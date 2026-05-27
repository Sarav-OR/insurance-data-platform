# Databricks notebook source

# ═══════════════════════════════════════════════════════════
# CELL 1 — Repository Path Setup
# Purpose : Add repo root to Python path so src/ imports work.
#           Must be first cell in every notebook.
#           Update path to match your Databricks username.
# ═══════════════════════════════════════════════════════════

import sys
import os

# Update YOUR_USERNAME to your actual Databricks username
REPO_ROOT = "/Workspace/Repos/saravanakumar.or@live.com/insurance-data-platform"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"✅ Repo root added to path: {REPO_ROOT}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 2 — Imports
# Purpose : Import all required libraries and src/ modules.
#           Silver imports are lighter than Bronze —
#           no data generation libraries needed here.
# ═══════════════════════════════════════════════════════════

import logging
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

# Platform shared modules
from src.config     import (BATCH_ID, DATABASES,
                             DELTA_SETTINGS, PARTITION_COLS)
from src.utils      import (safe_date, standardise_string,
                             deduplicate, write_delta,
                             drop_bronze_audit_cols,
                             apply_delta_settings)
from src.audit      import add_silver_audit
from src.monitoring import (apply_dq_rules, write_dq_monitoring,
                             log_pipeline_error,
                             reconcile_row_counts)
from src.dq_rules   import get_rules

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("silver_layer")

# ── Convenience aliases ───────────────────────────────────
BRONZE_DB = DATABASES["bronze"]
SILVER_DB = DATABASES["silver"]

print(f"✅ All imports successful")
print(f"   Batch ID  : {BATCH_ID}")
print(f"   Source DB : {BRONZE_DB}")
print(f"   Target DB : {SILVER_DB}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 3 — Database Setup
# Purpose : Create Silver database and apply Delta settings.
#           Silver reads from Bronze — Bronze must exist first.
#           Validates Bronze tables exist before proceeding.
# ═══════════════════════════════════════════════════════════

# Create Silver database
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DB}")
spark.sql(f"USE {SILVER_DB}")

# Apply Delta optimisation settings
apply_delta_settings(spark, DELTA_SETTINGS)

# Validate Bronze tables exist before starting
# Silver cannot run if Bronze failed or was never run
print("\nValidating Bronze tables exist...")
required_bronze = [
    "customers", "policies", "claims",
    "premiums", "fraud_signals"
]
for table in required_bronze:
    try:
        count = spark.table(f"{BRONZE_DB}.{table}").count()
        print(f"  {BRONZE_DB}.{table:<20} {count:>8,} rows ✅")
    except Exception as e:
        print(f"  {BRONZE_DB}.{table:<20} NOT FOUND ❌")
        raise RuntimeError(
            f"Bronze table '{table}' not found. "
            f"Run Bronze notebook first."
        )

print(f"\n✅ Database '{SILVER_DB}' ready")
print(f"✅ All Bronze source tables confirmed")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 4 — Silver Transformation Orchestrator
# Purpose : Single function that runs full Silver pipeline
#           for any domain. Encapsulates all transformation
#           steps in a consistent, repeatable pattern.
#
# Steps per domain:
#   1. Read from Bronze Delta table
#   2. Apply transformations (type cast, standardise, enrich)
#   3. Deduplicate — keep latest record per primary key
#   4. Drop Bronze audit columns — Silver adds its own
#   5. Add Silver audit columns
#   6. Apply Silver DQ rules
#   7. Write good records to Silver Delta table
#   8. Write bad records to quarantine table
#   9. Update DQ monitoring table
#  10. Reconcile row counts vs Bronze
# ═══════════════════════════════════════════════════════════

def transform_domain(domain: str,
                     transform_fn,
                     id_col: str,
                     partition_cols: list = None) -> int:
    """
    Full Silver transformation pipeline for one domain.

    Args:
        domain        : Domain name e.g. 'claims'
        transform_fn  : Function that applies transformations
                        Takes sdf, returns transformed sdf
        id_col        : Primary key column for deduplication
        partition_cols: Columns to partition Silver table by

    Returns:
        Count of records written to Silver
    """
    print(f"\n{'─'*50}")
    print(f"  TRANSFORMING: {domain.upper()}")
    print(f"{'─'*50}")

    try:
        # ── Step 1: Read from Bronze ──────────────────
        # Always read from Bronze — never from Silver
        # Silver is a clean transform of Bronze, not itself
        bronze_count = spark.table(
            f"{BRONZE_DB}.{domain}"
        ).count()
        log.info(
            f"[{domain}] Bronze records: {bronze_count:,}"
        )
        sdf = spark.table(f"{BRONZE_DB}.{domain}")

        # ── Step 2: Apply domain transformations ──────
        # transform_fn is different for each domain
        # Contains type casts, derived columns, enrichment
        sdf = transform_fn(sdf)
        log.info(
            f"[{domain}] Transformations applied"
        )

        # ── Step 3: Deduplicate ───────────────────────
        # Keep latest record per primary key
        # Handles resent records from source system
        sdf = deduplicate(sdf, id_col)
        log.info(f"[{domain}] Deduplication complete")

        # ── Step 4: Drop Bronze audit columns ─────────
        # Bronze audit columns not needed in Silver
        # Silver adds its own audit trail
        sdf = drop_bronze_audit_cols(sdf)

        # ── Step 5: Add Silver audit columns ──────────
        # _silver_batch_id, _silver_load_timestamp,
        # _silver_source_layer = 'bronze'
        sdf = add_silver_audit(sdf, BATCH_ID)

        # ── Step 6: Apply Silver DQ rules ─────────────
        # Silver rules validate derived columns are correct
        # e.g. claim_severity must be LOW/MEDIUM/HIGH/CATASTROPHIC
        rules = get_rules("silver", domain)
        if rules:
            good_sdf, bad_sdf, good_count, bad_count = \
                apply_dq_rules(sdf, domain, rules, BATCH_ID)
        else:
            good_sdf  = sdf
            good_count = sdf.count()
            bad_count  = 0
            bad_sdf    = None

        # ── Step 7: Write good records to Silver ──────
        final_count = write_delta(
            good_sdf, SILVER_DB, domain, partition_cols
        )

        # ── Step 8: Write bad records to quarantine ───
        if bad_sdf is not None and bad_count > 0:
            bad_sdf.write.format("delta").mode("overwrite") \
                   .saveAsTable(
                       f"{SILVER_DB}.rejected_{domain}"
                   )
            log.warning(
                f"Silver quarantine [{domain}]: "
                f"{bad_count:,} records"
            )

        # ── Step 9: DQ monitoring ─────────────────────
        write_dq_monitoring(
            spark, domain, good_count, bad_count, BATCH_ID
        )

        # ── Step 10: Reconcile vs Bronze ──────────────
        # Warn if Silver count differs from Bronze by > 1%
        reconcile_row_counts(
            spark,
            BRONZE_DB, domain,
            SILVER_DB, domain,
            tolerance_pct=1.0
        )

        print(
            f"  ✅ Complete: {final_count:,} records "
            f"transformed"
        )
        return final_count

    except Exception as e:
        log.error(
            f"[{domain}] Silver transform FAILED: {str(e)}"
        )
        log_pipeline_error(spark, domain, e, BATCH_ID)
        raise


print("✅ Silver transformation orchestrator ready")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 5 — Transform: Customers
# Purpose : Cleanse and enrich customer records.
# Reads   : insurance_bronze.customers
# Writes  : insurance_silver.customers
#
# Transformations applied:
#   date_of_birth/customer_since → proper date type
#   first_name/last_name         → title case (John Smith)
#   full_name                    → derived concatenation
#   email                        → lowercase + trim
#   gender/country/channel       → uppercase + trim
#   customer_tenure_years        → years since customer_since
#   customer_segment             → PREMIUM/LOYAL/STANDARD
#   is_active                    → derived boolean flag
# ═══════════════════════════════════════════════════════════

def transform_customers(sdf):
    """
    Apply Silver transformations to customer records.
    Type casting, standardisation and business enrichment.
    """
    return sdf \
        .withColumn("date_of_birth",
                    safe_date("date_of_birth")) \
        .withColumn("customer_since",
                    safe_date("customer_since")) \
        .withColumn("gender",
                    standardise_string("gender")) \
        .withColumn("country",
                    standardise_string("country")) \
        .withColumn("channel",
                    standardise_string("channel")) \
        .withColumn("age_band",
                    standardise_string("age_band")) \
        .withColumn("email",
                    F.lower(F.trim(F.col("email")))) \
        .withColumn("first_name",
                    F.initcap(F.trim(F.col("first_name")))) \
        .withColumn("last_name",
                    F.initcap(F.trim(F.col("last_name")))) \
        .withColumn("full_name",
                    F.concat_ws(" ",
                        F.col("first_name"),
                        F.col("last_name"))) \
        .withColumn("customer_tenure_years",
                    F.round(
                        F.datediff(
                            F.current_date(),
                            F.col("customer_since")
                        ) / 365, 1
                    )) \
        .withColumn("customer_segment",
                    F.when(
                        F.col("is_high_value") == F.lit(True),
                        F.lit("PREMIUM")
                    ).when(
                        F.col("customer_tenure_years") >= 5,
                        F.lit("LOYAL")
                    ).otherwise(F.lit("STANDARD"))) \
        .withColumn("is_active",
                    F.when(
                        F.col("customer_since").isNotNull(),
                        F.lit(True)
                    ).otherwise(F.lit(False)))


customers_count = transform_domain(
    domain         = "customers",
    transform_fn   = transform_customers,
    id_col         = "customer_id",
    partition_cols = PARTITION_COLS["silver"].get("customers")
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 6 — Transform: Policies
# Purpose : Cleanse and enrich policy records.
# Reads   : insurance_bronze.policies
# Writes  : insurance_silver.policies
#
# Transformations applied:
#   start_date/end_date      → proper date type
#   policy_type/status       → uppercase + trim
#   policy_duration_days     → end_date minus start_date
#   is_expired               → derived boolean flag
#   days_to_expiry           → days until policy expires
#   expiry_band              → EXPIRED/EXPIRING_SOON/ACTIVE
#   risk_band                → LOW/MEDIUM/HIGH/VERY_HIGH
#   premium_band             → LOW/MEDIUM/HIGH/VERY_HIGH
#   coverage_to_premium_ratio → coverage / annual_premium
# ═══════════════════════════════════════════════════════════

def transform_policies(sdf):
    """
    Apply Silver transformations to policy records.
    """
    return sdf \
        .withColumn("start_date",
                    safe_date("start_date")) \
        .withColumn("end_date",
                    safe_date("end_date")) \
        .withColumn("policy_type",
                    standardise_string("policy_type")) \
        .withColumn("status",
                    standardise_string("status")) \
        .withColumn("currency",
                    standardise_string("currency")) \
        .withColumn("payment_frequency",
                    standardise_string("payment_frequency")) \
        .withColumn("distribution_channel",
                    standardise_string("distribution_channel")) \
        .withColumn("policy_duration_days",
                    F.datediff(
                        F.col("end_date"),
                        F.col("start_date")
                    )) \
        .withColumn("is_expired",
                    F.when(
                        F.col("end_date") < F.current_date(),
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("days_to_expiry",
                    F.datediff(
                        F.col("end_date"),
                        F.current_date()
                    )) \
        .withColumn("expiry_band",
                    F.when(
                        F.col("days_to_expiry") < 0,
                        F.lit("EXPIRED")
                    ).when(
                        F.col("days_to_expiry") <= 30,
                        F.lit("EXPIRING_SOON")
                    ).when(
                        F.col("days_to_expiry") <= 90,
                        F.lit("EXPIRING_90_DAYS")
                    ).otherwise(F.lit("ACTIVE"))) \
        .withColumn("risk_band",
                    F.when(
                        F.col("risk_score") < 0.3,
                        F.lit("LOW")
                    ).when(
                        F.col("risk_score") < 0.6,
                        F.lit("MEDIUM")
                    ).when(
                        F.col("risk_score") < 0.8,
                        F.lit("HIGH")
                    ).otherwise(F.lit("VERY_HIGH"))) \
        .withColumn("premium_band",
                    F.when(
                        F.col("annual_premium_chf") < 500,
                        F.lit("LOW")
                    ).when(
                        F.col("annual_premium_chf") < 2000,
                        F.lit("MEDIUM")
                    ).when(
                        F.col("annual_premium_chf") < 4000,
                        F.lit("HIGH")
                    ).otherwise(F.lit("VERY_HIGH"))) \
        .withColumn("coverage_to_premium_ratio",
                    F.round(
                        F.col("coverage_amount_chf") /
                        F.col("annual_premium_chf"),
                        2
                    ))


policies_count = transform_domain(
    domain         = "policies",
    transform_fn   = transform_policies,
    id_col         = "policy_id",
    partition_cols = PARTITION_COLS["silver"].get("policies")
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 7 — Transform: Claims
# Purpose : Cleanse and enrich claim records.
# Reads   : insurance_bronze.claims
# Writes  : insurance_silver.claims
#
# Transformations applied:
#   incident_date/submitted_date → proper date type
#   claim_status/claim_type      → uppercase + trim
#   claim_age_days               → days since incident
#   submission_delay_band        → IMMEDIATE/NORMAL/DELAYED
#   claim_severity               → LOW/MEDIUM/HIGH/CATASTROPHIC
#   settlement_ratio             → settled / claimed amount
#   is_high_value_claim          → amount >= 50,000 CHF
#   fraud_risk_level             → HIGH/MEDIUM/LOW
# ═══════════════════════════════════════════════════════════

def transform_claims(sdf):
    """
    Apply Silver transformations to claim records.
    """
    return sdf \
        .withColumn("incident_date",
                    safe_date("incident_date")) \
        .withColumn("submitted_date",
                    safe_date("submitted_date")) \
        .withColumn("claim_status",
                    standardise_string("claim_status")) \
        .withColumn("claim_type",
                    standardise_string("claim_type")) \
        .withColumn("claim_age_days",
                    F.datediff(
                        F.current_date(),
                        F.col("incident_date")
                    )) \
        .withColumn("submission_delay_band",
                    F.when(
                        F.col("days_to_submit") <= 7,
                        F.lit("IMMEDIATE")
                    ).when(
                        F.col("days_to_submit") <= 30,
                        F.lit("NORMAL")
                    ).when(
                        F.col("days_to_submit") <= 60,
                        F.lit("DELAYED")
                    ).otherwise(F.lit("VERY_DELAYED"))) \
        .withColumn("claim_severity",
                    F.when(
                        F.col("claim_amount_chf") < 5000,
                        F.lit("LOW")
                    ).when(
                        F.col("claim_amount_chf") < 25000,
                        F.lit("MEDIUM")
                    ).when(
                        F.col("claim_amount_chf") < 75000,
                        F.lit("HIGH")
                    ).otherwise(F.lit("CATASTROPHIC"))) \
        .withColumn("settlement_ratio",
                    F.when(
                        (F.col("settled_amount_chf").isNotNull()) &
                        (F.col("claim_amount_chf") > 0),
                        F.round(
                            F.col("settled_amount_chf") /
                            F.col("claim_amount_chf"),
                            3
                        )
                    ).otherwise(F.lit(None).cast("double"))) \
        .withColumn("is_high_value_claim",
                    F.when(
                        F.col("claim_amount_chf") >= 50_000,
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("fraud_risk_level",
                    F.when(
                        F.col("is_fraud_suspected") == F.lit(True),
                        F.lit("HIGH")
                    ).when(
                        (F.col("days_to_submit") > 45) |
                        (F.col("claim_amount_chf") > 100_000),
                        F.lit("MEDIUM")
                    ).otherwise(F.lit("LOW")))


claims_count = transform_domain(
    domain         = "claims",
    transform_fn   = transform_claims,
    id_col         = "claim_id",
    partition_cols = PARTITION_COLS["silver"].get("claims")
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 8 — Transform: Premiums
# Purpose : Cleanse and enrich premium payment records.
# Reads   : insurance_bronze.premiums
# Writes  : insurance_silver.premiums
#
# Transformations applied:
#   due_date/paid_date    → proper date type
#   payment_status/method → uppercase + trim
#   is_overdue            → derived boolean flag
#   arrears_amount_chf    → amount due if overdue else 0
#   payment_variance_chf  → paid minus due amount
#   days_to_pay           → paid_date minus due_date
#   overdue_band          → ON_TIME/1_30_DAYS/31_60_DAYS etc
# ═══════════════════════════════════════════════════════════

def transform_premiums(sdf):
    """
    Apply Silver transformations to premium payment records.
    """
    return sdf \
        .withColumn("due_date",
                    safe_date("due_date")) \
        .withColumn("paid_date",
                    safe_date("paid_date")) \
        .withColumn("payment_status",
                    standardise_string("payment_status")) \
        .withColumn("payment_method",
                    standardise_string("payment_method")) \
        .withColumn("currency",
                    standardise_string("currency")) \
        .withColumn("is_overdue",
                    F.when(
                        (F.col("payment_status") != "PAID") &
                        (F.col("due_date") < F.current_date()),
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("arrears_amount_chf",
                    F.when(
                        F.col("is_overdue") == F.lit(True),
                        F.col("amount_due_chf")
                    ).otherwise(F.lit(0.0))) \
        .withColumn("payment_variance_chf",
                    F.when(
                        F.col("amount_paid_chf").isNotNull(),
                        F.round(
                            F.col("amount_paid_chf") -
                            F.col("amount_due_chf"),
                            2
                        )
                    ).otherwise(F.lit(None).cast("double"))) \
        .withColumn("days_to_pay",
                    F.when(
                        F.col("paid_date").isNotNull(),
                        F.datediff(
                            F.col("paid_date"),
                            F.col("due_date")
                        )
                    ).otherwise(F.lit(None).cast("integer"))) \
        .withColumn("overdue_band",
                    F.when(
                        F.col("days_overdue") == 0,
                        F.lit("ON_TIME")
                    ).when(
                        F.col("days_overdue") <= 30,
                        F.lit("1_30_DAYS")
                    ).when(
                        F.col("days_overdue") <= 60,
                        F.lit("31_60_DAYS")
                    ).when(
                        F.col("days_overdue") <= 90,
                        F.lit("61_90_DAYS")
                    ).otherwise(F.lit("90_PLUS_DAYS")))


premiums_count = transform_domain(
    domain         = "premiums",
    transform_fn   = transform_premiums,
    id_col         = "payment_id",
    partition_cols = PARTITION_COLS["silver"].get("premiums")
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 9 — Transform: Fraud Signals
# Purpose : Cleanse and enrich fraud signal records.
# Reads   : insurance_bronze.fraud_signals
# Writes  : insurance_silver.fraud_signals
#
# Transformations applied:
#   detected_at        → proper timestamp type
#   signal_type/outcome → uppercase + trim
#   score_band         → LOW/MEDIUM/HIGH/CRITICAL
#   is_confirmed_fraud → outcome == CONFIRMED_FRAUD
#   needs_review       → not reviewed AND score >= 0.7
# ═══════════════════════════════════════════════════════════

def transform_fraud_signals(sdf):
    """
    Apply Silver transformations to fraud signal records.
    """
    return sdf \
        .withColumn("detected_at",
                    F.to_timestamp(F.col("detected_at"))) \
        .withColumn("signal_type",
                    standardise_string("signal_type")) \
        .withColumn("outcome",
                    standardise_string("outcome")) \
        .withColumn("score_band",
                    F.when(
                        F.col("signal_score") >= 0.8,
                        F.lit("CRITICAL")
                    ).when(
                        F.col("signal_score") >= 0.6,
                        F.lit("HIGH")
                    ).when(
                        F.col("signal_score") >= 0.4,
                        F.lit("MEDIUM")
                    ).otherwise(F.lit("LOW"))) \
        .withColumn("is_confirmed_fraud",
                    F.when(
                        F.col("outcome") == "CONFIRMED_FRAUD",
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("needs_review",
                    F.when(
                        (F.col("reviewed") == F.lit(False)) &
                        (F.col("signal_score") >= 0.7),
                        F.lit(True)
                    ).otherwise(F.lit(False)))


fraud_count = transform_domain(
    domain         = "fraud_signals",
    transform_fn   = transform_fraud_signals,
    id_col         = "signal_id",
    partition_cols = PARTITION_COLS["silver"].get("fraud_signals")
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 10 — Build Claims Enriched View
# Purpose : Join claims to policies, customers and fraud signals
#           into one wide analytical table.
#
# Why this matters:
#   Every Gold and BI query needs claim + policy + customer data.
#   Without this join they would repeat the same join logic
#   in every downstream query — error-prone and slow.
#   We do it once here in Silver — everyone reads this table.
#
# Reads   : insurance_silver.claims
#           insurance_silver.policies
#           insurance_silver.customers
#           insurance_silver.fraud_signals
# Writes  : insurance_silver.claims_enriched
#
# Key derived columns added:
#   net_claim_after_deductible — claim minus deductible
#   claim_exceeds_coverage     — claim > policy coverage
#   combined_risk_flag         — fraud OR high risk OR catastrophic
# ═══════════════════════════════════════════════════════════

print(f"\n{'─'*50}")
print(f"  BUILDING: CLAIMS ENRICHED VIEW")
print(f"{'─'*50}")

try:
    claims_sdf    = spark.table(f"{SILVER_DB}.claims")
    policies_sdf  = spark.table(f"{SILVER_DB}.policies")
    customers_sdf = spark.table(f"{SILVER_DB}.customers")
    fraud_sdf     = spark.table(f"{SILVER_DB}.fraud_signals") \
        .select(
            "claim_id", "signal_score", "score_band",
            "signal_type", "is_confirmed_fraud", "needs_review"
        )

    # Join claims to policies — get product and risk context
    # Left join — keep all claims even if policy missing
    claims_enriched = claims_sdf \
        .join(
            policies_sdf.select(
                "policy_id", "policy_type", "status",
                "annual_premium_chf", "coverage_amount_chf",
                "deductible_chf", "risk_score", "risk_band",
                "premium_band", "expiry_band"
            ),
            on="policy_id", how="left"
        ) \
        .join(
            customers_sdf.select(
                "customer_id", "full_name", "age_band",
                "customer_segment", "customer_tenure_years",
                "city", "country"
            ),
            on="customer_id", how="left"
        ) \
        .join(fraud_sdf, on="claim_id", how="left") \
        .withColumn("net_claim_after_deductible",
                    F.greatest(
                        F.col("claim_amount_chf") -
                        F.col("deductible_chf"),
                        F.lit(0.0)
                    )) \
        .withColumn("claim_exceeds_coverage",
                    F.when(
                        F.col("claim_amount_chf") >
                        F.col("coverage_amount_chf"),
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("combined_risk_flag",
                    F.when(
                        (F.col("is_fraud_suspected") ==
                         F.lit(True)) |
                        (F.col("risk_band").isin(
                            "HIGH", "VERY_HIGH"
                        )) |
                        (F.col("claim_severity") ==
                         "CATASTROPHIC"),
                        F.lit(True)
                    ).otherwise(F.lit(False)))

    claims_enriched = add_silver_audit(claims_enriched, BATCH_ID)

    enriched_count = write_delta(
        claims_enriched, SILVER_DB, "claims_enriched",
        PARTITION_COLS["silver"].get("claims_enriched")
    )

    print(f"  ✅ Complete: {enriched_count:,} enriched records")

except Exception as e:
    log.error(f"Claims enriched view FAILED: {str(e)}")
    log_pipeline_error(spark, "claims_enriched", e, BATCH_ID)
    raise

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 11 — Silver Layer Validation Report
# Purpose : Post-transformation checks.
#           Verifies all Silver tables populated correctly.
#           Reviews DQ monitoring for Silver pass rates.
#           Checks reconciliation between Bronze and Silver.
#           GO / NO-GO decision point before Gold runs.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("SILVER LAYER — VALIDATION REPORT")
print(f"{'='*60}")

# ── Silver table counts ──────────────────────────────────
silver_tables = [
    "customers", "policies", "claims",
    "premiums", "fraud_signals", "claims_enriched"
]
total = 0

print("\n SILVER TABLES:")
for table in silver_tables:
    count = spark.table(f"{SILVER_DB}.{table}").count()
    total += count
    print(f"  {table:<25} {count:>10,} rows ✅")

print(f"\n  {'TOTAL':<25} {total:>10,} rows")

# ── Bronze vs Silver reconciliation ─────────────────────
print("\n BRONZE → SILVER RECONCILIATION:")
for domain in ["customers","policies","claims",
               "premiums","fraud_signals"]:
    bronze_c = spark.table(f"{BRONZE_DB}.{domain}").count()
    silver_c = spark.table(f"{SILVER_DB}.{domain}").count()
    diff     = abs(bronze_c - silver_c)
    flag     = "✅" if diff == 0 else "⚠️ "
    print(
        f"  {domain:<20} "
        f"Bronze: {bronze_c:>8,} → "
        f"Silver: {silver_c:>8,} "
        f"Diff: {diff:>4} {flag}"
    )

# ── DQ monitoring ────────────────────────────────────────
print("\n DQ PASS RATES (THIS BATCH):")
spark.sql(f"""
    SELECT domain,
           good_count,
           bad_count,
           pass_rate_pct,
           date_format(load_timestamp,
               'yyyy-MM-dd HH:mm:ss') AS loaded_at
    FROM {DATABASES['bronze']}.dq_monitoring
    WHERE batch_id = '{BATCH_ID}'
    ORDER BY domain
""").show()

# ── Pipeline errors ──────────────────────────────────────
error_count = spark.table(
    f"{DATABASES['bronze']}.pipeline_errors"
).filter(F.col("batch_id") == BATCH_ID).count()

status = "✅ SUCCESS" if error_count == 0 else "❌ FAILED"
print(f"\n{'='*60}")
print(f"  Batch ID : {BATCH_ID}")
print(f"  Status   : {status}")
print(f"  Errors   : {error_count}")
print(f"{'='*60}")

if error_count > 0:
    raise RuntimeError(
        f"Silver pipeline failed with {error_count} error(s). "
        f"Review pipeline_errors table before proceeding."
    )

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 12 — Sample Verification Queries
# Purpose : Visual spot checks to confirm transformations
#           produced correct results.
#           Engineer reviews these outputs manually.
# ═══════════════════════════════════════════════════════════

print("\nSpot check 1: Customer segments")
spark.sql(f"""
    SELECT customer_segment,
           COUNT(*)                            AS total,
           ROUND(AVG(customer_tenure_years),1) AS avg_tenure,
           SUM(CASE WHEN is_high_value
                    THEN 1 ELSE 0 END)         AS high_value_count
    FROM {SILVER_DB}.customers
    GROUP BY customer_segment
    ORDER BY total DESC
""").show()

print("\nSpot check 2: Claims severity distribution")
spark.sql(f"""
    SELECT claim_severity,
           COUNT(*)                            AS total,
           ROUND(AVG(claim_amount_chf), 2)     AS avg_amount,
           ROUND(AVG(settlement_ratio), 3)     AS avg_settlement,
           SUM(CASE WHEN is_fraud_suspected
                    THEN 1 ELSE 0 END)         AS fraud_count
    FROM {SILVER_DB}.claims
    GROUP BY claim_severity
    ORDER BY avg_amount DESC
""").show()

print("\nSpot check 3: Policy expiry bands")
spark.sql(f"""
    SELECT expiry_band,
           COUNT(*)                           AS total,
           ROUND(AVG(annual_premium_chf), 2)  AS avg_premium
    FROM {SILVER_DB}.policies
    GROUP BY expiry_band
    ORDER BY total DESC
""").show()

print("\nSpot check 4: Enriched claims sample")
spark.sql(f"""
    SELECT claim_id,
           full_name,
           policy_type,
           claim_severity,
           ROUND(claim_amount_chf, 0)          AS claimed,
           ROUND(net_claim_after_deductible, 0) AS net_claim,
           fraud_risk_level,
           combined_risk_flag
    FROM {SILVER_DB}.claims_enriched
    WHERE combined_risk_flag = true
    LIMIT 5
""").show(truncate=False)

print(f"\n✅ Silver layer complete — Batch ID: {BATCH_ID}")