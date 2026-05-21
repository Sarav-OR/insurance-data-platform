# Databricks notebook source
"""
02_silver_layer.py
==================
Production-grade Silver Layer for Insurance Data Platform.

Responsibilities:
- Read from Bronze Delta tables
- Cast string columns to proper types (dates, booleans, doubles)
- Cleanse and standardise data
- Enrich with derived/calculated columns
- Deduplicate using _record_hash
- Join domains into unified analytical views
- Write to Silver Delta tables

Architecture:
    Bronze Delta → Transformations → Silver Delta
"""

# ─────────────────────────────────────────────
# CELL 1 — Imports & Configuration
# ─────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("insurance_silver")

BRONZE_DB  = "insurance_bronze"
SILVER_DB  = "insurance_silver"
BATCH_ID   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

print(f"Silver layer starting — Batch ID: {BATCH_ID}")

# ─────────────────────────────────────────────
# CELL 2 — Setup Silver Database
# ─────────────────────────────────────────────

spark.sql(f"DROP DATABASE IF EXISTS {SILVER_DB} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DB}")
spark.sql(f"USE {SILVER_DB}")
print(f"✅ Database '{SILVER_DB}' ready")

# ─────────────────────────────────────────────
# CELL 3 — Shared Utility Functions
# ─────────────────────────────────────────────

def add_silver_audit(sdf):
    """Add Silver layer audit columns to every table."""
    return sdf \
        .withColumn("_silver_batch_id",        F.lit(BATCH_ID)) \
        .withColumn("_silver_load_timestamp",   F.current_timestamp()) \
        .withColumn("_silver_source_layer",     F.lit("bronze"))


def deduplicate(sdf, id_col: str, order_col: str = "_ingestion_timestamp"):
    """
    Remove duplicates keeping the latest record per ID.
    Production pattern: always deduplicate in Silver,
    never assume Bronze is deduplicated.
    """
    window = Window.partitionBy(id_col).orderBy(F.col(order_col).desc())
    return sdf.withColumn("_row_num", F.row_number().over(window)) \
              .filter(F.col("_row_num") == 1) \
              .drop("_row_num")


def write_silver(sdf, table: str, partition_cols: list = None):
    """Write DataFrame to Silver Delta table."""
    full_table = f"{SILVER_DB}.{table}"
    writer = sdf.write.format("delta").mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table)
    count = spark.table(full_table).count()
    log.info(f"  ✅ {full_table} → {count:,} rows")
    return count


def standardise_string(col_name: str):
    """Trim whitespace and uppercase for categorical columns."""
    return F.upper(F.trim(F.col(col_name)))


def null_safe_date(col_name: str):
    """Safely cast string to date, nullify unparseable values."""
    return F.to_date(F.col(col_name), "yyyy-MM-dd")


print("✅ Utility functions defined")

# ─────────────────────────────────────────────
# CELL 4 — Silver Customers
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("TRANSFORMING: CUSTOMERS")
print(f"{'='*50}")

bronze_customers = spark.table(f"{BRONZE_DB}.customers")

silver_customers = bronze_customers \
    .withColumn("date_of_birth",    null_safe_date("date_of_birth")) \
    .withColumn("customer_since",   null_safe_date("customer_since")) \
    .withColumn("gender",           standardise_string("gender")) \
    .withColumn("country",          standardise_string("country")) \
    .withColumn("channel",          standardise_string("channel")) \
    .withColumn("age_band",         standardise_string("age_band")) \
    .withColumn("email",            F.lower(F.trim(F.col("email")))) \
    .withColumn("first_name",       F.initcap(F.trim(F.col("first_name")))) \
    .withColumn("last_name",        F.initcap(F.trim(F.col("last_name")))) \
    .withColumn("full_name",        F.concat_ws(" ",
                                        F.col("first_name"),
                                        F.col("last_name"))) \
    .withColumn("customer_tenure_years",
                F.round(
                    F.datediff(F.current_date(), F.col("customer_since")) / 365, 1
                )) \
    .withColumn("is_active",
                F.when(F.col("customer_since").isNotNull(), F.lit(True))
                 .otherwise(F.lit(False))) \
    .withColumn("customer_segment",
                F.when(F.col("is_high_value") == True, F.lit("PREMIUM"))
                 .when(F.col("customer_tenure_years") >= 5, F.lit("LOYAL"))
                 .otherwise(F.lit("STANDARD")))

# Deduplicate
silver_customers = deduplicate(silver_customers, "customer_id")

# Drop Bronze audit columns — Silver has its own
silver_customers = silver_customers \
    .drop("_ingestion_timestamp", "_source_system", "_record_hash", "_batch_id",
          "_bronze_load_timestamp", "_bronze_batch_id", "_bronze_domain")

silver_customers = add_silver_audit(silver_customers)

write_silver(silver_customers, "customers")

# Quick check
print("\nCustomer segment distribution:")
spark.sql("""
    SELECT customer_segment, COUNT(*) AS count
    FROM insurance_silver.customers
    GROUP BY customer_segment
    ORDER BY count DESC
""").show()

# ─────────────────────────────────────────────
# CELL 5 — Silver Policies
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("TRANSFORMING: POLICIES")
print(f"{'='*50}")

bronze_policies = spark.table(f"{BRONZE_DB}.policies")

silver_policies = bronze_policies \
    .withColumn("start_date",          null_safe_date("start_date")) \
    .withColumn("end_date",            null_safe_date("end_date")) \
    .withColumn("policy_type",         standardise_string("policy_type")) \
    .withColumn("status",              standardise_string("status")) \
    .withColumn("currency",            standardise_string("currency")) \
    .withColumn("payment_frequency",   standardise_string("payment_frequency")) \
    .withColumn("distribution_channel",standardise_string("distribution_channel")) \
    .withColumn("policy_duration_days",
                F.datediff(F.col("end_date"), F.col("start_date"))) \
    .withColumn("is_expired",
                F.when(F.col("end_date") < F.current_date(), F.lit(True))
                 .otherwise(F.lit(False))) \
    .withColumn("days_to_expiry",
                F.datediff(F.col("end_date"), F.current_date())) \
    .withColumn("expiry_band",
                F.when(F.col("days_to_expiry") < 0,   F.lit("EXPIRED"))
                 .when(F.col("days_to_expiry") <= 30,  F.lit("EXPIRING_SOON"))
                 .when(F.col("days_to_expiry") <= 90,  F.lit("EXPIRING_90_DAYS"))
                 .otherwise(F.lit("ACTIVE"))) \
    .withColumn("premium_band",
                F.when(F.col("annual_premium_chf") < 500,   F.lit("LOW"))
                 .when(F.col("annual_premium_chf") < 2000,  F.lit("MEDIUM"))
                 .when(F.col("annual_premium_chf") < 4000,  F.lit("HIGH"))
                 .otherwise(F.lit("VERY_HIGH"))) \
    .withColumn("risk_band",
                F.when(F.col("risk_score") < 0.3,  F.lit("LOW"))
                 .when(F.col("risk_score") < 0.6,  F.lit("MEDIUM"))
                 .when(F.col("risk_score") < 0.8,  F.lit("HIGH"))
                 .otherwise(F.lit("VERY_HIGH"))) \
    .withColumn("coverage_to_premium_ratio",
                F.round(
                    F.col("coverage_amount_chf") / F.col("annual_premium_chf"), 2
                ))

silver_policies = deduplicate(silver_policies, "policy_id")
silver_policies = silver_policies \
    .drop("_ingestion_timestamp", "_source_system", "_record_hash", "_batch_id",
          "_bronze_load_timestamp", "_bronze_batch_id", "_bronze_domain")
silver_policies = add_silver_audit(silver_policies)

write_silver(silver_policies, "policies", partition_cols=["policy_type"])

print("\nPolicy expiry band distribution:")
spark.sql("""
    SELECT expiry_band, COUNT(*) AS count,
           ROUND(AVG(annual_premium_chf), 2) AS avg_premium
    FROM insurance_silver.policies
    GROUP BY expiry_band
    ORDER BY count DESC
""").show()

# ─────────────────────────────────────────────
# CELL 6 — Silver Claims
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("TRANSFORMING: CLAIMS")
print(f"{'='*50}")

bronze_claims = spark.table(f"{BRONZE_DB}.claims")

silver_claims = bronze_claims \
    .withColumn("incident_date",    null_safe_date("incident_date")) \
    .withColumn("submitted_date",   null_safe_date("submitted_date")) \
    .withColumn("claim_status",     standardise_string("claim_status")) \
    .withColumn("claim_type",       standardise_string("claim_type")) \
    .withColumn("claim_age_days",
                F.datediff(F.current_date(), F.col("incident_date"))) \
    .withColumn("submission_delay_band",
                F.when(F.col("days_to_submit") <= 7,   F.lit("IMMEDIATE"))
                 .when(F.col("days_to_submit") <= 30,  F.lit("NORMAL"))
                 .when(F.col("days_to_submit") <= 60,  F.lit("DELAYED"))
                 .otherwise(F.lit("VERY_DELAYED"))) \
    .withColumn("claim_severity",
                F.when(F.col("claim_amount_chf") < 5_000,   F.lit("LOW"))
                 .when(F.col("claim_amount_chf") < 25_000,  F.lit("MEDIUM"))
                 .when(F.col("claim_amount_chf") < 75_000,  F.lit("HIGH"))
                 .otherwise(F.lit("CATASTROPHIC"))) \
    .withColumn("settlement_ratio",
                F.when(
                    (F.col("settled_amount_chf").isNotNull()) &
                    (F.col("claim_amount_chf") > 0),
                    F.round(F.col("settled_amount_chf") / F.col("claim_amount_chf"), 3)
                ).otherwise(F.lit(None).cast("double"))) \
    .withColumn("is_high_value_claim",
                F.when(F.col("claim_amount_chf") >= 50_000, F.lit(True))
                 .otherwise(F.lit(False))) \
    .withColumn("fraud_risk_level",
                F.when(F.col("is_fraud_suspected") == True, F.lit("HIGH"))
                 .when(
                     (F.col("days_to_submit") > 45) |
                     (F.col("claim_amount_chf") > 100_000),
                     F.lit("MEDIUM")
                 )
                 .otherwise(F.lit("LOW")))

silver_claims = deduplicate(silver_claims, "claim_id")
silver_claims = silver_claims \
    .drop("_ingestion_timestamp", "_source_system", "_record_hash", "_batch_id",
          "_bronze_load_timestamp", "_bronze_batch_id", "_bronze_domain")
silver_claims = add_silver_audit(silver_claims)

write_silver(silver_claims, "claims", partition_cols=["claim_status"])

print("\nClaim severity distribution:")
spark.sql("""
    SELECT claim_severity,
           COUNT(*)                            AS total_claims,
           ROUND(AVG(claim_amount_chf), 2)     AS avg_amount_chf,
           SUM(CASE WHEN is_fraud_suspected
                    THEN 1 ELSE 0 END)         AS fraud_suspected
    FROM insurance_silver.claims
    GROUP BY claim_severity
    ORDER BY avg_amount_chf DESC
""").show()

# ─────────────────────────────────────────────
# CELL 7 — Silver Premiums
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("TRANSFORMING: PREMIUMS")
print(f"{'='*50}")

bronze_premiums = spark.table(f"{BRONZE_DB}.premiums")

silver_premiums = bronze_premiums \
    .withColumn("due_date",         null_safe_date("due_date")) \
    .withColumn("paid_date",        null_safe_date("paid_date")) \
    .withColumn("payment_status",   standardise_string("payment_status")) \
    .withColumn("payment_method",   standardise_string("payment_method")) \
    .withColumn("currency",         standardise_string("currency")) \
    .withColumn("is_overdue",
                F.when(
                    (F.col("payment_status") != "PAID") &
                    (F.col("due_date") < F.current_date()),
                    F.lit(True)
                ).otherwise(F.lit(False))) \
    .withColumn("arrears_amount_chf",
                F.when(
                    F.col("is_overdue") == True,
                    F.col("amount_due_chf")
                ).otherwise(F.lit(0.0))) \
    .withColumn("payment_variance_chf",
                F.when(
                    F.col("amount_paid_chf").isNotNull(),
                    F.round(F.col("amount_paid_chf") - F.col("amount_due_chf"), 2)
                ).otherwise(F.lit(None).cast("double"))) \
    .withColumn("days_to_pay",
                F.when(
                    F.col("paid_date").isNotNull(),
                    F.datediff(F.col("paid_date"), F.col("due_date"))
                ).otherwise(F.lit(None).cast("integer"))) \
    .withColumn("overdue_band",
                F.when(F.col("days_overdue") == 0,        F.lit("ON_TIME"))
                 .when(F.col("days_overdue") <= 30,       F.lit("1_30_DAYS"))
                 .when(F.col("days_overdue") <= 60,       F.lit("31_60_DAYS"))
                 .when(F.col("days_overdue") <= 90,       F.lit("61_90_DAYS"))
                 .otherwise(F.lit("90_PLUS_DAYS")))

silver_premiums = deduplicate(silver_premiums, "payment_id")
silver_premiums = silver_premiums \
    .drop("_ingestion_timestamp", "_source_system", "_record_hash", "_batch_id",
          "_bronze_load_timestamp", "_bronze_batch_id", "_bronze_domain")
silver_premiums = add_silver_audit(silver_premiums)

write_silver(silver_premiums, "premiums", partition_cols=["payment_status"])

print("\nPremium overdue band distribution:")
spark.sql("""
    SELECT overdue_band,
           COUNT(*)                            AS total_payments,
           ROUND(SUM(arrears_amount_chf), 2)   AS total_arrears_chf
    FROM insurance_silver.premiums
    GROUP BY overdue_band
    ORDER BY total_payments DESC
""").show()

# ─────────────────────────────────────────────
# CELL 8 — Silver Fraud Signals
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("TRANSFORMING: FRAUD SIGNALS")
print(f"{'='*50}")

bronze_fraud = spark.table(f"{BRONZE_DB}.fraud_signals")

silver_fraud = bronze_fraud \
    .withColumn("detected_at",      F.to_timestamp(F.col("detected_at"))) \
    .withColumn("signal_type",      standardise_string("signal_type")) \
    .withColumn("outcome",          standardise_string("outcome")) \
    .withColumn("score_band",
                F.when(F.col("signal_score") >= 0.8, F.lit("CRITICAL"))
                 .when(F.col("signal_score") >= 0.6, F.lit("HIGH"))
                 .when(F.col("signal_score") >= 0.4, F.lit("MEDIUM"))
                 .otherwise(F.lit("LOW"))) \
    .withColumn("is_confirmed_fraud",
                F.when(F.col("outcome") == "CONFIRMED_FRAUD", F.lit(True))
                 .otherwise(F.lit(False))) \
    .withColumn("needs_review",
                F.when(
                    (F.col("reviewed") == False) &
                    (F.col("signal_score") >= 0.7),
                    F.lit(True)
                ).otherwise(F.lit(False)))

silver_fraud = deduplicate(silver_fraud, "signal_id")
silver_fraud = silver_fraud \
    .drop("_ingestion_timestamp", "_source_system", "_record_hash", "_batch_id",
          "_bronze_load_timestamp", "_bronze_batch_id", "_bronze_domain")
silver_fraud = add_silver_audit(silver_fraud)

write_silver(silver_fraud, "fraud_signals", partition_cols=["signal_type"])

print("\nFraud signal score bands:")
spark.sql("""
    SELECT score_band,
           COUNT(*)                    AS signal_count,
           ROUND(AVG(signal_score), 3) AS avg_score,
           SUM(CASE WHEN is_confirmed_fraud
                    THEN 1 ELSE 0 END) AS confirmed_fraud
    FROM insurance_silver.fraud_signals
    GROUP BY score_band
    ORDER BY avg_score DESC
""").show()

# ─────────────────────────────────────────────
# CELL 9 — Silver Enriched View: Claims + Policy + Customer
# ─────────────────────────────────────────────
# Production pattern: build a wide analytical table
# joining key domains — avoids repeated joins in Gold/BI layer.

print(f"\n{'='*50}")
print("BUILDING: CLAIM ENRICHED VIEW")
print(f"{'='*50}")

claims_sdf   = spark.table(f"{SILVER_DB}.claims")
policies_sdf = spark.table(f"{SILVER_DB}.policies")
customers_sdf= spark.table(f"{SILVER_DB}.customers")
fraud_sdf    = spark.table(f"{SILVER_DB}.fraud_signals") \
                    .select("claim_id", "signal_score", "score_band",
                            "signal_type", "is_confirmed_fraud", "needs_review")

silver_claims_enriched = claims_sdf \
    .join(
        policies_sdf.select(
            "policy_id", "policy_type", "status",
            "annual_premium_chf", "coverage_amount_chf",
            "deductible_chf", "risk_score", "risk_band",
            "premium_band", "expiry_band"
        ),
        on="policy_id",
        how="left"
    ) \
    .join(
        customers_sdf.select(
            "customer_id", "full_name", "age_band",
            "customer_segment", "customer_tenure_years",
            "city", "country"
        ),
        on="customer_id",
        how="left"
    ) \
    .join(
        fraud_sdf,
        on="claim_id",
        how="left"
    ) \
    .withColumn("claim_exceeds_coverage",
                F.when(
                    F.col("claim_amount_chf") > F.col("coverage_amount_chf"),
                    F.lit(True)
                ).otherwise(F.lit(False))) \
    .withColumn("net_claim_after_deductible",
                F.greatest(
                    F.col("claim_amount_chf") - F.col("deductible_chf"),
                    F.lit(0.0)
                )) \
    .withColumn("combined_risk_flag",
                F.when(
                    (F.col("is_fraud_suspected") == True) |
                    (F.col("risk_band").isin("HIGH", "VERY_HIGH")) |
                    (F.col("claim_severity") == "CATASTROPHIC"),
                    F.lit(True)
                ).otherwise(F.lit(False)))

silver_claims_enriched = add_silver_audit(silver_claims_enriched)

write_silver(
    silver_claims_enriched,
    "claims_enriched",
    partition_cols=["claim_status"]
)

print("\nEnriched claims sample:")
spark.sql("""
    SELECT claim_id,
           full_name,
           policy_type,
           claim_severity,
           claim_amount_chf,
           net_claim_after_deductible,
           fraud_risk_level,
           combined_risk_flag,
           customer_segment
    FROM insurance_silver.claims_enriched
    WHERE combined_risk_flag = true
    LIMIT 5
""").show(truncate=False)

# ─────────────────────────────────────────────
# CELL 10 — Final Silver Validation Report
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("SILVER LAYER — FINAL VALIDATION REPORT")
print(f"{'='*50}")

silver_tables = [
    "customers",
    "policies",
    "claims",
    "premiums",
    "fraud_signals",
    "claims_enriched",
]

total = 0
for table in silver_tables:
    count = spark.table(f"{SILVER_DB}.{table}").count()
    total += count
    print(f"  {table:<25} {count:>10,} rows ✅")

print(f"\n  {'TOTAL':<25} {total:>10,} rows")
print(f"  Batch ID: {BATCH_ID}")
print(f"{'='*50}")

# ─────────────────────────────────────────────
# CELL 11 — Delta Table History (Audit Trail)
# ─────────────────────────────────────────────
# Production pattern: always verify Delta history
# confirms ACID writes and version tracking.

print("\nDelta history for silver.claims:")
spark.sql("DESCRIBE HISTORY insurance_silver.claims LIMIT 3").show(truncate=False)

print(f"\n✅ Silver layer complete — Batch ID: {BATCH_ID}")