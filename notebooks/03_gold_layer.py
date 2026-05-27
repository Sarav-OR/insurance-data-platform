# Databricks notebook source
"""
notebooks/03_gold_layer.py
===========================
Gold Layer — Insurance Data Platform
=====================================
Purpose : Build business-facing KPI aggregation tables
          from Silver data for BI consumption.
Layer   : Gold (Business)
Reads   : insurance_silver.* Delta tables
Writes  : insurance_gold.* Delta tables

Depends on:
  src/config.py     — configuration and constants
  src/utils.py      — write_delta, apply_delta_settings
  src/audit.py      — add_gold_audit
  src/monitoring.py — log_pipeline_error, reconcile_row_counts

Gold layer responsibilities:
  1. Aggregate Silver data into business KPI tables
  2. Calculate core insurance metrics:
     - Loss ratio (claims / premium income)
     - Collection rate (collected / due)
     - Fraud rate (fraud suspected / total claims)
     - CLV indicator (premium - claims)
     - Settlement rate (settled / total claims)
  3. Build monthly executive summary for trend analysis
  4. Optimise tables for direct BI consumption

Gold design principle:
  No raw records here — only aggregations.
  Business teams query Gold — never Silver directly.
  Every table answers a specific business question.
  Column names are business-friendly, not technical.
"""

# ═══════════════════════════════════════════════════════════
# CELL 1 — Repository Path Setup
# Purpose : Add repo root to Python path so src/ imports work.
#           Must be first cell. Update with your username.
# ═══════════════════════════════════════════════════════════

import sys
import os

REPO_ROOT = "/Workspace/Repos/saravanakumar.or@live.com/insurance-data-platform"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"✅ Repo root added to path: {REPO_ROOT}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 2 — Imports
# Purpose : Import libraries and src/ modules.
#           Gold only needs aggregation and audit functions —
#           no data generation or DQ rule imports needed.
# ═══════════════════════════════════════════════════════════

import logging
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

from src.config     import (BATCH_ID, DATABASES,
                             DELTA_SETTINGS, PARTITION_COLS)
from src.utils      import (write_delta, apply_delta_settings)
from src.audit      import add_gold_audit
from src.monitoring import log_pipeline_error

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("gold_layer")

# ── Convenience aliases ───────────────────────────────────
SILVER_DB = DATABASES["silver"]
GOLD_DB   = DATABASES["gold"]

print(f"✅ All imports successful")
print(f"   Batch ID  : {BATCH_ID}")
print(f"   Source DB : {SILVER_DB}")
print(f"   Target DB : {GOLD_DB}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 3 — Database Setup and Silver Validation
# Purpose : Create Gold database and validate Silver tables
#           exist before starting aggregations.
#           Gold cannot run if Silver failed.
# ═══════════════════════════════════════════════════════════

spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_DB}")
spark.sql(f"USE {GOLD_DB}")
apply_delta_settings(spark, DELTA_SETTINGS)

# Validate Silver tables exist
print("\nValidating Silver tables exist...")
required_silver = [
    "customers", "policies", "claims",
    "premiums", "fraud_signals", "claims_enriched"
]
for table in required_silver:
    try:
        count = spark.table(f"{SILVER_DB}.{table}").count()
        print(f"  {SILVER_DB}.{table:<20} {count:>8,} rows ✅")
    except Exception as e:
        raise RuntimeError(
            f"Silver table '{table}' not found. "
            f"Run Silver notebook first."
        )

print(f"\n✅ Database '{GOLD_DB}' ready")
print(f"✅ All Silver source tables confirmed")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 4 — Gold Aggregation Orchestrator
# Purpose : Reusable function for writing Gold tables.
#           Adds audit columns, writes to Delta,
#           logs errors consistently across all Gold tables.
# ═══════════════════════════════════════════════════════════

def build_gold_table(sdf,
                     table_name: str,
                     partition_cols: list = None) -> int:
    """
    Write aggregated DataFrame to Gold Delta table.
    Adds Gold audit columns and handles errors consistently.

    Args:
        sdf           : Aggregated Spark DataFrame
        table_name    : Target Gold table name
        partition_cols: Optional partition columns

    Returns:
        Row count written to Gold table
    """
    print(f"\n{'─'*50}")
    print(f"  BUILDING: {table_name.upper()}")
    print(f"{'─'*50}")

    try:
        # Add Gold audit columns to every table
        sdf = add_gold_audit(sdf, BATCH_ID)

        # Write to Gold Delta table
        count = write_delta(
            sdf, GOLD_DB, table_name, partition_cols
        )

        print(f"  ✅ Complete: {count:,} rows")
        return count

    except Exception as e:
        log.error(f"[{table_name}] Gold build FAILED: {str(e)}")
        log_pipeline_error(
            spark, table_name, e, BATCH_ID
        )
        raise


print("✅ Gold orchestrator ready")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 5 — Portfolio Summary
# Purpose : Policy portfolio composition by product type.
# Business question:
#   What is our policy mix, premium income and
#   coverage exposure by product line?
# Reads   : insurance_silver.policies
# Writes  : insurance_gold.portfolio_summary
#
# Key metrics:
#   total_policies          — count of policies per type
#   total_premium_income_chf — sum of annual premiums
#   total_coverage_exposure  — total liability exposure
#   avg_risk_score          — average underwriting risk
#   auto_renewal_rate       — % policies on auto renewal
# ═══════════════════════════════════════════════════════════

silver_policies = spark.table(f"{SILVER_DB}.policies")

portfolio_summary = silver_policies \
    .groupBy(
        "policy_type", "status",
        "risk_band", "premium_band"
    ) \
    .agg(
        F.count("policy_id")
            .alias("total_policies"),
        F.round(F.sum("annual_premium_chf"), 2)
            .alias("total_premium_income_chf"),
        F.round(F.avg("annual_premium_chf"), 2)
            .alias("avg_premium_chf"),
        F.round(F.sum("coverage_amount_chf"), 2)
            .alias("total_coverage_exposure_chf"),
        F.round(F.avg("risk_score"), 3)
            .alias("avg_risk_score"),
        F.round(F.avg("coverage_to_premium_ratio"), 2)
            .alias("avg_coverage_to_premium_ratio"),
        F.sum(F.when(
            F.col("auto_renewal") == F.lit(True), F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("auto_renewal_count"),
        F.sum(F.when(
            F.col("is_expired") == F.lit(True), F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("expired_count"),
    ) \
    .withColumn("auto_renewal_rate",
                F.round(
                    F.col("auto_renewal_count") /
                    F.col("total_policies") * 100, 2
                ))

build_gold_table(
    portfolio_summary, "portfolio_summary",
    partition_cols=["policy_type"]
)

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 6 — Claims KPIs
# Purpose : Core claims performance metrics by product line.
# Business question:
#   What is our claims performance — loss ratios,
#   settlement rates and average costs?
# Reads   : insurance_silver.claims_enriched
# Writes  : insurance_gold.claims_kpis
#
# Key metrics:
#   loss_ratio         — claims paid / premium income
#                        < 0.6 is healthy for most lines
#   avg_settlement_ratio — how much of claim is paid out
#   fraud_rate_pct     — % of claims suspected fraudulent
#   avg_days_to_submit — claims submission speed
# ═══════════════════════════════════════════════════════════

silver_enriched = spark.table(f"{SILVER_DB}.claims_enriched")

claims_kpis = silver_enriched \
    .groupBy(
        "policy_type", "claim_status",
        "claim_severity", "claim_type"
    ) \
    .agg(
        F.count("claim_id")
            .alias("total_claims"),
        F.countDistinct("customer_id")
            .alias("unique_claimants"),
        F.round(F.sum("claim_amount_chf"), 2)
            .alias("total_claimed_chf"),
        F.round(F.avg("claim_amount_chf"), 2)
            .alias("avg_claim_chf"),
        F.round(F.max("claim_amount_chf"), 2)
            .alias("max_claim_chf"),
        F.round(F.sum("settled_amount_chf"), 2)
            .alias("total_settled_chf"),
        F.round(F.avg("settlement_ratio"), 4)
            .alias("avg_settlement_ratio"),
        F.round(F.avg("days_to_submit"), 1)
            .alias("avg_days_to_submit"),
        F.sum(F.when(
            F.col("is_fraud_suspected") == F.lit(True),
            F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("fraud_suspected_count"),
        F.sum(F.when(
            F.col("is_high_value_claim") == F.lit(True),
            F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("high_value_count"),
        F.sum("annual_premium_chf")
            .alias("total_premium_chf"),
    ) \
    .withColumn("loss_ratio",
                F.round(
                    F.col("total_claimed_chf") /
                    F.when(
                        F.col("total_premium_chf") > 0,
                        F.col("total_premium_chf")
                    ).otherwise(F.lit(None)),
                    4
                )) \
    .withColumn("fraud_rate_pct",
                F.round(
                    F.col("fraud_suspected_count") /
                    F.col("total_claims") * 100, 2
                )) \
    .withColumn("settlement_rate_pct",
                F.round(
                    F.sum(F.when(
                        F.col("claim_status") == "SETTLED",
                        F.lit(1)
                    ).otherwise(F.lit(0))).over(
                        Window.partitionBy("policy_type")
                    ) / F.col("total_claims") * 100, 2
                ))

build_gold_table(
    claims_kpis, "claims_kpis",
    partition_cols=["policy_type"]
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 7 — Customer Segments
# Purpose : Customer segment performance and value analysis.
# Business question:
#   Which customer segments drive the most value,
#   claims and fraud risk?
# Reads   : insurance_silver.customers (+ policies + claims)
# Writes  : insurance_gold.customer_segments
#
# Key metrics:
#   clv_indicator      — premium income minus claims
#                        positive = profitable segment
#   fraud_rate_pct     — fraud exposure per segment
#   avg_policies       — product diversity per customer
# ═══════════════════════════════════════════════════════════

silver_customers = spark.table(f"{SILVER_DB}.customers")
silver_claims    = spark.table(f"{SILVER_DB}.claims")

# Summarise policies per customer
customer_policies = silver_policies \
    .groupBy("customer_id") \
    .agg(
        F.count("policy_id").alias("policy_count"),
        F.sum("annual_premium_chf").alias("total_premium_chf"),
        F.countDistinct("policy_type").alias("product_diversity"),
    )

# Summarise claims per customer
customer_claims = silver_claims \
    .groupBy("customer_id") \
    .agg(
        F.count("claim_id").alias("claim_count"),
        F.sum("claim_amount_chf").alias("total_claimed_chf"),
        F.sum(F.when(
            F.col("is_fraud_suspected") == F.lit(True),
            F.lit(1)
        ).otherwise(F.lit(0))).alias("fraud_count"),
    )

# Join and aggregate by segment
customer_segments = silver_customers \
    .join(customer_policies, on="customer_id", how="left") \
    .join(customer_claims,   on="customer_id", how="left") \
    .fillna(0, subset=[
        "policy_count", "total_premium_chf",
        "claim_count", "total_claimed_chf", "fraud_count"
    ]) \
    .groupBy(
        "customer_segment", "age_band",
        "channel", "country"
    ) \
    .agg(
        F.count("customer_id")
            .alias("customer_count"),
        F.round(F.avg("customer_tenure_years"), 1)
            .alias("avg_tenure_years"),
        F.round(F.avg("policy_count"), 2)
            .alias("avg_policies_per_customer"),
        F.round(F.sum("total_premium_chf"), 2)
            .alias("segment_premium_income_chf"),
        F.round(F.avg("total_premium_chf"), 2)
            .alias("avg_premium_per_customer_chf"),
        F.sum("claim_count")
            .alias("total_claims"),
        F.round(F.sum("total_claimed_chf"), 2)
            .alias("total_claimed_chf"),
        F.sum("fraud_count")
            .alias("total_fraud_suspected"),
    ) \
    .withColumn("clv_indicator",
                F.round(
                    F.col("segment_premium_income_chf") -
                    F.col("total_claimed_chf"), 2
                )) \
    .withColumn("fraud_rate_pct",
                F.round(
                    F.col("total_fraud_suspected") /
                    F.when(
                        F.col("total_claims") > 0,
                        F.col("total_claims")
                    ).otherwise(F.lit(None)) * 100, 2
                ))

build_gold_table(customer_segments, "customer_segments")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 8 — Premium Collections
# Purpose : Premium payment health and arrears analysis.
# Business question:
#   What is our collection rate and arrears exposure
#   by product line?
# Reads   : insurance_silver.premiums + policies
# Writes  : insurance_gold.premium_collections
#
# Key metrics:
#   collection_rate_pct — % of premiums successfully collected
#   total_arrears_chf   — total overdue premium exposure
#   overdue_rate_pct    — % of payments overdue
# ═══════════════════════════════════════════════════════════

silver_premiums = spark.table(f"{SILVER_DB}.premiums")

premiums_with_policy = silver_premiums \
    .join(
        silver_policies.select(
            "policy_id", "policy_type", "customer_id"
        ),
        on="policy_id", how="left"
    )

premium_collections = premiums_with_policy \
    .groupBy(
        "policy_type", "payment_status",
        "payment_method", "overdue_band"
    ) \
    .agg(
        F.count("payment_id")
            .alias("total_payments"),
        F.round(F.sum("amount_due_chf"), 2)
            .alias("total_due_chf"),
        F.round(F.sum("amount_paid_chf"), 2)
            .alias("total_collected_chf"),
        F.round(F.sum("arrears_amount_chf"), 2)
            .alias("total_arrears_chf"),
        F.round(F.avg("days_to_pay"), 1)
            .alias("avg_days_to_pay"),
        F.sum(F.when(
            F.col("is_overdue") == F.lit(True), F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("overdue_count"),
    ) \
    .withColumn("collection_rate_pct",
                F.round(
                    F.col("total_collected_chf") /
                    F.when(
                        F.col("total_due_chf") > 0,
                        F.col("total_due_chf")
                    ).otherwise(F.lit(None)) * 100, 2
                )) \
    .withColumn("overdue_rate_pct",
                F.round(
                    F.col("overdue_count") /
                    F.col("total_payments") * 100, 2
                ))

build_gold_table(
    premium_collections, "premium_collections",
    partition_cols=["policy_type"]
)

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 9 — Fraud Summary
# Purpose : Fraud detection effectiveness and exposure.
# Business question:
#   How effective is our fraud detection?
#   What is the financial exposure by fraud signal type?
# Reads   : insurance_silver.fraud_signals + claims + policies
# Writes  : insurance_gold.fraud_summary
#
# Key metrics:
#   confirmation_rate_pct    — % of signals confirmed fraud
#   total_fraud_exposure_chf — financial exposure
#   review_completion_rate   — % of signals reviewed
# ═══════════════════════════════════════════════════════════

silver_fraud = spark.table(f"{SILVER_DB}.fraud_signals")

fraud_with_claims = silver_fraud \
    .join(
        silver_claims.select(
            "claim_id", "policy_id",
            "claim_amount_chf", "claim_type", "claim_severity"
        ),
        on="claim_id", how="left"
    ) \
    .join(
        silver_policies.select("policy_id", "policy_type"),
        on="policy_id", how="left"
    )

fraud_summary = fraud_with_claims \
    .groupBy(
        "signal_type", "score_band",
        "policy_type", "claim_type"
    ) \
    .agg(
        F.count("signal_id")
            .alias("total_signals"),
        F.round(F.avg("signal_score"), 3)
            .alias("avg_signal_score"),
        F.sum(F.when(
            F.col("is_confirmed_fraud") == F.lit(True),
            F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("confirmed_fraud_count"),
        F.sum(F.when(
            F.col("needs_review") == F.lit(True),
            F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("needs_review_count"),
        F.sum(F.when(
            F.col("reviewed") == F.lit(True), F.lit(1)
        ).otherwise(F.lit(0)))
            .alias("reviewed_count"),
        F.round(F.sum("claim_amount_chf"), 2)
            .alias("total_fraud_exposure_chf"),
        F.round(F.avg("claim_amount_chf"), 2)
            .alias("avg_fraud_claim_chf"),
    ) \
    .withColumn("confirmation_rate_pct",
                F.round(
                    F.col("confirmed_fraud_count") /
                    F.when(
                        F.col("total_signals") > 0,
                        F.col("total_signals")
                    ).otherwise(F.lit(None)) * 100, 2
                )) \
    .withColumn("review_completion_rate_pct",
                F.round(
                    F.col("reviewed_count") /
                    F.when(
                        F.col("total_signals") > 0,
                        F.col("total_signals")
                    ).otherwise(F.lit(None)) * 100, 2
                ))

build_gold_table(
    fraud_summary, "fraud_summary",
    partition_cols=["signal_type"]
)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 10 — Executive Summary
# Purpose : Monthly KPI time-series for executive reporting.
# Business question:
#   How is the business performing month by month?
#   What are the trends in loss ratio, fraud rate,
#   collection rate and settlement rate?
# Reads   : insurance_silver.policies + claims + premiums
# Writes  : insurance_gold.executive_summary
#
# Design:
#   One row per month — time-series ready for Power BI
#   All key KPIs in one place — no complex joins needed
#   Business teams can answer trend questions directly
# ═══════════════════════════════════════════════════════════

# Monthly policy metrics
monthly_policies = silver_policies \
    .withColumn("month",
                F.date_format("start_date", "yyyy-MM")) \
    .groupBy("month") \
    .agg(
        F.count("policy_id").alias("new_policies"),
        F.round(F.sum("annual_premium_chf"), 2)
            .alias("new_premium_income_chf"),
        F.round(F.avg("risk_score"), 3)
            .alias("avg_risk_score"),
    )

# Monthly claims metrics
monthly_claims = silver_claims \
    .withColumn("month",
                F.date_format("incident_date", "yyyy-MM")) \
    .groupBy("month") \
    .agg(
        F.count("claim_id").alias("total_claims"),
        F.round(F.sum("claim_amount_chf"), 2)
            .alias("total_claims_chf"),
        F.round(F.avg("claim_amount_chf"), 2)
            .alias("avg_claim_chf"),
        F.sum(F.when(
            F.col("is_fraud_suspected") == F.lit(True),
            F.lit(1)
        ).otherwise(F.lit(0))).alias("fraud_suspected"),
        F.sum(F.when(
            F.col("claim_status") == "SETTLED", F.lit(1)
        ).otherwise(F.lit(0))).alias("settled_claims"),
    )

# Monthly premium collections
monthly_premiums = silver_premiums \
    .withColumn("month",
                F.date_format("due_date", "yyyy-MM")) \
    .groupBy("month") \
    .agg(
        F.round(F.sum("amount_due_chf"), 2)
            .alias("total_due_chf"),
        F.round(F.sum("amount_paid_chf"), 2)
            .alias("total_collected_chf"),
        F.round(F.sum("arrears_amount_chf"), 2)
            .alias("total_arrears_chf"),
    )

# Join all monthly metrics
executive_summary = monthly_policies \
    .join(monthly_claims,   on="month", how="full") \
    .join(monthly_premiums, on="month", how="full") \
    .fillna(0) \
    .withColumn("loss_ratio",
                F.round(
                    F.col("total_claims_chf") /
                    F.when(
                        F.col("new_premium_income_chf") > 0,
                        F.col("new_premium_income_chf")
                    ).otherwise(F.lit(None)),
                    4
                )) \
    .withColumn("collection_rate_pct",
                F.round(
                    F.col("total_collected_chf") /
                    F.when(
                        F.col("total_due_chf") > 0,
                        F.col("total_due_chf")
                    ).otherwise(F.lit(None)) * 100, 2
                )) \
    .withColumn("fraud_rate_pct",
                F.round(
                    F.col("fraud_suspected") /
                    F.when(
                        F.col("total_claims") > 0,
                        F.col("total_claims")
                    ).otherwise(F.lit(None)) * 100, 2
                )) \
    .withColumn("settlement_rate_pct",
                F.round(
                    F.col("settled_claims") /
                    F.when(
                        F.col("total_claims") > 0,
                        F.col("total_claims")
                    ).otherwise(F.lit(None)) * 100, 2
                )) \
    .orderBy("month")

build_gold_table(executive_summary, "executive_summary")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 11 — Gold Layer Validation Report
# Purpose : Post-aggregation checks.
#           Verifies all Gold tables built correctly.
#           Checks pipeline errors.
#           GO / NO-GO before Fraud layer runs.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("GOLD LAYER — VALIDATION REPORT")
print(f"{'='*60}")

gold_tables = [
    "portfolio_summary",
    "claims_kpis",
    "customer_segments",
    "premium_collections",
    "fraud_summary",
    "executive_summary",
]

total = 0
print("\n GOLD TABLES:")
for table in gold_tables:
    try:
        count = spark.table(f"{GOLD_DB}.{table}").count()
        total += count
        print(f"  {table:<25} {count:>8,} rows ✅")
    except Exception as e:
        print(f"  {table:<25} ERROR ❌")

print(f"\n  {'TOTAL':<25} {total:>8,} rows")

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
        f"Gold pipeline failed with {error_count} error(s)."
    )


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 12 — Business KPI Spot Checks
# Purpose : Visual verification of key business metrics.
#           Engineer reviews to confirm KPIs are sensible.
# ═══════════════════════════════════════════════════════════

print("\nKPI 1: Loss ratio by policy type")
spark.sql(f"""
    SELECT policy_type,
           SUM(total_claims)                   AS total_claims,
           ROUND(SUM(total_claimed_chf), 0)    AS total_claimed,
           ROUND(SUM(total_premium_chf), 0)    AS total_premium,
           ROUND(AVG(loss_ratio), 4)           AS avg_loss_ratio,
           SUM(fraud_suspected_count)          AS fraud_suspected
    FROM {GOLD_DB}.claims_kpis
    GROUP BY policy_type
    ORDER BY avg_loss_ratio DESC
""").show()

print("\nKPI 2: Customer segment CLV")
spark.sql(f"""
    SELECT customer_segment,
           SUM(customer_count)                         AS customers,
           ROUND(SUM(segment_premium_income_chf), 0)   AS premium_income,
           ROUND(SUM(total_claimed_chf), 0)             AS claims_paid,
           ROUND(SUM(clv_indicator), 0)                 AS net_clv,
           ROUND(AVG(fraud_rate_pct), 2)                AS fraud_rate
    FROM {GOLD_DB}.customer_segments
    GROUP BY customer_segment
    ORDER BY net_clv DESC
""").show()

print("\nKPI 3: Executive summary — last 6 months")
spark.sql(f"""
    SELECT month,
           new_policies,
           ROUND(new_premium_income_chf, 0) AS premium_income,
           total_claims,
           loss_ratio,
           collection_rate_pct,
           fraud_rate_pct,
           settlement_rate_pct
    FROM {GOLD_DB}.executive_summary
    WHERE month >= '2024-07'
    ORDER BY month DESC
""").show()

print(f"\n✅ Gold layer complete — Batch ID: {BATCH_ID}")