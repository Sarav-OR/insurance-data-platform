# Databricks notebook source
"""
03_gold_layer.py
================
Production-grade Gold Layer for Insurance Data Platform.

Responsibilities:
- Read from Silver Delta tables
- Build business-facing aggregated KPI tables
- Calculate loss ratios, settlement rates, fraud rates
- Build executive summary table
- All tables optimised for BI consumption

Architecture:
    Silver Delta → Aggregations → Gold Delta Tables
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
log = logging.getLogger("insurance_gold")

SILVER_DB  = "insurance_silver"
GOLD_DB    = "insurance_gold"
BATCH_ID   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

print(f"Gold layer starting — Batch ID: {BATCH_ID}")

# ─────────────────────────────────────────────
# CELL 2 — Setup Gold Database
# ─────────────────────────────────────────────

spark.sql(f"DROP DATABASE IF EXISTS {GOLD_DB} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_DB}")
spark.sql(f"USE {GOLD_DB}")
print(f"✅ Database '{GOLD_DB}' ready")

# ─────────────────────────────────────────────
# CELL 3 — Shared Utility Functions
# ─────────────────────────────────────────────

def add_gold_audit(sdf):
    """Add Gold layer audit columns."""
    return sdf \
        .withColumn("_gold_batch_id",        F.lit(BATCH_ID)) \
        .withColumn("_gold_load_timestamp",  F.current_timestamp()) \
        .withColumn("_gold_source_layer",    F.lit("silver"))


def write_gold(sdf, table: str, partition_cols: list = None):
    """Write DataFrame to Gold Delta table."""
    full_table = f"{GOLD_DB}.{table}"
    writer = sdf.write.format("delta").mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table)
    count = spark.table(full_table).count()
    log.info(f"  ✅ {full_table} → {count:,} rows")
    return count


print("✅ Utility functions defined")

# ─────────────────────────────────────────────
# CELL 4 — Load Silver Tables
# ─────────────────────────────────────────────

print("\nLoading Silver tables...")

silver_customers  = spark.table(f"{SILVER_DB}.customers")
silver_policies   = spark.table(f"{SILVER_DB}.policies")
silver_claims     = spark.table(f"{SILVER_DB}.claims")
silver_premiums   = spark.table(f"{SILVER_DB}.premiums")
silver_fraud      = spark.table(f"{SILVER_DB}.fraud_signals")
silver_enriched   = spark.table(f"{SILVER_DB}.claims_enriched")

print("✅ All Silver tables loaded")

# ─────────────────────────────────────────────
# CELL 5 — Gold: Portfolio Summary
# ─────────────────────────────────────────────
# Business question:
# What is our policy portfolio composition,
# premium income and coverage exposure by product?

print(f"\n{'='*50}")
print("BUILDING: PORTFOLIO SUMMARY")
print(f"{'='*50}")

gold_portfolio = silver_policies \
    .groupBy("policy_type", "status", "risk_band", "premium_band") \
    .agg(
        F.count("policy_id")                        .alias("total_policies"),
        F.sum("annual_premium_chf")                 .alias("total_premium_income_chf"),
        F.avg("annual_premium_chf")                 .alias("avg_premium_chf"),
        F.sum("coverage_amount_chf")                .alias("total_coverage_exposure_chf"),
        F.avg("coverage_amount_chf")                .alias("avg_coverage_chf"),
        F.avg("deductible_chf")                     .alias("avg_deductible_chf"),
        F.avg("risk_score")                         .alias("avg_risk_score"),
        F.avg("coverage_to_premium_ratio")          .alias("avg_coverage_to_premium_ratio"),
        F.sum(F.when(F.col("auto_renewal") == True,
                     F.lit(1)).otherwise(F.lit(0))) .alias("auto_renewal_count"),
        F.sum(F.when(F.col("is_expired") == True,
                     F.lit(1)).otherwise(F.lit(0))) .alias("expired_count"),
    ) \
    .withColumn("avg_premium_chf",
                F.round(F.col("avg_premium_chf"), 2)) \
    .withColumn("avg_coverage_chf",
                F.round(F.col("avg_coverage_chf"), 2)) \
    .withColumn("avg_risk_score",
                F.round(F.col("avg_risk_score"), 3)) \
    .withColumn("avg_coverage_to_premium_ratio",
                F.round(F.col("avg_coverage_to_premium_ratio"), 2)) \
    .withColumn("total_premium_income_chf",
                F.round(F.col("total_premium_income_chf"), 2)) \
    .withColumn("total_coverage_exposure_chf",
                F.round(F.col("total_coverage_exposure_chf"), 2)) \
    .withColumn("auto_renewal_rate",
                F.round(F.col("auto_renewal_count") /
                        F.col("total_policies") * 100, 2))

gold_portfolio = add_gold_audit(gold_portfolio)
write_gold(gold_portfolio, "portfolio_summary", partition_cols=["policy_type"])

print("\nPortfolio summary by policy type:")
spark.sql("""
    SELECT policy_type,
           SUM(total_policies)                          AS total_policies,
           ROUND(SUM(total_premium_income_chf), 0)      AS total_premium_chf,
           ROUND(SUM(total_coverage_exposure_chf), 0)   AS total_exposure_chf,
           ROUND(AVG(avg_risk_score), 3)                AS avg_risk_score
    FROM insurance_gold.portfolio_summary
    GROUP BY policy_type
    ORDER BY total_policies DESC
""").show()

# ─────────────────────────────────────────────
# CELL 6 — Gold: Claims KPIs
# ─────────────────────────────────────────────
# Business question:
# What is our claims performance — loss ratios,
# settlement rates, average costs by product?

print(f"\n{'='*50}")
print("BUILDING: CLAIMS KPIs")
print(f"{'='*50}")

# Join claims to policies for premium context
claims_with_premium = silver_claims \
    .join(
        silver_policies.select(
            "policy_id", "policy_type",
            "annual_premium_chf", "coverage_amount_chf"
        ),
        on="policy_id",
        how="left"
    )

gold_claims_kpis = claims_with_premium \
    .groupBy("policy_type", "claim_status", "claim_severity", "claim_type") \
    .agg(
        F.count("claim_id")                             .alias("total_claims"),
        F.sum("claim_amount_chf")                       .alias("total_claimed_chf"),
        F.avg("claim_amount_chf")                       .alias("avg_claim_chf"),
        F.max("claim_amount_chf")                       .alias("max_claim_chf"),
        F.min("claim_amount_chf")                       .alias("min_claim_chf"),
        F.sum("settled_amount_chf")                     .alias("total_settled_chf"),
        F.avg("settlement_ratio")                       .alias("avg_settlement_ratio"),
        F.avg("days_to_submit")                         .alias("avg_days_to_submit"),
        F.avg("claim_age_days")                         .alias("avg_claim_age_days"),
        F.sum(F.when(F.col("is_fraud_suspected") == True,
                     F.lit(1)).otherwise(F.lit(0)))     .alias("fraud_suspected_count"),
        F.sum(F.when(F.col("is_high_value_claim") == True,
                     F.lit(1)).otherwise(F.lit(0)))     .alias("high_value_count"),
        F.sum(F.when(F.col("third_party_involved") == True,
                     F.lit(1)).otherwise(F.lit(0)))     .alias("third_party_count"),
        F.sum("annual_premium_chf")                     .alias("total_premium_chf"),
    ) \
    .withColumn("avg_claim_chf",
                F.round(F.col("avg_claim_chf"), 2)) \
    .withColumn("total_claimed_chf",
                F.round(F.col("total_claimed_chf"), 2)) \
    .withColumn("total_settled_chf",
                F.round(F.col("total_settled_chf"), 2)) \
    .withColumn("avg_settlement_ratio",
                F.round(F.col("avg_settlement_ratio"), 3)) \
    .withColumn("avg_days_to_submit",
                F.round(F.col("avg_days_to_submit"), 1)) \
    .withColumn("loss_ratio",
                F.round(
                    F.col("total_claimed_chf") /
                    F.when(F.col("total_premium_chf") > 0,
                           F.col("total_premium_chf"))
                    .otherwise(F.lit(None)),
                    4
                )) \
    .withColumn("fraud_rate",
                F.round(
                    F.col("fraud_suspected_count") /
                    F.col("total_claims") * 100,
                    2
                ))

gold_claims_kpis = add_gold_audit(gold_claims_kpis)
write_gold(gold_claims_kpis, "claims_kpis", partition_cols=["policy_type"])

print("\nClaims KPIs by severity:")
spark.sql("""
    SELECT claim_severity,
           SUM(total_claims)                    AS total_claims,
           ROUND(SUM(total_claimed_chf), 0)     AS total_claimed_chf,
           ROUND(AVG(avg_claim_chf), 0)         AS avg_claim_chf,
           ROUND(AVG(loss_ratio), 4)            AS avg_loss_ratio,
           ROUND(AVG(avg_settlement_ratio), 3)  AS avg_settlement_ratio,
           SUM(fraud_suspected_count)           AS fraud_suspected
    FROM insurance_gold.claims_kpis
    GROUP BY claim_severity
    ORDER BY avg_claim_chf DESC
""").show()

# ─────────────────────────────────────────────
# CELL 7 — Gold: Customer Segments
# ─────────────────────────────────────────────
# Business question:
# Which customer segments drive the most value,
# claims and fraud risk?

print(f"\n{'='*50}")
print("BUILDING: CUSTOMER SEGMENTS")
print(f"{'='*50}")

# Customer policy summary
customer_policies = silver_policies \
    .groupBy("customer_id") \
    .agg(
        F.count("policy_id")            .alias("policy_count"),
        F.sum("annual_premium_chf")     .alias("total_premium_chf"),
        F.countDistinct("policy_type")  .alias("product_diversity"),
    )

# Customer claims summary
customer_claims = silver_claims \
    .groupBy("customer_id") \
    .agg(
        F.count("claim_id")                             .alias("claim_count"),
        F.sum("claim_amount_chf")                       .alias("total_claimed_chf"),
        F.sum(F.when(F.col("is_fraud_suspected") == True,
                     F.lit(1)).otherwise(F.lit(0)))     .alias("fraud_count"),
    )

gold_customer_segments = silver_customers \
    .join(customer_policies, on="customer_id", how="left") \
    .join(customer_claims,   on="customer_id", how="left") \
    .fillna(0, subset=["policy_count", "total_premium_chf",
                       "claim_count", "total_claimed_chf",
                       "fraud_count", "product_diversity"]) \
    .groupBy("customer_segment", "age_band", "channel", "country") \
    .agg(
        F.count("customer_id")              .alias("customer_count"),
        F.avg("customer_tenure_years")      .alias("avg_tenure_years"),
        F.avg("policy_count")               .alias("avg_policies_per_customer"),
        F.sum("total_premium_chf")          .alias("segment_premium_income_chf"),
        F.avg("total_premium_chf")          .alias("avg_premium_per_customer_chf"),
        F.sum("claim_count")                .alias("total_claims"),
        F.sum("total_claimed_chf")          .alias("total_claimed_chf"),
        F.avg("claim_count")                .alias("avg_claims_per_customer"),
        F.sum("fraud_count")                .alias("total_fraud_suspected"),
        F.avg("product_diversity")          .alias("avg_product_diversity"),
    ) \
    .withColumn("avg_tenure_years",
                F.round(F.col("avg_tenure_years"), 1)) \
    .withColumn("avg_policies_per_customer",
                F.round(F.col("avg_policies_per_customer"), 2)) \
    .withColumn("avg_premium_per_customer_chf",
                F.round(F.col("avg_premium_per_customer_chf"), 2)) \
    .withColumn("avg_claims_per_customer",
                F.round(F.col("avg_claims_per_customer"), 3)) \
    .withColumn("segment_premium_income_chf",
                F.round(F.col("segment_premium_income_chf"), 2)) \
    .withColumn("total_claimed_chf",
                F.round(F.col("total_claimed_chf"), 2)) \
    .withColumn("fraud_rate_pct",
                F.round(
                    F.col("total_fraud_suspected") /
                    F.when(F.col("total_claims") > 0,
                           F.col("total_claims"))
                    .otherwise(F.lit(None)) * 100,
                    2
                )) \
    .withColumn("clv_indicator",
                F.round(
                    F.col("segment_premium_income_chf") -
                    F.col("total_claimed_chf"),
                    2
                ))

gold_customer_segments = add_gold_audit(gold_customer_segments)
write_gold(gold_customer_segments, "customer_segments")

print("\nCustomer segment performance:")
spark.sql("""
    SELECT customer_segment,
           SUM(customer_count)                          AS total_customers,
           ROUND(SUM(segment_premium_income_chf), 0)   AS total_premium_chf,
           ROUND(SUM(total_claimed_chf), 0)             AS total_claimed_chf,
           ROUND(SUM(clv_indicator), 0)                 AS net_value_chf,
           ROUND(AVG(fraud_rate_pct), 2)                AS avg_fraud_rate_pct
    FROM insurance_gold.customer_segments
    GROUP BY customer_segment
    ORDER BY total_premium_chf DESC
""").show()

# ─────────────────────────────────────────────
# CELL 8 — Gold: Premium Collections
# ─────────────────────────────────────────────
# Business question:
# What is our premium collection health —
# payment rates, arrears, overdue exposure?

print(f"\n{'='*50}")
print("BUILDING: PREMIUM COLLECTIONS")
print(f"{'='*50}")

premiums_with_policy = silver_premiums \
    .join(
        silver_policies.select("policy_id", "policy_type", "customer_id"),
        on="policy_id",
        how="left"
    )

gold_premium_collections = premiums_with_policy \
    .groupBy("policy_type", "payment_status", "payment_method", "overdue_band") \
    .agg(
        F.count("payment_id")                           .alias("total_payments"),
        F.sum("amount_due_chf")                         .alias("total_due_chf"),
        F.sum("amount_paid_chf")                        .alias("total_collected_chf"),
        F.sum("arrears_amount_chf")                     .alias("total_arrears_chf"),
        F.avg("days_to_pay")                            .alias("avg_days_to_pay"),
        F.avg("days_overdue")                           .alias("avg_days_overdue"),
        F.sum(F.when(F.col("is_overdue") == True,
                     F.lit(1)).otherwise(F.lit(0)))     .alias("overdue_count"),
        F.avg("payment_variance_chf")                   .alias("avg_payment_variance_chf"),
    ) \
    .withColumn("total_due_chf",
                F.round(F.col("total_due_chf"), 2)) \
    .withColumn("total_collected_chf",
                F.round(F.col("total_collected_chf"), 2)) \
    .withColumn("total_arrears_chf",
                F.round(F.col("total_arrears_chf"), 2)) \
    .withColumn("avg_days_to_pay",
                F.round(F.col("avg_days_to_pay"), 1)) \
    .withColumn("avg_days_overdue",
                F.round(F.col("avg_days_overdue"), 1)) \
    .withColumn("collection_rate_pct",
                F.round(
                    F.col("total_collected_chf") /
                    F.when(F.col("total_due_chf") > 0,
                           F.col("total_due_chf"))
                    .otherwise(F.lit(None)) * 100,
                    2
                )) \
    .withColumn("overdue_rate_pct",
                F.round(
                    F.col("overdue_count") /
                    F.col("total_payments") * 100,
                    2
                ))

gold_premium_collections = add_gold_audit(gold_premium_collections)
write_gold(gold_premium_collections, "premium_collections",
           partition_cols=["policy_type"])

print("\nPremium collection health:")
spark.sql("""
    SELECT policy_type,
           SUM(total_payments)                      AS total_payments,
           ROUND(SUM(total_due_chf), 0)             AS total_due_chf,
           ROUND(SUM(total_collected_chf), 0)       AS total_collected_chf,
           ROUND(SUM(total_arrears_chf), 0)         AS total_arrears_chf,
           ROUND(AVG(collection_rate_pct), 2)       AS avg_collection_rate_pct
    FROM insurance_gold.premium_collections
    GROUP BY policy_type
    ORDER BY total_due_chf DESC
""").show()

# ─────────────────────────────────────────────
# CELL 9 — Gold: Fraud Summary
# ─────────────────────────────────────────────
# Business question:
# How effective is our fraud detection?
# What are confirmed fraud rates and financial impact?

print(f"\n{'='*50}")
print("BUILDING: FRAUD SUMMARY")
print(f"{'='*50}")

fraud_with_claims = silver_fraud \
    .join(
        silver_claims.select(
            "claim_id", "policy_id", "claim_amount_chf",
            "claim_type", "claim_severity"
        ),
        on="claim_id",
        how="left"
    ) \
    .join(
        silver_policies.select("policy_id", "policy_type"),
        on="policy_id",
        how="left"
    )

gold_fraud_summary = fraud_with_claims \
    .groupBy("signal_type", "score_band", "policy_type", "claim_type") \
    .agg(
        F.count("signal_id")                                .alias("total_signals"),
        F.avg("signal_score")                               .alias("avg_signal_score"),
        F.sum(F.when(F.col("is_confirmed_fraud") == True,
                     F.lit(1)).otherwise(F.lit(0)))         .alias("confirmed_fraud_count"),
        F.sum(F.when(F.col("needs_review") == True,
                     F.lit(1)).otherwise(F.lit(0)))         .alias("needs_review_count"),
        F.sum(F.when(F.col("reviewed") == True,
                     F.lit(1)).otherwise(F.lit(0)))         .alias("reviewed_count"),
        F.sum("claim_amount_chf")                           .alias("total_fraud_exposure_chf"),
        F.avg("claim_amount_chf")                           .alias("avg_fraud_claim_chf"),
    ) \
    .withColumn("avg_signal_score",
                F.round(F.col("avg_signal_score"), 3)) \
    .withColumn("total_fraud_exposure_chf",
                F.round(F.col("total_fraud_exposure_chf"), 2)) \
    .withColumn("avg_fraud_claim_chf",
                F.round(F.col("avg_fraud_claim_chf"), 2)) \
    .withColumn("confirmation_rate_pct",
                F.round(
                    F.col("confirmed_fraud_count") /
                    F.when(F.col("total_signals") > 0,
                           F.col("total_signals"))
                    .otherwise(F.lit(None)) * 100,
                    2
                )) \
    .withColumn("review_completion_rate_pct",
                F.round(
                    F.col("reviewed_count") /
                    F.when(F.col("total_signals") > 0,
                           F.col("total_signals"))
                    .otherwise(F.lit(None)) * 100,
                    2
                ))

gold_fraud_summary = add_gold_audit(gold_fraud_summary)
write_gold(gold_fraud_summary, "fraud_summary", partition_cols=["signal_type"])

print("\nFraud summary by signal type:")
spark.sql("""
    SELECT signal_type,
           SUM(total_signals)                       AS total_signals,
           ROUND(AVG(avg_signal_score), 3)          AS avg_score,
           SUM(confirmed_fraud_count)               AS confirmed_fraud,
           ROUND(SUM(total_fraud_exposure_chf), 0)  AS fraud_exposure_chf,
           ROUND(AVG(confirmation_rate_pct), 2)     AS confirmation_rate_pct
    FROM insurance_gold.fraud_summary
    GROUP BY signal_type
    ORDER BY total_signals DESC
""").show(truncate=False)

# ─────────────────────────────────────────────
# CELL 10 — Gold: Executive Summary
# ─────────────────────────────────────────────
# Single aggregated KPI table for executive reporting.
# One row per month — designed for time-series dashboards.

print(f"\n{'='*50}")
print("BUILDING: EXECUTIVE SUMMARY")
print(f"{'='*50}")

# Monthly policy metrics
monthly_policies = silver_policies \
    .withColumn("month", F.date_format("start_date", "yyyy-MM")) \
    .groupBy("month") \
    .agg(
        F.count("policy_id")            .alias("new_policies"),
        F.sum("annual_premium_chf")     .alias("new_premium_income_chf"),
        F.avg("risk_score")             .alias("avg_risk_score"),
    )

# Monthly claims metrics
monthly_claims = silver_claims \
    .withColumn("month", F.date_format("incident_date", "yyyy-MM")) \
    .groupBy("month") \
    .agg(
        F.count("claim_id")                             .alias("total_claims"),
        F.sum("claim_amount_chf")                       .alias("total_claims_chf"),
        F.avg("claim_amount_chf")                       .alias("avg_claim_chf"),
        F.sum(F.when(F.col("is_fraud_suspected") == True,
                     F.lit(1)).otherwise(F.lit(0)))     .alias("fraud_suspected"),
        F.sum(F.when(F.col("claim_status") == "SETTLED",
                     F.lit(1)).otherwise(F.lit(0)))     .alias("settled_claims"),
    )

# Monthly premium collections
monthly_premiums = silver_premiums \
    .withColumn("month", F.date_format("due_date", "yyyy-MM")) \
    .groupBy("month") \
    .agg(
        F.sum("amount_due_chf")         .alias("total_due_chf"),
        F.sum("amount_paid_chf")        .alias("total_collected_chf"),
        F.sum("arrears_amount_chf")     .alias("total_arrears_chf"),
    )

# Join all monthly metrics
gold_executive = monthly_policies \
    .join(monthly_claims,   on="month", how="full") \
    .join(monthly_premiums, on="month", how="full") \
    .fillna(0) \
    .withColumn("loss_ratio",
                F.round(
                    F.col("total_claims_chf") /
                    F.when(F.col("new_premium_income_chf") > 0,
                           F.col("new_premium_income_chf"))
                    .otherwise(F.lit(None)),
                    4
                )) \
    .withColumn("collection_rate_pct",
                F.round(
                    F.col("total_collected_chf") /
                    F.when(F.col("total_due_chf") > 0,
                           F.col("total_due_chf"))
                    .otherwise(F.lit(None)) * 100,
                    2
                )) \
    .withColumn("fraud_rate_pct",
                F.round(
                    F.col("fraud_suspected") /
                    F.when(F.col("total_claims") > 0,
                           F.col("total_claims"))
                    .otherwise(F.lit(None)) * 100,
                    2
                )) \
    .withColumn("settlement_rate_pct",
                F.round(
                    F.col("settled_claims") /
                    F.when(F.col("total_claims") > 0,
                           F.col("total_claims"))
                    .otherwise(F.lit(None)) * 100,
                    2
                )) \
    .withColumn("avg_claim_chf",
                F.round(F.col("avg_claim_chf"), 2)) \
    .withColumn("avg_risk_score",
                F.round(F.col("avg_risk_score"), 3)) \
    .withColumn("new_premium_income_chf",
                F.round(F.col("new_premium_income_chf"), 2)) \
    .withColumn("total_claims_chf",
                F.round(F.col("total_claims_chf"), 2)) \
    .orderBy("month")

gold_executive = add_gold_audit(gold_executive)
write_gold(gold_executive, "executive_summary")

print("\nExecutive summary — last 6 months sample:")
spark.sql("""
    SELECT month,
           new_policies,
           ROUND(new_premium_income_chf, 0)  AS premium_income_chf,
           total_claims,
           ROUND(total_claims_chf, 0)        AS claims_chf,
           loss_ratio,
           collection_rate_pct,
           fraud_rate_pct,
           settlement_rate_pct
    FROM insurance_gold.executive_summary
    WHERE month >= '2024-07'
    ORDER BY month DESC
""").show()

# ─────────────────────────────────────────────
# CELL 11 — Final Gold Validation Report
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("GOLD LAYER — FINAL VALIDATION REPORT")
print(f"{'='*50}")

gold_tables = [
    "portfolio_summary",
    "claims_kpis",
    "customer_segments",
    "premium_collections",
    "fraud_summary",
    "executive_summary",
]

total = 0
for table in gold_tables:
    count = spark.table(f"{GOLD_DB}.{table}").count()
    total += count
    print(f"  {table:<25} {count:>10,} rows ✅")

print(f"\n  {'TOTAL':<25} {total:>10,} rows")
print(f"  Batch ID: {BATCH_ID}")
print(f"{'='*50}")

# ─────────────────────────────────────────────
# CELL 12 — Full Platform Summary
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("FULL PLATFORM SUMMARY")
print(f"{'='*50}")

layers = {
    "insurance_bronze": ["customers","policies","claims","premiums","fraud_signals"],
    "insurance_silver": ["customers","policies","claims","premiums","fraud_signals","claims_enriched"],
    "insurance_gold":   gold_tables,
}

grand_total = 0
for db, tables in layers.items():
    layer_total = 0
    print(f"\n  {db.upper()}")
    for table in tables:
        try:
            count = spark.table(f"{db}.{table}").count()
            layer_total += count
            grand_total += count
            print(f"    {table:<25} {count:>10,} rows")
        except Exception as e:
            print(f"    {table:<25} ERROR ❌")
    print(f"    {'--- Layer Total ---':<25} {layer_total:>10,} rows")

print(f"\n  {'GRAND TOTAL':<25} {grand_total:>10,} rows")
print(f"  Batch ID: {BATCH_ID}")
print(f"{'='*50}")
print(f"\n✅ Gold layer complete — Batch ID: {BATCH_ID}")