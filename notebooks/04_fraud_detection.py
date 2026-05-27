# Databricks notebook source
"""
notebooks/04_fraud_detection.py
================================
Fraud Detection Layer — Insurance Data Platform
================================================
Purpose : Build a multi-signal composite fraud scoring engine
          on top of Silver claims data.
Layer   : Fraud (Analytical)
Reads   : insurance_silver.claims
          insurance_silver.policies
          insurance_silver.customers
          insurance_silver.fraud_signals
Writes  : insurance_fraud.* Delta tables

Depends on:
  src/config.py     — FRAUD_WEIGHTS, FRAUD_THRESHOLDS, BATCH_ID
  src/utils.py      — write_delta, apply_delta_settings
  src/audit.py      — add_fraud_audit
  src/monitoring.py — log_pipeline_error

Fraud scoring architecture:
  Four independent signals → Composite weighted score → Risk tier

  Signal 1: Rule-based scoring
    Known fraud patterns — late submission, high amounts,
    third party involvement, policy inception timing

  Signal 2: Statistical anomaly detection
    Z-score analysis on claim amounts per policy type
    Claims beyond 2 standard deviations flagged as anomalous

  Signal 3: Behavioural analysis
    Customer claim velocity — how many claims in lifetime
    Prior fraud history — previous fraud flags on record
    Submission patterns — consistently late submissions

  Signal 4: Network scoring
    Duplicate detection — same customer, multiple policies
    Claims clustering — multiple claims in short window

  Composite score:
    Weighted combination of all 4 signals (weights sum to 1.0)
    CRITICAL: score >= 0.75
    HIGH:     score >= 0.50
    MEDIUM:   score >= 0.25
    LOW:      score <  0.25

  Investigation queue:
    Priority-ordered list for fraud analysts
    CRITICAL claims investigated first
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
# Purpose : Import all required libraries and src/ modules.
#           Fraud layer needs statistical functions and
#           window operations for velocity detection.
# ═══════════════════════════════════════════════════════════

import logging
from datetime import datetime

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window

from src.config     import (BATCH_ID, DATABASES,
                             DELTA_SETTINGS, FRAUD_WEIGHTS,
                             FRAUD_THRESHOLDS)
from src.utils      import (write_delta, apply_delta_settings)
from src.audit      import add_fraud_audit
from src.monitoring import log_pipeline_error

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("fraud_detection")

# ── Convenience aliases ───────────────────────────────────
SILVER_DB = DATABASES["silver"]
FRAUD_DB  = DATABASES["fraud"]

# ── Fraud scoring weights from config ────────────────────
# Weights defined in src/config.py — change there not here
# Must sum to 1.0 — validated at import time in config.py
WEIGHTS = FRAUD_WEIGHTS
log.info(f"Fraud weights: {WEIGHTS}")

print(f"✅ All imports successful")
print(f"   Batch ID  : {BATCH_ID}")
print(f"   Source DB : {SILVER_DB}")
print(f"   Target DB : {FRAUD_DB}")
print(f"   Weights   : {WEIGHTS}")


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 3 — Database Setup and Silver Validation
# Purpose : Create Fraud database and validate all required
#           Silver tables exist before scoring begins.
# ═══════════════════════════════════════════════════════════

spark.sql(f"DROP DATABASE IF EXISTS {FRAUD_DB} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {FRAUD_DB}")
spark.sql(f"USE {FRAUD_DB}")
apply_delta_settings(spark, DELTA_SETTINGS)

# Validate Silver tables exist
print("\nValidating Silver tables exist...")
required_silver = [
    "claims", "policies", "customers", "fraud_signals"
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

print(f"\n✅ Database '{FRAUD_DB}' ready")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 4 — Load Silver Tables
# Purpose : Load all required Silver tables into memory.
#           Loading once here avoids repeated table reads
#           across multiple scoring cells.
# ═══════════════════════════════════════════════════════════

silver_claims    = spark.table(f"{SILVER_DB}.claims")
silver_policies  = spark.table(f"{SILVER_DB}.policies")
silver_customers = spark.table(f"{SILVER_DB}.customers")
silver_fraud     = spark.table(f"{SILVER_DB}.fraud_signals")

# Cache claims — used by multiple scoring signals
# Caching avoids repeated Delta reads for the same data
silver_claims.cache()

print(f"✅ Silver tables loaded")
print(f"   claims:    {silver_claims.count():,}")
print(f"   policies:  {silver_policies.count():,}")
print(f"   customers: {silver_customers.count():,}")
print(f"   signals:   {silver_fraud.count():,}")

# ═══════════════════════════════════════════════════════════
# CELL 5 — Fraud Orchestrator
# Purpose : Reusable function for writing Fraud tables.
#           Adds audit columns and handles errors consistently.
# ═══════════════════════════════════════════════════════════

def build_fraud_table(sdf,
                      table_name: str,
                      partition_cols: list = None) -> int:
    """
    Write fraud scoring DataFrame to Fraud Delta table.
    Adds audit columns and handles errors consistently.

    Args:
        sdf           : Fraud scoring DataFrame
        table_name    : Target table name
        partition_cols: Optional partition columns

    Returns:
        Row count written
    """
    print(f"\n{'─'*50}")
    print(f"  BUILDING: {table_name.upper()}")
    print(f"{'─'*50}")

    try:
        sdf   = add_fraud_audit(sdf, BATCH_ID)
        count = write_delta(
            sdf, FRAUD_DB, table_name, partition_cols
        )
        print(f"  ✅ Complete: {count:,} rows")
        return count

    except Exception as e:
        log.error(f"[{table_name}] FAILED: {str(e)}")
        log_pipeline_error(spark, table_name, e, BATCH_ID)
        raise


print("✅ Fraud orchestrator ready")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 6 — Signal 1: Rule-Based Scoring
# Purpose : Score each claim against known fraud patterns.
#           Rules are based on insurance industry experience.
#           Each rule that fires adds to the rule score.
#
# Rules applied:
#   late_submission     — submitted > 45 days after incident
#   high_amount         — claim > 100,000 CHF
#   third_party         — third party involved (common in fraud)
#   already_suspected   — already flagged in Bronze
#   catastrophic        — severity = CATASTROPHIC
#   very_delayed        — submission_delay_band = VERY_DELAYED
#   no_police_report    — high amount but no police report
#
# Scoring:
#   Each fired rule contributes equally to rule_score
#   rule_score = rules_fired / total_rules (0.0 to 1.0)
# ═══════════════════════════════════════════════════════════

try:
    RULE_DEFINITIONS = [
        # (rule_name, condition_expression, weight)
        ("late_submission",
         "days_to_submit > 45",
         0.15),
        ("high_amount",
         "claim_amount_chf > 100000",
         0.20),
        ("third_party_involved",
         "third_party_involved = true",
         0.10),
        ("already_fraud_suspected",
         "is_fraud_suspected = true",
         0.25),
        ("catastrophic_severity",
         "claim_severity = 'CATASTROPHIC'",
         0.15),
        ("very_delayed_submission",
         "submission_delay_band = 'VERY_DELAYED'",
         0.10),
        ("high_amount_no_police",
         "claim_amount_chf > 50000 AND police_report_filed = false",
         0.05),
    ]

    # Apply each rule and calculate weighted score
    rule_sdf = silver_claims.select(
        "claim_id", "policy_id", "customer_id",
        "claim_amount_chf", "days_to_submit",
        "is_fraud_suspected", "third_party_involved",
        "police_report_filed", "claim_severity",
        "submission_delay_band"
    )

    # Add binary flag for each rule (1 = fired, 0 = not fired)
    for rule_name, condition, _ in RULE_DEFINITIONS:
        rule_sdf = rule_sdf.withColumn(
            f"rule_{rule_name}",
            F.when(F.expr(condition), F.lit(1))
             .otherwise(F.lit(0))
        )

    # Calculate weighted rule score
    # Each rule has a weight — weighted sum = rule_score
    total_weight = sum(w for _, _, w in RULE_DEFINITIONS)
    score_expr   = sum(
        F.col(f"rule_{name}") * F.lit(weight)
        for name, _, weight in RULE_DEFINITIONS
    ) / F.lit(total_weight)

    rule_cols   = [f"rule_{name}" for name, _, _ in RULE_DEFINITIONS]
    rules_fired = sum(F.col(c) for c in rule_cols)

    rule_sdf = rule_sdf \
        .withColumn("rule_score",
                    F.round(score_expr, 4)) \
        .withColumn("rules_fired_count",
                    rules_fired) \
        .select(
            "claim_id", "rule_score",
            "rules_fired_count", *rule_cols
        )

    build_fraud_table(rule_sdf, "fraud_rule_hits")

except Exception as e:
    log.error(f"Rule scoring FAILED: {str(e)}")
    log_pipeline_error(spark, "fraud_rule_hits", e, BATCH_ID)
    raise

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 7 — Signal 2: Statistical Anomaly Detection
# Purpose : Detect claims with statistically unusual amounts
#           using Z-score analysis per policy type.
#
# How Z-score works:
#   Z = (value - mean) / standard_deviation
#   Z > 2.0 means value is 2 standard deviations above mean
#   Z > 2.0 flags as statistical anomaly
#
# Why per policy type:
#   A 50,000 CHF claim is normal for commercial insurance
#   but highly unusual for travel insurance.
#   Grouping by policy_type makes comparison fair.
#
# Output:
#   z_score_amount       — Z-score of claim amount
#   is_statistical_anomaly — Z > threshold (default 2.0)
#   statistical_score    — normalised 0-1 score
# ═══════════════════════════════════════════════════════════

try:
    # Join claims to policies to get policy_type for grouping
    claims_with_type = silver_claims \
        .join(
            silver_policies.select("policy_id", "policy_type"),
            on="policy_id", how="left"
        )

    # Calculate mean and std per policy type
    # Window function — no groupBy needed, preserves all rows
    policy_window = Window.partitionBy("policy_type")

    stat_sdf = claims_with_type \
        .withColumn("mean_amount",
                    F.avg("claim_amount_chf")
                     .over(policy_window)) \
        .withColumn("std_amount",
                    F.stddev("claim_amount_chf")
                     .over(policy_window)) \
        .withColumn("z_score_amount",
                    F.when(
                        F.col("std_amount") > 0,
                        F.round(
                            (F.col("claim_amount_chf") -
                             F.col("mean_amount")) /
                            F.col("std_amount"),
                            3
                        )
                    ).otherwise(F.lit(0.0))) \
        .withColumn("is_statistical_anomaly",
                    F.when(
                        F.col("z_score_amount") >
                        F.lit(FRAUD_THRESHOLDS["z_score_anomaly"]),
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("statistical_score",
                    F.round(
                        F.when(
                            F.col("z_score_amount") > 0,
                            F.least(
                                F.col("z_score_amount") /
                                F.lit(5.0),
                                F.lit(1.0)
                            )
                        ).otherwise(F.lit(0.0)),
                        4
                    )) \
        .select(
            "claim_id", "policy_type",
            "claim_amount_chf", "mean_amount",
            "std_amount", "z_score_amount",
            "is_statistical_anomaly", "statistical_score"
        )

    build_fraud_table(
        stat_sdf, "fraud_statistical_scores"
    )

except Exception as e:
    log.error(f"Statistical scoring FAILED: {str(e)}")
    log_pipeline_error(
        spark, "fraud_statistical_scores", e, BATCH_ID
    )
    raise

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 8 — Signal 3: Behavioural Analysis
# Purpose : Analyse customer claim behaviour patterns.
#           Frequent claimants are higher fraud risk.
#
# Behavioural signals:
#   lifetime_claim_count — total claims ever by this customer
#   prior_fraud_flags    — how many previous fraud suspicions
#   avg_days_to_submit   — consistently late = suspicious
#   high_value_claim_count — how many large claims
#
# Scoring:
#   behavioural_score derived from combination of signals
#   Normalised to 0-1 range
# ═══════════════════════════════════════════════════════════

try:
    # Customer-level claim statistics
    customer_stats = silver_claims \
        .groupBy("customer_id") \
        .agg(
            F.count("claim_id")
                .alias("lifetime_claim_count"),
            F.sum(F.when(
                F.col("is_fraud_suspected") == F.lit(True),
                F.lit(1)
            ).otherwise(F.lit(0)))
                .alias("prior_fraud_flags"),
            F.round(F.avg("days_to_submit"), 1)
                .alias("avg_days_to_submit"),
            F.sum(F.when(
                F.col("is_high_value_claim") == F.lit(True),
                F.lit(1)
            ).otherwise(F.lit(0)))
                .alias("high_value_claim_count"),
            F.round(F.avg("claim_amount_chf"), 2)
                .alias("avg_claim_amount_chf"),
        )

    # Join back to claims — one row per claim
    behav_sdf = silver_claims \
        .select(
            "claim_id", "customer_id",
            "claim_amount_chf", "days_to_submit"
        ) \
        .join(customer_stats, on="customer_id", how="left") \
        .withColumn("behavioural_flags_count",
                    (F.when(
                        F.col("lifetime_claim_count") >=
                        F.lit(FRAUD_THRESHOLDS[
                            "max_claims_per_window"
                        ]),
                        F.lit(1)
                    ).otherwise(F.lit(0))) +
                    (F.when(
                        F.col("prior_fraud_flags") > 0,
                        F.lit(1)
                    ).otherwise(F.lit(0))) +
                    (F.when(
                        F.col("avg_days_to_submit") > 30,
                        F.lit(1)
                    ).otherwise(F.lit(0))) +
                    (F.when(
                        F.col("high_value_claim_count") > 1,
                        F.lit(1)
                    ).otherwise(F.lit(0)))) \
        .withColumn("behavioural_score",
                    F.round(
                        F.least(
                            F.col("behavioural_flags_count") /
                            F.lit(4.0),
                            F.lit(1.0)
                        ),
                        4
                    )) \
        .select(
            "claim_id",
            "lifetime_claim_count",
            "prior_fraud_flags",
            "avg_days_to_submit",
            "high_value_claim_count",
            "behavioural_flags_count",
            "behavioural_score",
        )

    build_fraud_table(behav_sdf, "fraud_behavioural_flags")

except Exception as e:
    log.error(f"Behavioural scoring FAILED: {str(e)}")
    log_pipeline_error(
        spark, "fraud_behavioural_flags", e, BATCH_ID
    )
    raise


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 9 — Signal 4: Network Scoring
# Purpose : Detect suspicious patterns across the claim network.
#           Fraudsters often make multiple claims in short windows
#           or use the same details across different policies.
#
# Network signals:
#   claims_in_30d_window — claims by same customer in 30 days
#   policies_with_claims — how many policies have claims
#   network_flags_count  — total network red flags
#
# Scoring:
#   network_score normalised to 0-1
# ═══════════════════════════════════════════════════════════

try:
    # Claims in rolling 30-day window per customer
    # Window ordered by incident date
    date_window = Window \
        .partitionBy("customer_id") \
        .orderBy(F.col("incident_date").cast("long")) \
        .rangeBetween(
            -FRAUD_THRESHOLDS["velocity_window_days"] * 86400,
            0
        )

    # Policies with claims per customer
    policies_with_claims = silver_claims \
        .groupBy("customer_id") \
        .agg(
            F.countDistinct("policy_id")
                .alias("policies_with_claims")
        )

    network_sdf = silver_claims \
        .select(
            "claim_id", "customer_id",
            "incident_date", "policy_id",
            "claim_amount_chf"
        ) \
        .withColumn("claims_in_30d_window",
                    F.count("claim_id").over(date_window)) \
        .join(
            policies_with_claims,
            on="customer_id", how="left"
        ) \
        .withColumn("network_flags_count",
                    F.when(
                        F.col("claims_in_30d_window") >=
                        F.lit(FRAUD_THRESHOLDS[
                            "max_claims_per_window"
                        ]),
                        F.lit(1)
                    ).otherwise(F.lit(0)) +
                    F.when(
                        F.col("policies_with_claims") > 2,
                        F.lit(1)
                    ).otherwise(F.lit(0))) \
        .withColumn("network_score",
                    F.round(
                        F.least(
                            F.col("network_flags_count") /
                            F.lit(2.0),
                            F.lit(1.0)
                        ),
                        4
                    )) \
        .select(
            "claim_id",
            "claims_in_30d_window",
            "policies_with_claims",
            "network_flags_count",
            "network_score",
        )

    build_fraud_table(network_sdf, "fraud_network_scores")

except Exception as e:
    log.error(f"Network scoring FAILED: {str(e)}")
    log_pipeline_error(
        spark, "fraud_network_scores", e, BATCH_ID
    )
    raise

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 10 — Composite Fraud Score
# Purpose : Combine all 4 signals into one weighted score.
#           This is the final fraud score used for decisions.
#
# Formula:
#   composite_score =
#     (rule_score        × 0.35) +
#     (statistical_score × 0.25) +
#     (behavioural_score × 0.25) +
#     (network_score     × 0.15)
#
# Weights from src/config.py — change there not here.
# Must sum to 1.0 — validated at config import time.
#
# Risk tiers:
#   CRITICAL : score >= 0.75 — immediate investigation
#   HIGH     : score >= 0.50 — review within 24 hours
#   MEDIUM   : score >= 0.25 — review within 5 days
#   LOW      : score <  0.25 — no action required
# ═══════════════════════════════════════════════════════════

try:
    rule_sdf  = spark.table(f"{FRAUD_DB}.fraud_rule_hits") \
                     .select("claim_id", "rule_score",
                             "rules_fired_count")
    stat_sdf  = spark.table(
                    f"{FRAUD_DB}.fraud_statistical_scores"
                ).select("claim_id", "statistical_score",
                         "z_score_amount",
                         "is_statistical_anomaly")
    behav_sdf = spark.table(
                    f"{FRAUD_DB}.fraud_behavioural_flags"
                ).select("claim_id", "behavioural_score",
                         "lifetime_claim_count",
                         "prior_fraud_flags",
                         "behavioural_flags_count")
    net_sdf   = spark.table(
                    f"{FRAUD_DB}.fraud_network_scores"
                ).select("claim_id", "network_score",
                         "network_flags_count",
                         "claims_in_30d_window")

    composite = silver_claims \
        .select(
            "claim_id", "policy_id", "customer_id",
            "claim_amount_chf", "claim_status",
            "claim_type", "claim_severity",
            "incident_date", "submitted_date",
            "days_to_submit", "is_fraud_suspected",
            "fraud_indicators", "handler_id"
        ) \
        .join(rule_sdf,  on="claim_id", how="left") \
        .join(stat_sdf,  on="claim_id", how="left") \
        .join(behav_sdf, on="claim_id", how="left") \
        .join(net_sdf,   on="claim_id", how="left") \
        .fillna(0.0, subset=[
            "rule_score", "statistical_score",
            "behavioural_score", "network_score"
        ]) \
        .withColumn("composite_score",
                    F.round(
                        (F.col("rule_score") *
                         F.lit(WEIGHTS["rule_score"])) +
                        (F.col("statistical_score") *
                         F.lit(WEIGHTS["statistical_score"])) +
                        (F.col("behavioural_score") *
                         F.lit(WEIGHTS["behavioural_score"])) +
                        (F.col("network_score") *
                         F.lit(WEIGHTS["network_score"])),
                        4
                    )) \
        .withColumn("fraud_risk_tier",
                    F.when(
                        F.col("composite_score") >=
                        F.lit(FRAUD_THRESHOLDS["critical"]),
                        F.lit("CRITICAL")
                    ).when(
                        F.col("composite_score") >=
                        F.lit(FRAUD_THRESHOLDS["high"]),
                        F.lit("HIGH")
                    ).when(
                        F.col("composite_score") >=
                        F.lit(FRAUD_THRESHOLDS["medium"]),
                        F.lit("MEDIUM")
                    ).otherwise(F.lit("LOW"))) \
        .withColumn("recommend_investigation",
                    F.when(
                        (F.col("composite_score") >=
                         F.lit(FRAUD_THRESHOLDS["high"])) |
                        (F.col("is_fraud_suspected") ==
                         F.lit(True)),
                        F.lit(True)
                    ).otherwise(F.lit(False))) \
        .withColumn("investigation_priority",
                    F.when(
                        F.col("fraud_risk_tier") == "CRITICAL",
                        F.lit(1)
                    ).when(
                        F.col("fraud_risk_tier") == "HIGH",
                        F.lit(2)
                    ).when(
                        F.col("fraud_risk_tier") == "MEDIUM",
                        F.lit(3)
                    ).otherwise(F.lit(4)))

    build_fraud_table(
        composite, "fraud_composite_scores",
        partition_cols=["fraud_risk_tier"]
    )

except Exception as e:
    log.error(f"Composite scoring FAILED: {str(e)}")
    log_pipeline_error(
        spark, "fraud_composite_scores", e, BATCH_ID
    )
    raise

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 11 — Investigation Queue
# Purpose : Build prioritised list for fraud analysts.
#           Sorted by investigation_priority then score.
#           Analysts work top-to-bottom — highest risk first.
#
# Reads   : insurance_fraud.fraud_composite_scores
# Writes  : insurance_fraud.fraud_investigation_queue
#
# Includes:
#   All claims recommended for investigation
#   Full context — customer, policy, amount, signals
#   Priority ranking for analyst workflow
# ═══════════════════════════════════════════════════════════

try:
    composite_sdf = spark.table(
        f"{FRAUD_DB}.fraud_composite_scores"
    )

    investigation_queue = composite_sdf \
        .filter(
            F.col("recommend_investigation") == F.lit(True)
        ) \
        .join(
            silver_customers.select(
                "customer_id", "full_name",
                "customer_segment", "city"
            ),
            on="customer_id", how="left"
        ) \
        .join(
            silver_policies.select(
                "policy_id", "policy_type",
                "annual_premium_chf", "risk_band"
            ),
            on="policy_id", how="left"
        ) \
        .select(
            "investigation_priority",
            "fraud_risk_tier",
            "composite_score",
            "claim_id",
            "full_name",
            "customer_segment",
            "policy_type",
            "claim_type",
            "claim_severity",
            "claim_amount_chf",
            "annual_premium_chf",
            "risk_band",
            "incident_date",
            "days_to_submit",
            "is_fraud_suspected",
            "rule_score",
            "statistical_score",
            "behavioural_score",
            "network_score",
            "rules_fired_count",
            "handler_id",
            "city",
        ) \
        .orderBy(
            "investigation_priority",
            F.col("composite_score").desc()
        )

    build_fraud_table(
        investigation_queue,
        "fraud_investigation_queue",
        partition_cols=["fraud_risk_tier"]
    )

except Exception as e:
    log.error(f"Investigation queue FAILED: {str(e)}")
    log_pipeline_error(
        spark, "fraud_investigation_queue", e, BATCH_ID
    )
    raise


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 12 — Fraud Layer Validation Report
# Purpose : Post-scoring checks.
#           Verifies all Fraud tables built correctly.
#           Reviews score distribution for sanity.
#           Checks pipeline errors.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("FRAUD LAYER — VALIDATION REPORT")
print(f"{'='*60}")

fraud_tables = [
    "fraud_rule_hits",
    "fraud_statistical_scores",
    "fraud_behavioural_flags",
    "fraud_network_scores",
    "fraud_composite_scores",
    "fraud_investigation_queue",
]

total = 0
print("\n FRAUD TABLES:")
for table in fraud_tables:
    try:
        count = spark.table(f"{FRAUD_DB}.{table}").count()
        total += count
        print(f"  {table:<30} {count:>8,} rows ✅")
    except Exception as e:
        print(f"  {table:<30} ERROR ❌")

print(f"\n  {'TOTAL':<30} {total:>8,} rows")

# ── Score distribution ───────────────────────────────────
print("\n COMPOSITE SCORE DISTRIBUTION:")
spark.sql(f"""
    SELECT fraud_risk_tier,
           COUNT(*)                           AS claim_count,
           ROUND(AVG(composite_score), 4)     AS avg_score,
           ROUND(MIN(composite_score), 4)     AS min_score,
           ROUND(MAX(composite_score), 4)     AS max_score,
           SUM(CASE WHEN is_fraud_suspected
                    THEN 1 ELSE 0 END)        AS confirmed_suspected
    FROM {FRAUD_DB}.fraud_composite_scores
    GROUP BY fraud_risk_tier
    ORDER BY avg_score DESC
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
        f"Fraud pipeline failed with {error_count} error(s)."
    )


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 13 — Investigation Queue Preview
# Purpose : Show top priority claims for investigation.
#           This is what a fraud analyst sees first.
# ═══════════════════════════════════════════════════════════

print("\nTop 10 Priority Claims for Investigation:")
spark.sql(f"""
    SELECT investigation_priority,
           fraud_risk_tier,
           ROUND(composite_score, 4)  AS score,
           claim_id,
           full_name,
           policy_type,
           claim_type,
           ROUND(claim_amount_chf, 0) AS claim_amount,
           rule_score,
           statistical_score,
           behavioural_score,
           network_score
    FROM {FRAUD_DB}.fraud_investigation_queue
    ORDER BY investigation_priority,
             composite_score DESC
    LIMIT 10
""").show(truncate=False)

print("\nFraud signal type breakdown:")
spark.sql(f"""
    SELECT signal_type,
           COUNT(*)                       AS total_signals,
           ROUND(AVG(signal_score), 3)    AS avg_score,
           SUM(CASE WHEN is_confirmed_fraud
                    THEN 1 ELSE 0 END)    AS confirmed_fraud
    FROM {SILVER_DB}.fraud_signals
    GROUP BY signal_type
    ORDER BY total_signals DESC
""").show(truncate=False)

# Uncache claims now that all scoring is complete
silver_claims.unpersist()

print(f"\n✅ Fraud layer complete — Batch ID: {BATCH_ID}")