# Databricks notebook source
"""
04_fraud_detection.py
=====================
Production-grade Fraud Detection Layer for Insurance Data Platform.

Approaches:
1. Rule-based detection    — known fraud patterns
2. Statistical scoring     — Z-score anomaly detection
3. Behavioural flags       — customer claim frequency patterns
4. Network signals         — multi-claim customer detection
5. Composite scoring       — weighted combination of all signals
6. Investigation queue     — prioritised list for fraud investigators

Architecture:
    Silver Delta → Fraud Engine → Fraud Delta Tables

NOTE: All boolean expressions use PySpark operators:
      & for AND, | for OR, ~ for NOT
      Never use Python's and/or/not with Spark columns.
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
log = logging.getLogger("insurance_fraud")

SILVER_DB  = "insurance_silver"
FRAUD_DB   = "insurance_fraud"
BATCH_ID   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

WEIGHTS = {
    "rule_score":        0.35,
    "statistical_score": 0.25,
    "behavioural_score": 0.25,
    "network_score":     0.15,
}

print(f"Fraud detection starting — Batch ID: {BATCH_ID}")

# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 2 — Setup Fraud Database
# ─────────────────────────────────────────────

spark.sql(f"DROP DATABASE IF EXISTS {FRAUD_DB} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {FRAUD_DB}")
spark.sql(f"USE {FRAUD_DB}")
print(f"✅ Database '{FRAUD_DB}' ready")

# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 3 — Utility Functions
# ─────────────────────────────────────────────

def add_fraud_audit(sdf):
    return sdf \
        .withColumn("_fraud_batch_id",       F.lit(BATCH_ID)) \
        .withColumn("_fraud_load_timestamp", F.current_timestamp()) \
        .withColumn("_fraud_source_layer",   F.lit("silver"))


def write_fraud(sdf, table: str, partition_cols: list = None):
    full_table = f"{FRAUD_DB}.{table}"
    writer = sdf.write.format("delta").mode("overwrite")
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table)
    count = spark.table(full_table).count()
    log.info(f"  ✅ {full_table} → {count:,} rows")
    return count


print("✅ Utility functions defined")


# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 4 — Load Silver Tables
# ─────────────────────────────────────────────

print("\nLoading Silver tables...")

silver_claims    = spark.table(f"{SILVER_DB}.claims")
silver_policies  = spark.table(f"{SILVER_DB}.policies")
silver_customers = spark.table(f"{SILVER_DB}.customers")
silver_enriched  = spark.table(f"{SILVER_DB}.claims_enriched")

print(f"  Claims:    {silver_claims.count():,}")
print(f"  Policies:  {silver_policies.count():,}")
print(f"  Customers: {silver_customers.count():,}")
print("✅ Silver tables loaded")


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 5 — Rule-Based Detection
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("RUNNING: RULE-BASED DETECTION")
print(f"{'='*50}")

HIGH_CLAIM_THRESHOLD  = 75_000.0
VERY_HIGH_CLAIM       = 100_000.0
LATE_SUBMISSION_DAYS  = 45
VERY_LATE_SUBMISSION  = 55

rule_based = silver_claims \
    .join(
        silver_policies.select(
            "policy_id", "coverage_amount_chf",
            "deductible_chf", "start_date", "risk_score"
        ),
        on="policy_id",
        how="left"
    ) \
    .withColumn("rule_high_claim_amount",
                F.when(F.col("claim_amount_chf") > F.lit(HIGH_CLAIM_THRESHOLD),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_claim_exceeds_coverage",
                F.when(F.col("claim_amount_chf") > F.col("coverage_amount_chf"),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_late_submission",
                F.when(F.col("days_to_submit") > F.lit(LATE_SUBMISSION_DAYS),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_third_party_high_value",
                F.when(
                    (F.col("third_party_involved") == True) &
                    (F.col("claim_amount_chf") > F.lit(HIGH_CLAIM_THRESHOLD)),
                    F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_no_police_report_high_value",
                F.when(
                    (F.col("police_report_filed") == False) &
                    (F.col("claim_amount_chf") > F.lit(HIGH_CLAIM_THRESHOLD)),
                    F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_high_risk_policy",
                F.when(F.col("risk_score") > F.lit(0.8),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_very_late_submission",
                F.when(F.col("days_to_submit") > F.lit(VERY_LATE_SUBMISSION),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rule_very_high_claim",
                F.when(F.col("claim_amount_chf") > F.lit(VERY_HIGH_CLAIM),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("rules_fired_count",
                F.col("rule_high_claim_amount") +
                F.col("rule_claim_exceeds_coverage") +
                F.col("rule_late_submission") +
                F.col("rule_third_party_high_value") +
                F.col("rule_no_police_report_high_value") +
                F.col("rule_high_risk_policy") +
                F.col("rule_very_late_submission") +
                F.col("rule_very_high_claim")) \
    .withColumn("rule_score",
                F.least(F.lit(1.0),
                        F.greatest(F.lit(0.0),
                                   F.col("rules_fired_count") / F.lit(8.0)))) \
    .select(
        "claim_id", "policy_id", "customer_id",
        "claim_amount_chf", "days_to_submit",
        "rule_high_claim_amount",
        "rule_claim_exceeds_coverage",
        "rule_late_submission",
        "rule_third_party_high_value",
        "rule_no_police_report_high_value",
        "rule_high_risk_policy",
        "rule_very_late_submission",
        "rule_very_high_claim",
        "rules_fired_count",
        F.round("rule_score", 3).alias("rule_score")
    )

rule_based = add_fraud_audit(rule_based)
write_fraud(rule_based, "fraud_rule_hits")

print("\nRule hit distribution:")
spark.sql("""
    SELECT rules_fired_count,
           COUNT(*)                    AS claim_count,
           ROUND(AVG(rule_score), 3)   AS avg_rule_score
    FROM insurance_fraud.fraud_rule_hits
    GROUP BY rules_fired_count
    ORDER BY rules_fired_count DESC
""").show()


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 6 — Statistical Anomaly Scoring
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("RUNNING: STATISTICAL ANOMALY SCORING")
print(f"{'='*50}")

window_claim_type = Window.partitionBy("claim_type")

statistical_scores = silver_claims \
    .withColumn("mean_amount_by_type",
                F.avg("claim_amount_chf").over(window_claim_type)) \
    .withColumn("stddev_amount_by_type",
                F.stddev("claim_amount_chf").over(window_claim_type)) \
    .withColumn("z_score_amount",
                F.when(
                    F.col("stddev_amount_by_type") > F.lit(0.0),
                    F.abs(
                        (F.col("claim_amount_chf") - F.col("mean_amount_by_type")) /
                        F.col("stddev_amount_by_type")
                    )
                ).otherwise(F.lit(0.0))) \
    .withColumn("mean_days_by_type",
                F.avg("days_to_submit").over(window_claim_type)) \
    .withColumn("stddev_days_by_type",
                F.stddev("days_to_submit").over(window_claim_type)) \
    .withColumn("z_score_days",
                F.when(
                    F.col("stddev_days_by_type") > F.lit(0.0),
                    F.abs(
                        (F.col("days_to_submit") - F.col("mean_days_by_type")) /
                        F.col("stddev_days_by_type")
                    )
                ).otherwise(F.lit(0.0))) \
    .withColumn("combined_z_score",
                (F.col("z_score_amount") * F.lit(0.7)) +
                (F.col("z_score_days")   * F.lit(0.3))) \
    .withColumn("statistical_score",
                F.least(F.lit(1.0),
                        F.greatest(F.lit(0.0),
                                   F.lit(1.0) - (F.lit(1.0) /
                                   (F.lit(1.0) + F.col("combined_z_score")))))) \
    .withColumn("is_statistical_anomaly",
                F.when(F.col("z_score_amount") > F.lit(2.0),
                       F.lit(True)).otherwise(F.lit(False))) \
    .select(
        "claim_id", "claim_type",
        "claim_amount_chf",
        F.round("mean_amount_by_type",   2).alias("mean_amount_by_type"),
        F.round("stddev_amount_by_type", 2).alias("stddev_amount_by_type"),
        F.round("z_score_amount",        3).alias("z_score_amount"),
        F.round("z_score_days",          3).alias("z_score_days"),
        F.round("combined_z_score",      3).alias("combined_z_score"),
        F.round("statistical_score",     3).alias("statistical_score"),
        "is_statistical_anomaly"
    )

statistical_scores = add_fraud_audit(statistical_scores)
write_fraud(statistical_scores, "fraud_statistical_scores")

print("\nStatistical anomaly summary:")
spark.sql("""
    SELECT claim_type,
           COUNT(*)                            AS total_claims,
           SUM(CASE WHEN is_statistical_anomaly
                    THEN 1 ELSE 0 END)         AS anomalies_detected,
           ROUND(AVG(z_score_amount), 3)       AS avg_z_score,
           ROUND(AVG(statistical_score), 3)    AS avg_stat_score
    FROM insurance_fraud.fraud_statistical_scores
    GROUP BY claim_type
    ORDER BY anomalies_detected DESC
""").show()


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 7 — Behavioural Flag Scoring
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("RUNNING: BEHAVIOURAL FLAG SCORING")
print(f"{'='*50}")

customer_claim_stats = silver_claims \
    .groupBy("customer_id") \
    .agg(
        F.count("claim_id")                 .alias("lifetime_claim_count"),
        F.sum("claim_amount_chf")           .alias("lifetime_claimed_chf"),
        F.avg("claim_amount_chf")           .alias("avg_claim_amount"),
        F.min("incident_date")              .alias("first_claim_date"),
        F.max("incident_date")              .alias("last_claim_date"),
        F.countDistinct("policy_id")        .alias("policies_claimed_on"),
        F.countDistinct("claim_type")       .alias("claim_type_diversity"),
        F.sum(F.when(F.col("is_fraud_suspected") == True,
                     F.lit(1)).otherwise(F.lit(0))).alias("prior_fraud_flags"),
    ) \
    .withColumn("claim_span_days",
                F.datediff(F.col("last_claim_date"), F.col("first_claim_date"))) \
    .withColumn("flag_high_frequency",
                F.when(F.col("lifetime_claim_count") > F.lit(3),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("flag_high_lifetime_value",
                F.when(F.col("lifetime_claimed_chf") > F.lit(200_000.0),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("flag_multi_policy_claims",
                F.when(F.col("policies_claimed_on") > F.lit(2),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("flag_prior_fraud",
                F.when(F.col("prior_fraud_flags") > F.lit(0),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("flag_rapid_successive",
                F.when(
                    (F.col("lifetime_claim_count") > F.lit(1)) &
                    (F.col("claim_span_days") < F.lit(90)),
                    F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("behavioural_flags_count",
                F.col("flag_high_frequency") +
                F.col("flag_high_lifetime_value") +
                F.col("flag_multi_policy_claims") +
                F.col("flag_prior_fraud") +
                F.col("flag_rapid_successive")) \
    .withColumn("behavioural_score",
                F.least(F.lit(1.0),
                        F.greatest(F.lit(0.0),
                                   F.col("behavioural_flags_count") / F.lit(5.0))))

behavioural_flags = silver_claims \
    .select("claim_id", "customer_id") \
    .join(customer_claim_stats, on="customer_id", how="left")

behavioural_flags = add_fraud_audit(behavioural_flags)
write_fraud(behavioural_flags, "fraud_behavioural_flags")

print("\nBehavioural flag distribution:")
spark.sql("""
    SELECT behavioural_flags_count,
           COUNT(DISTINCT customer_id)     AS customer_count,
           COUNT(claim_id)                 AS claim_count,
           ROUND(AVG(behavioural_score),3) AS avg_behavioural_score
    FROM insurance_fraud.fraud_behavioural_flags
    GROUP BY behavioural_flags_count
    ORDER BY behavioural_flags_count DESC
""").show()


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 8 — Network Signal Scoring
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("RUNNING: NETWORK SIGNAL SCORING")
print(f"{'='*50}")

policy_claim_concentration = silver_claims \
    .groupBy("policy_id") \
    .agg(
        F.count("claim_id")             .alias("claims_per_policy"),
        F.sum("claim_amount_chf")       .alias("total_claimed_per_policy"),
        F.countDistinct("customer_id")  .alias("claimants_per_policy"),
    )

window_30d = Window \
    .partitionBy("customer_id") \
    .orderBy(F.col("incident_date").cast("long")) \
    .rangeBetween(-30 * 86400, 0)

network_scores = silver_claims \
    .join(policy_claim_concentration, on="policy_id", how="left") \
    .withColumn("claims_in_30d_window",
                F.count("claim_id").over(window_30d)) \
    .withColumn("network_flag_multi_claim_policy",
                F.when(F.col("claims_per_policy") > F.lit(2),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("network_flag_high_policy_exposure",
                F.when(F.col("total_claimed_per_policy") > F.lit(300_000.0),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("network_flag_clustered_claims",
                F.when(F.col("claims_in_30d_window") > F.lit(1),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("network_flag_multi_claimant",
                F.when(F.col("claimants_per_policy") > F.lit(1),
                       F.lit(1)).otherwise(F.lit(0))) \
    .withColumn("network_flags_count",
                F.col("network_flag_multi_claim_policy") +
                F.col("network_flag_high_policy_exposure") +
                F.col("network_flag_clustered_claims") +
                F.col("network_flag_multi_claimant")) \
    .withColumn("network_score",
                F.least(F.lit(1.0),
                        F.greatest(F.lit(0.0),
                                   F.col("network_flags_count") / F.lit(4.0)))) \
    .select(
        "claim_id", "policy_id", "customer_id",
        "claims_per_policy",
        F.round("total_claimed_per_policy", 2).alias("total_claimed_per_policy"),
        "claimants_per_policy",
        "claims_in_30d_window",
        "network_flag_multi_claim_policy",
        "network_flag_high_policy_exposure",
        "network_flag_clustered_claims",
        "network_flag_multi_claimant",
        "network_flags_count",
        F.round("network_score", 3).alias("network_score")
    )

network_scores = add_fraud_audit(network_scores)
write_fraud(network_scores, "fraud_network_scores")

print("\nNetwork signal distribution:")
spark.sql("""
    SELECT network_flags_count,
           COUNT(*)                        AS claim_count,
           ROUND(AVG(network_score), 3)    AS avg_network_score
    FROM insurance_fraud.fraud_network_scores
    GROUP BY network_flags_count
    ORDER BY network_flags_count DESC
""").show()


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 9 — Composite Fraud Score
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("RUNNING: COMPOSITE FRAUD SCORING")
print(f"{'='*50}")

rule_sdf  = spark.table(f"{FRAUD_DB}.fraud_rule_hits") \
                 .select("claim_id", "rule_score", "rules_fired_count")

stat_sdf  = spark.table(f"{FRAUD_DB}.fraud_statistical_scores") \
                 .select("claim_id", "statistical_score",
                         "z_score_amount", "is_statistical_anomaly")

                 
# Remove customer_id — it already exists in silver_claims
behav_sdf = spark.table(f"{FRAUD_DB}.fraud_behavioural_flags") \
                 .select("claim_id", "behavioural_score",
                         "lifetime_claim_count", "prior_fraud_flags",
                         "behavioural_flags_count")                 

net_sdf   = spark.table(f"{FRAUD_DB}.fraud_network_scores") \
                 .select("claim_id", "network_score",
                         "network_flags_count", "claims_in_30d_window")

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
    .fillna(0.0, subset=["rule_score", "statistical_score",
                          "behavioural_score", "network_score"]) \
    .withColumn("composite_score",
                F.round(
                    (F.col("rule_score")        * F.lit(WEIGHTS["rule_score"])) +
                    (F.col("statistical_score") * F.lit(WEIGHTS["statistical_score"])) +
                    (F.col("behavioural_score") * F.lit(WEIGHTS["behavioural_score"])) +
                    (F.col("network_score")     * F.lit(WEIGHTS["network_score"])),
                    4
                )) \
    .withColumn("fraud_risk_tier",
                F.when(F.col("composite_score") >= F.lit(0.75), F.lit("CRITICAL"))
                 .when(F.col("composite_score") >= F.lit(0.50), F.lit("HIGH"))
                 .when(F.col("composite_score") >= F.lit(0.25), F.lit("MEDIUM"))
                 .otherwise(F.lit("LOW"))) \
    .withColumn("recommend_investigation",
                F.when(
                    (F.col("composite_score") >= F.lit(0.50)) |
                    (F.col("is_fraud_suspected") == True),
                    F.lit(True)
                ).otherwise(F.lit(False))) \
    .withColumn("investigation_priority",
                F.when(F.col("fraud_risk_tier") == "CRITICAL", F.lit(1))
                 .when(F.col("fraud_risk_tier") == "HIGH",     F.lit(2))
                 .when(F.col("fraud_risk_tier") == "MEDIUM",   F.lit(3))
                 .otherwise(F.lit(4)))

composite = add_fraud_audit(composite)
write_fraud(composite, "fraud_composite_scores",
            partition_cols=["fraud_risk_tier"])

print("\nComposite fraud score distribution:")
spark.sql("""
    SELECT fraud_risk_tier,
           COUNT(*)                            AS claim_count,
           ROUND(AVG(composite_score), 4)      AS avg_composite_score,
           ROUND(AVG(claim_amount_chf), 2)     AS avg_claim_chf,
           ROUND(SUM(claim_amount_chf), 0)     AS total_exposure_chf,
           SUM(CASE WHEN is_fraud_suspected
                    THEN 1 ELSE 0 END)         AS confirmed_suspected
    FROM insurance_fraud.fraud_composite_scores
    GROUP BY fraud_risk_tier
    ORDER BY avg_composite_score DESC
""").show()


# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 10 — Investigation Queue
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("BUILDING: INVESTIGATION QUEUE")
print(f"{'='*50}")

investigation_queue = spark.table(f"{FRAUD_DB}.fraud_composite_scores") \
    .filter(F.col("recommend_investigation") == True) \
    .join(
        silver_customers.select(
            "customer_id", "full_name", "email",
            "customer_segment", "customer_tenure_years",
            "city", "country"
        ),
        on="customer_id",
        how="left"
    ) \
    .join(
        silver_policies.select(
            "policy_id", "policy_type", "status",
            "annual_premium_chf", "coverage_amount_chf",
            "risk_band"
        ),
        on="policy_id",
        how="left"
    ) \
    .select(
        "investigation_priority",
        "fraud_risk_tier",
        "claim_id",
        F.round("claim_amount_chf",   2).alias("claim_amount_chf"),
        F.round("composite_score",    4).alias("composite_score"),
        F.round("rule_score",         3).alias("rule_score"),
        F.round("statistical_score",  3).alias("statistical_score"),
        F.round("behavioural_score",  3).alias("behavioural_score"),
        F.round("network_score",      3).alias("network_score"),
        "rules_fired_count",
        "z_score_amount",
        "behavioural_flags_count",
        "network_flags_count",
        "claims_in_30d_window",
        "lifetime_claim_count",
        "prior_fraud_flags",
        "is_fraud_suspected",
        "fraud_indicators",
        "claim_type",
        "claim_severity",
        "incident_date",
        "submitted_date",
        "days_to_submit",
        "handler_id",
        "full_name",
        "email",
        "customer_segment",
        "customer_tenure_years",
        "city",
        "policy_type",
        F.round("annual_premium_chf",  2).alias("annual_premium_chf"),
        F.round("coverage_amount_chf", 2).alias("coverage_amount_chf"),
        "risk_band",
        "_fraud_batch_id",
        "_fraud_load_timestamp"
    ) \
    .orderBy("investigation_priority", F.col("composite_score").desc())

write_fraud(investigation_queue, "fraud_investigation_queue",
            partition_cols=["fraud_risk_tier"])

print("\nInvestigation queue summary:")
spark.sql("""
    SELECT fraud_risk_tier,
           COUNT(*)                            AS cases_in_queue,
           ROUND(AVG(composite_score), 4)      AS avg_score,
           ROUND(SUM(claim_amount_chf), 0)     AS total_exposure_chf,
           ROUND(AVG(claim_amount_chf), 0)     AS avg_claim_chf
    FROM insurance_fraud.fraud_investigation_queue
    GROUP BY fraud_risk_tier
    ORDER BY avg_score DESC
""").show()

print("\nTop 10 priority cases:")
spark.sql("""
    SELECT claim_id,
           full_name,
           fraud_risk_tier,
           ROUND(composite_score, 4)       AS composite_score,
           ROUND(claim_amount_chf, 0)      AS claim_chf,
           claim_type,
           rules_fired_count,
           prior_fraud_flags,
           claims_in_30d_window
    FROM insurance_fraud.fraud_investigation_queue
    ORDER BY investigation_priority,
             composite_score DESC
    LIMIT 10
""").show(truncate=False)


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 11 — Final Fraud Validation Report
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("FRAUD DETECTION — FINAL VALIDATION REPORT")
print(f"{'='*50}")

fraud_tables = [
    "fraud_rule_hits",
    "fraud_statistical_scores",
    "fraud_behavioural_flags",
    "fraud_network_scores",
    "fraud_composite_scores",
    "fraud_investigation_queue",
]

total = 0
for table in fraud_tables:
    count = spark.table(f"{FRAUD_DB}.{table}").count()
    total += count
    print(f"  {table:<35} {count:>8,} rows ✅")

print(f"\n  {'TOTAL':<35} {total:>8,} rows")

print(f"\n  KEY FRAUD KPIs")
print(f"  {'─'*45}")
spark.sql("""
    SELECT
        COUNT(*)                                        AS total_claims_scored,
        SUM(CASE WHEN fraud_risk_tier = 'CRITICAL'
                 THEN 1 ELSE 0 END)                    AS critical_tier,
        SUM(CASE WHEN fraud_risk_tier = 'HIGH'
                 THEN 1 ELSE 0 END)                    AS high_tier,
        SUM(CASE WHEN recommend_investigation = true
                 THEN 1 ELSE 0 END)                    AS flagged_for_investigation,
        ROUND(SUM(CASE WHEN recommend_investigation = true
                       THEN claim_amount_chf
                       ELSE 0 END), 0)                 AS total_exposure_chf,
        ROUND(AVG(composite_score), 4)                 AS avg_composite_score
    FROM insurance_fraud.fraud_composite_scores
""").show()

print(f"  Batch ID: {BATCH_ID}")
print(f"{'='*50}")
print(f"\n✅ Fraud detection complete — Batch ID: {BATCH_ID}")