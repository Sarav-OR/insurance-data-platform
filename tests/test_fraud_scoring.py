"""
tests/test_fraud_scoring.py
============================
Unit tests for fraud scoring logic.
Tests composite score calculation, tier assignment,
weight validation and edge cases.

Run:
    pytest tests/test_fraud_scoring.py -v
"""

import sys
import os
import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import FRAUD_WEIGHTS, FRAUD_THRESHOLDS


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("fraud_scoring_tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def calc_composite(rule, stat, behav, network):
    """Calculate expected composite score using config weights."""
    return round(
        rule    * FRAUD_WEIGHTS["rule_score"] +
        stat    * FRAUD_WEIGHTS["statistical_score"] +
        behav   * FRAUD_WEIGHTS["behavioural_score"] +
        network * FRAUD_WEIGHTS["network_score"],
        4
    )


def apply_fraud_tier(score):
    """Apply fraud tier logic matching Silver/Fraud layer."""
    if score >= FRAUD_THRESHOLDS["critical"]:
        return "CRITICAL"
    elif score >= FRAUD_THRESHOLDS["high"]:
        return "HIGH"
    elif score >= FRAUD_THRESHOLDS["medium"]:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────
# TESTS — WEIGHT CONFIGURATION
# ─────────────────────────────────────────────

class TestFraudWeights:

    def test_weights_sum_to_one(self):
        total = sum(FRAUD_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_weights_positive(self):
        for key, val in FRAUD_WEIGHTS.items():
            assert val > 0, f"Weight {key} not positive"

    def test_all_weight_keys_present(self):
        required = [
            "rule_score", "statistical_score",
            "behavioural_score", "network_score"
        ]
        for key in required:
            assert key in FRAUD_WEIGHTS

    def test_no_single_weight_dominates(self):
        for key, val in FRAUD_WEIGHTS.items():
            assert val < 0.6, \
                f"Weight {key}={val} dominates — review scoring balance"


# ─────────────────────────────────────────────
# TESTS — COMPOSITE SCORE CALCULATION
# ─────────────────────────────────────────────

class TestCompositeScore:

    def test_zero_scores_produce_zero_composite(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",          T.StringType(), True),
            T.StructField("rule_score",        T.DoubleType(), True),
            T.StructField("statistical_score", T.DoubleType(), True),
            T.StructField("behavioural_score", T.DoubleType(), True),
            T.StructField("network_score",     T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("CLM-001", 0.0, 0.0, 0.0, 0.0)], schema
        )
        result = sdf.withColumn(
            "composite_score",
            F.round(
                (F.col("rule_score")        * F.lit(FRAUD_WEIGHTS["rule_score"])) +
                (F.col("statistical_score") * F.lit(FRAUD_WEIGHTS["statistical_score"])) +
                (F.col("behavioural_score") * F.lit(FRAUD_WEIGHTS["behavioural_score"])) +
                (F.col("network_score")     * F.lit(FRAUD_WEIGHTS["network_score"])),
                4
            )
        ).collect()[0]["composite_score"]
        assert result == 0.0

    def test_max_scores_produce_one_composite(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",          T.StringType(), True),
            T.StructField("rule_score",        T.DoubleType(), True),
            T.StructField("statistical_score", T.DoubleType(), True),
            T.StructField("behavioural_score", T.DoubleType(), True),
            T.StructField("network_score",     T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("CLM-001", 1.0, 1.0, 1.0, 1.0)], schema
        )
        result = sdf.withColumn(
            "composite_score",
            F.round(
                (F.col("rule_score")        * F.lit(FRAUD_WEIGHTS["rule_score"])) +
                (F.col("statistical_score") * F.lit(FRAUD_WEIGHTS["statistical_score"])) +
                (F.col("behavioural_score") * F.lit(FRAUD_WEIGHTS["behavioural_score"])) +
                (F.col("network_score")     * F.lit(FRAUD_WEIGHTS["network_score"])),
                4
            )
        ).collect()[0]["composite_score"]
        assert abs(result - 1.0) < 1e-3

    def test_composite_score_within_bounds(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",          T.StringType(), True),
            T.StructField("rule_score",        T.DoubleType(), True),
            T.StructField("statistical_score", T.DoubleType(), True),
            T.StructField("behavioural_score", T.DoubleType(), True),
            T.StructField("network_score",     T.DoubleType(), True),
        ])
        data = [
            ("CLM-001", 0.8, 0.6, 0.4, 0.2),
            ("CLM-002", 0.1, 0.2, 0.3, 0.1),
            ("CLM-003", 0.5, 0.5, 0.5, 0.5),
            ("CLM-004", 0.0, 0.0, 0.0, 0.0),
            ("CLM-005", 1.0, 1.0, 1.0, 1.0),
        ]
        sdf = spark.createDataFrame(data, schema)
        results = sdf.withColumn(
            "composite_score",
            F.round(
                (F.col("rule_score")        * F.lit(FRAUD_WEIGHTS["rule_score"])) +
                (F.col("statistical_score") * F.lit(FRAUD_WEIGHTS["statistical_score"])) +
                (F.col("behavioural_score") * F.lit(FRAUD_WEIGHTS["behavioural_score"])) +
                (F.col("network_score")     * F.lit(FRAUD_WEIGHTS["network_score"])),
                4
            )
        ).collect()
        for row in results:
            score = row["composite_score"]
            assert 0.0 <= score <= 1.0, \
                f"Score {score} out of bounds for {row['claim_id']}"

    def test_weighted_composite_correct(self):
        score = calc_composite(0.8, 0.6, 0.4, 0.2)
        expected = (
            0.8 * FRAUD_WEIGHTS["rule_score"] +
            0.6 * FRAUD_WEIGHTS["statistical_score"] +
            0.4 * FRAUD_WEIGHTS["behavioural_score"] +
            0.2 * FRAUD_WEIGHTS["network_score"]
        )
        assert abs(score - round(expected, 4)) < 1e-4


# ─────────────────────────────────────────────
# TESTS — FRAUD TIER ASSIGNMENT
# ─────────────────────────────────────────────

class TestFraudTierAssignment:

    def test_critical_tier_assigned_above_threshold(self):
        assert apply_fraud_tier(0.75) == "CRITICAL"
        assert apply_fraud_tier(0.90) == "CRITICAL"
        assert apply_fraud_tier(1.00) == "CRITICAL"

    def test_high_tier_assigned_correctly(self):
        assert apply_fraud_tier(0.50) == "HIGH"
        assert apply_fraud_tier(0.65) == "HIGH"
        assert apply_fraud_tier(0.74) == "HIGH"

    def test_medium_tier_assigned_correctly(self):
        assert apply_fraud_tier(0.25) == "MEDIUM"
        assert apply_fraud_tier(0.40) == "MEDIUM"
        assert apply_fraud_tier(0.49) == "MEDIUM"

    def test_low_tier_assigned_below_threshold(self):
        assert apply_fraud_tier(0.00) == "LOW"
        assert apply_fraud_tier(0.10) == "LOW"
        assert apply_fraud_tier(0.24) == "LOW"

    def test_critical_boundary(self):
        assert apply_fraud_tier(0.7499) == "HIGH"
        assert apply_fraud_tier(0.75)   == "CRITICAL"

    def test_high_boundary(self):
        assert apply_fraud_tier(0.4999) == "MEDIUM"
        assert apply_fraud_tier(0.50)   == "HIGH"

    def test_medium_boundary(self):
        assert apply_fraud_tier(0.2499) == "LOW"
        assert apply_fraud_tier(0.25)   == "MEDIUM"


# ─────────────────────────────────────────────
# TESTS — INVESTIGATION PRIORITY
# ─────────────────────────────────────────────

class TestInvestigationPriority:

    def _apply_priority(self, spark, tier):
        schema = T.StructType([
            T.StructField("claim_id",        T.StringType(), True),
            T.StructField("fraud_risk_tier", T.StringType(), True),
        ])
        sdf = spark.createDataFrame([("CLM-001", tier)], schema)
        return sdf.withColumn(
            "investigation_priority",
            F.when(F.col("fraud_risk_tier") == "CRITICAL", F.lit(1))
             .when(F.col("fraud_risk_tier") == "HIGH",     F.lit(2))
             .when(F.col("fraud_risk_tier") == "MEDIUM",   F.lit(3))
             .otherwise(F.lit(4))
        ).collect()[0]["investigation_priority"]

    def test_critical_is_priority_one(self, spark):
        assert self._apply_priority(spark, "CRITICAL") == 1

    def test_high_is_priority_two(self, spark):
        assert self._apply_priority(spark, "HIGH") == 2

    def test_medium_is_priority_three(self, spark):
        assert self._apply_priority(spark, "MEDIUM") == 3

    def test_low_is_priority_four(self, spark):
        assert self._apply_priority(spark, "LOW") == 4

    def test_priority_ordering_correct(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",        T.StringType(), True),
            T.StructField("fraud_risk_tier", T.StringType(), True),
        ])
        data = [
            ("CLM-001", "LOW"),
            ("CLM-002", "CRITICAL"),
            ("CLM-003", "MEDIUM"),
            ("CLM-004", "HIGH"),
        ]
        sdf = spark.createDataFrame(data, schema)
        results = sdf.withColumn(
            "investigation_priority",
            F.when(F.col("fraud_risk_tier") == "CRITICAL", F.lit(1))
             .when(F.col("fraud_risk_tier") == "HIGH",     F.lit(2))
             .when(F.col("fraud_risk_tier") == "MEDIUM",   F.lit(3))
             .otherwise(F.lit(4))
        ).orderBy("investigation_priority").collect()

        assert results[0]["fraud_risk_tier"] == "CRITICAL"
        assert results[1]["fraud_risk_tier"] == "HIGH"
        assert results[2]["fraud_risk_tier"] == "MEDIUM"
        assert results[3]["fraud_risk_tier"] == "LOW"


# ─────────────────────────────────────────────
# TESTS — SIGNAL SCORE BANDS
# ─────────────────────────────────────────────

class TestSignalScoreBands:

    def _apply_score_band(self, spark, score):
        schema = T.StructType([
            T.StructField("signal_id",    T.StringType(), True),
            T.StructField("signal_score", T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame([("SIG-001", float(score))], schema)
        return sdf.withColumn(
            "score_band",
            F.when(F.col("signal_score") >= 0.8, F.lit("CRITICAL"))
             .when(F.col("signal_score") >= 0.6, F.lit("HIGH"))
             .when(F.col("signal_score") >= 0.4, F.lit("MEDIUM"))
             .otherwise(F.lit("LOW"))
        ).collect()[0]["score_band"]

    def test_critical_signal(self, spark):
        assert self._apply_score_band(spark, 0.9) == "CRITICAL"

    def test_high_signal(self, spark):
        assert self._apply_score_band(spark, 0.7) == "HIGH"

    def test_medium_signal(self, spark):
        assert self._apply_score_band(spark, 0.5) == "MEDIUM"

    def test_low_signal(self, spark):
        assert self._apply_score_band(spark, 0.2) == "LOW"

    def test_boundary_critical(self, spark):
        assert self._apply_score_band(spark, 0.79) == "HIGH"
        assert self._apply_score_band(spark, 0.80) == "CRITICAL"
