"""
tests/test_gold_aggregations.py
================================
Unit tests for Gold layer KPI calculations.
Tests every formula independently with known inputs and expected outputs.

Run:
    pytest tests/test_gold_aggregations.py -v
"""

import sys
import os
import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("gold_aggregation_tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


# ─────────────────────────────────────────────
# TESTS — LOSS RATIO
# ─────────────────────────────────────────────

class TestLossRatio:
    """
    Loss ratio = total claims paid / total premium income.
    Industry benchmark: < 0.6 is healthy for most lines.
    """

    def _calc_loss_ratio(self, spark, claims_total, premium_total):
        schema = T.StructType([
            T.StructField("policy_type",       T.StringType(), True),
            T.StructField("total_claimed_chf", T.DoubleType(), True),
            T.StructField("total_premium_chf", T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("motor", float(claims_total), float(premium_total))],
            schema
        )
        result = sdf.withColumn(
            "loss_ratio",
            F.round(
                F.col("total_claimed_chf") /
                F.when(F.col("total_premium_chf") > 0,
                       F.col("total_premium_chf"))
                 .otherwise(F.lit(None)),
                4
            )
        ).collect()[0]["loss_ratio"]
        return result

    def test_healthy_loss_ratio(self, spark):
        ratio = self._calc_loss_ratio(spark, 400000.0, 1000000.0)
        assert ratio == 0.4

    def test_break_even_loss_ratio(self, spark):
        ratio = self._calc_loss_ratio(spark, 1000000.0, 1000000.0)
        assert ratio == 1.0

    def test_unprofitable_loss_ratio(self, spark):
        ratio = self._calc_loss_ratio(spark, 1500000.0, 1000000.0)
        assert ratio == 1.5

    def test_zero_premium_returns_null(self, spark):
        ratio = self._calc_loss_ratio(spark, 500000.0, 0.0)
        assert ratio is None

    def test_loss_ratio_precision(self, spark):
        ratio = self._calc_loss_ratio(spark, 333333.0, 1000000.0)
        assert ratio == round(333333.0 / 1000000.0, 4)


# ─────────────────────────────────────────────
# TESTS — COLLECTION RATE
# ─────────────────────────────────────────────

class TestCollectionRate:
    """
    Collection rate = total collected / total due * 100.
    Target: > 95% in healthy insurance operations.
    """

    def _calc_collection_rate(self, spark, collected, due):
        schema = T.StructType([
            T.StructField("policy_type",          T.StringType(), True),
            T.StructField("total_collected_chf",  T.DoubleType(), True),
            T.StructField("total_due_chf",        T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("home", float(collected), float(due))],
            schema
        )
        result = sdf.withColumn(
            "collection_rate_pct",
            F.round(
                F.col("total_collected_chf") /
                F.when(F.col("total_due_chf") > 0,
                       F.col("total_due_chf"))
                 .otherwise(F.lit(None)) * 100,
                2
            )
        ).collect()[0]["collection_rate_pct"]
        return result

    def test_full_collection(self, spark):
        rate = self._calc_collection_rate(spark, 100000.0, 100000.0)
        assert rate == 100.0

    def test_partial_collection(self, spark):
        rate = self._calc_collection_rate(spark, 88000.0, 100000.0)
        assert rate == 88.0

    def test_zero_collection(self, spark):
        rate = self._calc_collection_rate(spark, 0.0, 100000.0)
        assert rate == 0.0

    def test_zero_due_returns_null(self, spark):
        rate = self._calc_collection_rate(spark, 0.0, 0.0)
        assert rate is None


# ─────────────────────────────────────────────
# TESTS — FRAUD RATE
# ─────────────────────────────────────────────

class TestFraudRate:
    """
    Fraud rate = fraud suspected count / total claims * 100.
    """

    def _calc_fraud_rate(self, spark, fraud_count, total_claims):
        schema = T.StructType([
            T.StructField("policy_type",           T.StringType(),  True),
            T.StructField("fraud_suspected_count", T.IntegerType(), True),
            T.StructField("total_claims",          T.IntegerType(), True),
        ])
        sdf = spark.createDataFrame(
            [("life", fraud_count, total_claims)],
            schema
        )
        result = sdf.withColumn(
            "fraud_rate_pct",
            F.round(
                F.col("fraud_suspected_count") /
                F.when(F.col("total_claims") > 0,
                       F.col("total_claims"))
                 .otherwise(F.lit(None)) * 100,
                2
            )
        ).collect()[0]["fraud_rate_pct"]
        return result

    def test_typical_fraud_rate(self, spark):
        rate = self._calc_fraud_rate(spark, 5, 100)
        assert rate == 5.0

    def test_zero_fraud(self, spark):
        rate = self._calc_fraud_rate(spark, 0, 100)
        assert rate == 0.0

    def test_all_fraud(self, spark):
        rate = self._calc_fraud_rate(spark, 100, 100)
        assert rate == 100.0

    def test_zero_claims_returns_null(self, spark):
        rate = self._calc_fraud_rate(spark, 0, 0)
        assert rate is None


# ─────────────────────────────────────────────
# TESTS — CLV INDICATOR
# ─────────────────────────────────────────────

class TestCLVIndicator:
    """
    CLV indicator = segment premium income - total claimed.
    Positive = profitable segment. Negative = loss-making.
    """

    def _calc_clv(self, spark, premium, claimed):
        schema = T.StructType([
            T.StructField("customer_segment",          T.StringType(), True),
            T.StructField("segment_premium_income_chf", T.DoubleType(), True),
            T.StructField("total_claimed_chf",         T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("PREMIUM", float(premium), float(claimed))],
            schema
        )
        return sdf.withColumn(
            "clv_indicator",
            F.round(
                F.col("segment_premium_income_chf") -
                F.col("total_claimed_chf"),
                2
            )
        ).collect()[0]["clv_indicator"]

    def test_profitable_segment(self, spark):
        clv = self._calc_clv(spark, 1000000.0, 600000.0)
        assert clv == 400000.0

    def test_loss_making_segment(self, spark):
        clv = self._calc_clv(spark, 500000.0, 750000.0)
        assert clv == -250000.0

    def test_breakeven_segment(self, spark):
        clv = self._calc_clv(spark, 1000000.0, 1000000.0)
        assert clv == 0.0


# ─────────────────────────────────────────────
# TESTS — MONTHLY AGGREGATION
# ─────────────────────────────────────────────

class TestMonthlyAggregation:

    def test_monthly_claim_count(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",       T.StringType(), True),
            T.StructField("incident_date",  T.StringType(), True),
        ])
        data = [
            ("CLM-001", "2024-01-15"),
            ("CLM-002", "2024-01-20"),
            ("CLM-003", "2024-02-10"),
            ("CLM-004", "2024-02-28"),
            ("CLM-005", "2024-02-05"),
        ]
        sdf = spark.createDataFrame(data, schema)
        result = sdf \
            .withColumn("month", F.date_format(
                F.to_date(F.col("incident_date"), "yyyy-MM-dd"),
                "yyyy-MM"
            )) \
            .groupBy("month") \
            .agg(F.count("claim_id").alias("claim_count")) \
            .orderBy("month") \
            .collect()

        assert len(result) == 2
        assert result[0]["month"] == "2024-01"
        assert result[0]["claim_count"] == 2
        assert result[1]["month"] == "2024-02"
        assert result[1]["claim_count"] == 3

    def test_monthly_premium_sum(self, spark):
        schema = T.StructType([
            T.StructField("payment_id",     T.StringType(), True),
            T.StructField("due_date",       T.StringType(), True),
            T.StructField("amount_due_chf", T.DoubleType(), True),
        ])
        data = [
            ("PAY-001", "2024-01-01", 1000.0),
            ("PAY-002", "2024-01-15", 2000.0),
            ("PAY-003", "2024-02-01", 1500.0),
        ]
        sdf = spark.createDataFrame(data, schema)
        result = sdf \
            .withColumn("month", F.date_format(
                F.to_date(F.col("due_date"), "yyyy-MM-dd"),
                "yyyy-MM"
            )) \
            .groupBy("month") \
            .agg(F.sum("amount_due_chf").alias("total_due")) \
            .orderBy("month") \
            .collect()

        assert result[0]["total_due"] == 3000.0
        assert result[1]["total_due"] == 1500.0
