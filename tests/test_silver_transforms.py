"""
tests/test_silver_transforms.py
================================
Unit tests for Silver layer transformation functions.
Tests every derived column calculation independently.

Run:
    pytest tests/test_silver_transforms.py -v
"""

import sys
import os
import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("silver_transform_tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


# ─────────────────────────────────────────────
# TESTS — CLAIM SEVERITY BANDS
# ─────────────────────────────────────────────

class TestClaimSeverityBands:

    def _apply_severity(self, spark, amount):
        schema = T.StructType([
            T.StructField("claim_id",         T.StringType(), True),
            T.StructField("claim_amount_chf", T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("CLM-001", float(amount))], schema
        )
        return sdf.withColumn(
            "claim_severity",
            F.when(F.col("claim_amount_chf") < 5000,   F.lit("LOW"))
             .when(F.col("claim_amount_chf") < 25000,  F.lit("MEDIUM"))
             .when(F.col("claim_amount_chf") < 75000,  F.lit("HIGH"))
             .otherwise(F.lit("CATASTROPHIC"))
        ).collect()[0]["claim_severity"]

    def test_low_severity(self, spark):
        assert self._apply_severity(spark, 1000.0) == "LOW"

    def test_medium_severity(self, spark):
        assert self._apply_severity(spark, 10000.0) == "MEDIUM"

    def test_high_severity(self, spark):
        assert self._apply_severity(spark, 50000.0) == "HIGH"

    def test_catastrophic_severity(self, spark):
        assert self._apply_severity(spark, 100000.0) == "CATASTROPHIC"

    def test_boundary_low_to_medium(self, spark):
        assert self._apply_severity(spark, 4999.99) == "LOW"
        assert self._apply_severity(spark, 5000.0) == "MEDIUM"

    def test_boundary_medium_to_high(self, spark):
        assert self._apply_severity(spark, 24999.99) == "MEDIUM"
        assert self._apply_severity(spark, 25000.0) == "HIGH"

    def test_boundary_high_to_catastrophic(self, spark):
        assert self._apply_severity(spark, 74999.99) == "HIGH"
        assert self._apply_severity(spark, 75000.0) == "CATASTROPHIC"


# ─────────────────────────────────────────────
# TESTS — RISK BANDS
# ─────────────────────────────────────────────

class TestRiskBands:

    def _apply_risk_band(self, spark, score):
        schema = T.StructType([
            T.StructField("policy_id",   T.StringType(), True),
            T.StructField("risk_score",  T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame([("POL-001", float(score))], schema)
        return sdf.withColumn(
            "risk_band",
            F.when(F.col("risk_score") < 0.3,  F.lit("LOW"))
             .when(F.col("risk_score") < 0.6,  F.lit("MEDIUM"))
             .when(F.col("risk_score") < 0.8,  F.lit("HIGH"))
             .otherwise(F.lit("VERY_HIGH"))
        ).collect()[0]["risk_band"]

    def test_low_risk(self, spark):
        assert self._apply_risk_band(spark, 0.1) == "LOW"

    def test_medium_risk(self, spark):
        assert self._apply_risk_band(spark, 0.4) == "MEDIUM"

    def test_high_risk(self, spark):
        assert self._apply_risk_band(spark, 0.7) == "HIGH"

    def test_very_high_risk(self, spark):
        assert self._apply_risk_band(spark, 0.9) == "VERY_HIGH"

    def test_boundary_low_to_medium(self, spark):
        assert self._apply_risk_band(spark, 0.29) == "LOW"
        assert self._apply_risk_band(spark, 0.30) == "MEDIUM"

    def test_boundary_high_to_very_high(self, spark):
        assert self._apply_risk_band(spark, 0.79) == "HIGH"
        assert self._apply_risk_band(spark, 0.80) == "VERY_HIGH"


# ─────────────────────────────────────────────
# TESTS — SETTLEMENT RATIO
# ─────────────────────────────────────────────

class TestSettlementRatio:

    def _apply_ratio(self, spark, claimed, settled):
        schema = T.StructType([
            T.StructField("claim_id",           T.StringType(), True),
            T.StructField("claim_amount_chf",   T.DoubleType(), True),
            T.StructField("settled_amount_chf", T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("CLM-001", float(claimed),
              float(settled) if settled is not None else None)],
            schema
        )
        result = sdf.withColumn(
            "settlement_ratio",
            F.when(
                (F.col("settled_amount_chf").isNotNull()) &
                (F.col("claim_amount_chf") > 0),
                F.round(
                    F.col("settled_amount_chf") /
                    F.col("claim_amount_chf"), 3
                )
            ).otherwise(F.lit(None).cast("double"))
        ).collect()[0]["settlement_ratio"]
        return result

    def test_full_settlement(self, spark):
        ratio = self._apply_ratio(spark, 10000.0, 10000.0)
        assert ratio == 1.0

    def test_partial_settlement(self, spark):
        ratio = self._apply_ratio(spark, 10000.0, 5000.0)
        assert ratio == 0.5

    def test_no_settlement_returns_null(self, spark):
        ratio = self._apply_ratio(spark, 10000.0, None)
        assert ratio is None

    def test_over_settlement(self, spark):
        ratio = self._apply_ratio(spark, 10000.0, 12000.0)
        assert ratio == 1.2

    def test_settlement_ratio_precision(self, spark):
        ratio = self._apply_ratio(spark, 3000.0, 1000.0)
        assert ratio == round(1000.0 / 3000.0, 3)


# ─────────────────────────────────────────────
# TESTS — SUBMISSION DELAY BANDS
# ─────────────────────────────────────────────

class TestSubmissionDelayBands:

    def _apply_delay_band(self, spark, days):
        schema = T.StructType([
            T.StructField("claim_id",       T.StringType(),  True),
            T.StructField("days_to_submit", T.IntegerType(), True),
        ])
        sdf = spark.createDataFrame([("CLM-001", days)], schema)
        return sdf.withColumn(
            "submission_delay_band",
            F.when(F.col("days_to_submit") <= 7,   F.lit("IMMEDIATE"))
             .when(F.col("days_to_submit") <= 30,  F.lit("NORMAL"))
             .when(F.col("days_to_submit") <= 60,  F.lit("DELAYED"))
             .otherwise(F.lit("VERY_DELAYED"))
        ).collect()[0]["submission_delay_band"]

    def test_immediate_submission(self, spark):
        assert self._apply_delay_band(spark, 3) == "IMMEDIATE"

    def test_normal_submission(self, spark):
        assert self._apply_delay_band(spark, 15) == "NORMAL"

    def test_delayed_submission(self, spark):
        assert self._apply_delay_band(spark, 45) == "DELAYED"

    def test_very_delayed_submission(self, spark):
        assert self._apply_delay_band(spark, 90) == "VERY_DELAYED"

    def test_boundary_immediate_to_normal(self, spark):
        assert self._apply_delay_band(spark, 7)  == "IMMEDIATE"
        assert self._apply_delay_band(spark, 8)  == "NORMAL"

    def test_boundary_normal_to_delayed(self, spark):
        assert self._apply_delay_band(spark, 30) == "NORMAL"
        assert self._apply_delay_band(spark, 31) == "DELAYED"


# ─────────────────────────────────────────────
# TESTS — OVERDUE BANDS
# ─────────────────────────────────────────────

class TestOverdueBands:

    def _apply_overdue_band(self, spark, days_overdue):
        schema = T.StructType([
            T.StructField("payment_id",   T.StringType(),  True),
            T.StructField("days_overdue", T.IntegerType(), True),
        ])
        sdf = spark.createDataFrame([("PAY-001", days_overdue)], schema)
        return sdf.withColumn(
            "overdue_band",
            F.when(F.col("days_overdue") == 0,       F.lit("ON_TIME"))
             .when(F.col("days_overdue") <= 30,      F.lit("1_30_DAYS"))
             .when(F.col("days_overdue") <= 60,      F.lit("31_60_DAYS"))
             .when(F.col("days_overdue") <= 90,      F.lit("61_90_DAYS"))
             .otherwise(F.lit("90_PLUS_DAYS"))
        ).collect()[0]["overdue_band"]

    def test_on_time_payment(self, spark):
        assert self._apply_overdue_band(spark, 0) == "ON_TIME"

    def test_early_overdue(self, spark):
        assert self._apply_overdue_band(spark, 15) == "1_30_DAYS"

    def test_mid_overdue(self, spark):
        assert self._apply_overdue_band(spark, 45) == "31_60_DAYS"

    def test_late_overdue(self, spark):
        assert self._apply_overdue_band(spark, 75) == "61_90_DAYS"

    def test_very_late_overdue(self, spark):
        assert self._apply_overdue_band(spark, 120) == "90_PLUS_DAYS"


# ─────────────────────────────────────────────
# TESTS — CUSTOMER SEGMENT
# ─────────────────────────────────────────────

class TestCustomerSegment:

    def _apply_segment(self, spark, is_high_value, tenure_years):
        schema = T.StructType([
            T.StructField("customer_id",           T.StringType(), True),
            T.StructField("is_high_value",         T.BooleanType(), True),
            T.StructField("customer_tenure_years", T.DoubleType(), True),
        ])
        sdf = spark.createDataFrame(
            [("CUST-001", is_high_value, float(tenure_years))],
            schema
        )
        return sdf.withColumn(
            "customer_segment",
            F.when(F.col("is_high_value") F.lit(True),     F.lit("PREMIUM"))
             .when(F.col("customer_tenure_years") >= 5,    F.lit("LOYAL"))
             .otherwise(F.lit("STANDARD"))
        ).collect()[0]["customer_segment"]

    def test_high_value_is_premium(self, spark):
        assert self._apply_segment(spark, True, 1.0) == "PREMIUM"

    def test_long_tenure_is_loyal(self, spark):
        assert self._apply_segment(spark, False, 6.0) == "LOYAL"

    def test_short_tenure_is_standard(self, spark):
        assert self._apply_segment(spark, False, 2.0) == "STANDARD"

    def test_high_value_overrides_tenure(self, spark):
        assert self._apply_segment(spark, True, 10.0) == "PREMIUM"

    def test_boundary_tenure_loyal(self, spark):
        assert self._apply_segment(spark, False, 4.9) == "STANDARD"
        assert self._apply_segment(spark, False, 5.0) == "LOYAL"


# ─────────────────────────────────────────────
# TESTS — DEDUPLICATION
# ─────────────────────────────────────────────

class TestDeduplication:

    def test_dedup_keeps_latest_record(self, spark):
        from pyspark.sql import Window
        schema = T.StructType([
            T.StructField("claim_id",             T.StringType(), True),
            T.StructField("claim_status",         T.StringType(), True),
            T.StructField("_ingestion_timestamp", T.StringType(), True),
        ])
        data = [
            ("CLM-001", "submitted",    "2024-01-01T10:00:00"),
            ("CLM-001", "under_review", "2024-01-02T10:00:00"),
            ("CLM-001", "approved",     "2024-01-03T10:00:00"),
            ("CLM-002", "submitted",    "2024-01-01T10:00:00"),
        ]
        sdf = spark.createDataFrame(data, schema)
        window = Window.partitionBy("claim_id") \
                       .orderBy(F.col("_ingestion_timestamp").desc())
        deduped = sdf.withColumn("_row_num", F.row_number().over(window)) \
                     .filter(F.col("_row_num") == 1) \
                     .drop("_row_num")

        assert deduped.count() == 2
        clm001 = deduped.filter(
            F.col("claim_id") == "CLM-001"
        ).collect()[0]["claim_status"]
        assert clm001 == "approved"

    def test_dedup_no_duplicates_unchanged(self, spark):
        from pyspark.sql import Window
        schema = T.StructType([
            T.StructField("claim_id",             T.StringType(), True),
            T.StructField("claim_status",         T.StringType(), True),
            T.StructField("_ingestion_timestamp", T.StringType(), True),
        ])
        data = [
            ("CLM-001", "submitted", "2024-01-01T10:00:00"),
            ("CLM-002", "approved",  "2024-01-02T10:00:00"),
        ]
        sdf = spark.createDataFrame(data, schema)
        window = Window.partitionBy("claim_id") \
                       .orderBy(F.col("_ingestion_timestamp").desc())
        deduped = sdf.withColumn("_row_num", F.row_number().over(window)) \
                     .filter(F.col("_row_num") == 1) \
                     .drop("_row_num")
        assert deduped.count() == 2
