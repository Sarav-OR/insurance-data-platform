"""
tests/test_great_expectations.py
==================================
Great Expectations style validation suite for Insurance Data Platform.
Implements schema, volume, completeness and range validations.

Note: Uses pure PySpark assertions to avoid GE dependency in CI.
Structure mirrors Great Expectations expectation patterns exactly —
direct GE integration is straightforward from this foundation.

Run:
    pytest tests/test_great_expectations.py -v
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
        .appName("great_expectations_tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


@pytest.fixture(scope="session")
def sample_claims(spark):
    schema = T.StructType([
        T.StructField("claim_id",           T.StringType(),  False),
        T.StructField("policy_id",          T.StringType(),  False),
        T.StructField("customer_id",        T.StringType(),  False),
        T.StructField("claim_amount_chf",   T.DoubleType(),  True),
        T.StructField("claim_status",       T.StringType(),  True),
        T.StructField("claim_type",         T.StringType(),  True),
        T.StructField("incident_date",      T.StringType(),  True),
        T.StructField("days_to_submit",     T.IntegerType(), True),
        T.StructField("is_fraud_suspected", T.BooleanType(), True),
    ])
    data = [
        ("CLM-001", "POL-001", "CUST-001", 5000.0,   "settled",      "accident", "2023-01-15", 7,  False),
        ("CLM-002", "POL-002", "CUST-002", 12000.0,  "approved",     "theft",    "2023-02-20", 14, False),
        ("CLM-003", "POL-003", "CUST-003", 75000.0,  "under_review", "fire",     "2023-03-10", 30, True),
        ("CLM-004", "POL-004", "CUST-004", 3000.0,   "rejected",     "flood",    "2023-04-05", 5,  False),
        ("CLM-005", "POL-005", "CUST-005", 150000.0, "submitted",    "medical",  "2023-05-01", 45, True),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture(scope="session")
def sample_policies(spark):
    schema = T.StructType([
        T.StructField("policy_id",           T.StringType(), False),
        T.StructField("customer_id",         T.StringType(), False),
        T.StructField("policy_type",         T.StringType(), True),
        T.StructField("annual_premium_chf",  T.DoubleType(), True),
        T.StructField("coverage_amount_chf", T.DoubleType(), True),
        T.StructField("risk_score",          T.DoubleType(), True),
        T.StructField("start_date",          T.StringType(), True),
        T.StructField("end_date",            T.StringType(), True),
    ])
    data = [
        ("POL-001", "CUST-001", "motor",      1200.0,  120000.0, 0.3, "2022-01-01", "2023-01-01"),
        ("POL-002", "CUST-002", "home",       800.0,   200000.0, 0.2, "2021-06-01", "2024-06-01"),
        ("POL-003", "CUST-003", "life",       3000.0,  500000.0, 0.7, "2020-01-01", "2025-01-01"),
        ("POL-004", "CUST-004", "health",     2000.0,  100000.0, 0.5, "2023-01-01", "2024-01-01"),
        ("POL-005", "CUST-005", "commercial", 4500.0,  750000.0, 0.8, "2022-06-01", "2025-06-01"),
    ]
    return spark.createDataFrame(data, schema)


# ─────────────────────────────────────────────
# EXPECTATION HELPERS
# Mirror Great Expectations naming conventions
# ─────────────────────────────────────────────

def expect_column_values_to_not_be_null(sdf, column):
    null_count = sdf.filter(F.col(column).isNull()).count()
    return null_count == 0, null_count


def expect_column_values_to_be_between(sdf, column, min_val, max_val):
    out_of_range = sdf.filter(
        (F.col(column) < min_val) | (F.col(column) > max_val)
    ).count()
    return out_of_range == 0, out_of_range


def expect_column_values_to_be_in_set(sdf, column, valid_set):
    invalid = sdf.filter(~F.col(column).isin(list(valid_set))).count()
    return invalid == 0, invalid


def expect_column_to_exist(sdf, column):
    return column in sdf.columns


def expect_table_row_count_to_be_between(sdf, min_rows, max_rows):
    count = sdf.count()
    return min_rows <= count <= max_rows, count


def expect_column_values_to_be_unique(sdf, column):
    total = sdf.count()
    distinct = sdf.select(column).distinct().count()
    return total == distinct, total - distinct


def expect_column_values_to_match_regex(sdf, column, pattern):
    invalid = sdf.filter(
        ~F.col(column).rlike(pattern)
    ).count()
    return invalid == 0, invalid


# ─────────────────────────────────────────────
# TESTS — SCHEMA EXPECTATIONS
# ─────────────────────────────────────────────

class TestSchemaExpectations:

    def test_claims_required_columns_exist(self, sample_claims):
        required = [
            "claim_id", "policy_id", "customer_id",
            "claim_amount_chf", "claim_status",
            "incident_date", "is_fraud_suspected"
        ]
        for col in required:
            assert expect_column_to_exist(sample_claims, col), \
                f"Required column missing: {col}"

    def test_policies_required_columns_exist(self, sample_policies):
        required = [
            "policy_id", "customer_id", "policy_type",
            "annual_premium_chf", "coverage_amount_chf",
            "risk_score", "start_date", "end_date"
        ]
        for col in required:
            assert expect_column_to_exist(sample_policies, col), \
                f"Required column missing: {col}"


# ─────────────────────────────────────────────
# TESTS — COMPLETENESS EXPECTATIONS
# ─────────────────────────────────────────────

class TestCompletenessExpectations:

    def test_claim_id_never_null(self, sample_claims):
        passed, null_count = expect_column_values_to_not_be_null(
            sample_claims, "claim_id"
        )
        assert passed, f"Found {null_count} null claim_ids"

    def test_policy_id_never_null(self, sample_claims):
        passed, null_count = expect_column_values_to_not_be_null(
            sample_claims, "policy_id"
        )
        assert passed, f"Found {null_count} null policy_ids"

    def test_customer_id_never_null(self, sample_claims):
        passed, null_count = expect_column_values_to_not_be_null(
            sample_claims, "customer_id"
        )
        assert passed, f"Found {null_count} null customer_ids"

    def test_policy_id_never_null_in_policies(self, sample_policies):
        passed, null_count = expect_column_values_to_not_be_null(
            sample_policies, "policy_id"
        )
        assert passed, f"Found {null_count} null policy_ids"


# ─────────────────────────────────────────────
# TESTS — RANGE EXPECTATIONS
# ─────────────────────────────────────────────

class TestRangeExpectations:

    def test_claim_amount_always_positive(self, sample_claims):
        passed, count = expect_column_values_to_be_between(
            sample_claims, "claim_amount_chf", 0.01, 10_000_000.0
        )
        assert passed, f"{count} claims have invalid amounts"

    def test_risk_score_between_zero_and_one(self, sample_policies):
        passed, count = expect_column_values_to_be_between(
            sample_policies, "risk_score", 0.0, 1.0
        )
        assert passed, f"{count} policies have invalid risk scores"

    def test_annual_premium_positive(self, sample_policies):
        passed, count = expect_column_values_to_be_between(
            sample_policies, "annual_premium_chf", 0.01, 1_000_000.0
        )
        assert passed, f"{count} policies have invalid premiums"

    def test_coverage_exceeds_premium(self, sample_policies):
        invalid = sample_policies.filter(
            F.col("coverage_amount_chf") <= F.col("annual_premium_chf")
        ).count()
        assert invalid == 0, \
            f"{invalid} policies have coverage <= premium"


# ─────────────────────────────────────────────
# TESTS — SET MEMBERSHIP EXPECTATIONS
# ─────────────────────────────────────────────

class TestSetMembershipExpectations:

    def test_claim_status_in_valid_set(self, sample_claims):
        valid = {
            "submitted", "under_review", "approved",
            "rejected", "settled", "withdrawn"
        }
        passed, count = expect_column_values_to_be_in_set(
            sample_claims, "claim_status", valid
        )
        assert passed, f"{count} claims have invalid status"

    def test_policy_type_in_valid_set(self, sample_policies):
        valid = {
            "motor", "home", "life",
            "health", "travel", "commercial"
        }
        passed, count = expect_column_values_to_be_in_set(
            sample_policies, "policy_type", valid
        )
        assert passed, f"{count} policies have invalid type"

    def test_claim_type_in_valid_set(self, sample_claims):
        valid = {
            "accident", "theft", "fire", "flood",
            "liability", "medical", "travel_delay"
        }
        passed, count = expect_column_values_to_be_in_set(
            sample_claims, "claim_type", valid
        )
        assert passed, f"{count} claims have invalid type"


# ─────────────────────────────────────────────
# TESTS — UNIQUENESS EXPECTATIONS
# ─────────────────────────────────────────────

class TestUniquenessExpectations:

    def test_claim_ids_unique(self, sample_claims):
        passed, duplicates = expect_column_values_to_be_unique(
            sample_claims, "claim_id"
        )
        assert passed, f"Found {duplicates} duplicate claim_ids"

    def test_policy_ids_unique(self, sample_policies):
        passed, duplicates = expect_column_values_to_be_unique(
            sample_policies, "policy_id"
        )
        assert passed, f"Found {duplicates} duplicate policy_ids"


# ─────────────────────────────────────────────
# TESTS — FORMAT EXPECTATIONS
# ─────────────────────────────────────────────

class TestFormatExpectations:

    def test_claim_id_format(self, sample_claims):
        passed, count = expect_column_values_to_match_regex(
            sample_claims, "claim_id", r"^CLM-[A-Z0-9]+$"
        )
        assert passed, f"{count} claim_ids have invalid format"

    def test_policy_id_format(self, sample_policies):
        passed, count = expect_column_values_to_match_regex(
            sample_policies, "policy_id", r"^POL-[A-Z0-9]+$"
        )
        assert passed, f"{count} policy_ids have invalid format"

    def test_incident_date_format(self, sample_claims):
        passed, count = expect_column_values_to_match_regex(
            sample_claims, "incident_date", r"^\d{4}-\d{2}-\d{2}$"
        )
        assert passed, f"{count} incident_dates have invalid format"


# ─────────────────────────────────────────────
# TESTS — VOLUME EXPECTATIONS
# ─────────────────────────────────────────────

class TestVolumeExpectations:

    def test_claims_table_not_empty(self, sample_claims):
        passed, count = expect_table_row_count_to_be_between(
            sample_claims, min_rows=1, max_rows=10_000_000
        )
        assert passed, f"Claims count {count} outside expected range"

    def test_policies_table_not_empty(self, sample_policies):
        passed, count = expect_table_row_count_to_be_between(
            sample_policies, min_rows=1, max_rows=10_000_000
        )
        assert passed, f"Policies count {count} outside expected range"

    def test_claims_to_policies_ratio_reasonable(self, sample_claims, sample_policies):
        claim_count  = sample_claims.count()
        policy_count = sample_policies.count()
        ratio = claim_count / policy_count if policy_count > 0 else 0
        assert ratio <= 10.0, \
            f"Claims/policy ratio {ratio:.2f} unusually high — possible data issue"
