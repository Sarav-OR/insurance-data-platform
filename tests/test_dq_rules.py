"""
tests/test_dq_rules.py
======================
Unit tests for Data Quality rules and utility functions.

Run locally:
    pip install pytest pyspark faker pandas pyarrow
    pytest tests/ -v

Run in CI:
    pytest tests/ -v --tb=short --junitxml=test-results.xml
"""

import sys
import os
import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dq_rules import get_bronze_rules, BRONZE_DQ_RULES
from src.config import FRAUD_WEIGHTS


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .master("local[2]") \
        .appName("insurance_dq_tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


@pytest.fixture
def good_customers_df(spark):
    schema = T.StructType([
        T.StructField("customer_id",   T.StringType(),  True),
        T.StructField("email",         T.StringType(),  True),
        T.StructField("country",       T.StringType(),  True),
        T.StructField("is_high_value", T.BooleanType(), True),
    ])
    data = [
        ("CUST-001", "john@example.com", "CH", True),
        ("CUST-002", "jane@test.org",    "CH", False),
        ("CUST-003", "bob@company.com",  "DE", True),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def bad_customers_df(spark):
    schema = T.StructType([
        T.StructField("customer_id", T.StringType(), True),
        T.StructField("email",       T.StringType(), True),
        T.StructField("country",     T.StringType(), True),
    ])
    data = [
        (None,       "john@example.com", "CH"),
        ("CUST-002", None,               "CH"),
        ("CUST-003", "invalid-email",    "CH"),
        ("CUST-004", "valid@test.com",   None),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def good_claims_df(spark):
    schema = T.StructType([
        T.StructField("claim_id",         T.StringType(),  True),
        T.StructField("policy_id",        T.StringType(),  True),
        T.StructField("customer_id",      T.StringType(),  True),
        T.StructField("claim_amount_chf", T.DoubleType(),  True),
        T.StructField("incident_date",    T.StringType(),  True),
        T.StructField("days_to_submit",   T.IntegerType(), True),
    ])
    data = [
        ("CLM-001", "POL-001", "CUST-001", 5000.0,  "2023-01-15", 7),
        ("CLM-002", "POL-002", "CUST-002", 12000.0, "2023-03-20", 14),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def bad_claims_df(spark):
    schema = T.StructType([
        T.StructField("claim_id",         T.StringType(),  True),
        T.StructField("policy_id",        T.StringType(),  True),
        T.StructField("customer_id",      T.StringType(),  True),
        T.StructField("claim_amount_chf", T.DoubleType(),  True),
        T.StructField("incident_date",    T.StringType(),  True),
        T.StructField("days_to_submit",   T.IntegerType(), True),
    ])
    data = [
        (None,      "POL-001", "CUST-001", 5000.0,  "2023-01-15", 7),
        ("CLM-002", None,      "CUST-002", 12000.0, "2023-03-20", 14),
        ("CLM-003", "POL-003", "CUST-003", -100.0,  "2023-05-10", 5),
        ("CLM-004", "POL-004", "CUST-004", 8000.0,  None,         10),
        ("CLM-005", "POL-005", "CUST-005", 3000.0,  "2023-07-01", -1),
    ]
    return spark.createDataFrame(data, schema)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def apply_dq(sdf, rules):
    sdf_tagged = sdf.withColumn(
        "_dq_errors",
        F.concat_ws("|", *[
            F.when(F.expr(f"NOT ({rule})"), F.lit(code))
             .otherwise(F.lit(None).cast("string"))
            for rule, code in rules
        ])
    )
    good = sdf_tagged.filter(
        (F.col("_dq_errors").isNull()) | (F.col("_dq_errors") == "")
    ).drop("_dq_errors")
    bad = sdf_tagged.filter(
        F.col("_dq_errors").isNotNull() & (F.col("_dq_errors") != "")
    )
    return good, bad


# ─────────────────────────────────────────────
# TESTS — DQ RULES REGISTRY
# ─────────────────────────────────────────────

class TestDQRulesRegistry:

    def test_bronze_rules_exist_for_all_domains(self):
        expected = ["customers", "policies", "claims", "premiums", "fraud_signals"]
        for domain in expected:
            rules = get_bronze_rules(domain)
            assert len(rules) > 0, f"No Bronze DQ rules for {domain}"

    def test_unknown_domain_returns_empty(self):
        assert get_bronze_rules("nonexistent") == []

    def test_each_rule_is_tuple_of_two_strings(self):
        for domain, rules in BRONZE_DQ_RULES.items():
            for rule in rules:
                assert isinstance(rule, tuple)
                assert len(rule) == 2
                assert isinstance(rule[0], str)
                assert isinstance(rule[1], str)

    def test_error_codes_start_with_err_prefix(self):
        for domain, rules in BRONZE_DQ_RULES.items():
            for _, code in rules:
                assert code.startswith("ERR_"), \
                    f"Bad code '{code}' in {domain}"


# ─────────────────────────────────────────────
# TESTS — CUSTOMER DQ
# ─────────────────────────────────────────────

class TestCustomerDQ:

    def test_good_customers_all_pass(self, good_customers_df):
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(good_customers_df, rules)
        assert good.count() == 3
        assert bad.count() == 0

    def test_null_customer_id_rejected(self, spark):
        schema = T.StructType([
            T.StructField("customer_id", T.StringType(), True),
            T.StructField("email",       T.StringType(), True),
            T.StructField("country",     T.StringType(), True),
        ])
        data = [(None, "test@test.com", "CH")]
        sdf = spark.createDataFrame(data, schema)
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 1

    def test_invalid_email_rejected(self, spark):
        schema = T.StructType([
            T.StructField("customer_id", T.StringType(), True),
            T.StructField("email",       T.StringType(), True),
            T.StructField("country",     T.StringType(), True),
        ])
        data = [("CUST-001", "invalidemail", "CH")]
        sdf = spark.createDataFrame(data, schema)
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 1

    def test_all_bad_customers_rejected(self, bad_customers_df):
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(bad_customers_df, rules)
        assert good.count() == 0
        assert bad.count() == 4

    def test_rejected_records_contain_error_codes(self, bad_customers_df):
        rules = get_bronze_rules("customers")
        _, bad = apply_dq(bad_customers_df, rules)
        for row in bad.select("_dq_errors").collect():
            assert row["_dq_errors"] is not None
            assert len(row["_dq_errors"]) > 0


# ─────────────────────────────────────────────
# TESTS — CLAIMS DQ
# ─────────────────────────────────────────────

class TestClaimsDQ:

    def test_good_claims_all_pass(self, good_claims_df):
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(good_claims_df, rules)
        assert good.count() == 2
        assert bad.count() == 0

    def test_negative_claim_amount_rejected(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",         T.StringType(),  True),
            T.StructField("policy_id",        T.StringType(),  True),
            T.StructField("customer_id",      T.StringType(),  True),
            T.StructField("claim_amount_chf", T.DoubleType(),  True),
            T.StructField("incident_date",    T.StringType(),  True),
            T.StructField("days_to_submit",   T.IntegerType(), True),
        ])
        data = [("CLM-001", "POL-001", "CUST-001", -500.0, "2023-01-15", 5)]
        sdf = spark.createDataFrame(data, schema)
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 1

    def test_negative_days_to_submit_rejected(self, spark):
        schema = T.StructType([
            T.StructField("claim_id",         T.StringType(),  True),
            T.StructField("policy_id",        T.StringType(),  True),
            T.StructField("customer_id",      T.StringType(),  True),
            T.StructField("claim_amount_chf", T.DoubleType(),  True),
            T.StructField("incident_date",    T.StringType(),  True),
            T.StructField("days_to_submit",   T.IntegerType(), True),
        ])
        data = [("CLM-001", "POL-001", "CUST-001", 5000.0, "2023-01-15", -3)]
        sdf = spark.createDataFrame(data, schema)
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 1

    def test_all_bad_claims_rejected(self, bad_claims_df):
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(bad_claims_df, rules)
        assert good.count() == 0
        assert bad.count() == 5


# ─────────────────────────────────────────────
# TESTS — FRAUD CONFIG
# ─────────────────────────────────────────────

class TestFraudConfig:

    def test_fraud_weights_sum_to_one(self):
        total = sum(FRAUD_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, \
            f"Fraud weights sum to {total}, expected 1.0"

    def test_all_fraud_weight_keys_present(self):
        required = [
            "rule_score", "statistical_score",
            "behavioural_score", "network_score"
        ]
        for key in required:
            assert key in FRAUD_WEIGHTS, f"Missing key: {key}"

    def test_all_fraud_weights_positive(self):
        for key, value in FRAUD_WEIGHTS.items():
            assert value > 0, f"Weight '{key}' not positive: {value}"


# ─────────────────────────────────────────────
# TESTS — DQ PASS RATE
# ─────────────────────────────────────────────

class TestDQPassRate:

    def test_mixed_records_correct_split(self, spark):
        schema = T.StructType([
            T.StructField("customer_id", T.StringType(), True),
            T.StructField("email",       T.StringType(), True),
            T.StructField("country",     T.StringType(), True),
        ])
        data = [
            ("CUST-001", "valid@test.com", "CH"),
            ("CUST-002", "also@valid.org", "CH"),
            (None,       "test@test.com",  "CH"),
            ("CUST-004", "bademail",       "CH"),
        ]
        sdf = spark.createDataFrame(data, schema)
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 2
        assert bad.count() == 2

    def test_empty_dataframe_handled(self, spark):
        schema = T.StructType([
            T.StructField("customer_id", T.StringType(), True),
            T.StructField("email",       T.StringType(), True),
            T.StructField("country",     T.StringType(), True),
        ])
        sdf = spark.createDataFrame([], schema)
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 0
