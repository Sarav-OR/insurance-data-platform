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

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dq_rules import get_bronze_rules, get_silver_rules, BRONZE_DQ_RULES
from src.config import FRAUD_WEIGHTS


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    return SparkSession.builder \
        .master("local[2]") \
        .appName("insurance_dq_tests") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()


@pytest.fixture
def good_customers_df(spark):
    """Valid customer records — all should pass DQ."""
    data = [
        ("CUST-001", "john@example.com", "CH", True),
        ("CUST-002", "jane@test.org",    "CH", False),
        ("CUST-003", "bob@company.com",  "DE", True),
    ]
    return spark.createDataFrame(
        data,
        ["customer_id", "email", "country", "is_high_value"]
    )


@pytest.fixture
def bad_customers_df(spark):
    """Invalid customer records — all should fail DQ."""
    data = [
        (None,       "john@example.com", "CH"),   # null customer_id
        ("CUST-002", None,               "CH"),   # null email
        ("CUST-003", "invalid-email",    "CH"),   # no @ in email
        ("CUST-004", "valid@test.com",   None),   # null country
    ]
    return spark.createDataFrame(
        data,
        ["customer_id", "email", "country"]
    )


@pytest.fixture
def good_claims_df(spark):
    """Valid claim records."""
    data = [
        ("CLM-001", "POL-001", "CUST-001", 5000.0, "2023-01-15", 7),
        ("CLM-002", "POL-002", "CUST-002", 12000.0, "2023-03-20", 14),
    ]
    return spark.createDataFrame(
        data,
        ["claim_id", "policy_id", "customer_id",
         "claim_amount_chf", "incident_date", "days_to_submit"]
    )


@pytest.fixture
def bad_claims_df(spark):
    """Invalid claim records."""
    data = [
        (None,      "POL-001", "CUST-001", 5000.0,  "2023-01-15", 7),   # null claim_id
        ("CLM-002", None,      "CUST-002", 12000.0, "2023-03-20", 14),  # null policy_id
        ("CLM-003", "POL-003", "CUST-003", -100.0,  "2023-05-10", 5),   # negative amount
        ("CLM-004", "POL-004", "CUST-004", 8000.0,  None,         10),  # null incident_date
        ("CLM-005", "POL-005", "CUST-005", 3000.0,  "2023-07-01", -1),  # negative days
    ]
    return spark.createDataFrame(
        data,
        ["claim_id", "policy_id", "customer_id",
         "claim_amount_chf", "incident_date", "days_to_submit"]
    )


# ─────────────────────────────────────────────
# HELPER: Apply DQ rules for tests
# ─────────────────────────────────────────────

def apply_dq(sdf, rules):
    """Apply DQ rules and return (good, bad) DataFrames."""
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
        """All expected domains have Bronze DQ rules."""
        expected_domains = [
            "customers", "policies", "claims", "premiums", "fraud_signals"
        ]
        for domain in expected_domains:
            rules = get_bronze_rules(domain)
            assert len(rules) > 0, f"No Bronze DQ rules found for {domain}"

    def test_unknown_domain_returns_empty_rules(self):
        """Unknown domain returns empty rule list, not error."""
        rules = get_bronze_rules("nonexistent_domain")
        assert rules == []

    def test_each_rule_is_tuple_of_two_strings(self):
        """Every rule must be a (expression, error_code) tuple."""
        for domain, rules in BRONZE_DQ_RULES.items():
            for rule in rules:
                assert isinstance(rule, tuple), \
                    f"Rule in {domain} is not a tuple: {rule}"
                assert len(rule) == 2, \
                    f"Rule in {domain} does not have 2 elements: {rule}"
                assert isinstance(rule[0], str), \
                    f"Rule expression in {domain} is not a string"
                assert isinstance(rule[1], str), \
                    f"Rule error code in {domain} is not a string"

    def test_error_codes_start_with_err_prefix(self):
        """All error codes follow ERR_ naming convention."""
        for domain, rules in BRONZE_DQ_RULES.items():
            for _, code in rules:
                assert code.startswith("ERR_"), \
                    f"Error code '{code}' in {domain} does not start with ERR_"


# ─────────────────────────────────────────────
# TESTS — CUSTOMER DQ
# ─────────────────────────────────────────────

class TestCustomerDQ:

    def test_good_customers_all_pass(self, good_customers_df):
        """All valid customers pass DQ rules."""
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(good_customers_df, rules)
        assert good.count() == 3
        assert bad.count() == 0
        
    def test_null_customer_id_rejected(self, spark):
        """Customer with null ID is rejected."""
        schema = T.StructType([
        T.StructField("customer_id", T.StringType(), True),
        T.StructField("email",       T.StringType(), True),
        T.StructField("country",     T.StringType(), True),
        ])
        data = [(None, "test@test.com", "CH")]
        sdf = spark.createDataFrame(data, schema)        

    
    def test_invalid_email_rejected(self, spark):
        """Customer with email missing @ is rejected."""
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
        """All invalid customer records are rejected."""
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(bad_customers_df, rules)
        assert good.count() == 0
        assert bad.count() == 4

    def test_rejected_records_contain_error_codes(self, bad_customers_df):
        """Rejected records have non-empty _dq_errors column."""
        rules = get_bronze_rules("customers")
        _, bad = apply_dq(bad_customers_df, rules)
        error_codes = [
            row["_dq_errors"]
            for row in bad.select("_dq_errors").collect()
        ]
        for code in error_codes:
            assert code is not None and len(code) > 0


# ─────────────────────────────────────────────
# TESTS — CLAIMS DQ
# ─────────────────────────────────────────────

class TestClaimsDQ:

    def test_good_claims_all_pass(self, good_claims_df):
        """All valid claims pass DQ rules."""
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(good_claims_df, rules)
        assert good.count() == 2
        assert bad.count() == 0

    def test_negative_claim_amount_rejected(self, spark):
        """Claim with negative amount is rejected."""
        schema = T.StructType([
        T.StructField("claim_id",          T.StringType(),  True),
        T.StructField("policy_id",         T.StringType(),  True),
        T.StructField("customer_id",       T.StringType(),  True),
        T.StructField("claim_amount_chf",  T.DoubleType(),  True),
        T.StructField("incident_date",     T.StringType(),  True),
        T.StructField("days_to_submit",    T.IntegerType(), True),
    ])
    data = [("CLM-001", "POL-001", "CUST-001", -500.0, "2023-01-15", 5)]
    sdf = spark.createDataFrame(data, schema)
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 1

    def test_negative_days_to_submit_rejected(self, spark):
        """Claim with negative days_to_submit is rejected."""
        schema = T.StructType([
        T.StructField("claim_id",          T.StringType(),  True),
        T.StructField("policy_id",         T.StringType(),  True),
        T.StructField("customer_id",       T.StringType(),  True),
        T.StructField("claim_amount_chf",  T.DoubleType(),  True),
        T.StructField("incident_date",     T.StringType(),  True),
        T.StructField("days_to_submit",    T.IntegerType(), True),
    ])
    data = [("CLM-001", "POL-001", "CUST-001", 5000.0, "2023-01-15", -3)]
    sdf = spark.createDataFrame(data, schema)   "claim_amount_chf", "incident_date", "days_to_submit"]
        )
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(sdf, rules)
        assert good.count() == 0
        assert bad.count() == 1

    def test_all_bad_claims_rejected(self, bad_claims_df):
        """All invalid claim records are rejected."""
        rules = get_bronze_rules("claims")
        good, bad = apply_dq(bad_claims_df, rules)
        assert good.count() == 0
        assert bad.count() == 5


# ─────────────────────────────────────────────
# TESTS — FRAUD CONFIG
# ─────────────────────────────────────────────

class TestFraudConfig:

    def test_fraud_weights_sum_to_one(self):
        """Fraud scoring weights must sum to exactly 1.0."""
        total = sum(FRAUD_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, \
            f"Fraud weights sum to {total}, expected 1.0"

    def test_all_fraud_weight_keys_present(self):
        """All required fraud weight keys are present."""
        required = [
            "rule_score", "statistical_score",
            "behavioural_score", "network_score"
        ]
        for key in required:
            assert key in FRAUD_WEIGHTS, \
                f"Missing fraud weight key: {key}"

    def test_all_fraud_weights_positive(self):
        """All fraud weights are positive values."""
        for key, value in FRAUD_WEIGHTS.items():
            assert value > 0, f"Fraud weight '{key}' is not positive: {value}"


# ─────────────────────────────────────────────
# TESTS — DQ PASS RATE CALCULATION
# ─────────────────────────────────────────────

class TestDQPassRate:

    def test_mixed_records_correct_split(self, spark):
        """Mixed good/bad records split correctly."""
        data = [
            ("CUST-001", "valid@test.com",  "CH"),   # good
            ("CUST-002", "also@valid.org",  "CH"),   # good
            (None,       "test@test.com",   "CH"),   # bad — null id
            ("CUST-004", "bademail",        "CH"),   # bad — invalid email
        ]
        sdf = spark.createDataFrame(data, ["customer_id", "email", "country"])
        rules = get_bronze_rules("customers")
        good, bad = apply_dq(sdf, rules)

        assert good.count() == 2
        assert bad.count() == 2

    def test_empty_dataframe_handled(self, spark):
        """Empty DataFrame handled without error."""
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
