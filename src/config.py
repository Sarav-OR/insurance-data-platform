"""
src/config.py
=============
Central Configuration for Insurance Data Platform
==================================================
Purpose : Single source of truth for ALL configuration.
          No hardcoded values anywhere else in the codebase.
          Change environment (dev/staging/prod) by changing
          values here — notebooks never need to change.

Usage   : from src.config import CONFIG, BATCH_ID, DOMAINS

Design  : In production these values come from:
          - Azure Key Vault (secrets)
          - Databricks Widgets (runtime parameters)
          - Environment variables
          Here we define defaults for development.
"""

from datetime import datetime, date

# ─────────────────────────────────────────────────────────
# BATCH IDENTIFIER
# Unique ID for every pipeline run.
# Every record written to Delta carries this ID.
# Answers: "show me everything loaded in run X"
# ─────────────────────────────────────────────────────────

BATCH_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────────────────
# DATABASE NAMES
# Separate databases per layer — clear ownership boundary.
# Bronze team owns bronze db, silver team owns silver db.
# In Unity Catalog these become catalog.schema references.
# ─────────────────────────────────────────────────────────

DATABASES = {
    "bronze": "insurance_bronze",
    "silver": "insurance_silver",
    "gold":   "insurance_gold",
    "fraud":  "insurance_fraud",
}

# ─────────────────────────────────────────────────────────
# DATA GENERATION CONFIG
# Controls how much synthetic data is generated.
# Increase these numbers to test at larger scale.
# seed=42 ensures same data every run — reproducible tests.
# ─────────────────────────────────────────────────────────

GENERATION = {
    "seed":          42,
    "num_customers": 10_000,
    "num_policies":  15_000,
    "num_claims":    8_000,
    "num_premiums":  50_000,
    "fraud_rate":    0.05,       # 5% of claims are fraudulent
    "start_date":    date(2019, 1, 1),
    "end_date":      date(2024, 12, 31),
}

# ─────────────────────────────────────────────────────────
# VOLUME THRESHOLDS
# Expected min/max record counts per domain.
# If count falls outside range → volume anomaly alert.
# Set based on historical averages ± 10% buffer.
# Catches: source outage, duplicate send, partial file.
# ─────────────────────────────────────────────────────────

VOLUME_THRESHOLDS = {
    "customers":     {"min": 9_000,  "max": 11_000},
    "policies":      {"min": 14_000, "max": 16_000},
    "claims":        {"min": 7_000,  "max": 9_000},
    "premiums":      {"min": 45_000, "max": 55_000},
    "fraud_signals": {"min": 100,    "max": 1_000},
}

# ─────────────────────────────────────────────────────────
# DELTA LAKE SETTINGS
# Applied via spark.conf at cluster/notebook level.
# optimizeWrite: auto-sizes files on write
# autoCompact:   auto-merges small files after write
# Both reduce need for manual OPTIMIZE runs.
# ─────────────────────────────────────────────────────────

DELTA_SETTINGS = {
    "spark.databricks.delta.optimizeWrite.enabled": "true",
    "spark.databricks.delta.autoCompact.enabled":   "true",
    "spark.sql.shuffle.partitions":                 "8",
}

# ─────────────────────────────────────────────────────────
# FRAUD SCORING WEIGHTS
# How much each signal type contributes to composite score.
# Must sum to exactly 1.0.
# Adjust weights based on model performance review.
# ─────────────────────────────────────────────────────────

FRAUD_WEIGHTS = {
    "rule_score":          0.35,
    "statistical_score":   0.25,
    "behavioural_score":   0.25,
    "network_score":       0.15,
}

# Validate weights sum to 1.0 at import time
# Catches misconfiguration before pipeline runs
_weight_sum = sum(FRAUD_WEIGHTS.values())
assert abs(_weight_sum - 1.0) < 1e-9, \
    f"Fraud weights must sum to 1.0, got {_weight_sum}"

# ─────────────────────────────────────────────────────────
# FRAUD THRESHOLDS
# Score boundaries for tier assignment.
# CRITICAL → immediate investigation required
# HIGH     → review within 24 hours
# MEDIUM   → review within 5 business days
# LOW      → no action required
# ─────────────────────────────────────────────────────────

FRAUD_THRESHOLDS = {
    "critical":              0.75,
    "high":                  0.50,
    "medium":                0.25,
    "z_score_anomaly":       2.0,
    "velocity_window_days":  30,
    "max_claims_per_window": 3,
}

# ─────────────────────────────────────────────────────────
# REFERENCE DATA
# Lookup lists used across all domains.
# Defined once here — imported by all notebooks.
# ─────────────────────────────────────────────────────────

POLICY_TYPES   = ["motor","home","life","health","travel","commercial"]
POLICY_STATUS  = ["active","lapsed","cancelled","expired","pending"]
CLAIM_STATUSES = ["submitted","under_review","approved",
                  "rejected","settled","withdrawn"]
CLAIM_TYPES    = ["accident","theft","fire","flood",
                  "liability","medical","travel_delay"]
PAY_METHODS    = ["direct_debit","credit_card",
                  "bank_transfer","cheque"]
CHANNELS       = ["online","broker","agent",
                  "direct_call","mobile_app"]
CURRENCIES     = ["CHF","EUR"]
FRAUD_TYPES    = [
    "multiple_claims_same_period",
    "claim_exceeds_policy_limit",
    "late_policy_inception",
    "duplicate_claimant_details",
    "high_frequency_claimant",
    "address_mismatch",
    "third_party_anomaly",
]

# ─────────────────────────────────────────────────────────
# BAND DEFINITIONS
# Business classification bands used in Silver layer.
# Tuple format: (min, max, label)
# max=None means no upper bound.
# ─────────────────────────────────────────────────────────

CLAIM_SEVERITY_BANDS = [
    (0,      5_000,   "LOW"),
    (5_000,  25_000,  "MEDIUM"),
    (25_000, 75_000,  "HIGH"),
    (75_000, None,    "CATASTROPHIC"),
]

RISK_BANDS = [
    (0.0,  0.3,  "LOW"),
    (0.3,  0.6,  "MEDIUM"),
    (0.6,  0.8,  "HIGH"),
    (0.8,  None, "VERY_HIGH"),
]

PREMIUM_BANDS = [
    (0,    500,   "LOW"),
    (500,  2000,  "MEDIUM"),
    (2000, 4000,  "HIGH"),
    (4000, None,  "VERY_HIGH"),
]

OVERDUE_BANDS = [
    (0,  0,   "ON_TIME"),
    (1,  30,  "1_30_DAYS"),
    (31, 60,  "31_60_DAYS"),
    (61, 90,  "61_90_DAYS"),
    (91, None,"90_PLUS_DAYS"),
]

# ─────────────────────────────────────────────────────────
# PARTITION COLUMNS PER DOMAIN
# Defines which columns to partition by for each table.
# Partitioning separates data into folders — improves
# query performance when filtering on these columns.
# ─────────────────────────────────────────────────────────

PARTITION_COLS = {
    "bronze": {
        "customers":     None,
        "policies":      ["policy_type"],
        "claims":        ["claim_status"],
        "premiums":      ["payment_status"],
        "fraud_signals": ["signal_type"],
    },
    "silver": {
        "customers":     None,
        "policies":      ["policy_type"],
        "claims":        ["claim_status"],
        "premiums":      ["payment_status"],
        "fraud_signals": ["signal_type"],
        "claims_enriched": ["claim_status"],
    },
    "gold": {
        "portfolio_summary":   ["policy_type"],
        "claims_kpis":         ["policy_type"],
        "customer_segments":   None,
        "premium_collections": ["policy_type"],
        "fraud_summary":       ["signal_type"],
        "executive_summary":   None,
    },
}

# ─────────────────────────────────────────────────────────
# CONVENIENCE ALIAS
# Single CONFIG dict for backward compatibility
# and simple import in notebooks.
# ─────────────────────────────────────────────────────────

CONFIG = {
    "batch_id":          BATCH_ID,
    "databases":         DATABASES,
    "generation":        GENERATION,
    "volume_thresholds": VOLUME_THRESHOLDS,
    "delta_settings":    DELTA_SETTINGS,
    "fraud_weights":     FRAUD_WEIGHTS,
    "fraud_thresholds":  FRAUD_THRESHOLDS,
    "partition_cols":    PARTITION_COLS,
}
