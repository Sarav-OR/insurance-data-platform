"""
src/config.py
=============
Centralised configuration for Insurance Data Platform.
No hardcoded values anywhere else in the codebase.
All environment-specific values overridable via environment variables.
"""

import os
from datetime import date

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

ENV = os.getenv("ENVIRONMENT", "dev")  # dev | staging | prod

# ─────────────────────────────────────────────
# DATABASE NAMES
# ─────────────────────────────────────────────

BRONZE_DB = os.getenv("BRONZE_DB", "insurance_bronze")
SILVER_DB = os.getenv("SILVER_DB", "insurance_silver")
GOLD_DB   = os.getenv("GOLD_DB",   "insurance_gold")
FRAUD_DB  = os.getenv("FRAUD_DB",  "insurance_fraud")

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────

GENERATOR = {
    "seed":             42,
    "num_customers":    10_000,
    "num_policies":     15_000,
    "num_claims":       8_000,
    "num_premiums":     50_000,
    "fraud_rate":       0.05,
    "start_date":       date(2019, 1, 1),
    "end_date":         date(2024, 12, 31),
}

# ─────────────────────────────────────────────
# FRAUD SCORING WEIGHTS
# Must sum to 1.0
# ─────────────────────────────────────────────

FRAUD_WEIGHTS = {
    "rule_score":          0.35,
    "statistical_score":   0.25,
    "behavioural_score":   0.25,
    "network_score":       0.15,
}

assert abs(sum(FRAUD_WEIGHTS.values()) - 1.0) < 1e-9, \
    "Fraud weights must sum to 1.0"

# ─────────────────────────────────────────────
# FRAUD THRESHOLDS
# ─────────────────────────────────────────────

FRAUD_THRESHOLDS = {
    "critical":     0.75,
    "high":         0.50,
    "medium":       0.25,
    "z_score":      2.0,
    "velocity_days": 30,
    "max_claims_per_window": 3,
}

# ─────────────────────────────────────────────
# DELTA LAKE SETTINGS
# ─────────────────────────────────────────────

DELTA = {
    "merge_schema":       False,   # strict schema — no drift
    "auto_optimize":      True,
    "auto_compact":       True,
    "target_file_size_mb": 128,
}

# ─────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────

POLICY_TYPES   = ["motor", "home", "life", "health", "travel", "commercial"]
POLICY_STATUS  = ["active", "lapsed", "cancelled", "expired", "pending"]
CLAIM_STATUSES = ["submitted", "under_review", "approved", "rejected", "settled"]
PAY_METHODS    = ["direct_debit", "credit_card", "bank_transfer", "cheque"]
CHANNELS       = ["online", "broker", "agent", "direct_call", "mobile_app"]
CURRENCIES     = ["CHF", "EUR"]
CLAIM_TYPES    = ["accident", "theft", "fire", "flood", "liability",
                  "medical", "travel_delay"]
FRAUD_TYPES    = [
    "multiple_claims_same_period",
    "claim_exceeds_policy_limit",
    "late_policy_inception",
    "duplicate_claimant_details",
    "high_frequency_claimant",
    "address_mismatch",
    "third_party_anomaly",
]

# ─────────────────────────────────────────────
# BAND DEFINITIONS
# ─────────────────────────────────────────────

PREMIUM_BANDS = [
    (0,    500,   "LOW"),
    (500,  2000,  "MEDIUM"),
    (2000, 4000,  "HIGH"),
    (4000, None,  "VERY_HIGH"),
]

RISK_BANDS = [
    (0.0,  0.3,  "LOW"),
    (0.3,  0.6,  "MEDIUM"),
    (0.6,  0.8,  "HIGH"),
    (0.8,  None, "VERY_HIGH"),
]

CLAIM_SEVERITY_BANDS = [
    (0,      5_000,   "LOW"),
    (5_000,  25_000,  "MEDIUM"),
    (25_000, 75_000,  "HIGH"),
    (75_000, None,    "CATASTROPHIC"),
]
