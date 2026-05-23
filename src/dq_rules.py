"""
src/dq_rules.py
===============
Data Quality Rules for Insurance Data Platform
===============================================
Purpose : Declarative DQ rules for every domain at every layer.
          Rules defined as (SQL expression, error code) tuples.
          Adding a new rule = one line change.
          Rules are testable, auditable and documentable.

Usage   : from src.dq_rules import get_rules, BRONZE_RULES

Design  : Rules are SQL expressions evaluated by Spark.
          If expression evaluates to FALSE → record is rejected.
          Error codes follow ERR_[DESCRIPTION] convention.
          Multiple errors on one record are pipe-separated:
          ERR_NULL_CLAIM_ID|ERR_INVALID_AMOUNT

Rule levels:
  BRONZE rules → structural validity (nulls, types, ranges)
  SILVER rules → business validity (derived column correctness)
"""

from typing import Dict, List, Tuple

# Type aliases for clarity
DQRule    = Tuple[str, str]           # (sql_expression, error_code)
DQRuleSet = Dict[str, List[DQRule]]   # domain → list of rules

# ─────────────────────────────────────────────────────────
# BRONZE DQ RULES
# Applied at raw ingestion — catches structural issues.
# These are the minimum validity requirements.
# A record failing these cannot be trusted at all.
# ─────────────────────────────────────────────────────────

BRONZE_RULES: DQRuleSet = {

    "customers": [
        # Primary key must exist and be non-empty
        ("customer_id IS NOT NULL",
         "ERR_NULL_CUSTOMER_ID"),
        ("length(customer_id) > 0",
         "ERR_EMPTY_CUSTOMER_ID"),
        # Email must exist and contain @ sign
        ("email IS NOT NULL",
         "ERR_NULL_EMAIL"),
        ("email LIKE '%@%'",
         "ERR_INVALID_EMAIL_FORMAT"),
        # Country must exist — used for regulatory compliance
        ("country IS NOT NULL",
         "ERR_NULL_COUNTRY"),
    ],

    "policies": [
        # Primary and foreign keys
        ("policy_id IS NOT NULL",
         "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",
         "ERR_NULL_CUSTOMER_ID"),
        # Premium must be positive — zero premium invalid
        ("annual_premium_chf > 0",
         "ERR_INVALID_PREMIUM"),
        # Coverage must be positive
        ("coverage_amount_chf > 0",
         "ERR_INVALID_COVERAGE"),
        # Both dates required for policy validity period
        ("start_date IS NOT NULL",
         "ERR_NULL_START_DATE"),
        ("end_date IS NOT NULL",
         "ERR_NULL_END_DATE"),
    ],

    "claims": [
        # All three keys required for referential integrity
        ("claim_id IS NOT NULL",
         "ERR_NULL_CLAIM_ID"),
        ("policy_id IS NOT NULL",
         "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",
         "ERR_NULL_CUSTOMER_ID"),
        # Claim amount must be positive
        ("claim_amount_chf > 0",
         "ERR_INVALID_CLAIM_AMOUNT"),
        # Incident date required for timeline analysis
        ("incident_date IS NOT NULL",
         "ERR_NULL_INCIDENT_DATE"),
        # Days to submit cannot be negative
        # (submitted before incident = data error)
        ("days_to_submit >= 0",
         "ERR_NEGATIVE_DAYS_TO_SUBMIT"),
    ],

    "premiums": [
        ("payment_id IS NOT NULL",
         "ERR_NULL_PAYMENT_ID"),
        ("policy_id IS NOT NULL",
         "ERR_NULL_POLICY_ID"),
        # Amount due must be positive
        ("amount_due_chf > 0",
         "ERR_INVALID_AMOUNT_DUE"),
        # Due date required for arrears calculation
        ("due_date IS NOT NULL",
         "ERR_NULL_DUE_DATE"),
    ],

    "fraud_signals": [
        ("signal_id IS NOT NULL",
         "ERR_NULL_SIGNAL_ID"),
        ("claim_id IS NOT NULL",
         "ERR_NULL_CLAIM_ID"),
        # Score must be between 0 and 1 (probability)
        ("signal_score >= 0",
         "ERR_INVALID_SIGNAL_SCORE"),
        ("signal_score <= 1",
         "ERR_SIGNAL_SCORE_OUT_OF_RANGE"),
    ],
}

# ─────────────────────────────────────────────────────────
# SILVER DQ RULES
# Applied post-transformation — catches business logic issues.
# These validate that derived columns were calculated correctly.
# ─────────────────────────────────────────────────────────

SILVER_RULES: DQRuleSet = {

    "claims": [
        # Claim amount still positive after transformation
        ("claim_amount_chf > 0",
         "ERR_INVALID_CLAIM_AMOUNT"),
        # Days to submit non-negative after type cast
        ("days_to_submit >= 0",
         "ERR_NEGATIVE_DAYS_TO_SUBMIT"),
        # Derived column must exist
        ("claim_severity IS NOT NULL",
         "ERR_NULL_CLAIM_SEVERITY"),
        # Derived column must be valid value
        ("claim_severity IN ('LOW','MEDIUM','HIGH','CATASTROPHIC')",
         "ERR_INVALID_CLAIM_SEVERITY"),
        # Fraud risk level must be calculated
        ("fraud_risk_level IS NOT NULL",
         "ERR_NULL_FRAUD_RISK_LEVEL"),
    ],

    "policies": [
        ("annual_premium_chf > 0",
         "ERR_INVALID_PREMIUM"),
        # Risk band must be valid
        ("risk_band IN ('LOW','MEDIUM','HIGH','VERY_HIGH')",
         "ERR_INVALID_RISK_BAND"),
        # Premium band must be valid
        ("premium_band IN ('LOW','MEDIUM','HIGH','VERY_HIGH')",
         "ERR_INVALID_PREMIUM_BAND"),
        # Duration must be positive
        ("policy_duration_days > 0",
         "ERR_INVALID_POLICY_DURATION"),
    ],

    "premiums": [
        ("amount_due_chf > 0",
         "ERR_INVALID_AMOUNT_DUE"),
        # Overdue band must be valid
        ("""overdue_band IN (
            'ON_TIME','1_30_DAYS','31_60_DAYS',
            '61_90_DAYS','90_PLUS_DAYS'
         )""",
         "ERR_INVALID_OVERDUE_BAND"),
    ],

    "fraud_signals": [
        ("signal_score >= 0",
         "ERR_INVALID_SIGNAL_SCORE"),
        ("signal_score <= 1",
         "ERR_SIGNAL_SCORE_OUT_OF_RANGE"),
        # Score band must be valid
        ("score_band IN ('LOW','MEDIUM','HIGH','CRITICAL')",
         "ERR_INVALID_SCORE_BAND"),
    ],
}

# ─────────────────────────────────────────────────────────
# PUBLIC API
# Functions to retrieve rules by layer and domain.
# ─────────────────────────────────────────────────────────

def get_rules(layer: str, domain: str) -> List[DQRule]:
    """
    Get DQ rules for a specific layer and domain.

    Args:
        layer  : 'bronze' or 'silver'
        domain : e.g. 'claims', 'policies', 'customers'

    Returns:
        List of (sql_expression, error_code) tuples.
        Empty list if no rules defined — never raises error.

    Example:
        rules = get_rules('bronze', 'claims')
        # Returns list of claim validation rules
    """
    ruleset = {
        "bronze": BRONZE_RULES,
        "silver": SILVER_RULES,
    }.get(layer.lower(), {})

    return ruleset.get(domain.lower(), [])


def list_all_rules() -> None:
    """
    Print all rules across all layers and domains.
    Useful for documentation and review.
    """
    for layer, ruleset in [
        ("BRONZE", BRONZE_RULES),
        ("SILVER", SILVER_RULES),
    ]:
        print(f"\n{'='*50}")
        print(f"{layer} DQ RULES")
        print(f"{'='*50}")
        for domain, rules in ruleset.items():
            print(f"\n  {domain.upper()} ({len(rules)} rules):")
            for rule, code in rules:
                print(f"    [{code}]")
                print(f"      Condition: {rule}")
                
                
# ─────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY ALIASES
# Tests and CI reference these names — keep them working
# ─────────────────────────────────────────────────────────

# Old name → new name mapping
BRONZE_DQ_RULES = BRONZE_RULES
SILVER_DQ_RULES = SILVER_RULES

def get_bronze_rules(domain: str) -> list:
    """Alias for get_rules('bronze', domain)."""
    return get_rules("bronze", domain)

def get_silver_rules(domain: str) -> list:
    """Alias for get_rules('silver', domain)."""
    return get_rules("silver", domain)                


if __name__ == "__main__":
    list_all_rules()
