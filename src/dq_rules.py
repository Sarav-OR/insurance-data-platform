"""
src/dq_rules.py
===============
Declarative Data Quality rules for Insurance Data Platform.

Rules defined as (sql_expression, error_code) tuples.
Adding a new rule = one line change.
Rules are testable, auditable, and documentable.
"""

from typing import Dict, List, Tuple

DQRule = Tuple[str, str]
DQRuleSet = Dict[str, List[DQRule]]

# ─────────────────────────────────────────────
# BRONZE DQ RULES
# Applied at raw ingestion — catch structural issues
# ─────────────────────────────────────────────

BRONZE_DQ_RULES: DQRuleSet = {
    "customers": [
        ("customer_id IS NOT NULL",         "ERR_NULL_CUSTOMER_ID"),
        ("length(customer_id) > 0",         "ERR_EMPTY_CUSTOMER_ID"),
        ("email IS NOT NULL",               "ERR_NULL_EMAIL"),
        ("email LIKE '%@%'",                "ERR_INVALID_EMAIL"),
        ("country IS NOT NULL",             "ERR_NULL_COUNTRY"),
    ],
    "policies": [
        ("policy_id IS NOT NULL",           "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",         "ERR_NULL_CUSTOMER_ID"),
        ("annual_premium_chf > 0",          "ERR_INVALID_PREMIUM"),
        ("coverage_amount_chf > 0",         "ERR_INVALID_COVERAGE"),
        ("start_date IS NOT NULL",          "ERR_NULL_START_DATE"),
        ("end_date IS NOT NULL",            "ERR_NULL_END_DATE"),
    ],
    "claims": [
        ("claim_id IS NOT NULL",            "ERR_NULL_CLAIM_ID"),
        ("policy_id IS NOT NULL",           "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",         "ERR_NULL_CUSTOMER_ID"),
        ("claim_amount_chf > 0",            "ERR_INVALID_CLAIM_AMOUNT"),
        ("incident_date IS NOT NULL",       "ERR_NULL_INCIDENT_DATE"),
        ("days_to_submit >= 0",             "ERR_NEGATIVE_DAYS"),
    ],
    "premiums": [
        ("payment_id IS NOT NULL",          "ERR_NULL_PAYMENT_ID"),
        ("policy_id IS NOT NULL",           "ERR_NULL_POLICY_ID"),
        ("amount_due_chf > 0",              "ERR_INVALID_AMOUNT_DUE"),
        ("due_date IS NOT NULL",            "ERR_NULL_DUE_DATE"),
    ],
    "fraud_signals": [
        ("signal_id IS NOT NULL",           "ERR_NULL_SIGNAL_ID"),
        ("claim_id IS NOT NULL",            "ERR_NULL_CLAIM_ID"),
        ("signal_score >= 0",               "ERR_INVALID_SCORE"),
        ("signal_score <= 1",               "ERR_SCORE_OUT_OF_RANGE"),
    ],
}

# ─────────────────────────────────────────────
# SILVER DQ RULES
# Applied post-transformation — catch business logic issues
# ─────────────────────────────────────────────

SILVER_DQ_RULES: DQRuleSet = {
    "claims": [
        ("claim_amount_chf > 0",                        "ERR_INVALID_CLAIM_AMOUNT"),
        ("days_to_submit >= 0",                         "ERR_NEGATIVE_DAYS"),
        ("claim_severity IS NOT NULL",                  "ERR_NULL_SEVERITY"),
        ("fraud_risk_level IS NOT NULL",                "ERR_NULL_FRAUD_RISK"),
    ],
    "policies": [
        ("annual_premium_chf > 0",                      "ERR_INVALID_PREMIUM"),
        ("risk_band IS NOT NULL",                       "ERR_NULL_RISK_BAND"),
        ("premium_band IS NOT NULL",                    "ERR_NULL_PREMIUM_BAND"),
        ("policy_duration_days > 0",                    "ERR_INVALID_DURATION"),
    ],
    "premiums": [
        ("amount_due_chf > 0",                          "ERR_INVALID_AMOUNT"),
        ("overdue_band IS NOT NULL",                    "ERR_NULL_OVERDUE_BAND"),
    ],
}

# ─────────────────────────────────────────────
# RULE REGISTRY
# ─────────────────────────────────────────────

def get_bronze_rules(domain: str) -> List[DQRule]:
    """Get Bronze DQ rules for a domain."""
    return BRONZE_DQ_RULES.get(domain, [])


def get_silver_rules(domain: str) -> List[DQRule]:
    """Get Silver DQ rules for a domain."""
    return SILVER_DQ_RULES.get(domain, [])


def list_all_rules() -> None:
    """Print all rules across all domains."""
    for layer, ruleset in [("BRONZE", BRONZE_DQ_RULES),
                            ("SILVER", SILVER_DQ_RULES)]:
        print(f"\n{layer} RULES")
        print("-" * 40)
        for domain, rules in ruleset.items():
            print(f"\n  {domain}:")
            for rule, code in rules:
                print(f"    [{code}] {rule}")


if __name__ == "__main__":
    list_all_rules()
