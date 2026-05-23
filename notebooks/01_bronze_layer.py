"""
notebooks/01_bronze_layer.py
=============================
Bronze Layer — Insurance Data Platform
=======================================
Purpose : Raw data ingestion with schema enforcement,
          data quality validation and quarantine pattern.
Layer   : Bronze (Raw)
Reads   : Synthetic data generated in-memory
Writes  : insurance_bronze.* Delta tables

Depends on:
  src/config.py     — configuration and constants
  src/dq_rules.py   — DQ rules per domain
  src/utils.py      — helper functions
  src/audit.py      — audit column functions
  src/monitoring.py — DQ tracking and error logging

Pipeline flow per domain:
  Generate data
    → Convert to Spark DataFrame
    → Add Bronze audit columns
    → Apply DQ rules
    → Good records  → Bronze Delta table
    → Bad records   → Quarantine table
    → DQ stats      → dq_monitoring table
    → Volume check  → alert if anomaly
"""

# ═══════════════════════════════════════════════════════════
# CELL 1 — Repository Path Setup
# Purpose : Add repo root to Python path so src/ imports work.
#           This must be the first cell in every notebook.
# Note    : Update the path to match your Databricks username.
# ═══════════════════════════════════════════════════════════

import sys
import os

# Add repo root to path — enables: from src.config import CONFIG
# Update YOUR_USERNAME to your Databricks workspace username
REPO_ROOT = "/Workspace/Repos/YOUR_USERNAME/insurance-data-platform"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"✅ Repo root added to path: {REPO_ROOT}")

# ═══════════════════════════════════════════════════════════
# CELL 2 — Imports
# Purpose : Import all required libraries and src/ modules.
#           All shared logic comes from src/ — no duplication.
# Imports :
#   Standard  — logging, datetime, random, uuid, hashlib
#   PySpark   — functions, types
#   pandas    — for data generation (local) before Spark
#   faker     — synthetic data generation
#   src/      — shared platform modules
# ═══════════════════════════════════════════════════════════

# Standard library
import logging
import random
import uuid
import hashlib
from datetime import datetime, timedelta, date

# Third party
import pandas as pd
from faker import Faker

# PySpark
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Platform shared modules
from src.config     import (CONFIG, BATCH_ID, DATABASES,
                             GENERATION, VOLUME_THRESHOLDS,
                             DELTA_SETTINGS, PARTITION_COLS,
                             POLICY_TYPES, POLICY_STATUS,
                             CLAIM_STATUSES, CLAIM_TYPES,
                             PAY_METHODS, CHANNELS,
                             CURRENCIES, FRAUD_TYPES)
from src.dq_rules   import get_rules
from src.utils      import (gen_id, rand_date, rand_amount,
                             audit_cols, age_band, write_delta,
                             apply_delta_settings)
from src.audit      import add_bronze_audit
from src.monitoring import (apply_dq_rules, write_dq_monitoring,
                             check_volume, log_pipeline_error)

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("bronze_layer")

# ── Convenience aliases ───────────────────────────────────
BRONZE_DB = DATABASES["bronze"]

print(f"✅ All imports successful")
print(f"   Batch ID : {BATCH_ID}")
print(f"   Database : {BRONZE_DB}")

# ═══════════════════════════════════════════════════════════
# CELL 3 — Seeding and Database Setup
# Purpose : Fix random seed for reproducible data generation.
#           Create database and apply Delta optimisations.
#           Using CREATE DATABASE IF NOT EXISTS — safe to
#           run multiple times without error.
# ═══════════════════════════════════════════════════════════

# Fix seed — same data generated every run
# Critical for testing: expected values are predictable
fake = Faker("en_GB")
Faker.seed(GENERATION["seed"])
random.seed(GENERATION["seed"])

# Create database if not exists
spark.sql(f"CREATE DATABASE IF NOT EXISTS {BRONZE_DB}")
spark.sql(f"USE {BRONZE_DB}")

# Apply Delta settings from config
# optimizeWrite: auto-sizes files on write
# autoCompact:   merges small files automatically
apply_delta_settings(spark, DELTA_SETTINGS)

print(f"✅ Database '{BRONZE_DB}' ready")
print(f"✅ Seed set to {GENERATION['seed']}")
print(f"✅ Delta optimisations applied")

# ═══════════════════════════════════════════════════════════
# CELL 4 — Domain Ingestion Orchestrator
# Purpose : Single function that runs the full pipeline
#           for any domain. All Bronze notebooks call this.
#           Encapsulates: convert → audit → DQ → write
#           → quarantine → monitoring → volume check
#
# Why one function:
#   Consistency — every domain follows identical pattern
#   Maintainability — fix logic once, applies everywhere
#   Testability — single function is easy to unit test
# ═══════════════════════════════════════════════════════════

def ingest_domain(pdf: pd.DataFrame,
                  domain: str) -> int:
    """
    Full ingestion pipeline for one domain.

    Steps:
    1. Convert Pandas DataFrame to Spark DataFrame
    2. Add Bronze audit columns (batch_id, timestamp, domain)
    3. Apply DQ rules — split into good and bad records
    4. Write good records to Bronze Delta table
    5. Write bad records to quarantine table
    6. Write DQ statistics to monitoring table
    7. Check volume within expected range

    Args:
        pdf    : Pandas DataFrame of generated records
        domain : Domain name e.g. 'claims', 'policies'

    Returns:
        Count of good records written to Delta
    """
    print(f"\n{'─'*50}")
    print(f"  INGESTING: {domain.upper()}")
    print(f"{'─'*50}")

    try:
        # ── Step 1: Convert to Spark ──────────────────
        # Pandas used for generation (simpler loops)
        # Spark used for processing (distributed, scalable)
        sdf = spark.createDataFrame(pdf)
        log.info(f"[{domain}] Raw records: {len(pdf):,}")

        # ── Step 2: Add Bronze audit columns ──────────
        # _bronze_batch_id, _bronze_load_timestamp,
        # _bronze_domain added to every record
        sdf = add_bronze_audit(sdf, domain, BATCH_ID)

        # ── Step 3: Apply DQ rules ────────────────────
        # Rules fetched from src/dq_rules.py
        # Returns (good_sdf, bad_sdf, good_count, bad_count)
        rules = get_rules("bronze", domain)
        good_sdf, bad_sdf, good_count, bad_count = \
            apply_dq_rules(sdf, domain, rules, BATCH_ID)

        # ── Step 4: Write good records to Delta ───────
        # partition_cols from config — no hardcoding
        partition_cols = PARTITION_COLS["bronze"].get(domain)
        final_count = write_delta(
            good_sdf, BRONZE_DB, domain, partition_cols
        )

        # ── Step 5: Write bad records to quarantine ───
        # Always create table even if empty
        # Monitoring queries expect table to exist
        _write_quarantine(bad_sdf, domain, good_sdf)

        # ── Step 6: DQ monitoring ─────────────────────
        # Appends pass rate stats to dq_monitoring table
        # Query this to track quality trends over time
        write_dq_monitoring(
            spark, domain, good_count, bad_count, BATCH_ID
        )

        # ── Step 7: Volume anomaly check ──────────────
        # Alert if record count outside expected range
        thresholds = VOLUME_THRESHOLDS.get(domain, {})
        check_volume(spark, domain, final_count,
                     thresholds, BATCH_ID)

        print(f"  ✅ Complete: {final_count:,} records ingested")
        return final_count

    except Exception as e:
        log.error(f"[{domain}] Pipeline FAILED: {str(e)}")
        log_pipeline_error(spark, domain, e, BATCH_ID)
        raise


def _write_quarantine(bad_sdf, domain: str, good_sdf) -> int:
    """
    Write rejected records to quarantine Delta table.
    Always creates table structure even if no bad records.
    Quarantine table = rejected_{domain}.

    Why always create:
    Monitoring dashboards query rejected_* tables.
    If table doesn't exist query fails even if pass rate = 100%.
    """
    table = f"{BRONZE_DB}.rejected_{domain}"

    if bad_sdf is None or bad_sdf.count() == 0:
        # Create empty table matching good record schema
        spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {table}
            USING DELTA
            AS SELECT * FROM {BRONZE_DB}.{domain}
            WHERE 1=0
        """)
        log.info(f"Quarantine [{domain}]: empty table created")
        return 0

    count = bad_sdf.count()
    bad_sdf.write.format("delta").mode("overwrite") \
           .saveAsTable(table)
    log.warning(
        f"Quarantine [{domain}]: {count:,} records rejected"
    )
    return count


print("✅ Ingestion orchestrator ready")

# ═══════════════════════════════════════════════════════════
# CELL 5 — Generate and Ingest: Customers
# Purpose : Generate synthetic customer records and ingest
#           to Bronze Delta table.
# Records : 10,000 customers
# Output  : insurance_bronze.customers
#           insurance_bronze.rejected_customers
# Key fields:
#   customer_id   — unique identifier (CUST-XXXXXXXXXX)
#   email         — validated by DQ rule (must contain @)
#   is_high_value — 10% of customers flagged premium
#   customer_since — drives tenure calculation in Silver
# ═══════════════════════════════════════════════════════════

log.info(f"Generating {GENERATION['num_customers']:,} customers...")

customers = []
for _ in range(GENERATION["num_customers"]):
    dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
    customers.append({
        "customer_id":    gen_id("CUST"),
        "first_name":     fake.first_name(),
        "last_name":      fake.last_name(),
        "date_of_birth":  dob.isoformat(),
        "age_band":       age_band(dob),
        "gender":         random.choice(["M","F","Other"]),
        "email":          fake.email(),
        "phone":          fake.phone_number(),
        "address_line1":  fake.street_address(),
        "city":           fake.city(),
        "postcode":       fake.postcode(),
        "country":        "CH",
        "customer_since": rand_date(
            date(2005,1,1), date(2023,12,31)
        ).isoformat(),
        "channel":        random.choice(CHANNELS),
        "is_high_value":  random.random() < 0.10,
        **audit_cols(BATCH_ID)
    })

customers_pdf = pd.DataFrame(customers)
# Store IDs for use in downstream domain generation
customer_ids  = [c["customer_id"] for c in customers]

customers_count = ingest_domain(customers_pdf, "customers")

# ═══════════════════════════════════════════════════════════
# CELL 6 — Generate and Ingest: Policies
# Purpose : Generate synthetic policy records linked to
#           real customer IDs from Cell 5.
# Records : 15,000 policies
# Output  : insurance_bronze.policies (partitioned by policy_type)
#           insurance_bronze.rejected_policies
# Key fields:
#   policy_id     — unique identifier (POL-XXXXXXXXXX)
#   customer_id   — FK to customers (referential integrity)
#   risk_score    — 0.0 to 1.0, drives risk_band in Silver
#   coverage/premium — ratio used for loss analysis in Gold
# ═══════════════════════════════════════════════════════════

log.info(f"Generating {GENERATION['num_policies']:,} policies...")

policies = []
for _ in range(GENERATION["num_policies"]):
    start = rand_date(GENERATION["start_date"],
                      GENERATION["end_date"])
    end   = start + timedelta(
        days=random.choice([180, 365, 730])
    )
    prem  = rand_amount(200, 5000)
    policies.append({
        "policy_id":            gen_id("POL"),
        "customer_id":          random.choice(customer_ids),
        "policy_type":          random.choice(POLICY_TYPES),
        "status":               random.choice(POLICY_STATUS),
        "start_date":           start.isoformat(),
        "end_date":             end.isoformat(),
        "annual_premium_chf":   prem,
        "coverage_amount_chf":  rand_amount(prem*10, prem*200),
        "deductible_chf":       rand_amount(500, 5000),
        "payment_frequency":    random.choice(
            ["monthly","quarterly","annual"]
        ),
        "currency":             random.choice(CURRENCIES),
        "distribution_channel": random.choice(CHANNELS),
        "underwriter_id":       f"UW-{random.randint(100,999)}",
        "risk_score":           round(random.uniform(0.1,1.0), 3),
        "auto_renewal":         random.random() < 0.65,
        **audit_cols(BATCH_ID)
    })

policies_pdf = pd.DataFrame(policies)
policy_ids   = [p["policy_id"] for p in policies]

policies_count = ingest_domain(policies_pdf, "policies")

# ═══════════════════════════════════════════════════════════
# CELL 7 — Generate and Ingest: Claims
# Purpose : Generate synthetic claim records with realistic
#           fraud injection at 5% rate.
# Records : 8,000 claims
# Output  : insurance_bronze.claims (partitioned by claim_status)
#           insurance_bronze.rejected_claims
# Key fields:
#   claim_id           — unique identifier (CLM-XXXXXXXXXX)
#   policy_id/customer_id — FK to policies and customers
#   is_fraud_suspected — 5% of claims flagged (realistic rate)
#   fraud_indicators   — which fraud pattern triggered
#   days_to_submit     — late submissions are fraud signal
#   settled_amount_chf — None if not yet settled
# ═══════════════════════════════════════════════════════════

log.info(f"Generating {GENERATION['num_claims']:,} claims...")

claims      = []
fraud_count = 0

for _ in range(GENERATION["num_claims"]):
    incident  = rand_date(GENERATION["start_date"],
                          GENERATION["end_date"])
    submitted = incident + timedelta(
        days=random.randint(1, 60)
    )
    is_fraud  = random.random() < GENERATION["fraud_rate"]
    if is_fraud:
        fraud_count += 1
    amt = rand_amount(500, 150_000)

    claims.append({
        "claim_id":             gen_id("CLM"),
        "policy_id":            random.choice(policy_ids),
        "customer_id":          random.choice(customer_ids),
        "incident_date":        incident.isoformat(),
        "submitted_date":       submitted.isoformat(),
        "days_to_submit":       (submitted - incident).days,
        "claim_amount_chf":     amt,
        "settled_amount_chf":   rand_amount(0, amt)
                                if random.random() < 0.6
                                else None,
        "claim_status":         random.choice(CLAIM_STATUSES),
        "claim_type":           random.choice(CLAIM_TYPES),
        "third_party_involved": random.random() < 0.3,
        "police_report_filed":  random.random() < 0.4,
        "handler_id":           f"CH-{random.randint(1000,9999)}",
        "is_fraud_suspected":   is_fraud,
        "fraud_indicators":     random.choice(FRAUD_TYPES)
                                if is_fraud else None,
        "recovery_amount_chf":  rand_amount(0, 5000)
                                if random.random() < 0.1
                                else None,
        **audit_cols(BATCH_ID)
    })

log.info(
    f"Fraud injected: {fraud_count} claims "
    f"({fraud_count/GENERATION['num_claims']*100:.1f}%)"
)

claims_pdf   = pd.DataFrame(claims)
claims_count = ingest_domain(claims_pdf, "claims")

# ═══════════════════════════════════════════════════════════
# CELL 8 — Generate and Ingest: Premiums
# Purpose : Generate synthetic premium payment records.
#           88% paid rate simulates realistic collection health.
# Records : 50,000 payments
# Output  : insurance_bronze.premiums (partitioned by payment_status)
#           insurance_bronze.rejected_premiums
# Key fields:
#   payment_id      — unique identifier (PAY-XXXXXXXXXX)
#   policy_id       — FK to policies
#   payment_status  — paid/pending/failed (88% paid)
#   days_overdue    — 0 for paid, 1-90 for overdue
#   amount_paid_chf — None if not paid yet
# ═══════════════════════════════════════════════════════════

log.info(f"Generating {GENERATION['num_premiums']:,} premiums...")

premiums = []
for _ in range(GENERATION["num_premiums"]):
    due  = rand_date(GENERATION["start_date"],
                     GENERATION["end_date"])
    paid = random.random() < 0.88
    amt  = rand_amount(50, 1500)

    premiums.append({
        "payment_id":         gen_id("PAY"),
        "policy_id":          random.choice(policy_ids),
        "due_date":           due.isoformat(),
        "paid_date":          (due + timedelta(
            days=random.randint(0, 30)
        )).isoformat() if paid else None,
        "amount_due_chf":     amt,
        "amount_paid_chf":    amt if paid else None,
        "payment_status":     "paid" if paid
                              else random.choice(["pending","failed"]),
        "payment_method":     random.choice(PAY_METHODS),
        "days_overdue":       0 if paid
                              else random.randint(1, 90),
        "installment_number": random.randint(1, 12),
        "currency":           "CHF",
        "transaction_ref":    gen_id("TXN"),
        **audit_cols(BATCH_ID)
    })

premiums_pdf   = pd.DataFrame(premiums)
premiums_count = ingest_domain(premiums_pdf, "premiums")

# ═══════════════════════════════════════════════════════════
# CELL 9 — Generate and Ingest: Fraud Signals
# Purpose : Derive fraud signal records from flagged claims.
#           One signal per fraud indicator per claim.
#           In production this would be a streaming pipeline.
#           Here we batch-generate for analytical processing.
# Records : ~400 signals (5% of 8,000 claims)
# Output  : insurance_bronze.fraud_signals
#           insurance_bronze.rejected_fraud_signals
# Key fields:
#   signal_id    — unique identifier (SIG-XXXXXXXXXX)
#   claim_id     — FK to claims (only fraud-suspected claims)
#   signal_score — 0.5-1.0 (higher = more suspicious)
#   signal_type  — which fraud pattern was detected
#   outcome      — confirmed_fraud/false_positive/under_review
# ═══════════════════════════════════════════════════════════

log.info("Generating fraud signals from flagged claims...")

fraud_claims = [c for c in claims if c["is_fraud_suspected"]]
signals      = []

for c in fraud_claims:
    signals.append({
        "signal_id":    gen_id("SIG"),
        "claim_id":     c["claim_id"],
        "customer_id":  c["customer_id"],
        "signal_type":  c["fraud_indicators"],
        "signal_score": round(random.uniform(0.5, 1.0), 3),
        "detected_at":  datetime.utcnow().isoformat(),
        "reviewed":     random.random() < 0.4,
        "reviewer_id":  f"FR-{random.randint(100,999)}"
                        if random.random() < 0.4 else None,
        "outcome":      random.choice([
            "confirmed_fraud","false_positive",
            "under_review", None
        ]),
        **audit_cols(BATCH_ID)
    })

log.info(f"Fraud signals generated: {len(signals):,}")

signals_pdf   = pd.DataFrame(signals)
signals_count = ingest_domain(signals_pdf, "fraud_signals")

# ═══════════════════════════════════════════════════════════
# CELL 10 — Validation Report
# Purpose : Post-ingestion checks — confirms every table
#           has expected row counts, reviews DQ pass rates,
#           checks for rejected records and pipeline errors.
#           This is the GO / NO-GO decision point.
#           Silver layer should NOT run if errors exist here.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("BRONZE LAYER — VALIDATION REPORT")
print(f"{'='*60}")

# ── Good records ────────────────────────────────────────
domains = ["customers","policies","claims",
           "premiums","fraud_signals"]
total_good = 0

print("\n GOOD RECORDS:")
for domain in domains:
    count = spark.table(f"{BRONZE_DB}.{domain}").count()
    total_good += count
    print(f"  {domain:<20} {count:>10,} rows ✅")

print(f"\n  {'TOTAL':<20} {total_good:>10,} rows")

# ── Quarantine records ──────────────────────────────────
print("\n QUARANTINE (REJECTED) RECORDS:")
total_bad = 0
for domain in domains:
    try:
        count = spark.table(
            f"{BRONZE_DB}.rejected_{domain}"
        ).count()
        total_bad += count
        flag = "⚠️ " if count > 0 else "✅"
        print(f"  rejected_{domain:<15} {count:>8,} rows {flag}")
    except Exception as e:
        print(f"  rejected_{domain:<15} NOT FOUND ❌")

# ── DQ monitoring ───────────────────────────────────────
print("\n DQ PASS RATES (THIS BATCH):")
spark.sql(f"""
    SELECT domain,
           good_count,
           bad_count,
           pass_rate_pct,
           date_format(load_timestamp,
               'yyyy-MM-dd HH:mm:ss') AS loaded_at
    FROM {BRONZE_DB}.dq_monitoring
    WHERE batch_id = '{BATCH_ID}'
    ORDER BY domain
""").show()

# ── Pipeline errors ─────────────────────────────────────
error_count = spark.table(
    f"{BRONZE_DB}.pipeline_errors"
).filter(F.col("batch_id") == BATCH_ID).count()

print(f"\n PIPELINE ERRORS THIS BATCH: {error_count}")
if error_count > 0:
    print("  ❌ Errors detected — review before running Silver:")
    spark.sql(f"""
        SELECT domain, error_type,
               left(error_message, 100) AS error_summary,
               failed_at
        FROM {BRONZE_DB}.pipeline_errors
        WHERE batch_id = '{BATCH_ID}'
    """).show(truncate=False)

# ── Final status ────────────────────────────────────────
status = "✅ SUCCESS" if error_count == 0 else "❌ FAILED"
print(f"\n{'='*60}")
print(f"  Batch ID : {BATCH_ID}")
print(f"  Status   : {status}")
print(f"  Total    : {total_good:,} good | {total_bad:,} rejected")
print(f"{'='*60}")

if error_count > 0:
    raise RuntimeError(
        f"Bronze pipeline failed with {error_count} error(s). "
        f"Review pipeline_errors table before proceeding."
    )

# ═══════════════════════════════════════════════════════════
# CELL 11 — Sample Data Spot Checks
# Purpose : Visual verification that data looks realistic.
#           Run these queries to confirm values are sensible.
#           Not automated — engineer reviews output visually.
# ═══════════════════════════════════════════════════════════

print("\nSpot check 1: Claims by status and fraud rate")
spark.sql(f"""
    SELECT claim_status,
           COUNT(*)                        AS total_claims,
           ROUND(AVG(claim_amount_chf), 2) AS avg_amount_chf,
           SUM(CASE WHEN is_fraud_suspected
                    THEN 1 ELSE 0 END)     AS fraud_suspected,
           ROUND(
               SUM(CASE WHEN is_fraud_suspected
                        THEN 1.0 ELSE 0.0 END)
               / COUNT(*) * 100, 2
           )                               AS fraud_rate_pct
    FROM {BRONZE_DB}.claims
    GROUP BY claim_status
    ORDER BY total_claims DESC
""").show()

print("\nSpot check 2: Policy distribution by type")
spark.sql(f"""
    SELECT policy_type,
           COUNT(*)                           AS total_policies,
           ROUND(AVG(annual_premium_chf), 2)  AS avg_premium_chf,
           ROUND(AVG(risk_score), 3)           AS avg_risk_score
    FROM {BRONZE_DB}.policies
    GROUP BY policy_type
    ORDER BY total_policies DESC
""").show()

print("\nSpot check 3: Premium collection health")
spark.sql(f"""
    SELECT payment_status,
           COUNT(*)                       AS total_payments,
           ROUND(SUM(amount_due_chf), 0)  AS total_due_chf
    FROM {BRONZE_DB}.premiums
    GROUP BY payment_status
    ORDER BY total_payments DESC
""").show()

print(f"\n✅ Bronze layer complete — Batch ID: {BATCH_ID}")
