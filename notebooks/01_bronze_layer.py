# Databricks notebook source
# MAGIC %pip install faker pandas pyarrow

# COMMAND ----------

"""
insurance_bronze_complete.py
=============================
Complete Bronze Layer for Insurance Data Platform.
Combines data generation + Delta table ingestion in one script.

Architecture:
    Faker → Pandas DataFrame → Spark DataFrame → Delta Tables

No file storage, no DBFS paths, no mounts needed.
Pure Spark to Delta — production grade.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql import types as T
from datetime import datetime, timedelta, date
import random
import uuid
import hashlib
import logging
import pandas as pd
from faker import Faker

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "seed":              42,
    "num_customers":     10_000,
    "num_policies":      15_000,
    "num_claims":        8_000,
    "num_premiums":      50_000,
    "fraud_rate":        0.05,
    "db_name":           "insurance_bronze",
    "start_date":        date(2019, 1, 1),
    "end_date":          date(2024, 12, 31),
}

BATCH_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("insurance_bronze")

# ─────────────────────────────────────────────
# SEEDING
# ─────────────────────────────────────────────

fake = Faker("en_GB")
Faker.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

# ─────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────

POLICY_TYPES   = ["motor", "home", "life", "health", "travel", "commercial"]
POLICY_STATUS  = ["active", "lapsed", "cancelled", "expired", "pending"]
CLAIM_STATUSES = ["submitted", "under_review", "approved", "rejected", "settled"]
PAY_METHODS    = ["direct_debit", "credit_card", "bank_transfer", "cheque"]
PAY_STATUSES   = ["paid", "pending", "failed"]
CHANNELS       = ["online", "broker", "agent", "direct_call", "mobile_app"]
CURRENCIES     = ["CHF", "EUR"]
CLAIM_TYPES    = ["accident", "theft", "fire", "flood", "liability", "medical", "travel_delay"]
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
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"

def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))

def rand_amount(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)

def audit_cols() -> dict:
    """Standard audit columns for every Bronze record."""
    return {
        "_ingestion_timestamp": datetime.utcnow().isoformat(),
        "_source_system":       "synthetic_generator_v2",
        "_record_hash":         hashlib.md5(str(uuid.uuid4()).encode()).hexdigest(),
        "_batch_id":            BATCH_ID,
    }

def age_band(dob: date) -> str:
    age = (date.today() - dob).days // 365
    if age < 25:  return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    elif age < 65: return "55-64"
    else:          return "65+"

# ─────────────────────────────────────────────
# DATA QUALITY RULES
# ─────────────────────────────────────────────
# Declarative DQ rules per domain.
# Each rule: (sql_expression, error_code)
# Records failing any rule go to rejected table.

DQ_RULES = {
    "customers": [
        ("customer_id IS NOT NULL",      "ERR_NULL_CUSTOMER_ID"),
        ("length(customer_id) > 0",      "ERR_EMPTY_CUSTOMER_ID"),
        ("email IS NOT NULL",            "ERR_NULL_EMAIL"),
        ("email LIKE '%@%'",             "ERR_INVALID_EMAIL"),
        ("country IS NOT NULL",          "ERR_NULL_COUNTRY"),
    ],
    "policies": [
        ("policy_id IS NOT NULL",        "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",      "ERR_NULL_CUSTOMER_ID"),
        ("annual_premium_chf > 0",       "ERR_INVALID_PREMIUM"),
        ("coverage_amount_chf > 0",      "ERR_INVALID_COVERAGE"),
        ("start_date IS NOT NULL",       "ERR_NULL_START_DATE"),
        ("end_date IS NOT NULL",         "ERR_NULL_END_DATE"),
    ],
    "claims": [
        ("claim_id IS NOT NULL",         "ERR_NULL_CLAIM_ID"),
        ("policy_id IS NOT NULL",        "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",      "ERR_NULL_CUSTOMER_ID"),
        ("claim_amount_chf > 0",         "ERR_INVALID_CLAIM_AMOUNT"),
        ("incident_date IS NOT NULL",    "ERR_NULL_INCIDENT_DATE"),
        ("days_to_submit >= 0",          "ERR_NEGATIVE_DAYS"),
    ],
    "premiums": [
        ("payment_id IS NOT NULL",       "ERR_NULL_PAYMENT_ID"),
        ("policy_id IS NOT NULL",        "ERR_NULL_POLICY_ID"),
        ("amount_due_chf > 0",           "ERR_INVALID_AMOUNT_DUE"),
        ("due_date IS NOT NULL",         "ERR_NULL_DUE_DATE"),
    ],
    "fraud_signals": [
        ("signal_id IS NOT NULL",        "ERR_NULL_SIGNAL_ID"),
        ("claim_id IS NOT NULL",         "ERR_NULL_CLAIM_ID"),
        ("signal_score >= 0",            "ERR_INVALID_SCORE"),
        ("signal_score <= 1",            "ERR_SCORE_OUT_OF_RANGE"),
    ],
}

# ─────────────────────────────────────────────
# CORE INGESTION FUNCTIONS
# ─────────────────────────────────────────────

def pandas_to_spark(pdf: pd.DataFrame, domain: str):
    """Convert Pandas DataFrame to Spark DataFrame with audit columns."""
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("_bronze_load_timestamp", F.current_timestamp()) \
             .withColumn("_bronze_batch_id",        F.lit(BATCH_ID)) \
             .withColumn("_bronze_domain",           F.lit(domain))
    return sdf


def apply_dq_rules(sdf, domain: str):
    """
    Split DataFrame into good and rejected records.
    Bad records captured with error codes — never silently dropped.
    Production rule: every rejected record must be traceable.
    """
    rules = DQ_RULES.get(domain, [])
    if not rules:
        return sdf, None

    # Tag each record with triggered error codes
    sdf_tagged = sdf.withColumn(
        "_dq_errors",
        F.concat_ws("|", *[
            F.when(F.expr(f"NOT ({rule})"), F.lit(code))
             .otherwise(F.lit(""))
            for rule, code in rules
        ])
    )

    good_sdf = sdf_tagged.filter(F.col("_dq_errors") == "").drop("_dq_errors")
    bad_sdf  = sdf_tagged.filter(F.col("_dq_errors") != "") \
                         .withColumn("_rejected_at",    F.current_timestamp()) \
                         .withColumn("_rejected_batch", F.lit(BATCH_ID))

    good_count = good_sdf.count()
    bad_count  = bad_sdf.count()
    total      = good_count + bad_count
    pass_rate  = (good_count / total * 100) if total > 0 else 0

    log.info(f"  DQ [{domain}] → Good: {good_count:,} | Rejected: {bad_count:,} | Pass Rate: {pass_rate:.2f}%")

    return good_sdf, bad_sdf


def write_delta(sdf, table: str, partition_cols: list = None, mode: str = "overwrite"):
    """Write Spark DataFrame to Delta table."""
    full_table = f"{CONFIG['db_name']}.{table}"
    writer = sdf.write.format("delta").mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table)
    count = spark.table(full_table).count()
    log.info(f"  ✅ {full_table} → {count:,} rows written")
    return count


def write_rejected(sdf, domain: str):
    """Write rejected records to quarantine table."""
    if sdf is None:
        return
    count = sdf.count()
    if count == 0:
        log.info(f"  ✅ No rejected records for {domain}")
        return
    full_table = f"{CONFIG['db_name']}.rejected_{domain}"
    sdf.write.format("delta").mode("overwrite").saveAsTable(full_table)
    log.info(f"  ⚠️  {count:,} rejected records → {full_table}")


def ingest_domain(pdf: pd.DataFrame, domain: str, partition_cols: list = None):
    """Full ingestion pipeline for one domain."""
    print(f"\n{'='*50}")
    print(f"INGESTING: {domain.upper()}")
    print(f"{'='*50}")

    sdf          = pandas_to_spark(pdf, domain)
    good, bad    = apply_dq_rules(sdf, domain)
    count        = write_delta(good, domain, partition_cols)
    write_rejected(bad, domain)

    return count

# ─────────────────────────────────────────────
# SETUP DATABASE
# ─────────────────────────────────────────────

print(f"\nBatch ID: {BATCH_ID}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {CONFIG['db_name']}")
spark.sql(f"USE {CONFIG['db_name']}")
print(f"✅ Database '{CONFIG['db_name']}' ready")

# ─────────────────────────────────────────────
# GENERATE & INGEST CUSTOMERS
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_customers']:,} customers...")
customers = []
for _ in range(CONFIG["num_customers"]):
    dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
    customers.append({
        "customer_id":    gen_id("CUST"),
        "first_name":     fake.first_name(),
        "last_name":      fake.last_name(),
        "date_of_birth":  dob.isoformat(),
        "age_band":       age_band(dob),
        "gender":         random.choice(["M", "F", "Other"]),
        "email":          fake.email(),
        "phone":          fake.phone_number(),
        "address_line1":  fake.street_address(),
        "city":           fake.city(),
        "postcode":       fake.postcode(),
        "country":        "CH",
        "customer_since": rand_date(date(2005,1,1), date(2023,12,31)).isoformat(),
        "channel":        random.choice(CHANNELS),
        "is_high_value":  random.random() < 0.10,
        **audit_cols()
    })
customers_pdf = pd.DataFrame(customers)
customer_ids  = [c["customer_id"] for c in customers]
ingest_domain(customers_pdf, "customers")

# ─────────────────────────────────────────────
# GENERATE & INGEST POLICIES
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_policies']:,} policies...")
policies = []
for _ in range(CONFIG["num_policies"]):
    start = rand_date(CONFIG["start_date"], CONFIG["end_date"])
    end   = start + timedelta(days=random.choice([180, 365, 730]))
    prem  = rand_amount(200, 5000)
    policies.append({
        "policy_id":           gen_id("POL"),
        "customer_id":         random.choice(customer_ids),
        "policy_type":         random.choice(POLICY_TYPES),
        "status":              random.choice(POLICY_STATUS),
        "start_date":          start.isoformat(),
        "end_date":            end.isoformat(),
        "annual_premium_chf":  prem,
        "coverage_amount_chf": rand_amount(prem * 10, prem * 200),
        "deductible_chf":      rand_amount(500, 5000),
        "payment_frequency":   random.choice(["monthly", "quarterly", "annual"]),
        "currency":            random.choice(CURRENCIES),
        "distribution_channel":random.choice(CHANNELS),
        "underwriter_id":      f"UW-{random.randint(100,999)}",
        "risk_score":          round(random.uniform(0.1, 1.0), 3),
        "auto_renewal":        random.random() < 0.65,
        **audit_cols()
    })
policies_pdf = pd.DataFrame(policies)
policy_ids   = [p["policy_id"] for p in policies]
ingest_domain(policies_pdf, "policies", partition_cols=["policy_type"])

# ─────────────────────────────────────────────
# GENERATE & INGEST CLAIMS
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_claims']:,} claims...")
claims      = []
fraud_count = 0
for _ in range(CONFIG["num_claims"]):
    incident  = rand_date(CONFIG["start_date"], CONFIG["end_date"])
    submitted = incident + timedelta(days=random.randint(1, 60))
    is_fraud  = random.random() < CONFIG["fraud_rate"]
    if is_fraud: fraud_count += 1
    amt = rand_amount(500, 150_000)
    claims.append({
        "claim_id":             gen_id("CLM"),
        "policy_id":            random.choice(policy_ids),
        "customer_id":          random.choice(customer_ids),
        "incident_date":        incident.isoformat(),
        "submitted_date":       submitted.isoformat(),
        "days_to_submit":       (submitted - incident).days,
        "claim_amount_chf":     amt,
        "settled_amount_chf":   rand_amount(0, amt) if random.random() < 0.6 else None,
        "claim_status":         random.choice(CLAIM_STATUSES),
        "claim_type":           random.choice(CLAIM_TYPES),
        "third_party_involved": random.random() < 0.3,
        "police_report_filed":  random.random() < 0.4,
        "handler_id":           f"CH-{random.randint(1000,9999)}",
        "is_fraud_suspected":   is_fraud,
        "fraud_indicators":     random.choice(FRAUD_TYPES) if is_fraud else None,
        "recovery_amount_chf":  rand_amount(0, 5000) if random.random() < 0.1 else None,
        **audit_cols()
    })
claims_pdf = pd.DataFrame(claims)
log.info(f"  Fraud injected: {fraud_count} ({fraud_count/CONFIG['num_claims']*100:.1f}%)")
ingest_domain(claims_pdf, "claims", partition_cols=["claim_status"])

# ─────────────────────────────────────────────
# GENERATE & INGEST PREMIUMS
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_premiums']:,} premium payments...")
premiums = []
for _ in range(CONFIG["num_premiums"]):
    due  = rand_date(CONFIG["start_date"], CONFIG["end_date"])
    paid = random.random() < 0.88
    amt  = rand_amount(50, 1500)
    premiums.append({
        "payment_id":       gen_id("PAY"),
        "policy_id":        random.choice(policy_ids),
        "due_date":         due.isoformat(),
        "paid_date":        (due + timedelta(days=random.randint(0,30))).isoformat() if paid else None,
        "amount_due_chf":   amt,
        "amount_paid_chf":  amt if paid else None,
        "payment_status":   "paid" if paid else random.choice(["pending", "failed"]),
        "payment_method":   random.choice(PAY_METHODS),
        "days_overdue":     0 if paid else random.randint(1, 90),
        "installment_number": random.randint(1, 12),
        "currency":         "CHF",
        "transaction_ref":  gen_id("TXN"),
        **audit_cols()
    })
premiums_pdf = pd.DataFrame(premiums)
ingest_domain(premiums_pdf, "premiums", partition_cols=["payment_status"])

# ─────────────────────────────────────────────
# GENERATE & INGEST FRAUD SIGNALS
# ─────────────────────────────────────────────

log.info("Generating fraud signals...")
fraud_claims = [c for c in claims if c["is_fraud_suspected"]]
signals = []
for c in fraud_claims:
    signals.append({
        "signal_id":    gen_id("SIG"),
        "claim_id":     c["claim_id"],
        "customer_id":  c["customer_id"],
        "signal_type":  c["fraud_indicators"],
        "signal_score": round(random.uniform(0.5, 1.0), 3),
        "detected_at":  datetime.utcnow().isoformat(),
        "reviewed":     random.random() < 0.4,
        "reviewer_id":  f"FR-{random.randint(100,999)}" if random.random() < 0.4 else None,
        "outcome":      random.choice(["confirmed_fraud","false_positive","under_review", None]),
        **audit_cols()
    })
signals_pdf = pd.DataFrame(signals)
ingest_domain(signals_pdf, "fraud_signals", partition_cols=["signal_type"])

# ─────────────────────────────────────────────
# FINAL VALIDATION REPORT
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("BRONZE LAYER — FINAL VALIDATION REPORT")
print(f"{'='*50}")

domains      = ["customers", "policies", "claims", "premiums", "fraud_signals"]
total        = 0
all_tables   = [t.name for t in spark.catalog.listTables(CONFIG["db_name"])]

for domain in domains:
    count = spark.table(f"{CONFIG['db_name']}.{domain}").count()
    total += count
    print(f"  {domain:<25} {count:>10,} rows ✅")

print(f"\n  {'TOTAL':<25} {total:>10,} rows")
print(f"  Batch ID: {BATCH_ID}")

# Show rejected tables if any
rejected = [t for t in all_tables if t.startswith("rejected_")]
if rejected:
    print(f"\n  Quarantine tables:")
    for t in rejected:
        count = spark.table(f"{CONFIG['db_name']}.{t}").count()
        print(f"  {t:<25} {count:>10,} rows ⚠️")
else:
    print(f"\n  ✅ No rejected records across all domains")

print(f"{'='*50}")

# ─────────────────────────────────────────────
# SAMPLE VERIFICATION QUERIES
# ─────────────────────────────────────────────

print("\nSample: Claims by status")
spark.sql("""
    SELECT claim_status,
           COUNT(*)                        AS total_claims,
           ROUND(AVG(claim_amount_chf), 2) AS avg_claim_chf,
           SUM(CASE WHEN is_fraud_suspected THEN 1 ELSE 0 END) AS fraud_count
    FROM   insurance_bronze.claims
    GROUP  BY claim_status
    ORDER  BY total_claims DESC
""").show()

print("\nSample: Policy distribution")
spark.sql("""
    SELECT policy_type,
           COUNT(*)                            AS total_policies,
           ROUND(AVG(annual_premium_chf), 2)   AS avg_premium_chf,
           ROUND(AVG(risk_score), 3)            AS avg_risk_score
    FROM   insurance_bronze.policies
    GROUP  BY policy_type
    ORDER  BY total_policies DESC
""").show()

print("\nSample: Premium payment health")
spark.sql("""
    SELECT payment_status,
           COUNT(*)                          AS total_payments,
           ROUND(SUM(amount_due_chf), 2)     AS total_due_chf,
           ROUND(SUM(amount_paid_chf), 2)    AS total_paid_chf
    FROM   insurance_bronze.premiums
    GROUP  BY payment_status
    ORDER  BY total_payments DESC
""").show()

print("\nSample: Fraud signal breakdown")
spark.sql("""
    SELECT signal_type,
           COUNT(*)                       AS signal_count,
           ROUND(AVG(signal_score), 3)    AS avg_score,
           SUM(CASE WHEN reviewed THEN 1 ELSE 0 END) AS reviewed_count
    FROM   insurance_bronze.fraud_signals
    GROUP  BY signal_type
    ORDER  BY signal_count DESC
""").show(truncate=False)

print(f"\n✅ Bronze layer complete — Batch ID: {BATCH_ID}")

# COMMAND ----------

spark.table("insurance_bronze.rejected_customers") \
     .select("customer_id", "email", "country", "_dq_errors") \
     .show(5, truncate=False)

# COMMAND ----------



# COMMAND ----------

"""
insurance_bronze_complete.py
=============================
Complete Bronze Layer for Insurance Data Platform.
Combines data generation + Delta table ingestion in one script.

Architecture:
    Faker → Pandas DataFrame → Spark DataFrame → Delta Tables

No file storage, no DBFS paths, no mounts needed.
Pure Spark to Delta — production grade.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql import types as T
from datetime import datetime, timedelta, date
import random
import uuid
import hashlib
import logging
import pandas as pd
from faker import Faker

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CONFIG = {
    "seed":              42,
    "num_customers":     10_000,
    "num_policies":      15_000,
    "num_claims":        8_000,
    "num_premiums":      50_000,
    "fraud_rate":        0.05,
    "db_name":           "insurance_bronze",
    "start_date":        date(2019, 1, 1),
    "end_date":          date(2024, 12, 31),
}

BATCH_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("insurance_bronze")

# ─────────────────────────────────────────────
# SEEDING
# ─────────────────────────────────────────────

fake = Faker("en_GB")
Faker.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

# ─────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────

POLICY_TYPES   = ["motor", "home", "life", "health", "travel", "commercial"]
POLICY_STATUS  = ["active", "lapsed", "cancelled", "expired", "pending"]
CLAIM_STATUSES = ["submitted", "under_review", "approved", "rejected", "settled"]
PAY_METHODS    = ["direct_debit", "credit_card", "bank_transfer", "cheque"]
PAY_STATUSES   = ["paid", "pending", "failed"]
CHANNELS       = ["online", "broker", "agent", "direct_call", "mobile_app"]
CURRENCIES     = ["CHF", "EUR"]
CLAIM_TYPES    = ["accident", "theft", "fire", "flood", "liability", "medical", "travel_delay"]
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
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"

def rand_date(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))

def rand_amount(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)

def audit_cols() -> dict:
    """Standard audit columns for every Bronze record."""
    return {
        "_ingestion_timestamp": datetime.utcnow().isoformat(),
        "_source_system":       "synthetic_generator_v2",
        "_record_hash":         hashlib.md5(str(uuid.uuid4()).encode()).hexdigest(),
        "_batch_id":            BATCH_ID,
    }

def age_band(dob: date) -> str:
    age = (date.today() - dob).days // 365
    if age < 25:  return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    elif age < 65: return "55-64"
    else:          return "65+"

# ─────────────────────────────────────────────
# DATA QUALITY RULES
# ─────────────────────────────────────────────
# Declarative DQ rules per domain.
# Each rule: (sql_expression, error_code)
# Records failing any rule go to rejected table.

DQ_RULES = {
    "customers": [
        ("customer_id IS NOT NULL",      "ERR_NULL_CUSTOMER_ID"),
        ("length(customer_id) > 0",      "ERR_EMPTY_CUSTOMER_ID"),
        ("email IS NOT NULL",            "ERR_NULL_EMAIL"),
        ("email LIKE '%@%'",             "ERR_INVALID_EMAIL"),
        ("country IS NOT NULL",          "ERR_NULL_COUNTRY"),
    ],
    "policies": [
        ("policy_id IS NOT NULL",        "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",      "ERR_NULL_CUSTOMER_ID"),
        ("annual_premium_chf > 0",       "ERR_INVALID_PREMIUM"),
        ("coverage_amount_chf > 0",      "ERR_INVALID_COVERAGE"),
        ("start_date IS NOT NULL",       "ERR_NULL_START_DATE"),
        ("end_date IS NOT NULL",         "ERR_NULL_END_DATE"),
    ],
    "claims": [
        ("claim_id IS NOT NULL",         "ERR_NULL_CLAIM_ID"),
        ("policy_id IS NOT NULL",        "ERR_NULL_POLICY_ID"),
        ("customer_id IS NOT NULL",      "ERR_NULL_CUSTOMER_ID"),
        ("claim_amount_chf > 0",         "ERR_INVALID_CLAIM_AMOUNT"),
        ("incident_date IS NOT NULL",    "ERR_NULL_INCIDENT_DATE"),
        ("days_to_submit >= 0",          "ERR_NEGATIVE_DAYS"),
    ],
    "premiums": [
        ("payment_id IS NOT NULL",       "ERR_NULL_PAYMENT_ID"),
        ("policy_id IS NOT NULL",        "ERR_NULL_POLICY_ID"),
        ("amount_due_chf > 0",           "ERR_INVALID_AMOUNT_DUE"),
        ("due_date IS NOT NULL",         "ERR_NULL_DUE_DATE"),
    ],
    "fraud_signals": [
        ("signal_id IS NOT NULL",        "ERR_NULL_SIGNAL_ID"),
        ("claim_id IS NOT NULL",         "ERR_NULL_CLAIM_ID"),
        ("signal_score >= 0",            "ERR_INVALID_SCORE"),
        ("signal_score <= 1",            "ERR_SCORE_OUT_OF_RANGE"),
    ],
}

# ─────────────────────────────────────────────
# CORE INGESTION FUNCTIONS
# ─────────────────────────────────────────────

def pandas_to_spark(pdf: pd.DataFrame, domain: str):
    """Convert Pandas DataFrame to Spark DataFrame with audit columns."""
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("_bronze_load_timestamp", F.current_timestamp()) \
             .withColumn("_bronze_batch_id",        F.lit(BATCH_ID)) \
             .withColumn("_bronze_domain",           F.lit(domain))
    return sdf

def apply_dq_rules(sdf, domain: str):
    rules = DQ_RULES.get(domain, [])
    if not rules:
        return sdf, None

    # Key fix: use NULL instead of empty string
    # concat_ws ignores NULLs — so passing records produce ""
    sdf_tagged = sdf.withColumn(
        "_dq_errors",
        F.concat_ws("|", *[
            F.when(F.expr(f"NOT ({rule})"), F.lit(code))
             .otherwise(F.lit(None).cast("string"))  # ← NULL not ""
            for rule, code in rules
        ])
    )

    good_sdf = sdf_tagged.filter(
        (F.col("_dq_errors").isNull()) | 
        (F.col("_dq_errors") == "")
    ).drop("_dq_errors")

    bad_sdf = sdf_tagged.filter(
        F.col("_dq_errors").isNotNull() & 
        (F.col("_dq_errors") != "")
    ).withColumn("_rejected_at",    F.current_timestamp()) \
     .withColumn("_rejected_batch", F.lit(BATCH_ID))

    good_count = good_sdf.count()
    bad_count  = bad_sdf.count()
    total      = good_count + bad_count
    pass_rate  = (good_count / total * 100) if total > 0 else 0

    log.info(f"  DQ [{domain}] → Good: {good_count:,} | Rejected: {bad_count:,} | Pass Rate: {pass_rate:.2f}%")

    return good_sdf, bad_sdf


def write_delta(sdf, table: str, partition_cols: list = None, mode: str = "overwrite"):
    """Write Spark DataFrame to Delta table."""
    full_table = f"{CONFIG['db_name']}.{table}"
    writer = sdf.write.format("delta").mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table)
    count = spark.table(full_table).count()
    log.info(f"  ✅ {full_table} → {count:,} rows written")
    return count


def write_rejected(sdf, domain: str):
    """Write rejected records to quarantine table."""
    if sdf is None:
        return
    count = sdf.count()
    if count == 0:
        log.info(f"  ✅ No rejected records for {domain}")
        return
    full_table = f"{CONFIG['db_name']}.rejected_{domain}"
    sdf.write.format("delta").mode("overwrite").saveAsTable(full_table)
    log.info(f"  ⚠️  {count:,} rejected records → {full_table}")


def ingest_domain(pdf: pd.DataFrame, domain: str, partition_cols: list = None):
    """Full ingestion pipeline for one domain."""
    print(f"\n{'='*50}")
    print(f"INGESTING: {domain.upper()}")
    print(f"{'='*50}")

    sdf          = pandas_to_spark(pdf, domain)
    good, bad    = apply_dq_rules(sdf, domain)
    count        = write_delta(good, domain, partition_cols)
    write_rejected(bad, domain)

    return count

# ─────────────────────────────────────────────
# SETUP DATABASE
# ─────────────────────────────────────────────

print(f"\nBatch ID: {BATCH_ID}")
spark.sql("DROP DATABASE IF EXISTS insurance_bronze CASCADE")
spark.sql("CREATE DATABASE IF NOT EXISTS insurance_bronze")
spark.sql("USE insurance_bronze")
print(f"✅ Database '{CONFIG['db_name']}' ready")

# ─────────────────────────────────────────────
# GENERATE & INGEST CUSTOMERS
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_customers']:,} customers...")
customers = []
for _ in range(CONFIG["num_customers"]):
    dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
    customers.append({
        "customer_id":    gen_id("CUST"),
        "first_name":     fake.first_name(),
        "last_name":      fake.last_name(),
        "date_of_birth":  dob.isoformat(),
        "age_band":       age_band(dob),
        "gender":         random.choice(["M", "F", "Other"]),
        "email":          fake.email(),
        "phone":          fake.phone_number(),
        "address_line1":  fake.street_address(),
        "city":           fake.city(),
        "postcode":       fake.postcode(),
        "country":        "CH",
        "customer_since": rand_date(date(2005,1,1), date(2023,12,31)).isoformat(),
        "channel":        random.choice(CHANNELS),
        "is_high_value":  random.random() < 0.10,
        **audit_cols()
    })
customers_pdf = pd.DataFrame(customers)
customer_ids  = [c["customer_id"] for c in customers]
ingest_domain(customers_pdf, "customers")

# ─────────────────────────────────────────────
# GENERATE & INGEST POLICIES
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_policies']:,} policies...")
policies = []
for _ in range(CONFIG["num_policies"]):
    start = rand_date(CONFIG["start_date"], CONFIG["end_date"])
    end   = start + timedelta(days=random.choice([180, 365, 730]))
    prem  = rand_amount(200, 5000)
    policies.append({
        "policy_id":           gen_id("POL"),
        "customer_id":         random.choice(customer_ids),
        "policy_type":         random.choice(POLICY_TYPES),
        "status":              random.choice(POLICY_STATUS),
        "start_date":          start.isoformat(),
        "end_date":            end.isoformat(),
        "annual_premium_chf":  prem,
        "coverage_amount_chf": rand_amount(prem * 10, prem * 200),
        "deductible_chf":      rand_amount(500, 5000),
        "payment_frequency":   random.choice(["monthly", "quarterly", "annual"]),
        "currency":            random.choice(CURRENCIES),
        "distribution_channel":random.choice(CHANNELS),
        "underwriter_id":      f"UW-{random.randint(100,999)}",
        "risk_score":          round(random.uniform(0.1, 1.0), 3),
        "auto_renewal":        random.random() < 0.65,
        **audit_cols()
    })
policies_pdf = pd.DataFrame(policies)
policy_ids   = [p["policy_id"] for p in policies]
ingest_domain(policies_pdf, "policies", partition_cols=["policy_type"])

# ─────────────────────────────────────────────
# GENERATE & INGEST CLAIMS
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_claims']:,} claims...")
claims      = []
fraud_count = 0
for _ in range(CONFIG["num_claims"]):
    incident  = rand_date(CONFIG["start_date"], CONFIG["end_date"])
    submitted = incident + timedelta(days=random.randint(1, 60))
    is_fraud  = random.random() < CONFIG["fraud_rate"]
    if is_fraud: fraud_count += 1
    amt = rand_amount(500, 150_000)
    claims.append({
        "claim_id":             gen_id("CLM"),
        "policy_id":            random.choice(policy_ids),
        "customer_id":          random.choice(customer_ids),
        "incident_date":        incident.isoformat(),
        "submitted_date":       submitted.isoformat(),
        "days_to_submit":       (submitted - incident).days,
        "claim_amount_chf":     amt,
        "settled_amount_chf":   rand_amount(0, amt) if random.random() < 0.6 else None,
        "claim_status":         random.choice(CLAIM_STATUSES),
        "claim_type":           random.choice(CLAIM_TYPES),
        "third_party_involved": random.random() < 0.3,
        "police_report_filed":  random.random() < 0.4,
        "handler_id":           f"CH-{random.randint(1000,9999)}",
        "is_fraud_suspected":   is_fraud,
        "fraud_indicators":     random.choice(FRAUD_TYPES) if is_fraud else None,
        "recovery_amount_chf":  rand_amount(0, 5000) if random.random() < 0.1 else None,
        **audit_cols()
    })
claims_pdf = pd.DataFrame(claims)
log.info(f"  Fraud injected: {fraud_count} ({fraud_count/CONFIG['num_claims']*100:.1f}%)")
ingest_domain(claims_pdf, "claims", partition_cols=["claim_status"])

# ─────────────────────────────────────────────
# GENERATE & INGEST PREMIUMS
# ─────────────────────────────────────────────

log.info(f"Generating {CONFIG['num_premiums']:,} premium payments...")
premiums = []
for _ in range(CONFIG["num_premiums"]):
    due  = rand_date(CONFIG["start_date"], CONFIG["end_date"])
    paid = random.random() < 0.88
    amt  = rand_amount(50, 1500)
    premiums.append({
        "payment_id":       gen_id("PAY"),
        "policy_id":        random.choice(policy_ids),
        "due_date":         due.isoformat(),
        "paid_date":        (due + timedelta(days=random.randint(0,30))).isoformat() if paid else None,
        "amount_due_chf":   amt,
        "amount_paid_chf":  amt if paid else None,
        "payment_status":   "paid" if paid else random.choice(["pending", "failed"]),
        "payment_method":   random.choice(PAY_METHODS),
        "days_overdue":     0 if paid else random.randint(1, 90),
        "installment_number": random.randint(1, 12),
        "currency":         "CHF",
        "transaction_ref":  gen_id("TXN"),
        **audit_cols()
    })
premiums_pdf = pd.DataFrame(premiums)
ingest_domain(premiums_pdf, "premiums", partition_cols=["payment_status"])

# ─────────────────────────────────────────────
# GENERATE & INGEST FRAUD SIGNALS
# ─────────────────────────────────────────────

log.info("Generating fraud signals...")
fraud_claims = [c for c in claims if c["is_fraud_suspected"]]
signals = []
for c in fraud_claims:
    signals.append({
        "signal_id":    gen_id("SIG"),
        "claim_id":     c["claim_id"],
        "customer_id":  c["customer_id"],
        "signal_type":  c["fraud_indicators"],
        "signal_score": round(random.uniform(0.5, 1.0), 3),
        "detected_at":  datetime.utcnow().isoformat(),
        "reviewed":     random.random() < 0.4,
        "reviewer_id":  f"FR-{random.randint(100,999)}" if random.random() < 0.4 else None,
        "outcome":      random.choice(["confirmed_fraud","false_positive","under_review", None]),
        **audit_cols()
    })
signals_pdf = pd.DataFrame(signals)
ingest_domain(signals_pdf, "fraud_signals", partition_cols=["signal_type"])

# ─────────────────────────────────────────────
# FINAL VALIDATION REPORT
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print("BRONZE LAYER — FINAL VALIDATION REPORT")
print(f"{'='*50}")

domains      = ["customers", "policies", "claims", "premiums", "fraud_signals"]
total        = 0
all_tables   = [t.name for t in spark.catalog.listTables(CONFIG["db_name"])]

for domain in domains:
    count = spark.table(f"{CONFIG['db_name']}.{domain}").count()
    total += count
    print(f"  {domain:<25} {count:>10,} rows ✅")

print(f"\n  {'TOTAL':<25} {total:>10,} rows")
print(f"  Batch ID: {BATCH_ID}")

# Show rejected tables if any
rejected = [t for t in all_tables if t.startswith("rejected_")]
if rejected:
    print(f"\n  Quarantine tables:")
    for t in rejected:
        count = spark.table(f"{CONFIG['db_name']}.{t}").count()
        print(f"  {t:<25} {count:>10,} rows ⚠️")
else:
    print(f"\n  ✅ No rejected records across all domains")

print(f"{'='*50}")

# ─────────────────────────────────────────────
# SAMPLE VERIFICATION QUERIES
# ─────────────────────────────────────────────

print("\nSample: Claims by status")
spark.sql("""
    SELECT claim_status,
           COUNT(*)                        AS total_claims,
           ROUND(AVG(claim_amount_chf), 2) AS avg_claim_chf,
           SUM(CASE WHEN is_fraud_suspected THEN 1 ELSE 0 END) AS fraud_count
    FROM   insurance_bronze.claims
    GROUP  BY claim_status
    ORDER  BY total_claims DESC
""").show()

print("\nSample: Policy distribution")
spark.sql("""
    SELECT policy_type,
           COUNT(*)                            AS total_policies,
           ROUND(AVG(annual_premium_chf), 2)   AS avg_premium_chf,
           ROUND(AVG(risk_score), 3)            AS avg_risk_score
    FROM   insurance_bronze.policies
    GROUP  BY policy_type
    ORDER  BY total_policies DESC
""").show()

print("\nSample: Premium payment health")
spark.sql("""
    SELECT payment_status,
           COUNT(*)                          AS total_payments,
           ROUND(SUM(amount_due_chf), 2)     AS total_due_chf,
           ROUND(SUM(amount_paid_chf), 2)    AS total_paid_chf
    FROM   insurance_bronze.premiums
    GROUP  BY payment_status
    ORDER  BY total_payments DESC
""").show()

print("\nSample: Fraud signal breakdown")
spark.sql("""
    SELECT signal_type,
           COUNT(*)                       AS signal_count,
           ROUND(AVG(signal_score), 3)    AS avg_score,
           SUM(CASE WHEN reviewed THEN 1 ELSE 0 END) AS reviewed_count
    FROM   insurance_bronze.fraud_signals
    GROUP  BY signal_type
    ORDER  BY signal_count DESC
""").show(truncate=False)

print(f"\n✅ Bronze layer complete — Batch ID: {BATCH_ID}")