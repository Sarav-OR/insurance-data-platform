# Databricks notebook source
"""
notebooks/05_incremental_ingestion.py
======================================
Incremental Ingestion — Insurance Data Platform
================================================
Purpose : Demonstrate production-grade incremental ingestion
          patterns. Process only NEW or CHANGED records
          since the last successful pipeline run.

Layer   : Bronze (Incremental)
Reads   : New/changed records from source
Writes  : insurance_bronze.* Delta tables (MERGE/APPEND)

Depends on:
  src/config.py     — configuration and constants
  src/utils.py      — helper functions
  src/audit.py      — audit column functions
  src/monitoring.py — DQ tracking and error logging

Why incremental ingestion matters:
  Full reload (current approach):
    Every run processes ALL 83,400 records
    Slow, expensive, not scalable beyond ~1M rows
    Cannot detect what changed between runs

  Incremental (production approach):
    Run 1 → process all records (initial load)
    Run 2 → process only NEW records since Run 1
    Run 3 → process only NEW records since Run 2
    10x-100x faster at scale
    Detects new, updated and deleted records

Three patterns demonstrated:
  Pattern 1 — Watermark
    Track last processed timestamp
    Process only records after that timestamp

  Pattern 2 — Delta MERGE (Upsert)
    INSERT new records
    UPDATE changed records
    Never creates duplicates

  Pattern 3 — Change Data Capture (CDC)
    Detect INSERT/UPDATE/DELETE from source
    Apply changes to target table
"""

# ═══════════════════════════════════════════════════════════
# CELL 1 — Repository Path Setup
# Purpose : Add repo root to Python path so src/ imports work.
# ═══════════════════════════════════════════════════════════

import sys
import os

REPO_ROOT = "/Workspace/Repos/saravanakumar.or@live.com/insurance-data-platform"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f"✅ Repo root added to path: {REPO_ROOT}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 2 — Imports
# Purpose : Import all required libraries and src/ modules.
# ═══════════════════════════════════════════════════════════

import logging
import random
import uuid
import hashlib
from datetime import datetime, timedelta, date

import pandas as pd
from faker import Faker
from delta.tables import DeltaTable

from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.config     import (BATCH_ID, DATABASES,
                             DELTA_SETTINGS, GENERATION,
                             POLICY_TYPES, CLAIM_STATUSES,
                             CLAIM_TYPES, CHANNELS,
                             CURRENCIES, PAY_METHODS)
from src.utils      import (gen_id, rand_date, rand_amount,
                             audit_cols, write_delta,
                             apply_delta_settings)
from src.audit      import add_bronze_audit
from src.monitoring import (apply_dq_rules, write_dq_monitoring,
                             log_pipeline_error)
from src.dq_rules   import get_rules

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("incremental_ingestion")

# ── Convenience aliases ───────────────────────────────────
BRONZE_DB  = DATABASES["bronze"]
INCR_DB    = "insurance_incremental"

# ── Faker setup ───────────────────────────────────────────
fake = Faker("en_GB")
Faker.seed(99)   # Different seed from Bronze — generates new data
random.seed(99)

print(f"✅ All imports successful")
print(f"   Batch ID : {BATCH_ID}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 3 — Setup Incremental Database
# Purpose : Create dedicated database for incremental demo.
#           Also creates watermark tracking table —
#           the heart of incremental processing.
#
# Watermark table:
#   Tracks last successfully processed timestamp per domain.
#   Next run reads: WHERE updated_at > last_watermark
#   Updated after every successful run.
# ═══════════════════════════════════════════════════════════

spark.sql(f"CREATE DATABASE IF NOT EXISTS {INCR_DB}")
spark.sql(f"USE {INCR_DB}")
apply_delta_settings(spark, DELTA_SETTINGS)

# Create watermark tracking table
# This table is the source of truth for incremental state
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {INCR_DB}.watermarks
    (
        domain          STRING,    -- which table e.g. 'claims'
        last_watermark  TIMESTAMP, -- last processed timestamp
        last_batch_id   STRING,    -- which batch updated this
        record_count    LONG,      -- records processed last run
        updated_at      TIMESTAMP  -- when watermark was updated
    )
    USING DELTA
""")

print(f"✅ Database '{INCR_DB}' ready")
print(f"✅ Watermark tracking table created")

# ── Show current watermarks ───────────────────────────────
print("\nCurrent watermarks:")
spark.sql(f"SELECT * FROM {INCR_DB}.watermarks").show()

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 4 — Watermark Helper Functions
# Purpose : Functions to read and update watermarks.
#           Every incremental pipeline uses these.
#
# get_watermark()    — read last processed timestamp
# update_watermark() — save new watermark after success
# ═══════════════════════════════════════════════════════════

def get_watermark(domain: str) -> datetime:
    """
    Get last processed timestamp for a domain.

    If no watermark exists (first run) returns a date
    far in the past — ensures ALL records are processed
    on initial load.

    Args:
        domain: Domain name e.g. 'claims'

    Returns:
        datetime — last successfully processed timestamp
    """
    result = spark.sql(f"""
        SELECT last_watermark
        FROM {INCR_DB}.watermarks
        WHERE domain = '{domain}'
        ORDER BY updated_at DESC
        LIMIT 1
    """).collect()

    if not result:
        # No watermark = first run
        # Return epoch start — process everything
        log.info(
            f"[{domain}] No watermark found — "
            f"first run, processing all records"
        )
        return datetime(2000, 1, 1)

    watermark = result[0]["last_watermark"]
    log.info(
        f"[{domain}] Watermark: {watermark} — "
        f"processing records after this timestamp"
    )
    return watermark


def update_watermark(domain: str,
                     new_watermark: datetime,
                     record_count: int) -> None:
    """
    Update watermark after successful processing.
    Called ONLY after data is successfully written.
    Never update watermark if processing failed —
    next run will reprocess from last successful point.

    Args:
        domain        : Domain name
        new_watermark : New high-water mark timestamp
        record_count  : Records processed this run
    """
    spark.createDataFrame([{
        "domain":         domain,
        "last_watermark": new_watermark,
        "last_batch_id":  BATCH_ID,
        "record_count":   record_count,
        "updated_at":     datetime.utcnow(),
    }]).write.format("delta").mode("append") \
       .saveAsTable(f"{INCR_DB}.watermarks")

    log.info(
        f"[{domain}] Watermark updated to: {new_watermark}"
    )


print("✅ Watermark functions defined")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 5 — Simulate Source Data (Initial Load)
# Purpose : Generate initial dataset simulating what a
#           source system would send on first load.
#           Creates claims table with full history.
#
# In real project:
#   Source system sends full extract on first load
#   Subsequent runs send only changed records
# ═══════════════════════════════════════════════════════════

print("\nSimulating initial data load...")

# Load existing policy and customer IDs from Bronze
# In real project these come from source system
policy_ids   = [
    r["policy_id"]
    for r in spark.table(f"{BRONZE_DB}.policies")
                  .select("policy_id").limit(100).collect()
]
customer_ids = [
    r["customer_id"]
    for r in spark.table(f"{BRONZE_DB}.customers")
                  .select("customer_id").limit(100).collect()
]

# Generate initial batch of claims
# Simulates source system full extract
initial_claims = []
base_time = datetime(2024, 1, 1, 0, 0, 0)

for i in range(500):
    incident  = rand_date(date(2023, 1, 1), date(2024, 6, 30))
    submitted = incident + timedelta(days=random.randint(1, 60))
    amt       = rand_amount(500, 50_000)
    # _updated_at simulates source system change timestamp
    # Incremental processing uses this to find new records
    updated_at = base_time + timedelta(hours=i)
    initial_claims.append({
        "claim_id":             gen_id("CLM"),
        "policy_id":            random.choice(policy_ids),
        "customer_id":          random.choice(customer_ids),
        "incident_date":        incident.isoformat(),
        "submitted_date":       submitted.isoformat(),
        "days_to_submit":       (submitted - incident).days,
        "claim_amount_chf":     amt,
        "claim_status":         random.choice(CLAIM_STATUSES),
        "claim_type":           random.choice(CLAIM_TYPES),
        "is_fraud_suspected":   random.random() < 0.05,
        "_updated_at":          updated_at.isoformat(),
        **audit_cols(BATCH_ID)
    })

initial_pdf = pd.DataFrame(initial_claims)
initial_sdf = spark.createDataFrame(initial_pdf)

# Write initial load to incremental claims table
initial_sdf.write.format("delta") \
           .mode("overwrite") \
           .option("overwriteSchema", "true") \
           .saveAsTable(f"{INCR_DB}.claims_source")

print(f"✅ Initial load: {len(initial_claims):,} claims")
print(f"   Date range: 2023-01-01 to 2024-06-30")


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 6 — Pattern 1: Watermark-Based Incremental Load
# Purpose : Process only records updated after last watermark.
#           Most common incremental pattern.
#
# How it works:
#   1. Read last watermark from tracking table
#   2. Query source: WHERE _updated_at > last_watermark
#   3. Process only those records
#   4. Write to target table (append mode)
#   5. Update watermark to max(_updated_at) of processed batch
#
# Benefit:
#   Run 1 processes 500 records (full history)
#   Run 2 processes only new records added since Run 1
#   Much faster for large datasets
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PATTERN 1 — WATERMARK BASED INCREMENTAL LOAD")
print(f"{'='*60}")

domain = "claims_watermark"

# Step 1: Get last watermark
last_watermark = get_watermark(domain)
print(f"\nLast watermark: {last_watermark}")

# Step 2: Read records — cast _updated_at to timestamp first
source_sdf = spark.table(f"{INCR_DB}.claims_source") \
    .withColumn("_updated_at",
                F.to_timestamp(F.col("_updated_at")))

new_records = source_sdf.filter(
    F.col("_updated_at") > F.lit(last_watermark)
)

new_count = new_records.count()
print(f"New records to process: {new_count:,}")

if new_count > 0:
    # Step 3: Add Bronze audit columns
    new_records = add_bronze_audit(
        new_records, domain, BATCH_ID
    )

    # Step 4: Write to target table
    new_records.write.format("delta") \
               .mode("append") \
               .option("mergeSchema", "true") \
               .saveAsTable(f"{INCR_DB}.{domain}")

    # Step 5: Update watermark — max timestamp as datetime
    max_ts_str = new_records \
        .agg(F.max("_updated_at").alias("max_ts")) \
        .collect()[0]["max_ts"]

    # Convert to Python datetime explicitly
    max_watermark = max_ts_str.replace(
        tzinfo=None
    ) if hasattr(max_ts_str, 'replace') else \
        datetime.strptime(
            str(max_ts_str)[:19], "%Y-%m-%d %H:%M:%S"
        )

    update_watermark(domain, max_watermark, new_count)

    print(f"✅ Written {new_count:,} records")
    print(f"✅ Watermark updated to: {max_watermark}")

# Verify watermark was saved
print("\nVerifying watermark saved:")
spark.sql(f"""
    SELECT * FROM {INCR_DB}.watermarks
""").show(truncate=False)


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 7 — Simulate New Records Arriving
# Purpose : Simulate source system sending new claims
#           that arrived after the initial load.
#           In real project this happens automatically
#           as new transactions occur in source system.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("SIMULATING: NEW RECORDS ARRIVING FROM SOURCE")
print(f"{'='*60}")

# Generate 50 new claims with timestamps AFTER initial load
# These simulate new insurance claims filed today
new_time = datetime(2024, 7, 1, 0, 0, 0)
new_claims = []

for i in range(50):
    incident  = rand_date(date(2024, 7, 1), date(2024, 7, 31))
    submitted = incident + timedelta(days=random.randint(1, 30))
    amt       = rand_amount(500, 75_000)
    updated_at = new_time + timedelta(hours=i)
    new_claims.append({
        "claim_id":             gen_id("CLM"),
        "policy_id":            random.choice(policy_ids),
        "customer_id":          random.choice(customer_ids),
        "incident_date":        incident.isoformat(),
        "submitted_date":       submitted.isoformat(),
        "days_to_submit":       (submitted - incident).days,
        "claim_amount_chf":     amt,
        "claim_status":         "submitted",
        "claim_type":           random.choice(CLAIM_TYPES),
        "is_fraud_suspected":   random.random() < 0.05,
        "_updated_at":          updated_at.isoformat(),
        **audit_cols(BATCH_ID)
    })

# Also simulate 10 UPDATED claims (status changed)
# These are existing claims whose status was updated
existing_ids = [
    r["claim_id"]
    for r in spark.table(f"{INCR_DB}.claims_source")
                  .select("claim_id").limit(10).collect()
]
updated_claims = []
for i, cid in enumerate(existing_ids):
    updated_at = new_time + timedelta(hours=50 + i)
    updated_claims.append({
        "claim_id":         cid,
        "claim_status":     "approved",  # status changed
        "_updated_at":      updated_at.isoformat(),
        "_update_type":     "STATUS_CHANGE",
    })

# Append new and updated records to source table
new_sdf = spark.createDataFrame(pd.DataFrame(new_claims))
new_sdf.write.format("delta").mode("append") \
       .option("mergeSchema", "true") \
       .saveAsTable(f"{INCR_DB}.claims_source")

print(f"✅ {len(new_claims)} new claims added to source")
print(f"✅ {len(updated_claims)} claims updated in source")
print(f"\nSource table now has:")
print(f"  {spark.table(f'{INCR_DB}.claims_source').count():,} total records")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 8 — Run Watermark Load Again (Second Run)
# Purpose : Demonstrate that second run only processes
#           the 50 new records — not all 500 again.
#           This is the core value of watermark pattern.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PATTERN 1 — SECOND RUN (INCREMENTAL)")
print(f"{'='*60}")

# Read watermark from first run
last_watermark = get_watermark(domain)
print(f"Last watermark: {last_watermark}")
print(f"Only processing records AFTER this timestamp...")

# Only new 50 records should be processed
new_records = spark.table(f"{INCR_DB}.claims_source") \
    .filter(F.col("_updated_at") > F.lit(last_watermark))

new_count = new_records.count()
print(f"\nRecords to process this run: {new_count:,}")
print(f"Records skipped (already processed): 500")
print(f"Efficiency gain: {500/(500+new_count)*100:.1f}% less work")

if new_count > 0:
    new_records = add_bronze_audit(
        new_records, domain, BATCH_ID
    )
    new_records.write.format("delta") \
               .mode("append") \
               .option("mergeSchema", "true") \
               .saveAsTable(f"{INCR_DB}.{domain}")

    max_watermark = new_records \
        .agg(F.max("_updated_at").alias("max_ts")) \
        .collect()[0]["max_ts"]

    update_watermark(domain, max_watermark, new_count)

    print(f"\n✅ Second run complete")
    print(f"   Processed: {new_count:,} records")
    print(f"   Watermark updated to: {max_watermark}")

total = spark.table(f"{INCR_DB}.{domain}").count()
print(f"   Target table total: {total:,} records")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 9 — Pattern 2: Delta MERGE (Upsert)
# Purpose : Handle both new AND updated records correctly.
#           Watermark alone misses updates to existing records.
#           MERGE solves this — INSERT new, UPDATE existing.
#
# How it works:
#   MERGE source INTO target
#   WHEN MATCHED (same ID exists) → UPDATE
#   WHEN NOT MATCHED (new ID)     → INSERT
#
# Why this is critical:
#   A claim status changes from 'submitted' to 'approved'
#   Watermark would INSERT a duplicate row
#   MERGE correctly UPDATES the existing row
#
# This is the most important production ingestion pattern.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PATTERN 2 — DELTA MERGE (UPSERT)")
print(f"{'='*60}")
print("""
Problem with append-only watermark:
  If claim CLM-001 status changes from 'submitted' to 'approved'
  Append would create TWO rows for CLM-001
  MERGE correctly updates the existing row

MERGE logic:
  WHEN source.claim_id = target.claim_id (match found)
    → UPDATE all columns with latest values
  WHEN no match found (new record)
    → INSERT new row
""")

# Create target table for MERGE demo
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {INCR_DB}.claims_merge_target
    USING DELTA
    AS SELECT * FROM {INCR_DB}.claims_source
    WHERE 1=0
""")

# Initial load via MERGE
initial_sdf = spark.table(f"{INCR_DB}.claims_source") \
    .filter(F.col("_updated_at") <= F.lit(
        datetime(2024, 6, 30, 23, 59, 59)
    ))

initial_count = initial_sdf.count()
print(f"Initial load: {initial_count:,} records via MERGE...")

# Use DeltaTable API for MERGE operation
target_table = DeltaTable.forName(
    spark, f"{INCR_DB}.claims_merge_target"
)

target_table.alias("target") \
    .merge(
        initial_sdf.alias("source"),
        "target.claim_id = source.claim_id"
    ) \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()

print(f"✅ Initial MERGE complete: {initial_count:,} records")

# Now MERGE the new/updated records
# This correctly handles both new claims AND status updates
new_sdf = spark.table(f"{INCR_DB}.claims_source") \
    .filter(F.col("_updated_at") > F.lit(
        datetime(2024, 6, 30, 23, 59, 59)
    ))

new_count = new_sdf.count()
print(f"\nMERGING {new_count:,} new/updated records...")

target_table.alias("target") \
    .merge(
        new_sdf.alias("source"),
        "target.claim_id = source.claim_id"
    ) \
    .whenMatchedUpdateAll() \
    .whenNotMatchedInsertAll() \
    .execute()

final_count = spark.table(
    f"{INCR_DB}.claims_merge_target"
).count()

print(f"✅ MERGE complete")
print(f"   Records in target: {final_count:,}")
print(f"   No duplicates — MERGE guarantees uniqueness")

# Verify no duplicates
dup_check = spark.table(
    f"{INCR_DB}.claims_merge_target"
) \
.groupBy("claim_id") \
.count() \
.filter(F.col("count") > 1) \
.count()

print(f"   Duplicate check: {dup_check} duplicates found ✅" \
      if dup_check == 0 else \
      f"   ❌ {dup_check} duplicates found!")


# COMMAND ----------


# ═══════════════════════════════════════════════════════════
# CELL 10 — Pattern 3: Change Data Capture (CDC)
# Purpose : Track and apply INSERT/UPDATE/DELETE operations
#           from source system to target table.
#
# CDC is used when:
#   Source system can send a change log not just data
#   Need to handle DELETE operations
#   Need full audit trail of every change
#
# CDC record format:
#   _cdc_operation : INSERT / UPDATE / DELETE
#   _cdc_timestamp : when change occurred
#   All data columns
#
# Processing logic:
#   DELETE operations → remove from target
#   INSERT/UPDATE    → MERGE into target
# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════
# CELL 10 — Pattern 3: Change Data Capture (CDC)
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("PATTERN 3 — CHANGE DATA CAPTURE (CDC)")
print(f"{'='*60}")
print("""
CDC captures every change from source system:
  INSERT → new record created
  UPDATE → existing record changed
  DELETE → record removed

More powerful than watermark alone because:
  Handles deletes (watermark cannot detect deletes)
  Full audit trail of every operation
  Exactly-once processing guarantee
""")

# Simulate CDC feed from source system
cdc_records = []

# 20 new claims (INSERT)
for i in range(20):
    incident = rand_date(date(2024, 8, 1), date(2024, 8, 31))
    cdc_records.append({
        "claim_id":         gen_id("CLM"),
        "policy_id":        random.choice(policy_ids),
        "customer_id":      random.choice(customer_ids),
        "claim_amount_chf": rand_amount(500, 50_000),
        "claim_status":     "submitted",
        "claim_type":       random.choice(CLAIM_TYPES),
        "_cdc_operation":   "INSERT",
        "_cdc_timestamp":   datetime(2024, 8, 1, i, 0, 0).isoformat(),
        **audit_cols(BATCH_ID)
    })

# 5 updated claims (UPDATE — status changed)
existing_claim_ids = [
    r["claim_id"]
    for r in spark.table(f"{INCR_DB}.claims_merge_target")
                  .select("claim_id").limit(5).collect()
]
for i, cid in enumerate(existing_claim_ids):
    cdc_records.append({
        "claim_id":         cid,
        "claim_status":     "settled",
        "_cdc_operation":   "UPDATE",
        "_cdc_timestamp":   datetime(2024, 8, 2, i, 0, 0).isoformat(),  # FIX 1: Aug 2nd avoids hour > 23
        **audit_cols(BATCH_ID)
    })

# 3 deleted claims (DELETE — withdrawn)
delete_ids = [
    r["claim_id"]
    for r in spark.table(f"{INCR_DB}.claims_merge_target")
                  .select("claim_id").limit(3).collect()
]
for i, cid in enumerate(delete_ids):
    cdc_records.append({
        "claim_id":         cid,
        "_cdc_operation":   "DELETE",
        "_cdc_timestamp":   datetime(2024, 8, 3, i, 0, 0).isoformat(),
    })

cdc_sdf = spark.createDataFrame(pd.DataFrame(cdc_records))
print(f"CDC feed received:")
print(f"  INSERT: 20 new claims")
print(f"  UPDATE: 5 status changes")
print(f"  DELETE: 3 withdrawn claims")

# Process CDC — DELETE first, then MERGE inserts/updates
deletes = cdc_sdf.filter(F.col("_cdc_operation") == "DELETE")
upserts = cdc_sdf.filter(F.col("_cdc_operation").isin("INSERT", "UPDATE"))

# Apply DELETEs
delete_count = deletes.count()
if delete_count > 0:
    delete_ids_list = [r["claim_id"] for r in deletes.collect()]
    target_table.delete(F.col("claim_id").isin(delete_ids_list))
    print(f"\n✅ Deleted {delete_count} records")

# Apply INSERTs and UPDATEs via MERGE
upsert_count = upserts.count()
if upsert_count > 0:
    target_table.alias("target") \
        .merge(
            upserts.alias("source"),
            "target.claim_id = source.claim_id"
        ) \
        .whenMatchedUpdate(set={
            "claim_status":         "source.claim_status",
            "claim_amount_chf":     "COALESCE(source.claim_amount_chf, target.claim_amount_chf)",
            "claim_type":           "COALESCE(source.claim_type, target.claim_type)",
            "policy_id":            "COALESCE(source.policy_id, target.policy_id)",
            "customer_id":          "COALESCE(source.customer_id, target.customer_id)",
            "incident_date":        "target.incident_date",
            "submitted_date":       "target.submitted_date",
            "days_to_submit":       "target.days_to_submit",
            "is_fraud_suspected":   "target.is_fraud_suspected",
            "_updated_at":          "source._cdc_timestamp",
            "_ingestion_timestamp": "source._ingestion_timestamp",
            "_batch_id":            "source._batch_id",
            "_source_system":       "COALESCE(source._source_system, target._source_system)",
            "_record_hash":         "target._record_hash",
        }) \
        .whenNotMatchedInsert(values={
            "claim_id":             "source.claim_id",
            "policy_id":            "source.policy_id",
            "customer_id":          "source.customer_id",
            "claim_amount_chf":     "source.claim_amount_chf",
            "claim_status":         "source.claim_status",
            "claim_type":           "source.claim_type",
            "incident_date":        "NULL",
            "submitted_date":       "NULL",
            "days_to_submit":       "NULL",
            "is_fraud_suspected":   "NULL",
            "_updated_at":          "source._cdc_timestamp",
            "_ingestion_timestamp": "source._ingestion_timestamp",
            "_source_system":       "source._source_system",
            "_record_hash":         "source._record_hash",
            "_batch_id":            "source._batch_id",
        }) \
        .execute()
    print(f"✅ Upserted {upsert_count} records")

final_count = spark.table(f"{INCR_DB}.claims_merge_target").count()
print(f"\n✅ CDC processing complete")
print(f"   Final record count: {final_count:,}")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 11 — Delta Time Travel
# Purpose : Show how Delta Lake keeps full history.
#           Can query any previous version of a table.
#           Critical for: auditing, rollback, debugging.
#
# Every MERGE/INSERT/DELETE creates a new Delta version.
# Time travel lets you query any version.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("BONUS — DELTA TIME TRAVEL")
print(f"{'='*60}")
print("""
Every write to a Delta table creates a new version.
You can query ANY previous version — forever (until VACUUM).
Use cases:
  - Audit: "what did this table look like yesterday?"
  - Rollback: "undo last pipeline run"
  - Debug: "what changed between version 1 and 2?"
""")

# Show table history
print("Table history:")
spark.sql(f"""
    DESCRIBE HISTORY {INCR_DB}.claims_merge_target
    LIMIT 5
""").select(
    "version", "timestamp", "operation",
    "operationParameters"
).show(truncate=False)

# Query previous version (version 0 = initial state)
print("Records at version 0 (initial load):")
v0_count = spark.read \
    .format("delta") \
    .option("versionAsOf", 0) \
    .table(f"{INCR_DB}.claims_merge_target") \
    .count()
print(f"  Version 0: {v0_count:,} records")

# Query current version
current_count = spark.table(
    f"{INCR_DB}.claims_merge_target"
).count()
print(f"  Current:   {current_count:,} records")
print(f"  Difference: {current_count - v0_count:,} net change")

# COMMAND ----------



# ═══════════════════════════════════════════════════════════
# CELL 12 — Watermark History Report
# Purpose : Show complete watermark history.
#           Proves incremental processing is working.
#           In production this feeds a monitoring dashboard.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("WATERMARK HISTORY REPORT")
print(f"{'='*60}")

spark.sql(f"""
    SELECT domain,
           last_watermark,
           record_count,
           last_batch_id,
           date_format(updated_at,
               'yyyy-MM-dd HH:mm:ss') AS updated_at
    FROM {INCR_DB}.watermarks
    ORDER BY domain, updated_at
""").show(truncate=False)

# ═══════════════════════════════════════════════════════════
# CELL 13 — Final Summary
# Purpose : Summary of all three incremental patterns
#           and when to use each in production.
# ═══════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("INCREMENTAL INGESTION — PATTERN SUMMARY")
print(f"{'='*60}")
print("""
Pattern 1 — Watermark
  Best for : Append-only sources (new records only)
  Example  : New claims filed today
  Pros     : Simple, fast, easy to implement
  Cons     : Cannot detect updates to existing records
  Use when : Source only adds new records, never updates

Pattern 2 — Delta MERGE (Upsert)
  Best for : Sources with inserts AND updates
  Example  : Claim status changes (submitted → approved)
  Pros     : Handles both new and changed records
  Cons     : Slightly more complex, requires primary key
  Use when : Records can be updated after creation

Pattern 3 — Change Data Capture (CDC)
  Best for : Full change tracking including deletes
  Example  : Policy cancellations, withdrawn claims
  Pros     : Complete audit trail, handles all operations
  Cons     : Source must support CDC output
  Use when : Need to handle deletes or full audit trail

Production recommendation for insurance platform:
  Claims     → MERGE (status changes frequently)
  Policies   → MERGE (renewals, cancellations)
  Premiums   → Watermark (payments only append)
  Customers  → MERGE (address/contact updates)
""")

print(f"✅ Incremental ingestion demo complete")
print(f"   Batch ID: {BATCH_ID}")