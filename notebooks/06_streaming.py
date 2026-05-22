# Databricks notebook source
"""
06_streaming.py
================
Production-grade Streaming Pipeline for Insurance Data Platform.

Parts:
    A — Structured Streaming: Rate Source (synthetic event generation)
    B — Structured Streaming: Autoloader (cloud file ingestion)
    C — Structured Streaming: Socket Stream (real-time input)
    D — Live Fraud Detection: Windowed aggregations + watermarking
    E — Delta Live Tables: Declarative pipeline definition

Architecture:
    Event Source → Structured Streaming → Delta Tables → Fraud Detection

Run cell by cell in Databricks notebook.
"""

# ─────────────────────────────────────────────
# CELL 1 — Imports & Configuration
# ─────────────────────────────────────────────

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.streaming import StreamingQuery
from datetime import datetime
import logging
import time
import json
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("insurance_streaming")

STREAMING_DB       = "insurance_streaming"
CHECKPOINT_BASE    = "dbfs:/tmp/insurance_streaming/checkpoints"
STREAMING_OUTPUT   = "dbfs:/tmp/insurance_streaming/output"
AUTOLOADER_INPUT   = "dbfs:/tmp/insurance_streaming/input"
BATCH_ID           = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# Streaming trigger interval
TRIGGER_INTERVAL   = "10 seconds"

print(f"Streaming pipeline starting — Batch ID: {BATCH_ID}")
print(f"Checkpoint path: {CHECKPOINT_BASE}")


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 2 — Setup Streaming Database & Paths
# ─────────────────────────────────────────────

spark.sql(f"DROP DATABASE IF EXISTS {STREAMING_DB} CASCADE")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {STREAMING_DB}")
spark.sql(f"USE {STREAMING_DB}")

# Create required DBFS directories
dbutils.fs.mkdirs(CHECKPOINT_BASE)
dbutils.fs.mkdirs(STREAMING_OUTPUT)
dbutils.fs.mkdirs(AUTOLOADER_INPUT)

print(f"✅ Database '{STREAMING_DB}' ready")
print(f"✅ DBFS paths created")


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 3 — Define Streaming Schemas
# ─────────────────────────────────────────────
# Production rule: always define explicit schemas
# for streaming — never infer from data

CLAIM_EVENT_SCHEMA = T.StructType([
    T.StructField("claim_id",           T.StringType(),    False),
    T.StructField("policy_id",          T.StringType(),    False),
    T.StructField("customer_id",        T.StringType(),    False),
    T.StructField("claim_amount_chf",   T.DoubleType(),    True),
    T.StructField("claim_type",         T.StringType(),    True),
    T.StructField("claim_status",       T.StringType(),    True),
    T.StructField("incident_date",      T.StringType(),    True),
    T.StructField("is_fraud_suspected", T.BooleanType(),   True),
    T.StructField("event_timestamp",    T.TimestampType(), True),
    T.StructField("handler_id",         T.StringType(),    True),
])

POLICY_EVENT_SCHEMA = T.StructType([
    T.StructField("policy_id",          T.StringType(),    False),
    T.StructField("customer_id",        T.StringType(),    False),
    T.StructField("policy_type",        T.StringType(),    True),
    T.StructField("event_type",         T.StringType(),    True),
    T.StructField("annual_premium_chf", T.DoubleType(),    True),
    T.StructField("event_timestamp",    T.TimestampType(), True),
])

print("✅ Streaming schemas defined")

# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 4 — Helper: Generate Claim Event JSON
# ─────────────────────────────────────────────

import uuid

CLAIM_TYPES    = ["accident", "theft", "fire", "flood", "medical"]
CLAIM_STATUSES = ["submitted", "under_review", "approved", "rejected"]
POLICY_TYPES   = ["motor", "home", "life", "health", "travel"]

def generate_claim_event():
    """Generate a single realistic claim event as JSON string."""
    is_fraud = random.random() < 0.05
    return json.dumps({
        "claim_id":           f"CLM-{uuid.uuid4().hex[:8].upper()}",
        "policy_id":          f"POL-{uuid.uuid4().hex[:8].upper()}",
        "customer_id":        f"CUST-{uuid.uuid4().hex[:8].upper()}",
        "claim_amount_chf":   round(random.uniform(500, 150000), 2),
        "claim_type":         random.choice(CLAIM_TYPES),
        "claim_status":       random.choice(CLAIM_STATUSES),
        "incident_date":      "2024-01-15",
        "is_fraud_suspected": is_fraud,
        "event_timestamp":    datetime.utcnow().isoformat(),
        "handler_id":         f"CH-{random.randint(1000, 9999)}",
    })

# Test the generator
print("Sample claim event:")
print(generate_claim_event())


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 5 — PART A: Rate Source Streaming
# ─────────────────────────────────────────────
# Rate source generates rows at a fixed rate.
# Use case: testing, benchmarking, learning streaming concepts.
# NOT for production — synthetic only.

print(f"\n{'='*55}")
print("PART A — RATE SOURCE STREAMING")
print(f"{'='*55}")
print("""
Rate Source:
  - Built into Spark — no external dependencies
  - Generates rows at a fixed rate (rowsPerSecond)
  - Each row has: timestamp, value (incrementing integer)
  - Perfect for testing streaming logic
  - Use case: load testing, pipeline validation
""")

# Create rate stream — 10 rows per second
rate_stream = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 10) \
    .option("numPartitions", 2) \
    .load()

print("Rate stream schema:")
rate_stream.printSchema()

# Transform rate stream into claim events
claim_events_from_rate = rate_stream \
    .withColumn("claim_id",
                F.concat(F.lit("CLM-"), F.lpad(F.col("value").cast("string"), 8, "0"))) \
    .withColumn("claim_amount_chf",
                F.round(F.abs(F.sin(F.col("value"))) * 150000, 2)) \
    .withColumn("claim_type",
                F.element_at(
                    F.array([F.lit(t) for t in CLAIM_TYPES]),
                    (F.col("value") % 5 + 1).cast("int")
                )) \
    .withColumn("claim_status",
                F.element_at(
                    F.array([F.lit(s) for s in CLAIM_STATUSES]),
                    (F.col("value") % 4 + 1).cast("int")
                )) \
    .withColumn("is_fraud_suspected",
                (F.col("value") % 20 == 0)) \
    .withColumn("event_timestamp", F.col("timestamp")) \
    .select(
        "claim_id", "claim_amount_chf", "claim_type",
        "claim_status", "is_fraud_suspected", "event_timestamp"
    )

# Write rate stream to Delta table
rate_query = claim_events_from_rate \
    .writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/rate_claims") \
    .trigger(processingTime=TRIGGER_INTERVAL) \
    .toTable(f"{STREAMING_DB}.claims_rate_stream")

print(f"✅ Rate stream started — writing to {STREAMING_DB}.claims_rate_stream")
print(f"   Trigger interval: {TRIGGER_INTERVAL}")
print(f"   Checkpoint: {CHECKPOINT_BASE}/rate_claims")

# Let it run for 30 seconds
print("\nRunning for 30 seconds...")
time.sleep(30)

# Check how many records landed
count = spark.table(f"{STREAMING_DB}.claims_rate_stream").count()
print(f"✅ Records ingested in 30s: {count:,}")

# Stop the query
rate_query.stop()
print("✅ Rate stream stopped")

# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 6 — PART A: Rate Stream Analysis
# ─────────────────────────────────────────────

print("\nRate stream results:")
spark.sql(f"""
    SELECT claim_type,
           COUNT(*) AS event_count,
           ROUND(AVG(claim_amount_chf), 2) AS avg_amount,
           SUM(CASE WHEN is_fraud_suspected THEN 1 ELSE 0 END) AS fraud_count
    FROM {STREAMING_DB}.claims_rate_stream
    GROUP BY claim_type
    ORDER BY event_count DESC
""").show()


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 7 — PART B: Autoloader Setup
# ─────────────────────────────────────────────
# Autoloader (cloudFiles) monitors a cloud storage path
# and automatically processes new files as they arrive.
# Production standard for file-based streaming in Azure.

print(f"\n{'='*55}")
print("PART B — AUTOLOADER (cloudFiles)")
print(f"{'='*55}")
print("""
Autoloader:
  - Monitors ADLS/DBFS path for new files
  - Automatically detects and processes new arrivals
  - Supports JSON, CSV, Parquet, Avro, ORC
  - Exactly-once processing via checkpointing
  - Scales to millions of files
  - Production standard for file landing zones
  - Handles schema evolution automatically
""")

# First — generate sample JSON files to simulate file landing
print("Generating sample claim event files...")

import os
for i in range(5):
    events = [generate_claim_event() for _ in range(100)]
    file_content = "\n".join(events)
    dbfs_path = f"dbfs:/tmp/insurance_streaming/input/claims_batch_{i:03d}.json"

    # Write using dbutils — works on all Databricks runtimes
    dbutils.fs.put(dbfs_path, file_content, overwrite=True)
    print(f"  Created: claims_batch_{i:03d}.json (100 events)")

print(f"✅ 5 files × 100 events = 500 events ready for Autoloader")

# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 8 — PART B: Autoloader Stream
# ─────────────────────────────────────────────

# Define schema for JSON events
autoloader_schema = T.StructType([
    T.StructField("claim_id",           T.StringType(),  True),
    T.StructField("policy_id",          T.StringType(),  True),
    T.StructField("customer_id",        T.StringType(),  True),
    T.StructField("claim_amount_chf",   T.DoubleType(),  True),
    T.StructField("claim_type",         T.StringType(),  True),
    T.StructField("claim_status",       T.StringType(),  True),
    T.StructField("incident_date",      T.StringType(),  True),
    T.StructField("is_fraud_suspected", T.BooleanType(), True),
    T.StructField("event_timestamp",    T.StringType(),  True),
    T.StructField("handler_id",         T.StringType(),  True),
])

# Read using Autoloader
autoloader_stream = spark.readStream \
    .format("cloudFiles") \
    .option("cloudFiles.format", "json") \
    .option("cloudFiles.schemaLocation",
            f"{CHECKPOINT_BASE}/autoloader_schema") \
    .schema(autoloader_schema) \
    .load(AUTOLOADER_INPUT)

# Add ingestion metadata
autoloader_enriched = autoloader_stream \
    .withColumn("_source_file",    F.col("_metadata.file_path")) \
    .withColumn("_ingestion_time", F.current_timestamp()) \
    .withColumn("event_timestamp",
                F.to_timestamp(F.col("event_timestamp"))) \
    .withColumn("claim_severity",
                F.when(F.col("claim_amount_chf") < 5000,   F.lit("LOW"))
                 .when(F.col("claim_amount_chf") < 25000,  F.lit("MEDIUM"))
                 .when(F.col("claim_amount_chf") < 75000,  F.lit("HIGH"))
                 .otherwise(F.lit("CATASTROPHIC")))

# Write to Delta with Autoloader
autoloader_query = autoloader_enriched \
    .writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/autoloader_claims") \
    .trigger(availableNow=True) \
    .toTable(f"{STREAMING_DB}.claims_autoloader")

# Wait for completion
autoloader_query.awaitTermination()
print("✅ Autoloader processing complete")

count = spark.table(f"{STREAMING_DB}.claims_autoloader").count()
print(f"✅ Records loaded via Autoloader: {count:,}")

# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 9 — PART B: Autoloader Results
# ─────────────────────────────────────────────

print("\nAutoloader results — source file tracking:")
spark.sql(f"""
    SELECT _source_file,
           COUNT(*) AS records_loaded,
           ROUND(AVG(claim_amount_chf), 2) AS avg_claim,
           SUM(CASE WHEN is_fraud_suspected THEN 1 ELSE 0 END) AS fraud_count
    FROM {STREAMING_DB}.claims_autoloader
    GROUP BY _source_file
    ORDER BY _source_file
""").show(truncate=False)

print("\nAutoloader — simulate new file arriving:")
print("Generating new batch file...")

new_events = [generate_claim_event() for _ in range(200)]
dbutils.fs.put(
    "dbfs:/tmp/insurance_streaming/input/claims_batch_NEW.json",
    "\n".join(new_events),
    overwrite=True
)
print("✅ New file dropped: claims_batch_NEW.json (200 events)")
print("   In production: Autoloader detects this automatically")
print("   Re-run the Autoloader query to process it")


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 10 — PART C: Explain Trigger Modes
# ─────────────────────────────────────────────

print(f"\n{'='*55}")
print("STREAMING TRIGGER MODES — EXPLAINED")
print(f"{'='*55}")
print("""
Three trigger modes in Structured Streaming:

1. processingTime="10 seconds"
   → Runs micro-batch every N seconds
   → Use for: near-real-time dashboards, fraud detection
   → Example: process claims every 10 seconds

2. availableNow=True
   → Process all available data then stop
   → Use for: scheduled batch-style streaming jobs
   → Example: process all files that landed overnight
   → Similar to traditional batch but with streaming semantics

3. Continuous (experimental)
   → Sub-millisecond latency
   → Use for: ultra-low latency requirements
   → Limited operations supported

Production choice for insurance:
   → Claims fraud detection: processingTime="30 seconds"
   → File landing zone:      availableNow=True (scheduled)
   → Policy changes:         processingTime="1 minute"
""")

# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 11 — PART D: Live Fraud Detection
# ─────────────────────────────────────────────
# Windowed aggregation — count claims per customer
# in a sliding window to detect high frequency claimants.
# Watermarking — tells Spark how late data can arrive.

print(f"\n{'='*55}")
print("PART D — LIVE FRAUD DETECTION")
print(f"{'='*55}")
print("""
Key streaming fraud detection patterns:

1. Watermarking
   → Tells Spark: "data can arrive up to N minutes late"
   → Spark keeps state for late data within that window
   → After watermark threshold: state is dropped
   → Prevents unbounded state growth

2. Windowed Aggregation
   → Count events in a time window (e.g. last 30 minutes)
   → Slide the window forward as time progresses
   → Detect: customer submitting 3+ claims in 30 minutes

3. Tumbling Window vs Sliding Window
   Tumbling: [0-10min] [10-20min] [20-30min] — no overlap
   Sliding:  [0-10min] [5-15min] [10-20min] — overlaps
""")

# Read from our autoloader Delta table as a stream
fraud_stream = spark.readStream \
    .format("delta") \
    .table(f"{STREAMING_DB}.claims_autoloader")

# Apply watermark — allow up to 10 minutes late data
fraud_watermarked = fraud_stream \
    .withWatermark("event_timestamp", "10 minutes")

# Tumbling window — count claims per customer per 5-min window
fraud_windowed = fraud_watermarked \
    .groupBy(
        F.window(F.col("event_timestamp"), "5 minutes"),
        F.col("customer_id")
    ) \
    .agg(
        F.count("claim_id")                             .alias("claims_in_window"),
        F.sum("claim_amount_chf")                       .alias("total_amount_in_window"),
        F.sum(F.when(F.col("is_fraud_suspected"),
                     F.lit(1)).otherwise(F.lit(0)))     .alias("fraud_signals_in_window"),
        F.collect_list("claim_type")                    .alias("claim_types"),
    ) \
    .withColumn("window_start",     F.col("window.start")) \
    .withColumn("window_end",       F.col("window.end")) \
    .withColumn("is_velocity_flag",
                F.when(F.col("claims_in_window") >= 3,  F.lit(True))
                 .otherwise(F.lit(False))) \
    .withColumn("velocity_risk_level",
                F.when(F.col("claims_in_window") >= 5,  F.lit("CRITICAL"))
                 .when(F.col("claims_in_window") >= 3,  F.lit("HIGH"))
                 .when(F.col("claims_in_window") >= 2,  F.lit("MEDIUM"))
                 .otherwise(F.lit("NORMAL"))) \
    .drop("window")

# Write fraud signals to Delta
fraud_query = fraud_windowed \
    .writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", f"{CHECKPOINT_BASE}/fraud_detection") \
    .trigger(availableNow=True) \
    .toTable(f"{STREAMING_DB}.fraud_velocity_signals")

fraud_query.awaitTermination()
print("✅ Fraud velocity detection complete")

count = spark.table(f"{STREAMING_DB}.fraud_velocity_signals").count()
print(f"✅ Velocity signal windows generated: {count:,}")


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 12 — PART D: Fraud Detection Results
# ─────────────────────────────────────────────

print("\nFraud velocity signals:")
spark.sql(f"""
    SELECT customer_id,
           window_start,
           window_end,
           claims_in_window,
           ROUND(total_amount_in_window, 2) AS total_amount_chf,
           fraud_signals_in_window,
           velocity_risk_level,
           is_velocity_flag
    FROM {STREAMING_DB}.fraud_velocity_signals
    ORDER BY claims_in_window DESC
    LIMIT 10
""").show(truncate=False)

print("\nVelocity risk level distribution:")
spark.sql(f"""
    SELECT velocity_risk_level,
           COUNT(*) AS window_count,
           SUM(CASE WHEN is_velocity_flag THEN 1 ELSE 0 END) AS flagged_windows
    FROM {STREAMING_DB}.fraud_velocity_signals
    GROUP BY velocity_risk_level
    ORDER BY window_count DESC
""").show()

# COMMAND ----------

# ─────────────────────────────────────────────
# CELL 13 — PART E: Delta Live Tables Explained
# ─────────────────────────────────────────────

print(f"\n{'='*55}")
print("PART E — DELTA LIVE TABLES")
print(f"{'='*55}")
print("""
Delta Live Tables (DLT) is Databricks' declarative
pipeline framework. Instead of writing HOW to run
a pipeline, you declare WHAT the tables should contain.

Key differences vs Structured Streaming:

  Structured Streaming    Delta Live Tables
  ─────────────────────   ─────────────────────────
  Imperative (how)        Declarative (what)
  Manual DQ checks        Built-in expectations
  Manual retry logic      Automatic retry
  Manual monitoring       Pipeline UI + metrics
  You manage state        DLT manages state
  Any cluster             DLT cluster (managed)

When to use DLT:
  → Team delivery with governance requirements
  → Non-expert engineers maintaining pipelines
  → Built-in data quality enforcement needed
  → Automatic lineage tracking required

When to use Structured Streaming:
  → Complex custom logic
  → Full control over execution
  → Advanced windowing patterns
  → Integration with non-Delta sinks
""")

# COMMAND ----------



# ─────────────────────────────────────────────
# CELL 14 — PART E: DLT Pipeline Definition
# ─────────────────────────────────────────────
# DLT pipelines are defined as Python notebooks
# with @dlt.table and @dlt.expect decorators.
# The actual DLT execution happens in a Pipeline UI.
# Below is the complete DLT notebook code.

dlt_notebook_code = '''
# ── DLT Notebook: insurance_dlt_pipeline.py ──
# Deploy this as a separate Databricks notebook
# Then create a Delta Live Tables Pipeline pointing to it.

import dlt
from pyspark.sql import functions as F
from pyspark.sql import types as T

AUTOLOADER_PATH = "dbfs:/tmp/insurance_streaming/input"

CLAIM_SCHEMA = T.StructType([
    T.StructField("claim_id",           T.StringType(),  True),
    T.StructField("policy_id",          T.StringType(),  True),
    T.StructField("customer_id",        T.StringType(),  True),
    T.StructField("claim_amount_chf",   T.DoubleType(),  True),
    T.StructField("claim_type",         T.StringType(),  True),
    T.StructField("claim_status",       T.StringType(),  True),
    T.StructField("is_fraud_suspected", T.BooleanType(), True),
    T.StructField("event_timestamp",    T.StringType(),  True),
])

# ── Bronze: Raw ingestion with expectations ──
@dlt.table(
    name="bronze_claims_stream",
    comment="Raw claims events from landing zone",
    table_properties={"quality": "bronze"}
)
@dlt.expect("valid_claim_id",         "claim_id IS NOT NULL")
@dlt.expect("valid_claim_amount",     "claim_amount_chf > 0")
@dlt.expect_or_drop("valid_policy",   "policy_id IS NOT NULL")
def bronze_claims_stream():
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("cloudFiles.schemaLocation",
                    "dbfs:/tmp/dlt_schema/claims")
            .schema(CLAIM_SCHEMA)
            .load(AUTOLOADER_PATH)
            .withColumn("_ingestion_time", F.current_timestamp())
            .withColumn("_source_file",    F.input_file_name())
    )

# ── Silver: Cleansed + enriched ──
@dlt.table(
    name="silver_claims_stream",
    comment="Cleansed and enriched claims stream",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_fail("no_negative_amounts", "claim_amount_chf >= 0")
def silver_claims_stream():
    return (
        dlt.read_stream("bronze_claims_stream")
            .withColumn("event_timestamp",
                        F.to_timestamp(F.col("event_timestamp")))
            .withColumn("claim_severity",
                F.when(F.col("claim_amount_chf") < 5000,
                       F.lit("LOW"))
                 .when(F.col("claim_amount_chf") < 25000,
                       F.lit("MEDIUM"))
                 .when(F.col("claim_amount_chf") < 75000,
                       F.lit("HIGH"))
                 .otherwise(F.lit("CATASTROPHIC")))
            .withColumn("fraud_risk",
                F.when(F.col("is_fraud_suspected") == F.lit(True),
                       F.lit("HIGH"))
                 .otherwise(F.lit("NORMAL")))
    )

# ── Gold: Live aggregations ──
@dlt.table(
    name="gold_claims_summary",
    comment="Live claims summary by type and severity",
    table_properties={"quality": "gold"}
)
def gold_claims_summary():
    return (
        dlt.read_stream("silver_claims_stream")
            .withWatermark("event_timestamp", "10 minutes")
            .groupBy(
                F.window("event_timestamp", "5 minutes"),
                "claim_type",
                "claim_severity"
            )
            .agg(
                F.count("claim_id")
                    .alias("claim_count"),
                F.sum("claim_amount_chf")
                    .alias("total_amount_chf"),
                F.sum(F.when(F.col("is_fraud_suspected"),
                             F.lit(1)).otherwise(F.lit(0)))
                    .alias("fraud_count")
            )
    )
'''

print("DLT Pipeline code:")
print(dlt_notebook_code)
print("""
To deploy this DLT pipeline in Databricks:
  1. Create a new notebook — paste the code above
  2. Left sidebar → Delta Live Tables → Create Pipeline
  3. Pipeline name: insurance_dlt_pipeline
  4. Notebook path: point to the notebook above
  5. Target schema: insurance_streaming
  6. Pipeline mode: Triggered (for batch) or Continuous
  7. Click Start
""")


# COMMAND ----------


# ─────────────────────────────────────────────
# CELL 15 — Streaming Summary
# ─────────────────────────────────────────────

print(f"\n{'='*55}")
print("STREAMING — FINAL SUMMARY")
print(f"{'='*55}")

streaming_tables = [
    "claims_rate_stream",
    "claims_autoloader",
    "fraud_velocity_signals",
]

total = 0
for table in streaming_tables:
    try:
        count = spark.table(f"{STREAMING_DB}.{table}").count()
        total += count
        print(f"  {table:<30} {count:>8,} rows ✅")
    except Exception as e:
        print(f"  {table:<30} ERROR ❌")

print(f"\n  {'TOTAL':<30} {total:>8,} rows")
print(f"  Batch ID: {BATCH_ID}")
print(f"{'='*55}")

print("""
STREAMING PATTERNS DEMONSTRATED:
  ✅ Rate Source          — synthetic event generation
  ✅ Autoloader           — cloud file ingestion (production)
  ✅ Trigger modes        — processingTime vs availableNow
  ✅ Watermarking         — late data handling
  ✅ Windowed aggregation — tumbling windows for fraud
  ✅ Velocity detection   — claim frequency per customer
  ✅ DLT pipeline         — declarative Bronze/Silver/Gold
  ✅ DLT expectations     — built-in data quality
  ✅ Checkpointing        — fault tolerant exactly-once
""")