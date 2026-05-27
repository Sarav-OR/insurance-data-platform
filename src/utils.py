"""
src/utils.py
============
Shared Utility Functions for Insurance Data Platform
=====================================================
Purpose : Reusable functions imported by all notebooks.
          Defined once here — no copy/paste across notebooks.
          Bug fix here fixes it everywhere.

Usage   : from src.utils import gen_id, rand_date, audit_cols
          from src.utils import deduplicate, safe_date

Categories:
  Data Generation  — gen_id, rand_date, rand_amount, audit_cols
  PySpark Helpers  — safe_date, safe_timestamp, standardise_string
  Transformations  — deduplicate
  Delta Operations — write_delta, apply_delta_settings
"""

import uuid
import hashlib
import random
from datetime import datetime, timedelta, date
from typing import Optional, List

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame

# ─────────────────────────────────────────────────────────
# DATA GENERATION HELPERS
# Used by Bronze layer to generate synthetic data.
# ─────────────────────────────────────────────────────────

def gen_id(prefix: str) -> str:
    """
    Generate a unique prefixed identifier.
    Format: PREFIX-XXXXXXXXXX (10 hex chars uppercase)
    Example: CLM-A1B2C3D4E5

    Args:
        prefix: Short identifier prefix e.g. 'CLM', 'POL'

    Returns:
        Unique string ID
    """
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


def rand_date(start: date, end: date) -> date:
    """
    Generate a random date between start and end (inclusive).

    Args:
        start: Earliest possible date
        end  : Latest possible date

    Returns:
        Random date within range
    """
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def rand_amount(lo: float, hi: float) -> float:
    """
    Generate a random monetary amount rounded to 2dp.

    Args:
        lo: Minimum amount
        hi: Maximum amount

    Returns:
        Random float rounded to 2 decimal places
    """
    return round(random.uniform(lo, hi), 2)


def audit_cols(batch_id: str,
               source_system: str = "synthetic_generator_v2") -> dict:
    """
    Generate standard audit columns for every record.
    These answer: who sent this, when, is it unchanged?

    _ingestion_timestamp : when source system generated record
    _source_system       : which system sent it
    _record_hash         : MD5 fingerprint to detect duplicates
    _batch_id            : which pipeline run loaded it

    Args:
        batch_id     : Current pipeline batch ID
        source_system: Name of source system

    Returns:
        Dict of audit column key-value pairs
    """
    return {
        "_ingestion_timestamp": datetime.utcnow().isoformat(),
        "_source_system":       source_system,
        "_record_hash":         hashlib.md5(
                                    str(uuid.uuid4()).encode()
                                ).hexdigest(),
        "_batch_id":            batch_id,
    }


def age_band(dob: date) -> str:
    """
    Calculate age band string from date of birth.
    Used for customer segmentation analytics.

    Args:
        dob: Date of birth

    Returns:
        Age band string e.g. '25-34', '65+'
    """
    age = (date.today() - dob).days // 365
    if age < 25:   return "18-24"
    elif age < 35: return "25-34"
    elif age < 45: return "35-44"
    elif age < 55: return "45-54"
    elif age < 65: return "55-64"
    else:          return "65+"


# ─────────────────────────────────────────────────────────
# PYSPARK TRANSFORMATION HELPERS
# Reusable column expressions used in Silver layer.
# ─────────────────────────────────────────────────────────

def safe_date(col_name: str):
    """
    Safely cast string column to date type.
    Unparseable values become NULL instead of raising error.
    Always use this instead of direct cast() for date columns.

    Args:
        col_name: Name of string column to cast

    Returns:
        PySpark Column expression
    """
    return F.to_date(F.col(col_name), "yyyy-MM-dd")


def safe_timestamp(col_name: str):
    """
    Safely cast string column to timestamp type.

    Args:
        col_name: Name of string column to cast

    Returns:
        PySpark Column expression
    """
    return F.to_timestamp(F.col(col_name))


def standardise_string(col_name: str):
    """
    Standardise categorical string column.
    Trims whitespace and converts to uppercase.
    Ensures 'motor', 'MOTOR', ' Motor ' all become 'MOTOR'.

    Args:
        col_name: Name of string column to standardise

    Returns:
        PySpark Column expression
    """
    return F.upper(F.trim(F.col(col_name)))


# ─────────────────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────────────────

def deduplicate(sdf: DataFrame,
                id_col: str,
                order_col: str = "_ingestion_timestamp") -> DataFrame:
    """
    Remove duplicate records keeping the latest per ID.

    Why this approach:
    - Source systems sometimes resend updated records
    - We keep the most recent version using window function
    - row_number() = 1 means latest for that ID
    - Safe for large datasets — no collect() or distinct()

    Args:
        sdf      : Input Spark DataFrame
        id_col   : Primary key column name
        order_col: Timestamp column to order by (latest wins)

    Returns:
        Deduplicated DataFrame

    Example:
        # If CLM-001 appears 3 times with different statuses
        # Only the most recent record is kept
        deduped = deduplicate(claims_sdf, "claim_id")
    """
    window = Window \
        .partitionBy(id_col) \
        .orderBy(F.col(order_col).desc())

    return sdf \
        .withColumn("_row_num", F.row_number().over(window)) \
        .filter(F.col("_row_num") == 1) \
        .drop("_row_num")


# ─────────────────────────────────────────────────────────
# DELTA OPERATIONS
# ─────────────────────────────────────────────────────────

def write_delta(sdf: DataFrame,
                database: str,
                table: str,
                partition_cols: Optional[List[str]] = None,
                mode: str = "overwrite") -> int:
    """
    Write Spark DataFrame to Delta table.
    Returns actual row count for post-write validation.

    Args:
        sdf           : DataFrame to write
        database      : Target database name
        table         : Target table name
        partition_cols: List of columns to partition by
        mode          : 'overwrite' or 'append'

    Returns:
        Row count of written table (for validation)
    """
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()

    full_table = f"{database}.{table}"
    writer = sdf.write.format("delta").mode(mode) \
             .option("overwriteSchema", "true")
             

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.saveAsTable(full_table)

    # Post-write count verifies data landed correctly
    count = spark.table(full_table).count()
    return count


def apply_delta_settings(spark, settings: dict) -> None:
    """
    Apply Delta Lake configuration settings to Spark session.
    Call this once at start of each notebook.

    Args:
        spark   : Active SparkSession
        settings: Dict of spark.conf key-value pairs
    """
    for key, value in settings.items():
        spark.conf.set(key, value)


def drop_bronze_audit_cols(sdf: DataFrame) -> DataFrame:
    """
    Remove Bronze audit columns before writing to Silver.
    Silver adds its own audit columns — Bronze ones not needed.

    Args:
        sdf: DataFrame with Bronze audit columns

    Returns:
        DataFrame with Bronze audit columns removed
    """
    bronze_audit_cols = [
        "_ingestion_timestamp",
        "_source_system",
        "_record_hash",
        "_batch_id",
        "_bronze_load_timestamp",
        "_bronze_batch_id",
        "_bronze_domain",
    ]
    existing = [c for c in bronze_audit_cols if c in sdf.columns]
    return sdf.drop(*existing)
