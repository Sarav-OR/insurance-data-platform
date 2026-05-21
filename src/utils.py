"""
src/utils.py
============
Shared utility functions for Insurance Data Platform.
Imported by all notebook layers.
"""

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame
from datetime import datetime
from typing import Optional, List, Tuple
import hashlib
import uuid
import logging

log = logging.getLogger("insurance_utils")


# ─────────────────────────────────────────────
# AUDIT FUNCTIONS
# ─────────────────────────────────────────────

def add_bronze_audit(sdf: DataFrame, domain: str, batch_id: str) -> DataFrame:
    """Add Bronze layer audit columns."""
    return sdf \
        .withColumn("_bronze_batch_id",        F.lit(batch_id)) \
        .withColumn("_bronze_load_timestamp",  F.current_timestamp()) \
        .withColumn("_bronze_domain",          F.lit(domain)) \
        .withColumn("_bronze_file_path",       F.input_file_name())


def add_silver_audit(sdf: DataFrame, batch_id: str) -> DataFrame:
    """Add Silver layer audit columns."""
    return sdf \
        .withColumn("_silver_batch_id",        F.lit(batch_id)) \
        .withColumn("_silver_load_timestamp",  F.current_timestamp()) \
        .withColumn("_silver_source_layer",    F.lit("bronze"))


def add_gold_audit(sdf: DataFrame, batch_id: str) -> DataFrame:
    """Add Gold layer audit columns."""
    return sdf \
        .withColumn("_gold_batch_id",          F.lit(batch_id)) \
        .withColumn("_gold_load_timestamp",    F.current_timestamp()) \
        .withColumn("_gold_source_layer",      F.lit("silver"))


def add_fraud_audit(sdf: DataFrame, batch_id: str) -> DataFrame:
    """Add Fraud layer audit columns."""
    return sdf \
        .withColumn("_fraud_batch_id",         F.lit(batch_id)) \
        .withColumn("_fraud_load_timestamp",   F.current_timestamp())


def drop_bronze_audit(sdf: DataFrame) -> DataFrame:
    """Drop Bronze audit columns before writing to Silver."""
    cols_to_drop = [
        "_ingestion_timestamp", "_source_system",
        "_record_hash", "_batch_id",
        "_bronze_load_timestamp", "_bronze_batch_id", "_bronze_domain"
    ]
    existing = [c for c in cols_to_drop if c in sdf.columns]
    return sdf.drop(*existing)


# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────

def deduplicate(
    sdf: DataFrame,
    id_col: str,
    order_col: str = "_ingestion_timestamp"
) -> DataFrame:
    """
    Deduplicate DataFrame keeping latest record per ID.
    Uses window function — production-safe for large datasets.
    """
    window = Window.partitionBy(id_col).orderBy(F.col(order_col).desc())
    return sdf \
        .withColumn("_row_num", F.row_number().over(window)) \
        .filter(F.col("_row_num") == 1) \
        .drop("_row_num")


# ─────────────────────────────────────────────
# TYPE CASTING
# ─────────────────────────────────────────────

def safe_date(col_name: str):
    """Safely cast string to date. Bad values become NULL."""
    return F.to_date(F.col(col_name), "yyyy-MM-dd")


def safe_timestamp(col_name: str):
    """Safely cast string to timestamp. Bad values become NULL."""
    return F.to_timestamp(F.col(col_name))


def standardise_string(col_name: str):
    """Trim and uppercase categorical string columns."""
    return F.upper(F.trim(F.col(col_name)))


# ─────────────────────────────────────────────
# DATA QUALITY
# ─────────────────────────────────────────────

def apply_dq_rules(
    sdf: DataFrame,
    domain: str,
    rules: List[Tuple[str, str]],
    batch_id: str
):
    """
    Apply declarative DQ rules and split into good/rejected.
    Uses NULL-based concat_ws pattern — avoids empty string collision.

    Args:
        sdf: Input DataFrame
        domain: Domain name for logging
        rules: List of (sql_expression, error_code) tuples
        batch_id: Current batch ID for rejected record tracking

    Returns:
        Tuple of (good_sdf, bad_sdf)
    """
    if not rules:
        return sdf, None

    sdf_tagged = sdf.withColumn(
        "_dq_errors",
        F.concat_ws("|", *[
            F.when(F.expr(f"NOT ({rule})"), F.lit(code))
             .otherwise(F.lit(None).cast("string"))
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
    ) \
    .withColumn("_rejected_at",    F.current_timestamp()) \
    .withColumn("_rejected_batch", F.lit(batch_id))

    good_count = good_sdf.count()
    bad_count  = bad_sdf.count()
    total      = good_count + bad_count
    pass_rate  = (good_count / total * 100) if total > 0 else 0

    log.info(
        f"DQ [{domain}] Good: {good_count:,} | "
        f"Rejected: {bad_count:,} | "
        f"Pass Rate: {pass_rate:.2f}%"
    )

    return good_sdf, bad_sdf


# ─────────────────────────────────────────────
# DELTA WRITE
# ─────────────────────────────────────────────

def write_delta(
    sdf: DataFrame,
    db: str,
    table: str,
    partition_cols: Optional[List[str]] = None,
    mode: str = "overwrite"
) -> int:
    """
    Write DataFrame to Delta table.
    Returns row count for validation.
    """
    full_table = f"{db}.{table}"
    writer = sdf.write.format("delta").mode(mode)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    writer.saveAsTable(full_table)

    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    count = spark.table(full_table).count()
    log.info(f"Written {count:,} rows → {full_table}")
    return count


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_layer(
    db: str,
    tables: List[str],
    batch_id: str,
    layer_name: str
):
    """Print validation report for a layer."""
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()

    print(f"\n{'='*50}")
    print(f"{layer_name.upper()} LAYER — VALIDATION REPORT")
    print(f"{'='*50}")

    total = 0
    for table in tables:
        try:
            count = spark.table(f"{db}.{table}").count()
            total += count
            print(f"  {table:<25} {count:>10,} rows ✅")
        except Exception as e:
            print(f"  {table:<25} ERROR: {e} ❌")

    print(f"\n  {'TOTAL':<25} {total:>10,} rows")
    print(f"  Batch ID: {batch_id}")
    print(f"{'='*50}")
    return total
