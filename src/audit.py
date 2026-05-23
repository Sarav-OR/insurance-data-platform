"""
src/audit.py
============
Audit Column Functions for Insurance Data Platform
===================================================
Purpose : Add layer-specific audit columns to every DataFrame.
          Every record at every layer carries a full audit trail
          answering: when was this loaded, by which batch,
          from which layer.

Usage   : from src.audit import add_bronze_audit, add_silver_audit

Audit trail per record:
  Bronze → _bronze_batch_id, _bronze_load_timestamp, _bronze_domain
  Silver → _silver_batch_id, _silver_load_timestamp
  Gold   → _gold_batch_id,   _gold_load_timestamp
  Fraud  → _fraud_batch_id,  _fraud_load_timestamp
"""

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame


def add_bronze_audit(sdf: DataFrame,
                     domain: str,
                     batch_id: str) -> DataFrame:
    """
    Add Bronze layer audit columns to DataFrame.

    Columns added:
      _bronze_batch_id       : Unique ID of this pipeline run
      _bronze_load_timestamp : When Bronze ingested this record
      _bronze_domain         : Which domain (claims/policies etc)

    These enable queries like:
      "Show me all claims loaded in batch 20240115_060000"
      "Which domain had issues in yesterday's run?"

    Args:
        sdf     : Input DataFrame
        domain  : Domain name e.g. 'claims'
        batch_id: Current pipeline batch ID

    Returns:
        DataFrame with Bronze audit columns added
    """
    return sdf \
        .withColumn("_bronze_batch_id",
                    F.lit(batch_id)) \
        .withColumn("_bronze_load_timestamp",
                    F.current_timestamp()) \
        .withColumn("_bronze_domain",
                    F.lit(domain))


def add_silver_audit(sdf: DataFrame,
                     batch_id: str) -> DataFrame:
    """
    Add Silver layer audit columns to DataFrame.

    Columns added:
      _silver_batch_id       : Unique ID of Silver pipeline run
      _silver_load_timestamp : When Silver processed this record
      _silver_source_layer   : Always 'bronze' — lineage tracking

    Args:
        sdf     : Input DataFrame
        batch_id: Current pipeline batch ID

    Returns:
        DataFrame with Silver audit columns added
    """
    return sdf \
        .withColumn("_silver_batch_id",
                    F.lit(batch_id)) \
        .withColumn("_silver_load_timestamp",
                    F.current_timestamp()) \
        .withColumn("_silver_source_layer",
                    F.lit("bronze"))


def add_gold_audit(sdf: DataFrame,
                   batch_id: str) -> DataFrame:
    """
    Add Gold layer audit columns to DataFrame.

    Columns added:
      _gold_batch_id       : Unique ID of Gold pipeline run
      _gold_load_timestamp : When Gold aggregated this record
      _gold_source_layer   : Always 'silver' — lineage tracking

    Args:
        sdf     : Input DataFrame
        batch_id: Current pipeline batch ID

    Returns:
        DataFrame with Gold audit columns added
    """
    return sdf \
        .withColumn("_gold_batch_id",
                    F.lit(batch_id)) \
        .withColumn("_gold_load_timestamp",
                    F.current_timestamp()) \
        .withColumn("_gold_source_layer",
                    F.lit("silver"))


def add_fraud_audit(sdf: DataFrame,
                    batch_id: str) -> DataFrame:
    """
    Add Fraud layer audit columns to DataFrame.

    Args:
        sdf     : Input DataFrame
        batch_id: Current pipeline batch ID

    Returns:
        DataFrame with Fraud audit columns added
    """
    return sdf \
        .withColumn("_fraud_batch_id",
                    F.lit(batch_id)) \
        .withColumn("_fraud_load_timestamp",
                    F.current_timestamp())
