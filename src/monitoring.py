"""
src/monitoring.py
=================
Monitoring and Error Handling for Insurance Data Platform
=========================================================
Purpose : Centralised functions for:
          - DQ pass rate tracking over time
          - Pipeline error logging
          - Volume anomaly detection
          - Row count reconciliation between layers

Usage   : from src.monitoring import (
              apply_dq_rules,
              write_dq_monitoring,
              check_volume,
              log_pipeline_error,
              reconcile_row_counts
          )

Monitoring tables written to insurance_bronze database:
  dq_monitoring   — DQ pass rates per domain per batch
  pipeline_errors — Pipeline failures with full details

Why centralise here:
  Every layer (Bronze/Silver/Gold) uses the same monitoring.
  One place to improve alerting, add Slack/email notifications,
  or integrate with external monitoring tools.
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.dataframe import DataFrame

log = logging.getLogger("insurance_monitoring")

# Monitoring database — always write to Bronze
# so monitoring works even if Silver/Gold fails
MONITORING_DB = "insurance_bronze"


# ─────────────────────────────────────────────────────────
# DQ RULE ENGINE
# ─────────────────────────────────────────────────────────

def apply_dq_rules(sdf: DataFrame,
                   domain: str,
                   rules: list,
                   batch_id: str) -> Tuple[DataFrame,
                                           DataFrame,
                                           int, int]:
    """
    Apply declarative DQ rules and split good/bad records.

    How it works:
    1. For each rule: tag record with error code if rule fails
    2. concat_ws joins all error codes with | separator
    3. Using NULL (not "") as fallback — concat_ws ignores NULLs
       so passing records get "" not "||||"
    4. Good records: _dq_errors is NULL or ""
    5. Bad records:  _dq_errors has at least one error code

    Production principle:
    NEVER silently drop bad data.
    Every rejected record is preserved with its error code(s).
    Source team can investigate and resend.

    Args:
        sdf     : Input DataFrame to validate
        domain  : Domain name for logging
        rules   : List of (sql_expr, error_code) tuples
        batch_id: Current pipeline batch ID

    Returns:
        Tuple of (good_sdf, bad_sdf, good_count, bad_count)
    """
    if not rules:
        log.warning(f"No DQ rules defined for '{domain}'")
        count = sdf.count()
        return sdf, None, count, 0

    # Tag each record with triggered error codes
    sdf_tagged = sdf.withColumn(
        "_dq_errors",
        F.concat_ws("|", *[
            F.when(F.expr(f"NOT ({rule})"), F.lit(code))
             .otherwise(F.lit(None).cast("string"))
            for rule, code in rules
        ])
    )

    # Good records — all rules passed
    good_sdf = sdf_tagged.filter(
        (F.col("_dq_errors").isNull()) |
        (F.col("_dq_errors") == "")
    ).drop("_dq_errors")

    # Bad records — at least one rule failed
    bad_sdf = sdf_tagged.filter(
        F.col("_dq_errors").isNotNull() &
        (F.col("_dq_errors") != "")
    ) \
    .withColumn("_rejected_at",    F.current_timestamp()) \
    .withColumn("_rejected_batch", F.lit(batch_id)) \
    .withColumn("_domain",         F.lit(domain))

    good_count = good_sdf.count()
    bad_count  = bad_sdf.count()
    total      = good_count + bad_count
    pass_rate  = round(good_count / total * 100, 2) \
                 if total > 0 else 100.0

    log.info(
        f"DQ [{domain}] "
        f"Good: {good_count:,} | "
        f"Rejected: {bad_count:,} | "
        f"Pass Rate: {pass_rate}%"
    )

    return good_sdf, bad_sdf, good_count, bad_count


# ─────────────────────────────────────────────────────────
# DQ MONITORING
# ─────────────────────────────────────────────────────────

def write_dq_monitoring(spark,
                        domain: str,
                        good_count: int,
                        bad_count: int,
                        batch_id: str) -> None:
    """
    Append DQ statistics to monitoring table.

    Why this matters:
    Over time you can query this table to answer:
    - Is data quality improving or degrading?
    - Which domain has the most rejections?
    - Did a source system change break our rules?
    - What was the pass rate 30 days ago vs today?

    Args:
        spark      : Active SparkSession
        domain     : Domain name e.g. 'claims'
        good_count : Number of records that passed DQ
        bad_count  : Number of records that failed DQ
        batch_id   : Current pipeline batch ID
    """
    total     = good_count + bad_count
    pass_rate = round(good_count / total * 100, 2) \
                if total > 0 else 100.0

    record = spark.createDataFrame([{
        "batch_id":       batch_id,
        "domain":         domain,
        "good_count":     good_count,
        "bad_count":      bad_count,
        "total_count":    total,
        "pass_rate_pct":  pass_rate,
        "load_timestamp": datetime.utcnow(),
    }])

    record.write.format("delta").mode("append") \
          .saveAsTable(f"{MONITORING_DB}.dq_monitoring")

    log.info(
        f"DQ monitoring [{domain}] written: "
        f"pass_rate={pass_rate}%"
    )


# ─────────────────────────────────────────────────────────
# VOLUME ANOMALY DETECTION
# ─────────────────────────────────────────────────────────

def check_volume(spark,
                 domain: str,
                 actual_count: int,
                 thresholds: dict,
                 batch_id: str) -> bool:
    """
    Detect unexpected record volumes.

    Catches these real-world issues:
    - Source system sent 0 records (outage or connection issue)
    - Source system sent 10x normal (duplicate file send)
    - Only partial file received (network timeout)

    Args:
        spark        : Active SparkSession
        domain       : Domain name
        actual_count : Actual record count received
        thresholds   : Dict with 'min' and 'max' keys
        batch_id     : Current pipeline batch ID

    Returns:
        True if volume is normal, False if anomaly detected
    """
    if not thresholds:
        return True

    min_exp = thresholds.get("min", 0)
    max_exp = thresholds.get("max", float("inf"))

    if actual_count < min_exp:
        msg = (
            f"VOLUME ANOMALY [{domain}]: "
            f"Received {actual_count:,} rows. "
            f"Expected minimum: {min_exp:,}. "
            f"Possible: source outage or data loss."
        )
        log.error(msg)
        log_pipeline_error(
            spark, domain,
            ValueError(msg), batch_id,
            status="VOLUME_ANOMALY"
        )
        return False

    elif actual_count > max_exp:
        msg = (
            f"VOLUME WARNING [{domain}]: "
            f"Received {actual_count:,} rows. "
            f"Expected maximum: {max_exp:,}. "
            f"Possible: duplicate file send."
        )
        log.warning(msg)
        return False

    else:
        log.info(
            f"Volume check [{domain}]: "
            f"{actual_count:,} rows ✅ "
            f"(expected {min_exp:,}–{max_exp:,})"
        )
        return True


# ─────────────────────────────────────────────────────────
# PIPELINE ERROR LOGGING
# ─────────────────────────────────────────────────────────

def log_pipeline_error(spark,
                       domain: str,
                       error: Exception,
                       batch_id: str,
                       status: str = "FAILED") -> None:
    """
    Write pipeline failure details to error tracking table.

    Used for:
    - Post-mortem analysis after pipeline failures
    - Alerting: query this table for failures in last 24h
    - SLA reporting: how many failures per month?

    In production this function would also:
    - Send Slack/Teams notification
    - Create PagerDuty incident
    - Update monitoring dashboard

    Args:
        spark    : Active SparkSession
        domain   : Domain where error occurred
        error    : Exception that was raised
        batch_id : Current pipeline batch ID
        status   : Error classification (FAILED/VOLUME_ANOMALY)
    """
    try:
        spark.createDataFrame([{
            "batch_id":      batch_id,
            "domain":        domain,
            "error_message": str(error)[:1000],
            "error_type":    type(error).__name__,
            "failed_at":     datetime.utcnow(),
            "status":        status,
        }]).write.format("delta").mode("append") \
           .saveAsTable(
               f"{MONITORING_DB}.pipeline_errors"
           )
        log.info(
            f"Error logged to pipeline_errors: "
            f"[{domain}] {type(error).__name__}"
        )
    except Exception as inner_e:
        # NEVER let error logging crash the main pipeline
        # Log locally and continue
        log.error(
            f"Could not write to pipeline_errors: {inner_e}"
        )


# ─────────────────────────────────────────────────────────
# ROW COUNT RECONCILIATION
# ─────────────────────────────────────────────────────────

def reconcile_row_counts(spark,
                         source_db: str,
                         source_table: str,
                         target_db: str,
                         target_table: str,
                         tolerance_pct: float = 1.0) -> bool:
    """
    Validate row counts between source and target tables.

    Used between layers to confirm no unexpected data loss:
    - Bronze claims count should match Silver claims count
    - Silver customer count should match Gold customer count

    A small tolerance is allowed (default 1%) for records
    that fail Silver DQ rules and are filtered out.

    Args:
        spark         : Active SparkSession
        source_db     : Source database e.g. 'insurance_bronze'
        source_table  : Source table e.g. 'claims'
        target_db     : Target database e.g. 'insurance_silver'
        target_table  : Target table e.g. 'claims'
        tolerance_pct : Max allowed % difference (default 1%)

    Returns:
        True if counts are within tolerance, False otherwise
    """
    source_count = spark.table(
        f"{source_db}.{source_table}"
    ).count()
    target_count = spark.table(
        f"{target_db}.{target_table}"
    ).count()

    if source_count == 0:
        log.warning("Source table is empty — skipping reconciliation")
        return False

    diff_pct = abs(source_count - target_count) \
               / source_count * 100

    if diff_pct > tolerance_pct:
        log.warning(
            f"Row count mismatch: "
            f"{source_db}.{source_table}={source_count:,} vs "
            f"{target_db}.{target_table}={target_count:,} "
            f"({diff_pct:.2f}% difference, "
            f"tolerance={tolerance_pct}%)"
        )
        return False

    log.info(
        f"Row count reconciliation ✅ "
        f"{source_table}: {source_count:,} → "
        f"{target_table}: {target_count:,} "
        f"({diff_pct:.2f}% difference)"
    )
    return True
