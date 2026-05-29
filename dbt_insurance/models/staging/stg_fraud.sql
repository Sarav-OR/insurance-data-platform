-- staging/stg_fraud.sql
with source as (
    select * from adb_insurance_platform.insurance_gold.fraud_summary
),

renamed as (
    select
        signal_type,
        score_band,
        policy_type,
        claim_type,
        total_signals,
        avg_signal_score,
        confirmed_fraud_count,
        needs_review_count,
        reviewed_count,
        total_fraud_exposure_chf,
        avg_fraud_claim_chf,
        confirmation_rate_pct,
        review_completion_rate_pct,
        _gold_batch_id,
        _gold_load_timestamp,
        _gold_source_layer
    from source
)

select * from renamed