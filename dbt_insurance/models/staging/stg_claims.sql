-- staging/stg_claims.sql
with source as (
    select * from adb_insurance_platform.insurance_gold.claims_kpis
),

renamed as (
    select
        policy_type,
        claim_status,
        claim_severity,
        claim_type,
        total_claims,
        total_claimed_chf,
        avg_claim_chf,
        max_claim_chf,
        min_claim_chf,
        total_settled_chf,
        avg_settlement_ratio,
        avg_days_to_submit,
        avg_claim_age_days,
        fraud_suspected_count,
        high_value_count,
        third_party_count,
        total_premium_chf,
        loss_ratio,
        fraud_rate,
        _gold_batch_id,
        _gold_load_timestamp,
        _gold_source_layer
    from source
)

select * from renamed