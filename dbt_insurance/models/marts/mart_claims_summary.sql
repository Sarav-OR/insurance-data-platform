-- marts/mart_claims_summary.sql
-- Business KPI model for claims performance reporting

with claims as (
    select * from {{ ref('stg_claims') }}
),

summary as (
    select
        policy_type,
        claim_type,
        claim_status,
        claim_severity,
        sum(total_claims)           as total_claims,
        sum(total_claimed_chf)      as total_claimed_chf,
        avg(avg_claim_chf)          as avg_claim_chf,
        avg(loss_ratio)             as avg_loss_ratio,
        avg(fraud_rate)             as avg_fraud_rate,
        sum(fraud_suspected_count)  as total_fraud_suspected,
        sum(high_value_count)       as total_high_value,
        avg(avg_settlement_ratio)   as avg_settlement_ratio,
        avg(avg_days_to_submit)     as avg_days_to_submit
    from claims
    group by
        policy_type,
        claim_type,
        claim_status,
        claim_severity
)

select * from summary