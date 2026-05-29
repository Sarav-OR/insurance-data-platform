-- marts/mart_fraud_summary.sql
-- Business KPI model for fraud analytics reporting

with fraud as (
    select * from {{ ref('stg_fraud') }}
),

summary as (
    select
        signal_type,
        score_band,
        policy_type,
        claim_type,
        sum(total_signals)              as total_signals,
        avg(avg_signal_score)           as avg_signal_score,
        sum(confirmed_fraud_count)      as confirmed_fraud_count,
        sum(needs_review_count)         as needs_review_count,
        sum(reviewed_count)             as reviewed_count,
        sum(total_fraud_exposure_chf)   as total_fraud_exposure_chf,
        avg(avg_fraud_claim_chf)        as avg_fraud_claim_chf,
        avg(confirmation_rate_pct)      as avg_confirmation_rate_pct,
        avg(review_completion_rate_pct) as avg_review_completion_rate_pct
    from fraud
    group by
        signal_type,
        score_band,
        policy_type,
        claim_type
)

select * from summary