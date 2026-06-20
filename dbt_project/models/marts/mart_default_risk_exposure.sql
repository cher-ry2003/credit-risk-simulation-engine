{{
    config(
        materialized='table',
        tags=['marts', 'risk', 'default']
    )
}}

/*
  Default Risk Exposure mart.
  Computes per-customer rolling 90-day default rate and assigns a risk bucket.

  Risk bucket logic:
    CRITICAL  : 90d_default_rate >= 0.15  OR  lifetime_defaults >= 5
    HIGH      : 90d_default_rate >= 0.08  OR  credit_tier IN ('SUBPRIME','DEEP_SUBPRIME')
    MEDIUM    : 90d_default_rate >= 0.03
    LOW       : otherwise
*/

with transactions as (
    select * from {{ ref('stg_transactions') }}
),

customers as (
    select * from {{ ref('stg_customers') }}
),

rolling_defaults as (
    select
        t.customer_id,
        count(*) filter (where t.txn_date >= dateadd('day', -90, current_date()))
            as txns_last_90d,
        sum(t.default_flag) filter (where t.txn_date >= dateadd('day', -90, current_date()))
            as defaults_last_90d,
        sum(t.default_flag)
            as lifetime_defaults,
        max(t.txn_date)
            as most_recent_txn_date
    from transactions t
    group by t.customer_id
),

risk_scores as (
    select
        rd.customer_id,
        rd.txns_last_90d,
        rd.defaults_last_90d,
        rd.lifetime_defaults,
        rd.most_recent_txn_date,
        c.dominant_credit_tier,
        c.customer_tenure_days,
        c.total_spend,
        -- 90-day default rate; guard against zero-division
        case
            when rd.txns_last_90d = 0 then 0.0
            else round(rd.defaults_last_90d::float / rd.txns_last_90d, 4)
        end as rolling_90d_default_rate
    from rolling_defaults rd
    left join customers c using (customer_id)
),

bucketed as (
    select
        *,
        case
            when rolling_90d_default_rate >= 0.15 or lifetime_defaults >= 5
                then 'CRITICAL'
            when rolling_90d_default_rate >= 0.08
                or dominant_credit_tier in ('SUBPRIME', 'DEEP_SUBPRIME')
                then 'HIGH'
            when rolling_90d_default_rate >= 0.03
                then 'MEDIUM'
            else 'LOW'
        end as risk_bucket
    from risk_scores
)

select
    customer_id,
    dominant_credit_tier,
    txns_last_90d,
    defaults_last_90d,
    rolling_90d_default_rate,
    lifetime_defaults,
    risk_bucket,
    customer_tenure_days,
    total_spend,
    most_recent_txn_date,
    current_timestamp() as mart_updated_at
from bucketed
