{{
    config(
        materialized='table',
        tags=['marts', 'churn', 'rfm']
    )
}}

/*
  Churn Probability mart using Recency-Frequency-Monetary (RFM) scoring.

  Each dimension is scored 1–5 (5 = best):
    Recency   : days since last transaction (lower = better → higher score)
    Frequency : total transaction count (higher = better)
    Monetary  : total spend (higher = better)

  Composite RFM score = recency_score + frequency_score + monetary_score (3–15)
  Churn risk quintile (1=lowest churn risk, 5=highest) is the inverse of RFM score.

  Churn risk labels:
    quintile 1 → VERY_LOW
    quintile 2 → LOW
    quintile 3 → MEDIUM
    quintile 4 → HIGH
    quintile 5 → VERY_HIGH
*/

with customers as (
    select * from {{ ref('stg_customers') }}
),

rfm_raw as (
    select
        customer_id,
        datediff('day', last_txn_date, current_date()) as days_since_last_txn,
        total_transactions                              as frequency,
        total_spend                                     as monetary,
        ever_churned,
        current_ltv_segment
    from customers
),

rfm_scored as (
    select
        *,
        -- Recency: lower days = better (score 5); quintile reversed
        ntile(5) over (order by days_since_last_txn asc)   as recency_raw_ntile,
        ntile(5) over (order by frequency desc)             as frequency_score,
        ntile(5) over (order by monetary desc)              as monetary_score
    from rfm_raw
),

rfm_final as (
    select
        *,
        -- Flip recency so 5 = most recent
        (6 - recency_raw_ntile)                             as recency_score,
        (6 - recency_raw_ntile) + frequency_score + monetary_score
                                                            as rfm_composite_score
    from rfm_scored
),

churn_bucketed as (
    select
        *,
        -- quintile over composite score: low RFM = high churn risk
        ntile(5) over (order by rfm_composite_score desc)   as churn_risk_quintile
    from rfm_final
)

select
    customer_id,
    days_since_last_txn,
    frequency,
    round(monetary, 2)                      as monetary,
    recency_score,
    frequency_score,
    monetary_score,
    rfm_composite_score,
    churn_risk_quintile,
    case churn_risk_quintile
        when 1 then 'VERY_LOW'
        when 2 then 'LOW'
        when 3 then 'MEDIUM'
        when 4 then 'HIGH'
        when 5 then 'VERY_HIGH'
    end                                     as churn_risk_label,
    ever_churned,
    current_ltv_segment,
    current_timestamp()                     as mart_updated_at
from churn_bucketed
