{{
    config(
        materialized='view',
        tags=['staging', 'customers']
    )
}}

/*
  Derives a customer dimension at the customer_id grain from stg_transactions.
  first_txn_date is the acquisition proxy; dominant_credit_tier is mode-based.
*/

with txns as (
    select * from {{ ref('stg_transactions') }}
),

customer_agg as (
    select
        customer_id,
        min(txn_date)                                   as first_txn_date,
        max(txn_date)                                   as last_txn_date,
        count(*)                                        as total_transactions,
        sum(amount)                                     as total_spend,
        avg(amount)                                     as avg_transaction_amount,
        -- mode credit tier: highest-frequency tier per customer
        mode(credit_tier)                               as dominant_credit_tier,
        max_by(ltv_segment, ingested_at)                as current_ltv_segment,
        sum(default_flag)                               as lifetime_defaults,
        max(churn_flag)                                 as ever_churned,
        min(ingested_at)                                as first_ingested_at
    from txns
    group by customer_id
)

select
    customer_id,
    first_txn_date,
    last_txn_date,
    total_transactions,
    round(total_spend, 2)                               as total_spend,
    round(avg_transaction_amount, 2)                    as avg_transaction_amount,
    dominant_credit_tier,
    current_ltv_segment,
    lifetime_defaults,
    cast(ever_churned as integer)                       as ever_churned,
    datediff('day', first_txn_date, last_txn_date)      as customer_tenure_days,
    first_ingested_at
from customer_agg
