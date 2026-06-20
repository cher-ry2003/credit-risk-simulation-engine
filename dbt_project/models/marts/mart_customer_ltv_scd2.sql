{{
    config(
        materialized='table',
        tags=['marts', 'ltv', 'scd2']
    )
}}

/*
  Customer LTV Lifecycle — Slowly Changing Dimension Type 2 (SCD Type 2) pattern.

  This mart simulates the SCD2 output that a dbt snapshot would produce,
  implemented as a portable mart for scaffolding purposes.

  Each customer gets one "current" row reflecting their latest LTV segment,
  plus historical rows synthesised from the min/max transaction date window
  to demonstrate the SCD2 structure:

  Columns:
    surrogate_key   : MD5(customer_id || effective_date)  — unique row identifier
    customer_id     : natural key
    ltv_segment     : segment value valid during [effective_date, expiry_date)
    effective_date  : date segment became active
    expiry_date     : date segment was superseded (9999-12-31 for current row)
    is_current      : boolean flag — true for the active segment row

  Segment transition path (ascending lifecycle):
    STARTER → GROWTH → LOYAL → CHAMPION
    Any segment → CHURNED (terminal)
*/

with customers as (
    select * from {{ ref('stg_customers') }}
),

-- Simulate historical epoch windows based on tenure quartiles
tenure_epochs as (
    select
        customer_id,
        current_ltv_segment,
        first_txn_date,
        last_txn_date,
        customer_tenure_days,
        -- Divide tenure into up to 4 epochs of ~equal length
        case
            when customer_tenure_days <= 0 then 1
            else least(4, greatest(1, floor(customer_tenure_days / 90.0)))
        end as n_epochs
    from customers
),

-- Generate one row per epoch per customer (simulates historical SCD2 rows)
epoch_series as (
    select
        e.customer_id,
        e.first_txn_date,
        e.last_txn_date,
        e.customer_tenure_days,
        e.current_ltv_segment,
        e.n_epochs,
        seq.epoch_num,
        -- Effective date for this epoch
        dateadd(
            'day',
            floor((seq.epoch_num - 1) * e.customer_tenure_days / e.n_epochs),
            e.first_txn_date
        ) as effective_date,
        -- Expiry date = next epoch's start (or last_txn_date for current)
        case
            when seq.epoch_num < e.n_epochs
            then dateadd(
                    'day',
                    floor(seq.epoch_num * e.customer_tenure_days / e.n_epochs),
                    e.first_txn_date
                 )
            else to_date('9999-12-31')
        end as expiry_date,
        seq.epoch_num = e.n_epochs as is_current
    from tenure_epochs e
    -- Cross-join to a small integer sequence (1..4)
    join (
        select 1 as epoch_num union all
        select 2 union all
        select 3 union all
        select 4
    ) seq on seq.epoch_num <= e.n_epochs
),

-- Assign a plausible historical LTV segment per epoch
segment_assigned as (
    select
        *,
        case
            when is_current then current_ltv_segment
            when n_epochs = 1 then current_ltv_segment
            when epoch_num = 1 then 'STARTER'
            when epoch_num = 2 and n_epochs >= 3 then 'GROWTH'
            when epoch_num = 3 and n_epochs >= 4 then 'LOYAL'
            else current_ltv_segment
        end as ltv_segment
    from epoch_series
)

select
    md5(customer_id || '|' || cast(effective_date as varchar)) as surrogate_key,
    customer_id,
    ltv_segment,
    effective_date,
    expiry_date,
    is_current,
    current_timestamp() as dbt_updated_at
from segment_assigned
order by customer_id, effective_date
