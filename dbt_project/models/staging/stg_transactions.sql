{{
    config(
        materialized='view',
        tags=['staging', 'transactions']
    )
}}

with source as (
    select * from {{ source('raw', 'consumer_transactions') }}
),

renamed as (
    select
        lower(trim(TRANSACTION_ID))                   as transaction_id,
        lower(trim(CUSTOMER_ID))                      as customer_id,
        cast(TXN_DATE as date)                        as txn_date,
        cast(AMOUNT as numeric(18, 2))                as amount,
        lower(trim(MERCHANT_CATEGORY))                as merchant_category,
        lower(trim(CHANNEL))                          as channel,
        upper(trim(CREDIT_TIER))                      as credit_tier,
        cast(coalesce(DEFAULT_FLAG, 0) as integer)    as default_flag,
        cast(coalesce(CHURN_FLAG, 0) as integer)      as churn_flag,
        upper(trim(LTV_SEGMENT))                      as ltv_segment,
        cast(INGESTED_AT as timestamp_ntz)            as ingested_at,
        row_number() over (
            partition by lower(trim(TRANSACTION_ID))
            order by cast(INGESTED_AT as timestamp_ntz) desc
        )                                             as _dedup_rank
    from source
),

deduplicated as (
    select
        transaction_id,
        customer_id,
        txn_date,
        amount,
        merchant_category,
        channel,
        credit_tier,
        default_flag,
        churn_flag,
        ltv_segment,
        ingested_at
    from renamed
    where _dedup_rank = 1
)

select * from deduplicated
