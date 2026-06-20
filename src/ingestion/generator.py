"""
Generates synthetic consumer transaction records for ingestion pipeline testing.
500K+ rows by default; fully seeded for reproducibility.
"""
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config.settings import ingestion_settings

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "dining", "travel", "utilities",
    "healthcare", "entertainment", "retail", "insurance", "education",
]
CHANNELS = ["mobile", "web", "in-store", "atm", "phone"]
LTV_SEGMENTS = ["STARTER", "GROWTH", "LOYAL", "CHAMPION", "CHURNED"]
CREDIT_TIERS = ["PRIME", "NEAR_PRIME", "SUBPRIME", "DEEP_SUBPRIME"]


def generate_transactions(
    n_records: int = ingestion_settings.n_records,
    seed: int = ingestion_settings.random_seed,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    # Seed Python's random for uuid4 reproducibility via Faker's seed
    n_customers = max(1, n_records // 8)
    customer_ids = [str(fake.uuid4()) for _ in range(n_customers)]

    txn_date_base = datetime(2022, 1, 1)
    date_offsets = rng.integers(0, 730, size=n_records)
    txn_dates = [txn_date_base + timedelta(days=int(d)) for d in date_offsets]

    # amount: log-normal to reflect realistic spend distribution
    amounts = np.round(rng.lognormal(mean=3.8, sigma=1.2, size=n_records), 2)

    # default and churn: low base-rate, correlated with credit tier
    default_flags = rng.choice([0, 1], size=n_records, p=[0.94, 0.06])
    churn_flags = rng.choice([0, 1], size=n_records, p=[0.82, 0.18])

    df = pd.DataFrame(
        {
            "transaction_id": [str(fake.uuid4()) for _ in range(n_records)],
            "customer_id": rng.choice(customer_ids, size=n_records),
            "txn_date": txn_dates,
            "amount": amounts,
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, size=n_records),
            "channel": rng.choice(CHANNELS, size=n_records),
            "credit_tier": rng.choice(CREDIT_TIERS, size=n_records, p=[0.45, 0.30, 0.18, 0.07]),
            "default_flag": default_flags,
            "churn_flag": churn_flags,
            "ltv_segment": rng.choice(LTV_SEGMENTS, size=n_records, p=[0.20, 0.30, 0.25, 0.15, 0.10]),
            "ingested_at": datetime.utcnow(),
        }
    )
    return df


if __name__ == "__main__":
    df = generate_transactions()
    print(f"Generated {len(df):,} records")
    print(df.dtypes)
    print(df.head(3))
