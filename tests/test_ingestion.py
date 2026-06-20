"""Tests for the data ingestion pipeline (generator + Snowflake loader mock mode)."""
import os
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.generator import (
    CHANNELS,
    LTV_SEGMENTS,
    MERCHANT_CATEGORIES,
    generate_transactions,
)

REQUIRED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "txn_date",
    "amount",
    "merchant_category",
    "channel",
    "credit_tier",
    "default_flag",
    "churn_flag",
    "ltv_segment",
    "ingested_at",
}


class TestTransactionGenerator:
    def test_default_record_count(self):
        df = generate_transactions(n_records=500_000, seed=42)
        assert len(df) >= 500_000, f"Expected >= 500,000 rows; got {len(df)}"

    def test_custom_record_count(self):
        df = generate_transactions(n_records=1_000, seed=0)
        assert len(df) == 1_000

    def test_required_columns_present(self):
        df = generate_transactions(n_records=100, seed=1)
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_null_transaction_id(self):
        df = generate_transactions(n_records=1_000, seed=2)
        assert df["transaction_id"].isna().sum() == 0

    def test_no_null_customer_id(self):
        df = generate_transactions(n_records=1_000, seed=3)
        assert df["customer_id"].isna().sum() == 0

    def test_transaction_id_uniqueness(self):
        df = generate_transactions(n_records=5_000, seed=4)
        assert df["transaction_id"].nunique() == len(df), "Duplicate transaction_ids detected"

    def test_amount_positive(self):
        df = generate_transactions(n_records=1_000, seed=5)
        assert (df["amount"] > 0).all()

    def test_default_flag_binary(self):
        df = generate_transactions(n_records=1_000, seed=6)
        assert set(df["default_flag"].unique()).issubset({0, 1})

    def test_churn_flag_binary(self):
        df = generate_transactions(n_records=1_000, seed=7)
        assert set(df["churn_flag"].unique()).issubset({0, 1})

    def test_merchant_category_values(self):
        df = generate_transactions(n_records=5_000, seed=8)
        assert set(df["merchant_category"].unique()).issubset(set(MERCHANT_CATEGORIES))

    def test_channel_values(self):
        df = generate_transactions(n_records=5_000, seed=9)
        assert set(df["channel"].unique()).issubset(set(CHANNELS))

    def test_ltv_segment_values(self):
        df = generate_transactions(n_records=5_000, seed=10)
        assert set(df["ltv_segment"].unique()).issubset(set(LTV_SEGMENTS))

    def test_reproducibility(self):
        df1 = generate_transactions(n_records=500, seed=99)
        df2 = generate_transactions(n_records=500, seed=99)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


class TestSnowflakeLoaderMockMode:
    def test_mock_writes_parquet(self, tmp_path, monkeypatch):
        from src.config.settings import SnowflakeSettings
        from src.ingestion import snowflake_loader as loader

        mock_path = tmp_path / "landing.parquet"
        monkeypatch.setattr(loader, "LOCAL_MOCK_PATH", mock_path)

        cfg = SnowflakeSettings(mock=True)
        df = generate_transactions(n_records=100, seed=0)
        rows = loader.load_dataframe(df, cfg=cfg)

        assert rows == 100
        assert mock_path.exists()
        written = pd.read_parquet(mock_path)
        assert len(written) == 100

    def test_mock_preserves_schema(self, tmp_path, monkeypatch):
        from src.config.settings import SnowflakeSettings
        from src.ingestion import snowflake_loader as loader

        mock_path = tmp_path / "landing2.parquet"
        monkeypatch.setattr(loader, "LOCAL_MOCK_PATH", mock_path)

        cfg = SnowflakeSettings(mock=True)
        df = generate_transactions(n_records=50, seed=11)
        loader.load_dataframe(df, cfg=cfg)

        written = pd.read_parquet(mock_path)
        assert set(REQUIRED_COLUMNS).issubset(set(written.columns))
