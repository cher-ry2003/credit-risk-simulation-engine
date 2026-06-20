"""
Snowflake landing layer loader.
When SNOWFLAKE_MOCK=true (default), writes a local Parquet file instead of
connecting to Snowflake, allowing fully offline development and CI.
"""
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pandas as pd

from src.config.settings import SnowflakeSettings, ingestion_settings

logger = logging.getLogger(__name__)

LANDING_TABLE = "CONSUMER_TRANSACTIONS"
LOCAL_MOCK_PATH = Path("data/landing_consumer_transactions.parquet")

CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {{database}}.{{schema}}.{LANDING_TABLE} (
    TRANSACTION_ID   VARCHAR(36)    NOT NULL,
    CUSTOMER_ID      VARCHAR(36)    NOT NULL,
    TXN_DATE         DATE           NOT NULL,
    AMOUNT           FLOAT          NOT NULL,
    MERCHANT_CATEGORY VARCHAR(64),
    CHANNEL          VARCHAR(32),
    CREDIT_TIER      VARCHAR(32),
    DEFAULT_FLAG     NUMBER(1),
    CHURN_FLAG       NUMBER(1),
    LTV_SEGMENT      VARCHAR(32),
    INGESTED_AT      TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (TRANSACTION_ID)
)
"""


@contextmanager
def _snowflake_connection(cfg: SnowflakeSettings) -> Generator:
    import snowflake.connector  # deferred; not installed in mock mode

    conn = snowflake.connector.connect(
        account=cfg.account,
        user=cfg.user,
        password=cfg.password,
        warehouse=cfg.warehouse,
        database=cfg.database,
        schema=cfg.schema_name,
        role=cfg.role,
        session_parameters={"QUERY_TAG": "consumer_risk_engine_ingestion"},
    )
    try:
        yield conn
    finally:
        conn.close()


def create_landing_table(cfg: SnowflakeSettings) -> None:
    ddl = CREATE_TABLE_DDL.format(database=cfg.database, schema=cfg.schema_name)
    with _snowflake_connection(cfg) as conn:
        conn.cursor().execute(ddl)
    logger.info("Landing table %s.%s.%s ensured.", cfg.database, cfg.schema_name, LANDING_TABLE)


def load_dataframe(df: pd.DataFrame, cfg: SnowflakeSettings | None = None) -> int:
    if cfg is None:
        cfg = SnowflakeSettings()

    if cfg.mock:
        LOCAL_MOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(LOCAL_MOCK_PATH, index=False)
        logger.info("[MOCK] Wrote %d rows to %s", len(df), LOCAL_MOCK_PATH)
        return len(df)

    from snowflake.connector.pandas_tools import write_pandas

    create_landing_table(cfg)
    total_loaded = 0
    chunk_size = ingestion_settings.chunk_size

    with _snowflake_connection(cfg) as conn:
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]
            success, nchunks, nrows, _ = write_pandas(
                conn,
                chunk,
                table_name=LANDING_TABLE,
                database=cfg.database,
                schema=cfg.schema_name,
                auto_create_table=False,
                overwrite=False,
            )
            if not success:
                raise RuntimeError(f"write_pandas failed at chunk starting row {start}")
            total_loaded += nrows
            logger.info("Loaded chunk rows %d–%d (%d rows committed)", start, start + len(chunk), nrows)

    logger.info("Total rows loaded to Snowflake: %d", total_loaded)
    return total_loaded
