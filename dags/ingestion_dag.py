"""
Airflow DAG: consumer_transaction_ingestion
Schedule: @daily

Pipeline:
  1. generate_mock_data   — synthesise 500K+ transaction records, write temp Parquet
  2. validate_record_count — assert row count >= 500_000, fail fast otherwise
  3. load_to_snowflake     — push Parquet to Snowflake landing (or local mock)
  4. trigger_dbt_run       — run dbt staging models downstream

Requires: AIRFLOW_CONN_SNOWFLAKE_DEFAULT set in Airflow connections (or SNOWFLAKE_MOCK=true).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

logger = logging.getLogger(__name__)

TEMP_PARQUET_PATH = "/tmp/consumer_transactions_raw.parquet"
MIN_RECORDS = 500_000
DBT_PROJECT_DIR = os.getenv("DBT_PROJECT_DIR", "/opt/airflow/dbt_project")
DBT_PROFILES_DIR = os.getenv("DBT_PROFILES_DIR", "/opt/airflow/dbt_project")


default_args = {
    "owner": "risk-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="consumer_transaction_ingestion",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "risk", "snowflake"],
    doc_md=__doc__,
)
def consumer_transaction_ingestion():

    @task()
    def generate_mock_data() -> str:
        from src.ingestion.generator import generate_transactions

        df = generate_transactions()
        df.to_parquet(TEMP_PARQUET_PATH, index=False)
        logger.info("Generated %d records → %s", len(df), TEMP_PARQUET_PATH)
        return TEMP_PARQUET_PATH

    @task()
    def validate_record_count(parquet_path: str) -> str:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        row_count = len(df)
        if row_count < MIN_RECORDS:
            raise ValueError(
                f"Record count validation FAILED: got {row_count:,}, expected >= {MIN_RECORDS:,}"
            )
        logger.info("Record count OK: %d rows (>= %d)", row_count, MIN_RECORDS)
        return parquet_path

    @task()
    def load_to_snowflake(parquet_path: str) -> int:
        import pandas as pd

        from src.ingestion.snowflake_loader import load_dataframe

        df = pd.read_parquet(parquet_path)
        rows_loaded = load_dataframe(df)
        logger.info("Snowflake load complete: %d rows", rows_loaded)
        return rows_loaded

    trigger_dbt_run = BashOperator(
        task_id="trigger_dbt_run",
        bash_command=(
            f"dbt run --select staging "
            f"--project-dir {DBT_PROJECT_DIR} "
            f"--profiles-dir {DBT_PROFILES_DIR}"
        ),
        env={**os.environ, "DBT_TARGET": "dev"},
    )

    parquet_path = generate_mock_data()
    validated_path = validate_record_count(parquet_path)
    load_result = load_to_snowflake(validated_path)
    load_result >> trigger_dbt_run


consumer_transaction_ingestion()
