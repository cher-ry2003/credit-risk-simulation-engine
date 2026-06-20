from pydantic_settings import BaseSettings, SettingsConfigDict


class SnowflakeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    account: str = "mock_account.us-east-1"
    user: str = "mock_user"
    password: str = "mock_password"
    warehouse: str = "COMPUTE_WH"
    database: str = "CONSUMER_RISK_DB"
    schema_name: str = "RAW"
    role: str = "SYSADMIN"
    mock: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SNOWFLAKE_",
        extra="ignore",
    )


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    n_records: int = 500_000
    random_seed: int = 42
    chunk_size: int = 10_000


snowflake_settings = SnowflakeSettings()
ingestion_settings = IngestionSettings()
