"""Application configuration loaded from environment variables."""

from functools import lru_cache
import re
import textwrap

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API service."""

    app_env: str = "development"
    app_name: str = "ARIS Platform API"
    app_version: str = "1.0.0"

    database_url: str = "sqlite+aiosqlite:///./aris_platform.db"
    database_echo: bool = False

    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    celery_task_always_eager: bool = False

    s3_bucket_name: str = "aris-documents"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_presign_expire_seconds: int = 900
    local_object_store_dir: str = ".aris_object_store"
    local_object_store_base_url: str = "http://127.0.0.1:8000"

    jwt_private_key: str = ""
    jwt_public_key: str = ""
    jwt_algorithm: str = "RS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 30
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    @classmethod
    def _normalize_pem_value(cls, value: str, kind: str) -> str:
        """Normalize PEM values loaded from env files and compose interpolation."""

        normalized = value.strip().strip('"').strip("'")
        normalized = normalized.replace("\\n", "\n").replace("\r\n", "\n")

        match = re.search(
            r"-----BEGIN [A-Z ]+-----[\s\S]*?-----END [A-Z ]+-----",
            normalized,
        )
        if match is not None:
            return match.group(0).strip()

        # Fallback: recover raw key bodies where accidental prefixes exist.
        start = normalized.find("MII")
        if start != -1:
            body = re.sub(r"\s+", "", normalized[start:])
            wrapped = "\n".join(textwrap.wrap(body, 64))
            return f"-----BEGIN {kind} KEY-----\n{wrapped}\n-----END {kind} KEY-----"

        return normalized

    @field_validator("jwt_private_key", mode="before")
    @classmethod
    def normalize_private_pem(cls, value: str) -> str:
        if isinstance(value, str):
            return cls._normalize_pem_value(value, "PRIVATE")
        return value

    @field_validator("jwt_public_key", mode="before")
    @classmethod
    def normalize_public_pem(cls, value: str) -> str:
        if isinstance(value, str):
            return cls._normalize_pem_value(value, "PUBLIC")
        return value

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings object for dependency injection."""

    return Settings()
