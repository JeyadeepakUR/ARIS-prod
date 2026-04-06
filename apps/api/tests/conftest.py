"""Shared test fixtures for API auth tests."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apps.api.config import Settings
from apps.api.config import get_settings
from apps.api.core.database import Base
from apps.api.dependencies import get_config, get_db
from apps.api.main import create_app


TEST_JWT_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCvOszvsl8DZJur
QgRQXhRJ58AF3G7g37hivTGqpgq6/MBlrE0FldKtXkIS7mxxvwCSgDekS5IkMAjq
VnqLWjY0wlvPmwf+Tscc4PsrfBRwfX7DBpgiN6e1LTe7mQqM5r96IQPv1xdvu7w6
46iN/D4XozqjArQKYY/arL6QwyKLsgVba+GdU1KkqJ81lJfQGca0v8+GyzLIwTkX
7dhiSv4mClVBEDYpo80cDBhlyJsT15JRds4TrULalWO+SAorckASYZUPXEAZEyPf
yhdA4JmPNBpujr+ax8mBfqioZH1o0JpmKqKTYbaNMs6+d6ZPCwp4GmSLxhfLD2CE
WjK3PstTAgMBAAECggEAK9oqSV45Ou8lqgE2dSpmJ2yw+IaG1Q4H5eX1FQDCtLLm
L0ukZt5sv+hT3sr+JqUnLoKZ6irjYt0GjiagOlmUMTXmphjlMjZizA5drS87Kj+t
xn7S+dU1yjiLtw4AvuxMRPMYOiB5BARWgeJRX18d5up5lSvrBKIPzyjl8/JxpcW4
HB8vMehCPZIrRsgYkxaPJCGVlfKPW+TFA43FspJXL2kOtPFebNuylyOGWX8UAmS4
c5ZYUC7AYV03eFWNQEG6EQ6HrmF6D9MitiV4JPSsoMZdR6//SB/R4S+HbIRH/DZK
O9p3/aUWkNt9CcjTvrIe5VXRLy3XaxFnRkehSrF3UQKBgQDjcnyO5dTj6MhS2U5s
jvbTQR56L4x+BB7M6U44g2dYzY0pDpVqfqVZ1lhnHU5jYwZFCDo1DVU1cKXGTGU2
RKmmSH/Y1wjASjU62IweIfYqDCw8J094+dbRj9lge6xwRjAigawkkBL7d1PH3WjN
/QsjVeb4aPGtd1X0HT03UlXRuwKBgQDFOjCQHtEsKj1YQU/XQrX/eazTa6C1gp9x
V/WADfb9Lidyf830G1gdqRjUdwwi6HOay2rVJEPYh4VeVHKPYcAGxcxu7TbRp+t3
SFP2HgNmi52eSAn06ikdA8bU0y0EDxCQVW2vGdBvyhbwceznN/NcGU7uwCaGILOw
jkb19Q+nSQKBgQCwRV8hRB/czeDKzJ1J5vaFvNcI7ObeFwVj24CCrdwfZ5Z59lJ4
KVSurj7vEzhYMDuArqKl0QJzmyzu5PAfwdEVDOUAQY8Hr7tXMtJM3BcyeSKjL/gY
ktAYs3pNmyuGC+9sHsExyPLdLpqgsAh0dCL4rK+HX5XF0VGtEigKQuY91wKBgFTQ
ZEfl6L/cXksQsxv35To7AfZdR1wnExfz4nAyES/pZC9aBKBgDfGbYUEk/MQaQHSl
24hTMxXvmYvqNuWv/Jss/nAJdNSOKLVAFlM1rvKvQZXqltWKySlgEWY+dhJTxCS9
iBCPwlIAjwLRizYXmoDVpVsIqMhvUbawSJXGz/d5AoGAO+0S3wZVlKL6NGfXJ3WP
hcgXXvW6tzDl9ZSQVDSNQuhsIEMDHpRzeqk6MggPYTiV8AfBL4ZCRMJdmfRRJ0Z5
OgGWi+x4sWGlkxDy5wrQgEykVyq9NVh6En/obMTsjp2yriOQzEWnewHIKn/pwNdM
a1PEwjR8y76x28+1fLGTOlw=
-----END PRIVATE KEY-----"""

TEST_JWT_PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArzrM77JfA2Sbq0IEUF4U
SefABdxu4N+4Yr0xqqYKuvzAZaxNBZXSrV5CEu5scb8AkoA3pEuSJDAI6lZ6i1o2
NMJbz5sH/k7HHOD7K3wUcH1+wwaYIjentS03u5kKjOa/eiED79cXb7u8OuOojfw+
F6M6owK0CmGP2qy+kMMii7IFW2vhnVNSpKifNZSX0BnGtL/PhssyyME5F+3YYkr+
JgpVQRA2KaPNHAwYZcibE9eSUXbOE61C2pVjvkgKK3JAEmGVD1xAGRMj38oXQOCZ
jzQabo6/msfJgX6oqGR9aNCaZiqik2G2jTLOvnemTwsKeBpki8YXyw9ghFoytz7L
UwIDAQAB
-----END PUBLIC KEY-----"""


@pytest.fixture
def settings_override(tmp_path: Path) -> Settings:
    """Provide deterministic test settings."""

    db_file = tmp_path / "test_auth.db"

    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_file.as_posix()}",
        database_echo=False,
        jwt_private_key=TEST_JWT_PRIVATE_KEY,
        jwt_public_key=TEST_JWT_PUBLIC_KEY,
        celery_task_always_eager=True,
        celery_broker_url="memory://",
        celery_result_backend="cache+memory://",
    )


@pytest.fixture
def session_factory(settings_override: Settings) -> Generator[async_sessionmaker[AsyncSession], None, None]:
    """Create isolated async session factory with a fresh schema."""

    engine = create_async_engine(settings_override.database_url, echo=False, future=True)
    session_local = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def _prepare_schema() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_prepare_schema())
    yield session_local

    asyncio.run(engine.dispose())


@pytest.fixture
def client(
    session_factory: async_sessionmaker[AsyncSession],
    settings_override: Settings,
) -> Generator[TestClient, None, None]:
    """Return FastAPI test client with DB/settings overrides."""

    os.environ["DATABASE_URL"] = settings_override.database_url
    os.environ["JWT_PRIVATE_KEY"] = settings_override.jwt_private_key
    os.environ["JWT_PUBLIC_KEY"] = settings_override.jwt_public_key
    os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
    os.environ["CELERY_BROKER_URL"] = "memory://"
    os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"
    get_settings.cache_clear()

    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    def override_get_config() -> Settings:
        return settings_override

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_config] = override_get_config

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    get_settings.cache_clear()
