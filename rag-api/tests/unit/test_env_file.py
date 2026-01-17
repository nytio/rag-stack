from __future__ import annotations

import pytest

from tests.conftest import TEST_ENV_PATH, parse_env_file


pytestmark = pytest.mark.unit

REQUIRED_KEYS = {
    "DATABASE_URL",
    "OPENAI_API_BASE",
    "LLM_MODEL",
    "EMBED_MODEL",
}


def test_env_file_exists() -> None:
    assert TEST_ENV_PATH.exists(), ".env.test file is missing at repo root"


def test_env_file_contains_required_keys() -> None:
    data = parse_env_file(TEST_ENV_PATH)
    missing = [key for key in sorted(REQUIRED_KEYS) if not data.get(key)]
    assert not missing, f"Missing or empty keys in .env.test: {', '.join(missing)}"


def test_env_file_values_are_valid() -> None:
    data = parse_env_file(TEST_ENV_PATH)
    database_url = data.get("DATABASE_URL", "")
    openai_api_base = data.get("OPENAI_API_BASE", "")

    assert database_url.startswith("postgresql://"), "DATABASE_URL must start with postgresql://"
    assert openai_api_base.endswith("/v1"), "OPENAI_API_BASE must end with /v1"
