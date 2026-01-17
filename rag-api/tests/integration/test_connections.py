from __future__ import annotations

import os

import pytest

from tests.conftest import require_env, require_modules, require_url_resolvable

require_modules("httpx", "sqlalchemy")

import httpx
from sqlalchemy import create_engine, text


@pytest.mark.integration
def test_db_connection() -> None:
    require_env("DATABASE_URL")
    engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1")).scalar()
    assert result == 1


@pytest.mark.integration
def test_llm_chat_completion() -> None:
    require_env("OPENAI_API_BASE", "LLM_MODEL")
    api_base = os.environ["OPENAI_API_BASE"].rstrip("/")
    require_url_resolvable(api_base, var_name="OPENAI_API_BASE")
    url = f"{api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0,
        "max_tokens": 5,
    }

    response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    assert "choices" in data
    assert data["choices"]


@pytest.mark.integration
def test_embedding_call() -> None:
    require_env("OPENAI_API_BASE", "EMBED_MODEL")
    api_base = os.environ["OPENAI_API_BASE"].rstrip("/")
    require_url_resolvable(api_base, var_name="OPENAI_API_BASE")
    url = f"{api_base}/embeddings"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": os.environ["EMBED_MODEL"], "input": "ping"}

    response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    assert "data" in data
    assert data["data"]
    assert "embedding" in data["data"][0]
    assert len(data["data"][0]["embedding"]) > 0
