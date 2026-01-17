from __future__ import annotations

import os
import uuid

import pytest

from tests.conftest import require_env, require_modules

require_modules("httpx")

import httpx


pytestmark = pytest.mark.integration


def _api_base() -> str:
    require_env("RAG_API_BASE")
    return os.environ["RAG_API_BASE"].rstrip("/")


def _auth_headers() -> dict[str, str]:
    api_key = os.getenv("RAG_API_KEY", "").strip()
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def test_healthz() -> None:
    base = _api_base()
    response = httpx.get(f"{base}/healthz", timeout=10.0)
    response.raise_for_status()
    data = response.json()
    assert "ok" in data
    assert "db_ok" in data


def test_ingest_and_query() -> None:
    base = _api_base()
    headers = {"Content-Type": "application/json", **_auth_headers()}

    doc_id = f"e2e-{uuid.uuid4().hex}"
    unique_token = f"E2E_TOKEN_{uuid.uuid4().hex}"

    ingest_payload = {
        "doc_id": doc_id,
        "text": f"Documento de prueba. Token unico: {unique_token}.",
        "metadata": {"source": "e2e"},
        "chunk_size": 300,
        "chunk_overlap": 20,
    }

    ingest_resp = httpx.post(
        f"{base}/ingest",
        json=ingest_payload,
        headers=headers,
        timeout=30.0,
    )
    if ingest_resp.status_code == 401 and "X-API-Key" not in headers:
        pytest.skip("API requires RAG_API_KEY; set env var to run E2E ingest/query")
    ingest_resp.raise_for_status()
    ingest_data = ingest_resp.json()
    assert ingest_data["doc_id"] == doc_id
    assert ingest_data["chunks_indexed"] >= 1

    query_payload = {
        "question": f"Cual es el token unico del documento?\n{unique_token}",
        "top_k": 3,
        "strict": True,
    }

    query_resp = httpx.post(
        f"{base}/query",
        json=query_payload,
        headers=headers,
        timeout=60.0,
    )
    query_resp.raise_for_status()
    query_data = query_resp.json()

    assert "answer" in query_data
    assert "sources" in query_data
    assert isinstance(query_data["sources"], list)
