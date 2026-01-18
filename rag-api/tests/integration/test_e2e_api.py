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
    assert query_data["sources"], "Expected at least one source chunk"
    assert query_data["answer"] != (
        "No tengo evidencia suficiente en los documentos para responder con certeza."
    )
    assert any(unique_token in src.get("text", "") for src in query_data["sources"])
    assert any(src.get("metadata", {}).get("doc_id") == doc_id for src in query_data["sources"])

    filtered_payload = {
        "question": f"Encuentra el token del documento filtrado.\n{unique_token}",
        "top_k": 3,
        "strict": True,
        "filters": {"doc_id": doc_id},
    }

    filtered_resp = httpx.post(
        f"{base}/query",
        json=filtered_payload,
        headers=headers,
        timeout=60.0,
    )
    filtered_resp.raise_for_status()
    filtered_data = filtered_resp.json()
    assert filtered_data["sources"], "Expected sources with metadata filter applied"
    assert all(src.get("metadata", {}).get("doc_id") == doc_id for src in filtered_data["sources"])
    assert any(unique_token in src.get("text", "") for src in filtered_data["sources"])
    assert filtered_data["answer"] != (
        "No tengo evidencia suficiente en los documentos para responder con certeza."
    )


def test_ingest_chunks_with_real_embeddings() -> None:
    base = _api_base()
    headers = {"Content-Type": "application/json", **_auth_headers()}

    health_resp = httpx.get(f"{base}/healthz", timeout=10.0)
    health_resp.raise_for_status()
    health_data = health_resp.json()
    assert health_data.get("ok") is True
    assert health_data.get("db_ok") is True

    doc_id = f"e2e-chunks-{uuid.uuid4().hex}"
    token_a = f"CHUNK_TOKEN_{uuid.uuid4().hex}"
    token_b = f"CHUNK_TOKEN_{uuid.uuid4().hex}"

    payload = {
        "doc_id": doc_id,
        "metadata": {"source": "integration-test", "dataset": "ingest_chunks"},
        "chunks": [
            {
                "text": f"Chunk A de prueba. Token: {token_a}.",
                "metadata": {"page": 1},
                "chunk_id": "c1",
            },
            {
                "text": f"Chunk B de prueba. Token: {token_b}.",
                "metadata": {"page": 2},
                "chunk_id": "c2",
            },
        ],
    }

    resp = httpx.post(
        f"{base}/ingest_chunks",
        json=payload,
        headers=headers,
        timeout=120.0,
    )
    if resp.status_code == 401 and "X-API-Key" not in headers:
        pytest.skip("API requires RAG_API_KEY; set env var to run ingest_chunks E2E")
    resp.raise_for_status()
    data = resp.json()
    assert data["doc_id"] == doc_id
    assert data["chunks_indexed"] == len(payload["chunks"])


def test_requires_api_key_on_ingest() -> None:
    if not os.getenv("RAG_API_KEY"):
        pytest.skip("RAG_API_KEY not set; auth not enforced in this environment")
    base = _api_base()
    payload = {
        "doc_id": "auth-missing",
        "text": "Texto de prueba para auth.",
    }
    resp = httpx.post(f"{base}/ingest", json=payload, timeout=30.0)
    assert resp.status_code == 401


def test_requires_api_key_on_query() -> None:
    if not os.getenv("RAG_API_KEY"):
        pytest.skip("RAG_API_KEY not set; auth not enforced in this environment")
    base = _api_base()
    payload = {"question": "Hola", "top_k": 1}
    resp = httpx.post(f"{base}/query", json=payload, timeout=30.0)
    assert resp.status_code == 401


def test_ingest_validation_errors() -> None:
    base = _api_base()
    headers = {"Content-Type": "application/json", **_auth_headers()}
    invalid_payloads = [
        {"doc_id": "bad-1", "text": "x", "chunk_size": 199},
        {"doc_id": "bad-2", "text": "x", "chunk_size": 4001},
        {"doc_id": "bad-3", "text": "x", "chunk_overlap": -1},
        {"doc_id": "bad-4", "text": "x", "chunk_overlap": 1001},
    ]
    for payload in invalid_payloads:
        resp = httpx.post(
            f"{base}/ingest",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        assert resp.status_code == 422


def test_query_validation_errors() -> None:
    base = _api_base()
    headers = {"Content-Type": "application/json", **_auth_headers()}
    invalid_payloads = [
        {"question": "hola", "top_k": 0},
        {"question": "hola", "top_k": 21},
    ]
    for payload in invalid_payloads:
        resp = httpx.post(
            f"{base}/query",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        assert resp.status_code == 422
