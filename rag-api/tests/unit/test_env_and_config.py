from __future__ import annotations

import types

import pytest

from tests.conftest import require_modules

require_modules(
    "fastapi",
    "httpx",
    "sqlalchemy",
    "llama_index",
    "llama_index.vector_stores.postgres",
    "llama_index.llms.openai_like",
)

from fastapi import HTTPException

import main

pytestmark = pytest.mark.unit


def test_validate_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "DATABASE_URL", "")
    monkeypatch.setattr(main, "LLM_MODEL", "")
    monkeypatch.setattr(main, "EMBED_MODEL", "")
    monkeypatch.setattr(main, "OPENAI_API_BASE", "")

    with pytest.raises(RuntimeError, match="Missing required env vars"):
        main._validate_env()


def test_validate_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setattr(main, "LLM_MODEL", "model-x")
    monkeypatch.setattr(main, "EMBED_MODEL", "embed-x")
    monkeypatch.setattr(main, "OPENAI_API_BASE", "http://localhost:8000/v1")

    main._validate_env()


def test_get_embed_dim_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "EMBED_DIM", "")
    assert main._get_embed_dim() == main.DEFAULT_EMBED_DIM


def test_get_embed_dim_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "EMBED_DIM", "768")
    assert main._get_embed_dim() == 768


def test_get_embed_dim_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "EMBED_DIM", "not-int")
    with pytest.raises(RuntimeError, match="EMBED_DIM must be an integer"):
        main._get_embed_dim()


def test_get_embed_dim_wrong_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "EMBED_DIM", "1024")
    with pytest.raises(RuntimeError, match="EMBED_DIM is fixed to 768"):
        main._get_embed_dim()


def test_require_api_key_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    main._require_api_key(None)


def test_require_api_key_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "RAG_API_KEY", "secret")
    with pytest.raises(HTTPException):
        main._require_api_key(None)

    with pytest.raises(HTTPException):
        main._require_api_key("wrong")

    main._require_api_key("secret")


def test_make_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "LLM_MODEL", "model-x")
    monkeypatch.setattr(main, "OPENAI_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setattr(main, "OPENAI_API_KEY", "sk-test")

    llm = main._make_llm()
    assert llm is not None


def test_make_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "EMBED_MODEL", "embed-x")
    monkeypatch.setattr(main, "OPENAI_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setattr(main, "OPENAI_API_KEY", "sk-test")

    embedder = main._make_embedder()
    assert embedder is not None


def test_make_pgvector_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    store = main._make_pgvector_store(768)
    assert store is not None
    assert store.__class__.__name__ == "PGVectorStore"


def test_embedding_headers_include_bearer() -> None:
    emb = main.OpenAICompatibleEmbedding(
        model="embed-x",
        api_base="http://localhost:8000/v1",
        api_key="sk-test",
    )
    headers = emb._headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer sk-test"


def test_embedding_headers_omit_bearer_when_empty() -> None:
    emb = main.OpenAICompatibleEmbedding(
        model="embed-x",
        api_base="http://localhost:8000/v1",
        api_key="",
    )
    headers = emb._headers()
    assert headers["Content-Type"] == "application/json"
    assert "Authorization" not in headers


def test_embedding_post_embeddings_uses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyClient:
        def __init__(self) -> None:
            self.called = False
            self.args = None
            self.kwargs = None

        def post(self, *args, **kwargs):
            self.called = True
            self.args = args
            self.kwargs = kwargs
            return types.SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"data": [{"embedding": [0.1, 0.2]}]},
            )

    emb = main.OpenAICompatibleEmbedding(
        model="embed-x",
        api_base="http://localhost:8000/v1",
        api_key="sk-test",
    )
    dummy_client = DummyClient()
    emb._client = dummy_client

    result = emb._post_embeddings("hola")
    assert dummy_client.called is True
    assert result == [[0.1, 0.2]]
