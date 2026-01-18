from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

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


class DummyIndex:
    def __init__(self) -> None:
        self.inserted: List[Any] = []
        self.retriever_kwargs = None

    def insert_nodes(self, nodes: List[Any]) -> None:
        self.inserted = nodes

    def as_retriever(self, **kwargs: Any) -> str:
        self.retriever_kwargs = kwargs
        return "dummy-retriever"


@dataclass
class DummyNode:
    content: str
    metadata: dict

    def get_content(self, **_: Any) -> str:
        return self.content


@dataclass
class DummySourceNode:
    node: DummyNode
    score: float


class DummyResponse:
    def __init__(self, text: str, source_nodes: List[DummySourceNode]) -> None:
        self._text = text
        self.source_nodes = source_nodes

    def __str__(self) -> str:
        return self._text


class DummyQueryEngine:
    def __init__(self, response: DummyResponse) -> None:
        self.last_query = None
        self._response = response

    def query(self, query: str) -> DummyResponse:
        self.last_query = query
        return self._response


def test_ingest_requires_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_index", None)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    with pytest.raises(HTTPException):
        main.ingest(main.IngestRequest(doc_id="d1", text="hola"))


def test_ingest_calls_index_insert(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")

    dummy_nodes = [
        DummyNode(content="chunk-1", metadata={"doc_id": "doc-1"}),
        DummyNode(content="chunk-2", metadata={"doc_id": "doc-1"}),
    ]
    monkeypatch.setattr(
        main.SentenceSplitter,
        "get_nodes_from_documents",
        lambda self, docs: dummy_nodes,
    )

    resp = main.ingest(
        main.IngestRequest(
            doc_id="doc-1",
            text="texto",
            metadata={"title": "demo"},
            chunk_size=200,
            chunk_overlap=0,
        )
    )

    assert resp.doc_id == "doc-1"
    assert resp.chunks_indexed == 2
    assert dummy_index.inserted == dummy_nodes


def test_ingest_chunks_requires_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_index", None)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    with pytest.raises(HTTPException):
        main.ingest_chunks(
            main.IngestChunksRequest(doc_id="doc-1", chunks=[], metadata={})
        )


def test_ingest_chunks_requires_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")

    with pytest.raises(HTTPException) as excinfo:
        main.ingest_chunks(main.IngestChunksRequest(doc_id="doc-1", chunks=[], metadata={}))
    assert excinfo.value.status_code == 400


def test_ingest_chunks_rejects_all_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")

    req = main.IngestChunksRequest(
        doc_id="doc-1",
        chunks=[
            main.IngestChunk(text="   ", metadata={"page": 1}, chunk_id="c1"),
            main.IngestChunk(text="\n", metadata={"page": 2}),
        ],
        metadata={"title": "demo"},
    )

    with pytest.raises(HTTPException) as excinfo:
        main.ingest_chunks(req)
    assert excinfo.value.status_code == 400


def test_ingest_chunks_merges_metadata_and_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTextNode:
        def __init__(self, text: str, metadata: dict, id_: str | None = None) -> None:
            self.text = text
            self.metadata = metadata
            self.id_ = id_

    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    monkeypatch.setattr(main, "TextNode", DummyTextNode)
    monkeypatch.setattr(main, "_enforce_embed_token_limit", lambda *_: None)

    req = main.IngestChunksRequest(
        doc_id="doc-1",
        chunks=[
            main.IngestChunk(
                text="chunk-1",
                metadata={"page": 1, "title": "override"},
                chunk_id="c1",
            ),
            main.IngestChunk(text="chunk-2", metadata={"page": 2}),
        ],
        metadata={"title": "doc-title", "source": "pdf"},
    )

    resp = main.ingest_chunks(req)

    assert resp.doc_id == "doc-1"
    assert resp.chunks_indexed == 2
    assert len(dummy_index.inserted) == 2

    first = dummy_index.inserted[0]
    assert first.text == "chunk-1"
    assert first.id_ == "c1"
    assert first.metadata["doc_id"] == "doc-1"
    assert first.metadata["chunk_id"] == "c1"
    assert first.metadata["page"] == 1
    assert first.metadata["title"] == "override"
    assert first.metadata["source"] == "pdf"

    second = dummy_index.inserted[1]
    assert second.text == "chunk-2"
    assert second.id_ is None
    assert second.metadata["doc_id"] == "doc-1"
    assert "chunk_id" not in second.metadata
    assert second.metadata["page"] == 2
    assert second.metadata["title"] == "doc-title"


def test_ingest_chunks_enforces_token_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTextNode:
        def __init__(self, text: str, metadata: dict, id_: str | None = None) -> None:
            self.text = text
            self.metadata = metadata
            self.id_ = id_

    calls: List[tuple[str, str]] = []

    def _record_limit(text: str, label: str) -> None:
        calls.append((text, label))

    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    monkeypatch.setattr(main, "TextNode", DummyTextNode)
    monkeypatch.setattr(main, "_enforce_embed_token_limit", _record_limit)

    req = main.IngestChunksRequest(
        doc_id="doc-1",
        chunks=[
            main.IngestChunk(text="  hola  ", metadata={}, chunk_id="c1"),
            main.IngestChunk(text="   ", metadata={"page": 2}),
            main.IngestChunk(text="adios", metadata={"page": 3}),
        ],
        metadata={},
    )

    resp = main.ingest_chunks(req)

    assert resp.chunks_indexed == 2
    assert calls == [("hola", "chunk c1"), ("adios", "chunk 2")]

def test_query_requires_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_index", None)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    with pytest.raises(HTTPException):
        main.query(main.QueryRequest(question="hola"))


def test_query_builds_response_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    monkeypatch.setattr(main, "_llm", object())

    dummy_node = DummyNode(content="contenido", metadata={"doc_id": "doc-1"})
    dummy_source = DummySourceNode(node=dummy_node, score=0.9)
    dummy_response = DummyResponse(text="respuesta", source_nodes=[dummy_source])
    dummy_engine = DummyQueryEngine(response=dummy_response)

    def _from_args(retriever: Any, llm: Any, **kwargs: Any):
        assert retriever == "dummy-retriever"
        assert llm is main._llm
        assert kwargs.get("text_qa_template") is not None
        return dummy_engine

    monkeypatch.setattr(main, "_build_filters", lambda _: "filters")
    monkeypatch.setattr(main.RetrieverQueryEngine, "from_args", staticmethod(_from_args))

    resp = main.query(main.QueryRequest(question="pregunta", top_k=3, strict=True))

    assert dummy_index.retriever_kwargs == {"similarity_top_k": 3, "filters": "filters"}
    assert resp.answer == "respuesta"
    assert resp.sources
    assert resp.sources[0].text == "contenido"
    assert resp.sources[0].metadata["doc_id"] == "doc-1"
    assert dummy_engine.last_query == "pregunta"


def test_query_strict_false_does_not_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    monkeypatch.setattr(main, "_llm", object())

    dummy_response = DummyResponse(text="respuesta", source_nodes=[])
    dummy_engine = DummyQueryEngine(response=dummy_response)

    def _from_args(retriever: Any, llm: Any, **kwargs: Any):
        assert "text_qa_template" not in kwargs
        return dummy_engine

    monkeypatch.setattr(main, "_build_filters", lambda _: None)
    monkeypatch.setattr(main.RetrieverQueryEngine, "from_args", staticmethod(_from_args))

    question = "pregunta sin prefijo"
    resp = main.query(main.QueryRequest(question=question, top_k=3, strict=False))

    assert resp.answer == "respuesta"
    assert dummy_engine.last_query == question


def test_query_without_source_nodes_returns_empty_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_index = DummyIndex()
    monkeypatch.setattr(main, "_index", dummy_index)
    monkeypatch.setattr(main, "RAG_API_KEY", "")
    monkeypatch.setattr(main, "_llm", object())

    dummy_response = DummyResponse(text="respuesta", source_nodes=[])
    dummy_engine = DummyQueryEngine(response=dummy_response)

    def _from_args(retriever: Any, llm: Any, **kwargs: Any):
        assert kwargs.get("text_qa_template") is not None
        return dummy_engine

    monkeypatch.setattr(main, "_build_filters", lambda _: None)
    monkeypatch.setattr(main.RetrieverQueryEngine, "from_args", staticmethod(_from_args))

    resp = main.query(main.QueryRequest(question="pregunta", top_k=3, strict=True))

    assert resp.sources == []
