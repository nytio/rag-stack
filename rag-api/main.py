import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import asyncio
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
try:
    from llama_index.core.vector_stores import (
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    _FILTER_MODE = "advanced"
except Exception:
    try:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

        _FILTER_MODE = "exact"
    except Exception:
        MetadataFilters = None
        ExactMatchFilter = None
        _FILTER_MODE = "none"


# -----------------------------
# Config
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "")
EMBED_DIM = os.getenv("EMBED_DIM", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-local")

VECTOR_TABLE_NAME = os.getenv("VECTOR_TABLE_NAME", "rag_chunks")
RAG_API_KEY = os.getenv("RAG_API_KEY", "").strip()

# Chunking defaults (ajustables)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
DEFAULT_EMBED_DIM = 768


# -----------------------------
# API models
# -----------------------------
class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Identificador del documento (externo).")
    text: str = Field(..., description="Texto ya extraído y limpio (sin imágenes).")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: Optional[int] = Field(None, ge=200, le=4000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=1000)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    filters: Dict[str, Any] = Field(default_factory=dict)
    # Si quieres “modo estricto”: si no hay evidencia suficiente, el LLM debe decirlo.
    strict: bool = True


class SourceChunk(BaseModel):
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


# -----------------------------
# Helpers
# -----------------------------
def _require_api_key(x_api_key: Optional[str]) -> None:
    """Protección básica por header X-API-Key (opcional)."""
    if not RAG_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _validate_env() -> None:
    missing = []
    for k, v in [
        ("DATABASE_URL", DATABASE_URL),
        ("LLM_MODEL", LLM_MODEL),
        ("EMBED_MODEL", EMBED_MODEL),
        ("OPENAI_API_BASE", OPENAI_API_BASE),
    ]:
        if not v:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def _make_pgvector_store(embed_dim: int) -> PGVectorStore:
    """
    Construye PGVectorStore desde DATABASE_URL.

    Nota: PGVectorStore usa parámetros (host, port, user, password, db) en lugar de una connection string directa.
    """
    url = make_url(DATABASE_URL)

    # `PGVectorStore.from_params(...)` es el patrón típico en LlamaIndex para Postgres/pgvector
    # table_name = donde se guarda el embedding store
    return PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port or 5432,
        user=url.username,
        table_name=VECTOR_TABLE_NAME,
        embed_dim=embed_dim,
    )


def _make_llm() -> OpenAILike:
    """
    LLM OpenAI-compatible (Docker Model Runner).
    """
    return OpenAILike(
        model=LLM_MODEL,
        api_base=OPENAI_API_BASE,  # debe incluir /v1
        api_key=OPENAI_API_KEY,
        is_chat_model=True,
    )


class OpenAICompatibleEmbedding(BaseEmbedding):
    def __init__(self, model: str, api_base: str, api_key: str, timeout: float = 60.0):
        super().__init__(model_name=model)
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post_embeddings(self, inputs: Any) -> List[List[float]]:
        url = f"{self._api_base}/embeddings"
        payload = {"model": self._model, "input": inputs}
        response = self._client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item["embedding"] for item in data]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._post_embeddings(text)[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def get_text_embedding_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[List[float]]:
        return self._post_embeddings(texts)


def _make_embedder() -> OpenAICompatibleEmbedding:
    """
    Embeddings OpenAI-compatible (Docker Model Runner).
    """
    return OpenAICompatibleEmbedding(
        model=EMBED_MODEL,
        api_base=OPENAI_API_BASE,  # debe incluir /v1
        api_key=OPENAI_API_KEY,
    )


def _get_embed_dim() -> int:
    if EMBED_DIM:
        try:
            embed_dim = int(EMBED_DIM)
        except ValueError as exc:
            raise RuntimeError("EMBED_DIM must be an integer") from exc
        if embed_dim != DEFAULT_EMBED_DIM:
            raise RuntimeError("EMBED_DIM is fixed to 768 for granite-embedding-multilingual.")
        return DEFAULT_EMBED_DIM
    return DEFAULT_EMBED_DIM


def _build_filters(raw_filters: Dict[str, Any]) -> Optional[Any]:
    if not raw_filters:
        return None

    cleaned = {k: v for k, v in raw_filters.items() if v is not None}
    if not cleaned:
        return None

    if _FILTER_MODE == "advanced":
        filters = []
        for key, value in cleaned.items():
            if isinstance(value, list):
                if not value:
                    continue
                if not hasattr(FilterOperator, "IN"):
                    raise HTTPException(
                        status_code=400,
                        detail="List filters are not supported by this LlamaIndex build.",
                    )
                filters.append(
                    MetadataFilter(
                        key=key,
                        operator=FilterOperator.IN,
                        value=value,
                    )
                )
            else:
                filters.append(
                    MetadataFilter(
                        key=key,
                        operator=FilterOperator.EQ,
                        value=value,
                    )
                )
        if not filters:
            return None
        return MetadataFilters(filters=filters, condition=FilterCondition.AND)

    if _FILTER_MODE == "exact":
        for value in cleaned.values():
            if isinstance(value, list):
                raise HTTPException(
                    status_code=400,
                    detail="List filters require LlamaIndex FilterOperator.IN support.",
                )
        return MetadataFilters(
            filters=[ExactMatchFilter(key=k, value=v) for k, v in cleaned.items()]
        )

    raise HTTPException(
        status_code=400,
        detail="Metadata filters are not supported by this LlamaIndex build.",
    )


# -----------------------------
# App & lifecycle
# -----------------------------
app = FastAPI(title="rag-api (LlamaIndex + pgvector + DMR)", version="0.1.0")

_storage_context: Optional[StorageContext] = None
_index: Optional[VectorStoreIndex] = None
_vector_store: Optional[PGVectorStore] = None
_engine: Optional[Engine] = None
_llm = None
_embedder = None
_embed_dim: Optional[int] = None


@app.on_event("startup")
def startup() -> None:
    global _storage_context, _index, _llm, _embedder, _engine, _vector_store, _embed_dim

    _validate_env()
    if _FILTER_MODE == "none":
        raise RuntimeError("Metadata filters are not supported by this LlamaIndex build.")

    # DB connection (validación temprana)
    _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError("Database connection failed") from exc

    # Model clients (OpenAI-compatible)
    _llm = _make_llm()
    _embedder = _make_embedder()
    _embed_dim = _get_embed_dim()

    # Vector store in Postgres
    _vector_store = _make_pgvector_store(_embed_dim)
    _storage_context = StorageContext.from_defaults(vector_store=_vector_store)

    # Crear/abrir índice
    # Observe: VectorStoreIndex no “carga” todo a RAM; usa el vector store para retrieval.
    _index = VectorStoreIndex.from_vector_store(
        vector_store=_vector_store,
        storage_context=_storage_context,
        embed_model=_embedder,
    )


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    ok = all([_index is not None, _llm is not None, _embedder is not None])
    db_ok = False
    if _engine is not None:
        try:
            with _engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False
    return {
        "ok": ok,
        "db_ok": db_ok,
        "vector_table": VECTOR_TABLE_NAME,
        "embed_dim": _embed_dim,
        "filter_mode": _FILTER_MODE,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, x_api_key: Optional[str] = Header(default=None)) -> IngestResponse:
    _require_api_key(x_api_key)

    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    chunk_size = req.chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = req.chunk_overlap or DEFAULT_CHUNK_OVERLAP

    # 1) Documento base con metadata
    metadata = dict(req.metadata or {})
    metadata["doc_id"] = req.doc_id

    doc = Document(text=req.text, metadata=metadata)

    # 2) Chunking → Nodes
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([doc])

    # 3) Insertar en el índice (genera embeddings y escribe a pgvector)
    _index.insert_nodes(nodes)

    return IngestResponse(doc_id=req.doc_id, chunks_indexed=len(nodes))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, x_api_key: Optional[str] = Header(default=None)) -> QueryResponse:
    _require_api_key(x_api_key)

    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    # Instrucción “estricta” (anti-alucinación) si así lo quieres:
    strict_prefix = ""
    if req.strict:
        strict_prefix = (
            "Instrucciones: Responde únicamente con base en el CONTEXTO recuperado. "
            "Si el contexto no contiene evidencia suficiente, responde: "
            "'No tengo evidencia suficiente en los documentos para responder con certeza.'\n\n"
        )

    filters = _build_filters(req.filters)
    retriever_kwargs: Dict[str, Any] = {"similarity_top_k": req.top_k}
    if filters is not None:
        retriever_kwargs["filters"] = filters

    retriever = _index.as_retriever(**retriever_kwargs)
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=_llm)

    response = query_engine.query(strict_prefix + req.question)

    sources: List[SourceChunk] = []
    # LlamaIndex devuelve nodos fuente en response.source_nodes (si aplica).
    # Extraemos texto + score + metadata para citas.
    if hasattr(response, "source_nodes") and response.source_nodes:
        for sn in response.source_nodes:
            node = sn.node
            sources.append(
                SourceChunk(
                    text=(node.get_content() or "").strip(),
                    score=getattr(sn, "score", None),
                    metadata=(node.metadata or {}),
                )
            )

    return QueryResponse(answer=str(response), sources=sources)
