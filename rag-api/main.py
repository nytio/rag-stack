import os
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import asyncio
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
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
STRICT_NO_EVIDENCE_MESSAGE = (
    "No tengo evidencia suficiente en los documentos para responder con certeza."
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{8,}")
_SQL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


class IngestChunk(BaseModel):
    text: str = Field(..., description="Texto del chunk pre-construido.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = Field(None, description="ID opcional del chunk (estable).")


class IngestChunksRequest(BaseModel):
    doc_id: str = Field(..., description="Identificador del documento (externo).")
    chunks: List[IngestChunk]
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
        return self._post_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def get_text_embedding_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[List[float]]:
        try:
            return self._post_embeddings(texts)
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code not in (400, 422):
                raise
            # Some OpenAI-compatible runners error on batch embeddings; fall back to single requests.
            if len(texts) <= 1:
                raise
            embeddings: List[List[float]] = []
            for text in texts:
                embeddings.append(self._post_embeddings([text])[0])
            return embeddings


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


def _extract_search_tokens(text: str) -> List[str]:
    tokens = [t for t in _TOKEN_RE.findall(text) if any(ch.isdigit() for ch in t)]
    # Preserve order, remove duplicates
    seen = set()
    unique: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def _vector_table_name() -> Optional[str]:
    table_name = VECTOR_TABLE_NAME.lower()
    if not _SQL_NAME_RE.match(table_name):
        return None
    return f"data_{table_name}"


def _keyword_search_sources(tokens: List[str], limit: int) -> List["SourceChunk"]:
    if _engine is None:
        return []
    table_name = _vector_table_name()
    if not table_name:
        return []
    schema_name = getattr(_vector_store, "schema_name", "public")
    if not _SQL_NAME_RE.match(schema_name):
        return []
    where_clauses = []
    params: Dict[str, Any] = {"limit": limit}
    for idx, token in enumerate(tokens):
        key = f"token_{idx}"
        where_clauses.append(f"text ILIKE :{key}")
        params[key] = f"%{token}%"
    if not where_clauses:
        return []
    stmt = text(
        f"SELECT text, metadata_ FROM {schema_name}.{table_name} "
        f"WHERE {' OR '.join(where_clauses)} ORDER BY id DESC LIMIT :limit"
    )
    with _engine.connect() as conn:
        rows = conn.execute(stmt, params).fetchall()
    sources: List[SourceChunk] = []
    for row in rows:
        metadata = row.metadata_ or {}
        sources.append(
            SourceChunk(
                text=(row.text or "").strip(),
                score=None,
                metadata=metadata,
            )
        )
    return sources


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

_storage_context: Optional[StorageContext] = None
_index: Optional[VectorStoreIndex] = None
_vector_store: Optional[PGVectorStore] = None
_engine: Optional[Engine] = None
_llm = None
_embedder = None
_embed_dim: Optional[int] = None


def _init_app() -> None:
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


def _shutdown_app() -> None:
    if _engine is not None:
        _engine.dispose()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _init_app()
    try:
        yield
    finally:
        _shutdown_app()


app = FastAPI(
    title="rag-api (LlamaIndex + pgvector + DMR)",
    version="0.1.0",
    lifespan=lifespan,
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

    doc = Document(text=req.text, metadata=metadata, id_=req.doc_id)

    # 2) Chunking → Nodes
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([doc])

    # 3) Insertar en el índice (genera embeddings y escribe a pgvector)
    _index.insert_nodes(nodes)

    return IngestResponse(doc_id=req.doc_id, chunks_indexed=len(nodes))


@app.post("/ingest_chunks", response_model=IngestResponse)
def ingest_chunks(
    req: IngestChunksRequest, x_api_key: Optional[str] = Header(default=None)
) -> IngestResponse:
    _require_api_key(x_api_key)

    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    if not req.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided")

    doc_metadata = dict(req.metadata or {})
    nodes: List[TextNode] = []

    for chunk in req.chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue

        metadata = dict(doc_metadata)
        if chunk.metadata:
            metadata.update(chunk.metadata)
        metadata["doc_id"] = req.doc_id
        if chunk.chunk_id:
            metadata["chunk_id"] = chunk.chunk_id

        nodes.append(TextNode(text=text, metadata=metadata, id_=chunk.chunk_id))

    if not nodes:
        raise HTTPException(status_code=400, detail="All chunks are empty")

    _index.insert_nodes(nodes)

    return IngestResponse(doc_id=req.doc_id, chunks_indexed=len(nodes))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, x_api_key: Optional[str] = Header(default=None)) -> QueryResponse:
    _require_api_key(x_api_key)

    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    # Instrucción “estricta” (anti-alucinación) si así lo quieres:
    text_qa_template = None
    if req.strict:
        text_qa_template = PromptTemplate(
            "Instrucciones: Responde únicamente con base en el CONTEXTO recuperado. "
            "Si el contexto no contiene evidencia suficiente, responde: "
            f"'{STRICT_NO_EVIDENCE_MESSAGE}'\n"
            "---------------------\n"
            "CONTEXTO:\n{context_str}\n"
            "---------------------\n"
            "PREGUNTA: {query_str}\n"
        )

    filters = _build_filters(req.filters)
    retriever_kwargs: Dict[str, Any] = {"similarity_top_k": req.top_k}
    if filters is not None:
        retriever_kwargs["filters"] = filters

    retriever = _index.as_retriever(**retriever_kwargs)
    query_engine_kwargs: Dict[str, Any] = {"retriever": retriever, "llm": _llm}
    if text_qa_template is not None:
        query_engine_kwargs["text_qa_template"] = text_qa_template
    query_engine = RetrieverQueryEngine.from_args(**query_engine_kwargs)

    response = query_engine.query(req.question)

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

    answer_text = str(response)
    if req.strict:
        tokens = _extract_search_tokens(req.question)
        if tokens and not any(
            token in (src.text or "") for token in tokens for src in sources
        ):
            keyword_sources = _keyword_search_sources(tokens, req.top_k)
            if keyword_sources:
                sources = keyword_sources
        if answer_text.strip() == STRICT_NO_EVIDENCE_MESSAGE and sources:
            # If we did retrieve context but the LLM still declined, return the top chunk as a fallback.
            answer_text = sources[0].text

    return QueryResponse(answer=answer_text, sources=sources)
