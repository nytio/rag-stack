# rag-api (FastAPI + LlamaIndex + Postgres/pgvector + Docker Model Runner)

## Endpoints
- GET /healthz
- POST /ingest
- POST /query

Ejemplo:
```
curl http://localhost:8000/healthz
{"ok":true,"db_ok":true,"vector_table":"rag_chunks","embed_dim":768,"filter_mode":"advanced","llm_model":"ai/granite-4.0-h-micro","embed_model":"ai/granite-embedding-multilingual"}
```

Respuesta (parámetros):
- `ok`: estado general del servicio.
- `db_ok`: estado de la conexión a Postgres/pgvector.
- `vector_table`: tabla usada para los embeddings.
- `embed_dim`: dimensión de los embeddings.
- `filter_mode`: modo de filtros soportado por LlamaIndex.
- `llm_model`: modelo LLM configurado.
- `embed_model`: modelo de embeddings configurado.

## Auth (opcional)
Si defines `RAG_API_KEY`, debes enviar header:
```
X-API-Key: <tu-clave>
```

Ejemplo con `curl`:
```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"question":"¿Qué dice el documento sobre X?","top_k":5,"strict":true,"filters":{"doc_id":"doc-1"}}'
```

## Variables de entorno requeridas
- DATABASE_URL
- OPENAI_API_BASE (OpenAI-compatible, debe terminar en /v1)
- LLM_MODEL
- EMBED_MODEL
- OPENAI_API_KEY (puede ser dummy para local)
Opcional:
- EMBED_DIM (fijo en 768 para `ai/granite-embedding-multilingual`; si se define debe ser 768)

## Ejemplos
Ingest (con auth):
```
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"doc_id":"doc-1","text":"Texto del documento...","metadata":{"title":"Mi doc"},"chunk_size":900,"chunk_overlap":120}'
```

Query (con auth):
```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"question":"¿Qué dice el documento sobre X?","top_k":5,"strict":true,"filters":{"doc_id":"doc-1"}}'
```
