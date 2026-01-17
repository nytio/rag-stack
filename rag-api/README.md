# rag-api (FastAPI + LlamaIndex + Postgres/pgvector + Docker Model Runner)

## Endpoints
- GET /healthz
- POST /ingest
- POST /query

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
