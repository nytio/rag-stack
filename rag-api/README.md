# rag-api (FastAPI + LlamaIndex + Postgres/pgvector + Docker Model Runner)

## Endpoints
- GET /healthz
- POST /ingest
- POST /query

Ejemplo:
```
curl http://localhost:8000/healthz
```

Respuesta (parámetros):
- `ok`: estado general del servicio.
- `db_ok`: estado de la conexión a Postgres/pgvector.
- `vector_table`: tabla usada para los embeddings.
- `embed_dim`: dimensión de los embeddings.
- `filter_mode`: modo de filtros soportado por LlamaIndex.
- `llm_model`: modelo LLM configurado.
- `embed_model`: modelo de embeddings configurado.

## Auth (obligatoria)
La Auth no es opcional: siempre debes enviar el header `X-API-Key` en cada request.
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
Ingest:
```
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"doc_id":"doc-1","text":"Texto del documento...","metadata":{"title":"Mi doc"},"chunk_size":900,"chunk_overlap":120}'
```

Parámetros del `-d`:
- `doc_id` (string, requerido): identificador externo del documento.
- `text` (string, requerido): texto ya extraído y limpio; no debe incluir imágenes ni binarios.
- `metadata` (object, opcional): metadatos libres (p. ej. `title`, `source`).
- `chunk_size` (int, opcional): tamaño de chunk; mínimo 200, máximo 4000.
- `chunk_overlap` (int, opcional): solapamiento; mínimo 0, máximo 1000.

Query:
```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"question":"¿Qué dice el documento sobre X?","top_k":5,"strict":true,"filters":{"doc_id":"doc-1"}}'
```
