# rag-api

Servicio RAG construido con FastAPI, LlamaIndex y Postgres/pgvector, integrado con un runner de modelos OpenAI-compatible (Docker Model Runner).

## Endpoints
| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | /healthz | Estado del servicio y configuración básica |
| POST | /ingest | Indexa texto y metadata en pgvector |
| POST | /ingest_chunks | Indexa chunks pre-construidos (sin re-chunking) |
| POST | /query | Recupera contexto y responde con fuentes |
| POST | /delete | Borra un documento y sus chunks por `doc_id` |

### Health check
```bash
curl http://localhost:8000/healthz
```

Campos de respuesta:
| Campo | Descripción |
| --- | --- |
| ok | estado general del servicio |
| db_ok | estado de la conexión a Postgres/pgvector |
| vector_table | tabla usada para los embeddings |
| embed_dim | dimensión de los embeddings |
| filter_mode | modo de filtros soportado por LlamaIndex |
| llm_model | modelo LLM configurado |
| embed_model | modelo de embeddings configurado |

## Auth (obligatoria)
Siempre debes enviar el header `X-API-Key` en cada request. Si falta o es incorrecto, la API responde 401.
```
X-API-Key: <tu-clave>
```

## Variables de entorno
| Variable | Requerida | Descripción |
| --- | --- | --- |
| DATABASE_URL | sí | URL de conexión a Postgres |
| OPENAI_API_BASE | sí | Endpoint OpenAI-compatible (debe terminar en `/v1`) |
| LLM_MODEL | sí | modelo LLM |
| EMBED_MODEL | sí | modelo de embeddings |
| OPENAI_API_KEY | sí | puede ser dummy para local |
| RAG_API_KEY | sí | clave usada por `X-API-Key` |
| EMBED_DIM | no | si se define, debe ser 768 |

## Ejemplos
### Ingest
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"doc_id":"doc-1","text":"Texto del documento...","metadata":{"filename":"manual.pdf","page":3,"chunk_id":"doc-1:0","chunk_index":0},"chunk_size":900,"chunk_overlap":120}'
```

Parámetros del `-d`:
| Campo | Tipo | Requerido | Notas |
| --- | --- | --- | --- |
| doc_id | string | sí | identificador del documento; se guarda en metadata |
| text | string | sí | texto limpio; sin imágenes ni binarios |
| metadata | object | no | metadatos libres (p. ej. `filename`, `source`, `page`, `chunk_id`, `chunk_index`) |
| chunk_size | int | no | tamaño de chunk; mínimo 200, máximo 4000 |
| chunk_overlap | int | no | solapamiento; mínimo 0, máximo 1000 |

Ejemplos de Metadata recomendados:
- `page` o `page_number` (int): número de página del chunk.
- `page_start` / `page_end` (int): solo si el chunk cruza varias páginas.
- `document` (string): título del documento.
- `source` o `url` (string): origen del documento.
- `section` o `heading` (string): sección o encabezado si lo extraes.
- `chunk_id` (string): ID estable del chunk; permite upsert.
- `chunk_index` (int): índice del chunk dentro del documento (usable para upsert).
- `question` (string): preguntas que el chunk puede resolver.
- `adjacent_summary` (string): resumen de los chunks adyacentes.

Sugerencias:
- Usa tipos consistentes (si `page` es int, siempre int).
- Mantén los valores cortos y útiles para filtros/seguimiento.
- Guarda solo metadatos que planeas filtrar en Query.
- Si incluyes `chunk_id` o `chunk_index`, el servidor intentará hacer upsert del chunk en `/ingest`.

### Ingest de chunks pre-construidos

Permite subir chunks ya generados (por ejemplo desde `data-in`) sin que el servidor haga re-chunking.
Esta operación **reemplaza** el `doc_id`: primero borra los chunks previos y luego inserta los nuevos.

```bash
curl -X POST http://localhost:8000/ingest_chunks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{
    "doc_id": "doc-1",
    "metadata": {"source_path": "docs/manual.pdf"},
    "chunks": [
      {
        "chunk_index": 0,
        "chunk_id": "doc-1:0",
        "text": "Primer chunk...",
        "metadata": {"page_start": 1, "page_end": 1, "chunk_index": 0}
      },
      {
        "chunk_index": 1,
        "chunk_id": "doc-1:1",
        "text": "Segundo chunk...",
        "metadata": {"page_start": 2, "page_end": 2, "chunk_index": 1}
      }
    ]
  }'
```

Notas:
- `doc_id` se fuerza en metadata de todos los chunks (útil para filtros).
- `metadata` a nivel documento se mezcla con la metadata de cada chunk.
- `chunk_id` es opcional pero recomendado para estabilidad e idempotencia.
- Si no envías `chunk_id`, se intentará derivar usando `doc_id:chunk_index`.

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"question":"¿Qué dice el documento sobre X?","top_k":5,"strict":true,"filters":{"doc_id":"doc-1"}}'
```

Parámetros del `-d`:
| Campo | Tipo | Requerido | Notas |
| --- | --- | --- | --- |
| question | string | sí | pregunta del usuario |
| top_k | int | no | cantidad de chunks a recuperar; mínimo 1, máximo 20 |
| strict | bool | no | si es `true`, el LLM reconoce cuando no hay evidencia suficiente |
| filters | object | no | filtros por metadata (coincidencia exacta por clave); listas requieren `filter_mode=advanced` |

Limitaciones esperadas:
- La respuesta depende del contenido previamente ingestado.
- `filters` solo aplica sobre metadatos disponibles en los chunks.
- `top_k` está limitado a 20.

### Delete
```bash
curl -X POST http://localhost:8000/delete \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <tu-clave>" \
  -d '{"doc_id":"doc-1"}'
```

Respuesta esperada:
```json
{"doc_id":"doc-1","deleted":true}
```

Notas:
- `doc_id` debe coincidir con el identificador usado en ingest (p. ej. hash del documento).
- Borra todos los chunks asociados a ese `doc_id`.
