# rag-api (FastAPI + LlamaIndex + Postgres/pgvector + Docker Model Runner)

## Endpoints
| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | /healthz | Estado del servicio y configuración básica |
| POST | /ingest | Indexa texto y metadata en pgvector |
| POST | /query | Recupera contexto y responde con fuentes |

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
  -d '{"doc_id":"doc-1","text":"Texto del documento...","metadata":{"filename":"manual.pdf","page":3,"chunk_index":12,"source":"docs/manual.pdf"},"chunk_size":900,"chunk_overlap":120}'
```

Parámetros del `-d`:
| Campo | Tipo | Requerido | Notas |
| --- | --- | --- | --- |
| doc_id | string | sí | identificador del documento; se guarda en metadata |
| text | string | sí | texto limpio; sin imágenes ni binarios |
| metadata | object | no | metadatos libres (p. ej. `filename`, `source`, `page`) |
| chunk_size | int | no | tamaño de chunk; mínimo 200, máximo 4000 |
| chunk_overlap | int | no | solapamiento; mínimo 0, máximo 1000 |

Ejemplos de Metadata recomendados:
- `page` o `page_number` (int): número de página del chunk.
- `page_start` / `page_end` (int): solo si el chunk cruza varias páginas.
- `filename` (string): nombre del archivo.
- `source` o `url` (string): origen del documento.
- `section` o `heading` (string): sección o encabezado si lo extraes.
- `chunk_index` (int): índice del chunk dentro del documento.

Sugerencias:
- Usa tipos consistentes (si `page` es int, siempre int).
- Mantén los valores cortos y útiles para filtros/seguimiento.
- Guarda solo metadatos que planeas filtrar en Query.

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
