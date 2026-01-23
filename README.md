# rag-stack

Stack mínimo de **RAG (Retrieval-Augmented Generation)** para indexar texto y consultarlo con un LLM, usando:
- **FastAPI + LlamaIndex** (`rag-api/`) para ingestión y consulta.
- **Postgres + pgvector** (`postgres/`) como almacén de embeddings.
- **Docker Compose** (`docker-compose.yml`) para orquestación.
- **Docker Model Runner (OpenAI-compatible)** para servir modelos de chat y embeddings (vía endpoints `/v1`).
- (Opcional) **Open WebUI** como UI para chat y pruebas rápidas.

## ¿Qué incluye?
- API con endpoints: `GET /healthz`, `POST /ingest`, `POST /query` (ver `rag-api/README.md`).
- Chunking configurable (tamaño/overlap) y filtros por metadata (según versión de LlamaIndex).
- Persistencia en Postgres: el índice vive en `pgvector` (no depende de RAM).

## RAG-API: motivación y futuro

La idea de `rag-api` es separar el “experimento” del “servicio”.

- **Motivación**: Open WebUI es excelente para probar modelos, prompts y UX; pero cuando quieres integrar RAG en una app (web/backend/bot), necesitas una **API estable** que haga ingestión + retrieval + respuesta del modelo, con control de parámetros, metadatos y trazabilidad.
- **Qué se busca**: un servicio HTTP simple (FastAPI) que, dado un `question`, recupere contexto desde `pgvector` y responda con el LLM, devolviendo además **fuentes** para auditar.
- **Futuro esperado**:
  - **Rendimiento**: colas/background jobs para ingestión, caching, batch embeddings, streaming de respuestas y ajuste fino de chunking/retrieval.
  - **Seguridad**: auth robusta (JWT/OAuth), rate limiting, separación por tenant, validación de payloads y políticas de CORS.
  - **Observabilidad**: logs estructurados, métricas (latencia/recall), trazas y evaluaciones de calidad de RAG.

## Ventajas y bondades
- **Reproducible**: todo corre por contenedores, sin “instalación mágica”.
- **Modular**: puedes cambiar modelo/runner sin tocar el código mientras sea OpenAI-compatible.
- **Persistente**: embeddings y metadata quedan en Postgres, apto para reinicios.
- **Simple de integrar**: API HTTP con `curl` (ideal para prototipos o backends).

## Deficiencias y cosas a tomar en cuenta
- No hay suite de tests automatizados todavía (solo verificación manual).
- El rendimiento/costo depende mucho del modelo elegido y del hardware.
- No hay pipeline de extracción de PDFs/HTML: **la API recibe texto ya limpio**.
- Seguridad básica: `RAG_API_KEY` protege por header, pero no hay OAuth/JWT ni rate limiting.
- La “estrictez” anti-alucinación es una instrucción al LLM; no es garantía formal.

## Requisitos
- Docker + Docker Compose.
- Postgres/puerto local `127.0.0.1:5432` y API en `127.0.0.1:8000` (por defecto).
- Variables de entorno (ver más abajo).

## Instalación: Docker Model Runner (modelos)

1) Instala el plugin (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y docker-model-plugin
```

Si Docker no detecta el plugin, reinicia el servicio:
```bash
sudo systemctl restart docker
```

2) Verifica que el comando exista:
```bash
docker model --help
```

3) Instala y levanta el runner escuchando en localhost (si ya lo tienes, omite lo que no aplique):
```bash
docker model stop-runner
docker model uninstall-runner
docker model install-runner --host 127.0.0.1 --port 12434
docker model start-runner
```

4) Descarga el modelo:
```bash
docker model pull ai/smollm2
```

5) Ejecuta el modelo (lo deja disponible para servir solicitudes):
```bash
docker model run ai/smollm2
```

6) Uso en terminal (CLI):

- Ejecutar el modelo en modo interactivo (si aplica al modelo/runner):
```bash
docker model run ai/smollm2
```

- Ejecutar con un prompt directo:
```bash
docker model run ai/smollm2 --prompt "Calculate derivative of f(x) = exp(x)"
```

7) Verifica qué tienes descargado y qué está cargado/ejecutándose:
```bash
docker model list
docker model ps
```

8) Para quitarlo de memoria (sin borrarlo del disco), usa `unload`:
```bash
docker model unload smollm2
```

9) Para borrarlo del disco, usa `rm`:
```bash
docker model rm ai/smollm2
```

Nota: el `docker-compose.yml` está preparado para consumir endpoints OpenAI-compatible (URLs que terminan en `/v1`) usando un único `OPENAI_API_BASE` para chat y embeddings.

## Configuración (variables de entorno)

Para `rag-api` (requeridas):
- `DATABASE_URL`
- `OPENAI_API_BASE` (OpenAI-compatible, debe terminar en `/v1`)
- `LLM_MODEL`
- `EMBED_MODEL`

Opcionales:
- `EMBED_DIM` (fijo en 768 para `ai/granite-embedding-multilingual`; si se define debe ser 768)
- `OPENAI_API_KEY` (puede ser dummy en local)
- `RAG_API_KEY` (si se define, exige header `X-API-Key`)

### `.env` para `docker compose`

Para ejecutar el stack con `docker compose`, crea un archivo `.env` en la raíz del repo (ejemplo mínimo):
```bash
POSTGRES_PASSWORD=changeme
# URL-encode de la contraseña (si tiene caracteres especiales)
POSTGRES_PASSWORD_URLENC=changeme

WEBUI_SECRET_KEY=changeme
OPENAI_API_KEY=sk-local

# URL OpenAI-compatible del runner (desde contenedores)
OPENAI_API_BASE=http://model-runner:12434/v1

# Nombres de modelos tal como aparecen en `docker model list`
LLM_MODEL=ai/granite-4.0-h-small:32B-Q4_K_M
EMBED_MODEL=ai/granite-embedding-multilingual:278M-F16
```

Tip: para generar `POSTGRES_PASSWORD_URLENC`:
```bash
python3 -c 'import urllib.parse; print(urllib.parse.quote_plus("changeme"))'
```

En Compose, `DATABASE_URL` apunta al Postgres interno y los servicios consumen `OPENAI_API_BASE`, `LLM_MODEL` y `EMBED_MODEL` para conectarse al runner.

## Ejecutar el stack

Levantar todo:
```bash
docker compose up --build
```

Servicios típicos:
- API: `http://localhost:8000`
- Open WebUI: `http://localhost:3000`
- Postgres: `localhost:5432` (vinculado a `127.0.0.1`)

Logs:
```bash
docker compose logs -f rag-api
```

## Uso de Open WebUI

Open WebUI es una interfaz web para conversar con el/los modelos que expone el runner (y para pruebas rápidas).

1) Abre `http://localhost:3000/` y crea el usuario inicial (primer arranque).
2) En el chat, selecciona el modelo (dropdown de modelo) y envía prompts como en cualquier UI tipo “Chat”.
   - Barra lateral: historial de conversaciones y creación de nuevos chats.
   - Barra superior: selector de modelo (y opciones del chat según la versión).
3) Si no aparecen modelos o falla la conexión, revisa en la configuración de Open WebUI que el backend apunte al endpoint OpenAI-compatible (y que el runner esté activo).

Nota: Open WebUI es independiente de `rag-api`. Para el flujo RAG de esta repo, usa `POST /ingest` y `POST /query` (ver `rag-api/README.md`).

## Uso rápido (API)

Healthcheck:
```bash
curl http://localhost:8000/healthz
```

Ejemplos de ingest/query: ver `rag-api/README.md`.

## Arquitectura (alto nivel)
- `POST /ingest`: recibe texto + metadata → chunking → embeddings → escritura en pgvector.
- `POST /query`: recupera top-k similares (y filtros) → arma contexto → LLM responde → devuelve respuesta + fuentes.

## Troubleshooting
- `Missing required env vars`: revisa `OPENAI_API_BASE` y que termine en `/v1`.
- `EMBED_DIM is fixed to 768`: estás intentando cambiar la dimensión; debe ser 768.
- Si la DB no conecta: confirma `POSTGRES_PASSWORD_URLENC` y el estado de `postgres` en `docker compose ps`.
