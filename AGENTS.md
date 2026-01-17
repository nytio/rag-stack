# Repository Guidelines

This repository hosts a small RAG stack built around a FastAPI service, Postgres/pgvector, and containerized model runners.

## Project Structure & Module Organization
- `rag-api/` contains the FastAPI app (`main.py`), dependency list (`requirements.txt`), and Dockerfile.
- `postgres/init/` holds database bootstrap SQL (e.g., enabling `pgvector`).
- `docker-compose.yml` orchestrates Postgres, Open WebUI, and the `rag-api` service.

## Build, Test, and Development Commands
- `docker compose up --build` builds and starts the full stack.
- `docker compose up postgres` runs only the database service.
- `docker compose logs -f rag-api` tails API logs.
- `pip install -r rag-api/requirements.txt` installs Python deps for local runs.
- `uvicorn main:app --reload --host 0.0.0.0 --port 8000` starts the API locally (run from `rag-api/` with env vars set).

## Coding Style & Naming Conventions
- Python uses 4-space indentation and PEP 8 conventions.
- Functions/variables are `snake_case`; Pydantic models are `PascalCase`.
- Environment variables are uppercase with underscores (see `rag-api/main.py`).
- SQL init files in `postgres/init/` use numeric prefixes (e.g., `01-...sql`).

## Testing Guidelines
- No automated test suite is present. Favor manual checks:
  - `curl http://localhost:8000/healthz`
  - Ingest/query examples in `rag-api/README.md`.
- Pytest suites (unit + integration) live under `rag-api/tests/`:
  - `pip install -r rag-api/requirements-dev.txt`
  - `python -m pytest -m unit` (unit tests)
  - `python -m pytest -m integration` (requires live DB + LLM/embeddings; set `RAG_API_BASE` for API E2E tests)

## Commit & Pull Request Guidelines
- Commit history favors short, descriptive messages (often Spanish). Keep them concise and imperative.
- PRs should include a clear description, affected services, and any required env/config changes.
- For API changes, include a sample request/response or `curl` snippet.

## Configuration & Security Notes
- Required env vars: `DATABASE_URL`, `OPENAI_API_BASE`, `LLM_MODEL`, `EMBED_MODEL`.
- `OPENAI_API_KEY` may be a dummy value for local usage.
- `RAG_API_KEY` enables optional header auth (`X-API-Key`).
