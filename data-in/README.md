# data-in

Pipeline de pre-ingesta para generar JSON revisable por documento antes de subir a `rag-api`.

## Requisitos (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

> `poppler-utils` es necesario para `pdf2image` (OCR). Si no usas OCR, puedes omitir `tesseract-ocr`.
> Para extraer imagenes de PDFs se usa `PyMuPDF` (instalado via `requirements.txt`).

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r data-in/requirements.txt
```

## Uso rapido

Preprocesar documentos (descubrimiento + extraccion + chunking + export):

```bash
python data-in/cli.py preprocess --input ./docs --output ./data-out --recursive
```

Exportar tambien `chunks.jsonl`:

```bash
python data-in/cli.py preprocess --input ./docs --output ./data-out --jsonl
```

Forzar OCR en PDFs:

```bash
python data-in/cli.py preprocess --input ./docs --output ./data-out --ocr --recursive
```

OCR sobre imagenes extraidas de PDFs:

```bash
python data-in/cli.py preprocess --input ./docs --output ./data-out --ocr-images
```

Enriquecer con LLM (requiere `OPENAI_API_BASE` y `LLM_MODEL`):

```bash
python data-in/cli.py enrich \
  --input ./data-out \
  --mode chunk \
  --max-chunks 200 \
  --openai-api-base "http://localhost:12434/v1" \
  --llm-model "ai/gpt-oss"
```

Subir a `rag-api` (opcion A):

```bash
python data-in/cli.py push --input ./data-out --rag-api http://localhost:8000 --mode ingest
```

Subir chunks pre-construidos (opcion B, 1:1 con el JSON):

```bash
python data-in/cli.py push --input ./data-out --rag-api http://localhost:8000 --mode ingest_chunks
```

## Opcion A vs Opcion B

- **Opcion A (`/ingest`)**: se envia el documento completo y `rag-api` re-chunkea. Mas simple, pero los chunks revisados no quedan 1:1 con lo indexado. Util para prototipos o cuando no necesitas trazabilidad fina.
  - **Preprocess recomendado**: enfocar en **extraccion y limpieza** (normalizar espacios, remover headers/footers repetidos, OCR si aplica). El chunking es secundario porque lo controla `rag-api` (puedes ajustar `chunk_size`/`chunk_overlap` al hacer `push`).
  - **Comando ejemplo**:
    ```bash
    python data-in/cli.py preprocess --input ./docs --output ./data-out --recursive --ocr-auto
    ```
- **Opcion B (`/ingest_chunks`)**: se envian chunks pre-construidos. Conserva `chunk_index`, `page_start/end`, `heading_path` y garantiza 1:1 con el JSON. Recomendado cuando necesitas auditabilidad y control de calidad.
  - **Preprocess recomendado**: enfocar en **chunking de alta calidad** (parrafos completos, `chunk_size`/`chunk_overlap` afinados) y metadatos ricos por chunk. Revisar `document.json` antes de ingestar para asegurar cortes correctos y coherencia.
  - **Comando ejemplo**:
    ```bash
    python data-in/cli.py preprocess --input ./docs --output ./data-out --recursive --chunk-size 900 --chunk-overlap 120 --strategy by_chars
    ```

## Estructura

```
data-in/
  cli.py
  requirements.txt
  README.md
  data_in/
    config.py
    discovery.py
    extract/
    chunking.py
    enrich.py
    export.py
    push_api.py
    schema.py
```

## Artefactos generados

- `data-out/<doc_id>/document.json` (pretty-printed y listo para revision).
- `data-out/<doc_id>/chunks.jsonl` (un chunk por linea; util para pipelines streaming o cargas parciales).
- `data-out/manifest.json` (resumen de corrida).

## Configuracion

- Variables opcionales:
  - `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_MODEL`
  - `DATA_IN_CHUNK_SIZE`, `DATA_IN_CHUNK_OVERLAP`, `DATA_IN_CHUNK_STRATEGY`
  - `DATA_IN_DOC_ID_MODE`, `DATA_IN_OCR`, `DATA_IN_OCR_AUTO`
  - `DATA_IN_OCR_IMAGES`

## Notas de integracion

- El modo `push` soporta `ingest` (endpoint actual) y `ingest_chunks` (si se implementa en `rag-api`).
- En modo `auto`, intenta `ingest_chunks` y hace fallback a `ingest` si recibe 404.
- Se guarda un cache local en `data-out/.push_state.json` para idempotencia por `file_hash` y `chunk_hash`.
- `ingest_chunks` preserva `chunk_index`, `chunk_id`, `page_start/end` y demas metadatos generados en `data-in`.
