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
python data-in/cli.py enrich --input ./data-out --mode chunk --max-chunks 200
```

Subir a `rag-api` (opcion A):

```bash
python data-in/cli.py push --input ./data-out --rag-api http://localhost:8000 --mode ingest
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
