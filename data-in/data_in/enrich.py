from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from data_in.schema import ChunkEnrich, DocumentJSON


def enrich_documents(
    input_dir: str,
    mode: str,
    openai_api_base: str,
    llm_model: str,
    openai_api_key: Optional[str] = None,
    max_chunks: int = 200,
    concurrency: int = 2,
    cache_dir: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    if mode == "none":
        return

    docs = _load_documents(input_dir)
    cache_path = Path(cache_dir) if cache_dir else Path(input_dir) / ".cache"
    cache_path.mkdir(parents=True, exist_ok=True)

    client = OpenAIClient(openai_api_base, openai_api_key)

    for doc in docs:
        if mode in {"doc", "all"}:
            _enrich_document(doc, client, llm_model, cache_path, overwrite)
        if mode in {"chunk", "all"}:
            _enrich_chunks(
                doc,
                client,
                llm_model,
                cache_path,
                max_chunks=max_chunks,
                concurrency=concurrency,
                overwrite=overwrite,
            )
        _save_document(input_dir, doc)


def _load_documents(input_dir: str) -> List[DocumentJSON]:
    base = Path(input_dir)
    docs = []
    for path in base.glob("*/document.json"):
        with path.open("r", encoding="utf-8") as handle:
            docs.append(DocumentJSON.model_validate(json.load(handle)))
    return docs


def _save_document(input_dir: str, doc: DocumentJSON) -> None:
    path = Path(input_dir) / doc.document.doc_id / "document.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(doc.model_dump(exclude_none=True), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _enrich_document(
    doc: DocumentJSON,
    client: "OpenAIClient",
    model: str,
    cache_path: Path,
    overwrite: bool,
) -> None:
    if not overwrite and "doc_summary" in doc.document.metadata:
        return

    cache_key = f"doc-{doc.document.doc_id}-{model}.json"
    cached = _load_cache(cache_path, cache_key)
    if cached:
        doc.document.metadata.update(cached)
        return

    prompt = (
        "Resume el documento en 3-5 frases, lista 5-10 keywords y 5 preguntas "
        "que el documento responde. Devuelve JSON con keys: summary, keywords, questions."
    )
    response = client.chat_json(model, prompt, doc_text=_document_text(doc))
    payload = {
        "doc_summary": response.get("summary"),
        "doc_keywords": response.get("keywords", []),
        "doc_questions": response.get("questions", []),
    }
    doc.document.metadata.update(payload)
    _save_cache(cache_path, cache_key, payload)


def _enrich_chunks(
    doc: DocumentJSON,
    client: "OpenAIClient",
    model: str,
    cache_path: Path,
    max_chunks: int,
    concurrency: int,
    overwrite: bool,
) -> None:
    chunks = doc.chunks[:max_chunks]

    def worker(chunk_index: int) -> Dict[str, Any]:
        chunk = doc.chunks[chunk_index]
        cache_key = f"chunk-{chunk.metadata.chunk_hash}-{model}.json"
        cached = _load_cache(cache_path, cache_key)
        if cached:
            return {"index": chunk_index, "payload": cached}
        prompt = (
            "Genera un resumen corto (1-3 frases), 3-7 preguntas, "
            "keywords y entidades. Devuelve JSON con keys: summary, questions, keywords, entities, tags."
        )
        response = client.chat_json(model, prompt, doc_text=chunk.text)
        _save_cache(cache_path, cache_key, response)
        return {"index": chunk_index, "payload": response}

    futures = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        for idx, chunk in enumerate(chunks):
            if chunk.enrich and not overwrite:
                continue
            futures.append(executor.submit(worker, idx))

        for future in as_completed(futures):
            result = future.result()
            idx = result["index"]
            payload = result["payload"]
            doc.chunks[idx].enrich = ChunkEnrich(
                summary=payload.get("summary"),
                questions=payload.get("questions", []),
                keywords=payload.get("keywords", []),
                entities=payload.get("entities", []),
                tags=payload.get("tags", []),
            )


def _document_text(doc: DocumentJSON) -> str:
    if doc.document.extraction.pages:
        return "\n\n".join(page.text for page in doc.document.extraction.pages)
    return "\n\n".join(chunk.text for chunk in doc.chunks)


def _load_cache(cache_path: Path, key: str) -> Optional[Dict[str, Any]]:
    path = cache_path / key
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_cache(cache_path: Path, key: str, payload: Dict[str, Any]) -> None:
    path = cache_path / key
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


class OpenAIClient:
    def __init__(self, api_base: str, api_key: Optional[str]) -> None:
        if not api_base:
            raise ValueError("OPENAI_API_BASE is required for enrichment")
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or ""

    def chat_json(self, model: str, prompt: str, doc_text: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "Responde solo con JSON valido."},
            {
                "role": "user",
                "content": f"{prompt}\n\nTexto:\n{doc_text}",
            },
        ]
        payload = {"model": model, "messages": messages, "temperature": 0.2}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self._api_base}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _safe_json_parse(content)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


def _safe_json_parse(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"summary": content}
