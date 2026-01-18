from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

from data_in.chunking import MAX_EMBED_TOKENS, count_tokens
from data_in.chunking import (
    MAX_EMBED_TOKENS,
    count_tokens,
    make_chunk_hash,
    split_text_by_tokens,
)
from data_in.schema import DocumentJSON
from data_in.utils import normalize_text

STATE_FILE = ".push_state.json"


def load_documents(input_dir: str) -> List[DocumentJSON]:
    base = Path(input_dir)
    docs = []
    for doc_path in base.glob("*/document.json"):
        with doc_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        docs.append(DocumentJSON.model_validate(data))
    return docs


def push_documents(
    input_dir: str,
    rag_api: str,
    api_key: Optional[str] = None,
    mode: str = "auto",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    force: bool = False,
    timeout: float = 60.0,
    chunk_batch_size: Optional[int] = None,
) -> List[Tuple[str, int]]:
    docs = load_documents(input_dir)
    results: List[Tuple[str, int]] = []
    base_url = rag_api.rstrip("/")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    state = _load_state(input_dir)
    target_state = state.setdefault("targets", {}).setdefault(
        base_url, {"file_hashes": [], "chunk_hashes": {}}
    )
    file_hashes = set(target_state.get("file_hashes", []))
    chunk_hashes_map = {
        doc_id: set(hashes) for doc_id, hashes in target_state.get("chunk_hashes", {}).items()
    }

    with httpx.Client(timeout=timeout) as client:
        for doc in docs:
            status, sent_chunk_hashes = _push_document(
                client,
                base_url,
                headers,
                doc,
                mode,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                force=force,
                file_hashes=file_hashes,
                chunk_hashes_map=chunk_hashes_map,
                chunk_batch_size=chunk_batch_size,
            )
            results.append((doc.document.doc_id, status))
            if _is_success(status):
                file_hashes.add(doc.document.file_hash_sha256)
                if sent_chunk_hashes is None:
                    chunk_hashes_map[doc.document.doc_id] = {
                        chunk.metadata.chunk_hash for chunk in doc.chunks
                    }
                else:
                    existing = chunk_hashes_map.get(doc.document.doc_id, set())
                    existing.update(sent_chunk_hashes)
                    chunk_hashes_map[doc.document.doc_id] = existing

    target_state["file_hashes"] = sorted(file_hashes)
    target_state["chunk_hashes"] = {
        doc_id: sorted(list(hashes)) for doc_id, hashes in chunk_hashes_map.items()
    }
    _save_state(input_dir, state)

    return results


def _push_document(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    doc: DocumentJSON,
    mode: str,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    force: bool,
    file_hashes: set,
    chunk_hashes_map: Dict[str, set],
    chunk_batch_size: Optional[int],
) -> Tuple[int, Optional[List[str]]]:
    mode = mode.lower()
    doc_hash = doc.document.file_hash_sha256

    if not force and mode in {"ingest", "auto"} and doc_hash in file_hashes:
        return 208, None

    if mode in {"ingest_chunks", "auto"}:
        chunks_to_send, chunk_hashes = _filter_chunks(doc, chunk_hashes_map, force)
        if not chunks_to_send and not force:
            return 208, []
        status = _post_ingest_chunks_batched(
            client,
            base_url,
            headers,
            doc,
            chunks_to_send,
            chunk_batch_size=chunk_batch_size,
        )
        if status != 404 or mode == "ingest_chunks":
            return status, chunk_hashes

    status = _post_ingest(client, base_url, headers, doc, chunk_size, chunk_overlap)
    return status, None


def _post_ingest(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    doc: DocumentJSON,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
) -> int:
    url = f"{base_url}/ingest"
    text = _document_text(doc)
    metadata = dict(doc.document.metadata)
    metadata.update(
        {
            "source_path": doc.document.source_path,
            "file_hash_sha256": doc.document.file_hash_sha256,
            "chunk_count": len(doc.chunks),
        }
    )
    payload: Dict[str, Any] = {
        "doc_id": doc.document.doc_id,
        "text": text,
        "metadata": metadata,
    }
    if chunk_size is not None:
        if chunk_size > MAX_EMBED_TOKENS:
            chunk_size = MAX_EMBED_TOKENS
        payload["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        if chunk_size is not None and chunk_overlap >= chunk_size:
            chunk_overlap = max(chunk_size - 1, 0)
        payload["chunk_overlap"] = chunk_overlap

    response = client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.status_code


def _post_ingest_chunks(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    doc: DocumentJSON,
    chunks: List[Any],
) -> int:
    url = f"{base_url}/ingest_chunks"
    payload = {
        "doc_id": doc.document.doc_id,
        "chunks": chunks,
        "metadata": doc.document.metadata,
    }

    response = client.post(url, json=payload, headers=headers)
    if response.status_code == 404:
        return 404
    response.raise_for_status()
    return response.status_code


def _post_ingest_chunks_batched(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    doc: DocumentJSON,
    chunks: List[Any],
    chunk_batch_size: Optional[int],
) -> int:
    if not chunks:
        return 200
    if not chunk_batch_size or chunk_batch_size <= 0 or len(chunks) <= chunk_batch_size:
        return _post_ingest_chunks(client, base_url, headers, doc, chunks)

    status = 200
    for i in range(0, len(chunks), chunk_batch_size):
        batch = chunks[i : i + chunk_batch_size]
        status = _post_ingest_chunks(client, base_url, headers, doc, batch)
        if status == 404:
            return 404
    return status


def _document_text(doc: DocumentJSON) -> str:
    if doc.document.extraction.pages:
        return "\n\n".join(page.text for page in doc.document.extraction.pages)
    return "\n\n".join(chunk.text for chunk in doc.chunks)


def _filter_chunks(
    doc: DocumentJSON, chunk_hashes_map: Dict[str, set], force: bool
) -> Tuple[List[Dict[str, Any]], List[str]]:
    chunks: List[Dict[str, Any]] = []
    hashes: List[str] = []
    seen = chunk_hashes_map.get(doc.document.doc_id, set())
    for chunk in doc.chunks:
        if not force and chunk.metadata.chunk_hash in seen:
            continue
        token_count = count_tokens(chunk.text)
        if token_count > MAX_EMBED_TOKENS:
            parts = split_text_by_tokens(chunk.text, MAX_EMBED_TOKENS)
            source_text = chunk.text
            search_pos = 0
            for part_idx, part in enumerate(parts, start=1):
                part_norm = normalize_text(part)
                found = source_text.find(part_norm, search_pos)
                if found == -1:
                    found = search_pos
                start = found
                end = start + len(part_norm)
                search_pos = end

                metadata = chunk.metadata.model_dump(exclude_none=True)
                metadata["chunk_hash"] = make_chunk_hash(part_norm)
                metadata["char_start"] = chunk.metadata.char_start + start
                metadata["char_end"] = chunk.metadata.char_start + end

                item = {"text": part_norm, "metadata": metadata}
                base_id = chunk.chunk_id or f"{doc.document.doc_id}:{chunk.chunk_index}"
                item["chunk_id"] = f"{base_id}.part{part_idx}"
                chunks.append(item)
                hashes.append(metadata["chunk_hash"])
            continue

        item = {
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump(exclude_none=True),
        }
        if chunk.chunk_id:
            item["chunk_id"] = chunk.chunk_id
        chunks.append(item)
        hashes.append(chunk.metadata.chunk_hash)
    return chunks, hashes


def _is_success(status: int) -> bool:
    return 200 <= status < 300 or status == 208


def _load_state(input_dir: str) -> Dict[str, Any]:
    path = Path(input_dir) / STATE_FILE
    if not path.exists():
        return {"version": 1, "targets": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_state(input_dir: str, state: Dict[str, Any]) -> None:
    path = Path(input_dir) / STATE_FILE
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
