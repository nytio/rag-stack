from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_in.schema import DocumentJSON


def export_document(
    document: DocumentJSON,
    output_dir: str,
    pretty: bool = True,
    jsonl: bool = False,
) -> Path:
    out_root = Path(output_dir)
    doc_dir = out_root / document.document.doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_path = doc_dir / "document.json"

    payload = document.model_dump(exclude_none=True)
    with doc_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2 if pretty else None, ensure_ascii=False)
        handle.write("\n")

    if jsonl:
        jsonl_path = doc_dir / "chunks.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for chunk in document.chunks:
                line = {
                    "doc_id": document.document.doc_id,
                    **chunk.model_dump(exclude_none=True),
                }
                handle.write(json.dumps(line, ensure_ascii=False))
                handle.write("\n")

    return doc_path


def write_manifest(
    output_dir: str,
    run_config: Dict[str, Any],
    documents: Iterable[DocumentJSON],
) -> Path:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"

    docs: List[Dict[str, Any]] = []
    for doc in documents:
        docs.append(
            {
                "doc_id": doc.document.doc_id,
                "source_path": doc.document.source_path,
                "source_type": doc.document.source_type,
                "chunks": len(doc.chunks),
                "warnings": doc.document.extraction.warnings,
            }
        )

    payload = {
        "generated_at": run_config.get("generated_at"),
        "tool_version": run_config.get("tool_version"),
        "config": run_config.get("config", {}),
        "documents": docs,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return manifest_path
