from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class DocumentUnit:
    source_path: str
    source_type: str
    file_size_bytes: int
    modified_at: str
    file_hash_sha256: str
    doc_id: str


def discover_documents(
    input_path: str,
    glob_pattern: Optional[str] = None,
    recursive: bool = False,
    ignore: Optional[Iterable[str]] = None,
    doc_id_mode: str = "hash",
) -> List[DocumentUnit]:
    path = Path(input_path)
    ignore_set = set(ignore or [])
    files: List[Path] = []

    if path.is_file():
        files = [path]
    else:
        if glob_pattern:
            pattern = f"**/{glob_pattern}" if recursive else glob_pattern
            files = list(path.glob(pattern))
        else:
            if recursive:
                files = [p for p in path.rglob("*") if p.is_file()]
            else:
                files = [p for p in path.iterdir() if p.is_file()]

    documents: List[DocumentUnit] = []
    for file_path in files:
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in ignore_set for part in file_path.parts):
            continue

        file_hash = sha256_file(file_path)
        doc_id = build_doc_id(file_hash, file_path, doc_id_mode)
        stat = file_path.stat()
        documents.append(
            DocumentUnit(
                source_path=str(file_path),
                source_type=file_path.suffix.lower().lstrip("."),
                file_size_bytes=stat.st_size,
                modified_at=iso_from_ts(stat.st_mtime),
                file_hash_sha256=file_hash,
                doc_id=doc_id,
            )
        )

    return documents


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_doc_id(file_hash: str, path: Path, mode: str) -> str:
    if mode == "hash+path":
        digest = hashlib.sha256()
        digest.update(file_hash.encode("utf-8"))
        digest.update(str(path).encode("utf-8"))
        return digest.hexdigest()
    return file_hash


def iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
