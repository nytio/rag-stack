from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable, List


_WHITESPACE_RE = re.compile(r"[ \t\r]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S+", stripped):
        return True
    if stripped.isupper() and len(stripped) <= 80:
        return True
    if stripped.endswith(":") and len(stripped) <= 80:
        return True
    return False


def strip_repeated_headers(pages: Iterable[str], threshold: float = 0.6) -> List[str]:
    page_list = list(pages)
    if not page_list:
        return []
    total_pages = len(page_list)
    line_counts = {}
    page_lines = []

    for text in page_list:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_lines.append(lines)
        for line in set(lines):
            line_counts[line] = line_counts.get(line, 0) + 1

    repeated = {
        line
        for line, count in line_counts.items()
        if count / total_pages >= threshold and len(line) < 200
    }

    cleaned_pages = []
    for lines in page_lines:
        kept = [line for line in lines if line not in repeated]
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages
