from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from data_in.extract import ExtractedDocument, TextBlock
from data_in.utils import is_heading_line, normalize_text, split_paragraphs


def extract_text_document(path: str) -> ExtractedDocument:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".md":
        blocks, title = _parse_markdown(file_path)
    else:
        blocks, title = _parse_plain_text(file_path)

    return ExtractedDocument(blocks=blocks, title=title)


def _parse_markdown(file_path: Path) -> Tuple[List[TextBlock], Optional[str]]:
    blocks: List[TextBlock] = []
    heading_path: List[str] = []
    title: Optional[str] = None
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = normalize_text("\n".join(paragraph_lines))
        for para in split_paragraphs(text):
            blocks.append(
                TextBlock(
                    text=para,
                    source_path=str(file_path),
                    heading=heading_path[-1] if heading_path else None,
                    heading_path=list(heading_path),
                )
            )
        paragraph_lines.clear()

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush_paragraph()
            continue
        if line.lstrip().startswith("#"):
            flush_paragraph()
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            if level <= 0:
                continue
            heading_path[:] = heading_path[: max(level - 1, 0)]
            heading_path.append(heading_text)
            if title is None:
                title = heading_text
            blocks.append(
                TextBlock(
                    text=heading_text,
                    source_path=str(file_path),
                    heading=heading_text,
                    heading_path=list(heading_path),
                )
            )
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return blocks, title


def _parse_plain_text(file_path: Path) -> Tuple[List[TextBlock], Optional[str]]:
    blocks: List[TextBlock] = []
    heading_path: List[str] = []
    title: Optional[str] = None
    paragraph_lines: List[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = normalize_text("\n".join(paragraph_lines))
        for para in split_paragraphs(text):
            blocks.append(
                TextBlock(
                    text=para,
                    source_path=str(file_path),
                    heading=heading_path[-1] if heading_path else None,
                    heading_path=list(heading_path),
                )
            )
        paragraph_lines.clear()

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            flush_paragraph()
            continue
        if is_heading_line(line):
            flush_paragraph()
            heading_text = line.strip().strip(":")
            heading_path[:] = [heading_text]
            if title is None:
                title = heading_text
            blocks.append(
                TextBlock(
                    text=heading_text,
                    source_path=str(file_path),
                    heading=heading_text,
                    heading_path=list(heading_path),
                )
            )
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return blocks, title
