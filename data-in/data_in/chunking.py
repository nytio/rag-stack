from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from data_in.extract import TextBlock
from data_in.schema import ChunkJSON, ChunkMetadata
from data_in.utils import normalize_text, split_paragraphs, split_sentences


@dataclass
class Segment:
    text: str
    source_path: str
    page_number: Optional[int]
    heading: Optional[str]
    heading_path: List[str] = field(default_factory=list)
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0


def build_chunks(
    blocks: List[TextBlock],
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
) -> List[ChunkJSON]:
    segments = _build_segments(blocks, chunk_size, strategy)
    _apply_offsets(segments)
    size_fn = _measure_factory(strategy)
    chunks: List[ChunkJSON] = []

    current: List[Segment] = []
    current_size = 0

    for seg in segments:
        seg_size = size_fn(seg.text)
        if not current:
            current = [seg]
            current_size = seg_size
            continue

        if current_size + seg_size <= chunk_size:
            current.append(seg)
            current_size += seg_size
            continue

        chunks.append(_segments_to_chunk(current, doc_id, len(chunks)))
        overlap_segments = _select_overlap(current, chunk_overlap, size_fn)
        current = overlap_segments + [seg]
        current_size = sum(size_fn(s.text) for s in current)

    if current:
        chunks.append(_segments_to_chunk(current, doc_id, len(chunks)))

    return chunks


def _build_segments(blocks: List[TextBlock], chunk_size: int, strategy: str) -> List[Segment]:
    segments: List[Segment] = []
    size_fn = _measure_factory(strategy)

    for block in blocks:
        normalized = normalize_text(block.text)
        for para in split_paragraphs(normalized):
            parts = _split_long_paragraph(para, chunk_size, size_fn)
            for part in parts:
                segments.append(
                    Segment(
                        text=part,
                        source_path=block.source_path,
                        page_number=block.page_number,
                        heading=block.heading,
                        heading_path=list(block.heading_path),
                        section=block.section,
                    )
                )

    return segments


def _apply_offsets(segments: List[Segment]) -> None:
    offset = 0
    for idx, seg in enumerate(segments):
        seg.char_start = offset
        seg.char_end = offset + len(seg.text)
        offset = seg.char_end
        if idx < len(segments) - 1:
            offset += 2


def _segments_to_chunk(segments: List[Segment], doc_id: str, chunk_index: int) -> ChunkJSON:
    text = "\n\n".join(seg.text for seg in segments)
    text_norm = normalize_text(text)
    chunk_hash = hashlib.sha1(text_norm.encode("utf-8")).hexdigest()

    page_numbers = [seg.page_number for seg in segments if seg.page_number is not None]
    page_start = min(page_numbers) if page_numbers else None
    page_end = max(page_numbers) if page_numbers else None

    heading_path = []
    heading = None
    for seg in segments:
        if seg.heading_path:
            heading_path = list(seg.heading_path)
            heading = seg.heading

    metadata = ChunkMetadata(
        page_start=page_start,
        page_end=page_end,
        heading=heading,
        heading_path=heading_path,
        section=segments[-1].section if segments else None,
        char_start=segments[0].char_start,
        char_end=segments[-1].char_end,
        source_path=segments[0].source_path,
        chunk_hash=chunk_hash,
    )

    return ChunkJSON(
        chunk_index=chunk_index,
        chunk_id=f"{doc_id}:{chunk_index}",
        text=text_norm,
        metadata=metadata,
    )


def _select_overlap(
    segments: List[Segment],
    overlap_size: int,
    size_fn: Callable[[str], int],
) -> List[Segment]:
    if overlap_size <= 0:
        return []
    overlap: List[Segment] = []
    size_acc = 0
    for seg in reversed(segments):
        seg_size = size_fn(seg.text)
        if size_acc + seg_size > overlap_size:
            break
        overlap.insert(0, seg)
        size_acc += seg_size
    return overlap


def _measure_factory(strategy: str) -> Callable[[str], int]:
    strategy = strategy.lower()
    if strategy == "by_sentences":
        return lambda text: max(1, len(split_sentences(text)))
    if strategy == "by_tokens":
        return _token_count
    return lambda text: max(1, len(text))


def _token_count(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return max(1, len(encoding.encode(text)))
    except Exception:
        return max(1, len(text.split()))


def _split_long_paragraph(
    paragraph: str, chunk_size: int, size_fn: Callable[[str], int]
) -> List[str]:
    if size_fn(paragraph) <= chunk_size:
        return [paragraph]

    sentences = split_sentences(paragraph)
    if not sentences:
        return [paragraph]

    parts: List[str] = []
    current: List[str] = []
    current_size = 0

    for sentence in sentences:
        sent_size = size_fn(sentence)
        if current_size + sent_size <= chunk_size:
            current.append(sentence)
            current_size += sent_size
            continue

        if current:
            parts.append(" ".join(current))
        current = [sentence]
        current_size = sent_size

    if current:
        parts.append(" ".join(current))

    final_parts: List[str] = []
    for part in parts:
        if size_fn(part) <= chunk_size:
            final_parts.append(part)
        else:
            final_parts.extend(_split_by_chars(part, chunk_size))

    return final_parts


def _split_by_chars(text: str, chunk_size: int) -> List[str]:
    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        parts.append(text[start:end])
        start = end
    return parts
