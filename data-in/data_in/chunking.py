from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from data_in.extract import TextBlock
from data_in.schema import ChunkJSON, ChunkMetadata
from data_in.utils import normalize_text, split_paragraphs, split_sentences

# Embedding constraints (granite-embedding-multilingual): max 512 tokens, 768-dim vectors.
MAX_EMBED_TOKENS = 512
EMBED_DIM = 768
_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


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
    strategy = strategy.lower()
    if strategy == "by_tokens" and chunk_size > MAX_EMBED_TOKENS:
        chunk_size = MAX_EMBED_TOKENS
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(chunk_size - 1, 0)

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

    chunks = _split_oversize_chunks(chunks, doc_id)
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
    chunk_hash = make_chunk_hash(text_norm)

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
        return count_tokens
    return lambda text: max(1, len(text))

def count_tokens(text: str) -> int:
    return _token_count(text)


def _token_count(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return max(1, len(encoding.encode(text)))
    except Exception:
        return max(1, len(_FALLBACK_TOKEN_RE.findall(text)))


def _tokenize(text: str) -> Tuple[Optional[Sequence[int]], Optional[List[str]]]:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode(text), None
    except Exception:
        return None, _FALLBACK_TOKEN_RE.findall(text)


def split_text_by_tokens(text: str, max_tokens: int) -> List[str]:
    if max_tokens <= 0:
        return [text]
    token_ids, fallback_tokens = _tokenize(text)
    if token_ids is not None:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding("cl100k_base")
        parts = []
        for i in range(0, len(token_ids), max_tokens):
            parts.append(encoding.decode(token_ids[i : i + max_tokens]))
        return parts

    if not fallback_tokens:
        return [text]
    parts = []
    for i in range(0, len(fallback_tokens), max_tokens):
        parts.append(" ".join(fallback_tokens[i : i + max_tokens]))
    return parts


def make_chunk_hash(text: str) -> str:
    text_norm = normalize_text(text)
    return hashlib.sha1(text_norm.encode("utf-8")).hexdigest()


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


def _split_oversize_chunks(chunks: List[ChunkJSON], doc_id: str) -> List[ChunkJSON]:
    if not chunks:
        return chunks
    normalized: List[ChunkJSON] = []
    for chunk in chunks:
        token_count = count_tokens(chunk.text)
        if token_count <= MAX_EMBED_TOKENS:
            normalized.append(chunk)
            continue

        parts = split_text_by_tokens(chunk.text, MAX_EMBED_TOKENS)
        if not parts:
            normalized.append(chunk)
            continue

        source_text = chunk.text
        search_pos = 0
        for part in parts:
            part_norm = normalize_text(part)
            found = source_text.find(part_norm, search_pos)
            if found == -1:
                found = search_pos
            start = found
            end = start + len(part_norm)
            search_pos = end

            metadata = chunk.metadata.model_dump()
            metadata["chunk_hash"] = make_chunk_hash(part_norm)
            metadata["char_start"] = chunk.metadata.char_start + start
            metadata["char_end"] = chunk.metadata.char_start + end

            normalized.append(
                ChunkJSON(
                    chunk_index=-1,
                    chunk_id=None,
                    text=part_norm,
                    metadata=ChunkMetadata(**metadata),
                )
            )

    reindexed: List[ChunkJSON] = []
    for idx, chunk in enumerate(normalized):
        reindexed.append(
            chunk.model_copy(
                update={"chunk_index": idx, "chunk_id": f"{doc_id}:{idx}"}
            )
        )
    return reindexed
