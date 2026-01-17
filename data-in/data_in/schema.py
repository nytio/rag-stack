from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExtractionPage(BaseModel):
    page_number: int
    text: str
    char_count: int


class ExtractionStats(BaseModel):
    pages: List[ExtractionPage] = Field(default_factory=list)
    char_count_total: int = 0
    warnings: List[str] = Field(default_factory=list)
    ocr_used: bool = False
    ocr_engine: Optional[str] = None


class DocumentMeta(BaseModel):
    doc_id: str
    source_path: str
    source_type: str
    file_hash_sha256: str
    title: Optional[str] = None
    created_at: Optional[str] = None
    ingested_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction: ExtractionStats


class ChunkMetadata(BaseModel):
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    heading: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    char_start: int
    char_end: int
    source_path: str
    chunk_hash: str


class ChunkEnrich(BaseModel):
    summary: Optional[str] = None
    questions: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ChunkJSON(BaseModel):
    chunk_index: int
    chunk_id: Optional[str] = None
    text: str
    metadata: ChunkMetadata
    enrich: Optional[ChunkEnrich] = None


class AssetJSON(BaseModel):
    type: str
    path: str
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunInfo(BaseModel):
    generated_at: str
    tool_version: str
    config: Dict[str, Any] = Field(default_factory=dict)


class DocumentJSON(BaseModel):
    schema_version: str = "1.0"
    document: DocumentMeta
    chunks: List[ChunkJSON] = Field(default_factory=list)
    assets: List[AssetJSON] = Field(default_factory=list)
    run: RunInfo
