from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextBlock:
    text: str
    source_path: str
    page_number: Optional[int] = None
    heading: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    section: Optional[str] = None


@dataclass
class PageText:
    page_number: int
    text: str
    char_count: int


@dataclass
class ExtractedDocument:
    blocks: List[TextBlock]
    pages: List[PageText] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ocr_used: bool = False
    ocr_engine: Optional[str] = None
    title: Optional[str] = None
    assets: List["Asset"] = field(default_factory=list)


@dataclass
class Asset:
    type: str
    path: str
    page_number: Optional[int] = None
    caption: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "TextBlock",
    "PageText",
    "ExtractedDocument",
    "Asset",
]
