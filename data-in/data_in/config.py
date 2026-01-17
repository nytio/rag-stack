import os
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    chunk_size: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    chunk_strategy: str = Field(default="by_chars")
    doc_id_mode: str = Field(default="hash")
    ocr: bool = False
    ocr_auto: bool = False
    ocr_dpi: int = 300
    ocr_images: bool = False
    llm_model: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_key: Optional[str] = None
    enrich_mode: str = Field(default="none")
    max_chunks_enrich: int = 200
    concurrency: int = 2

    def apply_env(self) -> "AppConfig":
        data = self.model_dump()
        data["chunk_size"] = int(os.getenv("DATA_IN_CHUNK_SIZE", data["chunk_size"]))
        data["chunk_overlap"] = int(os.getenv("DATA_IN_CHUNK_OVERLAP", data["chunk_overlap"]))
        data["chunk_strategy"] = os.getenv("DATA_IN_CHUNK_STRATEGY", data["chunk_strategy"])
        data["doc_id_mode"] = os.getenv("DATA_IN_DOC_ID_MODE", data["doc_id_mode"])
        data["ocr"] = _env_bool("DATA_IN_OCR", data["ocr"])
        data["ocr_auto"] = _env_bool("DATA_IN_OCR_AUTO", data["ocr_auto"])
        data["ocr_dpi"] = int(os.getenv("DATA_IN_OCR_DPI", data["ocr_dpi"]))
        data["ocr_images"] = _env_bool("DATA_IN_OCR_IMAGES", data["ocr_images"])
        data["llm_model"] = os.getenv("LLM_MODEL", data["llm_model"])
        data["openai_api_base"] = os.getenv("OPENAI_API_BASE", data["openai_api_base"])
        data["openai_api_key"] = os.getenv("OPENAI_API_KEY", data["openai_api_key"])
        data["enrich_mode"] = os.getenv("DATA_IN_ENRICH_MODE", data["enrich_mode"])
        data["max_chunks_enrich"] = int(os.getenv("DATA_IN_MAX_CHUNKS_ENRICH", data["max_chunks_enrich"]))
        data["concurrency"] = int(os.getenv("DATA_IN_CONCURRENCY", data["concurrency"]))
        return AppConfig(**data)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: Optional[str]) -> AppConfig:
    if not path:
        return AppConfig().apply_env()

    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = AppConfig(**raw)
    return cfg.apply_env()


def merge_config(cfg: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    data = cfg.model_dump()
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    return AppConfig(**data)
