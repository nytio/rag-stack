from __future__ import annotations

import pytest

from tests.conftest import require_modules

require_modules(
    "fastapi",
    "httpx",
    "sqlalchemy",
    "llama_index",
    "llama_index.vector_stores.postgres",
    "llama_index.llms.openai_like",
)

from fastapi import HTTPException

import main

pytestmark = pytest.mark.unit


def test_build_filters_empty_returns_none() -> None:
    assert main._build_filters({}) is None
    assert main._build_filters({"a": None}) is None


@pytest.mark.skipif(main._FILTER_MODE == "none", reason="Metadata filters not supported in this build")
def test_build_filters_single_value() -> None:
    filters = main._build_filters({"doc_id": "doc-1"})
    assert filters is not None
    assert hasattr(filters, "filters")


@pytest.mark.skipif(main._FILTER_MODE != "advanced", reason="Advanced filter mode not available")
def test_build_filters_list_value_advanced() -> None:
    filters = main._build_filters({"tag": ["a", "b"], "doc_id": "doc-1"})
    assert filters is not None
    assert hasattr(filters, "filters")


@pytest.mark.skipif(main._FILTER_MODE != "exact", reason="Exact filter mode not available")
def test_build_filters_list_value_exact_raises() -> None:
    with pytest.raises(HTTPException):
        main._build_filters({"tag": ["a", "b"]})


@pytest.mark.skipif(main._FILTER_MODE != "none", reason="Filter mode is supported; this test expects none")
def test_build_filters_none_mode_raises() -> None:
    with pytest.raises(HTTPException):
        main._build_filters({"doc_id": "doc-1"})
