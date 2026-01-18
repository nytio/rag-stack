import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_in import __version__
from data_in.chunking import build_chunks
from data_in.config import load_config, merge_config
from data_in.discovery import DocumentUnit, discover_documents
from data_in.export import export_document, write_manifest
from data_in.extract import ExtractedDocument
from data_in.extract.pdf import extract_pdf_document
from data_in.extract.text import extract_text_document
from data_in.schema import AssetJSON, DocumentJSON, DocumentMeta, ExtractionStats, RunInfo
from data_in.utils import now_iso
from data_in.enrich import enrich_documents
from data_in.push_api import push_documents


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        return run_preprocess(args)
    if args.command == "enrich":
        return run_enrich(args)
    if args.command == "push":
        return run_push(args)

    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-ingest pipeline for RAG data.")
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess", help="discover -> extract -> chunk -> export")
    preprocess.add_argument("--input", "-i", required=True, help="File or folder input")
    preprocess.add_argument("--output", "-o", default="data-out", help="Output folder")
    preprocess.add_argument("--glob", default=None, help="Glob pattern (e.g. *.pdf)")
    preprocess.add_argument("--recursive", action="store_true", help="Search recursively")
    preprocess.add_argument("--ignore", action="append", default=[], help="Folder names to ignore")
    preprocess.add_argument("--config", default=None, help="Config YAML file")
    preprocess.add_argument("--chunk-size", type=int, default=None)
    preprocess.add_argument("--chunk-overlap", type=int, default=None)
    preprocess.add_argument(
        "--strategy",
        default=None,
        choices=["by_chars", "by_sentences", "by_tokens"],
    )
    preprocess.add_argument(
        "--doc-id-mode", default=None, choices=["hash", "hash+path"]
    )
    preprocess.add_argument("--ocr", action="store_true", help="Force OCR for PDFs")
    preprocess.add_argument("--ocr-auto", action="store_true", help="Auto OCR if needed")
    preprocess.add_argument("--ocr-dpi", type=int, default=None)
    preprocess.add_argument(
        "--ocr-images", action="store_true", help="Run OCR on extracted PDF images"
    )
    preprocess.add_argument("--jsonl", action="store_true", help="Also export chunks.jsonl")

    enrich = sub.add_parser("enrich", help="LLM enrichment")
    enrich.add_argument("--input", "-i", required=True, help="Output folder")
    enrich.add_argument(
        "--mode",
        default="none",
        choices=["none", "doc", "chunk", "all"],
    )
    enrich.add_argument("--llm-model", default=None)
    enrich.add_argument("--openai-api-base", default=None)
    enrich.add_argument("--openai-api-key", default=None)
    enrich.add_argument("--max-chunks", type=int, default=200)
    enrich.add_argument("--concurrency", type=int, default=2)
    enrich.add_argument("--cache-dir", default=None)
    enrich.add_argument("--overwrite", action="store_true")

    push = sub.add_parser("push", help="Send output to rag-api")
    push.add_argument("--input", "-i", required=True, help="Output folder")
    push.add_argument("--rag-api", required=True, help="Base URL for rag-api")
    push.add_argument("--api-key", default=None)
    push.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "ingest", "ingest_chunks"],
    )
    push.add_argument("--chunk-size", type=int, default=None)
    push.add_argument("--chunk-overlap", type=int, default=None)
    push.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
    push.add_argument(
        "--chunk-batch-size",
        type=int,
        default=None,
        help="Max chunks per /ingest_chunks request",
    )
    push.add_argument("--force", action="store_true", help="Ignore local idempotency cache")

    return parser


def run_preprocess(args: argparse.Namespace) -> int:
    base_cfg = load_config(args.config)
    cfg = merge_config(
        base_cfg,
        {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "chunk_strategy": args.strategy,
            "doc_id_mode": args.doc_id_mode,
            "ocr": args.ocr if args.ocr else None,
            "ocr_auto": args.ocr_auto if args.ocr_auto else None,
            "ocr_dpi": args.ocr_dpi,
            "ocr_images": args.ocr_images if args.ocr_images else None,
        },
    )

    documents = discover_documents(
        args.input,
        glob_pattern=args.glob,
        recursive=args.recursive,
        ignore=args.ignore,
        doc_id_mode=cfg.doc_id_mode,
    )

    outputs: List[DocumentJSON] = []
    run_config = {
        "generated_at": now_iso(),
        "tool_version": __version__,
        "config": {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "chunk_strategy": cfg.chunk_strategy,
            "doc_id_mode": cfg.doc_id_mode,
            "ocr": cfg.ocr,
            "ocr_auto": cfg.ocr_auto,
            "ocr_dpi": cfg.ocr_dpi,
            "ocr_images": cfg.ocr_images,
        },
    }

    for doc in documents:
        extracted = extract_document(doc, cfg, output_dir=Path(args.output))
        extraction_stats = build_extraction_stats(extracted)
        meta = DocumentMeta(
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            source_type=doc.source_type,
            file_hash_sha256=doc.file_hash_sha256,
            title=extracted.title,
            metadata={
                "file_size_bytes": doc.file_size_bytes,
                "modified_at": doc.modified_at,
            },
            extraction=extraction_stats,
        )

        chunks = build_chunks(
            extracted.blocks,
            doc.doc_id,
            cfg.chunk_size,
            cfg.chunk_overlap,
            cfg.chunk_strategy,
        )

        assets = [
            AssetJSON(
                type=asset.type,
                path=asset.path,
                page_number=asset.page_number,
                caption=asset.caption,
                metadata=asset.metadata,
            )
            for asset in extracted.assets
        ]

        document_json = DocumentJSON(
            document=meta,
            chunks=chunks,
            assets=assets,
            run=RunInfo(
                generated_at=run_config["generated_at"],
                tool_version=run_config["tool_version"],
                config=run_config["config"],
            ),
        )

        export_document(document_json, args.output, pretty=True, jsonl=args.jsonl)
        outputs.append(document_json)

    write_manifest(args.output, run_config, outputs)
    return 0


def extract_document(doc: DocumentUnit, cfg, output_dir: Path) -> ExtractedDocument:
    if doc.source_type == "pdf":
        assets_dir = output_dir / doc.doc_id / "assets"
        return extract_pdf_document(
            doc.source_path,
            ocr=cfg.ocr,
            ocr_auto=cfg.ocr_auto,
            ocr_dpi=cfg.ocr_dpi,
            assets_dir=str(assets_dir),
            ocr_images=cfg.ocr_images,
        )
    return extract_text_document(doc.source_path)


def build_extraction_stats(extracted: ExtractedDocument) -> ExtractionStats:
    if extracted.pages:
        pages = [
            {
                "page_number": page.page_number,
                "text": page.text,
                "char_count": page.char_count,
            }
            for page in extracted.pages
        ]
        char_total = sum(page["char_count"] for page in pages)
        return ExtractionStats(
            pages=pages,
            char_count_total=char_total,
            warnings=extracted.warnings,
            ocr_used=extracted.ocr_used,
            ocr_engine=extracted.ocr_engine,
        )

    text = "\n\n".join(block.text for block in extracted.blocks)
    return ExtractionStats(
        pages=[{"page_number": 1, "text": text, "char_count": len(text)}],
        char_count_total=len(text),
        warnings=extracted.warnings,
        ocr_used=extracted.ocr_used,
        ocr_engine=extracted.ocr_engine,
    )


def run_enrich(args: argparse.Namespace) -> int:
    base_cfg = load_config(None)
    cfg = merge_config(
        base_cfg,
        {
            "llm_model": args.llm_model,
            "openai_api_base": args.openai_api_base,
            "openai_api_key": args.openai_api_key,
        },
    )
    if args.mode == "none":
        return 0
    if not cfg.openai_api_base or not cfg.llm_model:
        raise SystemExit("OPENAI_API_BASE y LLM_MODEL son requeridos para enrich")

    enrich_documents(
        input_dir=args.input,
        mode=args.mode,
        openai_api_base=cfg.openai_api_base,
        llm_model=cfg.llm_model,
        openai_api_key=cfg.openai_api_key,
        max_chunks=args.max_chunks,
        concurrency=args.concurrency,
        cache_dir=args.cache_dir,
        overwrite=args.overwrite,
    )
    return 0


def run_push(args: argparse.Namespace) -> int:
    results = push_documents(
        input_dir=args.input,
        rag_api=args.rag_api,
        api_key=args.api_key,
        mode=args.mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force=args.force,
        timeout=args.timeout,
        chunk_batch_size=args.chunk_batch_size,
    )
    for doc_id, status in results:
        print(f"{doc_id}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
