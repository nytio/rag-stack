from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

from data_in.extract import Asset, ExtractedDocument, PageText, TextBlock
from data_in.extract.ocr import ocr_pdf_to_pages
from data_in.utils import is_heading_line, normalize_text, split_paragraphs, strip_repeated_headers


def extract_pdf_document(
    path: str,
    ocr: bool = False,
    ocr_auto: bool = False,
    ocr_dpi: int = 300,
    assets_dir: Optional[str] = None,
    ocr_images: bool = False,
) -> ExtractedDocument:
    file_path = Path(path)
    warnings: List[str] = []
    page_texts: List[str] = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_texts.append(normalize_text(text))

    ocr_used = False
    ocr_engine: Optional[str] = None

    if ocr or (ocr_auto and _needs_ocr(page_texts)):
        try:
            page_texts = [normalize_text(text) for text in ocr_pdf_to_pages(path, dpi=ocr_dpi)]
            ocr_used = True
            ocr_engine = "tesseract"
            warnings.append("OCR was used to extract text from the PDF.")
        except Exception as exc:
            warnings.append(f"OCR failed: {exc}")

    page_texts = strip_repeated_headers(page_texts)

    blocks: List[TextBlock] = []
    pages: List[PageText] = []
    title: Optional[str] = None
    heading_path: List[str] = []
    assets: List[Asset] = []

    if assets_dir:
        extracted_assets, asset_warnings = _extract_pdf_images(
            path, assets_dir, ocr_images=ocr_images
        )
        assets.extend(extracted_assets)
        warnings.extend(asset_warnings)

    for idx, text in enumerate(page_texts, start=1):
        normalized = normalize_text(text)
        if not normalized:
            warnings.append(f"Page {idx} appears empty after extraction.")
            pages.append(PageText(page_number=idx, text="", char_count=0))
            continue

        pages.append(PageText(page_number=idx, text=normalized, char_count=len(normalized)))
        for paragraph in _page_to_paragraphs(normalized):
            if is_heading_line(paragraph):
                heading_text = paragraph.strip().strip(":")
                heading_path = [heading_text]
                if title is None:
                    title = heading_text
                blocks.append(
                    TextBlock(
                        text=heading_text,
                        source_path=str(file_path),
                        page_number=idx,
                        heading=heading_text,
                        heading_path=list(heading_path),
                    )
                )
            else:
                blocks.append(
                    TextBlock(
                        text=paragraph,
                        source_path=str(file_path),
                        page_number=idx,
                        heading=heading_path[-1] if heading_path else None,
                        heading_path=list(heading_path),
                    )
                )

    return ExtractedDocument(
        blocks=blocks,
        pages=pages,
        warnings=warnings,
        ocr_used=ocr_used,
        ocr_engine=ocr_engine,
        title=title,
        assets=assets,
    )


def _page_to_paragraphs(text: str) -> List[str]:
    paragraphs: List[str] = []
    for para in split_paragraphs(text):
        paragraphs.append(para)
    return paragraphs


def _needs_ocr(page_texts: List[str]) -> bool:
    if not page_texts:
        return False
    empty_pages = sum(1 for text in page_texts if len(text.strip()) < 40)
    return empty_pages / len(page_texts) >= 0.6


def _extract_pdf_images(
    path: str, assets_dir: str, ocr_images: bool = False
) -> Tuple[List[Asset], List[str]]:
    warnings: List[str] = []
    assets: List[Asset] = []
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)

    try:
        import fitz  # PyMuPDF
    except Exception:
        warnings.append("Asset extraction skipped: PyMuPDF is not installed.")
        return assets, warnings

    doc = fitz.open(path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        page_number = page_index + 1
        for img_index, img in enumerate(images):
            xref = img[0]
            image = doc.extract_image(xref)
            image_bytes = image.get("image")
            if not image_bytes:
                continue
            ext = image.get("ext", "png")
            filename = f"page-{page_number:03d}-img-{img_index:02d}.{ext}"
            output_path = assets_path / filename
            output_path.write_bytes(image_bytes)

            caption = None
            metadata: Dict[str, object] = {
                "width": image.get("width"),
                "height": image.get("height"),
                "xref": xref,
            }

            if ocr_images:
                ocr_text, ocr_warning = _ocr_image_bytes(image_bytes)
                if ocr_warning:
                    warnings.append(ocr_warning)
                if ocr_text:
                    caption = ocr_text
                    metadata["ocr_text"] = ocr_text

            assets.append(
                Asset(
                    type="image",
                    path=str(Path("assets") / filename),
                    page_number=page_number,
                    caption=caption,
                    metadata=metadata,
                )
            )

    return assets, warnings


def _ocr_image_bytes(image_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return None, "OCR image skipped: pytesseract/Pillow not installed."

    try:
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image).strip()
        return text if text else None, None
    except Exception as exc:
        return None, f"OCR image failed: {exc}"
