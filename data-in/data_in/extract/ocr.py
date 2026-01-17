from __future__ import annotations

from typing import List


def ocr_pdf_to_pages(path: str, dpi: int = 300) -> List[str]:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OCR dependencies missing. Install pdf2image and pytesseract."
        ) from exc

    images = convert_from_path(path, dpi=dpi)
    texts: List[str] = []
    for image in images:
        text = pytesseract.image_to_string(image)
        texts.append(text)
    return texts
