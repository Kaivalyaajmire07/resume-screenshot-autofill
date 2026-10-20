"""PDF input: read the embedded text layer when there is one (instant), OCR the page only if it is a scan.

Uses ``pypdfium2`` (pure wheel, no system install). A text-based PDF needs no OCR at all,
which is the fastest path in the app.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from . import config

try:  # optional: image uploads keep working without it
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover - depends on environment
    pdfium = None

MIN_TEXT_CHARS = 40  # fewer characters than this on a page => treat it as a scan and OCR it


class PdfError(ValueError):
    """Unreadable / unsupported PDF (message is user-facing)."""


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


@dataclass
class PdfPage:
    number: int  # 1-based
    text: str  # embedded text ("" for scans)
    image: Image.Image | None = None  # rendered page, only when OCR is needed


def _open(data: bytes):
    if pdfium is None:
        raise PdfError("PDF support needs the 'pypdfium2' package (pip install pypdfium2).")
    if len(data) > config.MAX_IMAGE_BYTES:
        raise PdfError(f"file is larger than {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    try:
        return pdfium.PdfDocument(data)
    except Exception as exc:  # password-protected, corrupt, ...
        raise PdfError("not a readable PDF (it may be password-protected or damaged)") from exc


def page_count(data: bytes) -> int:
    doc = _open(data)
    try:
        return len(doc)
    finally:
        doc.close()


def thumbnail_png(data: bytes, max_size: int = 220) -> bytes:
    """First page as a small PNG for the attachment gallery."""
    doc = _open(data)
    try:
        if len(doc) == 0:
            raise PdfError("the PDF has no pages")
        img = doc[0].render(scale=0.5).to_pil().convert("RGB")
    finally:
        doc.close()
    img.thumbnail((max_size, max_size))
    out = BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def read_pages(data: bytes, max_pages: int | None = None) -> tuple[list[PdfPage], int]:
    """Return (pages, total_pages). Scanned pages come back with a rendered image, text pages do not."""
    max_pages = max_pages or config.MAX_PDF_PAGES
    doc = _open(data)
    pages: list[PdfPage] = []
    try:
        total = len(doc)
        if total == 0:
            raise PdfError("the PDF has no pages")
        for i in range(min(total, max_pages)):
            page = doc[i]
            text = ""
            try:
                textpage = page.get_textpage()
                text = (textpage.get_text_range() or "").replace("\r\n", "\n").replace("\r", "\n")
            except Exception:
                text = ""
            if len(text.strip()) >= MIN_TEXT_CHARS:
                pages.append(PdfPage(i + 1, text))
            else:  # scanned page: render for OCR (about 150 dpi keeps it quick and readable)
                pages.append(PdfPage(i + 1, "", page.render(scale=config.PDF_RENDER_SCALE).to_pil().convert("RGB")))
        return pages, total
    finally:
        doc.close()
