"""PDF sniffing and page rasterization for analyze uploads.

Small multi-page PDFs (at most ``MAX_PDF_PAGES`` pages) are a supported
analyze input: every page is rendered to a BGR array at ``PDF_DPI`` and
analyzed through the same per-image path as photo uploads. Bounds are
enforced in one place so the upload boundary and the worker render path
share one policy. ``pypdfium2`` is lazy-imported; it ships with the
``api`` extra and its absence raises ``BlitzIDError`` with an install
hint.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np

from blitzid._image import MAX_DIMENSION, MIN_DIMENSION
from blitzid.exceptions import BlitzIDError, ImageError

if TYPE_CHECKING:
    from pypdfium2 import PdfDocument, PdfPage

PDF_MAGIC = b"%PDF-"
MAX_PDF_PAGES = 10
PDF_DPI = 200
MAX_PDF_TOTAL_PIXELS = 60_000_000


def is_pdf_bytes(data: bytes) -> bool:
    """Return whether *data* starts with the PDF magic marker."""
    return data[: len(PDF_MAGIC)] == PDF_MAGIC


def pdf_page_count(data: bytes) -> int:
    """Return the page count of a small, readable PDF.

    Args:
        data: Raw PDF bytes.

    Returns:
        Number of pages.

    Raises:
        ImageError: If the bytes are not a readable, unencrypted PDF, or
            the document is outside the page-count or page-size bounds.
    """
    document = _open_pdf(data)
    try:
        return _validated_pages(document)
    finally:
        document.close()


def render_pdf_pages(data: bytes) -> list[np.ndarray]:
    """Rasterize every page to a 3-channel BGR array.

    Args:
        data: PDF bytes.

    Returns:
        One BGR array per page, in page order.

    Raises:
        ImageError: Same conditions as :func:`pdf_page_count`, or if a
            page fails to render.
    """
    return list(iter_pdf_pages(data))


def iter_pdf_pages(data: bytes) -> Generator[np.ndarray, None, None]:
    """Yield rendered pages one at a time so callers can release memory.

    Args:
        data: PDF bytes.

    Yields:
        3-channel BGR arrays, one per page, in page order.

    Raises:
        ImageError: Same conditions as :func:`pdf_page_count`, or if a
            page fails to render.
    """
    document = _open_pdf(data)
    try:
        count = _validated_pages(document)
        for index in range(count):
            yield _render_page(document, index)
    finally:
        document.close()


def _pdfium() -> Any:
    """Return the lazily imported pypdfium2 module."""
    import pypdfium2 as pdfium

    return pdfium


def _render_page(document: PdfDocument, index: int) -> np.ndarray:
    """Render one page, normalizing PDFium failures to ``ImageError``."""
    pdfium = _pdfium()
    try:
        array: np.ndarray | None = document[index].render(scale=PDF_DPI / 72).to_numpy()
    except pdfium.PdfiumError as e:
        raise ImageError(f"PDF page {index + 1} failed to render: {e}") from e
    if array is None:
        raise ImageError(f"PDF page {index + 1} failed to render")
    return array


def _open_pdf(data: bytes) -> PdfDocument:
    """Open PDF bytes, normalizing dependency and format errors."""
    try:
        pdfium = _pdfium()
    except ImportError as e:
        raise BlitzIDError(
            "PDF support requires pypdfium2; install the blitzid `api` extra"
        ) from e
    try:
        return pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise ImageError(f"unreadable PDF: {e}") from e


def _validated_pages(document: PdfDocument) -> int:
    """Return the page count after enforcing every render bound.

    Each page must render within the image dimension bounds, and the
    pages together within ``MAX_PDF_TOTAL_PIXELS``, so a small file of
    huge pages cannot balloon worker memory.
    """
    pdfium = _pdfium()
    try:
        count = len(document)
    except pdfium.PdfiumError as e:
        raise ImageError(f"unreadable PDF: {e}") from e
    if count < 1:
        raise ImageError("PDF has no pages")
    if count > MAX_PDF_PAGES:
        raise ImageError(f"PDF has {count} pages; the limit is {MAX_PDF_PAGES}")
    total_pixels = 0.0
    for index in range(count):
        total_pixels += _page_pixels(document[index])
    if total_pixels > MAX_PDF_TOTAL_PIXELS:
        raise ImageError(
            f"PDF pages render to {total_pixels / 1e6:.0f} MP in total; "
            f"the limit is {MAX_PDF_TOTAL_PIXELS / 1e6:.0f} MP"
        )
    return count


def _page_pixels(page: PdfPage) -> float:
    """Return a page's rendered pixel count, rejecting out-of-bounds pages."""
    pdfium = _pdfium()
    try:
        width, height = page.get_size()
    except pdfium.PdfiumError as e:
        raise ImageError(f"unreadable PDF page: {e}") from e
    pixels_wide = float(width) * PDF_DPI / 72
    pixels_high = float(height) * PDF_DPI / 72
    if not (
        min(pixels_wide, pixels_high) >= MIN_DIMENSION
        and max(pixels_wide, pixels_high) <= MAX_DIMENSION
    ):
        raise ImageError(
            f"PDF page renders to {pixels_wide:.0f}x{pixels_high:.0f} px at "
            f"{PDF_DPI} DPI; bounds are {MIN_DIMENSION}-{MAX_DIMENSION} px per side"
        )
    return pixels_wide * pixels_high
