"""Tests for PDF sniffing, page bounds, and rasterization (api/_pdf.py)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from blitzid.api._pdf import (
    MAX_PDF_PAGES,
    PDF_DPI,
    is_pdf_bytes,
    pdf_page_count,
    render_pdf_pages,
)
from blitzid.exceptions import ImageError


def test_is_pdf_bytes_sniffs_magic() -> None:
    assert is_pdf_bytes(b"%PDF-1.7 body")
    assert not is_pdf_bytes(b"\x89PNG\r\n\x1a\n body")
    assert not is_pdf_bytes(b"")


def test_page_count_reads_created_pdf(pdf_factory: Callable[..., bytes]) -> None:
    assert pdf_page_count(pdf_factory(3)) == 3


def test_page_count_rejects_corrupt_bytes() -> None:
    with pytest.raises(ImageError, match="unreadable PDF"):
        pdf_page_count(b"%PDF-1.4 garbage")


def test_page_count_rejects_non_pdf_bytes() -> None:
    with pytest.raises(ImageError, match="unreadable PDF"):
        pdf_page_count(b"not a pdf at all")


def test_page_count_rejects_too_many_pages(
    pdf_factory: Callable[..., bytes],
) -> None:
    with pytest.raises(ImageError, match=f"limit is {MAX_PDF_PAGES}"):
        pdf_page_count(pdf_factory(MAX_PDF_PAGES + 1))


def test_page_count_rejects_oversized_page(
    pdf_factory: Callable[..., bytes],
) -> None:
    with pytest.raises(ImageError, match="renders to"):
        pdf_page_count(pdf_factory(1, width=20000.0, height=100.0))


def test_page_count_rejects_tiny_page(pdf_factory: Callable[..., bytes]) -> None:
    with pytest.raises(ImageError, match="renders to"):
        pdf_page_count(pdf_factory(1, width=1.0, height=1.0))


def test_render_produces_bgr_pages(pdf_factory: Callable[..., bytes]) -> None:
    pages = render_pdf_pages(pdf_factory(2, width=200.0, height=100.0))
    assert len(pages) == 2
    for page in pages:
        assert isinstance(page, np.ndarray)
        assert page.dtype == np.uint8
        assert page.shape == (
            round(100.0 * PDF_DPI / 72),
            round(200.0 * PDF_DPI / 72),
            3,
        )


def test_render_rejects_oversized_page(pdf_factory: Callable[..., bytes]) -> None:
    with pytest.raises(ImageError, match="renders to"):
        render_pdf_pages(pdf_factory(2, width=20000.0, height=100.0))


def test_render_rejects_too_many_pages(pdf_factory: Callable[..., bytes]) -> None:
    with pytest.raises(ImageError, match=f"limit is {MAX_PDF_PAGES}"):
        render_pdf_pages(pdf_factory(MAX_PDF_PAGES + 1))


def test_render_failure_is_an_image_error(
    pdf_factory: Callable[..., bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    import pypdfium2 as pdfium

    def fail_render(*args: object, **kwargs: object) -> None:
        raise pdfium.PdfiumError("render failed")

    monkeypatch.setattr(pdfium.PdfPage, "render", fail_render)

    with pytest.raises(ImageError, match="page 1 failed to render"):
        render_pdf_pages(pdf_factory(1))


def test_page_count_rejects_render_budget(
    pdf_factory: Callable[..., bytes],
) -> None:
    # Two pages under the per-page cap but far past the total render budget.
    with pytest.raises(ImageError, match="MP in total"):
        pdf_page_count(pdf_factory(2, width=2000.0, height=2000.0))


def test_render_rejects_render_budget(pdf_factory: Callable[..., bytes]) -> None:
    with pytest.raises(ImageError, match="MP in total"):
        render_pdf_pages(pdf_factory(2, width=2000.0, height=2000.0))


def test_render_allows_ten_letter_pages(pdf_factory: Callable[..., bytes]) -> None:
    pages = render_pdf_pages(pdf_factory(10, width=612.0, height=792.0))
    assert len(pages) == 10
