"""Tests for blitzid.reading.structurize — prompt building and reader setup.

The pydantic-ai ``ocr`` extra is not available in core-only installs, so
the unit tests cover prompt formatting, schema behavior, and output
normalization plus the missing-extra guard. Endpoint-backed calls
belong in the smoke test (``scripts/structurize_smoke.py``), not here.
"""

from __future__ import annotations

import logging
import sys
from datetime import date

import pytest
from pydantic import ValidationError

from blitzid import BlitzIDError, ExtractedField, OCRText, StructuredOCR
from blitzid.reading._observability import langfuse_tracing
from blitzid.reading.structurize import StructuredOCRReader


class TestStructuredModels:
    def test_empty_record(self) -> None:
        record = StructuredOCR()
        assert record.document_type is None
        assert record.fields == []
        assert record.raw_text == ""

    def test_field_default_confidence(self) -> None:
        field = ExtractedField(name="surname", value="DOE")
        assert field.confidence == 0.0

    def test_field_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            ExtractedField(name="n", value="v", confidence=1.5)

    def test_invalid_document_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StructuredOCR(document_type="personalausweis")


class TestDateCoercion:
    def test_iso_date_coerced(self) -> None:
        field = ExtractedField(name="date_of_birth", value="1983-08-12")
        assert field.value == date(1983, 8, 12)

    def test_dot_date_coerced(self) -> None:
        field = ExtractedField(name="date_of_expiry", value="01.05.2034")
        assert field.value == date(2034, 5, 1)

    def test_plain_text_stays_text(self) -> None:
        field = ExtractedField(name="document_number", value="L898902C")
        assert field.value == "L898902C"


class TestFormatPrompt:
    def test_orders_lines_by_bbox_y(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        texts = [
            OCRText(bbox=(0, 200, 10, 10), text="BOTTOM", confidence=0.9),
            OCRText(bbox=(0, 10, 10, 10), text="TOP", confidence=0.8),
        ]
        assert reader._format_prompt(texts) == "TOP\nBOTTOM"

    def test_same_row_orders_left_to_right(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        texts = [
            OCRText(bbox=(50, 10, 10, 10), text="RIGHT", confidence=0.9),
            OCRText(bbox=(0, 10, 10, 10), text="LEFT", confidence=0.8),
        ]
        assert reader._format_prompt(texts) == "LEFT\nRIGHT"


class TestRead:
    def test_empty_input_returns_empty_record_without_endpoint(
        self,
    ) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        assert reader.read([]) == StructuredOCR()
        assert reader._agents == {}

    def test_empty_input_with_kind_stamps_document_type(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        record = reader.read([], kind="plate")
        assert record.document_type == "plate"
        assert record.fields == []

    def test_read_stamps_raw_text_and_normalizes_values(self) -> None:
        class _FakeResult:
            output: StructuredOCR = StructuredOCR(
                fields=[
                    ExtractedField(name="surname", value="mustermann"),
                    ExtractedField(name="date_of_birth", value="1983-08-12"),
                ],
            )

        class _FakeAgent:
            def run_sync(self, prompt: str) -> _FakeResult:
                assert prompt == "P<UTO\n<SAMPLE"
                return _FakeResult()

        reader = StructuredOCRReader(log_level=logging.WARNING)
        reader._agents["auto"] = _FakeAgent()  # type: ignore[assignment]
        texts = [
            OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9),
            OCRText(bbox=(20, 0, 10, 10), text="<SAMPLE", confidence=0.9),
        ]
        record = reader.read(texts)
        assert record.raw_text == "P<UTO\n<SAMPLE"
        assert record.document_type is None
        assert record.fields[0].value == "MUSTERMANN"
        assert record.fields[1].value == date(1983, 8, 12)

    def test_explicit_kind_stamps_document_type(self) -> None:
        class _FakeResult:
            output: StructuredOCR = StructuredOCR()

        class _FakeAgent:
            def run_sync(self, prompt: str) -> _FakeResult:
                return _FakeResult()

        reader = StructuredOCRReader(log_level=logging.WARNING)
        reader._agents["mrz"] = _FakeAgent()  # type: ignore[assignment]
        texts = [OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9)]
        record = reader.read(texts, kind="mrz")
        assert record.document_type == "mrz"


class TestMissingExtra:
    def test_create_agent_raises_without_pydantic_ai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pydantic_ai", None)
        reader = StructuredOCRReader(log_level=logging.WARNING)
        with pytest.raises(BlitzIDError, match=r"blitzid\[ocr\]"):
            reader._create_agent("auto")


class TestTracing:
    def test_tracing_disabled_without_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        langfuse_tracing.cache_clear()
        try:
            assert langfuse_tracing() is None
        finally:
            langfuse_tracing.cache_clear()
