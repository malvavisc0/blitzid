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
from blitzid.reading.structurize import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    Extraction,
    StructuredOCRReader,
    _system_prompt,
)


def _agent_key(kind: str) -> tuple[str, str]:
    """Cache key the reader uses for one kind on today's date."""
    return (kind, date.today().isoformat())


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a complete (unused) LLM configuration."""
    monkeypatch.setenv(LLM_BASE_URL_ENV, "http://127.0.0.1:9/v1")
    monkeypatch.setenv(LLM_API_KEY_ENV, "sk-test")
    monkeypatch.setenv(LLM_MODEL_ENV, "test-model")


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

    def test_llm_schema_hides_reader_managed_fields(self) -> None:
        """raw_text is stamped by the reader, never asked of the model."""
        assert "raw_text" not in Extraction.model_json_schema()["properties"]
        assert "raw_text" in StructuredOCR.model_json_schema()["properties"]


class TestSystemPrompt:
    def test_stamps_todays_date(self) -> None:
        prompt = _system_prompt("mrz", date(2026, 10, 1))
        assert "Today is 2026-10-01." in prompt

    def test_appends_common_rules_to_kind_rules(self) -> None:
        prompt = _system_prompt("plate", date(2026, 10, 1))
        assert prompt.startswith("Extract a vehicle license plate")
        assert "Never invent a value." in prompt


class TestDateCoercion:
    def test_iso_date_coerced(self) -> None:
        field = ExtractedField(name="date_of_birth", value="1983-08-12")
        assert field.value == date(1983, 8, 12)

    def test_dot_date_coerced(self) -> None:
        field = ExtractedField(name="date_of_expiry", value="01.05.2034")
        assert field.value == date(2034, 5, 1)

    def test_suffixed_date_name_coerced(self) -> None:
        field = ExtractedField(name="registration_date", value="01.05.2034")
        assert field.value == date(2034, 5, 1)

    def test_plain_text_stays_text(self) -> None:
        field = ExtractedField(name="document_number", value="L898902C")
        assert field.value == "L898902C"

    def test_date_shaped_value_on_non_date_field_stays_text(self) -> None:
        field = ExtractedField(name="document_number", value="1983-08-12")
        assert field.value == "1983-08-12"


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

    def test_row_jitter_does_not_interleave_columns(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        texts = [
            OCRText(bbox=(0, 12, 10, 10), text="LABEL", confidence=0.9),
            OCRText(bbox=(50, 10, 10, 10), text="VALUE", confidence=0.8),
        ]
        assert reader._format_prompt(texts) == "LABEL\nVALUE"


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
        reader._agents[_agent_key("auto")] = _FakeAgent()  # type: ignore[assignment]
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
        reader._agents[_agent_key("mrz")] = _FakeAgent()  # type: ignore[assignment]
        texts = [OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9)]
        record = reader.read(texts, kind="mrz")
        assert record.document_type == "mrz"

    def test_model_fabricated_raw_text_is_overwritten(self) -> None:
        class _FakeResult:
            output: StructuredOCR = StructuredOCR(raw_text="INVENTED BY MODEL")

        class _FakeAgent:
            def run_sync(self, prompt: str) -> _FakeResult:
                return _FakeResult()

        reader = StructuredOCRReader(log_level=logging.WARNING)
        reader._agents[_agent_key("auto")] = _FakeAgent()  # type: ignore[assignment]
        texts = [OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9)]
        record = reader.read(texts)
        assert record.raw_text == "P<UTO"

    def test_duplicate_names_collapse_to_highest_confidence(self) -> None:
        class _FakeResult:
            output: StructuredOCR = StructuredOCR(
                fields=[
                    ExtractedField(name="surname", value="loewen", confidence=0.5),
                    ExtractedField(name="given_names", value="hans", confidence=0.8),
                    ExtractedField(name="surname", value="mustermann", confidence=0.9),
                ]
            )

        class _FakeAgent:
            def run_sync(self, prompt: str) -> _FakeResult:
                return _FakeResult()

        reader = StructuredOCRReader(log_level=logging.WARNING)
        reader._agents[_agent_key("auto")] = _FakeAgent()  # type: ignore[assignment]
        texts = [OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9)]
        record = reader.read(texts)
        assert [(f.name, f.value, f.confidence) for f in record.fields] == [
            ("surname", "MUSTERMANN", 0.9),
            ("given_names", "HANS", 0.8),
        ]

    def test_invalid_kind_raises(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        with pytest.raises(BlitzIDError, match="kind must be one of"):
            reader.read([], kind="personalausweis")  # type: ignore[arg-type]

    def test_non_positive_timeout_raises(self) -> None:
        with pytest.raises(BlitzIDError, match="timeout"):
            StructuredOCRReader(timeout=0.0)


class TestConfiguration:
    @pytest.mark.parametrize(
        "missing", [LLM_BASE_URL_ENV, LLM_API_KEY_ENV, LLM_MODEL_ENV]
    )
    def test_missing_setting_raises(
        self, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        monkeypatch.delenv(missing)
        with pytest.raises(BlitzIDError, match=missing):
            StructuredOCRReader()

    def test_blank_setting_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(LLM_API_KEY_ENV, "   ")
        with pytest.raises(BlitzIDError, match=LLM_API_KEY_ENV):
            StructuredOCRReader()

    def test_explicit_arguments_bypass_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for env in (LLM_BASE_URL_ENV, LLM_API_KEY_ENV, LLM_MODEL_ENV):
            monkeypatch.delenv(env)
        reader = StructuredOCRReader(
            base_url="http://example/v1",
            api_key="sk-explicit",
            model_name="explicit-model",
            log_level=logging.WARNING,
        )
        assert reader.base_url == "http://example/v1"
        assert reader.api_key == "sk-explicit"
        assert reader.model_name == "explicit-model"

    def test_environment_provides_configuration(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        assert reader.base_url == "http://127.0.0.1:9/v1"
        assert reader.api_key == "sk-test"
        assert reader.model_name == "test-model"


class TestMissingExtra:
    def test_create_agent_raises_without_pydantic_ai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pydantic_ai", None)
        reader = StructuredOCRReader(log_level=logging.WARNING)
        with pytest.raises(BlitzIDError, match=r"blitzid\[ocr\]"):
            reader._create_agent("auto", date.today())


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
