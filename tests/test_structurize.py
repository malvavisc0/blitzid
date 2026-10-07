"""Tests for blitzid.reading.structurize — prompt building and reader setup.

The pydantic-ai ``ocr`` extra is not available in core-only installs, so
the unit tests cover prompt formatting, schema behavior, and output
normalization plus the missing-extra guard. Endpoint-backed calls
belong in the smoke test (``scripts/structurize_smoke.py``), not here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Barrier
from types import ModuleType

import pytest
from pydantic import ValidationError

from blitzid import BlitzIDError, ExtractedField, OCRText, StructuredOCR
from blitzid.reading._observability import langfuse_tracing
from blitzid.reading.mrz import _parse
from blitzid.reading.structurize import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    Extraction,
    StructuredOCRReader,
    _system_prompt,
)


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
    @pytest.mark.parametrize("kind", ["auto", "mrz"])
    def test_mrz_boundary_rules_apply_to_both_modes(self, kind: str) -> None:
        prompt = _system_prompt(kind, date(2026, 10, 1))
        assert "skip the first 5 characters of line 1" in prompt
        assert "exactly the first 9 characters of line 2" in prompt
        assert "document_code 'P<'" in prompt

    def test_stamps_todays_date(self) -> None:
        prompt = _system_prompt("mrz", date(2026, 10, 1))
        assert "Today is 2026-10-01." in prompt

    def test_appends_common_rules_to_kind_rules(self) -> None:
        prompt = _system_prompt("plate", date(2026, 10, 1))
        assert prompt.startswith("Extract a vehicle license plate")
        assert "Never invent a value." in prompt


class TestDateCoercion:
    @pytest.mark.parametrize(
        "name",
        [
            "date_of_birth",
            "date_of_expiry",
            "birth_date",
            "expiry_date",
            "date_of_issue",
        ],
    )
    def test_iso_date_coerced(self, name: str) -> None:
        field = ExtractedField(name=name, value="1983-08-12")
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

    @pytest.mark.parametrize("name", ["document_number", "candidate"])
    def test_date_shaped_value_on_non_date_field_stays_text(self, name: str) -> None:
        field = ExtractedField(name=name, value="1983-08-12")
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)

        async def _boom(prompt: str, kind: str, today: date) -> Extraction:
            raise AssertionError("the agent must not be built for empty input")

        monkeypatch.setattr(reader, "_extract", _boom)
        assert reader.read([]) == StructuredOCR()

    def test_empty_input_with_kind_stamps_document_type(self) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        record = reader.read([], kind="plate")
        assert record.document_type == "plate"
        assert record.fields == []

    def test_read_stamps_raw_text_and_normalizes_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def extract(prompt: str, kind: str, today: date) -> Extraction:
            assert prompt == "P<UTO\n<SAMPLE"
            return StructuredOCR(
                fields=[
                    ExtractedField(name="surname", value="mustermann"),
                    ExtractedField(name="date_of_birth", value="1983-08-12"),
                ],
            )

        reader = StructuredOCRReader(log_level=logging.WARNING)
        monkeypatch.setattr(reader, "_extract", extract)
        texts = [
            OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9),
            OCRText(bbox=(20, 0, 10, 10), text="<SAMPLE", confidence=0.9),
        ]
        record = reader.read(texts)
        assert record.raw_text == "P<UTO\n<SAMPLE"
        assert record.document_type is None
        assert record.fields[0].value == "MUSTERMANN"
        assert record.fields[1].value == date(1983, 8, 12)

    def test_explicit_kind_stamps_document_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def extract(prompt: str, kind: str, today: date) -> Extraction:
            return StructuredOCR()

        reader = StructuredOCRReader(log_level=logging.WARNING)
        monkeypatch.setattr(reader, "_extract", extract)
        texts = [OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9)]
        record = reader.read(texts, kind="mrz")
        assert record.document_type == "mrz"

    def test_model_fabricated_raw_text_is_overwritten(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def extract(prompt: str, kind: str, today: date) -> Extraction:
            return StructuredOCR(raw_text="INVENTED BY MODEL")

        reader = StructuredOCRReader(log_level=logging.WARNING)
        monkeypatch.setattr(reader, "_extract", extract)
        texts = [OCRText(bbox=(0, 0, 10, 10), text="P<UTO", confidence=0.9)]
        record = reader.read(texts)
        assert record.raw_text == "P<UTO"

    def test_duplicate_names_collapse_to_highest_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def extract(prompt: str, kind: str, today: date) -> Extraction:
            return StructuredOCR(
                fields=[
                    ExtractedField(name="surname", value="loewen", confidence=0.5),
                    ExtractedField(name="given_names", value="hans", confidence=0.8),
                    ExtractedField(name="surname", value="mustermann", confidence=0.9),
                ]
            )

        reader = StructuredOCRReader(log_level=logging.WARNING)
        monkeypatch.setattr(reader, "_extract", extract)
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

    def test_worker_calls_own_and_close_their_http_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("pydantic_ai")
        import httpx

        barrier = Barrier(2)

        def respond(request: httpx.Request) -> httpx.Response:
            barrier.wait(timeout=5)
            content = json.dumps(
                {
                    "document_type": "id",
                    "fields": [
                        {"name": "surname", "value": "SPECIMEN", "confidence": 1}
                    ],
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "synthetic-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        clients: list[httpx.AsyncClient] = []

        class Client(httpx.AsyncClient):
            def __init__(self) -> None:
                super().__init__(transport=httpx.MockTransport(respond))
                clients.append(self)

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        monkeypatch.setattr(
            "blitzid.reading.structurize.langfuse_tracing", lambda: None
        )
        reader = StructuredOCRReader(log_level=logging.WARNING)
        texts = [OCRText(bbox=(0, 0, 10, 10), text="SPECIMEN", confidence=1.0)]
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(reader.read, texts) for _ in range(2)]
            records = [future.result(timeout=10) for future in futures]
        assert [record.fields[0].value for record in records] == [
            "SPECIMEN",
            "SPECIMEN",
        ]
        assert len(clients) == 2
        assert all(client.is_closed for client in clients)


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
    def test_extraction_raises_without_pydantic_ai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader = StructuredOCRReader(log_level=logging.WARNING)
        monkeypatch.setitem(sys.modules, "pydantic_ai", None)
        with pytest.raises(BlitzIDError, match=r"blitzid\[ocr\]"):
            asyncio.run(reader._extract("SPECIMEN", "auto", date.today()))


class TestSmokeChecks:
    def test_sample_mrz_is_a_valid_complete_td3(
        self, smoke_module: Callable[[str], ModuleType]
    ) -> None:
        record = _parse([line.text for line in smoke_module("structurize")._MRZ_LINES])
        assert (record.mrz_type, record.document_number) == ("TD3", "L898902C3")

    @pytest.mark.parametrize(
        ("kind", "values"),
        [
            ("unknown", {"surname": "MUSTERMANN"}),
            ("id", {}),
            ("id", {"surname": "WRONG", "given_names": "ERIKA"}),
        ],
    )
    def test_smoke_rejects_wrong_classification_empty_or_incorrect_fields(
        self,
        smoke_module: Callable[[str], ModuleType],
        kind: str,
        values: dict[str, str],
    ) -> None:
        record = StructuredOCR(
            document_type=kind,  # type: ignore[arg-type]
            fields=[
                ExtractedField(name=name, value=value) for name, value in values.items()
            ],
        )
        with pytest.raises(AssertionError):
            smoke_module("structurize")._report(record, "id", "synthetic ID")

    def test_smoke_main_checks_auto_classification(
        self,
        smoke_module: Callable[[str], ModuleType],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class Reader:
            def read(
                self, texts: Sequence[OCRText], kind: str = "auto"
            ) -> StructuredOCR:
                return StructuredOCR(
                    document_type="id" if kind == "id" else "unknown",
                    fields=[
                        ExtractedField(name="surname", value="MUSTERMANN"),
                        ExtractedField(name="given_names", value="ERIKA"),
                        ExtractedField(name="document_number", value="LZ6311T47"),
                        ExtractedField(name="date_of_birth", value="1983-08-12"),
                    ],
                )

        smoke = smoke_module("structurize")
        monkeypatch.setattr(smoke, "StructuredOCRReader", lambda **kwargs: Reader())
        monkeypatch.setattr(smoke, "RapidOCRReader", lambda **kwargs: None)
        monkeypatch.setattr(
            smoke,
            "_PASSES",
            (("id", (OCRText(bbox=(0, 0, 10, 10), text="SPECIMEN", confidence=1.0),)),),
        )
        with pytest.raises(AssertionError, match="expected document_type 'id'"):
            smoke.main()


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
