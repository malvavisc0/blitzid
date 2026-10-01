"""StructuredOCRReader — LLM-backed structured output from OCR text lines.

Feeds the raw OCR lines read by :class:`~blitzid.RapidOCRReader` to an
OpenAI-compatible chat endpoint (e.g. a vLLM server) and reconstructs a
structured, typed document record via ``pydantic-ai``. The model is
forced to return JSON matching the extraction schema and is instructed
to omit fields it cannot recover rather than invent them (prompt
enforced — treat low-confidence fields as unreliable); values under
date-named fields are coerced to ``datetime.date``, text values are
upper-cased by the reader (not the model), duplicate field names
collapse to the highest-confidence entry, and ``raw_text`` is stamped
by the reader from the actual prompt.

Each document kind gets its own specialized system prompt —
:meth:`StructuredOCRReader.read` takes ``kind="id" | "plate" |
"mrz"`` for a focused extraction with the ``document_type`` stamped
by the caller, or ``kind="auto"`` to let the model classify and
extract with a combined prompt.

Configuration comes from constructor arguments or the environment —
all three settings are required (an unset one raises at construction):

- ``BLITZID_LLM_BASE_URL`` — OpenAI-compatible base URL.
- ``BLITZID_LLM_API_KEY`` — API key.
- ``BLITZID_LLM_MODEL`` — served model name.

The endpoint call is bounded by ``timeout`` (default 180 s).

Requires the ``ocr`` extra (``pip install blitzid[ocr]``).
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..exceptions import BlitzIDError, ModelError
from ._observability import langfuse_tracing
from .ocr import OCRText

if TYPE_CHECKING:
    from pydantic_ai import Agent

__all__ = ["ExtractedField", "StructuredOCR", "StructuredOCRReader"]

LLM_BASE_URL_ENV = "BLITZID_LLM_BASE_URL"
LLM_API_KEY_ENV = "BLITZID_LLM_API_KEY"
LLM_MODEL_ENV = "BLITZID_LLM_MODEL"

_DEFAULT_TIMEOUT = 180.0

DocumentType = Literal["id", "plate", "mrz", "unknown"]

ReadKind = Literal["auto", "id", "plate", "mrz"]

_PROMPTS: dict[str, str] = {
    "auto": (
        "Extract structured data from OCR text. Set document_type to exactly "
        "one of: 'id' (identity document with person data), 'plate' "
        "(a vehicle license plate), 'mrz' (ICAO 9303 machine-readable zone: "
        "lines of A-Z, 0-9, and '<'), or 'unknown'. "
        "Then return only fields you can recover. "
        "id fields: surname, given_names, date_of_birth, nationality, "
        "document_number, date_of_expiry, place_of_birth, sex — ignore '<' "
        "filler lines; dates may read DD.MM.YYYY or DD/MM/YYYY (DD.MM.YYYY "
        "is day-first); document_number is the document serial; sex is M, "
        "F, or X. "
        "plate fields: plate_number, issuing_region — plate_number is "
        "letters and digits only, drop spaces and separators ('ABC-1234' "
        "becomes 'ABC1234'); issuing_region is the standard country or "
        "state code when one exists ('NEW YORK' becomes 'NY'), else the "
        "printed name. "
        "mrz fields: document_code, issuer, document_number, nationality, "
        "surname, given_names, birth_date, sex, expiry_date — '<<' splits "
        "surname from given names and single '<' is a space; keep every "
        "letter of the names as printed; strip trailing "
        "'<' fillers from document_number and names; the 6-digit dates are "
        "YYMMDD (birth: most recent century not after today; expiry: always "
        "20YY); sex '<' means X; never emit check digits or optional/"
        "personal-number data. "
    ),
    "id": (
        "Extract person data from identity-document OCR (visual zone). "
        "Ignore machine-readable '<' filler lines. Return only recovered "
        "fields: surname, given_names, date_of_birth, nationality, "
        "document_number, date_of_expiry, place_of_birth, sex. "
        "document_number is the document serial, not a personal ID number. "
        "sex is M, F, or X. Dates may read DD.MM.YYYY or DD/MM/YYYY "
        "(DD.MM.YYYY is day-first) — emit YYYY-MM-DD. "
    ),
    "plate": (
        "Extract a vehicle license plate from OCR text. Return only "
        "recovered fields: plate_number (letters and digits only — drop "
        "spaces and separators, e.g. 'ABC-1234' becomes 'ABC1234'), "
        "issuing_region (the standard country or state code when one "
        "exists — 'NEW YORK' becomes 'NY' — else the printed name). "
    ),
    "mrz": (
        "Extract ICAO 9303 MRZ fields from OCR (TD1 3x30, TD2 2x36, TD3 "
        "2x44). Lines hold A-Z, 0-9, and '<' filler. TD3 line 1: document "
        "code (2 chars, e.g. 'P<'), issuer (3 letters), then the name. "
        "Line 2: document number (9 chars, '<'-padded) + check digit, "
        "nationality (3 letters), birth YYMMDD + check digit, sex, expiry "
        "YYMMDD + check digit, optional data + check digit. TD1 and TD2 "
        "pack the same fields into shorter lines (TD1 puts the document "
        "number on line 1 and personal number on line 2). Names: '<<' "
        "separates surname from given_names, single '<' is a space — "
        "'DOE<<JOHN<PAUL' gives surname 'DOE', given_names 'JOHN PAUL'. "
        "Keep every letter of the names as printed. "
        "Strip trailing '<' fillers from document_number and names. Dates: "
        "birth in the most recent century that does not put it after today "
        "('850102' gives 1985-01-02); expiry always 20YY ('271231' gives "
        "2027-12-31). sex is M, F, or X ('<' in the sex position means X). "
        "Return only recovered fields: document_code, issuer, "
        "document_number, nationality, surname, given_names, birth_date, "
        "sex, expiry_date. Never emit check digits or optional/"
        "personal-number data. "
    ),
}


def _common_rules(today: date) -> str:
    """Shared extraction rules for every document kind.

    ``today`` anchors relative date math (the MRZ birth-century pivot)
    and is stamped into the system prompt as "Today is <date>".
    """
    return (
        f"Today is {today.isoformat()}. "
        "Copy values from the OCR text. Fix only obvious OCR confusions "
        "(0/O, 1/I/l, 5/S, 8/B, 2/Z, 6/G); do not guess missing characters. "
        "Use only the listed snake_case field names, once each. "
        "Dates as YYYY-MM-DD only. "
        "Omit any field you cannot recover. Never invent a value. "
        "Set confidence in [0, 1] for every returned field: "
        "1.0 only if the text is unambiguous, below 0.5 if corrected or partial."
    )


def _system_prompt(kind: str, today: date) -> str:
    """Full system prompt: kind-specific rules plus the common rules."""
    return _PROMPTS[kind] + _common_rules(today)


_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y")

_DATE_FIELD_NAMES = frozenset(
    {"date_of_birth", "date_of_expiry", "birth_date", "expiry_date"}
)

_ROW_GROUPING = 0.5


def _is_date_field(name: str) -> bool:
    """Whether a snake_case field name designates a date.

    Covers the prompt's date fields (``date_of_birth``, ``birth_date``,
    …) and similarly shaped names (``date_*`` / ``*_date``), while
    leaving e.g. ``candidate`` alone.
    """
    return (
        name in _DATE_FIELD_NAMES or name.startswith("date_") or name.endswith("_date")
    )


def _parse_date(value: object) -> object:
    """Parse date-shaped strings (ISO 8601 or DD.MM.YYYY) into dates."""
    if not isinstance(value, str):
        return value
    for fmt in _DATE_FORMATS:
        with contextlib.suppress(ValueError):
            return datetime.strptime(value, fmt).date()
    return value


class ExtractedField(BaseModel):
    """One named field recovered from OCR text.

    Attributes:
        name: Field label in snake_case, e.g. "document_number" or
            "surname".
        value: Extracted value — a ``datetime.date`` for date-named
            fields (e.g. "date_of_birth", "birth_date"), plain text
            otherwise.
        confidence: Model-assigned confidence in [0, 1]; 0.0 when the
            model omitted it, so treat such fields as unreliable.
    """

    name: str = Field(
        description=(
            "Field label in snake_case — one of the names listed in the instructions."
        )
    )
    value: date | str = Field(
        description=(
            "The extracted value. Values of date-named fields (e.g. "
            "date_of_birth, date_of_expiry, birth_date, expiry_date) "
            "as ISO 8601 dates (YYYY-MM-DD); every other value as "
            "plain text."
        )
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence that the value is correct, in [0, 1]. "
            "Omitting it reports 0.0; treat such fields as unreliable."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_date_fields(cls, data: object) -> object:
        """Coerce date-shaped values into dates on date-named fields."""
        if isinstance(data, Mapping):
            name, value = data.get("name"), data.get("value")
            if isinstance(name, str) and _is_date_field(name):
                parsed = _parse_date(value)
                if isinstance(parsed, date):
                    return {**dict(data), "value": parsed}
        return data


class Extraction(BaseModel):
    """Schema handed to the model — only model-populated fields.

    ``raw_text`` is stamped by the reader afterwards and deliberately
    stays out of this schema, so the model is never asked to invent the
    audit trail.
    """

    document_type: DocumentType | None = Field(
        default=None,
        description=(
            "Document kind: 'id' for an identity document with person "
            "data, 'plate' for a license plate, 'mrz' for a machine "
            "readable zone, 'unknown' otherwise."
        ),
    )
    fields: list[ExtractedField] = Field(
        default_factory=list,
        description="Fields extracted from the OCR text, snake_case names.",
    )


class StructuredOCR(Extraction):
    """Structured document data derived from raw OCR lines.

    Attributes:
        document_type: Document kind — "id" (identity document with
            person data), "plate" (license plate), "mrz" (machine
            readable zone), or "unknown"; None when it cannot be
            determined.
        fields: Named fields extracted from the OCR text. One entry per
            name: duplicates collapse to the highest-confidence value
            (first wins on ties), first-seen order preserved.
        raw_text: The joined OCR lines fed to the model, stamped by the
            reader for auditability — never model-populated.
    """

    raw_text: str = Field(
        default="",
        description=(
            "The joined OCR lines fed to the model. Stamped by the "
            "reader from the actual prompt; never model-populated."
        ),
    )


class StructuredOCRReader:
    """Reconstructs structured document data from OCR lines via an LLM.

    Args:
        base_url: OpenAI-compatible base URL. Falls back to
            ``BLITZID_LLM_BASE_URL``; required when neither is set.
        api_key: API key. Falls back to ``BLITZID_LLM_API_KEY``;
            required when neither is set.
        model_name: Served model name. Falls back to
            ``BLITZID_LLM_MODEL``; required when neither is set.
        timeout: Endpoint call timeout in seconds.
        log_level: Logging level for the reader's logger.

    Raises:
        BlitzIDError: If a required setting is neither passed nor set
            in the environment (``base_url`` / ``BLITZID_LLM_BASE_URL``,
            ``api_key`` / ``BLITZID_LLM_API_KEY``, ``model_name`` /
            ``BLITZID_LLM_MODEL``), if ``timeout`` is not positive, or
            if the ``ocr`` extra (pydantic-ai) is not installed.
        ModelError: If the endpoint cannot be reached or returns
            unstructured output.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if timeout <= 0:
            raise BlitzIDError(f"timeout must be positive, got {timeout}")
        self.timeout = timeout

        self.base_url = self._require_config(base_url, LLM_BASE_URL_ENV, "base_url")
        self.api_key = self._require_config(api_key, LLM_API_KEY_ENV, "api_key")
        self.model_name = self._require_config(model_name, LLM_MODEL_ENV, "model_name")

        self._agents: dict[tuple[str, str], Agent[Any, Extraction]] = {}

    @staticmethod
    def _require_config(value: str | None, env: str, arg_name: str) -> str:
        """Resolve one setting from the constructor argument, else the environment.

        Raises:
            BlitzIDError: When neither provides a non-empty value.
        """
        resolved = (value or os.environ.get(env) or "").strip()
        if not resolved:
            raise BlitzIDError(
                f"missing LLM configuration: pass {arg_name} or set {env}"
            )
        return resolved

    def read(self, texts: Sequence[OCRText], kind: ReadKind = "auto") -> StructuredOCR:
        """Build a structured record from OCR text lines.

        Args:
            texts: OCR lines, e.g. the output of
                :meth:`RapidOCRReader.read`. Empty input yields an
                empty record without calling the endpoint.
            kind: Which document to extract — a specialized per-kind
                prompt with a typed ``document_type`` (``"id"``,
                ``"plate"``, ``"mrz"``), or ``"auto"`` to let the model
                classify and extract with the combined prompt.

        Returns:
            A :class:`StructuredOCR` record validated against the schema.
            With an explicit ``kind`` the ``document_type`` is stamped
            from it, not from the model.

        Raises:
            BlitzIDError: If ``kind`` is invalid or the ``ocr`` extra
                (pydantic-ai) is not installed.
            ModelError: If the endpoint call fails or returns no usable
                structured data.
        """
        if kind not in _PROMPTS:
            raise BlitzIDError(f"kind must be one of {sorted(_PROMPTS)}, got {kind!r}")
        if not texts:
            return StructuredOCR(document_type=None if kind == "auto" else kind)
        prompt = self._format_prompt(texts)
        today = date.today()
        cache_key = (kind, today.isoformat())
        agent = self._agents.get(cache_key) or self._create_agent(kind, today)
        self._agents[cache_key] = agent
        try:
            result = agent.run_sync(prompt)
        except BlitzIDError:
            raise
        except Exception as e:
            raise ModelError(
                f"Structured extraction failed against {self.base_url}: {e}"
            ) from e
        output = result.output
        if not isinstance(output, Extraction):
            raise ModelError(f"Expected StructuredOCR, got {type(output).__name__}")
        record = self._normalize(output, prompt, kind)
        self.logger.info(
            "Structured record: %s (%d fields)",
            record.document_type,
            len(record.fields),
        )
        return record

    def _normalize(
        self, output: Extraction, prompt: str, kind: ReadKind
    ) -> StructuredOCR:
        """Build the public record from raw model output.

        Stamps ``raw_text`` from the prompt, upper-cases text values
        (dates untouched), and collapses duplicate field names to the
        highest-confidence entry (first-seen order preserved). With an
        explicit ``kind`` the ``document_type`` is stamped from it
        rather than taken from the model.
        """
        unique: dict[str, ExtractedField] = {}
        for field in output.fields:
            if not isinstance(field.value, date):
                field = field.model_copy(update={"value": field.value.upper()})
            prior = unique.get(field.name)
            if prior is None or field.confidence > prior.confidence:
                unique[field.name] = field
        return StructuredOCR(
            document_type=kind if kind != "auto" else output.document_type,
            fields=list(unique.values()),
            raw_text=prompt,
        )

    def _format_prompt(self, texts: Sequence[OCRText]) -> str:
        """Join OCR lines in reading order into the prompt.

        Lines are grouped into rows — tolerant of OCR y-jitter within
        one row — and ordered top-to-bottom, left-to-right within a row.
        """
        ordered = sorted(
            texts, key=lambda text: (text.bbox[1] + text.bbox[3] / 2, text.bbox[0])
        )
        rows: list[list[OCRText]] = []
        row_center = 0.0
        for text in ordered:
            _, y, _, height = text.bbox
            center = y + height / 2
            if not rows or center - row_center > _ROW_GROUPING * height:
                rows.append([text])
                row_center = center
            else:
                rows[-1].append(text)
        return "\n".join(
            line.text
            for row in rows
            for line in sorted(row, key=lambda text: text.bbox[0])
        )

    def _create_agent(self, kind: ReadKind, today: date) -> Agent[Any, Extraction]:
        """Build the pydantic-ai agent for one document kind.

        Lazy import of the ``ocr`` extra.

        Raises:
            BlitzIDError: If the ``ocr`` extra cannot be imported.
        """
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.output import NativeOutput
            from pydantic_ai.providers.openai import OpenAIProvider
            from pydantic_ai.settings import ModelSettings
        except ImportError as e:
            raise BlitzIDError(
                f"Cannot import pydantic-ai (ocr extra): {e}. "
                "Install it: pip install blitzid[ocr]"
            ) from e

        self.logger.info(
            "Initializing StructuredOCRReader (kind: %s, model: %s, url: %s)",
            kind,
            self.model_name,
            self.base_url,
        )
        tracing = langfuse_tracing()
        if tracing is not None:
            self.logger.info("Langfuse tracing enabled for this agent")
        model = OpenAIChatModel(
            self.model_name,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key),
        )
        agent = Agent(
            model,
            output_type=NativeOutput(
                Extraction,
                name="structured_ocr",
                description="Structured fields extracted from document OCR text.",
            ),
            system_prompt=_system_prompt(kind, today),
            model_settings=ModelSettings(timeout=self.timeout),
        )
        agent.instrument = tracing is not None
        return agent
