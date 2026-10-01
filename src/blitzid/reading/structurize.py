"""StructuredOCRReader — LLM-backed structured output from OCR text lines.

Feeds the raw OCR lines read by :class:`~blitzid.RapidOCRReader` to an
OpenAI-compatible chat endpoint (e.g. a vLLM server) and reconstructs a
structured, typed document record via ``pydantic-ai``. The model is
forced to return JSON matching the :class:`StructuredOCR` schema; any
field it cannot recover is omitted rather than hallucinated, date
fields are coerced to ``datetime.date``, text values are upper-cased
by the reader (not the model), and ``raw_text`` is stamped by the
reader from the actual prompt.

Each document kind gets its own specialized system prompt —
:meth:`StructuredOCRReader.read` takes ``kind="id" | "plate" |
"mrz"`` for a focused extraction with the ``document_type`` stamped
by the caller, or ``kind="auto"`` to let the model classify and
extract with a combined prompt.

Configuration is read from the environment, with defaults that point at
a local vLLM server:

- ``BLITZID_LLM_BASE_URL`` — OpenAI-compatible base URL
  (default ``http://127.0.0.1:9090/v1``).
- ``BLITZID_LLM_API_KEY`` — API key (default ``sk-aria``).
- ``BLITZID_LLM_MODEL`` — served model name (default ``Qwen3.8-9B``).

Requires the ``ocr`` extra (``pip install blitzid[ocr]``).
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..exceptions import BlitzIDError, ModelError
from ._observability import langfuse_tracing
from .ocr import OCRText

if TYPE_CHECKING:
    from pydantic_ai import Agent

__all__ = ["ExtractedField", "StructuredOCR", "StructuredOCRReader"]

LLM_BASE_URL_ENV = "BLITZID_LLM_BASE_URL"
LLM_API_KEY_ENV = "BLITZID_LLM_API_KEY"
LLM_MODEL_ENV = "BLITZID_LLM_MODEL"

_DEFAULT_BASE_URL = "http://127.0.0.1:9090/v1"
_DEFAULT_API_KEY = "sk-aria"
_DEFAULT_MODEL = "Qwen3.8-9B"

DocumentType = Literal["id", "plate", "mrz", "unknown"]

ReadKind = Literal["auto", "id", "plate", "mrz"]

_COMMON_RULES = (
    "Return dates (e.g. date_of_birth, date_of_expiry, birth_date, "
    "expiry_date) as YYYY-MM-DD. Omit fields you cannot recover. "
    "Report a confidence in [0, 1] for every field. Never invent data."
)

_PROMPTS: dict[str, str] = {
    "auto": (
        "You extract structured data from OCR text. First classify the "
        "document as exactly one of: 'id' (an identity document with person "
        "data), 'plate' (a license plate), 'mrz' (a machine readable zone), "
        "or 'unknown'. Then return every field you are confident about, "
        "with snake_case names: for 'id' person fields (e.g. surname, "
        "given_names, date_of_birth, nationality, document_number, "
        "date_of_expiry); for 'plate' plate_number and the issuing region; "
        "for 'mrz' the ICAO 9303 fields (e.g. document_number, issuer, "
        "nationality, surname, given_names, birth_date, sex, expiry_date). "
        + _COMMON_RULES
    ),
    "id": (
        "You extract person data from the OCR text of an identity "
        "document. Return every field you are confident about, with "
        "snake_case names: surname, given_names, date_of_birth, "
        "nationality, document_number, date_of_expiry, place_of_birth. " + _COMMON_RULES
    ),
    "plate": (
        "You extract license plate data from OCR text. Return every "
        "field you are confident about, with snake_case names: "
        "plate_number, issuing_region. " + _COMMON_RULES
    ),
    "mrz": (
        "You extract fields from ICAO 9303 machine-readable zone lines "
        "(TD1, TD2, TD3) in OCR text. Return every field you are "
        "confident about, with snake_case names: document_code, issuer, "
        "document_number, surname, given_names, birth_date, sex, "
        "expiry_date. " + _COMMON_RULES
    ),
}

_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y")


def _coerce_date(value: object) -> object:
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
        value: Extracted value — a ``datetime.date`` for date fields,
            plain text otherwise.
        confidence: Model-assigned confidence in [0, 1].
    """

    name: str = Field(description="Field label in snake_case.")
    value: date | str = Field(
        description=(
            "The extracted value. Date fields (e.g. date_of_birth, "
            "date_of_expiry, birth_date, expiry_date) as ISO 8601 dates "
            "(YYYY-MM-DD); every other value as plain text."
        )
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence that the value is correct, in [0, 1]. "
            "Always report it; it defaults to 0.0 when omitted."
        ),
    )

    @field_validator("value", mode="before")
    @classmethod
    def _parse_date(cls, value: object) -> object:
        """Coerce date-shaped strings into dates."""
        return _coerce_date(value)


class StructuredOCR(BaseModel):
    """Structured document data derived from raw OCR lines.

    Attributes:
        document_type: Document kind — "id" (identity document with
            person data), "plate" (license plate), "mrz" (machine
            readable zone), or "unknown"; None when it cannot be
            determined.
        fields: Named fields extracted from the OCR text.
        raw_text: The joined OCR lines fed to the model, for auditability.
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
    raw_text: str = ""


class StructuredOCRReader:
    """Reconstructs structured document data from OCR lines via an LLM.

    Args:
        base_url: OpenAI-compatible base URL. Defaults to
            ``BLITZID_LLM_BASE_URL``, then a local vLLM server.
        api_key: API key. Defaults to ``BLITZID_LLM_API_KEY``.
        model_name: Served model name. Defaults to ``BLITZID_LLM_MODEL``.
        log_level: Logging level for the reader's logger.

    Raises:
        BlitzIDError: If the ``ocr`` extra (pydantic-ai) is not installed.
        ModelError: If the endpoint cannot be reached or returns
            unstructured output.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.base_url = base_url or os.environ.get(LLM_BASE_URL_ENV, _DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get(LLM_API_KEY_ENV, _DEFAULT_API_KEY)
        self.model_name = model_name or os.environ.get(LLM_MODEL_ENV, _DEFAULT_MODEL)

        self._agents: dict[str, Agent[Any, StructuredOCR]] = {}

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
            BlitzIDError: If the ``ocr`` extra (pydantic-ai) is not installed.
            ModelError: If the endpoint call fails or returns no usable
                structured data.
        """
        if not texts:
            return StructuredOCR(document_type=None if kind == "auto" else kind)
        prompt = self._format_prompt(texts)
        agent = self._agents.get(kind) or self._create_agent(kind)
        self._agents[kind] = agent
        try:
            result = agent.run_sync(prompt)
        except Exception as e:
            raise ModelError(
                f"Structured extraction failed against {self.base_url}: {e}"
            ) from e
        output = result.output
        if not isinstance(output, StructuredOCR):
            raise ModelError(f"Expected StructuredOCR, got {type(output).__name__}")
        record = self._normalize(output, prompt, kind)
        self.logger.info(
            "Structured record: %s (%d fields)",
            record.document_type,
            len(record.fields),
        )
        return record

    def _normalize(
        self, output: StructuredOCR, prompt: str, kind: ReadKind
    ) -> StructuredOCR:
        """Stamp ``raw_text``, upper-case text values; dates untouched.

        With an explicit ``kind`` the ``document_type`` is stamped from
        it rather than taken from the model.
        """
        fields = [
            field
            if isinstance(field.value, date)
            else field.model_copy(update={"value": field.value.upper()})
            for field in output.fields
        ]
        update: dict[str, Any] = {"raw_text": prompt, "fields": fields}
        if kind != "auto":
            update["document_type"] = kind
        return output.model_copy(update=update)

    def _format_prompt(self, texts: Sequence[OCRText]) -> str:
        """Join OCR lines (top-to-bottom, left-to-right) into the prompt."""
        ordered = sorted(texts, key=lambda text: (text.bbox[1], text.bbox[0]))
        return "\n".join(text.text for text in ordered)

    def _create_agent(self, kind: ReadKind) -> Agent[Any, StructuredOCR]:
        """Build the pydantic-ai agent for one document kind.

        Lazy import of the ``ocr`` extra.

        Raises:
            BlitzIDError: If pydantic-ai is not installed.
        """
        try:
            from pydantic_ai import Agent
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.output import NativeOutput
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError as e:
            raise BlitzIDError(
                "pydantic-ai is not installed. "
                "Install the OCR extra: pip install blitzid[ocr]"
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
                StructuredOCR,
                name="structured_ocr",
                description="Structured fields extracted from document OCR text.",
            ),
            system_prompt=_PROMPTS[kind],
        )
        agent.instrument = tracing is not None
        return agent
