"""Pydantic request/response models for the blitzid API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentType = Literal["id", "plate", "mrz", "unknown"]
Verdict = Literal["match", "partial", "mismatch", "unavailable"]


class JobCreated(BaseModel):
    """202 response for POST /analyze."""

    job_id: str = Field(
        description="Opaque job identifier.",
        examples=["0f8b9d2c4e6a4b1f9c3d5e7a8b0c1d2e"],
    )
    status: Literal["queued"] = Field(
        description="Always `queued`; poll `status_url` for the result."
    )
    status_url: str = Field(
        description="Path to poll for status and result.",
        examples=["/api/jobs/0f8b9d2c4e6a4b1f9c3d5e7a8b0c1d2e"],
    )


class CropResponse(BaseModel):
    """200 response for POST /crop — QC verdict plus canonical crop."""

    quad: list[list[int]] | None = Field(
        description="Detected document corners as `[x, y]` pairs in source pixels."
    )
    crop_base64: str | None = Field(
        description="Perspective-corrected JPEG crop, base64-encoded."
    )
    width: int = Field(description="Crop width in pixels.")
    height: int = Field(description="Crop height in pixels.")
    side: str = Field(description="Requested side: `front`, `back`, or `unknown`.")
    face_found: bool | None = Field(
        description="Whether a face was detected on the document."
    )
    checks: dict[str, str] = Field(
        description="Per-check verdicts keyed by check name."
    )
    verdict: str = Field(description="Overall capture QC verdict.")


class FaceRef(BaseModel):
    """A detected face's location and score (e.g. an unpicked candidate)."""

    bbox: list[int] = Field(description="`[x, y, w, h]` face box in source pixels.")
    confidence: float = Field(description="Detector confidence for this face.")


class ComparedFace(BaseModel):
    """One side of a /verify comparison — the evidence that was used."""

    bbox: list[int] = Field(description="`[x, y, w, h]` face box in source pixels.")
    confidence: float = Field(description="Detector confidence for this face.")
    crop_base64: str = Field(description="JPEG face crop, base64-encoded.")
    alternatives: list[FaceRef] = Field(
        description="Other faces found in the image that were not compared."
    )


class ChallengeResponse(BaseModel):
    """200 response for POST /liveness/challenge."""

    challenge_id: str = Field(description="One-use challenge identifier.")
    action: str = Field(description="Randomized action the user must perform.")
    expires_in: float = Field(description="Seconds until the challenge expires.")


class LivenessResponse(BaseModel):
    """200 response for POST /liveness/session."""

    live: bool = Field(description="Whether the action passed across all frames.")
    action: str = Field(description="Action that was checked.")
    nose_shift: float = Field(description="Measured nose displacement ratio.")
    mouth_ratio_change: float = Field(description="Measured mouth ratio change.")
    size_ratio: float = Field(description="Face size change ratio across frames.")
    processing_time_ms: float = Field(description="Wall-clock processing time.")


class VerifyResponse(BaseModel):
    """200 response for POST /verify — 1:1 face comparison with evidence."""

    verified: bool = Field(
        description="`similarity >= threshold` (or the backend default)."
    )
    similarity: float = Field(description="Face similarity score.")
    threshold: float = Field(description="Threshold applied to `verified`.")
    face1: ComparedFace = Field(description="The face used from `image1`.")
    face2: ComparedFace = Field(description="The face used from `image2`.")
    processing_time_ms: float = Field(description="Wall-clock processing time.")
    backend: str = Field(description="Verifier backend identifier.")


class HealthResponse(BaseModel):
    """Health response with package/build identity and engine/Redis/job state."""

    status: str = Field(description="`ok`, or `degraded` when Redis is down.")
    version: str = Field(description="Running package version.")
    revision: str | None = Field(description="Build revision, when set.")
    models: dict[str, bool] = Field(description="Per-analysis engine availability.")
    redis: bool = Field(description="Whether the job store is reachable.")
    jobs: dict[str, int] = Field(description="Job counts by state.")
    auth: bool | None = Field(
        default=None,
        description=(
            "Whether hosted key auth is ready; null when auth is disabled "
            "(self-hosted)."
        ),
    )


class AnalysisSection(BaseModel):
    """Fields shared by every analysis section of a job result."""

    error: str | None = Field(
        default=None,
        description=(
            "Set when this analysis failed; the job still completes and "
            "other sections still return."
        ),
    )
    processing_time_ms: float | None = Field(
        default=None, description="Wall-clock time spent on this section."
    )


class FaceItem(BaseModel):
    """One detected face."""

    bbox: list[int] = Field(description="`[x, y, w, h]` box in source pixels.")
    confidence: float = Field(description="Detector confidence.")
    landmarks: list[list[int]] = Field(
        description="Five landmark points as `[x, y]` pairs."
    )
    crop_base64: str = Field(description="JPEG face crop, base64-encoded.")


class FaceSection(AnalysisSection):
    """Face detection results (`face`)."""

    faces: list[FaceItem] = Field(description="Detected faces, highest score first.")


class FaceAttributeItem(BaseModel):
    """Predicted attributes for one face."""

    bbox: list[int] = Field(description="`[x, y, w, h]` box in source pixels.")
    confidence: float = Field(description="Detector confidence.")
    age: str = Field(description="Age band, e.g. `20-29`.", examples=["20-29"])
    gender: str = Field(description="`Male` or `Female`.", examples=["Female"])
    race: str = Field(description="Coarse race group label.")
    age_confidence: float = Field(description="Softmax score of the age band.")
    gender_confidence: float = Field(description="Softmax score of the gender.")
    race_confidence: float = Field(description="Softmax score of the race group.")


class AttributesSection(AnalysisSection):
    """Age, gender, and race predictions (`attributes`)."""

    faces: list[FaceAttributeItem] = Field(description="One entry per face.")


class AntiSpoofItem(BaseModel):
    """Spoof scores for one face."""

    bbox: list[int] = Field(description="`[x, y, w, h]` box in source pixels.")
    confidence: float = Field(description="Detector confidence.")
    live_score: float = Field(description="Score of the real/live class, in [0, 1].")
    paper_score: float = Field(description="Score of the printed-photo class.")
    screen_score: float = Field(description="Score of the screen-replay class.")


class AntiSpoofSection(AnalysisSection):
    """Presentation-attack scores (`antispoof`)."""

    faces: list[AntiSpoofItem] = Field(description="One entry per face.")


class OCRLine(BaseModel):
    """One text line detected by OCR."""

    bbox: list[int] = Field(description="`[x, y, w, h]` box in source pixels.")
    text: str = Field(description="Recognized text.")
    confidence: float = Field(description="Recognition confidence, in [0, 1].")


class OCRSection(AnalysisSection):
    """Recognized text lines (`ocr`)."""

    lines: list[OCRLine] = Field(
        description="Text lines, highest recognition confidence first."
    )


class MRZRecord(BaseModel):
    """Parsed ICAO 9303 machine-readable zone."""

    mrz_type: str = Field(description="`TD1`, `TD2`, or `TD3`.")
    document_code: str = Field(description="Two-character document code, e.g. `P<`.")
    issuer: str = Field(description="Three-character issuing state code.")
    document_number: str = Field(description="Document number, fillers stripped.")
    birth_date: str = Field(description="Birth date as `YYMMDD`.")
    sex: str = Field(description="`M`, `F`, or `X`.")
    expiry_date: str = Field(description="Expiry date as `YYMMDD`.")
    nationality: str = Field(description="Three-character nationality code.")
    surname: str = Field(description="Primary identifier, fillers as spaces.")
    given_names: str = Field(description="Secondary identifier, fillers as spaces.")
    optional_data1: str = Field(description="Raw first optional field.")
    optional_data2: str = Field(description="Raw second optional field; usually empty.")


class MRZSection(AnalysisSection):
    """Code-strip parsing (`mrz`)."""

    record: MRZRecord = Field(description="Parsed machine-readable zone.")


class BarcodeRecord(BaseModel):
    """Parsed AAMVA PDF417 driver's-license record."""

    jurisdiction: str = Field(
        description="Issuing jurisdiction's identification number."
    )
    aamva_version: str = Field(description="AAMVA version, `02` through `14`.")
    jurisdiction_version: str = Field(description="The jurisdiction's version number.")
    family_name: str = Field(description="Family name (DCS).")
    given_name: str = Field(description="Given name (DAC).")
    middle_name: str = Field(description="Middle name(s); empty when absent.")
    birth_date: str = Field(description="Date of birth (DBB) as an ISO 8601 date.")
    issue_date: str = Field(description="Issue date (DBD) as an ISO 8601 date.")
    expiry_date: str = Field(description="Expiry date (DBA) as an ISO 8601 date.")
    sex: str = Field(description="`M`, `F`, or `X`.")
    license_number: str = Field(description="Customer ID number (DAQ).")
    height: str | None = Field(description="Height with unit, e.g. `069 in`.")
    extra_tags: dict[str, str] = Field(
        description="Raw values of every un-mapped AAMVA element."
    )
    symbology: str | None = Field(description="Decoded symbology, e.g. `PDF417`.")


class BarcodeSection(AnalysisSection):
    """AAMVA barcode parsing (`barcode`)."""

    record: BarcodeRecord = Field(description="Parsed barcode record.")


class StructuredField(BaseModel):
    """One named field recovered from printed text."""

    name: str = Field(description="Field label in snake_case.", examples=["surname"])
    value: str = Field(
        description=(
            "Extracted value; date-named fields are ISO 8601 dates "
            "(`YYYY-MM-DD`), everything else is plain text."
        )
    )
    confidence: float = Field(
        description="Model confidence, in [0, 1]; `0.0` means unrated."
    )


class StructuredRecord(BaseModel):
    """Structured document data derived from OCR text."""

    document_type: DocumentType | None = Field(
        description="`id`, `plate`, `mrz`, `unknown`, or null when undetermined."
    )
    fields: list[StructuredField] = Field(
        description="Extracted fields, one entry per name."
    )
    raw_text: str = Field(description="OCR lines fed to the model, for audit.")


class StructuredSection(AnalysisSection):
    """LLM structured extraction (`structured`)."""

    record: StructuredRecord = Field(description="Structured document data.")


class FieldComparison(BaseModel):
    """One field compared between the printed text and the code strip."""

    field: str = Field(description="Compared field, e.g. `date_of_birth`.")
    verdict: Verdict = Field(description="Agreement verdict.")
    mrz_value: str | None = Field(description="The code strip's value.")
    printed_value: str | None = Field(description="The printed side's value.")
    printed_confidence: float | None = Field(
        description="Model confidence for the printed value."
    )


class PhotoComparison(BaseModel):
    """One portrait attribute compared against the document's data."""

    field: str = Field(description="`age` or `sex`.")
    verdict: Verdict = Field(description="Agreement verdict.")
    document_value: str | None = Field(description="The document's value.")
    photo_value: str | None = Field(description="The photo model's label.")
    photo_confidence: float | None = Field(
        description="Photo model confidence for the label."
    )


class ConsistencySection(AnalysisSection):
    """Cross-check of printed fields, code strip, and portrait (`consistency`)."""

    comparisons: list[FieldComparison] = Field(
        description="One row per compared field."
    )
    photo_comparisons: list[PhotoComparison] = Field(
        description="Photo-vs-document rows (age, sex)."
    )
    consistent: bool = Field(
        description="False when any row is a mismatch or no usable evidence exists."
    )


class AnalysisSections(BaseModel):
    """Sections returned by a finished job (one per requested type)."""

    face: FaceSection | None = None
    attributes: AttributesSection | None = None
    antispoof: AntiSpoofSection | None = None
    ocr: OCRSection | None = None
    mrz: MRZSection | None = None
    barcode: BarcodeSection | None = None
    structured: StructuredSection | None = None
    consistency: ConsistencySection | None = None


class JobPage(AnalysisSections):
    """One PDF page's analysis sections."""

    page: int = Field(description="1-based page number.")


class JobResponse(AnalysisSections):
    """200 response for GET /jobs/{job_id} — status, result, or error."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "queued"},
                {
                    "status": "done",
                    "mrz": {
                        "record": {
                            "mrz_type": "TD3",
                            "document_code": "P<",
                            "issuer": "UTO",
                            "document_number": "L898902C3",
                            "birth_date": "740812",
                            "sex": "F",
                            "expiry_date": "120415",
                            "nationality": "UTOPIAN",
                            "surname": "ERIKSSON",
                            "given_names": "ANNA MARIA",
                            "optional_data1": "ZE184226B<<<<<10",
                            "optional_data2": "",
                        },
                        "processing_time_ms": 12.4,
                    },
                },
                {"status": "failed", "error": "stored image bytes failed validation"},
            ]
        }
    )

    status: Literal["queued", "running", "done", "failed"] = Field(
        description="`queued` and `running` have no result yet."
    )
    error: str | None = Field(
        default=None, description="Failure reason when `status` is `failed`."
    )
    pages: list[JobPage] | None = Field(
        default=None,
        description=(
            "Per-page sections for PDF uploads; image jobs return "
            "sections at the top level."
        ),
    )
