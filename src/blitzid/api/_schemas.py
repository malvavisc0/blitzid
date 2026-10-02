"""Pydantic request/response models for the blitzid API."""

from __future__ import annotations

from pydantic import BaseModel


class JobCreated(BaseModel):
    """202 response for POST /analyze."""

    job_id: str
    status: str
    status_url: str


class CropResponse(BaseModel):
    """200 response for POST /crop — QC verdict plus canonical crop."""

    quad: list[list[int]] | None
    crop_base64: str | None
    width: int
    height: int
    side: str
    face_found: bool | None
    checks: dict[str, str]
    verdict: str


class FaceRef(BaseModel):
    """A detected face's location and score (e.g. an unpicked candidate)."""

    bbox: list[int]
    confidence: float


class ComparedFace(BaseModel):
    """One side of a /verify comparison — the evidence that was used."""

    bbox: list[int]
    confidence: float
    crop_base64: str
    alternatives: list[FaceRef]


class ChallengeResponse(BaseModel):
    """200 response for POST /liveness/challenge."""

    challenge_id: str
    action: str
    expires_in: float


class LivenessResponse(BaseModel):
    """200 response for POST /liveness/session."""

    live: bool
    action: str
    nose_shift: float
    mouth_ratio_change: float
    size_ratio: float
    processing_time_ms: float


class VerifyResponse(BaseModel):
    """200 response for POST /verify — 1:1 face comparison with evidence."""

    verified: bool
    similarity: float
    threshold: float
    face1: ComparedFace
    face2: ComparedFace
    processing_time_ms: float
    backend: str


class HealthResponse(BaseModel):
    """200 response for GET /health — engine/Redis/job availability."""

    status: str
    models: dict[str, bool]
    redis: bool
    jobs: dict[str, int]
