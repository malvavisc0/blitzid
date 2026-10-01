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


class VerifyResponse(BaseModel):
    """200 response for POST /verify — 1:1 face comparison."""

    verified: bool
    similarity: float
    threshold: float
    processing_time_ms: float
    backend: str


class HealthResponse(BaseModel):
    """200 response for GET /health — engine/Redis/job availability."""

    status: str
    models: dict[str, bool]
    redis: bool
    jobs: dict[str, int]
