"""DocumentCropper — ID document localization, perspective crop, and QC.

Classical CV (no neural networks): the largest high-contrast convex
quadrilateral is located via Canny edges + contour approximation, then
:func:`cv2.warpPerspective` lifts it into an axis-aligned canonical
crop at the quad's own aspect ratio (no forced ID-1 size). The crop is
quality-checked; thresholds are module constants.

Quality checks, each ``"pass"`` / ``"warn"`` / ``"fail"`` / ``"n/a"``:

- ``document_found`` — a plausible quad (convex, at least
  :data:`MIN_DOCUMENT_AREA_FRACTION` of the image area) was found.
- ``aspect_ratio`` — crop w/h near ID-1 or ID-3 (tolerances
  :data:`ASPECT_PASS_TOLERANCE` / :data:`ASPECT_WARN_TOLERANCE`).
- ``resolution`` — short side in pixels: >= 600 pass, >= 400 warn,
  below fail.
- ``sharpness`` — Laplacian variance of the crop: >=
  :data:`SHARPNESS_PASS_VAR` pass, >= :data:`SHARPNESS_WARN_VAR` warn,
  below fail.
- ``brightness`` — mean intensity in ``[60, 200]`` pass, in the warn
  margins around it warn, outside fail.
- ``face_present`` — side-aware (see :class:`DocumentCropper`); never
  fails a document.

The optional injected face detector keeps this module pure CV without
it — the detector type is imported only for type checking, so the
reading domain gains no runtime dependency on the face domain. The
optional ``detector_lock`` constructor argument serializes the
detection call when the detector is shared across threads.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from .._image import ImageInput, load_image
from ..exceptions import BlitzIDError

if TYPE_CHECKING:
    from ..face.detector import FaceDetectorDNN

__all__ = ["DocumentCropper", "QualityReport"]

CheckResult = Literal["pass", "warn", "fail", "n/a"]

ID1_ASPECT_RATIO = 1.585
ID3_ASPECT_RATIO = 1.389
ASPECT_PASS_TOLERANCE = 0.15
ASPECT_WARN_TOLERANCE = 0.35
RESOLUTION_PASS_PX = 600
RESOLUTION_WARN_PX = 400
SHARPNESS_PASS_VAR = 120.0
SHARPNESS_WARN_VAR = 25.0
BRIGHTNESS_MIN = 60.0
BRIGHTNESS_MAX = 200.0
BRIGHTNESS_WARN_MARGIN = 20.0
MIN_DOCUMENT_AREA_FRACTION = 0.08

_SIDES = frozenset({"front", "back", "unknown"})
_APPROX_EPSILONS = (0.02, 0.04, 0.08)


@dataclass(frozen=True)
class QualityReport:
    """Quality-control verdict for one document crop attempt.

    Attributes:
        quad: Detected document corners ``(x, y)`` clockwise from
            top-left, or None when no document was found.
        width: Warped crop width in pixels (0 when no document).
        height: Warped crop height in pixels (0 when no document).
        side: Reported document side (``"front"``, ``"back"``, or
            ``"unknown"``).
        face_found: Whether a face was detected on the crop. None when
            no detector is available or detection was not run (back
            side).
        checks: Quality checks, each ``"pass"``, ``"warn"``, ``"fail"``,
            or ``"n/a"``.
        verdict: ``"pass"`` (no fail), ``"warn"`` (warns, no fail), or
            ``"reject"`` (any fail).
    """

    quad: tuple[tuple[int, int], ...] | None
    width: int
    height: int
    side: str
    face_found: bool | None
    checks: dict[str, CheckResult]
    verdict: str


def _order_corners(points: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Order quad corners clockwise from top-left."""
    pts = points.reshape(4, 2).astype(np.float64)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]
    ordered = (
        pts[np.argmin(sums)],
        pts[np.argmin(diffs)],
        pts[np.argmax(sums)],
        pts[np.argmax(diffs)],
    )
    return tuple((round(x), round(y)) for x, y in ordered)


def _approx_quad(contour: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    """Approximate a contour as a convex quad, or None."""
    perimeter = cv2.arcLength(contour, True)
    for epsilon in _APPROX_EPSILONS:
        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx)
    return None


def _quad_area(quad: tuple[tuple[int, int], ...]) -> float:
    """Shoelace area of a quad."""
    pts = np.asarray(quad, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2.0


def _detect_document_quad(img: np.ndarray) -> tuple[tuple[int, int], ...] | None:
    """Find the largest plausible document quad, or None.

    Plausible: convex, and covering at least
    :data:`MIN_DOCUMENT_AREA_FRACTION` of the image area.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_DOCUMENT_AREA_FRACTION * float(img.shape[0] * img.shape[1])
    best: tuple[tuple[int, int], ...] | None = None
    best_area = min_area
    for contour in contours:
        quad = _approx_quad(contour)
        if quad is None:
            continue
        area = _quad_area(quad)
        if area > best_area:
            best, best_area = quad, area
    return best


def _warp_quad(img: np.ndarray, quad: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Warp the quad into an axis-aligned crop at its own aspect."""
    tl, tr, br, bl = (np.asarray(point, dtype=np.float64) for point in quad)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    src = np.asarray(quad, dtype=np.float32)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (width, height))


def _check_aspect_ratio(crop: np.ndarray) -> CheckResult:
    """Ratio near ID-1 or ID-3 passes; further out warns, then fails."""
    height, width = crop.shape[:2]
    ratio = width / height
    bases = (ID1_ASPECT_RATIO, ID3_ASPECT_RATIO)
    delta = min(abs(ratio - base) / base for base in bases)
    if delta <= ASPECT_PASS_TOLERANCE:
        return "pass"
    if delta <= ASPECT_WARN_TOLERANCE:
        return "warn"
    return "fail"


def _check_resolution(crop: np.ndarray) -> CheckResult:
    """Short side in pixels: >= 600 pass, >= 400 warn, below fail."""
    short_side = min(crop.shape[:2])
    if short_side >= RESOLUTION_PASS_PX:
        return "pass"
    if short_side >= RESOLUTION_WARN_PX:
        return "warn"
    return "fail"


def _check_sharpness(crop: np.ndarray) -> CheckResult:
    """Laplacian variance of the crop: high passes, low fails."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance >= SHARPNESS_PASS_VAR:
        return "pass"
    if variance >= SHARPNESS_WARN_VAR:
        return "warn"
    return "fail"


def _check_brightness(crop: np.ndarray) -> CheckResult:
    """Mean intensity in bounds passes; the margins warn; outside fails."""
    mean = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
    if BRIGHTNESS_MIN <= mean <= BRIGHTNESS_MAX:
        return "pass"
    if (
        BRIGHTNESS_MIN - BRIGHTNESS_WARN_MARGIN
        <= mean
        <= BRIGHTNESS_MAX + BRIGHTNESS_WARN_MARGIN
    ):
        return "warn"
    return "fail"


def _check_face_present(
    crop: np.ndarray,
    side: str,
    detector: FaceDetectorDNN | None,
    detector_lock: Lock | None,
) -> tuple[bool | None, CheckResult]:
    """Side-aware face check; never fails a document.

    front: face -> pass, no face -> warn. back: n/a (not expected —
    TD1 MRZ lives on the back; detection is not run). unknown: n/a with
    face_found reported as information only. No detector: n/a and
    face_found None. *detector_lock*, when given, serializes the
    detection call against other users of the same shared detector.
    """
    if detector is None or side == "back":
        return None, "n/a"
    with detector_lock if detector_lock is not None else nullcontext():
        found = bool(detector.detect_face_landmarks(crop))
    if side == "front":
        return found, "pass" if found else "warn"
    return found, "n/a"


def _verdict(checks: dict[str, CheckResult]) -> str:
    """Any fail rejects; otherwise any warn warns; else pass."""
    if "fail" in checks.values():
        return "reject"
    if "warn" in checks.values():
        return "warn"
    return "pass"


def _no_document_report(side: str) -> QualityReport:
    """Report for an image with no plausible document quad."""
    return QualityReport(
        quad=None,
        width=0,
        height=0,
        side=side,
        face_found=None,
        checks={
            "document_found": "fail",
            "aspect_ratio": "n/a",
            "resolution": "n/a",
            "sharpness": "n/a",
            "brightness": "n/a",
            "face_present": "n/a",
        },
        verdict="reject",
    )


class DocumentCropper:
    """Locates an ID document, warps it flat, and quality-checks the crop.

    Args:
        detector: Optional :class:`~blitzid.FaceDetectorDNN` used for the
            side-aware ``face_present`` check. Without it the check
            reports ``"n/a"`` and ``face_found`` is None; detection
            runs on the warped crop (the canonical frame clients save).
        detector_lock: Optional lock serializing the face-detection
            call against other threads sharing the same *detector*
            instance (the API injects its per-engine lock here).
        log_level: Logging level for the cropper's logger.

    Raises:
        BlitzIDError: If *side* is not ``"front"``, ``"back"``, or
            ``"unknown"`` (from :meth:`crop`).
    """

    def __init__(
        self,
        detector: FaceDetectorDNN | None = None,
        detector_lock: Lock | None = None,
        log_level: int = logging.WARNING,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self._detector = detector
        self._detector_lock = detector_lock

    def crop(
        self, image_input: ImageInput, side: str = "unknown"
    ) -> tuple[np.ndarray | None, QualityReport]:
        """Detect the document, warp it flat, and run quality checks.

        Args:
            image_input: Path, NumPy array (BGR), or PIL Image.
            side: Document side — ``"front"``, ``"back"``, or
                ``"unknown"`` (default); drives the side-aware
                ``face_present`` check.

        Returns:
            ``(crop, report)`` — the perspective-corrected BGR crop
            (None when no document was found) and its
            :class:`QualityReport`.

        Raises:
            ImageError: If the image cannot be loaded or is invalid.
            BlitzIDError: If *side* is not a known side.
        """
        if side not in _SIDES:
            raise BlitzIDError(f"side must be one of {sorted(_SIDES)}, got {side!r}")
        img = load_image(image_input, self.logger)
        quad = _detect_document_quad(img)
        if quad is None:
            self.logger.info("No document quad found")
            return None, _no_document_report(side)
        crop = _warp_quad(img, quad)
        report = self._quality_report(crop, quad, side)
        self.logger.info(
            "Document cropped: %dx%d, verdict %s",
            report.width,
            report.height,
            report.verdict,
        )
        return crop, report

    def _quality_report(
        self,
        crop: np.ndarray,
        quad: tuple[tuple[int, int], ...],
        side: str,
    ) -> QualityReport:
        """Build the full QualityReport for a warped crop."""
        face_found, face_check = _check_face_present(
            crop, side, self._detector, self._detector_lock
        )
        checks: dict[str, CheckResult] = {
            "document_found": "pass",
            "aspect_ratio": _check_aspect_ratio(crop),
            "resolution": _check_resolution(crop),
            "sharpness": _check_sharpness(crop),
            "brightness": _check_brightness(crop),
            "face_present": face_check,
        }
        height, width = crop.shape[:2]
        return QualityReport(
            quad=quad,
            width=width,
            height=height,
            side=side,
            face_found=face_found,
            checks=checks,
            verdict=_verdict(checks),
        )
