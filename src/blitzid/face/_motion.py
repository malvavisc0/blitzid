"""Challenge-response liveness from face motion across frames.

A capture session records three frames while the user performs one
random action. This module verifies the action actually happened from
the SCRFD landmarks and box alone: ``turn_head`` moves the nose along
the eye axis, ``smile`` widens the mouth relative to the eye gap, and
``move_closer`` grows the face. A printed photo or a static screen
image cannot do any of them, which is the whole liveness check.

All measurements are normalized by the inter-eye distance, so they are
scale- and resolution-independent. The actions are mirror-safe: a
flipped selfie preview changes the sign of the nose shift but not its
size, so ``turn_head`` accepts either direction.

Pure geometry over already-detected faces: no models, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from blitzid.exceptions import BlitzIDError
from blitzid.face._face import Face

__all__ = ["LivenessAction", "MotionEvidence", "verify_action"]

LivenessAction = Literal["turn_head", "smile", "move_closer"]

_TURN_MIN = 0.12
_SMILE_MIN = 0.15
_CLOSER_MIN = 1.25


@dataclass(frozen=True)
class MotionEvidence:
    """What one action check measured across three frames.

    Attributes:
        verified: Whether the action was performed convincingly.
        nose_shift: Nose travel along the eye axis, in eye-gap units.
        mouth_ratio_change: Mouth widening relative to the first
            frame (0.2 means 20% wider).
        size_ratio: Face height of the last frame over the first.
    """

    verified: bool
    nose_shift: float
    mouth_ratio_change: float
    size_ratio: float


def verify_action(action: LivenessAction, faces: Sequence[Face]) -> MotionEvidence:
    """Verify one liveness action from three detected faces.

    Args:
        action: The action the challenge requested.
        faces: One face per frame, in capture order (exactly three).

    Returns:
        The measurements and whether they prove the action.

    Raises:
        BlitzIDError: If *faces* does not hold exactly three faces.
    """
    if len(faces) != 3:
        raise BlitzIDError(f"verify_action needs exactly three faces, got {len(faces)}")
    first, _, last = faces[0], faces[1], faces[2]
    nose_shift = abs(_nose_t(last) - _nose_t(first))
    mouth_change = _mouth_ratio(last) / _mouth_ratio(first) - 1.0
    size_ratio = _height(last) / _height(first)
    if action == "turn_head":
        verified = nose_shift >= _TURN_MIN
    elif action == "smile":
        verified = mouth_change >= _SMILE_MIN
    else:
        verified = size_ratio >= _CLOSER_MIN
    return MotionEvidence(
        verified=verified,
        nose_shift=round(nose_shift, 4),
        mouth_ratio_change=round(mouth_change, 4),
        size_ratio=round(size_ratio, 4),
    )


def _eye_gap(face: Face) -> float:
    (x1, y1), (x2, y2), *_ = face.landmarks
    return math.hypot(x2 - x1, y2 - y1)


def _nose_t(face: Face) -> float:
    """Nose position along the eye axis, in eye-gap units from center."""
    (x1, y1), (x2, y2), (nx, ny), *_ = face.landmarks
    gap = _eye_gap(face)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    return ((nx - mid_x) * (x2 - x1) + (ny - mid_y) * (y2 - y1)) / (gap * gap)


def _mouth_ratio(face: Face) -> float:
    *_, (mx1, my1), (mx2, my2) = face.landmarks
    return math.hypot(mx2 - mx1, my2 - my1) / _eye_gap(face)


def _height(face: Face) -> float:
    return float(face.bbox[3])
