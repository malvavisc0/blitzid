"""Tests for challenge-response liveness motion checks (pure geometry)."""

from __future__ import annotations

import pytest

from blitzid import Face
from blitzid.face._motion import MotionEvidence, verify_action


def _face(nose_x: float = 50, mouth_half: float = 10, height: int = 60) -> Face:
    """One synthetic face: eye gap scales with the box (gap = height / 3)."""
    eye_half = height / 6
    landmarks = (
        (50.0 - eye_half, 40.0),
        (50.0 + eye_half, 40.0),
        (nose_x, 55.0),
        (50.0 - mouth_half, 70.0),
        (50.0 + mouth_half, 70.0),
    )
    return Face(bbox=(30, 10, 40, height), confidence=0.9, landmarks=landmarks)


class TestTurnHead:
    def test_nose_travel_verifies(self) -> None:
        evidence = verify_action("turn_head", [_face(48), _face(52), _face(58)])
        assert isinstance(evidence, MotionEvidence)
        assert evidence.verified is True
        assert evidence.nose_shift == 0.5

    def test_mirror_direction_also_verifies(self) -> None:
        assert verify_action("turn_head", [_face(58), _face(52), _face(48)]).verified

    def test_static_face_fails(self) -> None:
        evidence = verify_action("turn_head", [_face(50), _face(50.5), _face(50.5)])
        assert evidence.verified is False

    def test_small_jitter_fails(self) -> None:
        evidence = verify_action("turn_head", [_face(50), _face(50.4), _face(50.8)])
        assert evidence.verified is False


class TestSmile:
    def test_widening_mouth_verifies(self) -> None:
        evidence = verify_action("smile", [_face(50, 10), _face(50, 11), _face(50, 12)])
        assert evidence.verified is True
        assert evidence.mouth_ratio_change == 0.2

    def test_static_mouth_fails(self) -> None:
        evidence = verify_action("smile", [_face(50, 10), _face(50, 10), _face(50, 10)])
        assert evidence.verified is False

    def test_zoom_does_not_fake_a_smile(self) -> None:
        """Scaling a face keeps the mouth-to-eye ratio constant."""
        zoomed = [_face(50, 10, 60), _face(50, 12, 72), _face(50, 14, 84)]
        assert verify_action("smile", zoomed).verified is False


class TestMoveCloser:
    def test_growing_face_verifies(self) -> None:
        evidence = verify_action(
            "move_closer", [_face(height=60), _face(height=70), _face(height=80)]
        )
        assert evidence.verified is True
        assert evidence.size_ratio == pytest.approx(80 / 60, rel=1e-3)

    def test_static_distance_fails(self) -> None:
        zoomed = [_face(height=60), _face(height=62), _face(height=62)]
        assert verify_action("move_closer", zoomed).verified is False
