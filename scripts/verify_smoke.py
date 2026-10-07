"""FaceVerifier smoke test — runs the full verification pipeline offline.

Exercises ArcFace face verification on the committed specimen and historical fixtures:
the specimen ID portrait against itself (expect similarity ~1.0),
against a rescaled copy of itself (expect still verified), and two
different people from the 1927 Solvay conference photo against each
other (expect a low similarity, not verified). Checks that the same pair
verifies at its similarity threshold and fails just above it. Weights
download on first use; this is run on demand, not in the pytest suite.

Run example::

    uv run python scripts/verify_smoke.py

Exits non-zero if any expectation fails.
"""

from __future__ import annotations

import logging
from math import nextafter

import cv2
import numpy as np

from blitzid import FaceVerifier

_ID_IMAGE = "images/bub_der_personalausweis_kopie.jpg"
_CONFERENCE_IMAGE = "images/solvay_conference_1927.jpg"


def _load(path: str) -> np.ndarray:
    """Load an image, failing fast when it cannot be read."""
    img = cv2.imread(path)
    assert img is not None, f"could not read {path}"
    return img


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    """Run the smoke test and exit non-zero on failure."""
    logging.basicConfig(level=logging.WARNING)

    verifier = FaceVerifier(log_level=logging.WARNING)
    print(f"threshold: {verifier.threshold}")

    specimen = _load(_ID_IMAGE)
    rescaled = cv2.resize(specimen, None, fx=1.5, fy=1.5)

    print("— specimen portrait vs itself —")
    result = verifier.verify(specimen, specimen)
    print(f"  verified={result.verified} similarity={result.similarity:.3f}")
    _expect(
        result.verified
        and result.threshold == verifier.threshold
        and result.similarity >= verifier.threshold,
        "same image must verify at the configured threshold",
    )
    _expect(result.similarity > 0.99, f"expected ~1.0, got {result.similarity:.3f}")

    print("— specimen portrait vs 1.5x rescaled copy —")
    result = verifier.verify(specimen, rescaled)
    print(f"  verified={result.verified} similarity={result.similarity:.3f}")
    _expect(
        result.verified
        and result.threshold == verifier.threshold
        and result.similarity >= verifier.threshold,
        "rescaled same face must verify at the configured threshold",
    )

    print("— two people from the conference photo —")
    conference = _load(_CONFERENCE_IMAGE)
    faces = verifier.detector.detect_face_landmarks(conference)
    _expect(len(faces) >= 2, "expected at least two faces in the conference photo")
    result = verifier.verify_faces(conference, faces[0], conference, faces[1])
    print(f"  verified={result.verified} similarity={result.similarity:.3f}")
    _expect(
        not result.verified
        and result.threshold == verifier.threshold
        and result.similarity < verifier.threshold,
        "two different people must not verify at the configured threshold",
    )
    _expect(
        result.similarity < 0.25,
        f"expected a low similarity, got {result.similarity:.3f}",
    )

    similarity = result.similarity
    print("conference faces at the threshold boundary:")
    for threshold, expected_verified in (
        (similarity, True),
        (nextafter(similarity, 1.0), False),
    ):
        verifier.threshold = threshold
        result = verifier.verify_faces(conference, faces[0], conference, faces[1])
        print(
            f"  threshold={result.threshold!r} verified={result.verified} "
            f"similarity={result.similarity!r}"
        )
        _expect(
            result.threshold == threshold
            and result.similarity == similarity
            and result.verified == expected_verified,
            f"threshold {threshold!r} must yield verified={expected_verified}",
        )

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
