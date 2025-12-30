"""Quick assertion-based smoke tests for the refactored framework.

This is intentionally a plain Python script (no pytest dependency). It validates:
- importability and basic construction
- deterministic default `model_dir`
- bbox validity invariants (non-negative, within image bounds)
- cache hit behavior and metrics timing semantics

Run:
    python test_refactored.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from framework import (
    FaceDetectorDeepFace,
    FaceDetectorDNN,
    OptionalDependencyError,
)


def _assert_bbox_valid(
    faces: list[tuple[int, int, int, int, float]],
    image_size: tuple[int, int],
) -> None:
    img_w, img_h = image_size

    for x, y, w, h, conf in faces:
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(w, int) and isinstance(h, int)
        assert 0 <= x < img_w, (x, img_w)
        assert 0 <= y < img_h, (y, img_h)
        assert w > 0 and h > 0, (w, h)
        assert x + w <= img_w, (x, w, img_w)
        assert y + h <= img_h, (y, h, img_h)
        assert 0.0 <= float(conf) <= 1.0, conf


def main() -> None:
    """Run lightweight smoke tests against the refactored framework."""
    repo_root = Path(__file__).resolve().parent
    expected_model_dir = (repo_root / "models").resolve()

    print("Testing refactored framework...")

    print("\n1) Constructing detector")
    detector = FaceDetectorDNN(
        confidence_threshold=0.5,
        enable_cache=True,
        max_cache_size=16,
    )
    print(f"   backend: {detector.backend_type}")

    print("\n2) Checking deterministic model_dir")
    actual_model_dir = detector.model_manager.model_dir.resolve()
    assert (
        actual_model_dir == expected_model_dir
    ), f"Expected model_dir={expected_model_dir}, got {actual_model_dir}"

    print("\n3) Synthetic image input (numpy array)")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    faces = detector.detect_face(img)
    _assert_bbox_valid(faces, (img.shape[1], img.shape[0]))
    print(f"   faces detected on synthetic image: {len(faces)}")

    print("\n4) Cache + metrics behavior")

    # Ensure a known cache state: step (3) may have already populated the cache.
    detector.clear_cache()

    r1 = detector.detect_face_with_metrics(img)
    assert r1.cache_hit is False

    r2 = detector.detect_face_with_metrics(img)
    assert r2.cache_hit is True
    assert r2.processing_time == 0.0

    print(
        f"   first call: cache_hit={r1.cache_hit}, time_ms={r1.processing_time * 1000:.2f}"
    )
    print(
        f"   second call: cache_hit={r2.cache_hit}, time_ms={r2.processing_time * 1000:.2f}"
    )

    print("\n5) Path input (repo image)")
    image_path = repo_root / "images" / "IMG_3435.jpg"
    if image_path.exists():
        try:
            import cv2

            raw = cv2.imread(str(image_path))
            assert raw is not None
            img_h, img_w = raw.shape[:2]

            faces = detector.detect_face(image_path)
            _assert_bbox_valid(faces, (img_w, img_h))
            print(f"   faces detected on {image_path.name}: {len(faces)}")
        except Exception as e:
            raise AssertionError(f"Path input test failed: {e}") from e
    else:
        print(f"   skipped (missing file): {image_path}")

    print("\n6) Optional DeepFace backend (detection + alignment + analyze)")
    try:
        deep = FaceDetectorDeepFace(log_level=50)

        # Smoke-check extract_faces on a synthetic image.
        extracted = deep.extract_faces(img)
        assert isinstance(extracted, list)
        for face_img, (x, y, w, h), conf in extracted:
            assert face_img.ndim == 3 and face_img.shape[2] == 3
            assert w >= 0 and h >= 0
            assert 0.0 <= float(conf) <= 1.0
        print(
            f"   DeepFace backend available; extracted crops: {len(extracted)}"
        )

        # Smoke-check *our wrapper* analyze() on a real face crop (requires repo image).
        # This calls [`FaceDetectorDeepFace.analyze()`](framework/deepface_facade.py:130),
        # not `deepface.DeepFace.analyze()` directly.
        image_path = repo_root / "images" / "bub_der_personalausweis_kopie.jpg"
        if image_path.exists():
            extracted_real = deep.extract_faces(image_path)
            if extracted_real:
                face0, _bbox0, _conf0 = extracted_real[0]
                attrs = deep.analyze(
                    face0, actions=["age", "gender", "race", "emotion"]
                )

                # DeepFace may return either a dict or a list[dict].
                if isinstance(attrs, list):
                    assert attrs, "DeepFace.analyze returned an empty list"
                    assert isinstance(attrs[0], dict)
                    attrs0 = attrs[0]
                else:
                    assert isinstance(attrs, dict)
                    attrs0 = attrs

                assert "age" in attrs0, attrs0
                assert "gender" in attrs0, attrs0
                assert "race" in attrs0, attrs0
                assert "emotion" in attrs0, attrs0
                print("   analyze() OK (age/gender/race/emotion keys present)")
            else:
                print(
                    f"   skipped analyze (no face detected in {image_path.name})"
                )
        else:
            print(f"   skipped analyze (missing file): {image_path}")

    except OptionalDependencyError as e:
        print(f"   skipped (DeepFace not installed): {e}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
