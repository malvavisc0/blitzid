"""DeepFace-based detection + alignment backend.

Optional DeepFace integration that provides an alternative to the OpenCV DNN
detector for batch/offline pipelines where quality matters more than speed.

DeepFace is an optional dependency.  Import errors are converted to
:class:`blitzid.exceptions.BlitzIDError`.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from ._image import ImageInput, load_image
from ._models import default_model_dir
from .detector import DetectionMetrics, _draw_detections
from .exceptions import BlitzIDError, ImageError


class FaceDetectorDeepFace:
    """DeepFace-backed detector that returns bboxes and aligned crops.

    Public methods intentionally mirror the commonly-used subset of
    :class:`blitzid.detector.FaceDetectorDNN`.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        detector_backend: str = "retinaface",
        align: bool = True,
        enforce_detection: bool = False,
        log_level: int = logging.INFO,
    ):
        self.model_dir = (
            Path(model_dir) if model_dir is not None else None
        ) or default_model_dir("deepface")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.detector_backend = str(detector_backend)
        self.align = bool(align)
        self.enforce_detection = bool(enforce_detection)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Point DeepFace weight storage at our model_dir before importing.
        os.environ.setdefault("DEEPFACE_HOME", str(self.model_dir))

        self._deepface = self._import_deepface()

    @staticmethod
    def _import_deepface() -> Any:
        try:
            from deepface import DeepFace  # type: ignore

            return DeepFace
        except Exception as e:  # pragma: no cover
            raise BlitzIDError(
                "DeepFace is not installed or failed to import. "
                "Install optional dependencies (DeepFace + its TF RetinaFace "
                "backend) to use FaceDetectorDeepFace."
            ) from e

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_bgr(face: np.ndarray) -> np.ndarray:
        """Convert a DeepFace face crop into BGR uint8."""
        if not isinstance(face, np.ndarray):
            return face

        if face.ndim == 3 and face.shape[2] == 3:
            img = face
            if np.issubdtype(img.dtype, np.floating):
                mx = float(np.max(img)) if img.size else 0.0
                if mx <= 1.0:
                    img = (img * 255.0).clip(0.0, 255.0)
                img = img.clip(0.0, 255.0).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return face

    def _load_and_convert_rgb(
        self, image_input: ImageInput
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load image once, return ``(bgr, rgb)`` pair."""
        img_bgr = load_image(image_input, self.logger)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr, img_rgb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_input: ImageInput,
        actions: Sequence[str] | None = None,
        detector_backend: str | None = None,
        align: bool | None = True,
        enforce_detection: bool | None = None,
    ) -> Any:
        """Wrap ``DeepFace.analyze``."""
        backend = (
            detector_backend if detector_backend is not None else self.detector_backend
        )
        do_align = align if align is not None else self.align
        enforce = (
            enforce_detection
            if enforce_detection is not None
            else self.enforce_detection
        )

        if isinstance(image_input, (str, Path)):
            return self._deepface.analyze(
                img_path=str(image_input),
                actions=cast(Any, list(actions) if actions is not None else None),
                detector_backend=backend,
                enforce_detection=enforce,
                align=do_align,
            )

        _, img_rgb = self._load_and_convert_rgb(image_input)
        return self._deepface.analyze(
            img_path=img_rgb,
            actions=cast(Any, list(actions) if actions is not None else None),
            detector_backend=backend,
            enforce_detection=enforce,
            align=do_align,
        )

    def detect_face(
        self, image_input: ImageInput
    ) -> list[tuple[int, int, int, int, float]]:
        """Detect faces (bbox-only output contract)."""
        crops = self.extract_faces(image_input)
        faces: list[tuple[int, int, int, int, float]] = [
            (x, y, w, h, float(conf)) for _img, (x, y, w, h), conf in crops
        ]
        return sorted(faces, key=lambda t: t[4], reverse=True)

    def detect_face_with_metrics(self, image_input: ImageInput) -> DetectionMetrics:
        """Detect faces and return metrics (single load)."""
        img_bgr, img_rgb = self._load_and_convert_rgb(image_input)
        h, w = img_bgr.shape[:2]

        start = time.time()
        results = self._deepface.extract_faces(
            img_path=img_rgb,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )
        faces = self._results_to_faces(results)
        processing_time = time.time() - start

        return DetectionMetrics(
            faces=faces,
            processing_time=processing_time,
            image_size=(w, h),
            backend=f"DeepFace:{self.detector_backend}",
            num_faces=len(faces),
            cache_hit=False,
        )

    def extract_faces(
        self,
        image_input: ImageInput,
        padding: float = 0.0,
    ) -> list[tuple[NDArray[np.uint8], tuple[int, int, int, int], float]]:
        """Detect and extract aligned face crops.

        Note: ``padding`` is currently ignored by the DeepFace backend.
        """
        if padding != 0.0:
            self.logger.debug(
                "padding is ignored for DeepFace backend (requested=%s)",
                padding,
            )

        try:
            _, img_rgb = self._load_and_convert_rgb(image_input)
        except ImageError:
            raise
        except Exception as e:  # pragma: no cover
            raise ImageError(f"Failed to load input for DeepFace: {e}") from e

        results = self._deepface.extract_faces(
            img_path=img_rgb,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )

        extracted: list[tuple[NDArray[np.uint8], tuple[int, int, int, int], float]] = []
        for item in results:
            face = cast(np.ndarray, item.get("face"))
            facial_area: dict[str, int] = cast(dict[str, int], item.get("facial_area"))
            conf = float(item.get("confidence", 0.0))

            x = int(facial_area.get("x", 0))
            y = int(facial_area.get("y", 0))
            w = int(facial_area.get("w", 0))
            h = int(facial_area.get("h", 0))

            face_bgr = cast(NDArray[np.uint8], self._as_bgr(face))
            extracted.append((face_bgr, (x, y, w, h), conf))

        extracted.sort(key=lambda t: t[2], reverse=True)
        return extracted

    def visualize_detections(
        self,
        image_input: ImageInput,
        faces: list[tuple[int, int, int, int, float]] | None = None,
        output_path: Path | None = None,
        show_confidence: bool = True,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        """Draw bounding boxes on image (single load)."""
        img_bgr, img_rgb = self._load_and_convert_rgb(image_input)

        if faces is None:
            # Detect from the already-loaded RGB array (no double load).
            results = self._deepface.extract_faces(
                img_path=img_rgb,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
            )
            faces = self._results_to_faces(results)

        result = _draw_detections(img_bgr, faces, show_confidence, color, thickness)
        if output_path:
            cv2.imwrite(str(output_path), result)
            self.logger.info("Visualization saved to %s", output_path)
        return result

    def detect_faces_batch(
        self,
        image_paths: list[Path | str],
        show_progress: bool = True,
    ) -> dict[str, list[tuple[int, int, int, int, float]]]:
        """Process multiple images in batch."""
        results: dict[str, list[tuple[int, int, int, int, float]]] = {}
        total = len(image_paths)

        for idx, p in enumerate(image_paths, 1):
            path = Path(p)
            try:
                faces = self.detect_face(path)
                results[str(path)] = faces
                if show_progress:
                    self.logger.info(
                        "Progress: %d/%d - %s: %d faces",
                        idx,
                        total,
                        path.name,
                        len(faces),
                    )
            except BlitzIDError as e:  # pragma: no cover
                self.logger.error("Error processing %s: %s", path, e)
                results[str(path)] = []

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _results_to_faces(
        self, results: list[Any]
    ) -> list[tuple[int, int, int, int, float]]:
        """Convert DeepFace extract_faces results to bbox tuples."""
        faces: list[tuple[int, int, int, int, float]] = []
        for item in results:
            facial_area = item.get("facial_area", {})
            conf = float(item.get("confidence", 0.0))
            x = int(facial_area.get("x", 0))
            y = int(facial_area.get("y", 0))
            w = int(facial_area.get("w", 0))
            h = int(facial_area.get("h", 0))
            faces.append((x, y, w, h, conf))
        return sorted(faces, key=lambda t: t[4], reverse=True)
