"""DeepFace-based detection + alignment backend.

This module adds an optional DeepFace integration without replacing the existing
OpenCV DNN detector.

Design goals
------------
- Keep the current bbox-only OpenCV DNN detector intact (see
  [`framework.facade.FaceDetectorDNN`](framework/facade.py:42)).
- Offer an alternative backend that uses DeepFace for *detection + alignment*
  (e.g. `detector_backend="retinaface"`, `align=True`) for batch/offline
  pipelines where quality matters more than speed.
- Store DeepFace downloaded weights under the repo-local `models/` directory.

Important
---------
DeepFace is an optional dependency. Import errors are converted to a
[`framework.exceptions.OptionalDependencyError`](framework/exceptions.py:1).

Weight storage
--------------
DeepFace stores weights under a "home" directory. We set the environment
variable `DEEPFACE_HOME` to point at `models/deepface/` *before importing*
DeepFace. If your DeepFace version does not honor this variable, you may need
an upgrade/pin.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .exceptions import (
    FaceDetectorError,
    ImageLoadError,
    OptionalDependencyError,
)
from .image_loader import ImageLoader
from .metrics import DetectionMetrics, MetricsTracker
from .visualizer import FaceVisualizer

ImageInput = Union[Path, str, NDArray[np.uint8]]


class FaceDetectorDeepFace:
    """DeepFace-backed detector that returns bboxes and aligned crops.

    Public methods intentionally mirror the commonly-used subset of
    [`framework.facade.FaceDetectorDNN`](framework/facade.py:42).
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        detector_backend: str = "retinaface",
        align: bool = True,
        enforce_detection: bool = False,
        log_level: int = logging.INFO,
    ):
        self.model_dir = (
            Path(model_dir) if model_dir is not None else None
        ) or Path(__file__).resolve().parents[1] / "models" / "deepface"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.detector_backend = str(detector_backend)
        self.align = bool(align)
        self.enforce_detection = bool(enforce_detection)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # We reuse the existing loader to normalize inputs to BGR arrays.
        # Note: DeepFace can accept paths as well, but we standardize inputs.
        self.image_loader = ImageLoader(self.logger, preprocess=False)
        self.visualizer = FaceVisualizer(self.logger)
        self.metrics = MetricsTracker()

        # Configure DeepFace weights directory before importing.
        os.environ.setdefault("DEEPFACE_HOME", str(self.model_dir))

        # Lazy import: we validate availability at construction.
        self._deepface = self._import_deepface()

    def _import_deepface(self):
        try:
            from deepface import DeepFace  # type: ignore

            return DeepFace
        except Exception as e:  # pragma: no cover
            raise OptionalDependencyError(
                "DeepFace is not installed or failed to import. "
                "Install optional dependencies (DeepFace + its TF RetinaFace backend) "
                "to use FaceDetectorDeepFace."
            ) from e

    def _as_bgr(self, face: np.ndarray) -> np.ndarray:
        """Convert a DeepFace face crop into BGR uint8 for consistency.

        DeepFace commonly returns face crops in RGB. The dtype can vary across
        versions/backends (e.g., float64 arrays in [0, 1] or [0, 255]). OpenCV
        color conversion does not accept float64, so we normalize to uint8.
        """
        if not isinstance(face, np.ndarray):
            return face

        if face.ndim == 3 and face.shape[2] == 3:
            img = face

            # Normalize dtype for OpenCV.
            if np.issubdtype(img.dtype, np.floating):
                mx = float(np.max(img)) if img.size else 0.0
                if mx <= 1.0:
                    img = (img * 255.0).clip(0.0, 255.0)
                img = img.clip(0.0, 255.0).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Heuristic: assume RGB and convert to BGR to match the rest of the framework.
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return face

    def analyze(
        self,
        image_input: ImageInput,
        actions: Optional[Sequence[str]] = None,
        detector_backend: Optional[str] = None,
        align: Optional[bool] = True,
        enforce_detection: Optional[bool] = None,
    ) -> Any:
        """Wrap `DeepFace.analyze`.

        This is useful when you already have a cut-out face image on disk and
        want to run DeepFace's attribute analysis (age/gender/emotion/etc.).

        Args:
            image_input: Path/str or image array.
            actions: DeepFace actions list (e.g. ["age", "gender"]). If None,
                DeepFace defaults are used.
            detector_backend: Override detector backend for analysis.
            align: Override alignment behavior for analysis.
            enforce_detection: Override face-enforcement behavior.

        Returns:
            The value returned by `DeepFace.analyze` (dict or list of dicts).
        """

        backend = (
            detector_backend
            if detector_backend is not None
            else self.detector_backend
        )
        do_align = align if align is not None else self.align
        enforce = (
            enforce_detection
            if enforce_detection is not None
            else self.enforce_detection
        )

        # If a cropped face image is already on disk, pass the path directly.
        if isinstance(image_input, (str, Path)):
            return self._deepface.analyze(
                img_path=str(image_input),
                actions=cast(
                    Any, list(actions) if actions is not None else None
                ),
                detector_backend=backend,
                enforce_detection=enforce,
                align=do_align,
            )

        # Otherwise normalize via the loader and convert to RGB.
        img_bgr = self.image_loader.load(image_input)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self._deepface.analyze(
            img_path=img_rgb,
            actions=cast(Any, list(actions) if actions is not None else None),
            detector_backend=backend,
            enforce_detection=enforce,
            align=do_align,
        )

    def detect_face(
        self, image_input: ImageInput
    ) -> List[Tuple[int, int, int, int, float]]:
        """Detect faces (bbox-only output contract)."""
        _crops = self.extract_faces(image_input)
        faces: List[Tuple[int, int, int, int, float]] = []
        for _img, (x, y, w, h), conf in _crops:
            faces.append((x, y, w, h, float(conf)))
        return sorted(faces, key=lambda t: t[4], reverse=True)

    def detect_face_with_metrics(
        self, image_input: ImageInput
    ) -> DetectionMetrics:
        start = time.time()
        faces = self.detect_face(image_input)
        processing_time = time.time() - start

        # Load only for image_size (fast, no preprocessing).
        img = self.image_loader.load(image_input)
        h, w = img.shape[:2]

        metrics = DetectionMetrics(
            faces=faces,
            processing_time=processing_time,
            image_size=(w, h),
            backend=f"DeepFace:{self.detector_backend}",
            num_faces=len(faces),
            cache_hit=False,
        )
        self.metrics.record_detection(
            metrics.processing_time, metrics.num_faces
        )
        return metrics

    def extract_faces(
        self,
        image_input: ImageInput,
        padding: float = 0.0,
    ) -> List[Tuple[NDArray[np.uint8], Tuple[int, int, int, int], float]]:
        """Detect and extract aligned face crops.

        Returns a list of `(face_bgr, (x, y, w, h), confidence)`.

        Args:
            image_input: Path/str or image array
            padding: Percentage padding around face (0.0-1.0). Currently ignored
                by DeepFace backend.

        Notes:
            DeepFace internally performs cropping and alignment; the `padding`
            parameter is currently ignored. It is accepted only to keep a
            compatible call signature with
            [`framework.facade.FaceDetectorDNN.extract_faces()`](framework/facade.py:192).

            **Why padding is ignored:**
            - DeepFace's `extract_faces()` returns pre-cropped, aligned face images
            - The cropping is done internally by the DeepFace library
            - The bounding box coordinates are provided, but the crop itself is
              already extracted and aligned
            - To apply padding, we would need to re-crop from the original image
              using expanded bounding boxes, which would lose the alignment

            **Future enhancement:**
            If padding support is needed, it could be implemented by:
            1. Storing the original image
            2. Expanding the bounding box by the padding percentage
            3. Re-cropping from the original image (losing alignment)

            For now, if you need padding, use the DNN backend or manually expand
            the bounding boxes and re-crop from the original image.
        """
        if padding != 0.0:
            self.logger.debug(
                "padding is ignored for DeepFace backend (requested=%s)",
                padding,
            )

        # DeepFace supports multiple input types; we pass a BGR array for stability.
        try:
            img_bgr = self.image_loader.load(image_input)
        except FaceDetectorError:
            raise
        except Exception as e:  # pragma: no cover
            raise ImageLoadError(
                f"Failed to load input for DeepFace: {e}"
            ) from e

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = self._deepface.extract_faces(
            img_path=img_rgb,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )

        extracted: List[
            Tuple[NDArray[np.uint8], Tuple[int, int, int, int], float]
        ] = []
        for item in results:
            # DeepFace returns dicts with keys like: 'face', 'facial_area', 'confidence'.
            face = cast(np.ndarray, item.get("face"))
            facial_area: Dict[str, int] = cast(
                Dict[str, int], item.get("facial_area")
            )
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
        faces: Optional[List[Tuple[int, int, int, int, float]]] = None,
        output_path: Optional[Path] = None,
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> NDArray[np.uint8]:
        img = self.image_loader.load(image_input)
        if faces is None:
            faces = self.detect_face(image_input)

        result = self.visualizer.draw_detections(
            img, faces, show_confidence, color, thickness
        )
        if output_path:
            self.visualizer.save_visualization(result, output_path)
        return result

    def detect_faces_batch(
        self, image_paths: List[Union[Path, str]], show_progress: bool = True
    ) -> Dict[str, List[Tuple[int, int, int, int, float]]]:
        results: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
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
            except FaceDetectorError as e:  # pragma: no cover
                self.logger.error("Error processing %s: %s", path, e)
                results[str(path)] = []

        return results
