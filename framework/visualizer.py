"""Visualization utilities for face detections."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class FaceVisualizer:
    """Visualizes face detections on images."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize visualizer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def draw_detections(
        self,
        image: np.ndarray,
        faces: List[Tuple[int, int, int, int, float]],
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Image to draw on (will be copied)
            faces: List of (x, y, w, h, confidence) tuples
            show_confidence: Whether to show confidence scores
            color: BGR color for bounding boxes
            thickness: Line thickness

        Returns:
            Image with drawn bounding boxes
        """
        img_copy = image.copy()

        for x, y, w, h, conf in faces:
            # Draw rectangle
            self._draw_box(img_copy, x, y, w, h, color, thickness)

            # Draw confidence score
            if show_confidence:
                self._draw_label(img_copy, x, y, f"{conf:.2f}", color)

        return img_copy

    def save_visualization(self, image: np.ndarray, output_path: Path) -> None:
        """
        Save visualized image.

        Args:
            image: Image to save
            output_path: Path to save image
        """
        cv2.imwrite(str(output_path), image)
        self.logger.info("Visualization saved to %s", output_path)

    def _draw_box(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """
        Draw single bounding box.

        Args:
            image: Image to draw on (modified in place)
            x, y: Top-left corner
            w, h: Width and height
            color: BGR color
            thickness: Line thickness
        """
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

    def _draw_label(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        label: str,
        color: Tuple[int, int, int],
    ) -> None:
        """
        Draw confidence label.

        Args:
            image: Image to draw on (modified in place)
            x, y: Position for label
            label: Text to draw
            color: BGR color for background
        """
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(y, label_size[1] + 10)

        # Draw background rectangle
        cv2.rectangle(
            image,
            (x, label_y - label_size[1] - 10),
            (x + label_size[0], label_y),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
