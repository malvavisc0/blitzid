"""Post-processing for face detections (NMS, filtering, sorting)."""

from typing import List, Tuple


class FaceProcessor:
    """Post-processes face detections."""

    def __init__(self, nms_threshold: float = 0.3):
        """
        Initialize face processor.

        Args:
            nms_threshold: IoU threshold for Non-Maximum Suppression (0.0-1.0)
        """
        self.nms_threshold = nms_threshold

    def apply_nms(
        self, faces: List[Tuple[int, int, int, int, float]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.

        Args:
            faces: List of (x, y, w, h, confidence) tuples, should be sorted by confidence

        Returns:
            Filtered list of faces with overlaps removed
        """
        if len(faces) == 0:
            return []

        # Assumes faces are already sorted by confidence (descending)
        keep = []

        for face in faces:
            should_keep = True
            for kept_face in keep:
                iou = self.calculate_iou(face[:4], kept_face[:4])
                if iou > self.nms_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(face)

        return keep

    def filter_by_size(
        self,
        faces: List[Tuple[int, int, int, int, float]],
        min_size: Tuple[int, int],
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Filter faces by minimum size.

        Args:
            faces: List of (x, y, w, h, confidence) tuples
            min_size: Minimum (width, height) in pixels

        Returns:
            Filtered list of faces meeting size requirements
        """
        min_w, min_h = min_size
        return [
            face for face in faces if face[2] >= min_w and face[3] >= min_h
        ]

    def sort_by_confidence(
        self, faces: List[Tuple[int, int, int, int, float]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Sort faces by confidence descending.

        Args:
            faces: List of (x, y, w, h, confidence) tuples

        Returns:
            Sorted list with highest confidence first
        """
        return sorted(faces, key=lambda face: face[4], reverse=True)

    @staticmethod
    def calculate_iou(
        box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.

        Args:
            box1, box2: Boxes as (x, y, w, h)

        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0
