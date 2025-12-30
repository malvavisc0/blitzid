"""Performance tracking and metrics for face detection."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""

    faces: List[Tuple[int, int, int, int, float]]
    processing_time: float
    image_size: Tuple[int, int]
    backend: str
    num_faces: int
    cache_hit: bool = False


class MetricsTracker:
    """Tracks performance metrics with percentile tracking."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.total_detections = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.total_faces = 0
        self.processing_times: List[float] = []

    def record_detection(
        self, processing_time: float, num_faces: int, cache_hit: bool = False
    ) -> None:
        """
        Record a detection event.

        Args:
            processing_time: Time taken for detection in seconds
            num_faces: Number of faces detected
            cache_hit: Whether result was from cache
        """
        self.total_detections += 1
        self.total_time += processing_time
        self.total_faces += num_faces

        # Only record non-cached times for percentile calculations
        if not cache_hit:
            self.processing_times.append(processing_time)

        if cache_hit:
            self.cache_hits += 1

    def get_stats(self) -> Dict[str, float]:
        """
        Get aggregated statistics including percentiles.

        Returns:
            Dictionary with statistics:
                - total_detections: Total number of detections
                - total_time: Total processing time
                - avg_time: Average time per detection
                - cache_hit_rate: Percentage of cache hits
                - avg_faces: Average faces per detection
                - p50_time: Median processing time (50th percentile)
                - p95_time: 95th percentile processing time
                - p99_time: 99th percentile processing time
                - min_time: Minimum processing time
                - max_time: Maximum processing time
        """
        avg_time = (
            self.total_time / self.total_detections
            if self.total_detections > 0
            else 0.0
        )
        cache_hit_rate = (
            (self.cache_hits / self.total_detections * 100)
            if self.total_detections > 0
            else 0.0
        )
        avg_faces = (
            self.total_faces / self.total_detections
            if self.total_detections > 0
            else 0.0
        )

        # Calculate percentiles from non-cached processing times
        stats = {
            "total_detections": self.total_detections,
            "total_time": self.total_time,
            "avg_time": avg_time,
            "cache_hit_rate": cache_hit_rate,
            "avg_faces": avg_faces,
        }

        if self.processing_times:
            times_array = np.array(self.processing_times)
            stats.update(
                {
                    "p50_time": float(np.percentile(times_array, 50)),
                    "p95_time": float(np.percentile(times_array, 95)),
                    "p99_time": float(np.percentile(times_array, 99)),
                    "min_time": float(np.min(times_array)),
                    "max_time": float(np.max(times_array)),
                }
            )
        else:
            stats.update(
                {
                    "p50_time": 0.0,
                    "p95_time": 0.0,
                    "p99_time": 0.0,
                    "min_time": 0.0,
                    "max_time": 0.0,
                }
            )

        return stats

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_detections = 0
        self.total_time = 0.0
        self.cache_hits = 0
        self.total_faces = 0
        self.processing_times.clear()
