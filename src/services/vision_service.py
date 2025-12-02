"""
VisionService - High-level vision processing.
"""
from typing import Optional, List
import numpy as np


class VisionService:
    """High-level vision service for cube detection."""

    def __init__(self, detector, time_step: int):
        self.detector = detector
        self.time_step = time_step
        self._current_target = None
        self._last_detections = []
        self._locked_color = None

    def update(self, image: np.ndarray) -> None:
        """Update vision with new camera frame."""
        if image is None:
            return

        self._last_detections = self.detector.detect(image)

        if self._last_detections:
            # Track closest detection
            self._current_target = min(self._last_detections, key=lambda d: d.distance)
        else:
            self._current_target = None

    def get_target(self):
        """Get current tracking target (respects color lock)."""
        if self._locked_color:
            target = self.get_target_by_color(self._locked_color)
            if target:
                return target
            # If locked color not found, return None (don't fall back to other colors)
            return None
        return self._current_target

    def get_detections(self) -> List:
        """Get all current detections."""
        return self._last_detections

    def has_target(self) -> bool:
        """Check if there's a valid target."""
        return self._current_target is not None

    def get_target_by_color(self, color: str):
        """Get closest detection of specific color."""
        matching = [d for d in self._last_detections if d.color == color]
        if matching:
            return min(matching, key=lambda d: d.distance)
        return None

    def lock_color(self, color: str) -> None:
        """Lock tracking to specific color."""
        self._locked_color = color

    def unlock_color(self) -> None:
        """Unlock color tracking."""
        self._locked_color = None

    def unlock(self) -> None:
        """Alias for unlock_color()."""
        self.unlock_color()
