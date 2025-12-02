"""Cube detection logic tying camera and classifiers together."""

from __future__ import annotations

import math
from typing import Optional

from ..types import CubeHypothesis
from .. import config
from ..sensors.camera_stream import CameraFrame
from .color_classifier import ColorClassifier, ColorDetection


class CubeDetector:
    def __init__(self, classifier: Optional[ColorClassifier] = None):
        self._classifier = classifier or ColorClassifier()

    def detect(self, frame: Optional[CameraFrame]) -> CubeHypothesis:
        if not frame:
            return CubeHypothesis()

        detection: ColorDetection = self._classifier.classify_frame(frame)
        if not detection.color:
            return CubeHypothesis()

        width = frame.width or 1
        bearing = self._bearing_from_centroid(detection.centroid_x, width, frame.camera.getFov())
        alignment = ((detection.centroid_x / width) - 0.5) * config.CAMERA_ALIGNMENT_SCALE * 2

        distance = self._estimate_distance(detection.coverage)

        return CubeHypothesis(
            color=detection.color.upper(),
            bearing=bearing,
            distance=distance,
            alignment=alignment,
            confidence=detection.coverage,
        )

    @staticmethod
    def _bearing_from_centroid(cx: float, width: int, fov: float) -> float:
        normalized = (cx / width) - 0.5
        return normalized * fov

    @staticmethod
    def _estimate_distance(coverage: float) -> float:
        if coverage <= 0:
            return config.DEFAULT_DISTANCE
        approx = 0.2 / math.sqrt(coverage)
        return min(max(approx, 0.1), config.DEFAULT_DISTANCE)
