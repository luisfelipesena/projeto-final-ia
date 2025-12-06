"""Cube detection pipeline (YOLO + AdaBoost + heuristics)."""

from __future__ import annotations

import math
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

import config
from sensors.camera_stream import CameraFrame
from data_types import CubeHypothesis, Detection
from perception.adaboost_classifier import AdaBoostColorClassifier
from perception.color_classifier import ColorClassifier, ColorDetection
from perception.utils import frame_to_bgr
from perception.yolo_detector import YoloDetector


class CubeDetector:
    def __init__(
        self,
        classifier: Optional[ColorClassifier] = None,
        yolo: Optional[YoloDetector] = None,
        adaboost: Optional[AdaBoostColorClassifier] = None,
    ):
        self._adaboost = adaboost or AdaBoostColorClassifier()
        self._classifier = classifier or ColorClassifier(adaboost=self._adaboost)
        self._yolo = yolo or YoloDetector()

    def detect(self, frame: Optional[CameraFrame]) -> CubeHypothesis:
        if not frame:
            return CubeHypothesis()
        bgr = frame_to_bgr(frame)
        detection = None
        if bgr is not None and self._yolo.available():
            detection = self._select_best(self._yolo.detect(bgr))
        if detection and bgr is not None:
            return self._from_detection(detection, bgr, frame)

        # fallback para heurística HSV completa
        color_detection: ColorDetection = self._classifier.classify_frame(frame)
        if not color_detection.color:
            return CubeHypothesis()
        width = frame.width or 1
        bearing = self._bearing_from_centroid(color_detection.centroid_x, width, frame.camera.getFov())
        alignment = ((color_detection.centroid_x / width) - 0.5) * config.CAMERA_ALIGNMENT_SCALE * 2
        distance = self._estimate_distance(color_detection.coverage)
        return CubeHypothesis(
            color=color_detection.color.upper(),
            bearing=bearing,
            distance=distance,
            alignment=alignment,
            confidence=color_detection.coverage,
        )

    # ------------------------------------------------------------------
    def _from_detection(self, detection: Detection, bgr, frame: CameraFrame) -> CubeHypothesis:
        x1, y1, x2, y2 = detection.bbox
        height, width = bgr.shape[:2]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))
        crop = bgr[y1:y2, x1:x2] if bgr is not None else None
        label = detection.label.upper() if detection.label else None
        if crop is not None:
            label = self._classifier.classify_patch(crop) or label
        cx = (x1 + x2) / 2.0
        width = frame.width or 1
        bearing = self._bearing_from_centroid(cx, width, frame.camera.getFov())
        alignment = ((cx / width) - 0.5) * config.CAMERA_ALIGNMENT_SCALE * 2
        coverage = ((x2 - x1) * (y2 - y1)) / (frame.width * frame.height) if frame.width and frame.height else 0.0
        distance = self._estimate_distance(max(coverage, 0.0001))
        return CubeHypothesis(
            color=label.upper() if label else None,
            bearing=bearing,
            distance=distance,
            alignment=alignment,
            confidence=detection.confidence,
        )

    @staticmethod
    def _select_best(detections: list[Detection]) -> Optional[Detection]:
        if not detections:
            return None
        return max(detections, key=lambda det: det.confidence)

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
