"""Color classification utilities combining HSV heuristics and AdaBoost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

import config
from sensors.camera_stream import CameraFrame
from perception.adaboost_classifier import AdaBoostColorClassifier
from perception.utils import frame_to_bgr


@dataclass
class ColorDetection:
    color: Optional[str]
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    coverage: float = 0.0


class ColorClassifier:
    """Hybrid classifier que combina AdaBoost e heurística HSV."""

    def __init__(self, hsv_ranges: Optional[dict] = None, adaboost: Optional[AdaBoostColorClassifier] = None):
        self._ranges = hsv_ranges or config.HSV_RANGES
        self._adaboost = adaboost

    def classify_frame(self, frame: CameraFrame) -> ColorDetection:
        bgr = frame_to_bgr(frame)
        if bgr is None:
            return ColorDetection(None)

        height, width = bgr.shape[:2]
        if np is None:
            # Fallback to simple heuristic
            label = self.classify_patch(bgr)
            return ColorDetection(color=label) if label else ColorDetection(None)

        # Scan for ALL colors directly using HSV masks - find the one with highest coverage
        best_color = None
        best_coverage = 0.0
        best_cx = 0.0
        best_cy = 0.0

        for color in ["red", "green", "blue"]:
            mask = self._mask_for_label(bgr, color)
            if not isinstance(mask, np.ndarray):
                continue
            coverage = float(np.count_nonzero(mask)) / (width * height)
            if coverage > best_coverage and coverage > config.HSV_COVERAGE_THRESHOLD:
                coordinates = np.column_stack(np.where(mask > 0))
                if coordinates.size > 0:
                    cy, cx = coordinates.mean(axis=0)
                    best_color = color.upper()
                    best_coverage = coverage
                    best_cx = float(cx)
                    best_cy = float(cy)

        if best_color:
            return ColorDetection(color=best_color, centroid_x=best_cx, centroid_y=best_cy, coverage=best_coverage)

        # Debug: log coverage values when nothing detected (ALWAYS log to understand)
        if config.ENABLE_LOGGING:
            coverages = {}
            for color in ["red", "green", "blue"]:
                mask = self._mask_for_label(bgr, color)
                if isinstance(mask, np.ndarray):
                    coverages[color] = float(np.count_nonzero(mask)) / (width * height)
            # Also log mean RGB values to understand camera output
            mean_bgr = bgr.reshape(-1, 3).mean(axis=0)
            print(f"CAM_DEBUG: mean_BGR=({mean_bgr[0]:.0f},{mean_bgr[1]:.0f},{mean_bgr[2]:.0f}) cov_r={coverages.get('red',0):.4f} cov_g={coverages.get('green',0):.4f} cov_b={coverages.get('blue',0):.4f} thr={config.HSV_COVERAGE_THRESHOLD}")

        return ColorDetection(None)

    def classify_patch(self, patch) -> Optional[str]:
        if patch is None:
            return None
        if self._adaboost and self._adaboost.available():
            label = self._adaboost.predict(patch)
            if label:
                return label.upper()
        return self._heuristic_color(patch)

    # ------------------------------------------------------------------
    def _heuristic_color(self, patch) -> Optional[str]:
        if np is None or patch is None:
            return None
        mean_bgr = patch.reshape(-1, 3).mean(axis=0)
        b, g, r = mean_bgr
        if r > g and r > b:
            return "RED"
        if g > r and g > b:
            return "GREEN"
        if b > r and b > g:
            return "BLUE"
        return None

    def _mask_for_label(self, image, label: str):
        if np is None:
            return 0
        hsv = self._to_hsv(image)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        ranges = self._ranges.get(label.lower(), [])
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.float32)
            upper_np = np.array(upper, dtype=np.float32)
            part = np.all((hsv >= lower_np) & (hsv <= upper_np), axis=2)
            mask[part] = 255
        return mask

    @staticmethod
    def _to_hsv(image):
        if np is None:
            return image
        # Normalizar para 0-1 antes de converter
        bgr = image.astype(np.float32) / 255.0
        r = bgr[:, :, 2]
        g = bgr[:, :, 1]
        b = bgr[:, :, 0]
        c_max = np.maximum(np.maximum(r, g), b)
        c_min = np.minimum(np.minimum(r, g), b)
        delta = c_max - c_min

        hue = np.zeros_like(c_max)
        mask = delta != 0
        mask_r = (c_max == r) & mask
        mask_g = (c_max == g) & mask
        mask_b = (c_max == b) & mask
        hue[mask_r] = ((g - b)[mask_r] / delta[mask_r]) % 6
        hue[mask_g] = ((b - r)[mask_g] / delta[mask_g]) + 2
        hue[mask_b] = ((r - g)[mask_b] / delta[mask_b]) + 4
        hue *= 30  # Output H in 0-180 range (OpenCV convention) to match config.HSV_RANGES

        saturation = np.zeros_like(c_max)
        saturation[c_max != 0] = delta[c_max != 0] / c_max[c_max != 0]

        value = c_max
        hsv = np.stack([hue, saturation * 100, value * 100], axis=2)
        return hsv
