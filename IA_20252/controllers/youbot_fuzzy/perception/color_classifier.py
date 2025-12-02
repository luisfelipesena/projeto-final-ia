"""Color classification utilities based on HSV ranges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from controller import Camera

from ..sensors.camera_stream import CameraFrame
from .. import config


@dataclass
class ColorDetection:
    color: Optional[str]
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    coverage: float = 0.0


class ColorClassifier:
    """Map raw pixel statistics to known cube colors."""

    def __init__(self, hsv_ranges: Optional[dict] = None):
        self._ranges = hsv_ranges or config.HSV_RANGES

    def classify_frame(self, frame: CameraFrame) -> ColorDetection:
        if not frame or not frame.image:
            return ColorDetection(None)

        totals = {color: 0 for color in self._ranges}
        centroid_x = {color: 0.0 for color in self._ranges}
        centroid_y = {color: 0.0 for color in self._ranges}

        width, height = frame.width, frame.height
        if width == 0 or height == 0:
            return ColorDetection(None)

        for y in range(height):
            for x in range(width):
                r = frame.camera.imageGetRed(frame.image, width, x, y)
                g = frame.camera.imageGetGreen(frame.image, width, x, y)
                b = frame.camera.imageGetBlue(frame.image, width, x, y)
                h, s, v = self._rgb_to_hsv(r, g, b)
                for color, ranges in self._ranges.items():
                    if self._matches(h, s, v, ranges):
                        totals[color] += 1
                        centroid_x[color] += x
                        centroid_y[color] += y
                        break

        if not any(totals.values()):
            return ColorDetection(None)

        color = max(totals, key=totals.get)
        count = totals[color]
        coverage = count / (width * height)
        if coverage <= 0.005:
            return ColorDetection(None)

        cx = centroid_x[color] / count
        cy = centroid_y[color] / count

        return ColorDetection(color=color, centroid_x=cx, centroid_y=cy, coverage=coverage)

    @staticmethod
    def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
        r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
        c_max = max(r_f, g_f, b_f)
        c_min = min(r_f, g_f, b_f)
        delta = c_max - c_min

        if delta == 0:
            h = 0
        elif c_max == r_f:
            h = (60 * ((g_f - b_f) / delta) + 360) % 360
        elif c_max == g_f:
            h = 60 * ((b_f - r_f) / delta + 2)
        else:
            h = 60 * ((r_f - g_f) / delta + 4)

        s = 0 if c_max == 0 else delta / c_max
        v = c_max
        return h, s * 100, v * 100

    @staticmethod
    def _matches(h: float, s: float, v: float, ranges) -> bool:
        for lower, upper in ranges:
            if ColorClassifier._within(h, lower[0], upper[0]) and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return True
        return False

    @staticmethod
    def _within(value: float, lower: float, upper: float) -> bool:
        if lower <= upper:
            return lower <= value <= upper
        # wrap around 360
        return value >= lower or value <= upper
