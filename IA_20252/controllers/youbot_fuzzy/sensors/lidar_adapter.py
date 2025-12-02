"""Wrapper around Webots Lidar providing higher-level summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from controller import Lidar

from .. import config
from ..types import LidarSnapshot


@dataclass
class RawScan:
    distances: list[float]
    horizontal_resolution: int


class LidarAdapter:
    """Small facade responsible for enabling and polling the lidar sensor."""

    def __init__(self, robot):
        self._lidar: Optional[Lidar] = None
        try:
            self._lidar = robot.getDevice(config.LIDAR.name)
        except AttributeError:
            self._lidar = None
        if self._lidar:
            self._lidar.enable(config.LIDAR.sampling_period)
            self._lidar.enablePointCloud(False)

    def has_sensor(self) -> bool:
        return self._lidar is not None

    def get_raw_scan(self) -> Optional[RawScan]:
        if not self._lidar:
            return None
        image = self._lidar.getRangeImage()
        return RawScan(list(image), self._lidar.getHorizontalResolution())

    def summarize(self) -> LidarSnapshot:
        """Return distances aggregated into semantic sectors (front/left/right)."""
        if not self._lidar:
            return LidarSnapshot()
        raw = self.get_raw_scan()
        if not raw:
            return LidarSnapshot()

        front = self._segment_average(raw.distances, config.LIDAR.front_sector)
        left = self._segment_average(raw.distances, config.LIDAR.left_sector)
        right = self._segment_average(raw.distances, config.LIDAR.right_sector)

        density = self._obstacle_density(raw.distances)

        return LidarSnapshot(front_distance=front, left_distance=left, right_distance=right, obstacle_density=density)

    @staticmethod
    def _segment_average(distances: list[float], segment: tuple[int, int]) -> float:
        start, end = segment
        length = len(distances)
        if length == 0:
            return config.DEFAULT_DISTANCE

        if end <= start:
            end = start
        start = max(0, min(length - 1, start))
        end = max(start + 1, min(length, end))

        slice_values = [d for d in distances[start:end] if not math.isinf(d) and d > 0.0]
        if not slice_values:
            return config.DEFAULT_DISTANCE
        return min(sum(slice_values) / len(slice_values), config.DEFAULT_DISTANCE)

    @staticmethod
    def _obstacle_density(distances: list[float]) -> float:
        if not distances:
            return 0.0
        near = sum(1 for d in distances if 0.0 < d < config.OBSTACLE_THRESHOLD)
        return near / len(distances)
