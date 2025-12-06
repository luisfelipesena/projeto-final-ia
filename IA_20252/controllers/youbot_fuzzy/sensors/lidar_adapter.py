"""Dual-LIDAR adapter providing summaries and cube candidates."""

from __future__ import annotations

import math
from typing import List, Optional

from controller import Lidar  # type: ignore[import-not-found]

import config
from data_types import LidarPoint, LidarSnapshot


class LidarAdapter:
    """Facade that manages both high and low LIDARs on the YouBot."""

    def __init__(self, robot):
        self._high = self._init_device(robot, config.LIDAR_HIGH)
        self._low = self._init_device(robot, config.LIDAR_LOW)

    @staticmethod
    def _init_device(robot, cfg: config.LidarConfig) -> Optional[Lidar]:
        try:
            device = robot.getDevice(cfg.name)
        except AttributeError:
            device = None
        if device:
            device.enable(cfg.sampling_period)
            if cfg.name == config.LIDAR_HIGH_NAME:
                try:
                    device.enablePointCloud()
                except AttributeError:
                    pass
        return device

    # ------------------------------------------------------------------
    # Device presence helpers
    def has_high(self) -> bool:
        return self._high is not None

    def has_low(self) -> bool:
        return self._low is not None

    # ------------------------------------------------------------------
    # Scan utilities
    def _min_distances(self, device: Lidar, cfg: config.LidarConfig) -> List[float]:
        res = device.getHorizontalResolution()
        layers = device.getNumberOfLayers()
        image = device.getRangeImage()
        distances: List[float] = []
        for i in range(res):
            best = cfg.max_range
            for layer in range(layers):
                value = image[layer * res + i]
                if value <= 0 or math.isinf(value):
                    continue
                if value < cfg.min_range or value > cfg.max_range:
                    continue
                best = min(best, value)
            distances.append(best)
        return distances

    def _polar_to_points(self, distances: List[float], cfg: config.LidarConfig) -> List[LidarPoint]:
        if not distances:
            return []
        fov = cfg.field_of_view
        res = len(distances)
        start_angle = -fov / 2.0
        step = fov / res
        points: List[LidarPoint] = []
        for idx, dist in enumerate(distances):
            if dist <= cfg.min_range or dist > cfg.max_range:
                continue
            angle = start_angle + idx * step
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            points.append(LidarPoint(x=x, y=y, distance=dist))
        return points

    # ------------------------------------------------------------------
    def summarize(self) -> LidarSnapshot:
        if not self._high:
            return LidarSnapshot()
        distances = self._min_distances(self._high, config.LIDAR_HIGH)

        # DEBUG: Print distance at key indices to empirically find front direction
        if config.ENABLE_LOGGING and len(distances) >= 360:
            debug_indices = [0, 45, 90, 135, 180, 225, 270, 315]
            debug_vals = [f"{i}:{distances[i]:.2f}" for i in debug_indices if i < len(distances)]
            print(f"LIDAR_RAW: {' | '.join(debug_vals)}")

        front = self._segment_average(distances, config.LIDAR_HIGH.front_sector)
        left = self._segment_average(distances, config.LIDAR_HIGH.left_sector)
        right = self._segment_average(distances, config.LIDAR_HIGH.right_sector)
        density = self._obstacle_density(distances, config.LIDAR_HIGH)
        return LidarSnapshot(front_distance=front, left_distance=left, right_distance=right, obstacle_density=density)

    def cube_candidates(self, separation: float = 0.12) -> List[LidarPoint]:
        if not self._low:
            return []
        low_distances = self._min_distances(self._low, config.LIDAR_LOW)
        high_distances: List[float] = []
        if self._high:
            high_distances = self._min_distances(self._high, config.LIDAR_HIGH)
        scale = len(high_distances) / len(low_distances) if high_distances else 1.0
        candidates: List[LidarPoint] = []
        for idx, dist_low in enumerate(low_distances):
            if not (config.CUBE_DETECTION_MIN_DISTANCE <= dist_low <= config.CUBE_DETECTION_MAX_DISTANCE):
                continue
            if high_distances:
                high_idx = int(idx * scale) % len(high_distances)
                dist_high = high_distances[high_idx]
                if dist_high - dist_low < config.CUBE_HEIGHT_DIFFERENCE_THRESHOLD:
                    continue
            angle = self._index_to_angle(idx, len(low_distances), config.LIDAR_LOW.field_of_view)
            x = dist_low * math.cos(angle)
            y = dist_low * math.sin(angle)
            candidates.append(LidarPoint(x=x, y=y, distance=dist_low))
        return candidates

    def navigation_points(self) -> List[LidarPoint]:
        if not self._high:
            return []
        distances = self._min_distances(self._high, config.LIDAR_HIGH)
        return self._polar_to_points(distances, config.LIDAR_HIGH)

    # ------------------------------------------------------------------
    @staticmethod
    def _segment_average(distances: List[float], segment: tuple[int, int]) -> float:
        start, end = segment
        length = len(distances)
        if length == 0:
            return config.DEFAULT_DISTANCE

        # Handle wrap-around for sectors crossing 0° (e.g., front_sector=(330, 390))
        if end > length:
            indices = list(range(start, length)) + list(range(0, end - length))
        else:
            indices = list(range(max(0, start), min(length, end)))

        slice_values = [
            distances[i]
            for i in indices
            if 0 <= i < length and distances[i] > 0.0 and not math.isinf(distances[i])
        ]
        if not slice_values:
            return config.DEFAULT_DISTANCE
        return min(sum(slice_values) / len(slice_values), min(slice_values + [config.DEFAULT_DISTANCE]))

    @staticmethod
    def _obstacle_density(distances: List[float], cfg: config.LidarConfig) -> float:
        if not distances:
            return 0.0
        near = sum(1 for d in distances if cfg.min_range < d < config.DANGER_ZONE)
        return near / len(distances)

    @staticmethod
    def _point_near(point: LidarPoint, cloud: List[LidarPoint], threshold: float) -> bool:
        threshold_sq = threshold * threshold
        for other in cloud:
            dx = point.x - other.x
            dy = point.y - other.y
            if dx * dx + dy * dy <= threshold_sq:
                return True
        return False

    @staticmethod
    def _index_to_angle(idx: int, resolution: int, fov: float) -> float:
        return (idx / resolution) * fov - fov / 2.0
