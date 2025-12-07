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
        self._debug_counter = 0
        self._cube_debug_counter = 0

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
        # FIXED: LIDAR scans COUNTER-CLOCKWISE in Webots
        # Index 0=BACK(-π), 90=LEFT(+π/2), 180=FRONT(0), 270=RIGHT(-π/2)
        # So angle should go from +π down to -π as index increases
        start_angle = fov / 2.0  # Start at +π (back)
        step = fov / res
        points: List[LidarPoint] = []
        for idx, dist in enumerate(distances):
            if dist <= cfg.min_range or dist > cfg.max_range:
                continue
            # Decreasing angle as idx increases (CCW scan → CW angle)
            angle = start_angle - idx * step
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            points.append(LidarPoint(x=x, y=y, distance=dist))
        return points

    # ------------------------------------------------------------------
    def summarize(self) -> LidarSnapshot:
        if not self._high:
            return LidarSnapshot()
        distances = self._min_distances(self._high, config.LIDAR_HIGH)

        # DEBUG: Print distance at key indices (rarely to avoid spam)
        self._debug_counter += 1
        if config.ENABLE_LOGGING and len(distances) >= 360 and self._debug_counter % 200 == 0:
            # Verified from readings: 180=FRONT, 90=LEFT, 270=RIGHT, 0=BACK(body)
            debug_indices = [180, 170, 190, 90, 270, 0]
            debug_vals = [f"{i}:{distances[i]:.2f}" for i in debug_indices if i < len(distances)]
            print(f"LIDAR[F=180]: {' | '.join(debug_vals)}")

        front = self._segment_average(distances, config.LIDAR_HIGH.front_sector)
        left = self._segment_average(distances, config.LIDAR_HIGH.left_sector)
        right = self._segment_average(distances, config.LIDAR_HIGH.right_sector)
        density = self._obstacle_density(distances, config.LIDAR_HIGH)
        return LidarSnapshot(front_distance=front, left_distance=left, right_distance=right, obstacle_density=density)

    def cube_candidates(self, separation: float = 0.12) -> List[LidarPoint]:
        """Detect cubes: LIDAR low sees something that LIDAR high does NOT see.

        Cube (3cm tall): low sees it, high passes over it → dist_high >> dist_low
        Obstacle (30cm+): both see it at same distance → dist_high ≈ dist_low (REJECT)
        """
        if not self._low or not self._high:
            return []  # Need BOTH lidars for reliable cube detection

        low_distances = self._min_distances(self._low, config.LIDAR_LOW)
        high_distances = self._min_distances(self._high, config.LIDAR_HIGH)

        if not high_distances:
            return []  # Can't distinguish cubes from obstacles without high lidar

        scale = len(high_distances) / len(low_distances)
        candidates: List[LidarPoint] = []
        self._cube_debug_counter += 1
        log_this_call = False  # silence candidate logs to stop spam
        sample_log = None
        min_diff = max(config.CUBE_HEIGHT_DIFFERENCE_THRESHOLD, 0.6)
        max_low_distance = min(config.CUBE_DETECTION_MAX_DISTANCE, 0.55)
        max_high_for_cube = config.LIDAR_HIGH.max_range * 0.85

        for idx, dist_low in enumerate(low_distances):
            # Only consider objects within pickup range
            if not (config.CUBE_DETECTION_MIN_DISTANCE <= dist_low <= max_low_distance):
                continue

            # Get corresponding high lidar reading
            high_idx = int(idx * scale) % len(high_distances)
            dist_high = high_distances[high_idx]

            # CRITICAL: A cube is detected ONLY when:
            # 1. Low lidar sees something close (dist_low is small)
            # 2. High lidar does NOT see it (dist_high is much larger, or max range)
            #
            # If high lidar sees something at similar distance → it's an OBSTACLE, not a cube!
            height_diff = dist_high - dist_low

            # Cube: high lidar should see MUCH farther (it passes over the 3cm cube)
            # Threshold 0.2m means high must be at least 20cm farther than low
            if height_diff < min_diff:
                # Both lidars see something at similar distance = OBSTACLE (reject)
                continue

            # Additional check: reject when high is near max range (noise) or close obstacle
            if dist_high >= max_high_for_cube:
                continue
            if dist_high < 1.3:  # If high sees something within ~1.3m, probably obstacle
                continue

            angle = self._index_to_angle(idx, len(low_distances), config.LIDAR_LOW.field_of_view)
            # Ignore back/side detections; we only pick cubes in a ±75° frontal cone
            if abs(angle) > math.radians(75):
                continue

            x = dist_low * math.cos(angle)
            y = dist_low * math.sin(angle)
            candidates.append(LidarPoint(x=x, y=y, distance=dist_low))

            if log_this_call and sample_log is None:
                sample_log = (
                    f"idx={idx} low={dist_low:.2f}m high={dist_high:.2f}m "
                    f"diff={height_diff:.2f}m angle={math.degrees(angle):.1f}deg x={x:.2f} y={y:.2f}"
                )

        # Keep only up to 5 closest unique-angle candidates to avoid spam
        candidates.sort(key=lambda p: p.distance)
        filtered: List[LidarPoint] = []
        used_angles: List[float] = []
        for cand in candidates:
            ang = round(math.degrees(math.atan2(cand.y, cand.x)))
            if any(abs(ang - a) <= 5 for a in used_angles):
                continue
            used_angles.append(ang)
            filtered.append(cand)
            if len(filtered) >= 5:
                break

        if log_this_call and sample_log:
            print(f"CUBE_CANDIDATES count={len(filtered)} sample={sample_log}")
        return filtered

    def navigation_points(self) -> List[LidarPoint]:
        if not self._high:
            return []
        distances = self._min_distances(self._high, config.LIDAR_HIGH)
        return self._polar_to_points(distances, config.LIDAR_HIGH)

    # ------------------------------------------------------------------
    @staticmethod
    def _is_in_dead_zone(idx: int) -> bool:
        """Check if index is in any of the dead zones."""
        for dead_start, dead_end in config.LIDAR_DEAD_ZONES:
            if dead_start <= idx <= dead_end:
                return True
        return False

    @staticmethod
    def _segment_average(distances: List[float], segment: tuple[int, int]) -> float:
        start, end = segment
        length = len(distances)
        if length == 0:
            return config.DEFAULT_DISTANCE

        # Handle wrap-around for sectors crossing 0° (e.g., front_sector=(350, 370))
        if end > length:
            indices = list(range(start, length)) + list(range(0, end - length))
        else:
            indices = list(range(max(0, start), min(length, end)))

        # Filter out dead zone indices (robot body blocks these)
        slice_values = [
            distances[i]
            for i in indices
            if 0 <= i < length
            and not LidarAdapter._is_in_dead_zone(i)  # Skip any dead zone
            and distances[i] > 0.0
            and not math.isinf(distances[i])
        ]
        if not slice_values:
            return config.DEFAULT_DISTANCE

        # Use MINIMUM for obstacle avoidance (safest reading)
        # Filter extreme outliers (< 0.05m is probably noise)
        valid_vals = [v for v in slice_values if v >= 0.05]
        if not valid_vals:
            return config.DEFAULT_DISTANCE

        return min(valid_vals)

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
