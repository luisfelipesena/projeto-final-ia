"""
LIDAR data processing for obstacle detection.

Processes raw LIDAR range data into sector-based obstacle information
that can be used by navigation and neural network modules.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from utils.config import LIDAR


@dataclass
class SectorInfo:
    """Information about a LIDAR sector."""
    index: int                    # Sector index (0 = front-left)
    min_distance: float           # Minimum distance in sector
    mean_distance: float          # Mean distance in sector
    obstacle_count: int           # Number of points below threshold
    has_obstacle: bool            # True if obstacle detected
    angle_center: float           # Center angle of sector (radians)


class LidarProcessor:
    """Processes LIDAR data into sector-based obstacle information."""

    def __init__(self, num_points: int = None, num_sectors: int = None):
        """Initialize processor.

        Args:
            num_points: Number of LIDAR points (default from config)
            num_sectors: Number of sectors to divide data into
        """
        self.num_points = num_points or LIDAR.NUM_POINTS
        self.num_sectors = num_sectors or LIDAR.NUM_SECTORS
        self.points_per_sector = self.num_points // self.num_sectors

        # Calculate sector angles (assuming 360° coverage)
        self.sector_angles = np.linspace(
            -np.pi, np.pi, self.num_sectors + 1
        )[:-1] + np.pi / self.num_sectors

        # Cache for last processed data
        self._last_raw = None
        self._last_sectors = None

    def process(self, range_data: np.ndarray) -> List[SectorInfo]:
        """Process raw LIDAR ranges into sector information.

        Args:
            range_data: Array of range values from LIDAR

        Returns:
            List of SectorInfo for each sector
        """
        if range_data is None or len(range_data) == 0:
            return []

        # Convert to numpy if needed
        ranges = np.array(range_data)

        # Filter invalid readings
        valid_mask = (ranges > LIDAR.MIN_RANGE) & (ranges < LIDAR.MAX_RANGE)
        ranges_filtered = np.where(valid_mask, ranges, LIDAR.MAX_RANGE)

        sectors = []
        for i in range(self.num_sectors):
            start_idx = i * self.points_per_sector
            end_idx = start_idx + self.points_per_sector

            sector_data = ranges_filtered[start_idx:end_idx]

            # Calculate sector statistics
            min_dist = float(np.min(sector_data))
            mean_dist = float(np.mean(sector_data))

            # Count obstacles (points below threshold)
            obstacle_mask = sector_data < LIDAR.OBSTACLE_THRESHOLD
            obstacle_count = int(np.sum(obstacle_mask))

            has_obstacle = min_dist < LIDAR.OBSTACLE_THRESHOLD

            sector = SectorInfo(
                index=i,
                min_distance=min_dist,
                mean_distance=mean_dist,
                obstacle_count=obstacle_count,
                has_obstacle=has_obstacle,
                angle_center=self.sector_angles[i],
            )
            sectors.append(sector)

        self._last_raw = ranges_filtered
        self._last_sectors = sectors

        return sectors

    def get_obstacle_map(self, range_data: np.ndarray) -> np.ndarray:
        """Get binary obstacle map for each sector.

        Args:
            range_data: Raw LIDAR ranges

        Returns:
            Binary array (1 = obstacle, 0 = clear) for each sector
        """
        sectors = self.process(range_data)
        return np.array([1 if s.has_obstacle else 0 for s in sectors])

    def get_sector_distances(self, range_data: np.ndarray) -> np.ndarray:
        """Get minimum distance for each sector.

        Args:
            range_data: Raw LIDAR ranges

        Returns:
            Array of minimum distances per sector
        """
        sectors = self.process(range_data)
        return np.array([s.min_distance for s in sectors])

    def get_nearest_obstacle(self, range_data: np.ndarray) -> Tuple[float, float]:
        """Find nearest obstacle across all sectors.

        Args:
            range_data: Raw LIDAR ranges

        Returns:
            (distance, angle) of nearest obstacle
        """
        sectors = self.process(range_data)

        if not sectors:
            return (LIDAR.MAX_RANGE, 0.0)

        nearest = min(sectors, key=lambda s: s.min_distance)
        return (nearest.min_distance, nearest.angle_center)

    def is_path_clear(
        self,
        range_data: np.ndarray,
        direction: float = 0.0,
        width: float = 0.3,
    ) -> Tuple[bool, float]:
        """Check if path in given direction is clear.

        Args:
            range_data: Raw LIDAR ranges
            direction: Direction to check (radians, 0 = front)
            width: Path width to check (meters)

        Returns:
            (is_clear, min_distance) in that direction
        """
        sectors = self.process(range_data)

        if not sectors:
            return (True, LIDAR.MAX_RANGE)

        # Find sectors that overlap with the path direction
        relevant_sectors = []
        for sector in sectors:
            angle_diff = abs(sector.angle_center - direction)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            # Check if sector is within path width angle
            path_half_angle = np.arctan2(width / 2, sector.min_distance)
            if angle_diff < path_half_angle + np.pi / self.num_sectors:
                relevant_sectors.append(sector)

        if not relevant_sectors:
            return (True, LIDAR.MAX_RANGE)

        min_dist = min(s.min_distance for s in relevant_sectors)
        is_clear = min_dist > LIDAR.OBSTACLE_THRESHOLD

        return (is_clear, min_dist)

    def get_front_distance(self, range_data: np.ndarray) -> float:
        """Get distance to nearest obstacle in front.

        Args:
            range_data: Raw LIDAR ranges

        Returns:
            Distance in meters
        """
        _, distance = self.is_path_clear(range_data, direction=0.0)
        return distance

    def normalize_for_nn(self, range_data: np.ndarray) -> np.ndarray:
        """Normalize LIDAR data for neural network input.

        Args:
            range_data: Raw LIDAR ranges

        Returns:
            Normalized array (values in [0, 1])
        """
        ranges = np.array(range_data)

        # Clamp to valid range
        ranges = np.clip(ranges, LIDAR.MIN_RANGE, LIDAR.MAX_RANGE)

        # Normalize to [0, 1]
        normalized = (ranges - LIDAR.MIN_RANGE) / (LIDAR.MAX_RANGE - LIDAR.MIN_RANGE)

        return normalized.astype(np.float32)

    @staticmethod
    def webots_lidar_to_numpy(lidar_data) -> np.ndarray:
        """Convert Webots LIDAR data to numpy array.

        Args:
            lidar_data: Data from lidar.getRangeImage()

        Returns:
            Numpy array of range values
        """
        return np.array(lidar_data, dtype=np.float32)
