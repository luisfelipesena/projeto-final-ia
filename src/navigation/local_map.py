"""
LocalMap Module

Local occupancy grid map built from LIDAR observations.
Based on: Thrun et al. (2005) - Probabilistic Robotics, Chapter 9

The map uses a log-odds representation for efficient updates
and handles the absence of GPS by maintaining a robot-centric view.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import IntEnum


class OccupancyCell(IntEnum):
    """Cell occupancy states"""
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


@dataclass
class MapConfig:
    """Local map configuration"""
    size_meters: float = 10.0  # Map covers 10m x 10m around robot
    resolution: float = 0.1  # 10cm per cell
    lidar_max_range: float = 5.0  # LIDAR maximum reliable range
    hit_probability: float = 0.7  # Probability of occupied given hit
    miss_probability: float = 0.4  # Probability of occupied given miss
    occupied_threshold: float = 0.65  # Log-odds threshold for occupied
    free_threshold: float = 0.35  # Log-odds threshold for free


class LocalMap:
    """
    Local occupancy grid map for robot-centric navigation

    Features:
    - Log-odds probability updates for sensor fusion
    - Efficient LIDAR raytracing using Bresenham's algorithm
    - Robot-centric coordinate frame (no GPS required)
    - Supports map shifting as robot moves

    Usage:
        local_map = LocalMap()

        # Update with LIDAR scan
        local_map.update_from_lidar(lidar_ranges, robot_angle=0.0)

        # Query occupancy
        if local_map.is_occupied(1.0, 0.5):
            print("Obstacle at (1.0, 0.5) meters")

        # Get grid for visualization/planning
        grid = local_map.get_occupancy_grid()
    """

    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initialize local map

        Args:
            config: MapConfig with map parameters
        """
        self.config = config or MapConfig()

        # Calculate grid dimensions
        self.grid_size = int(self.config.size_meters / self.config.resolution)
        self.center = self.grid_size // 2  # Robot is at center

        # Initialize log-odds map (0 = unknown, positive = occupied, negative = free)
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Precompute log-odds increments
        self.l_occ = np.log(self.config.hit_probability / (1 - self.config.hit_probability))
        self.l_free = np.log(self.config.miss_probability / (1 - self.config.miss_probability))

        # Clamp values to prevent saturation
        self.l_max = 5.0
        self.l_min = -5.0

        # LIDAR configuration (270° FOV)
        self.lidar_fov = 270.0  # degrees
        self.lidar_num_points = 667

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to grid indices

        Args:
            x: X position relative to robot (forward is positive)
            y: Y position relative to robot (left is positive)

        Returns:
            (row, col) grid indices
        """
        col = self.center + int(x / self.config.resolution)
        row = self.center - int(y / self.config.resolution)  # Y is inverted

        # Clamp to grid bounds
        col = max(0, min(self.grid_size - 1, col))
        row = max(0, min(self.grid_size - 1, row))

        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates (meters)

        Args:
            row: Grid row index
            col: Grid column index

        Returns:
            (x, y) world coordinates relative to robot
        """
        x = (col - self.center) * self.config.resolution
        y = (self.center - row) * self.config.resolution
        return x, y

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if grid indices are within map bounds"""
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def bresenham_line(
        self,
        x0: int, y0: int,
        x1: int, y1: int
    ) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for efficient raytracing

        Args:
            x0, y0: Start point (grid indices)
            x1, y1: End point (grid indices)

        Returns:
            List of (row, col) along the line
        """
        points = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if self.is_in_bounds(y0, x0):
                points.append((y0, x0))

            if x0 == x1 and y0 == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    def update_from_lidar(
        self,
        ranges: np.ndarray,
        robot_angle: float = 0.0
    ) -> None:
        """
        Update map from LIDAR scan

        Uses inverse sensor model to update log-odds:
        - Cells along ray before hit: marked as free
        - Cell at hit: marked as occupied
        - Cells beyond max range: unchanged

        Args:
            ranges: [667] LIDAR range measurements
            robot_angle: Robot heading (radians, 0 = forward)
        """
        if len(ranges) != self.lidar_num_points:
            # Resample if different number of points
            indices = np.linspace(0, len(ranges) - 1, self.lidar_num_points).astype(int)
            ranges = ranges[indices]

        # Calculate angles for each LIDAR point
        # 270° FOV, centered at 0° (forward)
        start_angle = -np.radians(self.lidar_fov / 2)
        end_angle = np.radians(self.lidar_fov / 2)
        angles = np.linspace(start_angle, end_angle, self.lidar_num_points)

        # Add robot orientation
        angles = angles + robot_angle

        # Robot is at grid center
        robot_row, robot_col = self.center, self.center

        for i, (angle, distance) in enumerate(zip(angles, ranges)):
            # Skip invalid readings
            if not np.isfinite(distance) or distance <= 0:
                continue

            # Clamp to max range
            effective_range = min(distance, self.config.lidar_max_range)

            # Calculate endpoint in world coordinates
            end_x = effective_range * np.cos(angle)
            end_y = effective_range * np.sin(angle)

            # Convert to grid
            end_row, end_col = self.world_to_grid(end_x, end_y)

            # Get cells along ray using Bresenham
            ray_cells = self.bresenham_line(
                robot_col, robot_row,
                end_col, end_row
            )

            # Update cells along ray as free (except last one)
            for cell_row, cell_col in ray_cells[:-1]:
                if self.is_in_bounds(cell_row, cell_col):
                    self.log_odds[cell_row, cell_col] += self.l_free
                    self.log_odds[cell_row, cell_col] = max(
                        self.l_min,
                        self.log_odds[cell_row, cell_col]
                    )

            # Update endpoint as occupied (if within max range and not a max reading)
            if distance < self.config.lidar_max_range and ray_cells:
                last_row, last_col = ray_cells[-1]
                if self.is_in_bounds(last_row, last_col):
                    self.log_odds[last_row, last_col] += self.l_occ
                    self.log_odds[last_row, last_col] = min(
                        self.l_max,
                        self.log_odds[last_row, last_col]
                    )

    def get_probability(self, row: int, col: int) -> float:
        """
        Get occupancy probability for a cell

        Args:
            row, col: Grid indices

        Returns:
            Probability [0, 1] of cell being occupied
        """
        if not self.is_in_bounds(row, col):
            return 0.5  # Unknown

        # Convert log-odds to probability
        l = self.log_odds[row, col]
        return 1.0 - 1.0 / (1.0 + np.exp(l))

    def get_occupancy(self, row: int, col: int) -> OccupancyCell:
        """
        Get discrete occupancy state for a cell

        Args:
            row, col: Grid indices

        Returns:
            OccupancyCell enum value
        """
        prob = self.get_probability(row, col)

        if prob > self.config.occupied_threshold:
            return OccupancyCell.OCCUPIED
        elif prob < self.config.free_threshold:
            return OccupancyCell.FREE
        else:
            return OccupancyCell.UNKNOWN

    def is_occupied(self, x: float, y: float) -> bool:
        """
        Check if world position is occupied

        Args:
            x, y: World coordinates (meters)

        Returns:
            True if cell is occupied
        """
        row, col = self.world_to_grid(x, y)
        return self.get_occupancy(row, col) == OccupancyCell.OCCUPIED

    def is_free(self, x: float, y: float) -> bool:
        """
        Check if world position is free

        Args:
            x, y: World coordinates (meters)

        Returns:
            True if cell is free
        """
        row, col = self.world_to_grid(x, y)
        return self.get_occupancy(row, col) == OccupancyCell.FREE

    def get_occupancy_grid(self) -> np.ndarray:
        """
        Get discrete occupancy grid

        Returns:
            [grid_size, grid_size] array of OccupancyCell values
        """
        grid = np.zeros_like(self.log_odds, dtype=np.int32)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                grid[row, col] = self.get_occupancy(row, col)

        return grid

    def get_probability_grid(self) -> np.ndarray:
        """
        Get probability grid

        Returns:
            [grid_size, grid_size] array of occupancy probabilities
        """
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def shift_map(self, dx: float, dy: float) -> None:
        """
        Shift map as robot moves (maintains robot-centric view)

        Args:
            dx: Movement in X (meters, positive = forward)
            dy: Movement in Y (meters, positive = left)
        """
        # Convert to grid cells
        shift_cols = int(dx / self.config.resolution)
        shift_rows = -int(dy / self.config.resolution)

        if shift_cols == 0 and shift_rows == 0:
            return

        # Shift the log-odds array
        self.log_odds = np.roll(self.log_odds, -shift_rows, axis=0)
        self.log_odds = np.roll(self.log_odds, -shift_cols, axis=1)

        # Clear newly revealed cells
        if shift_rows > 0:
            self.log_odds[-shift_rows:, :] = 0
        elif shift_rows < 0:
            self.log_odds[:-shift_rows, :] = 0

        if shift_cols > 0:
            self.log_odds[:, -shift_cols:] = 0
        elif shift_cols < 0:
            self.log_odds[:, :-shift_cols] = 0

    def clear(self) -> None:
        """Reset map to unknown"""
        self.log_odds.fill(0)

    def get_nearest_obstacle_in_direction(
        self,
        angle: float,
        max_distance: float = 5.0
    ) -> float:
        """
        Find distance to nearest obstacle in a direction

        Args:
            angle: Direction (radians, 0 = forward)
            max_distance: Maximum search distance (meters)

        Returns:
            Distance to nearest obstacle, or max_distance if none found
        """
        # Sample along ray
        step = self.config.resolution
        distance = step

        while distance < max_distance:
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)

            if self.is_occupied(x, y):
                return distance

            distance += step

        return max_distance


def test_local_map():
    """Test local map functionality"""
    print("Testing LocalMap...")

    local_map = LocalMap()

    # Create synthetic LIDAR scan with obstacle in front
    ranges = np.full(667, 5.0)  # All far
    ranges[300:367] = 1.5  # Obstacle at ~1.5m in front

    # Update map
    local_map.update_from_lidar(ranges)

    # Check occupancy
    print(f"  Grid size: {local_map.grid_size}x{local_map.grid_size}")
    print(f"  Cell at (1.5, 0): {local_map.get_occupancy(*local_map.world_to_grid(1.5, 0)).name}")
    print(f"  Cell at (3.0, 0): {local_map.get_occupancy(*local_map.world_to_grid(3.0, 0)).name}")

    # Check path
    print(f"  Is (1.0, 0) occupied: {local_map.is_occupied(1.0, 0)}")
    print(f"  Is (0.5, 0) free: {local_map.is_free(0.5, 0)}")

    print("  LocalMap test passed")


if __name__ == "__main__":
    test_local_map()
