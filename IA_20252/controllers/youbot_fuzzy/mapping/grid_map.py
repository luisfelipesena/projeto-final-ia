"""Occupancy grid and patch tracking for the YouBot arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

import config
from data_types import LidarPoint

STATE_UNKNOWN = 0
STATE_FREE = 1
STATE_OBSTACLE = 2
STATE_CUBE = 3


@dataclass
class GridCell:
    x: int
    y: int


class GridMap:
    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.size_x, self.size_y = config.ARENA_SIZE
        # Arena is centered at (-0.79, 0) in world coordinates
        # Origin = bottom-left corner of arena
        arena_center_x = -0.79
        arena_center_y = 0.0
        self.origin = (arena_center_x - self.size_x / 2.0, arena_center_y - self.size_y / 2.0)
        # origin = (-4.29, -2.0) for 7x4m arena centered at (-0.79, 0)
        width = int(self.size_x / resolution) + 2
        height = int(self.size_y / resolution) + 2
        if np is not None:
            self._grid = np.zeros((width, height), dtype=np.uint8)
        else:
            self._grid = [[STATE_UNKNOWN for _ in range(height)] for _ in range(width)]
        self.visited: set[Tuple[int, int]] = set()
        self.cube_counts: Dict[Tuple[int, int], int] = {}

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        gx = max(0, min(self._grid_width - 1, gx))
        gy = max(0, min(self._grid_height - 1, gy))
        return gx, gy

    def cell_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        gx, gy = cell
        x = self.origin[0] + (gx + 0.5) * self.resolution
        y = self.origin[1] + (gy + 0.5) * self.resolution
        return x, y

    @property
    def _grid_width(self) -> int:
        return len(self._grid) if np is None else self._grid.shape[0]

    @property
    def _grid_height(self) -> int:
        return len(self._grid[0]) if np is None else self._grid.shape[1]

    def _set_cell(self, cell: Tuple[int, int], state: int) -> None:
        gx, gy = cell
        if np is not None:
            self._grid[gx, gy] = state
        else:
            self._grid[gx][gy] = state

    def mark_pose(self, pose: Tuple[float, float, float]) -> None:
        cell = self.world_to_cell(pose[0], pose[1])
        self.visited.add(cell)
        self._set_cell(cell, STATE_FREE)

    def mark_obstacles(self, pose: Tuple[float, float, float], points: Tuple[LidarPoint, ...]) -> None:
        cos_t, sin_t = math_cos_sin(pose[2])
        px, py = pose[0], pose[1]
        for point in points:
            gx = px + point.x * cos_t - point.y * sin_t
            gy = py + point.x * sin_t + point.y * cos_t
            cell = self.world_to_cell(gx, gy)
            self._set_cell(cell, STATE_OBSTACLE)

    def mark_cubes(self, pose: Tuple[float, float, float], cubes: Tuple[LidarPoint, ...]) -> None:
        cos_t, sin_t = math_cos_sin(pose[2])
        px, py = pose[0], pose[1]
        for point in cubes:
            gx = px + point.x * cos_t - point.y * sin_t
            gy = py + point.x * sin_t + point.y * cos_t
            cell = self.world_to_cell(gx, gy)
            self._set_cell(cell, STATE_CUBE)
            self.cube_counts[cell] = self.cube_counts.get(cell, 0) + 1

    def next_unvisited_patch(self, current_pose: Optional[Tuple[float, float, float]] = None) -> Optional[Tuple[int, int]]:
        """Return nearest unvisited cell to current position, or first if no pose given."""
        if current_pose is None:
            # Fallback to old behavior
            for x in range(self._grid_width):
                for y in range(self._grid_height):
                    if (x, y) not in self.visited:
                        return (x, y)
            return None

        # Find nearest unvisited cell
        current_cell = self.world_to_cell(current_pose[0], current_pose[1])
        best_cell = None
        best_dist_sq = float('inf')

        # Arena bounds in grid coordinates (filter cells outside valid area)
        # Arena: X from -4.29 to 2.71, Y from -2.0 to 2.0
        min_x_cell = self.world_to_cell(-4.0, 0)[0]  # Leave margin from walls
        max_x_cell = self.world_to_cell(2.5, 0)[0]
        min_y_cell = self.world_to_cell(0, -1.8)[1]
        max_y_cell = self.world_to_cell(0, 1.8)[1]

        for x in range(self._grid_width):
            for y in range(self._grid_height):
                if (x, y) in self.visited:
                    continue
                # Filter cells outside arena bounds
                if not (min_x_cell <= x <= max_x_cell and min_y_cell <= y <= max_y_cell):
                    continue

                # Calculate distance squared to current position
                dx = x - current_cell[0]
                dy = y - current_cell[1]
                dist_sq = dx * dx + dy * dy

                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_cell = (x, y)

        return best_cell


def math_cos_sin(theta: float) -> Tuple[float, float]:
    import math

    return math.cos(theta), math.sin(theta)
