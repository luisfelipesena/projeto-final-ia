"""World-state aggregation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, List

import config
from mapping.grid_map import GridMap
from data_types import CubeHypothesis, LidarPoint, LidarSnapshot

# Arena boundaries (from RectangleArena centered at -0.79, 0)
ARENA_CENTER = (-0.79, 0.0)
ARENA_HALF_X = 3.5  # 7m / 2
ARENA_HALF_Y = 2.0  # 4m / 2
ARENA_MIN_X = ARENA_CENTER[0] - ARENA_HALF_X  # -4.29
ARENA_MAX_X = ARENA_CENTER[0] + ARENA_HALF_X  # 2.71
ARENA_MIN_Y = ARENA_CENTER[1] - ARENA_HALF_Y  # -2.0
ARENA_MAX_Y = ARENA_CENTER[1] + ARENA_HALF_Y  # 2.0

# Wooden boxes as obstacles (30cm cubes)
WOODEN_BOX_SIZE = 0.30
WOODEN_BOX_POSITIONS = [
    (0.60, 0.00),   # A
    (1.96, -1.24),  # B
    (1.95, 1.25),   # C
    (-2.28, 1.50),  # D
    (-1.02, 0.75),  # E
    (-1.02, -0.74), # F
    (-2.27, -1.51), # G
]

# PlasticFruitBox deposit boxes - ALSO obstacles to avoid!
DEPOSIT_BOX_SIZE = 0.50  # Approximate size of PlasticFruitBox
DEPOSIT_BOX_POSITIONS = {
    "GREEN": (0.48, 1.58),
    "BLUE": (0.48, -1.62),
    "RED": (2.31, 0.01),
}


@dataclass
class GoalState:
    color: str
    position: Tuple[float, float]


class WorldModel:
    """Tracks static references and latest perception snapshots."""

    def __init__(self):
        self._goals: Dict[str, GoalState] = {
            color: GoalState(color, pose) for color, pose in config.BOX_TARGETS.items()
        }
        self._latest_cube = CubeHypothesis()
        self._latest_obstacles = LidarSnapshot()
        self._goal_index = 0
        self.grid = GridMap()
        self._pose = (0.0, 0.0, 0.0)

    def update_obstacles(self, snapshot: LidarSnapshot) -> None:
        self._latest_obstacles = snapshot

    def update_cube(self, cube: CubeHypothesis) -> None:
        self._latest_cube = cube

    def get_goal(self, color: str) -> GoalState:
        return self._goals[color]

    def current_goal_color(self) -> str:
        return config.GOAL_SEQUENCE[self._goal_index % len(config.GOAL_SEQUENCE)]

    def advance_goal(self) -> None:
        self._goal_index += 1

    def update_pose(self, pose: Tuple[float, float, float]) -> None:
        self._pose = pose
        self.grid.mark_pose(pose)

    def integrate_lidar(self, nav_points: Iterable[LidarPoint], cube_points: Iterable[LidarPoint]) -> None:
        pose = self._pose
        self.grid.mark_obstacles(pose, tuple(nav_points))
        self.grid.mark_cubes(pose, tuple(cube_points))

    def next_patch_vector(self) -> Tuple[float, float]:
        """Get vector to nearest unvisited patch in world frame."""
        target = self.grid.next_unvisited_patch(self._pose)
        if target is None:
            return (0.0, 0.0)
        tx, ty = self.grid.cell_to_world(target)
        return (tx - self._pose[0], ty - self._pose[1])

    def goal_vector(self) -> Tuple[float, float]:
        goal = self.get_goal(self.current_goal_color())
        return (goal.position[0] - self._pose[0], goal.position[1] - self._pose[1])

    @property
    def current_cube(self) -> CubeHypothesis:
        return self._latest_cube

    @property
    def obstacles(self) -> LidarSnapshot:
        return self._latest_obstacles

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self._pose

    def distance_to_walls(self) -> dict:
        """Calculate distance to arena walls from current pose (absolute, not rotated)."""
        x, y, theta = self._pose
        return {
            "front_wall": ARENA_MAX_X - x,  # Distance to front wall (positive X)
            "back_wall": x - ARENA_MIN_X,   # Distance to back wall (negative X)
            "left_wall": ARENA_MAX_Y - y,   # Distance to left wall (positive Y)
            "right_wall": y - ARENA_MIN_Y,  # Distance to right wall (negative Y)
        }

    def distance_to_walls_robot_frame(self) -> dict:
        """Calculate distance to arena walls in robot frame (accounts for heading)."""
        x, y, theta = self._pose
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # Calculate distances to each wall in world frame
        dist_pos_x = ARENA_MAX_X - x  # +X wall
        dist_neg_x = x - ARENA_MIN_X  # -X wall
        dist_pos_y = ARENA_MAX_Y - y  # +Y wall
        dist_neg_y = y - ARENA_MIN_Y  # -Y wall

        # Project distances onto robot's forward and lateral directions
        # Forward direction: (cos_t, sin_t)
        # Left direction: (-sin_t, cos_t)

        # Simplified ray-casting for front/left/right distances
        front_dist = float('inf')
        left_dist = float('inf')
        right_dist = float('inf')

        # Front: check which wall robot is facing
        if cos_t > 0.1:  # Facing +X
            front_dist = min(front_dist, dist_pos_x / cos_t)
        if cos_t < -0.1:  # Facing -X
            front_dist = min(front_dist, dist_neg_x / (-cos_t))
        if sin_t > 0.1:  # Facing +Y
            front_dist = min(front_dist, dist_pos_y / sin_t)
        if sin_t < -0.1:  # Facing -Y
            front_dist = min(front_dist, dist_neg_y / (-sin_t))

        # Left (perpendicular to front, +90 deg)
        left_cos = -sin_t
        left_sin = cos_t
        if left_cos > 0.1:
            left_dist = min(left_dist, dist_pos_x / left_cos)
        if left_cos < -0.1:
            left_dist = min(left_dist, dist_neg_x / (-left_cos))
        if left_sin > 0.1:
            left_dist = min(left_dist, dist_pos_y / left_sin)
        if left_sin < -0.1:
            left_dist = min(left_dist, dist_neg_y / (-left_sin))

        # Right (-90 deg from front)
        right_cos = sin_t
        right_sin = -cos_t
        if right_cos > 0.1:
            right_dist = min(right_dist, dist_pos_x / right_cos)
        if right_cos < -0.1:
            right_dist = min(right_dist, dist_neg_x / (-right_cos))
        if right_sin > 0.1:
            right_dist = min(right_dist, dist_pos_y / right_sin)
        if right_sin < -0.1:
            right_dist = min(right_dist, dist_neg_y / (-right_sin))

        return {
            "front": min(front_dist, 5.0),
            "left": min(left_dist, 5.0),
            "right": min(right_dist, 5.0),
        }

    def distance_to_nearest_obstacle(self) -> Tuple[float, float, float]:
        """Find nearest obstacle (wooden box OR deposit box). Returns (distance, dx_robot, dy_robot) in robot frame."""
        x, y, theta = self._pose
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        min_dist = float('inf')
        nearest_dx_robot, nearest_dy_robot = 0.0, 0.0

        # Check wooden boxes
        for bx, by in WOODEN_BOX_POSITIONS:
            dx_world = bx - x
            dy_world = by - y
            dx_robot = dx_world * cos_t + dy_world * sin_t
            dy_robot = -dx_world * sin_t + dy_world * cos_t
            dist = math.sqrt(dx_world*dx_world + dy_world*dy_world) - WOODEN_BOX_SIZE/2
            if dist < min_dist:
                min_dist = dist
                nearest_dx_robot = dx_robot
                nearest_dy_robot = dy_robot

        # Check deposit boxes (PlasticFruitBox) - except when heading to that goal
        current_goal_pos = DEPOSIT_BOX_POSITIONS.get(self.current_goal_color(), None)
        for color, (bx, by) in DEPOSIT_BOX_POSITIONS.items():
            # Skip the goal we're heading to when carrying a cube
            if current_goal_pos and (bx, by) == current_goal_pos:
                continue
            dx_world = bx - x
            dy_world = by - y
            dx_robot = dx_world * cos_t + dy_world * sin_t
            dy_robot = -dx_world * sin_t + dy_world * cos_t
            dist = math.sqrt(dx_world*dx_world + dy_world*dy_world) - DEPOSIT_BOX_SIZE/2
            if dist < min_dist:
                min_dist = dist
                nearest_dx_robot = dx_robot
                nearest_dy_robot = dy_robot

        return (min_dist, nearest_dx_robot, nearest_dy_robot)

    def is_path_clear(self, target_x: float, target_y: float, margin: float = 0.4) -> bool:
        """Check if straight path to target is clear of known obstacles."""
        x, y, _ = self._pose

        # Check if target is within arena
        if not (ARENA_MIN_X + margin < target_x < ARENA_MAX_X - margin):
            return False
        if not (ARENA_MIN_Y + margin < target_y < ARENA_MAX_Y - margin):
            return False

        # Check for wooden box collisions along path
        for bx, by in WOODEN_BOX_POSITIONS:
            box_radius = WOODEN_BOX_SIZE/2 + margin
            if self._point_near_line(x, y, target_x, target_y, bx, by, box_radius):
                return False

        # Check for deposit box collisions along path
        for color, (bx, by) in DEPOSIT_BOX_POSITIONS.items():
            box_radius = DEPOSIT_BOX_SIZE/2 + margin
            if self._point_near_line(x, y, target_x, target_y, bx, by, box_radius):
                return False

        return True

    @staticmethod
    def _point_near_line(x1, y1, x2, y2, px, py, threshold) -> bool:
        """Check if point (px,py) is within threshold of line segment (x1,y1)-(x2,y2)."""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx*dx + dy*dy
        if length_sq < 0.001:
            return math.sqrt((px-x1)**2 + (py-y1)**2) < threshold

        t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy) / length_sq))
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
        dist = math.sqrt((px-nearest_x)**2 + (py-nearest_y)**2)
        return dist < threshold

    def get_safe_direction(self) -> Tuple[float, float]:
        """Get direction vector pointing away from obstacles/walls in ROBOT FRAME.

        Returns (vx, vy) where positive vx = forward, positive vy = left.
        """
        walls_robot = self.distance_to_walls_robot_frame()
        obs_dist, obs_dx, obs_dy = self.distance_to_nearest_obstacle()

        # Compute repulsion in robot frame
        repel_x, repel_y = 0.0, 0.0

        # Wall repulsion in robot frame (stronger when closer)
        if walls_robot["front"] < 0.5:
            repel_x -= 1.0 / max(walls_robot["front"], 0.1)  # Push backward
        if walls_robot["left"] < 0.5:
            repel_y -= 1.0 / max(walls_robot["left"], 0.1)   # Push right
        if walls_robot["right"] < 0.5:
            repel_y += 1.0 / max(walls_robot["right"], 0.1)  # Push left

        # Obstacle repulsion (already in robot frame)
        if obs_dist < 0.6:
            # obs_dx > 0 means obstacle in front, so repel backward
            # obs_dy > 0 means obstacle on left, so repel right
            if abs(obs_dx) > 0.05:
                repel_x -= obs_dx / max(obs_dist, 0.1)
            if abs(obs_dy) > 0.05:
                repel_y -= obs_dy / max(obs_dist, 0.1)

        # Normalize
        mag = math.sqrt(repel_x**2 + repel_y**2)
        if mag > 0.1:
            return (repel_x/mag, repel_y/mag)
        return (0.0, 0.0)
