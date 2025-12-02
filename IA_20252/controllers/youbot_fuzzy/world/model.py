"""World-state aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .. import config
from ..mapping.grid_map import GridMap
from ..types import CubeHypothesis, LidarPoint, LidarSnapshot


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
        target = self.grid.next_unvisited_patch()
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
