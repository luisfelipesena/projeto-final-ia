"""World-state aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .. import config
from ..types import CubeHypothesis, LidarSnapshot


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

    @property
    def current_cube(self) -> CubeHypothesis:
        return self._latest_cube

    @property
    def obstacles(self) -> LidarSnapshot:
        return self._latest_obstacles
