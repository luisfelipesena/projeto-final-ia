"""Fuzzy planner translating sensor inputs into motion commands."""

from __future__ import annotations

from .. import config
from ..types import CubeHypothesis, LidarSnapshot, MotionCommand


def triangular(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


class FuzzyPlanner:
    """Implements the fuzzification → inference → defuzzification pipeline."""

    def __init__(self):
        self._approach_gain = 0.12
        self._turn_gain = 0.6

    def plan(self, obstacles: LidarSnapshot, cube: CubeHypothesis, load_state: bool) -> MotionCommand:
        front_terms = self._front_membership(obstacles.front_distance)
        left_terms = self._clearance_membership(obstacles.left_distance)
        right_terms = self._clearance_membership(obstacles.right_distance)
        cube_conf = cube.confidence or 0.0

        vx = 0.0
        vy = 0.0
        omega = 0.0
        lift_request = None
        gripper_request = None

        # --- Rule group 1: Avoid obstacles -------------------------------------------------
        avoidance = max(front_terms["very_close"], obstacles.obstacle_density)
        if avoidance > 0:
            vx -= avoidance * 0.08
            # steer towards the side with greater clearance
            omega += self._turn_gain * ((right_terms["clear"] - left_terms["clear"]) * avoidance)

        # --- Rule group 2: Approach cube when safe -----------------------------------------
        if not load_state and cube_conf > 0.01:
            alignment = cube.alignment or 0.0
            bearing = cube.bearing or 0.0
            approach = min(front_terms["far"], cube_conf)
            vx += approach * self._approach_gain
            omega -= bearing * approach
            vy = -alignment * 0.5

            if front_terms["close"] > 0.5 and abs(alignment) < 0.05:
                lift_request = "FLOOR"
                gripper_request = "GRIP"

        # --- Rule group 3: Carrying cube -> head towards drop zone -------------------------
        if load_state:
            vx = max(vx, 0.06)
            omega *= 0.5
            if front_terms["very_close"] > 0.6:
                lift_request = "PLATE"
                gripper_request = "RELEASE"

        return MotionCommand(vx=vx, vy=vy, omega=omega, lift_request=lift_request, gripper_request=gripper_request)

    @staticmethod
    def _front_membership(distance: float) -> dict[str, float]:
        return {
            "very_close": triangular(distance, 0.0, 0.15, 0.35),
            "close": triangular(distance, 0.2, 0.5, 0.85),
            "medium": triangular(distance, 0.7, 1.2, 1.8),
            "far": trapezoidal(distance, 1.5, 1.8, config.DEFAULT_DISTANCE, config.DEFAULT_DISTANCE),
        }

    @staticmethod
    def _clearance_membership(distance: float) -> dict[str, float]:
        return {
            "blocked": trapezoidal(distance, 0.0, 0.0, 0.4, 0.8),
            "partial": triangular(distance, 0.5, 1.0, 1.5),
            "clear": trapezoidal(distance, 1.2, 1.5, config.DEFAULT_DISTANCE, config.DEFAULT_DISTANCE),
        }
