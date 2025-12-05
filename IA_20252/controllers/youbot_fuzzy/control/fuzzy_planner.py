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

    def plan(
        self,
        obstacles: LidarSnapshot,
        cube: CubeHypothesis,
        load_state: bool,
        patch_vector: tuple[float, float] = (0.0, 0.0),
        goal_vector: tuple[float, float] = (0.0, 0.0),
        cube_candidates: int = 0,
    ) -> MotionCommand:
        import math

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
        elif load_state:
            gx, gy = goal_vector
            goal_dist = math.sqrt(gx * gx + gy * gy)
            if goal_dist > 0.1:
                # Navigate towards goal box
                goal_heading = math.atan2(gy, gx)
                omega += goal_heading * 0.5
                vx += min(0.1, goal_dist * 0.1)
            if front_terms["very_close"] > 0.6 and goal_dist < 0.5:
                lift_request = "PLATE"
                gripper_request = "RELEASE"

        # --- Rule group 4: Explore patches when no cube detected ---------------------------
        elif cube_conf < 0.01 and not load_state:
            px, py = patch_vector
            patch_dist = math.sqrt(px * px + py * py)
            if patch_dist > 0.1:
                patch_heading = math.atan2(py, px)
                omega += patch_heading * 0.3
                vx += min(0.08, patch_dist * 0.05)
            else:
                # No patch to explore, wander
                vx += 0.05 * front_terms["far"]
                omega += 0.1 * (right_terms["clear"] - left_terms["clear"])

        # --- Rule group 5: Slow down if uncertain cubes nearby -----------------------------
        if cube_candidates > 0 and cube_conf < 0.3:
            vx *= 0.5  # Slow down to get better detection

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
