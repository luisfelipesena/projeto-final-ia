"""Fuzzy planner translating sensor inputs into motion commands."""

from __future__ import annotations

import math
import config
from data_types import CubeHypothesis, LidarSnapshot, MotionCommand


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
        self._turn_gain = 0.8
        self._escape_counter = 0  # Track stuck state

    def plan(
        self,
        obstacles: LidarSnapshot,
        cube: CubeHypothesis,
        load_state: bool,
        patch_vector: tuple[float, float] = (0.0, 0.0),
        goal_vector: tuple[float, float] = (0.0, 0.0),
        cube_candidates: int = 0,
    ) -> MotionCommand:
        front_terms = self._front_membership(obstacles.front_distance)
        left_terms = self._clearance_membership(obstacles.left_distance)
        right_terms = self._clearance_membership(obstacles.right_distance)
        cube_conf = cube.confidence or 0.0

        vx = 0.0
        vy = 0.0
        omega = 0.0
        lift_request = None
        gripper_request = None

        # Danger zone detection
        in_danger = obstacles.front_distance < config.DANGER_ZONE
        left_blocked = obstacles.left_distance < 0.3
        right_blocked = obstacles.right_distance < 0.3

        # --- Rule group 1: ESCAPE when in danger (HIGHEST PRIORITY) ----------------------
        if in_danger:
            self._escape_counter += 1

            # Strong backward movement
            vx = -0.15

            # Determine escape direction
            if left_blocked and right_blocked:
                # Both sides blocked - rotate more aggressively
                omega = 0.8 if self._escape_counter % 2 == 0 else -0.8
            elif left_blocked:
                omega = -0.6  # Turn right (away from left wall)
            elif right_blocked:
                omega = 0.6   # Turn left (away from right wall)
            else:
                # Front blocked, sides clear - pick clearer side
                if obstacles.left_distance > obstacles.right_distance:
                    omega = 0.5  # Turn left
                else:
                    omega = -0.5  # Turn right

            # Add lateral escape if possible
            if obstacles.left_distance > obstacles.right_distance + 0.2:
                vy = 0.1  # Strafe left
            elif obstacles.right_distance > obstacles.left_distance + 0.2:
                vy = -0.1  # Strafe right

            return MotionCommand(vx=vx, vy=vy, omega=omega)

        # Reset escape counter when safe
        self._escape_counter = 0

        # --- Rule group 2: Approach cube when safe -----------------------------------------
        if not load_state and cube_conf > 0.01:
            alignment = cube.alignment or 0.0
            bearing = cube.bearing or 0.0
            approach = min(front_terms["far"], front_terms["medium"], cube_conf)

            # Only approach if path is clear
            if front_terms["very_close"] < 0.3:
                vx += approach * self._approach_gain
                omega -= bearing * approach * 1.5
                vy = -alignment * 0.5

            if front_terms["close"] > 0.5 and abs(alignment) < 0.05:
                lift_request = "FLOOR"
                gripper_request = "GRIP"

        # --- Rule group 3: Carrying cube -> head towards drop zone -------------------------
        elif load_state:
            gx, gy = goal_vector
            goal_dist = math.sqrt(gx * gx + gy * gy)
            if goal_dist > 0.1:
                goal_heading = math.atan2(gy, gx)
                omega += goal_heading * 0.5
                # Only move forward if path clear
                if front_terms["very_close"] < 0.3:
                    vx += min(0.1, goal_dist * 0.1)
            if front_terms["very_close"] > 0.6 and goal_dist < 0.5:
                lift_request = "PLATE"
                gripper_request = "RELEASE"

        # --- Rule group 4: Explore patches when no cube detected ---------------------------
        elif cube_conf < 0.01 and not load_state:
            px, py = patch_vector
            patch_dist = math.sqrt(px * px + py * py)

            # Explore even if front partially blocked (threshold increased)
            if front_terms["very_close"] < 0.5:
                if patch_dist > 0.1:
                    patch_heading = math.atan2(py, px)
                    omega += patch_heading * 0.6  # Increased turn gain
                    vx += min(0.12, patch_dist * 0.1)  # Increased forward speed
                else:
                    # At patch location - explore forward with gentle wander
                    vx += 0.10  # Constant forward exploration
                    omega += 0.1 * (obstacles.left_distance - obstacles.right_distance)
            else:
                # Front blocked - rotate to find clear path
                if obstacles.left_distance > obstacles.right_distance:
                    omega = 0.5   # Turn left
                else:
                    omega = -0.5  # Turn right

        # --- Rule group 5: Slow down near obstacles ----------------------------------------
        slow_factor = 1.0 - front_terms["close"] * 0.5
        vx *= slow_factor

        # --- Rule group 6: Slow down if uncertain cubes nearby -----------------------------
        if cube_candidates > 0 and cube_conf < 0.3:
            vx *= 0.5

        # Clamp velocities
        vx = max(-0.2, min(0.15, vx))
        vy = max(-0.1, min(0.1, vy))
        omega = max(-1.0, min(1.0, omega))

        return MotionCommand(vx=vx, vy=vy, omega=omega, lift_request=lift_request, gripper_request=gripper_request)

    @staticmethod
    def _front_membership(distance: float) -> dict[str, float]:
        return {
            "very_close": triangular(distance, 0.0, 0.15, 0.40),
            "close": triangular(distance, 0.25, 0.55, 0.90),
            "medium": triangular(distance, 0.70, 1.20, 1.80),
            "far": trapezoidal(distance, 1.50, 1.80, config.DEFAULT_DISTANCE, config.DEFAULT_DISTANCE),
        }

    @staticmethod
    def _clearance_membership(distance: float) -> dict[str, float]:
        return {
            "blocked": trapezoidal(distance, 0.0, 0.0, 0.3, 0.6),
            "partial": triangular(distance, 0.4, 0.8, 1.3),
            "clear": trapezoidal(distance, 1.0, 1.4, config.DEFAULT_DISTANCE, config.DEFAULT_DISTANCE),
        }
