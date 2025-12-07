"""Fuzzy planner translating sensor inputs into motion commands."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Tuple

import config
from data_types import CubeHypothesis, LidarSnapshot, MotionCommand

if TYPE_CHECKING:
    from world.model import WorldModel


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
        self._log_counter = 0  # For debug logging
        self._same_position_counter = 0  # Anti-stuck mechanism
        self._last_position = (0.0, 0.0)

    def plan(
        self,
        obstacles: LidarSnapshot,
        cube: CubeHypothesis,
        load_state: bool,
        patch_vector: tuple[float, float] = (0.0, 0.0),
        goal_vector: tuple[float, float] = (0.0, 0.0),
        cube_candidates: int = 0,
        world: "WorldModel | None" = None,
    ) -> MotionCommand:
        cube_conf = cube.confidence or 0.0

        vx = 0.0
        vy = 0.0
        omega = 0.0
        lift_request = None
        gripper_request = None

        # Use KNOWN MAP for collision avoidance (more reliable than LIDAR)
        map_front = 5.0
        map_left = 5.0
        map_right = 5.0
        safe_dir = (0.0, 0.0)

        if world is not None:
            # Get wall distances in robot frame (accounts for heading)
            walls_robot = world.distance_to_walls_robot_frame()
            obs_dist, obs_dx, obs_dy = world.distance_to_nearest_obstacle()

            # Start with wall distances in robot frame
            map_front = walls_robot["front"]
            map_left = walls_robot["left"]
            map_right = walls_robot["right"]

            # Refine with nearest obstacle position in robot frame
            # obs_dx > 0 means obstacle is in front, obs_dy > 0 means left
            if obs_dist < 2.0:  # Only consider nearby obstacles
                if obs_dx > 0.1 and abs(obs_dy) < 0.4:  # Roughly in front
                    map_front = min(map_front, obs_dist)
                elif obs_dy > 0.3:  # On left
                    map_left = min(map_left, obs_dist)
                elif obs_dy < -0.3:  # On right
                    map_right = min(map_right, obs_dist)

            safe_dir = world.get_safe_direction()

        # Smart fusion: trust map when LIDAR seems buggy (reading body interference)
        # If LIDAR reads very short distance but map shows open space, LIDAR is wrong
        if map_front > 1.0 and obstacles.front_distance < 0.20:
            front_dist = map_front * 0.7  # Trust map
        else:
            front_dist = min(obstacles.front_distance, map_front)

        if map_left > 0.5 and obstacles.left_distance < 0.20:
            left_dist = map_left * 0.7  # Trust map - LIDAR left often reads body
        else:
            left_dist = min(obstacles.left_distance, map_left)

        if map_right > 0.5 and obstacles.right_distance < 0.20:
            right_dist = map_right * 0.7  # Trust map
        else:
            right_dist = min(obstacles.right_distance, map_right)

        front_terms = self._front_membership(front_dist)
        left_terms = self._clearance_membership(left_dist)
        right_terms = self._clearance_membership(right_dist)

        # Anti-stuck detection: if robot hasn't moved for 100+ cycles, force movement
        if world is not None:
            current_pos = (world.pose[0], world.pose[1])
            dx = abs(current_pos[0] - self._last_position[0])
            dy = abs(current_pos[1] - self._last_position[1])
            if dx < 0.03 and dy < 0.03:
                self._same_position_counter += 1
            else:
                self._same_position_counter = 0
                self._last_position = current_pos

            # If stuck for 100+ cycles, force escape maneuver
            if self._same_position_counter > 100:
                escape_vx = 0.12
                escape_omega = 0.5 if (self._same_position_counter // 50) % 2 == 0 else -0.5
                self._same_position_counter = 0  # Reset after attempt
                return MotionCommand(vx=escape_vx, vy=0.0, omega=escape_omega)

        # Danger zone detection using KNOWN MAP
        in_danger = front_dist < config.DANGER_ZONE

        # --- Rule group 1: ESCAPE when in danger (HIGHEST PRIORITY) ----------------------
        if in_danger:
            self._escape_counter += 1

            # Use KNOWN MAP safe direction if available
            if safe_dir != (0.0, 0.0):
                # Move in safe direction calculated from known obstacles/walls
                safe_x, safe_y = safe_dir
                heading = math.atan2(safe_y, safe_x)
                omega = heading * 0.6
                vx = 0.06 if safe_x > 0 else 0.0
            elif front_dist < 0.18:
                # Emergency short reverse only
                vx = -0.05
                omega = 0.5 if left_dist > right_dist else -0.5
            else:
                vx = 0.04
                if left_dist > right_dist + 0.1:
                    omega = 0.7
                elif right_dist > left_dist + 0.1:
                    omega = -0.7
                else:
                    omega = 0.6 if self._escape_counter % 20 < 10 else -0.6

            return MotionCommand(vx=vx, vy=vy, omega=omega)

        # Reset escape counter when safe
        self._escape_counter = 0

        # --- Rule group 2: Approach cube when safe -----------------------------------------
        if not load_state and cube.color and cube_conf >= 0.3:
            alignment = cube.alignment or 0.0
            bearing = cube.bearing or 0.0
            # Use weighted average instead of min() for less conservative approach
            approach = ((front_terms["far"] + front_terms["medium"]) * 0.5) * cube_conf

            # Only approach if path is clear
            if front_terms["very_close"] < 0.25 and front_dist > 0.35:
                vx += min(0.06, approach * self._approach_gain)
                omega -= bearing * approach * 1.2

            # Only trigger arm when confident and centered
            if cube_conf >= 0.5 and 0.35 <= (cube.distance or front_dist) <= 0.55 and abs(alignment) < 0.04:
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

            # Check if path to patch is clear using known map
            path_clear = True
            if world is not None and patch_dist > 0.1:
                pose = world.pose
                target_x = pose[0] + px
                target_y = pose[1] + py
                path_clear = world.is_path_clear(target_x, target_y)

            # Determine clearance levels
            front_very_clear = front_dist > 1.2  # Lots of space ahead
            front_clear = front_dist > 0.8  # Reasonable space ahead
            front_tight = front_dist > 0.5  # Some space but need caution

            if front_very_clear:
                # Wide open - move fast towards goal
                if patch_dist > 0.1:
                    patch_heading = math.atan2(py, px)
                    omega = patch_heading * 0.4  # Gentle turn towards goal
                    vx = 0.12  # Full speed ahead (capped)
                else:
                    vx = 0.10
                    omega = 0.05 * (left_dist - right_dist)  # Slight wander

            elif front_clear and path_clear:
                # Good clearance - move forward with heading correction
                if patch_dist > 0.1:
                    patch_heading = math.atan2(py, px)
                    omega = patch_heading * 0.4
                    vx = 0.10
                else:
                    vx = 0.08
                    omega = 0.1 * (left_dist - right_dist)

            elif front_tight:
                # Tight space - slow down but keep moving
                vx = 0.06
                # Turn towards more open side
                if left_dist > right_dist + 0.2:
                    omega = 0.5
                elif right_dist > left_dist + 0.2:
                    omega = -0.5
                else:
                    omega = 0.3  # Default slight left turn

            else:
                # Front blocked - rotate to find clear path
                vx = 0.02  # Tiny forward to prevent stall
                if left_dist > right_dist:
                    omega = 0.7
                else:
                    omega = -0.7

        # --- Rule group 5: Slow down near obstacles ----------------------------------------
        slow_factor = 1.0 - front_terms["close"] * 0.5
        vx *= slow_factor

        # --- Rule group 6: Slow down if uncertain cubes nearby -----------------------------
        if cube_candidates > 0 and cube_conf < 0.3:
            vx *= 0.5

        # Clamp velocities and scale down when front space is limited
        vx = max(0.0, min(0.12, vx))
        if front_dist < 0.8:
            vx = min(vx, max(0.0, (front_dist - 0.4) * 0.2))
        vy = 0.0  # Avoid lateral strafing; prefer forward + yaw
        omega = max(-0.9, min(0.9, omega))

        # Debug: Log decision reasoning
        if config.ENABLE_LOGGING:
            self._log_counter += 1
            if self._log_counter % 200 == 0:
                mode = "ESCAPE" if in_danger else "EXPLORE" if cube_conf < 0.01 else "CUBE"
                print(f"FUZZY[{mode}]: f={front_dist:.2f} l={left_dist:.2f} r={right_dist:.2f} → vx={vx:.2f} ω={omega:.2f}")

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
