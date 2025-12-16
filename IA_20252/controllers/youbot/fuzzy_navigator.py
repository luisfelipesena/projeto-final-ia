"""
Fuzzy Logic Navigator for YouBot obstacle avoidance and navigation.
Implements membership functions and fuzzy rules for velocity control.
"""

import math
from constants import KNOWN_OBSTACLES
from routes import wrap_angle


class FuzzyNavigator:
    """Fuzzy logic controller for navigation with obstacle avoidance."""

    def __init__(self, max_speed=0.2):
        """Initialize fuzzy navigator.
        
        Args:
            max_speed: maximum linear velocity in m/s
        """
        self.max_speed = max_speed
        self.safety_margin = 0.4

    @staticmethod
    def _mu_close(x, threshold=0.45):
        """Membership function: distance is close."""
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_very_close(x, threshold=0.25):
        """Membership function: distance is very close."""
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_far(x, start=0.4, end=1.5):
        """Membership function: distance is far."""
        if x <= start:
            return 0.0
        if x >= end:
            return 1.0
        return (x - start) / (end - start)

    @staticmethod
    def _mu_small_angle(angle_rad):
        """Membership function: angle is small (near aligned)."""
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(5):
            return 1.0
        if a > math.radians(25):
            return 0.0
        return (math.radians(25) - a) / math.radians(20)

    @staticmethod
    def _mu_medium_angle(angle_rad):
        """Membership function: angle is medium."""
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(10) or a > math.radians(60):
            return 0.0
        if a < math.radians(35):
            return (a - math.radians(10)) / math.radians(25)
        return (math.radians(60) - a) / math.radians(25)

    @staticmethod
    def _mu_big_angle(angle_rad):
        """Membership function: angle is large."""
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(40):
            return 0.0
        if a > math.radians(90):
            return 1.0
        return (a - math.radians(40)) / math.radians(50)

    def check_known_obstacles(self, pose):
        """Check proximity to known static obstacles.
        
        Args:
            pose: (x, y, yaw) robot pose
            
        Returns:
            tuple: (front_dist, left_dist, right_dist) to obstacles
        """
        x, y, yaw = pose
        min_dist_left = 2.0
        min_dist_right = 2.0
        min_dist_front = 2.0

        for ox, oy, radius in KNOWN_OBSTACLES:
            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy) - radius
            obs_angle = wrap_angle(math.atan2(dy, dx) - yaw)

            if abs(obs_angle) < math.radians(30):
                min_dist_front = min(min_dist_front, dist)
            elif obs_angle > 0 and obs_angle < math.radians(120):
                min_dist_left = min(min_dist_left, dist)
            elif obs_angle < 0 and obs_angle > math.radians(-120):
                min_dist_right = min(min_dist_right, dist)

        return min_dist_front, min_dist_left, min_dist_right

    def check_rear_clearance(self, omega, obs_front, obs_left, obs_right,
                             rear_sensors=None):
        """Check if rotation will cause rear collision.
        
        Args:
            omega: desired rotation (positive = left, negative = right)
            obs_front, obs_left, obs_right: front LIDAR distances
            rear_sensors: dict with 'rear', 'rear_left', 'rear_right' in meters
            
        Returns:
            tuple: (is_safe, blocking_direction)
        """
        MIN_REAR_CLEARANCE = 0.35
        MIN_SIDE_CLEARANCE = 0.40

        if rear_sensors:
            rear = rear_sensors.get("rear", 2.0)
            rear_left = rear_sensors.get("rear_left", 2.0)
            rear_right = rear_sensors.get("rear_right", 2.0)

            # Direct rear blocked - no reverse
            if rear < MIN_REAR_CLEARANCE:
                return False, "rear_blocked"

            # Left rotation (omega > 0) -> rear goes right
            if omega > 0.05:
                if rear_right < MIN_SIDE_CLEARANCE:
                    return False, "rear_right"
            # Right rotation (omega < 0) -> rear goes left
            elif omega < -0.05:
                if rear_left < MIN_SIDE_CLEARANCE:
                    return False, "rear_left"
        else:
            # Fallback: use lateral LIDAR as estimate
            ROBOT_HALF_LENGTH = 0.30
            SAFETY_MARGIN = 0.20
            MIN_CLEARANCE = ROBOT_HALF_LENGTH + SAFETY_MARGIN

            if omega > 0.05 and obs_right < MIN_CLEARANCE:
                return False, "rear_right"
            elif omega < -0.05 and obs_left < MIN_CLEARANCE:
                return False, "rear_left"

        # Front too close for any rotation
        if obs_front < 0.35:
            return False, "front_blocked"

        return True, None

    def compute(self, has_target, target_distance, target_angle,
                obs_front, obs_left, obs_right, pose=None):
        """Compute velocity commands using fuzzy logic.
        
        Args:
            has_target: whether there's a navigation target
            target_distance: distance to target in meters
            target_angle: angle to target in radians
            obs_front, obs_left, obs_right: obstacle distances
            pose: (x, y, yaw) robot pose for known obstacle check
            
        Returns:
            tuple: (vx, vy, omega) velocity commands
        """
        if not has_target:
            return 0.0, 0.0, 0.25

        if pose is not None:
            known_front, known_left, known_right = self.check_known_obstacles(pose)
            obs_front = min(obs_front, known_front)
            obs_left = min(obs_left, known_left)
            obs_right = min(obs_right, known_right)

        # Fuzzification
        mu_front_close = self._mu_close(obs_front)
        mu_front_very_close = self._mu_very_close(obs_front)
        mu_left_close = self._mu_close(obs_left)
        mu_right_close = self._mu_close(obs_right)

        mu_target_far = self._mu_far(target_distance)
        mu_target_close = self._mu_close(target_distance, threshold=0.3)
        mu_angle_small = self._mu_small_angle(target_angle)
        mu_angle_medium = self._mu_medium_angle(target_angle)
        mu_angle_big = self._mu_big_angle(target_angle)

        # Rule 1: Very close obstacle -> STOP and strafe (NO excessive rotation)
        if mu_front_very_close > 0.5:
            vx = -0.04
            vy = 0.10 * (1 if obs_left < obs_right else -1)
            omega = 0.0  # DON'T ROTATE - only strafe to avoid spinning
            return vx, vy, omega

        # Rule 2: Close obstacle -> reduce speed
        speed_reduction = 1.0 - (0.7 * mu_front_close)

        # Rule 3: Forward velocity - PRIORITIZE FORWARD MOVEMENT
        vx = 0.20 * max(0.4, mu_target_far) * speed_reduction * (1.0 - 0.25 * mu_angle_big)

        # Rule 4: Near target and aligned -> final approach
        if mu_target_close > 0.3 and mu_angle_small > 0.5:
            vx = 0.10 * speed_reduction

        # Rule 4b: When aligned (small angle), ensure forward motion
        if mu_angle_small > 0.6:
            vx = max(vx, 0.12 * speed_reduction)

        # Rule 5: Strafe for obstacle avoidance
        vy = 0.10 * (mu_right_close - mu_left_close)

        # Rule 6: Rotation to align - PROPORTIONAL CONTROL
        angle_norm = target_angle / math.radians(90)
        angle_norm = max(-1.0, min(1.0, angle_norm))

        omega_fuzzy = (
            0.35 * mu_angle_big +
            0.18 * mu_angle_medium +
            0.05 * (1.0 - mu_angle_small)
        )
        omega = omega_fuzzy * angle_norm

        # Rule 7: Lateral obstacle adjustment
        omega += 0.2 * (mu_right_close - mu_left_close)

        # Limit velocities
        vx = max(-self.max_speed, min(self.max_speed, vx))
        vy = max(-self.max_speed * 0.8, min(self.max_speed * 0.8, vy))
        omega = max(-0.6, min(0.6, omega))

        return vx, vy, omega
