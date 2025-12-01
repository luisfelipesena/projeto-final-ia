"""
NavigationService - High-level navigation control.
"""
from typing import Optional, Tuple, Dict, Any
import math


class NavigationService:
    """High-level navigation service combining movement and vision."""

    def __init__(self, movement, vision, robot, camera, time_step: int, lidar=None):
        self.movement = movement
        self.vision = vision
        self.robot = robot
        self.camera = camera
        self.time_step = time_step
        self.lidar = lidar

    def search_for_cube(self, omega: float = 0.3) -> bool:
        """Rotate to search for cubes."""
        self.movement.rotate(omega)
        return self.robot.step(self.time_step) != -1

    def approach_target(self, max_speed: float = 0.15) -> Tuple[bool, str]:
        """Approach currently tracked target.

        Returns:
            (success, status) - status is 'approaching', 'aligned', 'lost', or 'reached'
        """
        target = self.vision.get_target()
        if not target:
            self.movement.stop()
            return False, 'lost'

        # Check if close enough
        if target.distance < 0.12:
            self.movement.stop()
            return True, 'reached'

        # Calculate velocities based on target
        angle_rad = math.radians(target.angle)

        # Proportional control
        angular = -0.01 * target.angle  # Negative for correction
        angular = max(-0.5, min(0.5, angular))

        # Linear velocity decreases as we get closer
        linear = min(max_speed, target.distance * 0.5)
        if abs(target.angle) > 20:
            linear *= 0.5  # Slow down if misaligned

        self.movement.move(linear, 0, angular)

        if abs(target.angle) < 5:
            return True, 'aligned'
        return True, 'approaching'

    def avoid_obstacle(self, lidar_data: Dict[str, Any]) -> bool:
        """Avoid detected obstacle based on LIDAR data."""
        if not lidar_data:
            return False

        sectors = lidar_data.get('sectors', {})
        front = sectors.get('front', {}).get('min', float('inf'))
        left = sectors.get('left', {}).get('min', float('inf'))
        right = sectors.get('right', {}).get('min', float('inf'))

        if front < 0.3:
            # Back up and turn
            self.movement.move(-0.1, 0, 0.5 if left < right else -0.5)
            return True
        elif left < 0.25:
            self.movement.move(0.05, 0, -0.3)
            return True
        elif right < 0.25:
            self.movement.move(0.05, 0, 0.3)
            return True

        return False

    def is_path_clear(self, lidar_data: Dict[str, Any], threshold: float = 0.3) -> bool:
        """Check if forward path is clear."""
        if not lidar_data:
            return True

        sectors = lidar_data.get('sectors', {})
        front = sectors.get('front', {}).get('min', float('inf'))
        front_left = sectors.get('front_left', {}).get('min', float('inf'))
        front_right = sectors.get('front_right', {}).get('min', float('inf'))

        return min(front, front_left, front_right) > threshold
