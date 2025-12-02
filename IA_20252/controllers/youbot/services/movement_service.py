"""
MovementService - High-level movement control.
"""
from typing import Tuple


class MovementService:
    """High-level movement service for base control."""

    def __init__(self, base, robot, time_step: int):
        self.base = base
        self.robot = robot
        self.time_step = time_step

    def move(self, vx: float, vy: float, omega: float) -> None:
        """Move base with velocity."""
        self.base.move(vx, vy, omega)

    def stop(self) -> None:
        """Stop base movement."""
        self.base.reset()

    def forward(self, speed: float = 0.2, distance_m: float = None) -> bool:
        """Move forward/backward. If distance_m provided, move that distance and stop.
        Negative distance_m moves backward."""
        if distance_m is not None:
            dt = self.time_step / 1000.0
            actual_speed = speed if distance_m >= 0 else -speed
            duration_steps = max(1, int((abs(distance_m) / speed) / dt))
            return self.move_for_steps(actual_speed, 0, 0, duration_steps)
        else:
            self.base.move(speed, 0, 0)
            return True

    def backward(self, speed: float = 0.2) -> None:
        """Move backward."""
        self.base.move(-speed, 0, 0)

    def rotate(self, omega: float = 0.5) -> None:
        """Rotate in place."""
        self.base.move(0, 0, omega)

    def strafe(self, vy: float = 0.2) -> None:
        """Strafe left/right."""
        self.base.move(0, vy, 0)

    def move_for_steps(self, vx: float, vy: float, omega: float, steps: int) -> bool:
        """Move for specified number of simulation steps."""
        self.move(vx, vy, omega)
        for _ in range(steps):
            if self.robot.step(self.time_step) == -1:
                return False
        self.stop()
        return True

    def turn(self, angle_deg: float, speed: float = 0.5) -> bool:
        """Turn in place by specified angle in degrees."""
        import math
        angle_rad = math.radians(abs(angle_deg))
        # time = angle / angular_speed, steps = time / dt
        dt = self.time_step / 1000.0
        duration_steps = max(1, int((angle_rad / speed) / dt))
        # Note: Sign may be inverted due to coordinate system
        # Positive angle_deg = turn toward positive camera angle direction
        omega = -speed if angle_deg > 0 else speed
        return self.move_for_steps(0, 0, omega, duration_steps)

    def forward_distance(self, distance_m: float, speed: float = 0.1) -> bool:
        """Move forward by specified distance in meters."""
        dt = self.time_step / 1000.0
        duration_steps = max(1, int((distance_m / speed) / dt))
        return self.move_for_steps(speed, 0, 0, duration_steps)
