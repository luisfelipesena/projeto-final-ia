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

    def forward(self, speed: float = 0.2) -> None:
        """Move forward."""
        self.base.move(speed, 0, 0)

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
