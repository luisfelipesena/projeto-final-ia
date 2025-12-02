"""
Omnidirectional base controller for YouBot (Mecanum wheels).

Implements proper kinematics for 4-wheel omnidirectional movement.
"""

from controller import Robot
from typing import Tuple
from utils.config import BASE


class BaseController:
    """Controls YouBot mobile base with Mecanum wheel kinematics."""

    # Wheel names in Webots
    WHEEL_NAMES = ('wheel1', 'wheel2', 'wheel3', 'wheel4')

    def __init__(self, robot: Robot):
        """Initialize base motors.

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Get wheel motors
        self.wheels = [robot.getDevice(name) for name in self.WHEEL_NAMES]

        # Set to velocity control mode (infinite position)
        for wheel in self.wheels:
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)

        # Current velocity state
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between bounds."""
        return max(min_val, min(max_val, value))

    def _compute_wheel_speeds(self, vx: float, vy: float, omega: float) -> Tuple[float, float, float, float]:
        """Compute wheel speeds from body velocities using Mecanum kinematics.

        Mecanum Wheel Layout (top view):
            [0] front-left    [1] front-right
            [2] rear-left     [3] rear-right

        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s, positive = left)
            omega: Angular velocity (rad/s, positive = CCW)

        Returns:
            Tuple of 4 wheel angular velocities (rad/s)
        """
        r = BASE.WHEEL_RADIUS
        lx_ly = BASE.LX + BASE.LY  # Combined distance factor

        # Inverse kinematics for Mecanum wheels
        # Each wheel has rollers at 45°, contributing to omnidirectional motion
        w0 = (1.0 / r) * (vx - vy - lx_ly * omega)  # front-left
        w1 = (1.0 / r) * (vx + vy + lx_ly * omega)  # front-right
        w2 = (1.0 / r) * (vx + vy - lx_ly * omega)  # rear-left
        w3 = (1.0 / r) * (vx - vy + lx_ly * omega)  # rear-right

        return (w0, w1, w2, w3)

    def _apply_wheel_speeds(self, speeds: Tuple[float, float, float, float]) -> None:
        """Apply wheel speeds to motors."""
        for wheel, speed in zip(self.wheels, speeds):
            wheel.setVelocity(speed)

    def move(self, vx: float, vy: float, omega: float) -> None:
        """Move base with omnidirectional velocity.

        Args:
            vx: Forward velocity (m/s), clamped to MAX_SPEED
            vy: Lateral velocity (m/s), positive = left
            omega: Angular velocity (rad/s), clamped to MAX_OMEGA
        """
        # Clamp velocities to safe limits
        vx = self._clamp(vx, -BASE.MAX_SPEED, BASE.MAX_SPEED)
        vy = self._clamp(vy, -BASE.MAX_SPEED, BASE.MAX_SPEED)
        omega = self._clamp(omega, -BASE.MAX_OMEGA, BASE.MAX_OMEGA)

        # Store state
        self._vx = vx
        self._vy = vy
        self._omega = omega

        # Compute and apply wheel speeds
        speeds = self._compute_wheel_speeds(vx, vy, omega)
        self._apply_wheel_speeds(speeds)

    def stop(self) -> None:
        """Stop all wheel movement."""
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._apply_wheel_speeds((0.0, 0.0, 0.0, 0.0))

    def forward(self, speed: float = None) -> None:
        """Move forward.

        Args:
            speed: Forward speed (m/s), defaults to MAX_SPEED
        """
        speed = speed if speed is not None else BASE.MAX_SPEED
        self.move(speed, 0.0, 0.0)

    def backward(self, speed: float = None) -> None:
        """Move backward."""
        speed = speed if speed is not None else BASE.MAX_SPEED
        self.move(-speed, 0.0, 0.0)

    def strafe_left(self, speed: float = None) -> None:
        """Strafe left."""
        speed = speed if speed is not None else BASE.MAX_SPEED
        self.move(0.0, speed, 0.0)

    def strafe_right(self, speed: float = None) -> None:
        """Strafe right."""
        speed = speed if speed is not None else BASE.MAX_SPEED
        self.move(0.0, -speed, 0.0)

    def rotate_left(self, omega: float = None) -> None:
        """Rotate counter-clockwise."""
        omega = omega if omega is not None else BASE.MAX_OMEGA
        self.move(0.0, 0.0, omega)

    def rotate_right(self, omega: float = None) -> None:
        """Rotate clockwise."""
        omega = omega if omega is not None else BASE.MAX_OMEGA
        self.move(0.0, 0.0, -omega)

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Current velocity state (vx, vy, omega)."""
        return (self._vx, self._vy, self._omega)

    @property
    def is_moving(self) -> bool:
        """True if base has non-zero velocity."""
        return self._vx != 0.0 or self._vy != 0.0 or self._omega != 0.0
