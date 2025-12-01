"""
Gripper controller for YouBot with REAL object detection.

CRITICAL FIX: Uses PositionSensor to detect actual finger position.
When gripper closes on an object, fingers can't reach MIN_POS,
so position > threshold indicates object is held.

The original implementation used just a boolean flag which failed
to detect actual grasping success/failure.
"""

from controller import Robot
from typing import Optional

from ..utils.config import GRIPPER


class GripperController:
    """Controls YouBot parallel gripper with object detection."""

    def __init__(self, robot: Robot):
        """Initialize gripper with position sensor.

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Get finger motor (controls both fingers symmetrically)
        self.finger_motor = robot.getDevice("finger::left")
        if not self.finger_motor:
            raise RuntimeError("Could not find gripper motor 'finger::left'")

        # CRITICAL: Get position sensor for object detection
        self.finger_sensor = robot.getDevice("finger::left:sensor")
        if self.finger_sensor:
            self.finger_sensor.enable(self.time_step)
        else:
            print("[GRIPPER] Warning: No position sensor, object detection disabled")

        # Set velocity for controlled movement
        self.finger_motor.setVelocity(GRIPPER.VELOCITY)

        # Track commanded state (not actual state)
        self._commanded_closed = False

    def grip(self) -> None:
        """Command gripper to close."""
        self.finger_motor.setPosition(GRIPPER.MIN_POS)
        self._commanded_closed = True

    def release(self) -> None:
        """Command gripper to open."""
        self.finger_motor.setPosition(GRIPPER.MAX_POS)
        self._commanded_closed = False

    def set_gap(self, gap: float) -> None:
        """Set specific gap between fingers.

        Args:
            gap: Desired gap in meters (between fingers)
        """
        # Convert gap to motor position (accounting for offset)
        position = 0.5 * (gap - GRIPPER.OFFSET_WHEN_LOCKED)
        position = max(GRIPPER.MIN_POS, min(GRIPPER.MAX_POS, position))

        self.finger_motor.setPosition(position)
        self._commanded_closed = position < GRIPPER.MAX_POS / 2

    def get_finger_position(self) -> Optional[float]:
        """Read actual finger position from sensor.

        Returns:
            Current position in meters, or None if sensor unavailable
        """
        if not self.finger_sensor:
            return None
        return self.finger_sensor.getValue()

    def has_object(self) -> bool:
        """Detect if an object is being held.

        CRITICAL: This is the real object detection using position sensor.
        If gripper was commanded closed but position > threshold,
        something is blocking the fingers = object detected.

        Returns:
            True if object is detected in gripper
        """
        if not self.finger_sensor:
            # Fallback to commanded state if no sensor
            return self._commanded_closed

        if not self._commanded_closed:
            return False

        position = self.finger_sensor.getValue()

        # If commanded closed but position is above threshold,
        # the fingers are blocked by an object
        has_obj = position > GRIPPER.GRIP_THRESHOLD

        return has_obj

    def is_fully_closed(self) -> bool:
        """Check if gripper is completely closed (no object).

        Returns:
            True if fingers are at minimum position
        """
        if not self.finger_sensor:
            return self._commanded_closed

        position = self.finger_sensor.getValue()
        return position <= GRIPPER.GRIP_THRESHOLD

    def is_fully_open(self) -> bool:
        """Check if gripper is completely open.

        Returns:
            True if fingers are at maximum position
        """
        if not self.finger_sensor:
            return not self._commanded_closed

        position = self.finger_sensor.getValue()
        # Allow small tolerance for "fully open"
        return position >= GRIPPER.MAX_POS - 0.002

    @property
    def is_closed_commanded(self) -> bool:
        """Whether close command was sent (not actual state)."""
        return self._commanded_closed

    def grasp_and_verify(self, wait_steps: int = 30) -> bool:
        """Close gripper and verify object was grasped.

        This method should be called repeatedly until it returns True/False
        or timeout is reached. Each call advances one time step.

        Args:
            wait_steps: Number of steps to wait for gripper to close

        Returns:
            True if object detected, False if gripper closed empty
        """
        self.grip()

        # After waiting period, check if object is held
        position = self.get_finger_position()
        if position is None:
            return self._commanded_closed

        return self.has_object()
