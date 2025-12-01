"""
5-DOF Arm controller for YouBot with joint limit validation.

CRITICAL: All joint positions are validated before applying to prevent
the "too low requested position" errors seen in failed implementations.
"""

from controller import Robot
from enum import IntEnum
from typing import Tuple, Optional, List
import math

from ..utils.config import ARM_JOINT_LIMITS, ARM_SEGMENT_LENGTHS


class ArmHeight(IntEnum):
    """Height presets for arm configuration."""
    FRONT_FLOOR = 0
    FRONT_PLATE = 1
    FRONT_CARDBOARD_BOX = 2
    RESET = 3
    BACK_PLATE_HIGH = 4
    BACK_PLATE_LOW = 5
    HANOI_PREPARE = 6


class ArmOrientation(IntEnum):
    """Base rotation presets."""
    BACK_LEFT = 0
    LEFT = 1
    FRONT_LEFT = 2
    FRONT = 3
    FRONT_RIGHT = 4
    RIGHT = 5
    BACK_RIGHT = 6


# Pre-validated height configurations: (arm2, arm3, arm4, arm5)
HEIGHT_POSITIONS = {
    ArmHeight.FRONT_FLOOR: (-0.97, -1.55, -0.61, 0.0),
    ArmHeight.FRONT_PLATE: (-0.62, -0.98, -1.53, 0.0),
    ArmHeight.FRONT_CARDBOARD_BOX: (0.0, -0.77, -1.21, 0.0),
    ArmHeight.RESET: (1.57, -2.635, 1.78, 0.0),
    ArmHeight.BACK_PLATE_HIGH: (0.678, 0.682, 1.74, 0.0),
    ArmHeight.BACK_PLATE_LOW: (0.92, 0.42, 1.78, 0.0),
    ArmHeight.HANOI_PREPARE: (-0.4, -1.2, -math.pi / 2, math.pi / 2),
}

# Orientation angles for arm1 (base rotation)
ORIENTATION_ANGLES = {
    ArmOrientation.BACK_LEFT: -2.949,
    ArmOrientation.LEFT: -math.pi / 2,
    ArmOrientation.FRONT_LEFT: -0.2,
    ArmOrientation.FRONT: 0.0,
    ArmOrientation.FRONT_RIGHT: 0.2,
    ArmOrientation.RIGHT: math.pi / 2,
    ArmOrientation.BACK_RIGHT: 2.949,
}


class ArmController:
    """Controls YouBot 5-DOF arm with joint limit validation."""

    MOTOR_NAMES = ('arm1', 'arm2', 'arm3', 'arm4', 'arm5')

    def __init__(self, robot: Robot):
        """Initialize arm motors with sensors.

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Get motors
        self.motors = [robot.getDevice(name) for name in self.MOTOR_NAMES]

        # Get position sensors for feedback
        self.sensors = []
        for i, name in enumerate(self.MOTOR_NAMES):
            sensor = robot.getDevice(f"{name}sensor")
            if sensor:
                sensor.enable(self.time_step)
                self.sensors.append(sensor)
            else:
                self.sensors.append(None)

        # Set arm2 to slower velocity for safety (from original code)
        self.motors[1].setVelocity(0.5)

        # Current state
        self._current_height = ArmHeight.RESET
        self._current_orientation = ArmOrientation.FRONT

        # Initialize to reset position
        self.set_height(ArmHeight.RESET)
        self.set_orientation(ArmOrientation.FRONT)

    @staticmethod
    def _clamp_to_limits(joint_name: str, position: float) -> Tuple[float, bool]:
        """Clamp position to joint limits.

        Args:
            joint_name: Motor name (arm1-arm5)
            position: Desired position in radians

        Returns:
            (clamped_position, was_clamped)
        """
        min_pos, max_pos = ARM_JOINT_LIMITS[joint_name]
        clamped = max(min_pos, min(max_pos, position))
        was_clamped = abs(clamped - position) > 0.001
        return clamped, was_clamped

    def _validate_and_set(self, joint_idx: int, position: float) -> bool:
        """Validate position and set motor.

        Args:
            joint_idx: Motor index (0-4)
            position: Target position in radians

        Returns:
            True if position was valid, False if clamped
        """
        joint_name = self.MOTOR_NAMES[joint_idx]
        clamped_pos, was_clamped = self._clamp_to_limits(joint_name, position)

        if was_clamped:
            min_pos, max_pos = ARM_JOINT_LIMITS[joint_name]
            print(f"[ARM] Warning: {joint_name} clamped {position:.3f} -> {clamped_pos:.3f} "
                  f"(limits: [{min_pos:.3f}, {max_pos:.3f}])")

        self.motors[joint_idx].setPosition(clamped_pos)
        return not was_clamped

    def set_joint_positions(self, positions: Tuple[float, ...]) -> bool:
        """Set all joint positions with validation.

        Args:
            positions: Tuple of 5 joint positions (arm1-arm5) in radians

        Returns:
            True if all positions were valid
        """
        if len(positions) != 5:
            raise ValueError("Must provide exactly 5 joint positions")

        all_valid = True
        for i, pos in enumerate(positions):
            if not self._validate_and_set(i, pos):
                all_valid = False

        return all_valid

    def set_height(self, height: ArmHeight) -> None:
        """Set arm to height preset.

        Args:
            height: Height preset from ArmHeight enum
        """
        if height not in HEIGHT_POSITIONS:
            print(f"[ARM] Error: Invalid height {height}")
            return

        positions = HEIGHT_POSITIONS[height]
        for i, pos in enumerate(positions):
            self._validate_and_set(i + 1, pos)  # arm2-arm5 are indices 1-4

        self._current_height = height

    def set_orientation(self, orientation: ArmOrientation) -> None:
        """Set arm base rotation.

        Args:
            orientation: Orientation preset from ArmOrientation enum
        """
        if orientation not in ORIENTATION_ANGLES:
            print(f"[ARM] Error: Invalid orientation {orientation}")
            return

        angle = ORIENTATION_ANGLES[orientation]
        self._validate_and_set(0, angle)  # arm1 is index 0
        self._current_orientation = orientation

    def reset(self) -> None:
        """Reset arm to initial position."""
        self.set_height(ArmHeight.RESET)
        self.set_orientation(ArmOrientation.FRONT)

    def inverse_kinematics(self, x: float, y: float, z: float) -> bool:
        """Move arm to (x, y, z) using inverse kinematics.

        Uses law of cosines to compute joint angles. All angles are
        validated before applying to prevent joint limit violations.

        Args:
            x: Target x position (meters)
            y: Target y position (meters)
            z: Target z position (meters)

        Returns:
            True if IK solution is within joint limits, False otherwise
        """
        # Segment lengths
        L0, L1, L2, L3, L4 = ARM_SEGMENT_LENGTHS

        # Calculate intermediate values
        y1 = math.sqrt(x * x + y * y)
        z1 = z + L3 + L4 - L0

        a = L1
        b = L2
        c = math.sqrt(y1 * y1 + z1 * z1)

        # Check reachability
        if c > a + b:
            print(f"[ARM] IK Error: Position ({x:.3f}, {y:.3f}, {z:.3f}) unreachable (too far)")
            return False
        if c < abs(a - b):
            print(f"[ARM] IK Error: Position ({x:.3f}, {y:.3f}, {z:.3f}) unreachable (too close)")
            return False

        # Calculate joint angles
        alpha = -math.asin(x / y1) if y1 > 0.001 else 0.0

        # Law of cosines for beta (arm2)
        cos_beta_part = (a * a + c * c - b * b) / (2.0 * a * c)
        cos_beta_part = max(-1.0, min(1.0, cos_beta_part))
        beta = -(math.pi / 2 - math.acos(cos_beta_part) - math.atan2(z1, y1))

        # Law of cosines for gamma (arm3)
        cos_gamma_part = (a * a + b * b - c * c) / (2.0 * a * b)
        cos_gamma_part = max(-1.0, min(1.0, cos_gamma_part))
        gamma = -(math.pi - math.acos(cos_gamma_part))

        # Derived angles
        delta = -(math.pi + (beta + gamma))
        epsilon = math.pi / 2 + alpha

        # Validate all positions before applying
        angles = [alpha, beta, gamma, delta, epsilon]
        all_valid = True

        for i, (name, angle) in enumerate(zip(self.MOTOR_NAMES, angles)):
            _, was_clamped = self._clamp_to_limits(name, angle)
            if was_clamped:
                all_valid = False

        # Apply (with clamping if needed)
        for i, angle in enumerate(angles):
            self._validate_and_set(i, angle)

        return all_valid

    def get_joint_positions(self) -> Optional[List[float]]:
        """Read current joint positions from sensors.

        Returns:
            List of 5 positions, or None if sensors unavailable
        """
        if not all(self.sensors):
            return None
        return [sensor.getValue() for sensor in self.sensors]

    @property
    def current_height(self) -> ArmHeight:
        """Current height preset."""
        return self._current_height

    @property
    def current_orientation(self) -> ArmOrientation:
        """Current orientation preset."""
        return self._current_orientation

    def prepare_for_grasp(self) -> None:
        """Move arm to position ready for grasping floor objects."""
        self.set_orientation(ArmOrientation.FRONT)
        self.set_height(ArmHeight.FRONT_FLOOR)

    def prepare_for_deposit(self) -> None:
        """Move arm to position for depositing objects."""
        self.set_orientation(ArmOrientation.FRONT)
        self.set_height(ArmHeight.FRONT_CARDBOARD_BOX)

    def retract(self) -> None:
        """Retract arm to safe transport position."""
        self.set_height(ArmHeight.RESET)
