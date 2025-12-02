"""
Global configuration and constants for YouBot controller.

All physical constants, limits, and parameters are centralized here
for easy tuning and consistency across modules.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


# =============================================================================
# ROBOT PHYSICAL CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class BaseConfig:
    """Mobile base (Mecanum wheels) configuration."""
    WHEEL_RADIUS: float = 0.05      # meters
    LX: float = 0.228               # Longitudinal distance to wheel (m)
    LY: float = 0.158               # Lateral distance to wheel (m)
    MAX_SPEED: float = 0.3          # Maximum linear velocity (m/s)
    MAX_OMEGA: float = 1.0          # Maximum angular velocity (rad/s)
    SPEED_INCREMENT: float = 0.05   # Velocity increment step


@dataclass(frozen=True)
class ArmConfig:
    """5-DOF arm configuration."""
    # Joint limits (radians) - CRITICAL for IK validation
    JOINT_LIMITS: Dict[str, Tuple[float, float]] = None

    # Segment lengths (meters)
    SEGMENT_LENGTHS: Tuple[float, ...] = (0.253, 0.155, 0.135, 0.081, 0.105)

    # Motor names
    MOTOR_NAMES: Tuple[str, ...] = ('arm1', 'arm2', 'arm3', 'arm4', 'arm5')

    # Velocity for arm2 (slower for safety)
    ARM2_VELOCITY: float = 0.5

    def __post_init__(self):
        # Can't use mutable default, set after
        pass


# Arm joint limits - extracted from Webots/C code
ARM_JOINT_LIMITS = {
    'arm1': (-2.949, 2.949),     # Base rotation
    'arm2': (-1.13, 1.57),       # Shoulder - MOST RESTRICTIVE
    'arm3': (-2.635, 2.548),     # Elbow
    'arm4': (-1.78, 1.78),       # Wrist pitch
    'arm5': (-2.949, 2.949),     # Wrist roll
}

# Arm segment lengths (meters)
ARM_SEGMENT_LENGTHS = (0.253, 0.155, 0.135, 0.081, 0.105)


@dataclass(frozen=True)
class GripperConfig:
    """Gripper configuration."""
    MIN_POS: float = 0.0            # Fully closed position
    MAX_POS: float = 0.025          # Fully open position
    OFFSET_WHEN_LOCKED: float = 0.021
    VELOCITY: float = 0.03          # Gripper motor velocity
    GRIP_THRESHOLD: float = 0.003   # Threshold for object detection
    CUBE_SIZE: float = 0.03         # Cube size (3cm)


# =============================================================================
# SENSOR CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class CameraConfig:
    """Camera sensor configuration."""
    WIDTH: int = 128
    HEIGHT: int = 128
    FOV_DEGREES: float = 57.0       # Field of view
    FOV_RADIANS: float = 0.995      # ~57 degrees in radians


@dataclass(frozen=True)
class LidarConfig:
    """LIDAR sensor configuration."""
    NUM_POINTS: int = 512           # Number of range readings
    MIN_RANGE: float = 0.01         # Minimum valid range (m)
    MAX_RANGE: float = 5.0          # Maximum valid range (m)
    NUM_SECTORS: int = 9            # Number of sectors for obstacle map
    OBSTACLE_THRESHOLD: float = 0.5 # Distance threshold for obstacle


# =============================================================================
# PERCEPTION CONFIGURATION
# =============================================================================

# HSV color ranges for cube detection (calibrated for Webots)
HSV_RANGES = {
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),      # Low red
        (np.array([170, 100, 100]), np.array([180, 255, 255])),   # High red
    ],
    'green': [
        (np.array([35, 100, 100]), np.array([85, 255, 255])),
    ],
    'blue': [
        (np.array([100, 100, 100]), np.array([130, 255, 255])),
    ],
}

# Minimum contour area to consider as cube (pixels^2)
# Reduced to detect distant cubes (10x10=100 is borderline)
MIN_CUBE_AREA = 50

# Distance estimation constants (for monocular camera)
CUBE_REAL_SIZE = 0.03  # 3cm cube
# Focal length: f = (width/2) / tan(FOV/2) = 64 / tan(28.5°) = 118
FOCAL_LENGTH_PIXELS = 118  # Calculated for 128px width, 57° FOV


# =============================================================================
# CONTROL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class NavigationConfig:
    """Navigation control parameters."""
    # Obstacle avoidance
    CRITICAL_DISTANCE: float = 0.25     # Emergency stop distance (m)
    SAFE_DISTANCE: float = 0.5          # Safe navigation distance (m)

    # Approach parameters
    GRASP_DISTANCE: float = 0.15        # Distance to start grasp (m)
    APPROACH_SPEED: float = 0.1         # Speed when approaching cube
    SEARCH_OMEGA: float = 0.4           # Angular velocity for search

    # Alignment
    ALIGN_THRESHOLD: float = 5.0        # Max angle error for grasp (degrees)


@dataclass(frozen=True)
class GraspConfig:
    """Grasp sequence parameters."""
    APPROACH_DISTANCE: float = 0.15     # Distance before grasp
    FINAL_APPROACH: float = 0.05        # Final forward movement
    WAIT_STEPS_ARM: int = 60            # Steps to wait for arm movement
    WAIT_STEPS_GRIPPER: int = 30        # Steps to wait for gripper
    MAX_ALIGN_ATTEMPTS: int = 50        # Max alignment iterations


# =============================================================================
# ARENA CONFIGURATION
# =============================================================================

# Deposit box positions (x, y) in world coordinates
DEPOSIT_BOXES = {
    'green': (0.48, 1.58),
    'blue': (0.48, -1.62),
    'red': (2.31, 0.01),
}

# Arena boundaries
ARENA_X_MIN = -4.29
ARENA_X_MAX = 2.71
ARENA_Y_MIN = -2.0
ARENA_Y_MAX = 2.0


# =============================================================================
# NEURAL NETWORK CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class LidarMLPConfig:
    """LIDAR MLP neural network configuration."""
    INPUT_SIZE: int = 512           # LIDAR points
    NUM_SECTORS: int = 9            # Output sectors
    HIDDEN_SIZE: int = 128          # Hidden layer size
    DROPOUT: float = 0.3            # Dropout rate
    MODEL_PATH: str = 'models/lidar_mlp.pth'


# =============================================================================
# TIMEOUTS AND LIMITS
# =============================================================================

# State machine timeouts (in simulation steps, 16ms each)
TIMEOUT_SEARCHING = 1000        # ~16 seconds
TIMEOUT_APPROACHING = 500       # ~8 seconds
TIMEOUT_GRASPING = 200          # ~3 seconds
TIMEOUT_DEPOSITING = 500        # ~8 seconds

# Maximum attempts before recovery
MAX_GRASP_ATTEMPTS = 3
MAX_APPROACH_LOST_FRAMES = 90   # ~1.5 seconds


# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_FILE = 'data/youbot.log'


# =============================================================================
# INSTANTIATED CONFIGS (for easy import)
# =============================================================================

BASE = BaseConfig()
GRIPPER = GripperConfig()
CAMERA = CameraConfig()
LIDAR = LidarConfig()
NAVIGATION = NavigationConfig()
GRASP = GraspConfig()
LIDAR_MLP = LidarMLPConfig()
