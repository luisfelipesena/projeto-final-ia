"""Centralized configuration constants for the YouBot fuzzy controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# --- Arena & object metrics -------------------------------------------------

ARENA_SIZE: Tuple[float, float] = (7.0, 4.0)  # meters (rectangle arena)
CUBE_SIZE = 0.03  # meters
CUBE_MASS = 0.03  # kilograms
DEFAULT_DISTANCE = 5.0  # safety cap for lidar readings
OBSTACLE_THRESHOLD = 1.0  # meters – closer than this is considered \"near\"

BOX_TARGETS: Dict[str, Tuple[float, float]] = {
    "GREEN": (0.48, 1.58),
    "BLUE": (0.48, -1.62),
    "RED": (2.31, 0.01),
}

WOODEN_BOXES = {
    "A": (0.60, 0.00),
    "B": (1.96, -1.24),
    "C": (1.95, 1.25),
    "D": (-2.28, 1.50),
    "E": (-1.02, 0.75),
    "F": (-1.02, -0.74),
    "G": (-2.27, -1.51),
}

# --- Robot & motion parameters ----------------------------------------------

BASE_MAX_SPEED = 0.3  # m/s, consistent with Base.MAX_SPEED
BASE_SLOW_SPEED = 0.05  # m/s approach speed validated in draft
ARM_WAIT_SECONDS = {
    "RESET": 1.5,
    "FLOOR": 2.5,
    "GRIP": 1.5,
    "PLATE": 2.0,
}

# --- Sensor configuration ---------------------------------------------------

TIME_STEP_MS = 32  # fallback; real value fetched from robot

@dataclass(frozen=True)
class LidarConfig:
    name: str = "lidar"
    sampling_period: int = 32  # milliseconds
    horizontal_resolution: int = 512
    number_of_layers: int = 3
    front_sector: Tuple[int, int] = (220, 292)
    left_sector: Tuple[int, int] = (292, 360)
    right_sector: Tuple[int, int] = (160, 220)


LIDAR = LidarConfig()
CAMERA_ALIGNMENT_SCALE = 0.5  # meters of lateral offset at image edges

HSV_RANGES = {
    "red": [((0, 100, 100), (10, 255, 255)), ((170, 100, 100), (180, 255, 255))],
    "green": [((35, 100, 100), (85, 255, 255))],
    "blue": [((100, 100, 100), (130, 255, 255))],
}

CAMERA_NAME = "camera"

# --- Mission parameters -----------------------------------------------------

MAX_CUBES = 15
GOAL_SEQUENCE = ("GREEN", "BLUE", "RED")
ENABLE_LOGGING = True
LOG_INTERVAL_STEPS = 10
