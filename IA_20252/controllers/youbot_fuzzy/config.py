"""Centralized configuration constants for the YouBot fuzzy controller."""

from __future__ import annotations

import math
from pathlib import Path
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
    name: str
    sampling_period: int
    horizontal_resolution: int
    number_of_layers: int
    field_of_view: float
    min_range: float
    max_range: float
    near_range: float = 0.0
    front_sector: Tuple[int, int] = (0, 0)
    left_sector: Tuple[int, int] = (0, 0)
    right_sector: Tuple[int, int] = (0, 0)


LIDAR_LOW_NAME = "lidar_low"
LIDAR_HIGH_NAME = "lidar_high"

LIDAR_HIGH = LidarConfig(
    name=LIDAR_HIGH_NAME,
    sampling_period=32,
    horizontal_resolution=360,
    number_of_layers=2,
    field_of_view=math.tau,
    min_range=0.1,
    max_range=7.0,
    near_range=0.05,
    # Webots LIDAR convention (empirically verified):
    # Index 0=left, 90=front, 180=right, 270=back
    front_sector=(75, 105),   # Around index 90 (±15°)
    left_sector=(345, 375),   # Around index 0/360 (±15°, wraps)
    right_sector=(165, 195),  # Around index 180 (±15°)
)

LIDAR_LOW = LidarConfig(
    name=LIDAR_LOW_NAME,
    sampling_period=32,
    horizontal_resolution=180,
    number_of_layers=1,
    field_of_view=math.pi,
    min_range=0.03,
    max_range=2.5,
    near_range=0.02,
)
CAMERA_ALIGNMENT_SCALE = 0.5  # meters of lateral offset at image edges

# Webots cube colors are pure RGB - adjusted for rendering variations
HSV_RANGES = {
    "red": [((0, 80, 80), (15, 255, 255)), ((165, 80, 80), (180, 255, 255))],
    "green": [((40, 80, 80), (80, 255, 255))],
    "blue": [((100, 80, 80), (135, 255, 255))],
}
HSV_COVERAGE_THRESHOLD = 0.001  # 0.1% of frame (cubes are small at distance)

CAMERA_NAME = "camera"

# --- Mission parameters -----------------------------------------------------

MAX_CUBES = 15
GOAL_SEQUENCE = ("GREEN", "BLUE", "RED")
ENABLE_LOGGING = True
LOG_INTERVAL_STEPS = 10

# --- Vision / ML configuration ----------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent / "models"
ENABLE_YOLO = True
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-cubes.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.35

ENABLE_ADABOOST = True
ADABOOST_MODEL_PATH = MODELS_DIR / "adaboost_color.pkl"

# --- LIDAR detection thresholds ---------------------------------------------

CUBE_DETECTION_MIN_DISTANCE = 0.05
CUBE_DETECTION_MAX_DISTANCE = 1.5
CUBE_HEIGHT_DIFFERENCE_THRESHOLD = 0.2
DANGER_ZONE = 0.15  # Reduced - robot should explore more before escaping
