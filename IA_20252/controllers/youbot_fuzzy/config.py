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

# LIDAR orientation after -90° rotation applied in world:
# High lidar: index ~90 = FRONT, 180 = LEFT, 0/360 = RIGHT
# Low lidar: 180° FOV centered forward, index ~45 = FRONT

LIDAR_HIGH = LidarConfig(
    name=LIDAR_HIGH_NAME,
    sampling_period=32,
    horizontal_resolution=360,
    number_of_layers=2,
    field_of_view=math.tau,
    min_range=0.1,
    max_range=7.0,
    near_range=0.05,
    # After rotation: center front at ~90; narrow sectors to avoid walls
    front_sector=(70, 110),   # Forward cone
    left_sector=(140, 200),   # Left side
    right_sector=(320, 360),  # Right side (wrap handled)
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
    # With rotation: center front at ~45
    front_sector=(30, 60),
)
CAMERA_ALIGNMENT_SCALE = 0.5  # meters of lateral offset at image edges

# Webots cube colors are pure RGB - adjusted for rendering variations
# HSV ranges: H=0-180, S=0-100, V=0-100 (matching color_classifier._to_hsv output)
HSV_RANGES = {
    "red": [((0, 50, 50), (15, 100, 100)), ((165, 50, 50), (180, 100, 100))],
    "green": [((40, 50, 50), (80, 100, 100))],
    "blue": [((100, 50, 50), (135, 100, 100))],
}
HSV_COVERAGE_THRESHOLD = 0.001  # 0.1% of frame (~16 pixels in 128x128) - cubes are tiny at distance

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

# Dead zones with -90° Z rotation (Index 0=FRONT, 90=LEFT, 180=BACK, 270=RIGHT):
# Robot body blocks rays going backward: ~150-210 (around 180° = back)
# Arm may interfere near front when extended, but usually tucked
# Left/right sectors (90/270) should now be clear of body
# Dead zones: robot body blocks rays around index 90 (left side)
# Verified from logs: index 90 reads 0.11m constantly
LIDAR_DEAD_ZONES = [(60, 120)]  # Left side blocked by robot body - expanded margin

CUBE_DETECTION_MIN_DISTANCE = 0.08  # Minimum distance to detect cube (avoid noise)
CUBE_DETECTION_MAX_DISTANCE = 0.50  # Only detect cubes within 50cm (arm reach ~25cm + approach ~25cm)
CUBE_HEIGHT_DIFFERENCE_THRESHOLD = 0.4  # High lidar must see 40cm+ farther than low (cube is only 3cm tall)
DANGER_ZONE = 0.40  # Trigger escape earlier to prevent wall collision
