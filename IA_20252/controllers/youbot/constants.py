"""
Constants and arena configuration for YouBot controller.
Extracted from youbot.py for modularity.
"""

# Cube dimensions
CUBE_SIZE = 0.03

# Arena configuration
ARENA_CENTER = (-0.79, 0.0)
ARENA_SIZE = (7.0, 4.0)

# Known obstacles (WoodenBoxes) - position (x, y) and safety radius
# Based on IA_20252.wbt world file
KNOWN_OBSTACLES = [
    (0.6, 0.0, 0.25),      # A - center
    (1.96, -1.24, 0.25),   # B - bottom right
    (1.95, 1.25, 0.25),    # C - top right
    (-2.28, 1.5, 0.25),    # D - top left
    (-1.02, 0.75, 0.25),   # E - upper center-left
    (-1.02, -0.74, 0.25),  # F - lower center-left
    (-2.27, -1.51, 0.25),  # G - bottom left
]

# Deposit boxes (PlasticFruitBox) positions
BOX_POSITIONS = {
    "green": (0.48, 1.58),
    "blue": (0.48, -1.62),
    "red": (2.31, 0.01),
}

# Robot spawn position
SPAWN_POSITION = (-3.91, 0.0)

# Grid configuration
DEFAULT_CELL_SIZE = 0.12

# Navigation thresholds
GRASP_DISTANCE = 0.32
GRASP_ANGLE_MAX_DEG = 10
APPROACH_ANGLE_THRESHOLD_DEG = 5

# Speed limits
MAX_SPEED = 0.25
MAX_OMEGA = 0.6

# Sensor thresholds
EMERGENCY_STOP_DISTANCE = 0.22
FRONT_DANGER_DISTANCE = 0.35
FRONT_WARN_DISTANCE = 0.55
LATERAL_WARN_DISTANCE = 0.35
REAR_CLEARANCE_MIN = 0.30

# Waypoint thresholds
WAYPOINT_THRESHOLD_INTERMEDIATE = 0.40
WAYPOINT_THRESHOLD_PREFINAL = 0.30
WAYPOINT_THRESHOLD_FINAL = 0.50

# Timing
GT_SYNC_INTERVAL_NAVIGATION = 0.5  # seconds
GT_SYNC_INTERVAL_DEFAULT = 2.0     # seconds
