"""
Test helper utilities for YouBot control validation tests.
Provides fixtures and utility functions for pytest.
"""

import sys
import os
import math

# Add controller path for importing YouBot modules
CONTROLLER_PATH = os.path.join(os.path.dirname(__file__), '..', 'IA_20252', 'controllers', 'youbot')
sys.path.insert(0, CONTROLLER_PATH)


def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two 2D positions.

    Args:
        pos1: tuple (x1, y1)
        pos2: tuple (x2, y2)

    Returns:
        float: distance in meters
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.sqrt(dx * dx + dy * dy)


def normalize_angle(angle):
    """Normalize angle to [-π, π] range.

    Args:
        angle: angle in radians

    Returns:
        float: normalized angle in [-π, π]
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_robot_position(robot):
    """Get robot position from GPS or supervisor field (if available).

    Args:
        robot: Webots Robot instance

    Returns:
        tuple: (x, y) position in meters, or None if unavailable
    """
    # Try to get GPS device
    gps = robot.getDevice("gps")
    if gps:
        gps.enable(int(robot.getBasicTimeStep()))
        robot.step(int(robot.getBasicTimeStep()))
        values = gps.getValues()
        return (values[0], values[1])

    # Fallback: Try supervisor field (requires supervisor mode)
    try:
        root = robot.getSelf()
        if root:
            translation_field = root.getField("translation")
            if translation_field:
                values = translation_field.getSFVec3f()
                return (values[0], values[1])
    except:
        pass

    return None


def get_robot_heading(robot):
    """Get robot heading angle from compass or rotation field.

    Args:
        robot: Webots Robot instance

    Returns:
        float: heading angle in radians, or None if unavailable
    """
    # Try to get compass device
    compass = robot.getDevice("compass")
    if compass:
        compass.enable(int(robot.getBasicTimeStep()))
        robot.step(int(robot.getBasicTimeStep()))
        values = compass.getValues()
        # Compass returns direction vector (x, z), calculate angle
        heading = math.atan2(values[0], values[1])
        return normalize_angle(heading)

    # Fallback: Try supervisor rotation field
    try:
        root = robot.getSelf()
        if root:
            rotation_field = root.getField("rotation")
            if rotation_field:
                # Rotation is axis-angle: [x, y, z, angle]
                values = rotation_field.getSFRotation()
                # For ground robot, rotation around Y-axis
                angle = values[3] if values[1] > 0 else -values[3]
                return normalize_angle(angle)
    except:
        pass

    return None


def wait_for_motion(robot, duration_ms):
    """Execute simulation steps for a specified duration.

    Args:
        robot: Webots Robot instance
        duration_ms: duration in milliseconds

    Returns:
        int: number of steps executed
    """
    time_step = int(robot.getBasicTimeStep())
    num_steps = int(duration_ms / time_step)

    for _ in range(num_steps):
        robot.step(time_step)

    return num_steps


def is_position_stable(positions, tolerance=0.01):
    """Check if robot position is stable (not moving).

    Args:
        positions: list of recent (x, y) positions
        tolerance: maximum position variance in meters

    Returns:
        bool: True if position is stable
    """
    if len(positions) < 3:
        return False

    # Calculate variance in x and y
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    x_var = max(xs) - min(xs)
    y_var = max(ys) - min(ys)

    return (x_var < tolerance) and (y_var < tolerance)
