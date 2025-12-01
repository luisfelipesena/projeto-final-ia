#!/usr/bin/env python3
"""
Generate synthetic LIDAR training data.

Creates realistic LIDAR readings based on known arena geometry:
- Arena: 7m x 4m
- Walls at boundaries
- 7 WoodenBoxes as obstacles
- Cubes and deposit boxes

Usage:
    python scripts/generate_synthetic_lidar.py --samples 2000
"""

import argparse
import json
import math
import os
import random
from typing import List, Tuple

import numpy as np


# Arena geometry
ARENA_WIDTH = 7.0   # X axis
ARENA_HEIGHT = 4.0  # Y axis

# Obstacle positions (from world file)
OBSTACLES = [
    # WoodenBoxes (x, y, width, depth)
    (0.61, 0.04, 0.6, 0.6),     # A
    (2.25, 0.80, 0.6, 0.6),     # B
    (2.07, -0.93, 0.6, 0.6),    # C
    (-0.75, 0.89, 0.6, 0.6),    # D
    (-2.25, -0.11, 0.6, 0.6),   # E
    (-0.84, -0.9, 0.6, 0.6),    # F
    (-1.7, 0.9, 0.6, 0.6),      # G
    # Deposit boxes
    (0.48, 1.58, 0.4, 0.4),     # GREEN
    (0.48, -1.62, 0.4, 0.4),    # BLUE
    (2.31, 0.01, 0.4, 0.4),     # RED
]

# LIDAR specs
NUM_POINTS = 512
LIDAR_FOV = 240.0  # degrees
MIN_RANGE = 0.01
MAX_RANGE = 5.0
LIDAR_NOISE_STD = 0.02

# Sectors
NUM_SECTORS = 9
SECTOR_NAMES = [
    'back_right', 'right', 'front_right',
    'front',
    'front_left', 'left', 'back_left',
    'back_center_left', 'back_center_right'
]


def generate_lidar_reading(robot_x: float, robot_y: float, robot_theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate LIDAR reading from robot position.

    Args:
        robot_x: Robot X position
        robot_y: Robot Y position
        robot_theta: Robot heading (radians)

    Returns:
        (ranges, sector_labels) - 512 ranges and 9 binary sector labels
    """
    ranges = np.full(NUM_POINTS, MAX_RANGE, dtype=np.float32)

    # Calculate angle for each LIDAR point
    start_angle = -math.radians(LIDAR_FOV / 2)
    angle_step = math.radians(LIDAR_FOV) / (NUM_POINTS - 1)

    for i in range(NUM_POINTS):
        local_angle = start_angle + i * angle_step
        world_angle = robot_theta + local_angle

        # Ray cast to find distance
        distance = ray_cast(robot_x, robot_y, world_angle)
        ranges[i] = np.clip(distance + np.random.normal(0, LIDAR_NOISE_STD), MIN_RANGE, MAX_RANGE)

    # Generate sector labels based on ranges
    sector_labels = compute_sector_labels(ranges)

    return ranges, sector_labels


def ray_cast(x: float, y: float, angle: float) -> float:
    """Cast ray and return distance to nearest obstacle."""
    min_dist = MAX_RANGE

    dx = math.cos(angle)
    dy = math.sin(angle)

    # Check walls
    # Top wall (y = ARENA_HEIGHT/2)
    if dy > 0:
        t = (ARENA_HEIGHT / 2 - y) / dy
        if 0 < t < min_dist:
            hit_x = x + t * dx
            if -ARENA_WIDTH / 2 <= hit_x <= ARENA_WIDTH / 2:
                min_dist = t

    # Bottom wall (y = -ARENA_HEIGHT/2)
    if dy < 0:
        t = (-ARENA_HEIGHT / 2 - y) / dy
        if 0 < t < min_dist:
            hit_x = x + t * dx
            if -ARENA_WIDTH / 2 <= hit_x <= ARENA_WIDTH / 2:
                min_dist = t

    # Right wall (x = ARENA_WIDTH/2)
    if dx > 0:
        t = (ARENA_WIDTH / 2 - x) / dx
        if 0 < t < min_dist:
            hit_y = y + t * dy
            if -ARENA_HEIGHT / 2 <= hit_y <= ARENA_HEIGHT / 2:
                min_dist = t

    # Left wall (x = -ARENA_WIDTH/2)
    if dx < 0:
        t = (-ARENA_WIDTH / 2 - x) / dx
        if 0 < t < min_dist:
            hit_y = y + t * dy
            if -ARENA_HEIGHT / 2 <= hit_y <= ARENA_HEIGHT / 2:
                min_dist = t

    # Check obstacles
    for obs_x, obs_y, obs_w, obs_h in OBSTACLES:
        dist = ray_box_intersection(x, y, dx, dy, obs_x, obs_y, obs_w, obs_h)
        if dist is not None and dist < min_dist:
            min_dist = dist

    return min_dist


def ray_box_intersection(rx: float, ry: float, dx: float, dy: float,
                         bx: float, by: float, bw: float, bh: float) -> float:
    """Check ray-box intersection."""
    half_w = bw / 2
    half_h = bh / 2

    # Box bounds
    left = bx - half_w
    right = bx + half_w
    bottom = by - half_h
    top = by + half_h

    t_min = 0.0
    t_max = MAX_RANGE

    # X slab
    if abs(dx) < 1e-8:
        if rx < left or rx > right:
            return None
    else:
        t1 = (left - rx) / dx
        t2 = (right - rx) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    # Y slab
    if abs(dy) < 1e-8:
        if ry < bottom or ry > top:
            return None
    else:
        t1 = (bottom - ry) / dy
        t2 = (top - ry) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    return t_min if t_min > 0 else None


def compute_sector_labels(ranges: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Compute binary sector labels from ranges.

    Args:
        ranges: 512 LIDAR ranges
        threshold: Distance threshold for obstacle

    Returns:
        9 binary labels (1 = obstacle present)
    """
    points_per_sector = NUM_POINTS // NUM_SECTORS
    labels = np.zeros(NUM_SECTORS, dtype=np.float32)

    for i in range(NUM_SECTORS):
        start = i * points_per_sector
        end = start + points_per_sector
        sector_ranges = ranges[start:end]

        # Label as obstacle if min range in sector < threshold
        if sector_ranges.min() < threshold:
            labels[i] = 1.0

    return labels


def generate_dataset(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate full training dataset.

    Args:
        num_samples: Number of samples to generate

    Returns:
        (readings, labels) - shapes (N, 512) and (N, 9)
    """
    readings = []
    labels = []

    # Safe spawn area (avoid spawning inside obstacles)
    safe_margin = 0.5

    for _ in range(num_samples):
        # Random position within arena
        x = random.uniform(-ARENA_WIDTH / 2 + safe_margin, ARENA_WIDTH / 2 - safe_margin)
        y = random.uniform(-ARENA_HEIGHT / 2 + safe_margin, ARENA_HEIGHT / 2 - safe_margin)

        # Skip if inside obstacle
        inside_obstacle = False
        for obs_x, obs_y, obs_w, obs_h in OBSTACLES:
            if (abs(x - obs_x) < obs_w / 2 + 0.3 and
                abs(y - obs_y) < obs_h / 2 + 0.3):
                inside_obstacle = True
                break

        if inside_obstacle:
            continue

        # Random heading
        theta = random.uniform(-math.pi, math.pi)

        # Generate reading
        lidar_ranges, sector_labels = generate_lidar_reading(x, y, theta)

        readings.append(lidar_ranges)
        labels.append(sector_labels)

    return np.array(readings), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic LIDAR training data')
    parser.add_argument('--samples', type=int, default=2000,
                        help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/lidar/synthetic_training.json',
                        help='Output file path')

    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic LIDAR samples...")

    readings, labels = generate_dataset(args.samples)

    print(f"Generated {len(readings)} valid samples")
    print(f"Readings shape: {readings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Stats
    obstacle_ratio = labels.mean(axis=0)
    print(f"\nObstacle ratio per sector:")
    for i, ratio in enumerate(obstacle_ratio):
        print(f"  Sector {i} ({SECTOR_NAMES[i] if i < len(SECTOR_NAMES) else 'unknown'}): {ratio:.2%}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    data = {
        'lidar_readings': readings.tolist(),
        'sector_labels': labels.tolist(),
        'metadata': {
            'num_samples': len(readings),
            'num_points': NUM_POINTS,
            'num_sectors': NUM_SECTORS,
            'arena_size': [ARENA_WIDTH, ARENA_HEIGHT],
            'synthetic': True,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(data, f)

    print(f"\nData saved to {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
