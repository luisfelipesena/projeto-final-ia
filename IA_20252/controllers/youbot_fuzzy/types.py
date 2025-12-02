"""Shared dataclasses across controller modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp helper reused everywhere."""
    return max(min_value, min(max_value, value))


@dataclass
class LidarSnapshot:
    front_distance: float = 5.0
    left_distance: float = 5.0
    right_distance: float = 5.0
    obstacle_density: float = 0.0


@dataclass
class CubeHypothesis:
    color: Optional[str] = None
    bearing: float = 0.0  # radians (relative to robot front)
    distance: float = 0.0
    alignment: float = 0.0  # meters offset relative to gripper center
    confidence: float = 0.0


@dataclass
class MotionCommand:
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0
    lift_request: Optional[str] = None  # e.g., "RESET", "FLOOR", "PLATE"
    gripper_request: Optional[str] = None  # "GRIP" or "RELEASE"
