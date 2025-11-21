"""
Neural Network Perception System

This module implements the perception layer for the YouBot autonomous system,
combining LIDAR obstacle detection and camera-based cube color classification.

Components:
- lidar_processor: LIDAR neural network for obstacle detection
- cube_detector: CNN for cube color classification
- perception_system: Unified perception interface
"""

from .lidar_processor import LIDARProcessor, ObstacleMap
from .cube_detector import CubeDetector, CubeObservation, BoundingBox
from .perception_system import PerceptionSystem, WorldState

__all__ = [
    'LIDARProcessor',
    'ObstacleMap',
    'CubeDetector',
    'CubeObservation',
    'BoundingBox',
    'PerceptionSystem',
    'WorldState',
]
