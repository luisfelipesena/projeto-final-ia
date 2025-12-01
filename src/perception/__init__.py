"""
Perception Module - Sensory processing for YouBot.

Components:
- CubeDetector: HSV-based color detection for cubes
- LidarProcessor: LIDAR data processing and obstacle detection
- LidarMLP: Neural network for obstacle classification
"""

from .cube_detector import CubeDetector, CubeDetection
from .lidar_processor import LidarProcessor, SectorInfo
from .lidar_mlp import LidarMLP

__all__ = [
    'CubeDetector',
    'CubeDetection',
    'LidarProcessor',
    'SectorInfo',
    'LidarMLP',
]
