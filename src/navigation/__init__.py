"""
Navigation Module

Provides local mapping and odometry for GPS-free navigation.

Components:
- local_map: Local occupancy grid from LIDAR
- odometry: Wheel-based position estimation
"""

from .local_map import LocalMap, OccupancyCell
from .odometry import Odometry, Pose2D

__all__ = [
    'LocalMap',
    'OccupancyCell',
    'Odometry',
    'Pose2D',
]
