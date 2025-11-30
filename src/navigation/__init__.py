"""
Navigation Module

Provides odometry for GPS-free navigation.

Components:
- odometry: Wheel-based position estimation
"""

from .odometry import Odometry, Pose2D, get_deposit_box_pose, DEPOSIT_BOXES

__all__ = [
    'Odometry',
    'Pose2D',
    'get_deposit_box_pose',
    'DEPOSIT_BOXES',
]
