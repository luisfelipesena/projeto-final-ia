"""
YouBot Modular Services

Step-by-step testable components for robot control.
Based on: Brooks (1986) - Subsumption Architecture

Services:
- MovementService: Basic motion commands
- ArmService: Grasp/deposit sequences
- VisionService: Stable cube tracking
- NavigationService: Movement + vision coordination
"""

from .movement_service import MovementService
from .arm_service import ArmService, GraspResult, ArmState
from .vision_service import VisionService, TrackedCube, VisionState
from .navigation_service import NavigationService, ApproachResult, ApproachPhase

__all__ = [
    'MovementService',
    'ArmService',
    'GraspResult',
    'ArmState',
    'VisionService',
    'TrackedCube',
    'VisionState',
    'NavigationService',
    'ApproachResult',
    'ApproachPhase',
]
