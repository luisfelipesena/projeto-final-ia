"""
Manipulation Module

Provides grasping and depositing sequences for cube collection.

Components:
- grasping: Grasp sequence controller
- depositing: Deposit sequence controller
"""

from .grasping import GraspController, GraspState, GraspResult
from .depositing import DepositController, DepositState, DepositResult

__all__ = [
    'GraspController',
    'GraspState',
    'GraspResult',
    'DepositController',
    'DepositState',
    'DepositResult',
]
