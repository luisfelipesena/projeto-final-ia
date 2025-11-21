"""
Fuzzy Logic Control System for YouBot Autonomous Navigation

This module implements Mamdani fuzzy inference system for robot control,
including obstacle avoidance, cube approach, and state machine coordination.

Based on: Zadeh (1965), Mamdani & Assilian (1975), Saffiotti (1997)
"""

__version__ = "0.1.0"

# Core exports
from .fuzzy_controller import FuzzyController, FuzzyInputs, FuzzyOutputs
from .state_machine import StateMachine, RobotState, StateTransitionConditions
from .robot_controller import RobotController

__all__ = [
    "FuzzyController",
    "FuzzyInputs",
    "FuzzyOutputs",
    "StateMachine",
    "RobotState",
    "StateTransitionConditions",
    "RobotController",
]

