"""
Fuzzy Logic Control System for YouBot Autonomous Navigation

This module implements Mamdani fuzzy inference system for robot control,
including obstacle avoidance and cube approach decisions.

Based on: Zadeh (1965), Mamdani & Assilian (1975), Saffiotti (1997)
"""

__version__ = "0.2.0"

# Core exports - only fuzzy controller remains after v2 cleanup
from .fuzzy_controller import FuzzyController, FuzzyInputs, FuzzyOutputs

__all__ = [
    "FuzzyController",
    "FuzzyInputs",
    "FuzzyOutputs",
]
