"""
Type Definitions for Fuzzy Control System

Provides type aliases for compatibility between specs/004 and specs/005 naming conventions.

Existing (specs/004): FuzzyInputs, FuzzyOutputs
New (specs/005): PerceptionInput, ControlOutput

This module provides both for backward compatibility.
"""

# Import existing types from fuzzy_controller
from .fuzzy_controller import FuzzyInputs, FuzzyOutputs
from .state_machine import RobotState

# Create type aliases for specs/005 compatibility
PerceptionInput = FuzzyInputs
ControlOutput = FuzzyOutputs

__all__ = [
    # specs/004 naming
    'FuzzyInputs',
    'FuzzyOutputs',
    # specs/005 naming (aliases)
    'PerceptionInput',
    'ControlOutput',
    # Common
    'RobotState',
]
