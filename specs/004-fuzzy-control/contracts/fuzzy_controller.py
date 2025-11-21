"""
FuzzyController Interface Contract

Purpose: Define interface for Mamdani fuzzy inference system used in Phase 3.
This contract enables independent development and testing with mock implementations.

Based on: specs/004-fuzzy-control/data-model.md
Scientific Foundation: Mamdani & Assilian (1975), Zadeh (1965)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class MembershipFunctionParams:
    """Parameters defining a membership function shape"""
    label: str  # e.g., 'very_near', 'near', 'medium', 'far'
    shape: str  # 'trimf' | 'trapmf' | 'gaussmf'
    params: Tuple[float, ...]  # Shape-specific parameters


@dataclass
class LinguisticVariable:
    """Fuzzy linguistic variable with membership functions"""
    name: str  # e.g., 'distance_to_obstacle'
    universe: Tuple[float, float]  # (min, max) range in SI units
    membership_functions: Dict[str, MembershipFunctionParams]
    mf_type: str  # 'triangular' | 'trapezoidal' | 'gaussian'


@dataclass
class FuzzyRuleCondition:
    """Single antecedent condition in fuzzy rule"""
    variable: str  # LinguisticVariable name
    membership_function: str  # MF label within that variable


@dataclass
class FuzzyRuleAssignment:
    """Single consequent assignment in fuzzy rule"""
    variable: str  # Output LinguisticVariable name
    membership_function: str  # MF label to assign


@dataclass
class FuzzyRule:
    """
    Complete fuzzy IF-THEN rule with priority weight

    Example:
        IF distance_to_obstacle IS very_near
        THEN linear_velocity IS stop AND angular_velocity IS strong_left
    """
    rule_id: str  # e.g., 'R001_emergency_stop'
    antecedents: List[FuzzyRuleCondition]  # IF conditions (AND logic)
    consequents: List[FuzzyRuleAssignment]  # THEN assignments
    weight: float  # 1.0-10.0 (10.0 = highest priority)
    category: str  # 'safety' | 'task' | 'exploration'


@dataclass
class FuzzyInputs:
    """Crisp input values to fuzzy controller"""
    distance_to_obstacle: float  # meters (0.0-5.0)
    angle_to_obstacle: float  # degrees (-135 to +135)
    distance_to_cube: float  # meters (0.0-3.0)
    angle_to_cube: float  # degrees (-135 to +135)
    cube_detected: bool  # True if cube visible in camera
    holding_cube: bool  # True if gripper has cube


@dataclass
class FuzzyOutputs:
    """Crisp output values from fuzzy controller"""
    linear_velocity: float  # m/s (-0.3 to 0.3)
    angular_velocity: float  # rad/s (-0.5 to 0.5)
    action: str  # 'search' | 'approach' | 'grasp' | 'navigate' | 'deposit'
    confidence: float  # [0, 1] - aggregated rule strength
    active_rules: List[str]  # Rule IDs that fired this cycle


class FuzzyController:
    """
    Mamdani fuzzy inference system interface

    Contract Requirements:
    - MUST execute inference in <50ms (FR-008)
    - MUST use centroid defuzzification (FR-006)
    - MUST prioritize safety rules (weight >= 8.0) over all others (FR-007)
    - MUST implement minimum 20 rules (FR-005)
    - MUST validate all membership functions overlap 50% (±20% tuning range)

    Usage:
        controller = FuzzyController()
        controller.initialize()  # Load rules and MFs

        inputs = FuzzyInputs(
            distance_to_obstacle=0.4,
            angle_to_obstacle=15.0,
            distance_to_cube=1.2,
            angle_to_cube=-30.0,
            cube_detected=True,
            holding_cube=False
        )

        outputs = controller.infer(inputs)
        print(f"Linear: {outputs.linear_velocity}, Angular: {outputs.angular_velocity}")
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize fuzzy controller

        Args:
            config: Optional configuration dict with:
                - 'defuzz_method': str = 'centroid'
                - 'enable_cache': bool = True
                - 'logging': bool = False
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def initialize(self) -> None:
        """
        Load linguistic variables, membership functions, and rules

        Must be called before infer(). Loads default rules from fuzzy_rules.py
        or custom rules if provided in config.

        Raises:
            ValueError: If MF overlap validation fails
            RuntimeError: If rule count < 20 (FR-005)
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def infer(self, inputs: FuzzyInputs) -> FuzzyOutputs:
        """
        Perform Mamdani fuzzy inference

        Steps:
        1. Fuzzification: Map crisp inputs to membership values
        2. Rule Evaluation: Compute activation levels for all rules
        3. Aggregation: Combine consequents weighted by activations
        4. Defuzzification: Convert fuzzy output to crisp values

        Args:
            inputs: Crisp sensor values

        Returns:
            FuzzyOutputs with velocities, action, confidence, active rules

        Raises:
            ValueError: If inputs outside universe ranges
            RuntimeError: If inference exceeds 50ms (FR-008)

        Performance:
        - Target: <50ms per call
        - Typical: 10-30ms with caching
        - Worst case: 45ms (all rules fire)
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def add_rule(self, rule: FuzzyRule) -> None:
        """
        Add custom fuzzy rule to rule base

        Args:
            rule: FuzzyRule instance with antecedents and consequents

        Raises:
            ValueError: If rule references undefined variables/MFs
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_linguistic_variables(self) -> Dict[str, LinguisticVariable]:
        """
        Get all defined linguistic variables (inputs + outputs)

        Returns:
            Dict mapping variable name → LinguisticVariable
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_rules(self, category: Optional[str] = None) -> List[FuzzyRule]:
        """
        Get fuzzy rules, optionally filtered by category

        Args:
            category: Optional filter ('safety' | 'task' | 'exploration')

        Returns:
            List of FuzzyRule instances
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def visualize_membership_functions(self, variable_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot membership functions for a linguistic variable

        Args:
            variable_name: Name of variable to plot (e.g., 'distance_to_obstacle')
            save_path: Optional path to save plot (e.g., 'docs/mfs_distance.png')
        """
        raise NotImplementedError("Must be implemented by concrete class")
