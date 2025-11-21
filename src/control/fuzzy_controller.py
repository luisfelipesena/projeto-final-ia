"""
FuzzyController Implementation

Mamdani fuzzy inference system for robot control.
Based on: Zadeh (1965), Mamdani & Assilian (1975), Saffiotti (1997)

Contract: specs/004-fuzzy-control/contracts/fuzzy_controller.py
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# ============================================================================
# Data Structures (Phase 2: Foundational)
# ============================================================================

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
    membership_functions: Dict[str, MembershipFunctionParams] = field(default_factory=dict)
    mf_type: str = 'triangular'  # 'triangular' | 'trapezoidal' | 'gaussian'


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
    weight: float = 1.0  # 1.0-10.0 (10.0 = highest priority)
    category: str = 'task'  # 'safety' | 'task' | 'exploration'


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
    active_rules: List[str] = field(default_factory=list)  # Rule IDs that fired this cycle


# ============================================================================
# FuzzyController Class (Phase 3+ implementation)
# ============================================================================

class FuzzyController:
    """
    Mamdani fuzzy inference system for robot control

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
        self.config = config or {}
        self.defuzz_method = self.config.get('defuzz_method', 'centroid')
        self.enable_cache = self.config.get('enable_cache', True)
        self.logging_enabled = self.config.get('logging', False)

        # Will be initialized in initialize()
        self.linguistic_vars: Dict[str, LinguisticVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.control_system: Optional[ctrl.ControlSystem] = None
        self.control_sim: Optional[ctrl.ControlSystemSimulation] = None
        self._initialized = False

        # Setup logging if enabled
        if self.logging_enabled:
            self.logger = logging.getLogger('fuzzy_controller')
            handler = logging.FileHandler('logs/fuzzy_decisions.log')
            formatter = logging.Formatter('[%(asctime)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None

    def initialize(self) -> None:
        """
        Load linguistic variables, membership functions, and rules

        Must be called before infer(). Loads default rules from fuzzy_rules.py
        or custom rules if provided in config.

        Raises:
            ValueError: If MF overlap validation fails
            RuntimeError: If rule count < 20 (FR-005)
        """
        # Import rules module (will be implemented in Phase 3)
        from .fuzzy_rules import create_linguistic_variables, create_rules

        # Create linguistic variables
        self.linguistic_vars = create_linguistic_variables()

        # Create rules
        self.rules = create_rules()

        # Validate minimum rule count (FR-005)
        if len(self.rules) < 20:
            raise RuntimeError(f"Rule count {len(self.rules)} < 20 (FR-005 requirement)")

        # Validate membership function overlaps (will be implemented in Phase 3)
        # self._validate_mf_overlaps()

        # Build scikit-fuzzy control system (will be implemented in Phase 3)
        # self._build_control_system()

        self._initialized = True

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
        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call initialize() first.")

        start_time = time.time()

        # Validate input ranges
        self._validate_inputs(inputs)

        # Perform inference (implementation in Phase 3)
        # For now, return default outputs
        outputs = FuzzyOutputs(
            linear_velocity=0.0,
            angular_velocity=0.0,
            action='search',
            confidence=0.0,
            active_rules=[]
        )

        # Check performance (FR-008)
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 50:
            raise RuntimeError(f"Inference time {elapsed_ms:.2f}ms exceeds 50ms limit (FR-008)")

        return outputs

    def add_rule(self, rule: FuzzyRule) -> None:
        """
        Add custom fuzzy rule to rule base

        Args:
            rule: FuzzyRule instance with antecedents and consequents

        Raises:
            ValueError: If rule references undefined variables/MFs
        """
        # Validate rule references
        self._validate_rule(rule)
        self.rules.append(rule)

    def get_linguistic_variables(self) -> Dict[str, LinguisticVariable]:
        """
        Get all defined linguistic variables (inputs + outputs)

        Returns:
            Dict mapping variable name → LinguisticVariable
        """
        return self.linguistic_vars.copy()

    def get_rules(self, category: Optional[str] = None) -> List[FuzzyRule]:
        """
        Get fuzzy rules, optionally filtered by category

        Args:
            category: Optional filter ('safety' | 'task' | 'exploration')

        Returns:
            List of FuzzyRule instances
        """
        if category:
            return [r for r in self.rules if r.category == category]
        return self.rules.copy()

    def visualize_membership_functions(self, variable_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot membership functions for a linguistic variable

        Args:
            variable_name: Name of variable to plot (e.g., 'distance_to_obstacle')
            save_path: Optional path to save plot (e.g., 'docs/mfs_distance.png')
        """
        # Implementation in Phase 7 (Polish)
        pass

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _validate_inputs(self, inputs: FuzzyInputs) -> None:
        """Validate input ranges match universe definitions"""
        if inputs.distance_to_obstacle < 0.0 or inputs.distance_to_obstacle > 5.0:
            raise ValueError(f"distance_to_obstacle {inputs.distance_to_obstacle} outside [0.0, 5.0]")
        if inputs.angle_to_obstacle < -135.0 or inputs.angle_to_obstacle > 135.0:
            raise ValueError(f"angle_to_obstacle {inputs.angle_to_obstacle} outside [-135, 135]")
        if inputs.distance_to_cube < 0.0 or inputs.distance_to_cube > 3.0:
            raise ValueError(f"distance_to_cube {inputs.distance_to_cube} outside [0.0, 3.0]")
        if inputs.angle_to_cube < -135.0 or inputs.angle_to_cube > 135.0:
            raise ValueError(f"angle_to_cube {inputs.angle_to_cube} outside [-135, 135]")

    def _validate_rule(self, rule: FuzzyRule) -> None:
        """Validate rule references valid variables and membership functions"""
        for condition in rule.antecedents:
            if condition.variable not in self.linguistic_vars:
                raise ValueError(f"Rule {rule.rule_id} references undefined variable: {condition.variable}")
            var = self.linguistic_vars[condition.variable]
            if condition.membership_function not in var.membership_functions:
                raise ValueError(
                    f"Rule {rule.rule_id} references undefined MF '{condition.membership_function}' "
                    f"in variable '{condition.variable}'"
                )

