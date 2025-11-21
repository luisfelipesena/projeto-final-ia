"""
Fuzzy Rules Database

Defines linguistic variables, membership functions, and fuzzy rules
for robot control system.

Based on: research.md membership function specifications
"""

from typing import Dict, List
from .fuzzy_controller import (
    LinguisticVariable,
    MembershipFunctionParams,
    FuzzyRule,
    FuzzyRuleCondition,
    FuzzyRuleAssignment
)


def create_linguistic_variables() -> Dict[str, LinguisticVariable]:
    """
    Create all linguistic variables with membership functions

    Returns:
        Dict mapping variable name → LinguisticVariable

    Membership Functions per Variable (from research.md):
    - distance_to_obstacle: 5 MFs {very_near: 0-0.5m, near: 0.3-1.0m, medium: 0.8-2.0m, far: 1.5-5.0m}
    - angle_to_obstacle: 7 MFs covering -135° to +135°
    - distance_to_cube: 5 MFs {very_near: 0-0.2m, near: 0.15-0.5m, medium: 0.4-1.5m, far: 1.0-3.0m}
    - angle_to_cube: 7 MFs {strong_left: -135° to -60°, left: -90° to -20°, ...}
    - linear_velocity: 4 MFs {stop: 0, slow: 0.05-0.15 m/s, medium: 0.1-0.25 m/s, fast: 0.2-0.3 m/s}
    - angular_velocity: 5 MFs {strong_left: 0.3-0.5 rad/s, left: 0.1-0.3, straight: -0.05-0.05, ...}
    """
    vars_dict: Dict[str, LinguisticVariable] = {}

    # Input: distance_to_obstacle [0.0, 5.0] meters
    vars_dict['distance_to_obstacle'] = LinguisticVariable(
        name='distance_to_obstacle',
        universe=(0.0, 5.0),
        membership_functions={
            'very_near': MembershipFunctionParams('very_near', 'trimf', (0.0, 0.3, 0.6)),
            'near': MembershipFunctionParams('near', 'trimf', (0.4, 0.8, 1.2)),
            'medium': MembershipFunctionParams('medium', 'trimf', (1.0, 1.8, 2.6)),
            'far': MembershipFunctionParams('far', 'trimf', (2.2, 3.5, 4.3)),
            'very_far': MembershipFunctionParams('very_far', 'trapmf', (4.0, 5.0, 5.0, 5.0)),
        },
        mf_type='triangular'
    )

    # Input: angle_to_obstacle [-135, 135] degrees
    vars_dict['angle_to_obstacle'] = LinguisticVariable(
        name='angle_to_obstacle',
        universe=(-135.0, 135.0),
        membership_functions={
            'negative_big': MembershipFunctionParams('negative_big', 'trapmf', (-135.0, -135.0, -90.0, -45.0)),
            'negative_medium': MembershipFunctionParams('negative_medium', 'trimf', (-90.0, -60.0, -30.0)),
            'negative_small': MembershipFunctionParams('negative_small', 'trimf', (-45.0, -15.0, 0.0)),
            'zero': MembershipFunctionParams('zero', 'trimf', (-15.0, 0.0, 15.0)),
            'positive_small': MembershipFunctionParams('positive_small', 'trimf', (0.0, 15.0, 45.0)),
            'positive_medium': MembershipFunctionParams('positive_medium', 'trimf', (30.0, 60.0, 90.0)),
            'positive_big': MembershipFunctionParams('positive_big', 'trapmf', (45.0, 90.0, 135.0, 135.0)),
        },
        mf_type='triangular'
    )

    # Input: distance_to_cube [0.0, 3.0] meters
    vars_dict['distance_to_cube'] = LinguisticVariable(
        name='distance_to_cube',
        universe=(0.0, 3.0),
        membership_functions={
            'very_near': MembershipFunctionParams('very_near', 'trimf', (0.0, 0.15, 0.3)),
            'near': MembershipFunctionParams('near', 'trimf', (0.2, 0.4, 0.6)),
            'medium': MembershipFunctionParams('medium', 'trimf', (0.5, 1.0, 1.5)),
            'far': MembershipFunctionParams('far', 'trimf', (1.2, 2.0, 3.0)),
            'very_far': MembershipFunctionParams('very_far', 'trapmf', (2.5, 3.0, 3.0, 3.0)),
        },
        mf_type='triangular'
    )

    # Input: angle_to_cube [-135, 135] degrees
    vars_dict['angle_to_cube'] = LinguisticVariable(
        name='angle_to_cube',
        universe=(-135.0, 135.0),
        membership_functions={
            'negative_big': MembershipFunctionParams('negative_big', 'trapmf', (-135.0, -135.0, -90.0, -45.0)),
            'negative_medium': MembershipFunctionParams('negative_medium', 'trimf', (-90.0, -60.0, -30.0)),
            'negative_small': MembershipFunctionParams('negative_small', 'trimf', (-45.0, -15.0, 0.0)),
            'zero': MembershipFunctionParams('zero', 'trimf', (-15.0, 0.0, 15.0)),
            'positive_small': MembershipFunctionParams('positive_small', 'trimf', (0.0, 15.0, 45.0)),
            'positive_medium': MembershipFunctionParams('positive_medium', 'trimf', (30.0, 60.0, 90.0)),
            'positive_big': MembershipFunctionParams('positive_big', 'trapmf', (45.0, 90.0, 135.0, 135.0)),
        },
        mf_type='triangular'
    )

    # Output: linear_velocity [0.0, 0.3] m/s
    vars_dict['linear_velocity'] = LinguisticVariable(
        name='linear_velocity',
        universe=(0.0, 0.3),
        membership_functions={
            'stop': MembershipFunctionParams('stop', 'trimf', (0.0, 0.0, 0.05)),
            'slow': MembershipFunctionParams('slow', 'trimf', (0.03, 0.08, 0.13)),
            'medium': MembershipFunctionParams('medium', 'trimf', (0.10, 0.18, 0.25)),
            'fast': MembershipFunctionParams('fast', 'trapmf', (0.20, 0.30, 0.30, 0.30)),
        },
        mf_type='triangular'
    )

    # Output: angular_velocity [-0.5, 0.5] rad/s
    vars_dict['angular_velocity'] = LinguisticVariable(
        name='angular_velocity',
        universe=(-0.5, 0.5),
        membership_functions={
            'strong_left': MembershipFunctionParams('strong_left', 'trapmf', (-0.5, -0.5, -0.3, -0.15)),
            'left': MembershipFunctionParams('left', 'trimf', (-0.3, -0.15, 0.0)),
            'straight': MembershipFunctionParams('straight', 'trimf', (-0.1, 0.0, 0.1)),
            'right': MembershipFunctionParams('right', 'trimf', (0.0, 0.15, 0.3)),
            'strong_right': MembershipFunctionParams('strong_right', 'trapmf', (0.15, 0.3, 0.5, 0.5)),
        },
        mf_type='triangular'
    )

    # Output: action (discrete)
    vars_dict['action'] = LinguisticVariable(
        name='action',
        universe=(0.0, 4.0),  # 0=search, 1=approach, 2=grasp, 3=navigate, 4=deposit
        membership_functions={
            'search': MembershipFunctionParams('search', 'trimf', (0.0, 0.0, 1.0)),
            'approach': MembershipFunctionParams('approach', 'trimf', (0.5, 1.0, 1.5)),
            'grasp': MembershipFunctionParams('grasp', 'trimf', (1.0, 2.0, 3.0)),
            'navigate': MembershipFunctionParams('navigate', 'trimf', (2.5, 3.0, 3.5)),
            'deposit': MembershipFunctionParams('deposit', 'trimf', (3.0, 4.0, 4.0)),
        },
        mf_type='triangular'
    )

    return vars_dict


def create_rules() -> List[FuzzyRule]:
    """
    Create fuzzy rule base

    Returns:
        List of FuzzyRule instances

    Rule categories:
    - Safety (weight 8.0-10.0): Obstacle avoidance, emergency stops
    - Task (weight 5.0-7.0): Cube approach, navigation to box
    - Exploration (weight 1.0-3.0): Search patterns

    Minimum 20 rules required (FR-005)
    """
    rules: List[FuzzyRule] = []

    # Safety rules (R001-R015) will be implemented in Phase 3
    # Task rules (R016-R025) will be implemented in Phase 4
    # Exploration rules will be implemented in Phase 4

    # Placeholder: Return empty list for now
    # Full rule implementation in Phase 3-4
    return rules

