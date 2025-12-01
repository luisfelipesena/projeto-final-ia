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
            'very_near': MembershipFunctionParams('very_near', 'trimf', (0.0, 0.4, 0.8)),
            'near': MembershipFunctionParams('near', 'trimf', (0.5, 1.0, 1.5)),
            'medium': MembershipFunctionParams('medium', 'trimf', (1.2, 2.0, 2.8)),
            'far': MembershipFunctionParams('far', 'trimf', (2.5, 3.5, 4.5)),
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
            'slow': MembershipFunctionParams('slow', 'trimf', (0.02, 0.06, 0.10)),
            'medium': MembershipFunctionParams('medium', 'trimf', (0.08, 0.14, 0.22)),
            'fast': MembershipFunctionParams('fast', 'trapmf', (0.18, 0.24, 0.30, 0.30)),
            'crawl': MembershipFunctionParams('crawl', 'trimf', (0.00, 0.02, 0.04)),
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

    # Additional boolean-like inputs for obstacle blocking
    vars_dict['front_blocked'] = LinguisticVariable(
        name='front_blocked',
        universe=(0.0, 1.0),
        membership_functions={
            'clear': MembershipFunctionParams('clear', 'trimf', (0.0, 0.0, 0.5)),
            'blocked': MembershipFunctionParams('blocked', 'trimf', (0.5, 1.0, 1.0)),
        },
        mf_type='triangular'
    )

    vars_dict['lateral_blocked'] = LinguisticVariable(
        name='lateral_blocked',
        universe=(0.0, 1.0),
        membership_functions={
            'clear': MembershipFunctionParams('clear', 'trimf', (0.0, 0.0, 0.5)),
            'blocked': MembershipFunctionParams('blocked', 'trimf', (0.5, 1.0, 1.0)),
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
    Total: 35-50 rules (15 safety + 20-25 task + 5-10 exploration)
    """
    rules: List[FuzzyRule] = []

    # ========================================================================
    # Safety Rules (R001-R015): Obstacle Avoidance - Highest Priority
    # Based on research.md Section 2.1: Minimum 15 safety-critical rules
    # ========================================================================

    # R001: Emergency Stop - Very Close obstacle ahead
    rules.append(FuzzyRule(
        rule_id='R001_emergency_stop',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'very_near'),
            FuzzyRuleCondition('angle_to_obstacle', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'stop'),
            FuzzyRuleAssignment('angular_velocity', 'strong_left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=10.0,
        category='safety'
    ))

    # R002: Very Close obstacle left → Turn right strongly
    rules.append(FuzzyRule(
        rule_id='R002_very_close_left',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'very_near'),
            FuzzyRuleCondition('angle_to_obstacle', 'negative_big')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'stop'),
            FuzzyRuleAssignment('angular_velocity', 'strong_right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=10.0,
        category='safety'
    ))

    # R003: Very Close obstacle right → Turn left strongly
    rules.append(FuzzyRule(
        rule_id='R003_very_close_right',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'very_near'),
            FuzzyRuleCondition('angle_to_obstacle', 'positive_big')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'stop'),
            FuzzyRuleAssignment('angular_velocity', 'strong_left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=10.0,
        category='safety'
    ))

    # R004: Close obstacle ahead → Slow + Turn right
    rules.append(FuzzyRule(
        rule_id='R004_close_ahead',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'near'),
            FuzzyRuleCondition('angle_to_obstacle', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=9.0,
        category='safety'
    ))

    # R005: Close obstacle left → Slow + Turn right
    rules.append(FuzzyRule(
        rule_id='R005_close_left',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'near'),
            FuzzyRuleCondition('angle_to_obstacle', 'negative_medium')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=9.0,
        category='safety'
    ))

    # R006: Close obstacle right → Slow + Turn left
    rules.append(FuzzyRule(
        rule_id='R006_close_right',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'near'),
            FuzzyRuleCondition('angle_to_obstacle', 'positive_medium')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=9.0,
        category='safety'
    ))

    # R007: Medium distance obstacle ahead → Medium speed + Slight turn
    rules.append(FuzzyRule(
        rule_id='R007_medium_ahead',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'medium'),
            FuzzyRuleCondition('angle_to_obstacle', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'medium'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=8.0,
        category='safety'
    ))

    # R008-R015: Additional safety rules covering all critical combinations
    # Very Close × All angles (complete coverage)
    for angle_mf in ['negative_small', 'positive_small', 'negative_medium', 'positive_medium']:
        turn_direction = 'strong_right' if 'negative' in angle_mf else 'strong_left'
        rules.append(FuzzyRule(
            rule_id=f'R008_very_close_{angle_mf}',
            antecedents=[
                FuzzyRuleCondition('distance_to_obstacle', 'very_near'),
                FuzzyRuleCondition('angle_to_obstacle', angle_mf)
            ],
            consequents=[
                FuzzyRuleAssignment('linear_velocity', 'stop'),
                FuzzyRuleAssignment('angular_velocity', turn_direction),
                FuzzyRuleAssignment('action', 'search')
            ],
            weight=10.0,
            category='safety'
        ))

    # Close × Side angles
    rules.append(FuzzyRule(
        rule_id='R012_close_negative_small',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'near'),
            FuzzyRuleCondition('angle_to_obstacle', 'negative_small')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=9.0,
        category='safety'
    ))

    rules.append(FuzzyRule(
        rule_id='R013_close_positive_small',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'near'),
            FuzzyRuleCondition('angle_to_obstacle', 'positive_small')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=9.0,
        category='safety'
    ))

    # R014: Far obstacle → Proceed normally
    rules.append(FuzzyRule(
        rule_id='R014_far_obstacle',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'medium'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=8.0,
        category='safety'
    ))

    # R015: Very Far obstacle → Fast forward
    rules.append(FuzzyRule(
        rule_id='R015_very_far_obstacle',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'very_far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'fast'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=8.0,
        category='safety'
    ))

    # ========================================================================
    # Task Rules (R016-R025): Cube Approach and Navigation
    # Basic implementation for Phase 3, will be expanded in Phase 4
    # ========================================================================

    # R016: Cube detected far ahead → Approach with medium speed
    rules.append(FuzzyRule(
        rule_id='R016_cube_far_ahead',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'far'),
            FuzzyRuleCondition('angle_to_cube', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'medium'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'approach')
        ],
        weight=6.0,
        category='task'
    ))

    # R017: Cube detected near ahead → Slow approach
    rules.append(FuzzyRule(
        rule_id='R017_cube_near_ahead',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'near'),
            FuzzyRuleCondition('angle_to_cube', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'approach')
        ],
        weight=6.0,
        category='task'
    ))

    # R018: Cube very near → Stop and prepare to grasp
    rules.append(FuzzyRule(
        rule_id='R018_cube_very_near',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'very_near'),
            FuzzyRuleCondition('angle_to_cube', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'stop'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'grasp')
        ],
        weight=7.0,
        category='task'
    ))

    # R019: Cube left → Turn left to align
    rules.append(FuzzyRule(
        rule_id='R019_cube_left',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'medium'),
            FuzzyRuleCondition('angle_to_cube', 'negative_medium')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'left'),
            FuzzyRuleAssignment('action', 'approach')
        ],
        weight=5.0,
        category='task'
    ))

    # R020: Cube right → Turn right to align
    rules.append(FuzzyRule(
        rule_id='R020_cube_right',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'medium'),
            FuzzyRuleCondition('angle_to_cube', 'positive_medium')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'approach')
        ],
        weight=5.0,
        category='task'
    ))

    # R027: Final approach creep when front clear
    rules.append(FuzzyRule(
        rule_id='R027_final_creep',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'near'),
            FuzzyRuleCondition('front_blocked', 'clear')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'approach')
        ],
        weight=7.5,
        category='task'
    ))

    # R028: Abort forward motion if cube near but front blocked
    rules.append(FuzzyRule(
        rule_id='R028_front_blocked_abort',
        antecedents=[
            FuzzyRuleCondition('distance_to_cube', 'near'),
            FuzzyRuleCondition('front_blocked', 'blocked')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'stop'),
            FuzzyRuleAssignment('angular_velocity', 'strong_left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=9.5,
        category='safety'
    ))

    # R029: Lateral blockage triggers re-orientation
    rules.append(FuzzyRule(
        rule_id='R029_lateral_clearance',
        antecedents=[
            FuzzyRuleCondition('lateral_blocked', 'blocked'),
            FuzzyRuleCondition('distance_to_obstacle', 'near')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=7.0,
        category='safety'
    ))

    # ========================================================================
    # Exploration Rules (R021-R025): Search Patterns
    # Basic implementation for Phase 3, will be expanded in Phase 4
    # ========================================================================

    # R021: No obstacles, no cube → Search forward
    rules.append(FuzzyRule(
        rule_id='R021_search_forward',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'very_far'),
            FuzzyRuleCondition('distance_to_cube', 'very_far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'medium'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=2.0,
        category='exploration'
    ))

    # R022: Search with slight left rotation
    rules.append(FuzzyRule(
        rule_id='R022_search_left',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'far'),
            FuzzyRuleCondition('distance_to_cube', 'very_far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=1.5,
        category='exploration'
    ))

    # R023: Search with slight right rotation
    rules.append(FuzzyRule(
        rule_id='R023_search_right',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'far'),
            FuzzyRuleCondition('distance_to_cube', 'very_far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'slow'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=1.5,
        category='exploration'
    ))

    # R024: Clear path → Fast forward
    rules.append(FuzzyRule(
        rule_id='R024_clear_path',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'very_far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'fast'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=2.0,
        category='exploration'
    ))

    # R025: Medium distance obstacle, safe → Continue forward
    rules.append(FuzzyRule(
        rule_id='R025_safe_continue',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'medium'),
            FuzzyRuleCondition('angle_to_obstacle', 'zero')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'medium'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=3.0,
        category='exploration'
    ))

    # R026: Far obstacle → Fast forward (coverage gap fix)
    rules.append(FuzzyRule(
        rule_id='R026_far_obstacle',
        antecedents=[
            FuzzyRuleCondition('distance_to_obstacle', 'far')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'fast'),
            FuzzyRuleAssignment('angular_velocity', 'straight'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=2.0,
        category='exploration'
    ))

    # Safety rule: front blocked -> stop
    rules.append(FuzzyRule(
        rule_id='R030_front_block_stop',
        antecedents=[
            FuzzyRuleCondition('front_blocked', 'blocked')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'stop'),
            FuzzyRuleAssignment('angular_velocity', 'strong_left'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=10.0,
        category='safety'
    ))

    # Safety rule: lateral blocked -> crawl
    rules.append(FuzzyRule(
        rule_id='R031_lateral_crawl',
        antecedents=[
            FuzzyRuleCondition('lateral_blocked', 'blocked'),
            FuzzyRuleCondition('distance_to_obstacle', 'medium')
        ],
        consequents=[
            FuzzyRuleAssignment('linear_velocity', 'crawl'),
            FuzzyRuleAssignment('angular_velocity', 'right'),
            FuzzyRuleAssignment('action', 'search')
        ],
        weight=6.5,
        category='safety'
    ))

    return rules

