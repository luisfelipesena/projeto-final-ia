"""
Integration tests for fuzzy control system

Tests interaction between fuzzy controller, state machine, and perception mock
"""

import pytest
import numpy as np
from src.control.fuzzy_controller import FuzzyController, FuzzyInputs
from src.control.state_machine import StateMachine, RobotState
from tests.control.fixtures.perception_mock import MockPerceptionSystem


class TestObstacleAvoidanceIntegration:
    """T019: Integration tests for obstacle avoidance scenarios"""

    def test_obstacle_front_scenario(self):
        """Test obstacle directly ahead"""
        controller = FuzzyController({'logging': False})
        controller.initialize()

        perception = MockPerceptionSystem()
        scenario = perception.get_scenario('obstacle_front')

        inputs = FuzzyInputs(
            distance_to_obstacle=scenario.obstacle_map.min_distance,
            angle_to_obstacle=scenario.obstacle_map.min_angle,
            distance_to_cube=3.0,
            angle_to_cube=0.0,
            cube_detected=False,
            holding_cube=False
        )

        outputs = controller.infer(inputs)

        # Should avoid obstacle (turn or stop)
        assert outputs.linear_velocity <= 0.15, "Should slow down or stop"
        assert abs(outputs.angular_velocity) > 0.05, "Should turn away"

    def test_obstacle_critical_scenario(self):
        """Test critical obstacle (<0.3m)"""
        controller = FuzzyController({'logging': False})
        controller.initialize()

        perception = MockPerceptionSystem()
        scenario = perception.get_scenario('obstacle_critical')

        inputs = FuzzyInputs(
            distance_to_obstacle=scenario.obstacle_map.min_distance,
            angle_to_obstacle=scenario.obstacle_map.min_angle,
            distance_to_cube=3.0,
            angle_to_cube=0.0,
            cube_detected=False,
            holding_cube=False
        )

        outputs = controller.infer(inputs)

        # Emergency stop
        assert outputs.linear_velocity <= 0.05, "Should stop"
        assert abs(outputs.angular_velocity) > 0.1, "Should turn strongly"

    def test_clear_path_scenario(self):
        """Test clear path (no obstacles)"""
        controller = FuzzyController({'logging': False})
        controller.initialize()

        perception = MockPerceptionSystem()
        scenario = perception.get_scenario('clear_all')

        inputs = FuzzyInputs(
            distance_to_obstacle=scenario.obstacle_map.min_distance,
            angle_to_obstacle=scenario.obstacle_map.min_angle,
            distance_to_cube=3.0,
            angle_to_cube=0.0,
            cube_detected=False,
            holding_cube=False
        )

        outputs = controller.infer(inputs)

        # Should proceed forward
        assert outputs.linear_velocity > 0.1, "Should move forward"
        assert abs(outputs.angular_velocity) < 0.2, "Should go relatively straight"


class TestStateMachineIntegration:
    """Test state machine with fuzzy controller"""

    def test_state_machine_initialization(self):
        """Test state machine initializes correctly"""
        sm = StateMachine()
        assert sm.current_state == RobotState.SEARCHING

    def test_state_transitions(self):
        """Test basic state transitions"""
        sm = StateMachine()

        # Transition to APPROACHING
        sm.transition_to(RobotState.APPROACHING, {'cube_id': 'cube_1'})
        assert sm.current_state == RobotState.APPROACHING

        # Transition to GRASPING
        sm.transition_to(RobotState.GRASPING, {'cube_id': 'cube_1'})
        assert sm.current_state == RobotState.GRASPING

    def test_avoiding_override(self):
        """Test AVOIDING state override"""
        sm = StateMachine()
        sm.transition_to(RobotState.APPROACHING, {'cube_id': 'cube_1'})

        # Obstacle detected → override to AVOIDING
        sm.transition_to(RobotState.AVOIDING, {'obstacle_distance': 0.3})
        assert sm.current_state == RobotState.AVOIDING

        # After avoiding, should return to previous state
        sm.transition_to(RobotState.SEARCHING)
        assert sm.current_state == RobotState.SEARCHING


