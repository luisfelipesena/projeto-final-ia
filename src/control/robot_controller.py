"""
RobotController Integration Layer

Integrates FuzzyController and StateMachine for complete robot control.
Connects perception data to fuzzy inference and state machine transitions.

Contract: Integration layer for Phase 6
"""

from typing import Optional
from .fuzzy_controller import FuzzyController, FuzzyInputs, FuzzyOutputs
from .state_machine import StateMachine, RobotState, StateTransitionConditions


class RobotController:
    """
    Integration layer connecting fuzzy controller and state machine

    This class coordinates:
    - Perception data → Fuzzy inputs conversion
    - Fuzzy inference → State machine transitions
    - Action commands → Robot actuator commands

    Implementation in Phase 6 (User Story 4)
    """

    def __init__(self):
        """Initialize robot controller with fuzzy controller and state machine"""
        self.fuzzy = FuzzyController()
        self.state_machine = StateMachine()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize fuzzy controller and state machine"""
        self.fuzzy.initialize()
        self._initialized = True

    def run(self, perception_data) -> None:
        """
        Main control loop

        Args:
            perception_data: PerceptionData from perception module

        Implementation in Phase 6
        """
        if not self._initialized:
            raise RuntimeError("Controller not initialized. Call initialize() first.")

        # Convert perception data to fuzzy inputs
        fuzzy_inputs = self._convert_perception_to_fuzzy(perception_data)

        # Perform fuzzy inference
        fuzzy_outputs = self.fuzzy.infer(fuzzy_inputs)

        # Update state machine
        conditions = self._create_transition_conditions(perception_data, fuzzy_inputs)
        self.state_machine.update(conditions)

        # Apply control commands (implementation in Phase 6)
        # self._apply_commands(fuzzy_outputs)

    def _convert_perception_to_fuzzy(self, perception_data) -> FuzzyInputs:
        """Convert PerceptionData to FuzzyInputs"""
        # Implementation in Phase 6
        return FuzzyInputs(
            distance_to_obstacle=0.0,
            angle_to_obstacle=0.0,
            distance_to_cube=999.0,
            angle_to_cube=0.0,
            cube_detected=False,
            holding_cube=False
        )

    def _create_transition_conditions(self, perception_data, fuzzy_inputs: FuzzyInputs) -> StateTransitionConditions:
        """Create state transition conditions from perception and fuzzy inputs"""
        # Implementation in Phase 6
        return StateTransitionConditions(
            cube_detected=fuzzy_inputs.cube_detected,
            cube_distance=fuzzy_inputs.distance_to_cube,
            cube_angle=fuzzy_inputs.angle_to_cube,
            obstacle_distance=fuzzy_inputs.distance_to_obstacle,
            holding_cube=fuzzy_inputs.holding_cube,
            at_target_box=False,
            grasp_success=False,
            deposit_complete=False
        )

