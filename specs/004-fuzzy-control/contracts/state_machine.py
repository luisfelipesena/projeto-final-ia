"""
StateMachine Interface Contract

Purpose: Define interface for robot operational state management in Phase 3.
Coordinates task execution flow: SEARCHING → APPROACHING → GRASPING → NAVIGATING → DEPOSITING

Based on: specs/004-fuzzy-control/data-model.md
Scientific Foundation: Thrun et al. (2005) - Probabilistic Robotics
"""

from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import time


class RobotState(Enum):
    """Robot operational states"""
    SEARCHING = 1           # Looking for cubes (exploration pattern)
    APPROACHING = 2         # Moving toward detected cube
    GRASPING = 3           # Executing grasp sequence
    NAVIGATING_TO_BOX = 4  # Moving toward deposit box with cube
    DEPOSITING = 5         # Executing deposit sequence
    AVOIDING = 6           # Override state for collision risk


@dataclass
class StateTransitionConditions:
    """Sensor-based conditions for state transitions"""
    cube_detected: bool
    cube_distance: float  # meters
    cube_angle: float  # degrees
    obstacle_distance: float  # meters (minimum across all sectors)
    holding_cube: bool
    at_target_box: bool
    grasp_success: bool
    deposit_complete: bool


@dataclass
class StateMetrics:
    """Performance metrics for state tracking"""
    state: RobotState
    entry_time: float  # Unix timestamp when state entered
    duration: float  # Seconds in current state
    transition_count: int  # Total transitions since start
    timeout_triggered: bool


class StateMachine:
    """
    Finite state machine for robot task coordination

    Contract Requirements:
    - MUST implement 6 states (FR-009)
    - MUST allow AVOIDING to override any state when obstacle <0.3m (FR-011)
    - MUST track cube color for correct box navigation (FR-012)
    - MUST return to SEARCHING after deposit or failed grasp (FR-013)
    - MUST implement 2-minute timeout per state (FR-022)

    State Transition Rules:
        SEARCHING → APPROACHING: cube_detected=True
        SEARCHING → AVOIDING: obstacle_distance < 0.3m

        APPROACHING → GRASPING: cube_distance < 0.15m AND |cube_angle| < 5°
        APPROACHING → SEARCHING: cube_detected=False (lost detection)
        APPROACHING → AVOIDING: obstacle_distance < 0.3m

        GRASPING → NAVIGATING_TO_BOX: grasp_success=True
        GRASPING → SEARCHING: grasp_attempts >= 3

        NAVIGATING_TO_BOX → DEPOSITING: at_target_box=True
        NAVIGATING_TO_BOX → AVOIDING: obstacle_distance < 0.3m

        DEPOSITING → SEARCHING: deposit_complete=True

        AVOIDING → previous_state: obstacle_distance > 0.5m

    Usage:
        sm = StateMachine(initial_state=RobotState.SEARCHING)

        # In control loop
        conditions = StateTransitionConditions(
            cube_detected=True,
            cube_distance=0.8,
            cube_angle=15.0,
            obstacle_distance=1.5,
            holding_cube=False,
            at_target_box=False,
            grasp_success=False,
            deposit_complete=False
        )

        new_state = sm.update(conditions)
        if new_state != sm.current_state:
            print(f"Transition: {sm.current_state} → {new_state}")
    """

    def __init__(self, initial_state: RobotState = RobotState.SEARCHING):
        """
        Initialize state machine

        Args:
            initial_state: Starting state (default: SEARCHING)
        """
        raise NotImplementedError("Must be implemented by concrete class")

    @property
    def current_state(self) -> RobotState:
        """Get current active state"""
        raise NotImplementedError("Must be implemented by concrete class")

    @property
    def previous_state(self) -> Optional[RobotState]:
        """Get previous state (for AVOIDING recovery)"""
        raise NotImplementedError("Must be implemented by concrete class")

    def update(self, conditions: StateTransitionConditions) -> RobotState:
        """
        Evaluate transition conditions and update state if needed

        Priority order:
        1. AVOIDING override (obstacle_distance < 0.3m) - FR-011
        2. Timeout check (>120s in state) - FR-022
        3. Normal transition rules per current state

        Args:
            conditions: Current sensor readings and flags

        Returns:
            New active state (may be same as current)

        Side Effects:
            - Updates internal state tracking
            - Logs state transitions
            - Resets state timers on transition
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def force_transition(self, target_state: RobotState, reason: str = "") -> None:
        """
        Manually force transition to target state (for testing/recovery)

        Args:
            target_state: Desired state to enter
            reason: Optional explanation for manual transition

        Raises:
            ValueError: If target_state not in allowed transitions
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def reset(self) -> None:
        """
        Reset state machine to initial state

        Clears:
        - Current/previous state → SEARCHING
        - Cube tracking data
        - Grasp attempt counter
        - State timers
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_metrics(self) -> StateMetrics:
        """
        Get performance metrics for current state

        Returns:
            StateMetrics with timing and transition data
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def set_target_cube_color(self, color: str) -> None:
        """
        Set color of cube being tracked (for box navigation)

        Args:
            color: 'green' | 'blue' | 'red'

        Raises:
            ValueError: If color not in valid set
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_target_cube_color(self) -> Optional[str]:
        """
        Get color of currently tracked cube

        Returns:
            Color string or None if no cube held
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def increment_grasp_attempt(self) -> int:
        """
        Increment failed grasp counter

        Returns:
            Current grasp attempt count (resets after successful deposit)

        Raises:
            RuntimeError: If called when not in GRASPING state
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_cubes_collected(self) -> int:
        """
        Get total cubes successfully deposited this session

        Returns:
            Cube count (0-15)
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def register_state_callback(self, state: RobotState, callback: Callable[[RobotState, RobotState], None]) -> None:
        """
        Register callback for state entry/exit

        Args:
            state: State to monitor
            callback: Function(from_state, to_state) called on transitions

        Example:
            def on_grasping(from_state, to_state):
                print(f"Entered GRASPING from {from_state}")

            sm.register_state_callback(RobotState.GRASPING, on_grasping)
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_allowed_transitions(self) -> Dict[RobotState, List[RobotState]]:
        """
        Get complete transition table

        Returns:
            Dict mapping state → list of allowed next states
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def is_timeout_exceeded(self) -> bool:
        """
        Check if current state has exceeded 120s timeout

        Returns:
            True if timeout exceeded (FR-022)
        """
        raise NotImplementedError("Must be implemented by concrete class")
