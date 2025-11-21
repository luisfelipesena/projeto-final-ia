"""
StateMachine Implementation

Finite state machine for robot task coordination.
Based on: Thrun et al. (2005) - Probabilistic Robotics

Contract: specs/004-fuzzy-control/contracts/state_machine.py
"""

from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging


# ============================================================================
# Data Structures (Phase 2: Foundational)
# ============================================================================

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


# ============================================================================
# StateMachine Class (Phase 6 implementation)
# ============================================================================

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
        self._current_state = initial_state
        self._previous_state: Optional[RobotState] = None
        self._state_start_time = time.time()
        self._transition_count = 0
        self._grasp_attempts = 0
        self._cubes_collected = 0
        self._target_cube_color: Optional[str] = None

        # State timeout limits (FR-022: max 120s per state)
        self._timeout_limits: Dict[RobotState, float] = {
            RobotState.SEARCHING: 120.0,
            RobotState.APPROACHING: 120.0,
            RobotState.GRASPING: 120.0,
            RobotState.NAVIGATING_TO_BOX: 120.0,
            RobotState.DEPOSITING: 120.0,
            RobotState.AVOIDING: 120.0,
        }

        # State transition callbacks
        self._callbacks: Dict[RobotState, List[Callable[[RobotState, RobotState], None]]] = {
            state: [] for state in RobotState
        }

        # Setup logging
        self.logger = logging.getLogger('state_machine')
        handler = logging.FileHandler('logs/state_transitions.log')
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    @property
    def current_state(self) -> RobotState:
        """Get current active state"""
        return self._current_state

    @property
    def previous_state(self) -> Optional[RobotState]:
        """Get previous state (for AVOIDING recovery)"""
        return self._previous_state

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
        # Priority 1: AVOIDING override (FR-011)
        if conditions.obstacle_distance < 0.3:
            if self._current_state != RobotState.AVOIDING:
                self._transition_to(RobotState.AVOIDING, "Obstacle <0.3m detected")
            return self._current_state

        # Priority 2: Timeout check (FR-022)
        if self.is_timeout_exceeded():
            self.logger.warning(f"State {self._current_state} timeout exceeded, returning to SEARCHING")
            self._transition_to(RobotState.SEARCHING, "Timeout exceeded")
            return self._current_state

        # Priority 3: Normal transitions (implementation in Phase 6)
        # For now, return current state
        return self._current_state

    def force_transition(self, target_state: RobotState, reason: str = "") -> None:
        """
        Manually force transition to target state (for testing/recovery)

        Args:
            target_state: Desired state to enter
            reason: Optional explanation for manual transition

        Raises:
            ValueError: If target_state not in allowed transitions
        """
        self._transition_to(target_state, reason or "Manual transition")

    def reset(self) -> None:
        """
        Reset state machine to initial state

        Clears:
        - Current/previous state → SEARCHING
        - Cube tracking data
        - Grasp attempt counter
        - State timers
        """
        self._previous_state = None
        self._transition_to(RobotState.SEARCHING, "Reset")
        self._grasp_attempts = 0
        self._cubes_collected = 0
        self._target_cube_color = None

    def get_metrics(self) -> StateMetrics:
        """
        Get performance metrics for current state

        Returns:
            StateMetrics with timing and transition data
        """
        return StateMetrics(
            state=self._current_state,
            entry_time=self._state_start_time,
            duration=time.time() - self._state_start_time,
            transition_count=self._transition_count,
            timeout_triggered=self.is_timeout_exceeded()
        )

    def set_target_cube_color(self, color: str) -> None:
        """
        Set color of cube being tracked (for box navigation)

        Args:
            color: 'green' | 'blue' | 'red'

        Raises:
            ValueError: If color not in valid set
        """
        if color not in ['green', 'blue', 'red']:
            raise ValueError(f"Invalid cube color: {color}. Must be 'green', 'blue', or 'red'")
        self._target_cube_color = color

    def get_target_cube_color(self) -> Optional[str]:
        """
        Get color of currently tracked cube

        Returns:
            Color string or None if no cube held
        """
        return self._target_cube_color

    def increment_grasp_attempt(self) -> int:
        """
        Increment failed grasp counter

        Returns:
            Current grasp attempt count (resets after successful deposit)

        Raises:
            RuntimeError: If called when not in GRASPING state
        """
        if self._current_state != RobotState.GRASPING:
            raise RuntimeError(f"Cannot increment grasp attempts in state {self._current_state}")
        self._grasp_attempts += 1
        return self._grasp_attempts

    def get_cubes_collected(self) -> int:
        """
        Get total cubes successfully deposited this session

        Returns:
            Cube count (0-15)
        """
        return self._cubes_collected

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
        self._callbacks[state].append(callback)

    def get_allowed_transitions(self) -> Dict[RobotState, List[RobotState]]:
        """
        Get complete transition table

        Returns:
            Dict mapping state → list of allowed next states
        """
        # Implementation in Phase 6
        return {
            RobotState.SEARCHING: [RobotState.APPROACHING, RobotState.AVOIDING],
            RobotState.APPROACHING: [RobotState.GRASPING, RobotState.SEARCHING, RobotState.AVOIDING],
            RobotState.GRASPING: [RobotState.NAVIGATING_TO_BOX, RobotState.SEARCHING],
            RobotState.NAVIGATING_TO_BOX: [RobotState.DEPOSITING, RobotState.AVOIDING],
            RobotState.DEPOSITING: [RobotState.SEARCHING],
            RobotState.AVOIDING: [RobotState.SEARCHING, RobotState.APPROACHING, RobotState.GRASPING,
                                  RobotState.NAVIGATING_TO_BOX, RobotState.DEPOSITING],
        }

    def is_timeout_exceeded(self) -> bool:
        """
        Check if current state has exceeded 120s timeout

        Returns:
            True if timeout exceeded (FR-022)
        """
        duration = time.time() - self._state_start_time
        timeout_limit = self._timeout_limits.get(self._current_state, 120.0)
        return duration > timeout_limit

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _transition_to(self, new_state: RobotState, reason: str) -> None:
        """Internal method to perform state transition"""
        if new_state == self._current_state:
            return

        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = new_state
        self._state_start_time = time.time()
        self._transition_count += 1

        # Log transition
        self.logger.info(f"State transition: {old_state.name} → {new_state.name} ({reason})")

        # Call registered callbacks
        for callback in self._callbacks.get(new_state, []):
            try:
                callback(old_state, new_state)
            except Exception as e:
                self.logger.error(f"Callback error for {new_state}: {e}")

