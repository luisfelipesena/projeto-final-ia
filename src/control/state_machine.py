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
        sm.update(conditions)
        current_state = sm.current_state
    """

    def __init__(self, initial_state: RobotState = RobotState.SEARCHING, timeout_seconds: float = 120.0):
        """
        Initialize state machine

        Args:
            initial_state: Starting state (default: SEARCHING)
            timeout_seconds: Maximum time per state before timeout (default: 120s, FR-022)
        """
        self.current_state = initial_state
        self.previous_state: Optional[RobotState] = None
        self.timeout_seconds = timeout_seconds
        self.state_entry_time = time.time()
        self.transition_count = 0
        self.grasp_attempts = 0
        self.tracked_cube_id: Optional[str] = None
        self.tracked_cube_color: Optional[str] = None  # 'green' | 'blue' | 'red'
        self.metrics = StateMetrics(
            state=initial_state,
            entry_time=time.time(),
            duration=0.0,
            transition_count=0,
            timeout_triggered=False
        )

        # Setup logging
        self.logger = logging.getLogger('state_machine')
        handler = logging.FileHandler('logs/state_transitions.log')
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def transition_to(self, new_state: RobotState, context: Optional[Dict] = None) -> None:
        """
        Transition to new state

        Args:
            new_state: Target state
            context: Optional context dict with transition data
        """
        if new_state == self.current_state:
            return  # No transition needed

        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = time.time()
        self.transition_count += 1

        # Update metrics
        self.metrics.state = new_state
        self.metrics.entry_time = self.state_entry_time
        self.metrics.transition_count = self.transition_count

        # Log transition
        self.logger.info(
            f"Transition: {self.previous_state.name} → {new_state.name} "
            f"(context: {context})"
        )

        # Handle state-specific initialization
        if new_state == RobotState.GRASPING:
            self.grasp_attempts += 1
        elif new_state == RobotState.SEARCHING:
            # Reset cube tracking when searching
            if self.previous_state != RobotState.AVOIDING:
                self.tracked_cube_id = None
                self.tracked_cube_color = None
                self.grasp_attempts = 0

    def update(self, conditions: StateTransitionConditions) -> None:
        """
        Update state machine based on sensor conditions

        Args:
            conditions: Current sensor-based conditions

        State Transition Logic:
        - AVOIDING override: Always check first (FR-011)
        - State-specific transitions: Based on current state
        - Timeout handling: Reset to SEARCHING if timeout exceeded (FR-022)
        """
        # Check timeout (FR-022)
        elapsed = time.time() - self.state_entry_time
        if elapsed > self.timeout_seconds:
            self.logger.warning(
                f"State timeout ({elapsed:.1f}s > {self.timeout_seconds}s) in {self.current_state.name}, "
                f"transitioning to SEARCHING"
            )
            self.metrics.timeout_triggered = True
            self.transition_to(RobotState.SEARCHING, {'reason': 'timeout'})
            return

        # Update metrics duration
        self.metrics.duration = elapsed

        # Priority 1: AVOIDING override (FR-011)
        if conditions.obstacle_distance < 0.3:
            if self.current_state != RobotState.AVOIDING:
                self.transition_to(RobotState.AVOIDING, {
                    'obstacle_distance': conditions.obstacle_distance,
                    'previous_state': self.previous_state
                })
            return  # Stay in AVOIDING until obstacle clears

        # Priority 2: Exit AVOIDING when safe
        if self.current_state == RobotState.AVOIDING:
            if conditions.obstacle_distance > 0.5:
                # Return to previous state or SEARCHING
                target_state = self.previous_state if self.previous_state else RobotState.SEARCHING
                self.transition_to(target_state, {'reason': 'obstacle_cleared'})
            return

        # Priority 3: State-specific transitions
        if self.current_state == RobotState.SEARCHING:
            if conditions.cube_detected:
                self.transition_to(RobotState.APPROACHING, {
                    'cube_distance': conditions.cube_distance,
                    'cube_angle': conditions.cube_angle
                })

        elif self.current_state == RobotState.APPROACHING:
            if not conditions.cube_detected:
                # Lost cube detection
                self.transition_to(RobotState.SEARCHING, {'reason': 'cube_lost'})
            elif conditions.cube_distance < 0.15 and abs(conditions.cube_angle) < 5.0:
                # Within grasping range
                self.transition_to(RobotState.GRASPING, {
                    'cube_distance': conditions.cube_distance,
                    'cube_angle': conditions.cube_angle
                })

        elif self.current_state == RobotState.GRASPING:
            if conditions.grasp_success:
                self.transition_to(RobotState.NAVIGATING_TO_BOX, {
                    'cube_color': self.tracked_cube_color
                })
            elif self.grasp_attempts >= 3:
                # Max retries exceeded
                self.transition_to(RobotState.SEARCHING, {'reason': 'grasp_failed'})

        elif self.current_state == RobotState.NAVIGATING_TO_BOX:
            if conditions.at_target_box:
                self.transition_to(RobotState.DEPOSITING, {
                    'cube_color': self.tracked_cube_color
                })

        elif self.current_state == RobotState.DEPOSITING:
            if conditions.deposit_complete:
                self.transition_to(RobotState.SEARCHING, {'reason': 'deposit_complete'})

    def get_state(self) -> RobotState:
        """Get current state"""
        return self.current_state

    def get_metrics(self) -> StateMetrics:
        """Get current state metrics"""
        self.metrics.duration = time.time() - self.metrics.entry_time
        return self.metrics

    def set_cube_tracking(self, cube_id: str, cube_color: str) -> None:
        """
        Track cube for navigation (FR-012)

        Args:
            cube_id: Unique identifier for cube
            cube_color: Color of cube ('green' | 'blue' | 'red')
        """
        self.tracked_cube_id = cube_id
        self.tracked_cube_color = cube_color
        self.logger.info(f"Tracking cube {cube_id} (color: {cube_color})")

        # Call registered callbacks
        for callback in self._callbacks.get(new_state, []):
            try:
                callback(old_state, new_state)
            except Exception as e:
                self.logger.error(f"Callback error for {new_state}: {e}")

