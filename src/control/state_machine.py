"""
State machine for YouBot task coordination.

Manages transitions between states:
IDLE -> SEARCHING -> APPROACHING -> GRASPING -> TRANSPORTING -> DEPOSITING -> SEARCHING

Each state has specific entry/exit conditions and timeouts.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any


class RobotState(Enum):
    """Robot operating states."""
    IDLE = auto()           # Waiting to start
    SEARCHING = auto()      # Looking for cubes
    APPROACHING = auto()    # Moving toward detected cube
    ALIGNING = auto()       # Fine alignment before grasp
    GRASPING = auto()       # Executing grasp sequence
    TRANSPORTING = auto()   # Moving to deposit location
    DEPOSITING = auto()     # Releasing cube at target
    RECOVERING = auto()     # Error recovery


@dataclass
class StateContext:
    """Context passed to state handlers."""
    cube_detected: bool = False
    cube_color: Optional[str] = None
    cube_distance: float = float('inf')
    cube_angle: float = 0.0
    has_object: bool = False
    front_clear: bool = True
    nearest_obstacle: float = float('inf')
    steps_in_state: int = 0
    grasp_attempts: int = 0
    cubes_collected: int = 0


class StateMachine:
    """Finite state machine for task coordination."""

    def __init__(self):
        """Initialize state machine."""
        self._state = RobotState.IDLE
        self._previous_state = RobotState.IDLE
        self._steps_in_state = 0

        # State handlers: state -> (update_fn, timeout)
        self._handlers: Dict[RobotState, Callable] = {}
        self._timeouts: Dict[RobotState, int] = {
            RobotState.SEARCHING: 1000,
            RobotState.APPROACHING: 500,
            RobotState.ALIGNING: 200,
            RobotState.GRASPING: 200,
            RobotState.TRANSPORTING: 500,
            RobotState.DEPOSITING: 200,
            RobotState.RECOVERING: 300,
        }

        # Transition callbacks
        self._on_enter: Dict[RobotState, Callable] = {}
        self._on_exit: Dict[RobotState, Callable] = {}

        # Context
        self.context = StateContext()

        # Statistics
        self._transition_count = 0
        self._total_steps = 0

    @property
    def state(self) -> RobotState:
        """Current state."""
        return self._state

    @property
    def previous_state(self) -> RobotState:
        """Previous state."""
        return self._previous_state

    @property
    def steps_in_state(self) -> int:
        """Steps spent in current state."""
        return self._steps_in_state

    def register_handler(
        self,
        state: RobotState,
        handler: Callable[[StateContext], Optional[RobotState]],
    ) -> None:
        """Register update handler for a state.

        Args:
            state: State to handle
            handler: Function that receives context, returns new state or None
        """
        self._handlers[state] = handler

    def register_enter_callback(
        self,
        state: RobotState,
        callback: Callable[[StateContext], None],
    ) -> None:
        """Register callback for state entry."""
        self._on_enter[state] = callback

    def register_exit_callback(
        self,
        state: RobotState,
        callback: Callable[[StateContext], None],
    ) -> None:
        """Register callback for state exit."""
        self._on_exit[state] = callback

    def set_timeout(self, state: RobotState, timeout_steps: int) -> None:
        """Set timeout for a state."""
        self._timeouts[state] = timeout_steps

    def transition_to(self, new_state: RobotState) -> None:
        """Force transition to new state.

        Args:
            new_state: State to transition to
        """
        if new_state == self._state:
            return

        # Exit current state
        if self._state in self._on_exit:
            self._on_exit[self._state](self.context)

        self._previous_state = self._state
        self._state = new_state
        self._steps_in_state = 0
        self._transition_count += 1

        # Enter new state
        if new_state in self._on_enter:
            self._on_enter[new_state](self.context)

        print(f"[STATE] {self._previous_state.name} -> {new_state.name}")

    def update(self) -> RobotState:
        """Update state machine (call once per time step).

        Returns:
            Current state after update
        """
        self._steps_in_state += 1
        self._total_steps += 1
        self.context.steps_in_state = self._steps_in_state

        # Check timeout
        timeout = self._timeouts.get(self._state)
        if timeout and self._steps_in_state > timeout:
            print(f"[STATE] Timeout in {self._state.name} after {self._steps_in_state} steps")
            self._handle_timeout()
            return self._state

        # Run state handler
        if self._state in self._handlers:
            new_state = self._handlers[self._state](self.context)
            if new_state is not None and new_state != self._state:
                self.transition_to(new_state)

        return self._state

    def _handle_timeout(self) -> None:
        """Handle state timeout."""
        if self._state == RobotState.SEARCHING:
            # Continue searching
            self._steps_in_state = 0
        elif self._state == RobotState.APPROACHING:
            # Lost target, go back to search
            self.transition_to(RobotState.SEARCHING)
        elif self._state == RobotState.GRASPING:
            # Grasp failed, recover
            self.context.grasp_attempts += 1
            if self.context.grasp_attempts >= 3:
                self.transition_to(RobotState.SEARCHING)
            else:
                self.transition_to(RobotState.RECOVERING)
        elif self._state == RobotState.TRANSPORTING:
            self.transition_to(RobotState.RECOVERING)
        elif self._state == RobotState.DEPOSITING:
            self.transition_to(RobotState.SEARCHING)
        else:
            self.transition_to(RobotState.SEARCHING)

    def reset(self) -> None:
        """Reset state machine to initial state."""
        self._state = RobotState.IDLE
        self._previous_state = RobotState.IDLE
        self._steps_in_state = 0
        self._transition_count = 0
        self._total_steps = 0
        self.context = StateContext()

    def start(self) -> None:
        """Start the state machine (transition from IDLE to SEARCHING)."""
        if self._state == RobotState.IDLE:
            self.transition_to(RobotState.SEARCHING)

    # Convenience methods for common transitions

    def cube_found(self, color: str, distance: float, angle: float) -> None:
        """Signal that a cube was detected."""
        self.context.cube_detected = True
        self.context.cube_color = color
        self.context.cube_distance = distance
        self.context.cube_angle = angle

        if self._state == RobotState.SEARCHING:
            self.transition_to(RobotState.APPROACHING)

    def cube_lost(self) -> None:
        """Signal that cube was lost from view."""
        self.context.cube_detected = False

        if self._state == RobotState.APPROACHING:
            self.transition_to(RobotState.SEARCHING)

    def ready_to_grasp(self) -> None:
        """Signal that robot is in position to grasp."""
        if self._state == RobotState.APPROACHING:
            self.transition_to(RobotState.ALIGNING)
        elif self._state == RobotState.ALIGNING:
            self.transition_to(RobotState.GRASPING)

    def grasp_complete(self, success: bool) -> None:
        """Signal grasp attempt complete."""
        self.context.has_object = success

        if success:
            self.transition_to(RobotState.TRANSPORTING)
        else:
            self.context.grasp_attempts += 1
            if self.context.grasp_attempts >= 3:
                self.transition_to(RobotState.SEARCHING)
            else:
                self.transition_to(RobotState.RECOVERING)

    def at_deposit_location(self) -> None:
        """Signal arrival at deposit location."""
        if self._state == RobotState.TRANSPORTING:
            self.transition_to(RobotState.DEPOSITING)

    def deposit_complete(self) -> None:
        """Signal deposit complete."""
        self.context.has_object = False
        self.context.cube_color = None
        self.context.cubes_collected += 1
        self.context.grasp_attempts = 0
        self.transition_to(RobotState.SEARCHING)

    def get_stats(self) -> Dict[str, Any]:
        """Get state machine statistics."""
        return {
            'current_state': self._state.name,
            'total_steps': self._total_steps,
            'transitions': self._transition_count,
            'cubes_collected': self.context.cubes_collected,
            'grasp_attempts': self.context.grasp_attempts,
        }
