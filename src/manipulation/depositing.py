"""
DepositController Module

Finite state machine for cube depositing sequences.
Based on: YouBot arm kinematics and box locations.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import time


class DepositState(Enum):
    """States in the deposit sequence"""
    IDLE = auto()
    POSITION_ARM = auto()     # Move arm to deposit position
    EXTEND_ARM = auto()       # Extend arm over box
    OPEN_GRIPPER = auto()     # Release cube
    RETRACT_ARM = auto()      # Retract arm
    COMPLETE = auto()         # Deposit sequence complete
    FAILED = auto()           # Deposit failed


@dataclass
class DepositResult:
    """Result of deposit attempt"""
    success: bool
    cube_color: Optional[str] = None
    duration: float = 0.0
    error_message: str = ""


class DepositController:
    """
    Finite state machine for cube depositing

    Controls the arm and gripper through a predefined deposit sequence.
    Moves arm to position over the correct colored box and releases cube.

    State Sequence:
    1. POSITION_ARM: Move arm to side position for box
    2. EXTEND_ARM: Lower/extend arm over box
    3. OPEN_GRIPPER: Release cube
    4. RETRACT_ARM: Pull arm back
    5. COMPLETE: Done

    Usage:
        deposit_ctrl = DepositController(arm, gripper)

        # Start deposit sequence for a color
        deposit_ctrl.start('green')

        # In control loop:
        while not deposit_ctrl.is_done():
            deposit_ctrl.update(dt=0.032)

        result = deposit_ctrl.get_result()
        if result.success:
            print(f"Deposited {result.cube_color} cube!")
    """

    # State timing (seconds)
    STATE_DURATIONS = {
        DepositState.POSITION_ARM: 1.5,
        DepositState.EXTEND_ARM: 1.0,
        DepositState.OPEN_GRIPPER: 0.5,
        DepositState.RETRACT_ARM: 1.0,
    }

    # Arm orientation for each box color
    # Based on arena layout: boxes are on the sides
    BOX_ORIENTATIONS = {
        'green': 'LEFT',      # Green box on left
        'blue': 'LEFT',       # Blue box on left (different position)
        'red': 'LEFT',        # Red box on left (different position)
    }

    def __init__(self, arm, gripper):
        """
        Initialize deposit controller

        Args:
            arm: Arm controller instance
            gripper: Gripper controller instance
        """
        self.arm = arm
        self.gripper = gripper

        self.state = DepositState.IDLE
        self.state_start_time = 0.0
        self.cube_color: Optional[str] = None
        self.sequence_start_time = 0.0

        # Result
        self._result: Optional[DepositResult] = None

    def start(self, cube_color: str) -> None:
        """
        Start deposit sequence

        Args:
            cube_color: Color of cube being deposited ('green', 'blue', 'red')
        """
        self.cube_color = cube_color
        self.sequence_start_time = time.time()
        self._result = None
        self._transition_to(DepositState.POSITION_ARM)

    def update(self, dt: float = 0.032) -> DepositState:
        """
        Update deposit state machine

        Args:
            dt: Time delta (seconds)

        Returns:
            Current DepositState
        """
        if self.state in (DepositState.IDLE, DepositState.COMPLETE, DepositState.FAILED):
            return self.state

        elapsed = time.time() - self.state_start_time
        duration = self.STATE_DURATIONS.get(self.state, 1.0)

        # State-specific actions and transitions
        if self.state == DepositState.POSITION_ARM:
            self._execute_position_arm()
            if elapsed >= duration:
                self._transition_to(DepositState.EXTEND_ARM)

        elif self.state == DepositState.EXTEND_ARM:
            self._execute_extend_arm()
            if elapsed >= duration:
                self._transition_to(DepositState.OPEN_GRIPPER)

        elif self.state == DepositState.OPEN_GRIPPER:
            self._execute_open_gripper()
            if elapsed >= duration:
                self._transition_to(DepositState.RETRACT_ARM)

        elif self.state == DepositState.RETRACT_ARM:
            self._execute_retract_arm()
            if elapsed >= duration:
                self._complete(success=True)

        return self.state

    def _transition_to(self, new_state: DepositState) -> None:
        """Transition to new state"""
        self.state = new_state
        self.state_start_time = time.time()

    def _execute_position_arm(self) -> None:
        """Position arm for deposit"""
        if self.arm is None:
            return

        # Determine orientation based on cube color
        orientation = self.BOX_ORIENTATIONS.get(self.cube_color, 'LEFT')

        if orientation == 'LEFT':
            self.arm.set_orientation(self.arm.LEFT)
        elif orientation == 'RIGHT':
            self.arm.set_orientation(self.arm.RIGHT)
        else:
            self.arm.set_orientation(self.arm.FRONT)

        # Raise arm first
        self.arm.set_height(self.arm.FRONT_PLATE)

    def _execute_extend_arm(self) -> None:
        """Extend arm over box"""
        if self.arm is not None:
            # Lower arm to deposit height
            # Using FRONT_CARDBOARD_BOX as approximate box height
            self.arm.set_height(self.arm.FRONT_CARDBOARD_BOX)

    def _execute_open_gripper(self) -> None:
        """Open gripper to release cube"""
        if self.gripper is not None:
            self.gripper.release()

    def _execute_retract_arm(self) -> None:
        """Retract arm after deposit"""
        if self.arm is not None:
            self.arm.set_height(self.arm.RESET)
            self.arm.set_orientation(self.arm.FRONT)

    def _complete(self, success: bool, error: str = "") -> None:
        """Complete deposit sequence"""
        self.state = DepositState.COMPLETE if success else DepositState.FAILED

        self._result = DepositResult(
            success=success,
            cube_color=self.cube_color,
            duration=time.time() - self.sequence_start_time,
            error_message=error
        )

    def is_done(self) -> bool:
        """Check if sequence is complete (success or failure)"""
        return self.state in (DepositState.COMPLETE, DepositState.FAILED, DepositState.IDLE)

    def is_success(self) -> bool:
        """Check if deposit was successful"""
        return self.state == DepositState.COMPLETE

    def get_result(self) -> Optional[DepositResult]:
        """Get deposit result (only valid after sequence completes)"""
        return self._result

    def reset(self) -> None:
        """Reset controller to idle state"""
        self.state = DepositState.IDLE
        self.cube_color = None
        self._result = None


def test_deposit_controller():
    """Test deposit controller (mock)"""
    print("Testing DepositController...")

    # Mock arm and gripper
    class MockArm:
        FRONT = 0
        LEFT = 1
        RIGHT = 2
        FRONT_PLATE = 1
        FRONT_CARDBOARD_BOX = 2
        RESET = 3

        def set_orientation(self, val): pass
        def set_height(self, val): pass

    class MockGripper:
        def release(self): pass
        def grip(self): pass
        def is_closed(self): return False

    deposit = DepositController(MockArm(), MockGripper())

    # Start deposit
    deposit.start('green')
    print(f"  Started deposit for green cube")

    # Simulate updates
    steps = 0
    while not deposit.is_done() and steps < 200:
        deposit.update(dt=0.032)
        steps += 1

    result = deposit.get_result()
    print(f"  Completed: success={result.success}, duration={result.duration:.2f}s")

    assert result.success, "Deposit should succeed"
    print("  DepositController test passed")


if __name__ == "__main__":
    test_deposit_controller()
