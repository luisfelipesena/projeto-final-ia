"""
GraspController Module

Finite state machine for cube grasping sequences.
Based on: YouBot arm kinematics and gripper control.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable
import time


class GraspState(Enum):
    """States in the grasp sequence"""
    IDLE = auto()
    PREPARE_ARM = auto()      # Move arm to pre-grasp position
    LOWER_ARM = auto()        # Lower arm to grasp height
    CLOSE_GRIPPER = auto()    # Close gripper on cube
    VERIFY_GRASP = auto()     # Verify cube is grasped
    LIFT_CUBE = auto()        # Lift cube off ground
    COMPLETE = auto()         # Grasp sequence complete
    FAILED = auto()           # Grasp failed


@dataclass
class GraspResult:
    """Result of grasp attempt"""
    success: bool
    cube_color: Optional[str] = None
    duration: float = 0.0
    attempts: int = 1
    error_message: str = ""


class GraspController:
    """
    Finite state machine for cube grasping

    Controls the arm and gripper through a predefined grasp sequence.
    Each state has timing requirements and transition conditions.

    State Sequence:
    1. PREPARE_ARM: Move arm to front position
    2. LOWER_ARM: Lower to floor level
    3. CLOSE_GRIPPER: Close gripper
    4. VERIFY_GRASP: Check if cube is held
    5. LIFT_CUBE: Raise arm
    6. COMPLETE: Done

    Usage:
        grasp_ctrl = GraspController(arm, gripper)

        # Start grasp sequence
        grasp_ctrl.start('green')

        # In control loop:
        while not grasp_ctrl.is_done():
            grasp_ctrl.update(dt=0.032)

        result = grasp_ctrl.get_result()
        if result.success:
            print(f"Grasped {result.cube_color} cube!")
    """

    # State timing (seconds)
    STATE_DURATIONS = {
        GraspState.PREPARE_ARM: 1.5,
        GraspState.LOWER_ARM: 1.0,
        GraspState.CLOSE_GRIPPER: 0.8,
        GraspState.VERIFY_GRASP: 0.3,
        GraspState.LIFT_CUBE: 1.0,
    }

    MAX_ATTEMPTS = 3

    def __init__(self, arm, gripper):
        """
        Initialize grasp controller

        Args:
            arm: Arm controller instance
            gripper: Gripper controller instance
        """
        self.arm = arm
        self.gripper = gripper

        self.state = GraspState.IDLE
        self.state_start_time = 0.0
        self.cube_color: Optional[str] = None
        self.attempt_count = 0
        self.sequence_start_time = 0.0

        # Result
        self._result: Optional[GraspResult] = None

    def start(self, cube_color: str) -> None:
        """
        Start grasp sequence

        Args:
            cube_color: Color of cube being grasped ('green', 'blue', 'red')
        """
        self.cube_color = cube_color
        self.attempt_count = 1
        self.sequence_start_time = time.time()
        self._result = None
        self._transition_to(GraspState.PREPARE_ARM)

    def update(self, dt: float = 0.032) -> GraspState:
        """
        Update grasp state machine

        Args:
            dt: Time delta (seconds)

        Returns:
            Current GraspState
        """
        if self.state in (GraspState.IDLE, GraspState.COMPLETE, GraspState.FAILED):
            return self.state

        elapsed = time.time() - self.state_start_time
        duration = self.STATE_DURATIONS.get(self.state, 1.0)

        # State-specific actions and transitions
        if self.state == GraspState.PREPARE_ARM:
            self._execute_prepare_arm()
            if elapsed >= duration:
                self._transition_to(GraspState.LOWER_ARM)

        elif self.state == GraspState.LOWER_ARM:
            self._execute_lower_arm()
            if elapsed >= duration:
                self._transition_to(GraspState.CLOSE_GRIPPER)

        elif self.state == GraspState.CLOSE_GRIPPER:
            self._execute_close_gripper()
            if elapsed >= duration:
                self._transition_to(GraspState.VERIFY_GRASP)

        elif self.state == GraspState.VERIFY_GRASP:
            if elapsed >= duration:
                if self._verify_grasp():
                    self._transition_to(GraspState.LIFT_CUBE)
                else:
                    # Grasp failed, retry or fail
                    if self.attempt_count < self.MAX_ATTEMPTS:
                        self.attempt_count += 1
                        self.gripper.release()
                        self._transition_to(GraspState.PREPARE_ARM)
                    else:
                        self._complete(success=False, error="Max attempts exceeded")

        elif self.state == GraspState.LIFT_CUBE:
            self._execute_lift_cube()
            if elapsed >= duration:
                self._complete(success=True)

        return self.state

    def _transition_to(self, new_state: GraspState) -> None:
        """Transition to new state"""
        self.state = new_state
        self.state_start_time = time.time()

    def _execute_prepare_arm(self) -> None:
        """Move arm to pre-grasp position"""
        if self.arm is not None:
            # Set arm to front position, raised
            self.arm.set_orientation(self.arm.FRONT)
            self.arm.set_height(self.arm.FRONT_PLATE)

        # Open gripper
        if self.gripper is not None:
            self.gripper.release()

    def _execute_lower_arm(self) -> None:
        """Lower arm to grasp height"""
        if self.arm is not None:
            self.arm.set_height(self.arm.FRONT_FLOOR)

    def _execute_close_gripper(self) -> None:
        """Close gripper to grasp cube"""
        if self.gripper is not None:
            self.gripper.grip()

    def _verify_grasp(self) -> bool:
        """
        Verify cube is successfully grasped

        Returns:
            True if cube appears to be grasped
        """
        if self.gripper is None:
            return True  # Assume success in test mode

        # In real implementation, could check:
        # - Gripper force feedback
        # - Camera verification
        # - Motor current
        return self.gripper.is_closed()

    def _execute_lift_cube(self) -> None:
        """Lift cube after grasping"""
        if self.arm is not None:
            self.arm.set_height(self.arm.FRONT_PLATE)

    def _complete(self, success: bool, error: str = "") -> None:
        """Complete grasp sequence"""
        self.state = GraspState.COMPLETE if success else GraspState.FAILED

        self._result = GraspResult(
            success=success,
            cube_color=self.cube_color,
            duration=time.time() - self.sequence_start_time,
            attempts=self.attempt_count,
            error_message=error
        )

    def is_done(self) -> bool:
        """Check if sequence is complete (success or failure)"""
        return self.state in (GraspState.COMPLETE, GraspState.FAILED, GraspState.IDLE)

    def is_success(self) -> bool:
        """Check if grasp was successful"""
        return self.state == GraspState.COMPLETE

    def get_result(self) -> Optional[GraspResult]:
        """Get grasp result (only valid after sequence completes)"""
        return self._result

    def reset(self) -> None:
        """Reset controller to idle state"""
        self.state = GraspState.IDLE
        self.cube_color = None
        self._result = None


def test_grasp_controller():
    """Test grasp controller (mock)"""
    print("Testing GraspController...")

    # Mock arm and gripper
    class MockArm:
        FRONT = 0
        FRONT_PLATE = 1
        FRONT_FLOOR = 0

        def set_orientation(self, val): pass
        def set_height(self, val): pass

    class MockGripper:
        def __init__(self):
            self._closed = False

        def grip(self):
            self._closed = True

        def release(self):
            self._closed = False

        def is_closed(self):
            return self._closed

    grasp = GraspController(MockArm(), MockGripper())

    # Start grasp
    grasp.start('green')
    print(f"  Started grasp for green cube")

    # Simulate updates
    steps = 0
    while not grasp.is_done() and steps < 200:
        grasp.update(dt=0.032)
        steps += 1

    result = grasp.get_result()
    print(f"  Completed: success={result.success}, attempts={result.attempts}, duration={result.duration:.2f}s")

    assert result.success, "Grasp should succeed"
    print("  GraspController test passed")


if __name__ == "__main__":
    test_grasp_controller()
