"""
ArmService - Grasp/deposit sequences WITHOUT movement

Based on: Craig (2005) - Introduction to Robotics
Test this with robot stationary, cube placed manually in front.
"""

import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto


class ArmState(Enum):
    """Current arm state"""
    REST = auto()           # Arm tucked away
    FRONT_HIGH = auto()     # Extended front, raised
    FRONT_LOW = auto()      # Extended front, floor level
    DEPOSIT = auto()        # Over deposit box
    MOVING = auto()         # Transitioning between states


@dataclass
class GraspResult:
    """Result of grasp attempt"""
    success: bool
    has_object: bool
    duration: float
    error: str = ""


class ArmService:
    """
    Arm/gripper control for stationary grasp sequences.

    NO movement dependency - robot must be positioned first.
    Test this BEFORE integrating with navigation.

    Usage:
        arm_svc = ArmService(arm, gripper, robot, time_step)

        # Full grasp cycle
        arm_svc.prepare_grasp()    # Arm to front, gripper open
        result = arm_svc.execute_grasp()  # Lower, close, lift
        if result.success:
            print("Cube grasped!")

        # Deposit
        arm_svc.prepare_deposit()
        arm_svc.execute_deposit()

    Test:
        # Place cube 25cm in front of stationary robot
        arm_svc.test_grasp_cycle()
    """

    # Timing constants (seconds) - generous timing for reliable grasping
    MOVE_TIME_HIGH = 2.5     # Time for arm to reach high position
    MOVE_TIME_LOW = 2.0      # Time for arm to lower to floor
    GRIP_TIME = 2.5          # Time for gripper to close fully (increased)
    LIFT_TIME = 2.5          # Time for arm to lift with cube

    def __init__(self, arm, gripper, robot, time_step: int):
        """
        Initialize ArmService.

        Args:
            arm: Arm controller instance (from arm.py)
            gripper: Gripper controller instance (from gripper.py)
            robot: Webots Robot instance (for step())
            time_step: Simulation time step in ms
        """
        self.arm = arm
        self.gripper = gripper
        self.robot = robot
        self.time_step = time_step

        self.state = ArmState.REST
        self._has_object = False

    def _step(self) -> bool:
        """Execute one simulation step. Returns False if simulation ended."""
        return self.robot.step(self.time_step) != -1

    def _wait(self, seconds: float) -> bool:
        """
        Wait for specified duration while stepping simulation.

        Args:
            seconds: Time to wait

        Returns:
            True if completed, False if simulation ended
        """
        steps = int(seconds * 1000 / self.time_step)
        for _ in range(steps):
            if not self._step():
                return False
        return True

    def reset(self) -> bool:
        """
        Reset arm to tucked position.

        Returns:
            True if completed
        """
        self.arm.set_height(self.arm.RESET)
        self.arm.set_orientation(self.arm.FRONT)
        self.state = ArmState.MOVING

        if not self._wait(self.MOVE_TIME_HIGH):
            return False

        self.state = ArmState.REST
        return True

    def prepare_grasp(self) -> bool:
        """
        Move arm to pre-grasp position (front, raised) and open gripper.

        Returns:
            True if completed
        """
        self.state = ArmState.MOVING

        # Open gripper first
        print(f"[ArmService] Opening gripper...")
        self.gripper.release()

        # Wait for gripper to open
        if not self._wait(0.5):
            return False

        # Log finger position after opening
        if self.gripper.finger_sensor:
            finger_pos = self.gripper.finger_sensor.getValue()
            print(f"[ArmService] Gripper opened: finger_pos={finger_pos:.5f} (should be ~0.025)")
            # Write to grasp log
            from pathlib import Path
            log_file = Path(__file__).parent.parent.parent / "youbot_mcp" / "data" / "youbot" / "grasp_log.txt"
            with open(log_file, 'a') as f:
                f.write(f"{time.time()}: [ARM] gripper_opened: finger_pos={finger_pos:.5f}\n")

        # Move arm to front, raised position
        self.arm.set_orientation(self.arm.FRONT)
        self.arm.set_height(self.arm.FRONT_PLATE)

        if not self._wait(self.MOVE_TIME_HIGH):
            return False

        self.state = ArmState.FRONT_HIGH
        return True

    def execute_grasp(self, use_ik: bool = False, forward_reach: float = 0.22) -> GraspResult:
        """
        Execute grasp sequence: lower arm, close gripper, verify, lift.

        Requires arm to be at FRONT_HIGH position (call prepare_grasp first).

        Args:
            use_ik: If True, use IK for precise positioning instead of preset
            forward_reach: Forward distance for IK (meters from arm base)

        Returns:
            GraspResult with success status
        """
        start_time = time.time()

        if self.state != ArmState.FRONT_HIGH:
            return GraspResult(
                success=False,
                has_object=False,
                duration=0,
                error="Must call prepare_grasp() first"
            )

        # Step 1: Lower arm to floor level
        if use_ik:
            # Use IK for precise positioning: (x=0 centered, y=forward, z=cube height)
            print(f"[ArmService] Using IK: x=0, y={forward_reach:.3f}, z=0.025")
            self.arm.inverse_kinematics(x=0.0, y=forward_reach, z=0.025)
        else:
            print(f"[ArmService] Lowering to FRONT_FLOOR")
            self.arm.set_height(self.arm.FRONT_FLOOR)
        self.state = ArmState.MOVING

        if not self._wait(self.MOVE_TIME_LOW):
            return GraspResult(
                success=False,
                has_object=False,
                duration=time.time() - start_time,
                error="Simulation ended during lower"
            )

        self.state = ArmState.FRONT_LOW

        # Step 2: Close gripper
        self.gripper.grip()

        if not self._wait(self.GRIP_TIME):
            return GraspResult(
                success=False,
                has_object=False,
                duration=time.time() - start_time,
                error="Simulation ended during grip"
            )

        # Step 3: Check if object grasped
        self._has_object = self.gripper.has_object()

        # Debug: log finger position to file for MCP visibility
        finger_pos = 0.0
        finger_pos_before = 0.0
        if self.gripper.finger_sensor:
            finger_pos = self.gripper.finger_sensor.getValue()
            print(f"[ArmService] Gripper finger position after close: {finger_pos:.5f} (threshold: 0.003)")
        print(f"[ArmService] has_object = {self._has_object}")

        # Write to grasp log for debugging
        from pathlib import Path
        log_file = Path(__file__).parent.parent.parent / "youbot_mcp" / "data" / "youbot" / "grasp_log.txt"
        with open(log_file, 'a') as f:
            f.write(f"{time.time()}: [ARM] finger_pos_after_close={finger_pos:.5f}, has_object={self._has_object}\n")

        # Step 4: Lift arm (regardless of success, so we can retry)
        self.arm.set_height(self.arm.FRONT_PLATE)
        self.state = ArmState.MOVING

        if not self._wait(self.LIFT_TIME):
            return GraspResult(
                success=False,
                has_object=self._has_object,
                duration=time.time() - start_time,
                error="Simulation ended during lift"
            )

        self.state = ArmState.FRONT_HIGH
        duration = time.time() - start_time

        if self._has_object:
            print(f"[ArmService] GRASP SUCCESS in {duration:.2f}s")
        else:
            print(f"[ArmService] GRASP FAILED - no object detected")

        return GraspResult(
            success=self._has_object,
            has_object=self._has_object,
            duration=duration
        )

    def prepare_deposit(self) -> bool:
        """
        Move arm to deposit position (over box).

        Returns:
            True if completed
        """
        self.state = ArmState.MOVING

        # Move to back plate position (over deposit box)
        self.arm.set_orientation(self.arm.BACK_LEFT)
        self.arm.set_height(self.arm.BACK_PLATE_HIGH)

        if not self._wait(self.MOVE_TIME_HIGH):
            return False

        self.state = ArmState.DEPOSIT
        return True

    def execute_deposit(self) -> bool:
        """
        Execute deposit: open gripper to release cube.

        Returns:
            True if completed
        """
        self.gripper.release()

        if not self._wait(0.5):
            return False

        self._has_object = False
        return True

    def return_to_rest(self) -> bool:
        """
        Return arm to tucked REST position.

        Returns:
            True if completed
        """
        return self.reset()

    def has_object(self) -> bool:
        """
        Check if gripper currently holds an object.

        Returns:
            True if object is held
        """
        return self._has_object

    # ==================== TEST METHODS ====================

    def test_grasp_cycle(self) -> bool:
        """
        Test: Complete grasp cycle with manually placed cube.

        Steps:
        1. Place cube ~25cm in front of robot (manually in Webots)
        2. Run this test
        3. Verify cube is grasped (has_object=True)
        4. Deposit cube
        5. Verify cube is released (has_object=False)

        Returns:
            True if test passed
        """
        print("=" * 50)
        print("[ArmService] TEST: Full grasp cycle")
        print("  Ensure cube is placed ~25cm in front of robot!")
        print("=" * 50)

        # Wait a moment for user to verify cube placement
        print("  Starting in 2 seconds...")
        if not self._wait(2.0):
            return False

        # Prepare grasp
        print("\n[TEST] Step 1: Prepare grasp")
        if not self.prepare_grasp():
            print("[TEST] FAILED at prepare_grasp")
            return False

        # Execute grasp
        print("\n[TEST] Step 2: Execute grasp")
        result = self.execute_grasp()
        print(f"[TEST] Grasp result: success={result.success}, has_object={result.has_object}")

        if not result.success:
            print("[TEST] FAILED - Grasp unsuccessful")
            print("  Check: Is cube placed correctly?")
            print("  Check: Is gripper reaching floor level?")
            self.reset()
            return False

        # Hold for a moment
        print("\n[TEST] Step 3: Holding cube...")
        if not self._wait(2.0):
            return False

        # Prepare deposit
        print("\n[TEST] Step 4: Prepare deposit")
        if not self.prepare_deposit():
            print("[TEST] FAILED at prepare_deposit")
            return False

        # Execute deposit
        print("\n[TEST] Step 5: Execute deposit")
        if not self.execute_deposit():
            print("[TEST] FAILED at execute_deposit")
            return False

        # Verify released
        if self.has_object():
            print("[TEST] WARNING: Still showing has_object after deposit")

        # Return to rest
        print("\n[TEST] Step 6: Return to rest")
        if not self.return_to_rest():
            print("[TEST] FAILED at return_to_rest")
            return False

        print("\n" + "=" * 50)
        print("[ArmService] TEST PASSED: Full grasp cycle complete")
        print("=" * 50)
        return True

    def test_arm_positions(self) -> bool:
        """
        Test: Cycle through arm positions without grasping.

        Useful to verify arm movement works before testing grasp.

        Returns:
            True if all positions reached
        """
        print("[ArmService] TEST: Arm positions")

        positions = [
            ("REST", self.arm.RESET, self.arm.FRONT),
            ("FRONT_PLATE", self.arm.FRONT_PLATE, self.arm.FRONT),
            ("FRONT_FLOOR", self.arm.FRONT_FLOOR, self.arm.FRONT),
            ("FRONT_PLATE", self.arm.FRONT_PLATE, self.arm.FRONT),
            ("BACK_PLATE_HIGH", self.arm.BACK_PLATE_HIGH, self.arm.BACK_LEFT),
            ("REST", self.arm.RESET, self.arm.FRONT),
        ]

        for name, height, orientation in positions:
            print(f"  Moving to: {name}")
            self.arm.set_height(height)
            self.arm.set_orientation(orientation)
            if not self._wait(2.0):
                return False

        print("[ArmService] TEST COMPLETE: Arm positions")
        return True

    def test_gripper(self) -> bool:
        """
        Test: Open/close gripper without arm movement.

        Returns:
            True if gripper responds
        """
        print("[ArmService] TEST: Gripper open/close")

        print("  Opening gripper...")
        self.gripper.release()
        if not self._wait(1.0):
            return False

        print("  Closing gripper...")
        self.gripper.grip()
        if not self._wait(1.0):
            return False

        has_obj = self.gripper.has_object()
        print(f"  has_object (should be False on air): {has_obj}")

        print("  Opening gripper...")
        self.gripper.release()
        if not self._wait(1.0):
            return False

        print("[ArmService] TEST COMPLETE: Gripper")
        return True


# ==================== STANDALONE TEST ====================

def test_arm_service():
    """
    Standalone test for ArmService.

    Run: python -m src.services.arm_service --test grasp
    """
    import sys

    try:
        from controller import Robot
        from IA_20252.controllers.youbot.arm import Arm
        from IA_20252.controllers.youbot.gripper import Gripper
    except ImportError:
        print("ERROR: Must run inside Webots simulation")
        return

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    arm = Arm(robot)
    gripper = Gripper(robot)

    arm_svc = ArmService(arm, gripper, robot, time_step)

    # Parse test type
    test_type = "grasp"
    if len(sys.argv) > 1:
        if "--test" in sys.argv:
            idx = sys.argv.index("--test")
            if idx + 1 < len(sys.argv):
                test_type = sys.argv[idx + 1]

    print(f"[ArmService] Running test: {test_type}")

    # Wait for simulation to start
    for _ in range(10):
        robot.step(time_step)

    if test_type == "grasp":
        arm_svc.test_grasp_cycle()
    elif test_type == "positions":
        arm_svc.test_arm_positions()
    elif test_type == "gripper":
        arm_svc.test_gripper()
    else:
        print(f"Unknown test: {test_type}")
        print("Available tests: grasp, positions, gripper")


if __name__ == "__main__":
    test_arm_service()
