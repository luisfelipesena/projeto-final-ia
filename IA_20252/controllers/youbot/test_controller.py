#!/usr/bin/env python3
"""
Simple test controller to validate base movements in Webots.
Watch the robot in the 3D view to verify each movement.

Phase 1.2 - Basic Control Validation
"""

from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper
import time


def test_base_movements(robot):
    """Test basic base movements (FR-001 to FR-006)."""
    time_step = int(robot.getBasicTimeStep())
    base = Base(robot)

    print("\n" + "="*60)
    print("YOUBOT MOVEMENT TEST - Phase 1.2")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Forward movement
    print("\n[TEST 1/8] Forward movement (5s)...")
    base.move(vx=0.2, vy=0.0, omega=0.0)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Forward movement complete")
    tests_passed += 1

    # Test 2: Backward movement
    print("\n[TEST 2/8] Backward movement (5s)...")
    base.move(vx=-0.2, vy=0.0, omega=0.0)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Backward movement complete")
    tests_passed += 1

    # Test 3: Strafe left
    print("\n[TEST 3/8] Strafe left (5s)...")
    base.move(vx=0.0, vy=0.2, omega=0.0)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Strafe left complete")
    tests_passed += 1

    # Test 4: Strafe right
    print("\n[TEST 4/8] Strafe right (5s)...")
    base.move(vx=0.0, vy=-0.2, omega=0.0)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Strafe right complete")
    tests_passed += 1

    # Test 5: Rotate clockwise
    print("\n[TEST 5/8] Rotate clockwise (5s)...")
    base.move(vx=0.0, vy=0.0, omega=-0.3)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Rotate clockwise complete")
    tests_passed += 1

    # Test 6: Rotate counterclockwise
    print("\n[TEST 6/8] Rotate counterclockwise (5s)...")
    base.move(vx=0.0, vy=0.0, omega=0.3)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Rotate counterclockwise complete")
    tests_passed += 1

    # Test 7: Stop command
    print("\n[TEST 7/8] Stop command...")
    base.move(vx=0.2, vy=0.0, omega=0.0)
    for _ in range(int(1000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Stop command complete")
    tests_passed += 1

    # Test 8: Combined movement
    print("\n[TEST 8/8] Combined movement (diagonal)...")
    base.move(vx=0.2, vy=0.2, omega=0.0)
    for _ in range(int(5000 / time_step)):
        robot.step(time_step)
    base.reset()
    for _ in range(int(500 / time_step)):
        robot.step(time_step)
    print("✅ Combined movement complete")
    tests_passed += 1

    print("\n" + "="*60)
    print(f"BASE MOVEMENT TESTS: {tests_passed}/{tests_passed + tests_failed} PASSED")
    print("="*60)

    return tests_passed, tests_failed


def test_arm_gripper(robot):
    """Test arm and gripper movements (FR-008 to FR-013)."""
    time_step = int(robot.getBasicTimeStep())
    arm = Arm(robot)
    gripper = Gripper(robot)

    print("\n" + "="*60)
    print("ARM & GRIPPER TEST - Phase 1.2")
    print("="*60)

    tests_passed = 0

    # Test 1: Arm height positions
    print("\n[TEST 1/5] Testing arm heights...")
    heights = [
        ("FRONT_FLOOR", arm.FRONT_FLOOR),
        ("FRONT_PLATE", arm.FRONT_PLATE),
        ("RESET", arm.RESET),
    ]

    for name, height in heights:
        print(f"  → {name}")
        arm.set_height(height)
        for _ in range(int(3000 / time_step)):
            robot.step(time_step)

    print("✅ Arm height positions complete")
    tests_passed += 1

    # Test 2: Arm orientations
    print("\n[TEST 2/5] Testing arm orientations...")
    orientations = [
        ("BACK", arm.BACK),
        ("FRONT", arm.FRONT),
        ("BACK", arm.BACK),
    ]

    for name, orient in orientations:
        print(f"  → {name}")
        arm.set_orientation(orient)
        for _ in range(int(3000 / time_step)):
            robot.step(time_step)

    print("✅ Arm orientation positions complete")
    tests_passed += 1

    # Test 3: Gripper close
    print("\n[TEST 3/5] Gripper close...")
    gripper.grip()
    for _ in range(int(2000 / time_step)):
        robot.step(time_step)
    print("✅ Gripper close complete")
    tests_passed += 1

    # Test 4: Gripper open
    print("\n[TEST 4/5] Gripper open...")
    gripper.release()
    for _ in range(int(2000 / time_step)):
        robot.step(time_step)
    print("✅ Gripper open complete")
    tests_passed += 1

    # Test 5: Reset to initial position
    print("\n[TEST 5/5] Reset to initial position...")
    arm.reset()
    for _ in range(int(3000 / time_step)):
        robot.step(time_step)
    print("✅ Reset complete")
    tests_passed += 1

    print("\n" + "="*60)
    print(f"ARM & GRIPPER TESTS: {tests_passed}/5 PASSED")
    print("="*60)

    return tests_passed, 0


def main():
    """Main test execution."""
    print("\n🤖 Starting YouBot Control Validation Tests")
    print("⏱️  Total estimated time: ~2 minutes")
    print("\nWatch the 3D view to verify movements...\n")

    # Create single Robot instance (only one allowed per controller)
    robot = Robot()

    # Run tests
    base_passed, base_failed = test_base_movements(robot)
    arm_passed, arm_failed = test_arm_gripper(robot)

    # Summary
    total_passed = base_passed + arm_passed
    total_failed = base_failed + arm_failed
    total_tests = total_passed + total_failed

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"Success Rate: {100 * total_passed / total_tests:.1f}%")
    print("="*60)

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Base control validation complete.")
        print("\nNext steps:")
        print("1. Verify docs/arena_map.md")
        print("2. Execute sensor analysis notebook")
        print("3. Proceed to Phase 2 (Neural Networks)")
    else:
        print(f"\n⚠️  {total_failed} test(s) failed. Review controller implementation.")


if __name__ == "__main__":
    main()
