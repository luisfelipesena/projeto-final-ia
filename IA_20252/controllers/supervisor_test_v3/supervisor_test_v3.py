"""
Supervisor Test V3 - Multi-Angle Grasp Validation

Orchestrates 5 grasp tests at different angles:
- Test 1: 0° (baseline)
- Test 2: +30° (CCW rotation)
- Test 3: -30° (CW rotation)
- Test 4: +60° (CCW rotation forte)
- Test 5: -60° (CW rotation forte)

Monitors status.json, resets robot/cube between tests.
"""

from controller import Supervisor
import json
import os
import time
import math

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_ROOT, "youbot_mcp", "data", "youbot")
STATUS_FILE = os.path.join(DATA_DIR, "status.json")
RESULTS_FILE = os.path.join(DATA_DIR, "test_results_v3.json")

# Robot initial position
ROBOT_INIT = {
    'x': -1.5,
    'y': 0.0,
    'z': 0.102,
    'rotation': [0, 0, 1, 0]  # facing +X
}

# Test positions: (x, y, angle_name)
# Robot at (-1.5, 0), cube at (X, Y)
# NOTE: Testing ONE angle at a time.
TEST_POSITIONS = [
    # (-1.0,  0.00, "0°"),     # Test 1: baseline - VALIDATED in V2
    # (-1.0,  0.29, "+30°"),   # Test 2: atan2(0.29, 0.5) ≈ 30° - VALIDATED
    # (-1.0, -0.29, "-30°"),   # Test 3: atan2(-0.29, 0.5) ≈ -30°
    # (-1.0,  0.87, "+60°"),   # Test 4: blocked by obstacle
    (-1.5,  0.60, "+90°"),   # Test 5: 90° - cube directly LEFT of robot
    # (-1.0, -0.87, "-60°"),   # Test 6
]

CUBE_Z = 0.016  # Half cube size (3cm cube on ground)
CUBE_SIZE = 0.03


class SupervisorTestV3:
    """Multi-angle grasp test supervisor."""

    def __init__(self):
        print("=" * 60)
        print("SUPERVISOR TEST V3 - Multi-Angle Grasp Validation")
        print("=" * 60)

        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Get world root
        self.root = self.supervisor.getRoot()
        self.root_children = self.root.getField("children")

        # Find YouBot
        self.youbot = self.supervisor.getFromDef("YOUBOT")
        if not self.youbot:
            # Try finding by name
            self.youbot = self._find_node_by_type("Youbot")

        if self.youbot:
            print(f"[INIT] Found YouBot: {self.youbot.getTypeName()}")
        else:
            print("[INIT] WARNING: YouBot not found!")

        # State
        self.current_test = 0
        self.test_results = []
        self.test_start_time = 0
        self.cube_node = None
        self.test_timeout = 60.0  # 60s per test

        # Ensure data dir exists
        os.makedirs(DATA_DIR, exist_ok=True)

        print(f"[INIT] Status file: {STATUS_FILE}")
        print(f"[INIT] Results file: {RESULTS_FILE}")
        print(f"[INIT] {len(TEST_POSITIONS)} tests configured")
        print("=" * 60)

    def _find_node_by_type(self, type_name: str):
        """Find first node of given type."""
        n = self.root_children.getCount()
        for i in range(n):
            node = self.root_children.getMFNode(i)
            if node.getTypeName() == type_name:
                return node
        return None

    def _find_node_by_name(self, name: str):
        """Find node by name field."""
        n = self.root_children.getCount()
        for i in range(n):
            node = self.root_children.getMFNode(i)
            name_field = node.getField("name")
            if name_field and name_field.getSFString() == name:
                return node
        return None

    def step(self) -> bool:
        """Advance simulation one step."""
        return self.supervisor.step(self.timestep) != -1

    def read_status(self) -> dict:
        """Read YouBot status from JSON file."""
        try:
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            pass
        return {}

    def write_status_reset(self):
        """Reset status file to trigger new test cycle."""
        try:
            status = {
                "version": "GRASP_TEST_V3",
                "state": "INIT",
                "test_index": self.current_test,
                "test_angle": TEST_POSITIONS[self.current_test][2] if self.current_test < len(TEST_POSITIONS) else "DONE",
                "supervisor_reset": True,
                "timestamp": time.time()
            }
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
            print(f"[SUPERVISOR] Status reset for test {self.current_test + 1}")
        except Exception as e:
            print(f"[SUPERVISOR] Error writing status: {e}")

    def save_results(self):
        """Save test results to JSON."""
        try:
            results = {
                "version": "GRASP_TEST_V3",
                "total_tests": len(TEST_POSITIONS),
                "completed": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r['success']),
                "failed": sum(1 for r in self.test_results if not r['success']),
                "tests": self.test_results,
                "timestamp": time.time()
            }
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[SUPERVISOR] Results saved to {RESULTS_FILE}")
        except Exception as e:
            print(f"[SUPERVISOR] Error saving results: {e}")

    def remove_cube(self):
        """Remove current test cube from world."""
        if self.cube_node:
            try:
                self.cube_node.remove()
                print("[SUPERVISOR] Cube removed")
            except:
                pass
            self.cube_node = None

        # Also try to find and remove any test cube by name
        cube = self._find_node_by_name("test_cube_v3")
        if cube:
            try:
                cube.remove()
            except:
                pass

    def spawn_cube(self, x: float, y: float, angle_name: str):
        """Spawn test cube at specified position."""
        # Remove any existing cube first
        self.remove_cube()

        # Wait a bit for removal to take effect
        for _ in range(10):
            self.step()

        # Cube node definition - matching original test_cube_green exactly
        node_string = f"""
        Solid {{
          translation {x} {y} {CUBE_Z}
          name "test_cube_v3"
          children [
            Shape {{
              appearance PBRAppearance {{
                baseColor 0 1 0
                roughness 0.8
                metalness 0
              }}
              geometry Box {{ size {CUBE_SIZE} {CUBE_SIZE} {CUBE_SIZE} }}
            }}
          ]
          boundingObject Box {{ size {CUBE_SIZE} {CUBE_SIZE} {CUBE_SIZE} }}
          physics Physics {{
            density -1
            mass 0.03
          }}
        }}
        """

        self.root_children.importMFNodeFromString(-1, node_string)
        self.cube_node = self._find_node_by_name("test_cube_v3")

        # Calculate actual angle
        actual_angle = math.degrees(math.atan2(y - ROBOT_INIT['y'], x - ROBOT_INIT['x']))
        print(f"[SUPERVISOR] Cube spawned at ({x:.2f}, {y:.2f}) - {angle_name} (actual: {actual_angle:.1f}°)")

    def reset_robot(self):
        """Reset YouBot to initial position."""
        if not self.youbot:
            print("[SUPERVISOR] Cannot reset robot - not found")
            return

        try:
            # Reset translation
            trans_field = self.youbot.getField("translation")
            if trans_field:
                trans_field.setSFVec3f([ROBOT_INIT['x'], ROBOT_INIT['y'], ROBOT_INIT['z']])

            # Reset rotation
            rot_field = self.youbot.getField("rotation")
            if rot_field:
                rot_field.setSFRotation(ROBOT_INIT['rotation'])

            # Reset physics
            self.youbot.resetPhysics()

            print(f"[SUPERVISOR] Robot reset to ({ROBOT_INIT['x']}, {ROBOT_INIT['y']})")
        except Exception as e:
            print(f"[SUPERVISOR] Error resetting robot: {e}")

    def setup_test(self, test_index: int):
        """Setup for a specific test."""
        if test_index >= len(TEST_POSITIONS):
            return False

        x, y, angle_name = TEST_POSITIONS[test_index]

        print(f"\n{'='*60}")
        print(f"SETTING UP TEST {test_index + 1}/{len(TEST_POSITIONS)}: {angle_name}")
        print(f"{'='*60}")

        # Reset robot position
        self.reset_robot()

        # Wait for physics to settle
        for _ in range(20):
            self.step()

        # Spawn cube
        self.spawn_cube(x, y, angle_name)

        # Wait for physics
        for _ in range(20):
            self.step()

        # Reset status file
        self.write_status_reset()

        # Record start time
        self.test_start_time = time.time()

        return True

    def check_test_complete(self) -> tuple:
        """Check if current test is complete. Returns (done, success)."""
        status = self.read_status()
        state = status.get('state', '')
        test_index = status.get('test_index', -1)
        tests_completed = status.get('tests_completed', 0)

        # Only check completion if controller is working on current test
        # (test_index matches current_test)
        if test_index != self.current_test:
            # Controller hasn't started this test yet
            return False, False

        # Check if controller reports SUCCESS/FAILED
        if state == 'SUCCESS':
            return True, True
        elif state == 'FAILED':
            return True, False
        elif state == 'WAITING_RESET':
            # Controller finished - check if it completed this test
            # tests_completed should be >= current_test+1 (0-indexed vs count)
            if tests_completed >= self.current_test + 1:
                has_object = status.get('has_object', False)
                return True, has_object
            # Still processing previous test
            return False, False

        # Check timeout
        elapsed = time.time() - self.test_start_time
        if elapsed > self.test_timeout:
            print(f"[SUPERVISOR] Test {self.current_test + 1} TIMEOUT after {elapsed:.1f}s")
            return True, False

        return False, False

    def run(self):
        """Main supervisor loop."""
        print("\n[SUPERVISOR] Starting multi-angle test sequence...")
        print(f"[SUPERVISOR] Running {len(TEST_POSITIONS)} tests\n")

        # Initial warmup
        print("[SUPERVISOR] Warmup (2s)...")
        for _ in range(int(2000 / self.timestep)):
            if not self.step():
                return

        # Setup first test
        self.setup_test(0)

        # Main loop
        while self.step():
            done, success = self.check_test_complete()

            if done:
                # Record result
                x, y, angle_name = TEST_POSITIONS[self.current_test]
                result = {
                    'test_index': self.current_test + 1,
                    'angle': angle_name,
                    'cube_position': {'x': x, 'y': y},
                    'success': success,
                    'duration': time.time() - self.test_start_time
                }
                self.test_results.append(result)

                status_str = "SUCCESS" if success else "FAILED"
                print(f"\n[SUPERVISOR] Test {self.current_test + 1} ({angle_name}): {status_str}")

                # Move to next test
                self.current_test += 1

                if self.current_test >= len(TEST_POSITIONS):
                    # All tests complete
                    print(f"\n{'='*60}")
                    print("ALL TESTS COMPLETE!")
                    print(f"{'='*60}")
                    passed = sum(1 for r in self.test_results if r['success'])
                    print(f"Results: {passed}/{len(TEST_POSITIONS)} passed")
                    for r in self.test_results:
                        status_str = "PASS" if r['success'] else "FAIL"
                        print(f"  Test {r['test_index']} ({r['angle']}): {status_str}")
                    print(f"{'='*60}")

                    self.save_results()
                    break
                else:
                    # Wait a bit before next test
                    print("[SUPERVISOR] Waiting 3s before next test...")
                    for _ in range(int(3000 / self.timestep)):
                        if not self.step():
                            return

                    # Setup next test
                    self.setup_test(self.current_test)

        print("[SUPERVISOR] Supervisor complete.")


def main():
    supervisor = SupervisorTestV3()
    supervisor.run()


if __name__ == "__main__":
    main()
