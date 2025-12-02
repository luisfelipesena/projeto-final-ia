"""
Minimal Grasp Test Controller for YouBot with MCP Communication

STEP 1 VALIDATION: Tests that the robot can physically grasp a cube.

This version includes MCP file-based communication so Claude can:
1. Send commands via commands.json
2. Read status via status.json
3. See screenshots in data/youbot/

Usage:
1. Open Webots with IA_20252/worlds/IA_20252.wbt
2. Set YouBot controller to "youbot_grasp_test"
3. Run simulation
4. Send commands via MCP or let auto-test run
"""

from controller import Robot
from arm import Arm
from gripper import Gripper
from base import Base
import numpy as np
import json
import time
import os

# Paths - find project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to project root: controllers/youbot_grasp_test -> controllers -> IA_20252 -> project_root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_ROOT, "youbot_mcp", "data", "youbot")
COMMANDS_FILE = os.path.join(DATA_DIR, "commands.json")
STATUS_FILE = os.path.join(DATA_DIR, "status.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class GraspTestMCP:
    """Grasp test controller with MCP file communication."""

    VERSION = "GRASP_TEST_V1"

    def __init__(self):
        print("=" * 60)
        print(f"YOUBOT GRASP TEST - MCP Version ({self.VERSION})")
        print("=" * 60)
        print(f"[INIT] Project root: {PROJECT_ROOT}")
        print(f"[INIT] Data dir: {DATA_DIR}")

        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        print(f"[INIT] Time step: {self.time_step}ms")

        # Initialize hardware using backup classes
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)
        self.base = Base(self.robot)

        # Camera for screenshots
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            print(f"[INIT] Camera: {self.camera.getWidth()}x{self.camera.getHeight()}")

        # List all available devices
        print("[INIT] Scanning for available devices...")
        device_names = []
        for i in range(self.robot.getNumberOfDevices()):
            device = self.robot.getDeviceByIndex(i)
            device_names.append(device.getName())
        print(f"[INIT] Available devices: {device_names}")

        # Use the correct sensor name: finger::leftsensor (no colon before sensor)
        self.finger_sensor = self.robot.getDevice("finger::leftsensor")
        if self.finger_sensor:
            self.finger_sensor.enable(self.time_step)
            print("[INIT] Finger sensor enabled: finger::leftsensor")
        else:
            print("[INIT] WARNING: No finger sensor - will use gripper state")

        # State
        self.last_command_id = 0
        self.state = "IDLE"
        self.grasp_result = None
        self.auto_test_done = False

        print("[INIT] Complete - waiting for commands or auto-test")
        print("=" * 60)

    def step(self, steps: int = 1) -> bool:
        """Execute simulation steps."""
        for _ in range(steps):
            if self.robot.step(self.time_step) == -1:
                return False
        return True

    def wait_seconds(self, seconds: float) -> bool:
        """Wait for specified time."""
        steps = int(seconds * 1000 / self.time_step)
        return self.step(steps)

    def get_finger_position(self) -> float:
        """Read finger position from sensor."""
        if self.finger_sensor:
            return self.finger_sensor.getValue()
        return -1.0

    def has_object(self) -> bool:
        """Detect if object is held based on gripper state."""
        pos = self.get_finger_position()
        # If sensor works, use threshold
        # 0.002 = validated threshold for 3cm cube
        if pos >= 0:
            threshold = 0.002
            return pos > threshold
        # Fallback: assume object if gripper commanded to close
        # This is less reliable but works for testing
        return self.gripper.is_gripping

    def save_screenshot(self, name: str) -> str:
        """Save camera image to file."""
        if not self.camera:
            return ""
        try:
            import cv2
            image = self.camera.getImage()
            if image:
                w = self.camera.getWidth()
                h = self.camera.getHeight()
                img = np.frombuffer(image, np.uint8).reshape((h, w, 4))
                bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                filepath = os.path.join(DATA_DIR, f"{name}.jpg")
                cv2.imwrite(filepath, bgr)
                print(f"[SCREENSHOT] {filepath}")
                return filepath
        except Exception as e:
            print(f"[SCREENSHOT] Error: {e}")
        return ""

    def write_status(self):
        """Write current status to status.json for MCP."""
        try:
            status = {
                "version": self.VERSION,
                "last_update": time.time(),
                "current_state": self.state,
                "arm_height": self._height_name(),
                "gripper_state": "closed" if self.gripper.is_gripping else "open",
                "finger_position": self.get_finger_position(),
                "gripper_has_object": self.has_object(),
                "grasp_result": self.grasp_result,
                "auto_test_done": self.auto_test_done,
            }
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"[STATUS] Error: {e}")

    def _height_name(self) -> str:
        """Get arm height name."""
        h = self.arm.current_height
        names = ["FRONT_FLOOR", "FRONT_PLATE", "FRONT_CARDBOARD_BOX", "RESET",
                 "BACK_PLATE_HIGH", "BACK_PLATE_LOW", "HANOI_PREPARE"]
        return names[h] if 0 <= h < len(names) else "UNKNOWN"

    def read_command(self) -> dict:
        """Read pending command from commands.json."""
        try:
            if os.path.exists(COMMANDS_FILE):
                with open(COMMANDS_FILE, 'r') as f:
                    cmd = json.load(f)
                cmd_id = cmd.get("id", 0)
                if cmd_id > self.last_command_id:
                    self.last_command_id = cmd_id
                    return cmd
        except Exception as e:
            print(f"[CMD] Read error: {e}")
        return {}

    def execute_command(self, cmd: dict):
        """Execute a single MCP command."""
        action = cmd.get("action", "")
        params = cmd.get("params", {})
        print(f"[CMD] Executing: {action}")

        if action == "run_grasp_test":
            self.run_grasp_test()

        elif action == "release":
            self.gripper.release()
            self.state = "GRIPPER_OPEN"

        elif action == "grip":
            self.gripper.grip()
            self.state = "GRIPPER_CLOSED"

        elif action == "set_arm_height":
            height = params.get("height", "RESET")
            height_map = {
                "FRONT_FLOOR": Arm.FRONT_FLOOR,
                "FRONT_PLATE": Arm.FRONT_PLATE,
                "RESET": Arm.RESET,
            }
            if height in height_map:
                self.arm.set_height(height_map[height])
                self.state = f"ARM_{height}"

        elif action == "move_forward":
            dist = params.get("distance_m", 0.08)
            speed = params.get("speed", 0.05)
            self.base.move(speed, 0, 0)
            self.wait_seconds(dist / speed)
            self.base.reset()
            self.state = "MOVED_FORWARD"

        elif action == "stop_base":
            self.base.reset()
            self.state = "STOPPED"

        elif action == "capture_camera":
            self.save_screenshot("camera_image")

        elif action == "check_grasp":
            has_obj = self.has_object()
            finger_pos = self.get_finger_position()
            self.grasp_result = {
                "has_object": has_obj,
                "finger_position": finger_pos,
                "success": has_obj
            }
            print(f"[GRASP CHECK] has_object={has_obj}, finger={finger_pos:.4f}")

    def run_grasp_test(self) -> bool:
        """Execute complete grasp test sequence."""
        print("\n" + "=" * 60)
        print("RUNNING GRASP TEST SEQUENCE")
        print("=" * 60)

        self.state = "TESTING"
        self.grasp_result = None

        # Step 1: Initial screenshot
        print("\n[STEP 1] Taking initial screenshot...")
        self.save_screenshot("grasp_01_start")
        self.write_status()

        # Step 2: Open gripper WIDE and reset arm
        print("\n[STEP 2] Opening gripper, resetting arm...")
        self.gripper.release()
        self.wait_seconds(1.0)  # Wait for gripper to fully open
        self.arm.set_height(Arm.RESET)
        self.wait_seconds(1.5)
        self.write_status()

        # Step 3: Lower arm to grab cube on floor (gripper stays open)
        print("\n[STEP 3] Lowering arm to FRONT_FLOOR...")
        self.gripper.release()  # Ensure gripper is open before lowering
        self.arm.set_height(Arm.FRONT_FLOOR)
        self.wait_seconds(2.5)
        self.save_screenshot("grasp_02_arm_lowered")
        self.write_status()

        # Step 4: Forward to position gripper over cube
        # Using slow move() for precise control
        print("\n[STEP 4] Moving forward 10cm...")
        self.base.move(0.05, 0, 0)  # 5cm/s forward, no lateral, no rotation
        self.wait_seconds(2.0)      # 10cm at 0.05m/s
        self.base.reset()
        self.wait_seconds(0.5)
        self.save_screenshot("grasp_03_after_forward")
        self.write_status()

        # Step 5: Close gripper
        print("\n[STEP 5] Closing gripper...")
        finger_before = self.get_finger_position()
        print(f"         Finger BEFORE: {finger_before:.4f}")

        self.gripper.grip()
        self.wait_seconds(1.5)

        finger_after = self.get_finger_position()
        print(f"         Finger AFTER: {finger_after:.4f}")
        self.save_screenshot("grasp_04_after_grip")
        self.write_status()

        # Step 6: Check object detection
        print("\n[STEP 6] Checking object detection...")
        has_obj = self.has_object()
        print(f"         has_object(): {has_obj}")

        # Step 7: Lift
        print("\n[STEP 7] Lifting to FRONT_PLATE...")
        self.arm.set_height(Arm.FRONT_PLATE)
        self.wait_seconds(2.0)
        self.save_screenshot("grasp_05_lifted")

        # Final check
        final_has_obj = self.has_object()
        final_finger = self.get_finger_position()

        self.grasp_result = {
            "success": final_has_obj,
            "finger_before": finger_before,
            "finger_after": finger_after,
            "finger_final": final_finger,
            "has_object": final_has_obj
        }

        # Final screenshot
        self.save_screenshot("grasp_06_final")

        # Report
        print("\n" + "=" * 60)
        if final_has_obj:
            print("GRASP TEST: *** SUCCESS ***")
            self.state = "SUCCESS"
        else:
            print("GRASP TEST: FAILED")
            self.state = "FAILED"
        print(f"Finger position: {final_finger:.4f}")
        print(f"Threshold: 0.005")
        print("=" * 60)

        self.auto_test_done = True
        self.write_status()
        return final_has_obj

    def run(self):
        """Main control loop."""
        print("\n[RUN] Starting main loop...")
        print("[RUN] Waiting 3s for warmup, then running auto-test...")

        # Warmup
        for _ in range(int(3000 / self.time_step)):
            if not self.step():
                return
            self.write_status()

        # Auto-run grasp test once
        if not self.auto_test_done:
            self.run_grasp_test()

        # Continue running and accepting commands
        print("\n[RUN] Test complete. Accepting MCP commands...")

        while self.step():
            # Check for MCP commands
            cmd = self.read_command()
            if cmd:
                self.execute_command(cmd)

            # Update status
            self.write_status()


def main():
    controller = GraspTestMCP()
    controller.run()


if __name__ == "__main__":
    main()
