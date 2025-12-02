"""
YouBot Grasp Test V2 - Cube Detection + Grasp Integration

STEP 2 VALIDATION: Robot scans area, detects cube, approaches, and grasps.

Integrates:
- CubeDetector from src/perception for color detection
- Validated grasp sequence from GRASP_TEST.md
- Simple approach logic

Usage:
1. Set controller to "youbot_grasp_test_v2" in Webots
2. Run simulation
3. Robot will: scan → detect → approach → grasp
"""

import sys
import os
import json
import time
import math
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from controller import Robot

# Import from grasp_test (same directory level)
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'youbot_grasp_test'))
from arm import Arm
from base import Base
from gripper import Gripper

# Import cube detector
from perception.cube_detector import CubeDetector, CubeDetection

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "youbot_mcp", "data", "youbot")
os.makedirs(DATA_DIR, exist_ok=True)
STATUS_FILE = os.path.join(DATA_DIR, "status.json")


class GraspTestV2:
    """Grasp test with cube detection integration."""

    VERSION = "GRASP_TEST_V2"

    # States
    STATE_INIT = "INIT"
    STATE_SCANNING = "SCANNING"
    STATE_APPROACHING = "APPROACHING"
    STATE_ALIGNING = "ALIGNING"
    STATE_GRASPING = "GRASPING"
    STATE_SUCCESS = "SUCCESS"
    STATE_FAILED = "FAILED"

    # Validated parameters from GRASP_TEST.md
    APPROACH_SPEED = 0.06          # m/s
    # Use cube pixel size instead of distance estimate (more reliable)
    # Cube 3cm at 50cm = ~7px, at 25cm = ~14px, at 15cm = ~24px
    MIN_CUBE_SIZE = 5              # Minimum size to consider (filter noise)
    MAX_CUBE_SIZE = 40             # Maximum size (filter deposit boxes - they're huge)
    GRASP_READY_SIZE = 22          # Cube size when ready to grasp (~15cm away)
    GRASP_READY_ANGLE = 8.0        # Max angle to start grasp sequence
    FINAL_APPROACH_DISTANCE = 0.10 # 10cm final approach
    FINAL_APPROACH_TIME = 2.0      # 10cm at 5cm/s
    SCAN_OMEGA = 0.30              # rad/s rotation during scan
    OBJECT_THRESHOLD = 0.002       # Finger sensor threshold

    def __init__(self):
        print("=" * 60)
        print(f"YOUBOT GRASP TEST V2 - Cube Detection ({self.VERSION})")
        print("=" * 60)

        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Hardware
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)
        self.base = Base(self.robot)

        # Camera
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            print(f"[INIT] Camera: {self.camera.getWidth()}x{self.camera.getHeight()}")

        # Finger sensor
        self.finger_sensor = self.robot.getDevice("finger::leftsensor")
        if self.finger_sensor:
            self.finger_sensor.enable(self.time_step)
            print("[INIT] Finger sensor enabled")

        # Cube detector
        self.detector = CubeDetector(
            camera_width=self.camera.getWidth() if self.camera else 128,
            camera_height=self.camera.getHeight() if self.camera else 128
        )
        print("[INIT] CubeDetector initialized")

        # State
        self.state = self.STATE_INIT
        self.target_cube = None
        self.target_color = None
        self.scan_angle = 0.0
        self.approach_lost_frames = 0

        print("[INIT] Complete")
        print("=" * 60)

    def step(self, steps: int = 1) -> bool:
        for _ in range(steps):
            if self.robot.step(self.time_step) == -1:
                return False
        return True

    def wait_seconds(self, seconds: float) -> bool:
        steps = int(seconds * 1000 / self.time_step)
        return self.step(steps)

    def get_finger_position(self) -> float:
        if self.finger_sensor:
            return self.finger_sensor.getValue()
        return -1.0

    def has_object(self) -> bool:
        pos = self.get_finger_position()
        if pos >= 0:
            return pos > self.OBJECT_THRESHOLD
        return False

    def get_camera_image(self) -> np.ndarray:
        """Get camera image as BGR numpy array."""
        if not self.camera:
            return None
        image = self.camera.getImage()
        if not image:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        # Webots returns BGRA
        img = np.frombuffer(image, np.uint8).reshape((h, w, 4))
        return img[:, :, :3]  # BGR

    def detect_cubes(self) -> list:
        """Detect cubes in current camera view, filtering out deposit boxes."""
        image = self.get_camera_image()
        if image is None:
            return []

        detections = self.detector.detect(image)

        # Filter: only keep small objects (actual cubes, not deposit boxes)
        valid = []
        for det in detections:
            size = max(det.width, det.height)
            if self.MIN_CUBE_SIZE <= size <= self.MAX_CUBE_SIZE:
                valid.append(det)

        return valid

    def save_screenshot(self, name: str) -> str:
        """Save camera image."""
        try:
            import cv2
            image = self.get_camera_image()
            if image is not None:
                filepath = os.path.join(DATA_DIR, f"{name}.jpg")
                cv2.imwrite(filepath, image)
                print(f"[SCREENSHOT] {filepath}")
                return filepath
        except Exception as e:
            print(f"[SCREENSHOT] Error: {e}")
        return ""

    def write_status(self):
        """Write status to JSON."""
        try:
            status = {
                "version": self.VERSION,
                "state": self.state,
                "target_color": self.target_color,
                "target_distance": self.target_cube.distance if self.target_cube else None,
                "target_angle": self.target_cube.angle if self.target_cube else None,
                "finger_position": self.get_finger_position(),
                "has_object": self.has_object(),
                "timestamp": time.time()
            }
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            print(f"[STATUS] Error: {e}")

    # =========================================================================
    # STATE MACHINE
    # =========================================================================

    def run_scanning(self):
        """Scan for cubes by rotating."""
        detections = self.detect_cubes()

        if detections:
            # Found cube! Lock onto nearest
            self.target_cube = detections[0]
            self.target_color = self.target_cube.color
            self.base.reset()
            print(f"\n[SCAN] *** CUBE DETECTED ***")
            print(f"       Color: {self.target_color.upper()}")
            print(f"       Distance: {self.target_cube.distance:.2f}m")
            print(f"       Angle: {self.target_cube.angle:.1f}°")
            self.save_screenshot("v2_01_detected")
            self.state = self.STATE_APPROACHING
            return

        # Continue scanning
        self.base.move(0, 0, self.SCAN_OMEGA)
        self.scan_angle += self.SCAN_OMEGA * self.time_step / 1000.0

        # Full rotation without finding cube
        if self.scan_angle > 2 * math.pi:
            print("[SCAN] Full rotation complete, no cubes found")
            self.base.reset()
            self.state = self.STATE_FAILED

    def run_approaching(self):
        """Approach and align with target cube simultaneously."""
        detections = self.detect_cubes()

        # Find our target color
        target = None
        for det in detections:
            if det.color == self.target_color:
                target = det
                break

        if not target:
            self.approach_lost_frames += 1
            if self.approach_lost_frames > 60:  # ~1s
                print("[APPROACH] Lost target, back to scanning")
                self.base.reset()
                self.target_cube = None
                self.state = self.STATE_SCANNING
                self.scan_angle = 0
                return
            # Pure rotation to find target again
            omega = 0.2 if not hasattr(self, '_last_target_angle') or self._last_target_angle >= 0 else -0.2
            self.base.move(0, 0, omega)
            return

        self.approach_lost_frames = 0
        self.target_cube = target
        self._last_target_angle = target.angle

        cube_size = max(target.width, target.height)
        angle = target.angle

        # Check if ready to grasp: big enough AND centered
        if cube_size >= self.GRASP_READY_SIZE and abs(angle) <= self.GRASP_READY_ANGLE:
            self.base.reset()
            print(f"\n[APPROACH] Ready to grasp! Size: {cube_size}px, Angle: {angle:.1f}°")
            self.save_screenshot("v2_02_ready")
            self.state = self.STATE_GRASPING
            return

        # Log progress
        if not hasattr(self, '_last_approach_log') or time.time() - self._last_approach_log > 0.5:
            print(f"[APPROACH] Size: {cube_size}px/{self.GRASP_READY_SIZE}, Angle: {angle:.1f}°/{self.GRASP_READY_ANGLE}")
            self._last_approach_log = time.time()

        # Combined approach + alignment
        # ALWAYS move forward a bit to get closer, adjust rotation based on angle
        # Positive angle = cube to the right = need negative omega (turn right)

        if abs(angle) > 20:
            # Large angle: slow forward + strong rotation
            omega = -0.35 if angle > 0 else 0.35
            self.base.move(0.02, 0, omega)
        elif abs(angle) > 10:
            # Medium angle: medium forward + medium rotation
            omega = -0.25 if angle > 0 else 0.25
            self.base.move(0.04, 0, omega)
        else:
            # Small angle: fast forward + light rotation
            omega = -0.10 if angle > 0 else 0.10
            if abs(angle) < 3:
                omega = 0  # Don't oscillate when nearly centered
            self.base.move(self.APPROACH_SPEED, 0, omega)

    def run_aligning(self):
        """Fine-tune alignment before grasp."""
        detections = self.detect_cubes()
        target = None
        for det in detections:
            if det.color == self.target_color:
                target = det
                break

        if not target:
            print("[ALIGN] Lost target")
            self.state = self.STATE_SCANNING
            self.scan_angle = 0
            return

        self.target_cube = target
        cube_size = max(target.width, target.height)

        if abs(target.angle) <= self.ALIGN_THRESHOLD_DEG:
            self.base.reset()
            print(f"\n[ALIGN] Aligned! Angle: {target.angle:.1f}°, Cube size: {cube_size}px")
            self.save_screenshot("v2_03_aligned")
            self.state = self.STATE_GRASPING
            return

        # Fine rotation
        omega = -0.15 if target.angle > 0 else 0.15
        self.base.move(0, 0, omega)

    def run_grasping(self):
        """Execute validated grasp sequence from GRASP_TEST.md."""
        print("\n" + "=" * 60)
        print("EXECUTING GRASP SEQUENCE")
        print("=" * 60)

        # Step 1: Open gripper
        print("\n[GRASP 1] Opening gripper...")
        self.gripper.release()
        self.wait_seconds(1.0)

        # Step 2: Reset arm
        print("[GRASP 2] Resetting arm...")
        self.arm.set_height(Arm.RESET)
        self.wait_seconds(1.5)

        # Step 3: Lower arm to FRONT_FLOOR (gripper stays open)
        print("[GRASP 3] Lowering arm to FRONT_FLOOR...")
        self.gripper.release()  # Ensure open
        self.arm.set_height(Arm.FRONT_FLOOR)
        self.wait_seconds(2.5)
        self.save_screenshot("v2_04_arm_lowered")

        # Step 4: Forward approach (10cm at 5cm/s) - validated in GRASP_TEST.md
        print(f"[GRASP 4] Moving forward {self.FINAL_APPROACH_DISTANCE*100:.0f}cm...")
        self.base.move(0.05, 0, 0)  # 5cm/s - validated speed
        self.wait_seconds(self.FINAL_APPROACH_TIME)  # 2.0s = 10cm
        self.base.reset()
        self.wait_seconds(0.5)
        self.save_screenshot("v2_05_after_forward")

        # Step 5: Close gripper
        print("[GRASP 5] Closing gripper...")
        finger_before = self.get_finger_position()
        print(f"         Finger BEFORE: {finger_before:.4f}")
        self.gripper.grip()
        self.wait_seconds(1.5)
        finger_after = self.get_finger_position()
        print(f"         Finger AFTER: {finger_after:.4f}")
        self.save_screenshot("v2_06_after_grip")

        # Step 6: Check object
        print("[GRASP 6] Checking object...")
        has_obj = self.has_object()
        print(f"         has_object(): {has_obj}")

        # Step 7: Lift
        print("[GRASP 7] Lifting to FRONT_PLATE...")
        self.arm.set_height(Arm.FRONT_PLATE)
        self.wait_seconds(2.0)
        self.save_screenshot("v2_07_lifted")

        # Final result
        final_has_obj = self.has_object()
        final_finger = self.get_finger_position()

        print("\n" + "=" * 60)
        if final_has_obj:
            print(f"GRASP TEST V2: *** SUCCESS ***")
            print(f"Cube color: {self.target_color.upper()}")
            self.state = self.STATE_SUCCESS
        else:
            print("GRASP TEST V2: FAILED")
            self.state = self.STATE_FAILED
        print(f"Finger position: {final_finger:.4f}")
        print(f"Threshold: {self.OBJECT_THRESHOLD}")
        print("=" * 60)

        self.save_screenshot("v2_08_final")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Main control loop."""
        print("\n[RUN] Starting main loop...")
        print("[RUN] Waiting 3s for warmup...")

        # Warmup
        for _ in range(int(3000 / self.time_step)):
            if not self.step():
                return
            self.write_status()

        # Start scanning
        self.state = self.STATE_SCANNING
        print("\n[RUN] Starting cube search...")

        while self.step():
            self.write_status()

            if self.state == self.STATE_SCANNING:
                self.run_scanning()

            elif self.state == self.STATE_APPROACHING:
                self.run_approaching()

            elif self.state == self.STATE_GRASPING:
                self.run_grasping()

            elif self.state in [self.STATE_SUCCESS, self.STATE_FAILED]:
                # Done - only print once
                if not hasattr(self, '_test_complete_printed'):
                    print("\n[RUN] Test complete.")
                    self._test_complete_printed = True
                self.write_status()
                self.base.reset()


def main():
    controller = GraspTestV2()
    controller.run()


if __name__ == "__main__":
    main()
