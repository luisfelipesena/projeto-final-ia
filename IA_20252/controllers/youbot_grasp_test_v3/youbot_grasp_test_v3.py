"""
YouBot Grasp Test V3 - Angular Validation Controller

Validates grasp at different angles. Robot must:
1. SCAN to find cube (may require rotation)
2. APPROACH with angle correction
3. GRASP and verify

Based on V2, optimized for any angle approach.
"""

import sys
import os
import json
import time
import math
import numpy as np
from typing import Tuple

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


class GraspTestV3:
    """Grasp test V3 - handles any angle approach."""

    VERSION = "GRASP_TEST_V3"

    # States
    STATE_INIT = "INIT"
    STATE_SCANNING = "SCANNING"
    STATE_APPROACHING = "APPROACHING"
    STATE_GRASPING = "GRASPING"
    STATE_SUCCESS = "SUCCESS"
    STATE_FAILED = "FAILED"

    # Parameters - V2-style rotation approach (proven working)
    APPROACH_SPEED = 0.06  # m/s forward during fine approach
    MIN_CUBE_SIZE = 8  # Cube at 0.6m is ~15px minimum
    MAX_CUBE_SIZE = 22  # STRICT - cube at 0.6m is ~15-20px, deposit box is larger
    GRASP_READY_SIZE = 16  # px - start grasp when cube is ~16px (closer)
    GRASP_READY_ANGLE = 5.0  # degrees - relaxed for 90° approach
    FINAL_APPROACH_DISTANCE = 0.10
    FINAL_APPROACH_TIME = 2.0  # Time to reach cube
    SCAN_OMEGA = 0.30  # Slightly slower for better detection at 90°
    OBJECT_THRESHOLD = 0.002  # Same as V1 - validated for 3cm cube
    MAX_SCAN_TIME = 40.0  # Max 40s scanning - allow more time for 90° rotation
    MAX_APPROACH_TIME = 40.0  # Max 40s approaching
    SAFE_DISTANCE = 0.35  # m - min obstacle distance during approach
    # Y position filter: cubes on floor appear in lower portion
    # But distant cubes appear higher, so be lenient
    MIN_CUBE_Y_RATIO = 0.30  # Relaxed: cube center must be below 30% from top
    # Distance filter: deposit boxes are FAR (>1.5m), cubes are NEAR
    # 30° cube at ~0.58m, 90° cube at 0.60m
    MAX_INITIAL_DISTANCE = 0.9  # m - allow slightly farther for 90° test
    MAX_SCAN_SIZE = 25  # Size filter for scan - cube at 0.6m is ~18-21px

    def __init__(self):
        print("=" * 60)
        print(f"YOUBOT GRASP TEST V3 - Angular Approach ({self.VERSION})")
        print("BUILD: 2024-12-02-v36-14CM")  # Unique build ID
        print("STRATEGY: arm2=-1.10 (GROUND), 14cm forward")
        print("=" * 60)

        # Write to file to confirm code is loaded
        try:
            with open(os.path.join(DATA_DIR, "controller_loaded.txt"), "w") as f:
                f.write(f"V3 Controller loaded at {time.time()}\n")
                f.write("BUILD: 2024-12-02-v36-14CM\n")
        except Exception as e:
            print(f"[DEBUG] Could not write load marker: {e}")

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

        # LIDAR for obstacle detection
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            print("[INIT] LIDAR enabled")

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
        self.scan_start_time = 0
        self.approach_start_time = 0
        self.approach_lost_frames = 0
        self._last_target_angle = 0

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
        if not self.camera:
            return None
        image = self.camera.getImage()
        if not image:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        img = np.frombuffer(image, np.uint8).reshape((h, w, 4))
        return img[:, :, :3]

    def _is_valid_cube(self, det) -> Tuple[bool, str]:
        """Filter to distinguish 3cm cube from large deposit box.

        STRATEGY: The deposit box is LARGE (appears 20-30px even from far).
        The cube is SMALL (3cm at 0.6m = ~6px, gets bigger as we approach).

        Key insight from research:
        - Deposit box at 2.5m appears as ~20-28px blob (edge visible)
        - Cube at 0.6m appears as ~6px, at 0.3m as ~12px, at 0.2m as ~18px

        So we need to find SMALL green blobs, not large ones!

        Returns:
            Tuple of (is_valid, reason_if_rejected)
        """
        size = max(det.width, det.height)
        min_dim = min(det.width, det.height)
        aspect = size / (min_dim + 0.001)

        # CRITICAL: Cube is SMALL, deposit box edge is LARGE
        # Cube at 0.6m = ~6px, at 0.4m = ~9px, at 0.3m = ~12px
        # Deposit box edge = 20-30px regardless of robot rotation
        MIN_SIZE = 5   # Cube might be very small at distance
        MAX_SIZE = 18  # Larger than this = deposit box edge!

        if size < MIN_SIZE:
            return False, f"TOO_SMALL({size}<{MIN_SIZE})"
        if size > MAX_SIZE:
            return False, f"TOO_BIG({size}>{MAX_SIZE})=deposit_box"

        # Aspect ratio - cube is square
        if aspect > 1.6:
            return False, f"BAD_ASPECT({aspect:.1f}>1.6)"

        # Y-position - cube on floor is in lower half
        cam_height = self.camera.getHeight() if self.camera else 128
        y_ratio = det.center_y / cam_height
        if y_ratio < 0.35:
            return False, f"TOO_HIGH(y={y_ratio:.2f}<0.35)"

        return True, "OK"

    def detect_cubes(self) -> list:
        image = self.get_camera_image()
        if image is None:
            return []

        detections = self.detector.detect(image)
        valid = []
        cam_height = self.camera.getHeight() if self.camera else 128

        for det in detections:
            size = max(det.width, det.height)
            # Basic size filter - more detailed filtering in _is_valid_cube
            if size < 5 or size > 50:
                continue
            valid.append(det)

        # Debug: log all raw detections periodically
        if not hasattr(self, '_last_detect_log') or time.time() - self._last_detect_log > 1.0:
            if detections:
                print(f"[DETECT] Raw: {len(detections)}, Valid: {len(valid)}")
                for d in detections[:5]:
                    sz = max(d.width, d.height)
                    aspect = max(d.width, d.height) / (min(d.width, d.height) + 0.001)
                    is_valid, reason = self._is_valid_cube(d)
                    print(f"         {d.color}: {d.width}x{d.height}={d.area}px², asp={aspect:.1f}, y={d.center_y:.0f} [{reason}]")
            self._last_detect_log = time.time()

        return valid

    def save_screenshot(self, name: str) -> str:
        try:
            import cv2
            image = self.get_camera_image()
            if image is not None:
                filepath = os.path.join(DATA_DIR, f"v3_{name}.jpg")
                cv2.imwrite(filepath, image)
                print(f"[SCREENSHOT] {filepath}")
                return filepath
        except Exception as e:
            print(f"[SCREENSHOT] Error: {e}")
        return ""

    def write_status(self):
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
    # OBSTACLE DETECTION
    # =========================================================================

    def _check_front_clearance(self) -> bool:
        """Check LIDAR front sector for obstacles.

        Returns:
            True if front is clear (> SAFE_DISTANCE), False if obstacle detected.
        """
        if not self.lidar:
            return True  # No LIDAR, assume clear

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return True

        n = len(ranges)
        # LIDAR has 512 points over 360°
        # Front sector = ±30° = 60° total
        points_per_degree = n / 360.0
        sector_half = int(30 * points_per_degree)  # 30° each side
        center = n // 2

        # Get front sector readings
        start = max(0, center - sector_half)
        end = min(n, center + sector_half)
        front_ranges = ranges[start:end]

        if not front_ranges:
            return True

        # Filter out invalid readings (0 or very large)
        valid_ranges = [r for r in front_ranges if 0.01 < r < 5.0]
        if not valid_ranges:
            return True

        min_dist = min(valid_ranges)
        return min_dist > self.SAFE_DISTANCE

    # =========================================================================
    # STATE MACHINE
    # =========================================================================

    def run_scanning(self):
        """Scan for cubes by rotating CCW. Stop when valid GREEN cube found."""
        # Initialize scan state
        if not hasattr(self, '_scan_screenshot_count'):
            self._scan_screenshot_count = 0
            self._scan_candidates = []  # Track consistent detections
            self._rotation_accumulated = 0.0  # Track how much we've rotated
            self._last_scan_time = time.time()
            print("[SCAN] Starting rotation scan (CCW to find GREEN cube)...")
            print("[SCAN] Cube at 90° requires ~90° CCW rotation to see it")

        detections = self.detect_cubes()

        # Debug: save screenshot periodically during scan
        elapsed = time.time() - self.scan_start_time
        if self._scan_screenshot_count < 10 and elapsed > self._scan_screenshot_count * 2.0:
            self.save_screenshot(f"scan_{self._scan_screenshot_count}")
            self._scan_screenshot_count += 1

        # Track rotation accumulation (for 90° test, need to rotate ~90° to see cube)
        current_time = time.time()
        dt = current_time - self._last_scan_time
        self._last_scan_time = current_time
        # omega = SCAN_OMEGA rad/s, rotating CCW (negative)
        self._rotation_accumulated += self.SCAN_OMEGA * dt  # in radians
        rotation_deg = math.degrees(self._rotation_accumulated)

        # === MULTI-FILTER CUBE DETECTION ===
        # Problem: Deposit box edges look like small cubes
        # Solution: Use aspect ratio + Y-position + size to distinguish
        #
        # Cube characteristics:
        # - Square aspect ratio (~1.0-1.3)
        # - Appears in lower portion of image (Y > 42%)
        # - Size 8-25px at detection distance
        #
        # Deposit box edge characteristics:
        # - Tall/narrow aspect ratio (1.4-2.5)
        # - Appears higher in image (Y < 45%)
        # - Size overlaps with cube but irregular shape

        # Use size-based filtering: cube is SMALL (5-18px), deposit box is BIG (>20px)
        # No rotation minimum needed - the size filter will reject deposit box

        # Log rotation progress
        if int(rotation_deg) % 15 == 0 and not hasattr(self, f'_logged_{int(rotation_deg)}'):
            setattr(self, f'_logged_{int(rotation_deg)}', True)
            print(f"[SCAN] Rotation: {rotation_deg:.1f}°")

        # Log all green detections for debugging
        green_detections = [d for d in detections if d.color == 'green']
        if green_detections:
            print(f"[SCAN] Found {len(green_detections)} green objects at rotation {rotation_deg:.1f}°:")
            for d in green_detections:
                sz = max(d.width, d.height)
                asp = sz / (min(d.width, d.height) + 0.001)
                is_valid, reason = self._is_valid_cube(d)
                print(f"       - size={sz}px, asp={asp:.1f}, dist={d.distance:.2f}m, angle={d.angle:.1f}° [{reason}]")

        # Apply size-based filtering (no rotation limit - size filter rejects deposit box)
        if True:
            # Apply multi-filter validation
            valid_greens = []
            for det in detections:
                if det.color != 'green':
                    continue

                is_valid, reason = self._is_valid_cube(det)
                cube_size = max(det.width, det.height)

                if not is_valid:
                    print(f"[SCAN] REJECT green: {reason}")
                    continue

                # Passed all filters!
                valid_greens.append(det)
                print(f"[SCAN] *** VALID GREEN ***: size={cube_size}px, rotation={rotation_deg:.1f}°")

        # Lock on first valid green
        target = None
        if valid_greens:
            self.base.reset()
            best = min(valid_greens, key=lambda d: d.distance)
            cube_size = max(best.width, best.height)
            print(f"[SCAN] *** FOUND CUBE *** size={cube_size}px, angle={best.angle:.1f}°")
            target = best

        if target:
            self.target_cube = target
            self.target_color = self.target_cube.color
            self.base.reset()

            print("\n[SCAN] *** CUBE DETECTED ***")
            print(f"       Color: {self.target_color.upper()}")
            print(f"       Distance: {self.target_cube.distance:.2f}m")
            print(f"       Angle: {self.target_cube.angle:.1f}°")
            print(f"       Size: {max(self.target_cube.width, self.target_cube.height)}px")
            print(f"       Center X: {self.target_cube.center_x:.0f}px")
            print(f"       Center Y: {self.target_cube.center_y:.0f}px")
            print(f"       Robot rotated: {rotation_deg:.1f}° CCW")

            self.save_screenshot("01_detected")
            self.state = self.STATE_APPROACHING
            self.approach_start_time = time.time()
            return

        # No target found yet - keep rotating (CCW = negative omega)
        # Negative omega rotates counter-clockwise, which should find cube on LEFT first
        self.base.move(0, 0, -self.SCAN_OMEGA)  # CCW rotation

        # Timeout check
        if elapsed > self.MAX_SCAN_TIME:
            print(f"[SCAN] Timeout after {elapsed:.1f}s - no cube found")
            self.base.reset()
            self.state = self.STATE_FAILED

    def run_approaching(self):
        """Approach cube with V2-style rotation alignment (proven working)."""
        # Check approach timeout
        elapsed = time.time() - self.approach_start_time
        if elapsed > self.MAX_APPROACH_TIME:
            print(f"[APPROACH] Timeout after {elapsed:.1f}s")
            self.base.reset()
            self.state = self.STATE_FAILED
            return

        detections = self.detect_cubes()

        # Find target by LOCKED color (don't switch targets!)
        # Also validate it's a real cube, not a deposit box
        target = None
        for det in detections:
            if det.color == self.target_color:
                is_valid, reason = self._is_valid_cube(det)
                if is_valid:
                    target = det
                    break
                else:
                    # Log rejected detection (might be deposit box)
                    print(f"[APPROACH] Rejected {det.color}: {reason}")

        if not target:
            self.approach_lost_frames += 1

            # If target lost, continue in last known direction
            # DON'T switch to another color!
            if self.approach_lost_frames > 90:  # 90 frames = ~1.5s at 60fps
                print(f"[APPROACH] Lost {self.target_color} for too long, back to scan")
                self.base.reset()
                self.target_cube = None
                self.state = self.STATE_SCANNING
                self.scan_start_time = time.time()
                return

            # Keep moving in last known direction (proportional to last angle)
            last_omega = self._last_target_angle * 0.02
            last_omega = max(-0.3, min(0.3, last_omega))
            self.base.move(0.03, 0, last_omega)  # Slow forward + rotation
            print(f"[APPROACH] Target lost ({self.approach_lost_frames}/90), continuing with omega={last_omega:.2f}")
            return

        # Target found - reset lost counter and update
        self.approach_lost_frames = 0
        self.target_cube = target
        self._last_target_angle = target.angle

        cube_size = max(target.width, target.height)
        angle = target.angle

        # Check for obstacles when angle is small (moving mostly forward)
        if abs(angle) < 15:
            if not self._check_front_clearance():
                self.base.reset()
                print("[APPROACH] Obstacle ahead! Pausing.")
                return

        # Log progress periodically
        if not hasattr(self, '_last_approach_log') or time.time() - self._last_approach_log > 0.5:
            print(f"[APPROACH] Size: {cube_size}px/{self.GRASP_READY_SIZE}, Angle: {angle:.1f}°/±{self.GRASP_READY_ANGLE}°")
            self._last_approach_log = time.time()

        # === ROTATION APPROACH WITH FORWARD BIAS ===
        # Mecanum kinematics: POSITIVE omega = rotate CW (right) from above
        # So: cube on RIGHT (angle>0) → rotate RIGHT → POSITIVE omega
        #     cube on LEFT  (angle<0) → rotate LEFT  → NEGATIVE omega
        # Therefore: omega = angle * k (SAME SIGN)

        # Calculate omega proportional to angle
        omega = angle * 0.02  # ~0.4 rad/s at 20°
        omega = max(-0.4, min(0.4, omega))  # Clamp

        # Forward speed based on alignment
        if abs(angle) > 20:
            vx = 0.03  # Slow forward at large angle
        elif abs(angle) > 10:
            vx = 0.05  # Medium forward
        else:
            vx = self.APPROACH_SPEED  # Full speed when aligned
            # Only stop rotation when VERY centered (< 1.5°)
            if abs(angle) < 1.5:
                omega = 0

        print(f"[MOVE] angle={angle:.1f}°, vx={vx:.2f}, omega={omega:.2f}")
        self.base.move(vx, 0, omega)

        # Check if ready to grasp - use ANGLE (not pixel offset!)
        if cube_size >= self.GRASP_READY_SIZE and abs(angle) <= self.GRASP_READY_ANGLE:
            self.base.reset()
            print("\n[APPROACH] *** READY TO GRASP ***")
            print(f"           Size: {cube_size}px (threshold: {self.GRASP_READY_SIZE})")
            print(f"           Angle: {angle:.1f}° (threshold: ±{self.GRASP_READY_ANGLE}°)")
            self._last_cube_size = cube_size  # Save for grasping phase
            self.save_screenshot("02_ready")
            self.state = self.STATE_GRASPING
            return

    def run_grasping(self):
        """Execute grasp sequence."""
        print("\n" + "=" * 60)
        print("EXECUTING GRASP SEQUENCE")
        print("=" * 60)

        # Check if too close - backup if needed
        TOO_CLOSE_SIZE = 25  # px - if cube appears this big, we're too close
        if hasattr(self, '_last_cube_size') and self._last_cube_size > TOO_CLOSE_SIZE:
            print(f"[GRASP 0] Too close (size={self._last_cube_size}px > {TOO_CLOSE_SIZE}), backing up...")
            self.base.move(-0.03, 0, 0)  # Backward 3cm/s
            self.wait_seconds(1.0)       # = 3cm back
            self.base.reset()
            self.wait_seconds(0.3)

        # Step 1: Open gripper
        print("\n[GRASP 1] Opening gripper...")
        self.gripper.release()
        self.wait_seconds(1.0)

        # Step 2: Reset arm
        print("[GRASP 2] Resetting arm...")
        self.arm.set_height(Arm.RESET)
        self.wait_seconds(1.5)

        # Step 3: Lower arm to GROUND level - LOWER than FRONT_FLOOR
        # FRONT_FLOOR: arm2=-0.97, arm3=-1.55, arm4=-0.61
        # GROUND: arm2=-1.10 (near limit -1.13), arm3=-1.45, arm4=-0.70
        print("[GRASP 3] Lowering arm to GROUND position (custom)...")
        self.gripper.release()

        arm2 = self.robot.getDevice("arm2")
        arm3 = self.robot.getDevice("arm3")
        arm4 = self.robot.getDevice("arm4")

        if arm2:
            arm2.setPosition(-1.10)  # Shoulder LOWER (limit is -1.13)
        if arm3:
            arm3.setPosition(-1.45)  # Elbow
        if arm4:
            arm4.setPosition(-0.70)  # Wrist

        self.wait_seconds(2.5)
        self.save_screenshot("03_arm_lowered")

        # Step 4: Forward approach - 14cm (18cm was too far, cube passed through)
        print("[GRASP 4] Forward approach (14cm)...")
        self.base.move(0.05, 0, 0)
        self.wait_seconds(2.8)  # 2.8s @ 5cm/s = 14cm
        self.base.reset()
        self.wait_seconds(0.5)
        self.save_screenshot("04_after_forward")

        # Step 5: Close gripper
        print("[GRASP 5] Closing gripper...")
        finger_before = self.get_finger_position()
        print(f"         Finger BEFORE: {finger_before:.4f}")
        self.gripper.grip()
        self.wait_seconds(1.5)
        finger_after = self.get_finger_position()
        print(f"         Finger AFTER: {finger_after:.4f}")
        self.save_screenshot("05_after_grip")

        # Step 6: Check object
        print("[GRASP 6] Checking object...")
        has_obj = self.has_object()
        print(f"         has_object(): {has_obj}")

        # Step 7: Lift
        print("[GRASP 7] Lifting to FRONT_PLATE...")
        self.arm.set_height(Arm.FRONT_PLATE)
        self.wait_seconds(2.0)
        self.save_screenshot("06_lifted")

        # Final result
        final_has_obj = self.has_object()
        final_finger = self.get_finger_position()

        print("\n" + "=" * 60)
        if final_has_obj:
            print("RESULT: *** SUCCESS ***")
            print(f"Cube color: {self.target_color.upper()}")
            self.state = self.STATE_SUCCESS
        else:
            print("RESULT: FAILED - No object detected")
            self.state = self.STATE_FAILED
        print(f"Finger position: {final_finger:.4f}")
        print(f"Threshold: {self.OBJECT_THRESHOLD}")
        print("=" * 60)

        self.save_screenshot("07_final")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        print("\n[RUN] Starting V3 controller...")
        print("[RUN] Warmup (2s)...")

        # Warmup
        for _ in range(int(2000 / self.time_step)):
            if not self.step():
                return
            self.write_status()

        # Start scanning
        self.state = self.STATE_SCANNING
        self.scan_start_time = time.time()
        print("\n[RUN] Starting SCAN phase...")

        while self.step():
            self.write_status()

            # State machine
            if self.state == self.STATE_SCANNING:
                self.run_scanning()

            elif self.state == self.STATE_APPROACHING:
                self.run_approaching()

            elif self.state == self.STATE_GRASPING:
                self.run_grasping()

            elif self.state in [self.STATE_SUCCESS, self.STATE_FAILED]:
                # Done - hold position
                self.base.reset()
                self.write_status()
                print("\n[RUN] Test complete. Holding position.")

                # Keep running to maintain status
                while self.step():
                    self.write_status()
                break


def main():
    controller = GraspTestV3()
    controller.run()


if __name__ == "__main__":
    main()
