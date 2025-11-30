"""
NavigationService - Coordinate movement + vision for approach

Combines MovementService and VisionService for cube approach.
Based on: Latombe (1991) - Robot Motion Planning
"""

import math
import time
from typing import Optional
from dataclasses import dataclass
from enum import Enum, auto

from .movement_service import MovementService
from .vision_service import VisionService, TrackedCube


class ApproachPhase(Enum):
    """Current phase of approach"""
    ALIGN = auto()      # Turning to face target
    APPROACH = auto()   # Moving toward target
    FINAL = auto()      # Close enough for grasp
    LOST = auto()       # Target lost during approach
    COMPLETE = auto()   # Successfully positioned


@dataclass
class ApproachResult:
    """Result of approach attempt"""
    success: bool
    phase: ApproachPhase
    final_distance: float
    final_angle: float
    reason: str = ""


class NavigationService:
    """
    Coordinate movement and vision for cube approach.

    Implements a two-phase approach:
    1. ALIGN: Turn to face target (angle < 5°)
    2. APPROACH: Move forward until close (distance < 0.28m)

    Usage:
        nav = NavigationService(movement, vision, robot, time_step)

        # Start approach
        result = nav.approach_target()

        if result.success:
            print(f"Positioned at {result.final_distance}m")
            # Now safe to grasp
    """

    # Approach parameters
    ALIGN_THRESHOLD_ENTER = 20.0  # degrees - start considering aligned (forgiving for approach)
    ALIGN_THRESHOLD_EXIT = 12.0   # degrees - stop aligning (hysteresis prevents oscillation)
    APPROACH_DISTANCE = 0.22     # meters - stop at 22cm, then forward approach
    TURN_GAIN = 0.5              # Proportional gain for turning
    MIN_TURN = 5.0               # Minimum turn angle (degrees)
    APPROACH_STEP = 0.10         # Forward step size (meters)
    APPROACH_SPEED = 0.08        # Forward speed (m/s) - slower for precision
    MAX_ATTEMPTS = 100           # Max iterations before giving up

    def __init__(self, movement: MovementService, vision: VisionService,
                 robot, camera, time_step: int, lidar=None, lidar_model=None):
        """
        Initialize NavigationService.

        Args:
            movement: MovementService instance
            vision: VisionService instance
            robot: Webots Robot instance
            camera: Webots Camera device
            time_step: Simulation time step in ms
            lidar: Webots LIDAR device (optional)
            lidar_model: RNA model for obstacle detection (optional)
        """
        self.movement = movement
        self.vision = vision
        self.robot = robot
        self.camera = camera
        self.time_step = time_step
        self.lidar = lidar
        self.lidar_model = lidar_model

        self.phase = ApproachPhase.ALIGN
        self._attempts = 0

    def _get_front_obstacle_distance(self) -> float:
        """
        Get minimum distance to obstacle in front sectors (3-5).
        Returns float('inf') if no LIDAR available.
        """
        if not self.lidar:
            return float('inf')

        try:
            ranges = self.lidar.getRangeImage()
            if not ranges:
                return float('inf')

            # Front sectors: divide 512 points into 9 sectors, check sectors 3-5 (front)
            num_points = len(ranges)
            points_per_sector = num_points // 9

            # Sectors 3, 4, 5 = front
            front_ranges = []
            for sector in [3, 4, 5]:
                start = sector * points_per_sector
                end = start + points_per_sector
                front_ranges.extend(ranges[start:end])

            # Filter valid ranges
            valid = [r for r in front_ranges if 0.01 < r < 5.0]
            return min(valid) if valid else float('inf')
        except Exception as e:
            print(f"[Nav] LIDAR error: {e}")
            return float('inf')

    def _step(self) -> bool:
        """Execute simulation step and update vision."""
        if self.robot.step(self.time_step) == -1:
            return False

        # Update vision with new camera frame
        image = self.camera.getImage()
        if image:
            import numpy as np
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            image_rgb = image_array[:, :, :3]
            self.vision.update(image_rgb)

        return True

    def approach_target(self) -> ApproachResult:
        """
        Approach currently tracked target.

        Implements two-phase approach:
        1. ALIGN: Turn in place until facing target
        2. APPROACH: Move forward until close enough

        Returns:
            ApproachResult with success status
        """
        self._attempts = 0
        self.phase = ApproachPhase.ALIGN

        while self._attempts < self.MAX_ATTEMPTS:
            self._attempts += 1

            # Update vision
            if not self._step():
                return ApproachResult(
                    success=False,
                    phase=self.phase,
                    final_distance=float('inf'),
                    final_angle=0,
                    reason="simulation_ended"
                )

            target = self.vision.get_target()

            if not target:
                self.phase = ApproachPhase.LOST
                return ApproachResult(
                    success=False,
                    phase=self.phase,
                    final_distance=float('inf'),
                    final_angle=0,
                    reason="target_lost"
                )

            # Phase 1: ALIGN
            if self.phase == ApproachPhase.ALIGN:
                result = self._do_align(target)
                if result:
                    return result
                continue

            # Phase 2: APPROACH
            if self.phase == ApproachPhase.APPROACH:
                result = self._do_approach(target)
                if result:
                    return result
                continue

        # Max attempts exceeded
        target = self.vision.get_target()
        return ApproachResult(
            success=False,
            phase=self.phase,
            final_distance=target.distance if target else float('inf'),
            final_angle=target.angle if target else 0,
            reason="max_attempts"
        )

    def _do_align(self, target: TrackedCube) -> Optional[ApproachResult]:
        """
        Execute alignment phase: turn to face target.
        Uses ALIGN_THRESHOLD_EXIT (stricter) to transition to approach.

        Returns:
            ApproachResult if phase complete/failed, None to continue
        """
        if abs(target.angle) <= self.ALIGN_THRESHOLD_EXIT:
            # Well aligned - transition to approach (use stricter threshold)
            print(f"[Navigation] ALIGNED: angle={target.angle:.1f}° → APPROACH")
            self.phase = ApproachPhase.APPROACH
            return None

        # Calculate turn angle (proportional control)
        # POSITIVE angle = cube to RIGHT → need POSITIVE omega to turn RIGHT
        # (YouBot convention: positive omega = turn right/clockwise from above)
        turn_angle = target.angle * self.TURN_GAIN

        # Ensure minimum turn for responsiveness
        if abs(turn_angle) < self.MIN_TURN:
            turn_angle = self.MIN_TURN if turn_angle > 0 else -self.MIN_TURN

        # Limit turn angle
        turn_angle = max(-30, min(30, turn_angle))

        # Execute turn (non-blocking single step)
        self.movement.move_continuous(vx=0, vy=0, omega=math.radians(turn_angle) * 2)

        return None

    def _do_approach(self, target: TrackedCube) -> Optional[ApproachResult]:
        """
        Execute approach phase: move forward toward target.
        Uses lateral movement if obstacle blocks direct path.

        Returns:
            ApproachResult if phase complete/failed, None to continue
        """
        if target.distance <= self.APPROACH_DISTANCE:
            # Close enough - complete
            print(f"[Navigation] COMPLETE: dist={target.distance:.2f}m angle={target.angle:.1f}°")
            self.movement.stop()
            self.phase = ApproachPhase.COMPLETE
            return ApproachResult(
                success=True,
                phase=self.phase,
                final_distance=target.distance,
                final_angle=target.angle
            )

        # Check if still aligned (use ALIGN_THRESHOLD_ENTER for consistency)
        # Re-align if angle drifts beyond threshold during approach
        if abs(target.angle) > self.ALIGN_THRESHOLD_ENTER:
            # Lost alignment - go back to align phase
            print(f"[Navigation] Lost alignment: angle={target.angle:.1f}° → ALIGN")
            self.phase = ApproachPhase.ALIGN
            return None

        # Check for obstacle in front (but ignore if it's the cube we're approaching)
        front_obstacle = self._get_front_obstacle_distance()

        # Only trigger obstacle avoidance if:
        # 1. Obstacle is closer than target by significant margin (>0.15m difference)
        # 2. This indicates a different obstacle between robot and cube
        obstacle_is_different = (front_obstacle < target.distance - 0.15)

        if obstacle_is_different and front_obstacle < 0.25:
            # There's a different obstacle blocking the path to cube
            vy = 0.08 if target.angle > 0 else -0.08

            if self._attempts % 20 == 0:
                print(f"[Navigation] OBSTACLE at {front_obstacle:.2f}m (target at {target.distance:.2f}m) - lateral dodge")

            self.movement.move_continuous(vx=0.05, vy=vy, omega=0)
            return None

        # Normal approach: move forward with angle correction
        omega = target.angle * 0.05  # Proportional correction for better tracking

        # Log distance periodically
        if self._attempts % 10 == 0:
            print(f"[Navigation] APPROACH: dist={target.distance:.2f}m (target={self.APPROACH_DISTANCE:.2f}m)")

        self.movement.move_continuous(vx=self.APPROACH_SPEED, vy=0, omega=omega)

        return None

    def stop(self) -> None:
        """Stop any ongoing movement."""
        self.movement.stop()

    # ==================== TEST METHODS ====================

    def test_approach(self) -> bool:
        """
        Test: Approach a cube placed 2m in front.

        Expected:
        - Robot turns to face cube
        - Robot moves forward
        - Robot stops at ~0.28m

        Returns:
            True if approach successful
        """
        print("=" * 50)
        print("[NavigationService] TEST: Approach cube")
        print("  Place cube ~2m in front of robot")
        print("=" * 50)

        # Wait for vision to acquire target
        print("  Waiting for target...")
        for i in range(50):
            if not self._step():
                return False

            target = self.vision.get_target()
            if target and target.is_reliable:
                print(f"  Found: {target.color} at {target.distance:.2f}m, {target.angle:.1f}°")
                break
        else:
            print("[TEST] FAILED: No target found")
            return False

        # Execute approach
        print("\n[TEST] Starting approach...")
        result = self.approach_target()

        print(f"\n[TEST] Result:")
        print(f"  Success: {result.success}")
        print(f"  Phase: {result.phase.name}")
        print(f"  Final distance: {result.final_distance:.2f}m")
        print(f"  Final angle: {result.final_angle:.1f}°")
        if result.reason:
            print(f"  Reason: {result.reason}")

        if result.success:
            print("\n[NavigationService] TEST PASSED")
        else:
            print("\n[NavigationService] TEST FAILED")

        return result.success


# ==================== STANDALONE TEST ====================

def test_navigation_service():
    """
    Standalone test for NavigationService.

    Run: python -m src.services.navigation_service --test approach
    """
    import sys
    import numpy as np

    try:
        from controller import Robot
    except ImportError:
        print("ERROR: Must run inside Webots simulation")
        return

    # Import other services
    sys.path.insert(0, '/Users/luisfelipesena/Development/Personal/projeto-final-ia/src')
    from perception.cube_detector import CubeDetector

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # Get devices
    from IA_20252.controllers.youbot.base import Base
    base = Base(robot)

    camera = robot.getDevice("camera")
    camera.enable(time_step)

    # Create services
    from services.movement_service import MovementService
    from services.vision_service import VisionService

    movement = MovementService(base, robot, time_step)
    detector = CubeDetector()
    vision = VisionService(detector, time_step)
    navigation = NavigationService(movement, vision, robot, camera, time_step)

    # Parse test type
    test_type = "approach"
    if len(sys.argv) > 1:
        if "--test" in sys.argv:
            idx = sys.argv.index("--test")
            if idx + 1 < len(sys.argv):
                test_type = sys.argv[idx + 1]

    print(f"[NavigationService] Running test: {test_type}")

    # Warmup
    for _ in range(10):
        robot.step(time_step)

    if test_type == "approach":
        navigation.test_approach()
    else:
        print(f"Unknown test: {test_type}")
        print("Available tests: approach")


if __name__ == "__main__":
    test_navigation_service()
