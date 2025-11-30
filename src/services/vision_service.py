"""
VisionService - Stable cube tracking with position-based persistence

Fixes the oscillation problem where robot switched between multiple
same-color cubes causing "trembling" behavior.

Based on: Bradski & Kaehler (2008) - Learning OpenCV
"""

import math
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrackedCube:
    """
    Tracked cube with stable identifier.

    Unlike raw detections, TrackedCube persists across frames
    and uses position matching (not just color) to maintain identity.
    """
    track_id: int
    color: str              # 'green', 'blue', 'red'
    distance: float         # meters
    angle: float            # degrees (positive = right of center)
    confidence: float       # 0-1
    frames_tracked: int     # consecutive frames this cube tracked
    last_seen: float        # timestamp of last detection
    last_position: Tuple[float, float]  # (x, y) relative position

    @property
    def is_reliable(self) -> bool:
        """Cube tracked for enough frames to be reliable (~100ms)."""
        return self.frames_tracked >= 3


@dataclass
class VisionState:
    """Current vision state for debugging"""
    timestamp: float
    raw_detections: int     # Number of raw detections
    tracked_target: Optional[TrackedCube]
    frames_locked: int
    frames_lost: int


class VisionService:
    """
    Stable cube tracking with position-based matching.

    Key improvements over raw detection:
    1. Position-based matching (not just color)
    2. Persistence during temporary occlusion
    3. Single target output (no oscillation)
    4. Hysteresis for stable lock/unlock

    Usage:
        vision = VisionService(cube_detector, time_step)

        # In control loop
        vision.update(camera_image)
        target = vision.get_target()

        if target and target.is_reliable:
            print(f"Tracking {target.color} cube at {target.distance}m")
    """

    # Tracking parameters
    LOST_THRESHOLD = 30         # Frames to wait before declaring lost (~1s) - balance stability/responsiveness
    MIN_CONFIDENCE = 0.60       # Minimum confidence to accept detection
    POSITION_TOLERANCE = 0.30   # Max distance change to consider same cube (meters)
    ANGLE_TOLERANCE = 20.0      # Max angle change to consider same cube (degrees)
    MIN_FRAMES_RELIABLE = 10    # Frames needed for reliable tracking (~320ms)

    def __init__(self, cube_detector, time_step: int):
        """
        Initialize VisionService.

        Args:
            cube_detector: CubeDetector instance
            time_step: Simulation time step in ms
        """
        self.detector = cube_detector
        self.time_step = time_step

        # Tracking state
        self.tracked_target: Optional[TrackedCube] = None
        self.frames_lost = 0
        self.next_track_id = 1

        # Lock mode
        self._locked_color: Optional[str] = None

        # Debug
        self._last_raw_count = 0
        self._update_count = 0

    def update(self, camera_image: np.ndarray) -> Optional[TrackedCube]:
        """
        Process camera frame and update tracking.

        Args:
            camera_image: RGB image [H, W, 3] uint8

        Returns:
            Currently tracked cube (or None)
        """
        self._update_count += 1

        if camera_image is None or camera_image.size == 0:
            self._handle_no_detection()
            return self.tracked_target

        # Get raw detections
        detections = self.detector.detect(camera_image)
        self._last_raw_count = len(detections)

        # Filter by validity and confidence
        valid_detections = [d for d in detections
                           if d.is_valid and d.confidence >= self.MIN_CONFIDENCE]

        # Apply color lock if active
        if self._locked_color:
            valid_detections = [d for d in valid_detections
                               if d.color == self._locked_color]

        if self.tracked_target:
            self._update_tracked(valid_detections)
        else:
            self._acquire_target(valid_detections)

        return self.tracked_target

    def _update_tracked(self, detections: list) -> None:
        """Update existing tracked target with new detections."""
        if not self.tracked_target:
            return

        # Find best match by position
        match = self._find_match(detections, self.tracked_target)

        if match:
            # Update tracked cube with new detection
            self.tracked_target.distance = match.distance
            self.tracked_target.angle = match.angle
            self.tracked_target.confidence = match.confidence
            self.tracked_target.frames_tracked += 1
            self.tracked_target.last_seen = time.time()
            self.tracked_target.last_position = self._calc_position(match)
            self.frames_lost = 0
        else:
            # Cube not found this frame
            self.frames_lost += 1

            if self.frames_lost >= self.LOST_THRESHOLD:
                # Lost for too long - release tracking
                if self._update_count % 30 == 0:
                    print(f"[VisionService] Lost {self.tracked_target.color} cube "
                          f"(track_id={self.tracked_target.track_id}) after "
                          f"{self.LOST_THRESHOLD} frames")
                self.tracked_target = None
                self._locked_color = None
            # else: Keep returning last known position (persistence)

    def _acquire_target(self, detections: list) -> None:
        """Acquire new target from valid detections."""
        if not detections:
            return

        # Pick closest cube
        closest = min(detections, key=lambda d: d.distance)

        # Create new tracked cube
        self.tracked_target = TrackedCube(
            track_id=self.next_track_id,
            color=closest.color,
            distance=closest.distance,
            angle=closest.angle,
            confidence=closest.confidence,
            frames_tracked=1,
            last_seen=time.time(),
            last_position=self._calc_position(closest)
        )
        self.next_track_id += 1
        self.frames_lost = 0

        print(f"[VisionService] Acquired {closest.color} cube "
              f"(track_id={self.tracked_target.track_id}) "
              f"at {closest.distance:.2f}m, {closest.angle:.1f}°")

    def _find_match(self, detections: list, target: TrackedCube):
        """
        Find detection matching tracked target by POSITION, not just color.

        Args:
            detections: List of CubeDetection
            target: Current tracked cube

        Returns:
            Matching detection or None
        """
        # Filter by color first
        same_color = [d for d in detections if d.color == target.color]

        if not same_color:
            return None

        # Match by position similarity
        best_match = None
        best_score = float('inf')

        for d in same_color:
            # Calculate position difference
            angle_diff = abs(d.angle - target.angle)
            dist_diff = abs(d.distance - target.distance)

            # Weighted score (angle more important when close)
            score = angle_diff / self.ANGLE_TOLERANCE + dist_diff / self.POSITION_TOLERANCE

            if score < best_score:
                best_score = score
                best_match = d

        # Accept if within tolerance
        if best_score < 2.0:  # Sum of normalized differences < 2
            return best_match

        return None

    def _calc_position(self, detection) -> Tuple[float, float]:
        """Calculate relative (x, y) position from distance and angle."""
        angle_rad = math.radians(detection.angle)
        x = detection.distance * math.sin(angle_rad)
        y = detection.distance * math.cos(angle_rad)
        return (x, y)

    def _handle_no_detection(self) -> None:
        """Handle frame with no detections."""
        if self.tracked_target:
            self.frames_lost += 1
            if self.frames_lost >= self.LOST_THRESHOLD:
                self.tracked_target = None
                self._locked_color = None

    def get_target(self) -> Optional[TrackedCube]:
        """
        Get currently tracked target.

        Returns stable target or None.
        Unlike raw detection, this persists during brief occlusions.
        """
        return self.tracked_target

    def lock_color(self, color: str) -> None:
        """
        Lock tracking to specific color.

        Args:
            color: 'green', 'blue', or 'red'
        """
        self._locked_color = color
        print(f"[VisionService] Locked to {color} color")

    def unlock(self) -> None:
        """Release color lock and tracking."""
        if self._locked_color:
            print(f"[VisionService] Unlocked from {self._locked_color}")
        self._locked_color = None
        self.tracked_target = None
        self.frames_lost = 0

    def get_state(self) -> VisionState:
        """Get current vision state for debugging."""
        return VisionState(
            timestamp=time.time(),
            raw_detections=self._last_raw_count,
            tracked_target=self.tracked_target,
            frames_locked=self.tracked_target.frames_tracked if self.tracked_target else 0,
            frames_lost=self.frames_lost
        )

    # ==================== TEST METHODS ====================

    def test_stability(self, robot, camera, num_frames: int = 100) -> bool:
        """
        Test: Verify tracking stability with multiple cubes visible.

        Expected:
        - track_id stays constant for entire test
        - No switching between same-color cubes
        - frames_tracked increases monotonically

        Returns:
            True if stable
        """
        print("[VisionService] TEST: Tracking stability")
        print(f"  Running for {num_frames} frames...")

        initial_track_id = None
        switches = 0

        for i in range(num_frames):
            if robot.step(int(robot.getBasicTimeStep())) == -1:
                return False

            # Get camera image
            image = camera.getImage()
            if image:
                width = camera.getWidth()
                height = camera.getHeight()
                image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
                image_rgb = image_array[:, :, :3]

                self.update(image_rgb)

            target = self.get_target()

            if target:
                if initial_track_id is None:
                    initial_track_id = target.track_id
                    print(f"  Initial target: {target.color} (id={target.track_id})")
                elif target.track_id != initial_track_id:
                    switches += 1
                    print(f"  WARNING: Track switched from {initial_track_id} to {target.track_id}")
                    initial_track_id = target.track_id

            # Log every 20 frames
            if (i + 1) % 20 == 0:
                if target:
                    print(f"  Frame {i+1}: {target.color} id={target.track_id} "
                          f"frames={target.frames_tracked} "
                          f"dist={target.distance:.2f}m angle={target.angle:.1f}°")
                else:
                    print(f"  Frame {i+1}: No target")

        print(f"\n[VisionService] TEST RESULT: {switches} track switches")
        if switches == 0:
            print("[VisionService] TEST PASSED: Stable tracking")
            return True
        else:
            print("[VisionService] TEST FAILED: Unstable tracking")
            return False


# ==================== STANDALONE TEST ====================

def test_vision_service():
    """
    Standalone test for VisionService.

    Run: python -m src.services.vision_service --test stability
    """
    import sys

    try:
        from controller import Robot
    except ImportError:
        print("ERROR: Must run inside Webots simulation")
        return

    # Import cube detector
    sys.path.insert(0, '/Users/luisfelipesena/Development/Personal/projeto-final-ia/src')
    from perception.cube_detector import CubeDetector

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    # Get camera
    camera = robot.getDevice("camera")
    camera.enable(time_step)

    # Create detector and service
    detector = CubeDetector()
    vision = VisionService(detector, time_step)

    # Parse test type
    test_type = "stability"
    if len(sys.argv) > 1:
        if "--test" in sys.argv:
            idx = sys.argv.index("--test")
            if idx + 1 < len(sys.argv):
                test_type = sys.argv[idx + 1]

    print(f"[VisionService] Running test: {test_type}")

    # Warmup
    for _ in range(10):
        robot.step(time_step)

    if test_type == "stability":
        vision.test_stability(robot, camera, num_frames=100)
    else:
        print(f"Unknown test: {test_type}")
        print("Available tests: stability")


if __name__ == "__main__":
    test_vision_service()
