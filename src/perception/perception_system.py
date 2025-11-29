"""
PerceptionSystem Module

Unified perception system integrating LIDAR obstacle detection
and camera-based cube detection.

Based on: Thrun et al. (2005) - Probabilistic Robotics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import time

from .lidar_processor import LIDARProcessor, ObstacleMap, HandCraftedFeatures
from .cube_detector import CubeDetector, CubeDetection


@dataclass
class PerceptionState:
    """
    Complete perception state at a given timestamp

    Combines LIDAR obstacle map with camera cube detections
    for unified world representation.
    """
    timestamp: float  # Unix timestamp

    # LIDAR-based obstacle detection
    obstacle_map: Optional[ObstacleMap] = None
    min_obstacle_distance: float = float('inf')
    obstacle_angle: float = 0.0  # Angle to closest obstacle (degrees)

    # Camera-based cube detection
    cube_detections: List[CubeDetection] = field(default_factory=list)
    closest_cube: Optional[CubeDetection] = None
    cube_detected: bool = False

    # Derived properties
    @property
    def is_path_clear(self) -> bool:
        """Check if forward path is clear (front sectors)"""
        if self.obstacle_map is None:
            return True
        return self.obstacle_map.is_path_clear([3, 4, 5], min_clearance=0.5)

    @property
    def should_avoid(self) -> bool:
        """Check if obstacle avoidance is needed"""
        return self.min_obstacle_distance < 0.3


class PerceptionSystem:
    """
    Unified perception system for YouBot

    Integrates:
    - LIDARProcessor: 9-sector obstacle detection from 667-point LIDAR
    - CubeDetector: Color classification and localization from RGB camera

    Contract Requirements:
    - MUST update at >10 Hz (SC-003, SC-004)
    - MUST provide unified PerceptionState
    - MUST handle sensor failures gracefully
    - MUST NOT use GPS (project requirement)

    Usage:
        perception = PerceptionSystem(
            lidar_model_path="models/lidar_net.pt",
            camera_model_path="models/camera_net.pt"
        )

        # In control loop:
        lidar_ranges = lidar.getRangeImage()
        camera_image = camera.getImage()

        state = perception.update(lidar_ranges, camera_image)

        if state.should_avoid:
            # Execute avoidance maneuver
            pass
        elif state.cube_detected:
            # Approach cube
            cube = state.closest_cube
            print(f"Cube at {cube.distance}m, {cube.angle}°")
    """

    def __init__(
        self,
        lidar_model_path: Optional[str] = None,
        camera_model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize perception system

        Args:
            lidar_model_path: Path to trained LIDAR model (optional)
            camera_model_path: Path to trained camera model (optional)
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = device

        # Initialize LIDAR processor
        self.lidar_processor: Optional[LIDARProcessor] = None
        if lidar_model_path and Path(lidar_model_path).exists():
            try:
                self.lidar_processor = LIDARProcessor(lidar_model_path, device)
            except Exception as e:
                print(f"PerceptionSystem: LIDAR model load failed: {e}")
        else:
            print("PerceptionSystem: Using heuristic LIDAR processing")

        # Initialize cube detector
        self.cube_detector = CubeDetector(camera_model_path, device)

        # State tracking
        self.last_state: Optional[PerceptionState] = None
        self.update_count = 0

        print("PerceptionSystem initialized")

    def process_lidar(self, ranges: np.ndarray) -> Tuple[Optional[ObstacleMap], float, float]:
        """
        Process LIDAR data

        Args:
            ranges: [667] LIDAR range measurements

        Returns:
            (obstacle_map, min_distance, angle_to_closest)
        """
        if ranges is None or len(ranges) == 0:
            return None, float('inf'), 0.0

        # Use neural network if available
        if self.lidar_processor is not None:
            obstacle_map = self.lidar_processor.process(ranges)
        else:
            # Heuristic fallback
            obstacle_map = self._heuristic_lidar_processing(ranges)

        # Find minimum distance and its angle
        min_dist = float('inf')
        min_angle = 0.0

        valid_ranges = ranges[np.isfinite(ranges)]
        if len(valid_ranges) > 0:
            min_idx = np.argmin(ranges)
            min_dist = ranges[min_idx]

            # Convert index to angle (270° FOV, centered at 0°)
            angle_per_point = 270.0 / len(ranges)
            min_angle = (min_idx - len(ranges) / 2) * angle_per_point

        return obstacle_map, min_dist, min_angle

    def _heuristic_lidar_processing(self, ranges: np.ndarray) -> ObstacleMap:
        """
        Simple heuristic LIDAR processing (fallback)

        Divides scan into 9 sectors and checks for obstacles.
        """
        num_sectors = 9
        sector_size = len(ranges) // num_sectors

        sectors = np.zeros(num_sectors, dtype=np.float32)
        probabilities = np.zeros(num_sectors, dtype=np.float32)
        min_distances = np.zeros(num_sectors, dtype=np.float32)

        OBSTACLE_THRESHOLD = 1.5  # meters (increased sensitivity)

        for i in range(num_sectors):
            start = i * sector_size
            end = start + sector_size
            sector_ranges = ranges[start:end]

            valid = sector_ranges[np.isfinite(sector_ranges)]
            if len(valid) > 0:
                min_distances[i] = np.min(valid)
                # Probability based on how many points are close
                close_ratio = np.mean(valid < OBSTACLE_THRESHOLD)
                probabilities[i] = close_ratio
                sectors[i] = 1.0 if close_ratio > 0.2 else 0.0  # Lower threshold
            else:
                min_distances[i] = float('inf')

        return ObstacleMap(
            sectors=sectors,
            probabilities=probabilities,
            min_distances=min_distances
        )

    def process_camera(self, image: np.ndarray) -> List[CubeDetection]:
        """
        Process camera image

        Args:
            image: RGB image [H, W, 3] uint8

        Returns:
            List of CubeDetection
        """
        if image is None or image.size == 0:
            return []

        return self.cube_detector.detect(image)

    def update(
        self,
        lidar_ranges: Optional[np.ndarray] = None,
        camera_image: Optional[np.ndarray] = None
    ) -> PerceptionState:
        """
        Update perception state with new sensor data

        Args:
            lidar_ranges: [667] LIDAR ranges (optional)
            camera_image: RGB image [H, W, 3] (optional)

        Returns:
            PerceptionState with all sensor fusion results
        """
        timestamp = time.time()

        # Process LIDAR
        obstacle_map = None
        min_obstacle_dist = float('inf')
        obstacle_angle = 0.0

        if lidar_ranges is not None:
            obstacle_map, min_obstacle_dist, obstacle_angle = self.process_lidar(lidar_ranges)

        # Process camera
        cube_detections = []
        if camera_image is not None:
            cube_detections = self.process_camera(camera_image)

        # Find closest valid cube
        closest_cube = None
        valid_cubes = [d for d in cube_detections if d.is_valid]
        if valid_cubes:
            closest_cube = valid_cubes[0]  # Already sorted by distance

        # Build state
        state = PerceptionState(
            timestamp=timestamp,
            obstacle_map=obstacle_map,
            min_obstacle_distance=min_obstacle_dist,
            obstacle_angle=obstacle_angle,
            cube_detections=cube_detections,
            closest_cube=closest_cube,
            cube_detected=(closest_cube is not None)
        )

        self.last_state = state
        self.update_count += 1

        return state

    def get_obstacle_info(self) -> Dict:
        """
        Get obstacle information for fuzzy controller

        Returns:
            Dict with distance_to_obstacle and angle_to_obstacle
        """
        if self.last_state is None:
            return {'distance_to_obstacle': 5.0, 'angle_to_obstacle': 0.0}

        return {
            'distance_to_obstacle': min(self.last_state.min_obstacle_distance, 5.0),
            'angle_to_obstacle': self.last_state.obstacle_angle
        }

    def get_cube_info(self) -> Dict:
        """
        Get cube information for fuzzy controller

        Returns:
            Dict with cube detection info
        """
        if self.last_state is None or not self.last_state.cube_detected:
            return {
                'cube_detected': False,
                'distance_to_cube': 3.0,
                'angle_to_cube': 0.0,
                'cube_color': None
            }

        cube = self.last_state.closest_cube
        return {
            'cube_detected': True,
            'distance_to_cube': cube.distance,
            'angle_to_cube': cube.angle,
            'cube_color': cube.color
        }


def test_perception_system():
    """Test perception system"""
    print("Testing PerceptionSystem...")

    perception = PerceptionSystem()

    # Test with synthetic LIDAR data
    lidar_ranges = np.random.uniform(0.5, 5.0, size=667)
    lidar_ranges[300:350] = 0.3  # Obstacle in front

    # Test with synthetic camera image
    camera_image = np.zeros((512, 512, 3), dtype=np.uint8)
    camera_image[200:300, 200:300] = [0, 255, 0]  # Green cube

    state = perception.update(lidar_ranges, camera_image)

    print(f"  Min obstacle distance: {state.min_obstacle_distance:.2f}m")
    print(f"  Obstacle angle: {state.obstacle_angle:.1f}°")
    print(f"  Path clear: {state.is_path_clear}")
    print(f"  Should avoid: {state.should_avoid}")
    print(f"  Cube detected: {state.cube_detected}")

    if state.closest_cube:
        print(f"  Closest cube: {state.closest_cube.color} at {state.closest_cube.distance:.2f}m")

    print("  PerceptionSystem test passed")


if __name__ == "__main__":
    test_perception_system()
