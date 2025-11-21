"""
API Contract: PerceptionSystem

Purpose: Integration layer combining LIDAR + camera perception
Phase: 2 - Perception (Neural Networks)
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from lidar_processor import LIDARProcessor, ObstacleMap
from cube_detector import CubeDetector, CubeObservation


@dataclass
class WorldState:
    """Combined sensor perception (LIDAR + camera)"""

    obstacle_map: ObstacleMap
    cube_observations: List[CubeObservation]
    timestamp: float

    def get_navigation_command(self) -> Tuple[float, float, float]:
        """
        Placeholder for Phase 3 fuzzy controller interface

        Returns:
            (vx, vy, omega) velocity commands

        Note: Implementation will be in Phase 3 (Fuzzy Logic Control)
        """
        raise NotImplementedError("Implemented in Phase 3")

    def to_dict(self) -> dict:
        """
        Serialize full state for logging

        Returns:
            Dictionary with all perception data
        """
        ...


class PerceptionSystem:
    """
    Unified perception interface combining obstacle detection and cube detection

    Architecture:
        ┌─────────────────────┐
        │ PerceptionSystem    │
        └──────────┬──────────┘
                   │
            ┌──────┴──────┐
            ▼             ▼
    ┌─────────────┐ ┌─────────────┐
    │  LIDAR      │ │   Camera    │
    │ Processor   │ │  Detector   │
    └─────────────┘ └─────────────┘
    """

    def __init__(self,
                 lidar_model_path: str,
                 camera_model_path: str,
                 device: str = "cpu"):
        """
        Initialize perception system with both neural networks

        Args:
            lidar_model_path: Path to LIDAR TorchScript model
            camera_model_path: Path to camera TorchScript model
            device: "cpu" or "cuda" (default: "cpu")

        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If model loading fails
        """
        ...

    def update(self,
               lidar_ranges: np.ndarray,
               camera_image: np.ndarray) -> WorldState:
        """
        Process both sensors and fuse results

        Args:
            lidar_ranges: [667] LIDAR distances (meters)
            camera_image: [512, 512, 3] RGB image (uint8)

        Returns:
            WorldState with obstacles + cubes

        Performance:
            - Combined latency: <150ms (target)
            - LIDAR: <100ms
            - Camera: <100ms
            - Can process in parallel if needed

        Raises:
            ValueError: If inputs have incorrect shapes
            RuntimeError: If inference fails
        """
        ...

    def get_obstacle_map(self) -> ObstacleMap:
        """
        Get latest obstacle map from LIDAR

        Returns:
            Most recent ObstacleMap

        Note: Returns cached result from last update() call
        """
        ...

    def get_cube_observations(self) -> List[CubeObservation]:
        """
        Get latest cube detections from camera

        Returns:
            Most recent list of CubeObservation

        Note: Returns cached result from last update() call
              Empty list if no cubes detected
        """
        ...

    def get_nearest_obstacle(self) -> Tuple[int, float]:
        """
        Get closest obstacle from LIDAR map

        Returns:
            (sector_id, distance) for nearest obstacle
            sector_id ∈ [0-8]
            distance in meters

        Use Case:
            Phase 3 fuzzy controller uses this for collision avoidance
        """
        ...

    def get_nearest_cube(self, color: Optional[str] = None) -> Optional[CubeObservation]:
        """
        Find nearest cube, optionally filtered by color

        Args:
            color: "green" | "blue" | "red" | None
                   If None, returns nearest cube of any color

        Returns:
            CubeObservation or None if not detected

        Use Case:
            Phase 3 fuzzy controller uses this for approach behavior:
            - Detect nearest target cube
            - Navigate toward it
            - Trigger grasping when close enough
        """
        ...

    def get_free_direction(self) -> Optional[float]:
        """
        Get angle of freest (most open) direction

        Returns:
            Angle in degrees [0, 360] or None if fully blocked
            0° = forward, 90° = left, 180° = rear, 270° = right

        Algorithm:
            1. Query obstacle_map.get_free_sectors()
            2. Find largest contiguous free region
            3. Return center angle of that region

        Use Case:
            Phase 3 fuzzy controller uses this for obstacle avoidance:
            - If blocked, turn toward free direction
            - Navigate through open space
        """
        ...

    def is_path_clear(self, target_angle: float, clearance: float = 0.5) -> bool:
        """
        Check if path is clear in given direction

        Args:
            target_angle: Direction to check (degrees)
            clearance: Minimum safe distance (meters)

        Returns:
            True if no obstacles within clearance distance

        Use Case:
            Phase 3 fuzzy controller uses this to verify safe navigation:
            - Before moving forward, check is_path_clear(0°, 0.5m)
            - Before turning, check is_path_clear(target_angle, 0.3m)
        """
        ...

    def align_cube_with_obstacles(self,
                                   cube: CubeObservation) -> Tuple[bool, Optional[float]]:
        """
        Check if cube is reachable (not blocked by obstacles)

        Args:
            cube: Target cube observation

        Returns:
            (is_reachable, alternative_angle)
                is_reachable: True if direct path is clear
                alternative_angle: Suggested approach angle if blocked

        Algorithm:
            1. Get cube bearing angle
            2. Map angle to LIDAR sector
            3. Check obstacle_map.is_obstacle(sector)
            4. If blocked, find nearest free sector
            5. Return alternative angle

        Use Case:
            Phase 3 fuzzy controller uses this to plan approach:
            - If cube is behind obstacle, navigate around first
            - If clear path exists, approach directly
        """
        ...

    def get_perception_latency(self) -> Tuple[float, float, float]:
        """
        Get average processing times

        Returns:
            (lidar_ms, camera_ms, total_ms)
                lidar_ms: LIDAR inference time
                camera_ms: Camera inference time
                total_ms: Combined update() latency

        Targets:
            - lidar_ms: <100ms
            - camera_ms: <100ms
            - total_ms: <150ms

        Use Case:
            Performance monitoring and optimization
        """
        ...

    def enable_logging(self, log_path: str):
        """
        Enable perception logging to file

        Args:
            log_path: Path to log file (e.g., logs/perception.log)

        Logging Format:
            Timestamp, LIDAR_time_ms, Camera_time_ms, Total_time_ms,
            Obstacles_detected, Cubes_detected, Cube_colors

        Use Case:
            Debugging, performance analysis, experiment tracking
        """
        ...

    def disable_logging(self):
        """Disable perception logging"""
        ...


# Example Usage (Phase 2 - Perception Only)
if __name__ == "__main__":
    # Initialize perception system
    perception = PerceptionSystem(
        lidar_model_path="models/lidar_net.pt",
        camera_model_path="models/cube_detector.pt"
    )

    # Simulate sensor data
    lidar_ranges = np.random.uniform(0.1, 3.0, size=667)
    camera_image = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)

    # Process sensors
    world_state = perception.update(lidar_ranges, camera_image)

    # Query results
    print(f"Obstacles detected: {world_state.obstacle_map.probabilities.sum():.1f}/9 sectors")
    print(f"Cubes detected: {len(world_state.cube_observations)}")

    # Find nearest obstacle
    sector, distance = perception.get_nearest_obstacle()
    print(f"Nearest obstacle: sector {sector}, {distance:.2f}m")

    # Find nearest green cube
    green_cube = perception.get_nearest_cube(color="green")
    if green_cube:
        print(f"Green cube at {green_cube.distance:.2f}m, angle {green_cube.angle:.1f}°")

        # Check if reachable
        is_reachable, alt_angle = perception.align_cube_with_obstacles(green_cube)
        if is_reachable:
            print("Direct path to cube is clear!")
        else:
            print(f"Cube blocked, approach from {alt_angle:.1f}° instead")

    # Check performance
    lidar_time, camera_time, total_time = perception.get_perception_latency()
    print(f"Latency: LIDAR={lidar_time:.1f}ms, Camera={camera_time:.1f}ms, Total={total_time:.1f}ms")


# Example Integration (Phase 3 - Fuzzy Controller)
"""
Phase 3 will use PerceptionSystem as input to fuzzy logic controller:

from perception_system import PerceptionSystem
from fuzzy_controller import FuzzyController

perception = PerceptionSystem(...)
fuzzy = FuzzyController()

while robot.step(timestep) != -1:
    # Get sensor data
    lidar = robot.getDevice("lidar").getRangeImage()
    camera = robot.getDevice("camera").getImageArray()

    # Process perception
    world_state = perception.update(lidar, camera)

    # Fuzzy inputs:
    obstacle_distance = perception.get_nearest_obstacle()[1]
    cube = perception.get_nearest_cube(color="green")
    path_clear = perception.is_path_clear(target_angle=0)

    # Fuzzy decision
    vx, vy, omega = fuzzy.compute(obstacle_distance, cube, path_clear)

    # Actuate
    base.move(vx, vy, omega)
"""
