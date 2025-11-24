"""
Mock Perception System for Testing

Provides simulated sensor data matching Phase 2 perception output format.
Enables independent Phase 3 fuzzy control development.

Contract: specs/004-fuzzy-control/contracts/perception_mock.py
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import time


@dataclass
class BoundingBox:
    """Camera space bounding box for detected object"""
    x_min: int  # pixels
    y_min: int  # pixels
    x_max: int  # pixels
    y_max: int  # pixels
    width: int  # pixels
    height: int  # pixels


@dataclass
class CubeObservation:
    """
    Detected cube from camera perception

    Matches output from Phase 2: src/perception/cube_detector.py
    """
    color: str  # 'green' | 'blue' | 'red'
    distance: float  # meters (from robot center)
    angle: float  # degrees (bearing from robot forward axis, [-135, +135])
    bbox: BoundingBox  # Camera pixel coordinates
    confidence: float  # [0, 1] detection confidence


@dataclass
class ObstacleMap:
    """
    9-sector LIDAR occupancy map

    Matches output from Phase 2: src/perception/lidar_processor.py

    Sectors layout (top-down view, 0° = forward):
        [0]: -135° to -90° (left-back)
        [1]: -90° to -45° (left)
        [2]: -45° to 0° (left-front)
        [3]: 0° (center-front)
        [4]: 0° to 45° (right-front)
        [5]: 45° to 90° (right)
        [6]: 90° to 135° (right-back)
        [7]: 135° to 180° (back-left)
        [8]: 180° to -135° (back-right)
    """
    sectors: np.ndarray  # shape (9,), dtype bool - binary occupancy
    probabilities: np.ndarray  # shape (9,), dtype float32 - confidence [0, 1]
    min_distances: np.ndarray  # shape (9,), dtype float32 - meters [0.1, 5.0]

    @property
    def min_distance(self) -> float:
        """Overall minimum distance across all sectors"""
        return float(np.min(self.min_distances))

    @property
    def min_angle(self) -> float:
        """Angle to closest obstacle (degrees)"""
        min_idx = int(np.argmin(self.min_distances))
        # Map sector index to angle: each sector is 30° (270° / 9)
        # Sector 3 is forward (0°), sector 0 is -135°
        sector_angles = [-135, -90, -45, 0, 45, 90, 135, 180, -180]
        return float(sector_angles[min_idx]) if min_idx < len(sector_angles) else 0.0


@dataclass
class PerceptionData:
    """
    Complete perception output consumed by fuzzy controller

    Aggregates obstacle map + cube detections + metadata
    """
    obstacle_map: ObstacleMap
    detected_cubes: List[CubeObservation]  # Empty list if no cubes visible
    timestamp: float  # Unix timestamp of sensor reading


class MockPerceptionSystem:
    """
    Mock perception system for Phase 3 testing

    Provides realistic but simulated sensor data for:
    - Obstacle detection scenarios (clear path, blocked, corner, etc.)
    - Cube detection scenarios (single, multiple, various distances/angles)
    - State-specific scenarios (searching, approaching, navigating)

    Usage:
        mock = MockPerceptionSystem(seed=42)

        # Scenario: Clear path ahead, cube at 1.5m left
        data = mock.get_scenario('cube_left_clear')

        # Custom scenario
        data = mock.create_custom(
            obstacle_distances=[2.0] * 9,  # All sectors clear
            cube_color='green',
            cube_distance=0.8,
            cube_angle=-30.0
        )
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize mock perception system

        Args:
            seed: Random seed for reproducible scenarios (default: None)
        """
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed

    def get_scenario(self, scenario_name: str) -> PerceptionData:
        """
        Get predefined test scenario

        Available scenarios:
        - 'clear_all': No obstacles, no cubes (exploration)
        - 'obstacle_front': Obstacle 0.5m ahead in center sector
        - 'obstacle_critical': Obstacle 0.2m ahead (emergency stop)
        - 'cube_center_near': Green cube 0.3m ahead, aligned
        - 'cube_left_far': Blue cube 2.0m at -45°
        - 'cube_right_close': Red cube 0.5m at 30°
        - 'multiple_cubes': 3 cubes visible at various positions
        - 'corner_trap': Obstacles on 3 sides (escape test)
        - 'narrow_passage': Obstacles left+right, clear center
        - 'approaching_cube': Cube 0.15m ahead (grasp range)

        Args:
            scenario_name: Name of predefined scenario

        Returns:
            PerceptionData matching scenario

        Raises:
            ValueError: If scenario_name not recognized
        """
        scenarios = {
            'clear_all': self._scenario_clear_all,
            'obstacle_front': self._scenario_obstacle_front,
            'obstacle_critical': self._scenario_obstacle_critical,
            'cube_center_near': self._scenario_cube_center_near,
            'cube_left_far': self._scenario_cube_left_far,
            'cube_right_close': self._scenario_cube_right_close,
            'multiple_cubes': self._scenario_multiple_cubes,
            'corner_trap': self._scenario_corner_trap,
            'narrow_passage': self._scenario_narrow_passage,
            'approaching_cube': self._scenario_approaching_cube,
        }

        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(scenarios.keys())}")

        return scenarios[scenario_name]()

    def create_custom(
        self,
        obstacle_distances: List[float],  # 9 values, meters per sector
        cube_color: Optional[str] = None,
        cube_distance: Optional[float] = None,
        cube_angle: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> PerceptionData:
        """
        Create custom perception data for specific test case

        Args:
            obstacle_distances: Distance to nearest obstacle per sector (9 values)
                                Use 5.0 for "no obstacle" in sector
            cube_color: Optional cube color ('green'|'blue'|'red')
            cube_distance: Optional cube distance in meters [0.1, 3.0]
            cube_angle: Optional cube bearing in degrees [-135, +135]
            timestamp: Optional timestamp (default: current time)

        Returns:
            PerceptionData with specified configuration

        Raises:
            ValueError: If distances length != 9 or values out of range
        """
        if len(obstacle_distances) != 9:
            raise ValueError(f"obstacle_distances must have 9 values, got {len(obstacle_distances)}")

        # Create obstacle map
        distances = np.array(obstacle_distances, dtype=np.float32)
        sectors = distances < 5.0  # Binary occupancy
        probabilities = np.where(sectors, 1.0 - (distances / 5.0), 0.0).astype(np.float32)
        obstacle_map = ObstacleMap(
            sectors=sectors,
            probabilities=probabilities,
            min_distances=distances
        )

        # Create cube observations
        cubes = []
        if cube_color and cube_distance is not None and cube_angle is not None:
            bbox = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200, width=100, height=100)
            cube = CubeObservation(
                color=cube_color,
                distance=cube_distance,
                angle=cube_angle,
                bbox=bbox,
                confidence=0.9
            )
            cubes.append(cube)

        return PerceptionData(
            obstacle_map=obstacle_map,
            detected_cubes=cubes,
            timestamp=timestamp or time.time()
        )

    # ========================================================================
    # Scenario Implementations
    # ========================================================================

    def _scenario_clear_all(self) -> PerceptionData:
        """No obstacles, no cubes (exploration)"""
        return self.create_custom(obstacle_distances=[5.0] * 9)

    def _scenario_obstacle_front(self) -> PerceptionData:
        """Obstacle 0.5m ahead in center sector"""
        distances = [5.0] * 9
        distances[3] = 0.5  # Center-front
        return self.create_custom(obstacle_distances=distances)

    def _scenario_obstacle_critical(self) -> PerceptionData:
        """Obstacle 0.2m ahead (emergency stop)"""
        distances = [5.0] * 9
        distances[3] = 0.2  # Center-front, very close
        return self.create_custom(obstacle_distances=distances)

    def _scenario_cube_center_near(self) -> PerceptionData:
        """Green cube 0.3m ahead, aligned"""
        return self.create_custom(
            obstacle_distances=[5.0] * 9,
            cube_color='green',
            cube_distance=0.3,
            cube_angle=0.0
        )

    def _scenario_cube_left_far(self) -> PerceptionData:
        """Blue cube 2.0m at -45°"""
        return self.create_custom(
            obstacle_distances=[5.0] * 9,
            cube_color='blue',
            cube_distance=2.0,
            cube_angle=-45.0
        )

    def _scenario_cube_right_close(self) -> PerceptionData:
        """Red cube 0.5m at 30°"""
        return self.create_custom(
            obstacle_distances=[5.0] * 9,
            cube_color='red',
            cube_distance=0.5,
            cube_angle=30.0
        )

    def _scenario_multiple_cubes(self) -> PerceptionData:
        """3 cubes visible at various positions"""
        data = self.create_custom(obstacle_distances=[5.0] * 9)
        # Add additional cubes
        bbox = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200, width=100, height=100)
        data.detected_cubes.extend([
            CubeObservation('green', 1.5, -30.0, bbox, 0.9),
            CubeObservation('blue', 2.0, 45.0, bbox, 0.85),
        ])
        return data

    def _scenario_corner_trap(self) -> PerceptionData:
        """Obstacles on 3 sides (escape test)"""
        distances = [5.0] * 9
        distances[1] = 0.4  # Left
        distances[3] = 0.4  # Front
        distances[5] = 0.4  # Right
        return self.create_custom(obstacle_distances=distances)

    def _scenario_narrow_passage(self) -> PerceptionData:
        """Obstacles left+right, clear center"""
        distances = [5.0] * 9
        distances[1] = 0.6  # Left
        distances[5] = 0.6  # Right
        return self.create_custom(obstacle_distances=distances)

    def _scenario_approaching_cube(self) -> PerceptionData:
        """Cube 0.15m ahead (grasp range)"""
        return self.create_custom(
            obstacle_distances=[5.0] * 9,
            cube_color='green',
            cube_distance=0.15,
            cube_angle=0.0
        )

    def add_noise(self, data: PerceptionData, noise_level: float = 0.1) -> PerceptionData:
        """
        Add realistic sensor noise to perception data

        Args:
            data: Clean PerceptionData
            noise_level: Noise magnitude [0, 1] (0 = no noise, 1 = max noise)

        Returns:
            PerceptionData with added Gaussian noise to distances/angles
        """
        # Add noise to obstacle distances
        noise = np.random.normal(0, noise_level * 0.1, size=9)
        noisy_distances = np.clip(data.obstacle_map.min_distances + noise, 0.1, 5.0)
        noisy_sectors = noisy_distances < 5.0
        noisy_probs = np.where(noisy_sectors, 1.0 - (noisy_distances / 5.0), 0.0).astype(np.float32)

        noisy_obstacle_map = ObstacleMap(
            sectors=noisy_sectors,
            probabilities=noisy_probs,
            min_distances=noisy_distances.astype(np.float32)
        )

        # Add noise to cube observations
        noisy_cubes = []
        for cube in data.detected_cubes:
            distance_noise = np.random.normal(0, noise_level * 0.05)
            angle_noise = np.random.normal(0, noise_level * 5.0)
            noisy_cubes.append(CubeObservation(
                color=cube.color,
                distance=max(0.1, cube.distance + distance_noise),
                angle=np.clip(cube.angle + angle_noise, -135.0, 135.0),
                bbox=cube.bbox,
                confidence=max(0.0, cube.confidence - noise_level * 0.1)
            ))

        return PerceptionData(
            obstacle_map=noisy_obstacle_map,
            detected_cubes=noisy_cubes,
            timestamp=data.timestamp
        )

    def get_state_specific_scenario(self, state: str) -> PerceptionData:
        """
        Get scenario matching robot operational state

        Args:
            state: 'searching' | 'approaching' | 'grasping' | 'navigating' | 'depositing' | 'avoiding'

        Returns:
            PerceptionData typical for that state
        """
        state_scenarios = {
            'searching': 'clear_all',
            'approaching': 'cube_center_near',
            'grasping': 'approaching_cube',
            'navigating': 'clear_all',
            'depositing': 'clear_all',
            'avoiding': 'obstacle_critical',
        }
        return self.get_scenario(state_scenarios.get(state, 'clear_all'))

    def simulate_sequence(
        self,
        scenario_sequence: List[str],
        dt: float = 0.05
    ) -> List[Tuple[float, PerceptionData]]:
        """
        Generate sequence of perception data over time

        Args:
            scenario_sequence: List of scenario names to chain
            dt: Time step between scenarios (seconds, default 0.05 = 20Hz)

        Returns:
            List of (timestamp, PerceptionData) tuples
        """
        sequence = []
        current_time = time.time()
        for scenario_name in scenario_sequence:
            data = self.get_scenario(scenario_name)
            data.timestamp = current_time
            sequence.append((current_time, data))
            current_time += dt
        return sequence


# Convenience functions for common test patterns

def mock_clear_path() -> PerceptionData:
    """Quick mock: Clear path, no obstacles, no cubes"""
    mock = MockPerceptionSystem()
    return mock.get_scenario('clear_all')


def mock_obstacle_ahead(distance: float = 0.5) -> PerceptionData:
    """Quick mock: Obstacle ahead at specified distance"""
    mock = MockPerceptionSystem()
    return mock.create_custom(
        obstacle_distances=[5.0, 5.0, 5.0, distance, 5.0, 5.0, 5.0, 5.0, 5.0]
    )


def mock_cube_detected(color: str = 'green', distance: float = 1.0, angle: float = 0.0) -> PerceptionData:
    """Quick mock: Cube detected at specified position"""
    mock = MockPerceptionSystem()
    return mock.create_custom(
        obstacle_distances=[5.0] * 9,
        cube_color=color,
        cube_distance=distance,
        cube_angle=angle
    )

