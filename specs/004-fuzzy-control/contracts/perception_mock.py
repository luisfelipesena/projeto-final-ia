"""
Perception Mock Interface Contract

Purpose: Mock perception module for independent Phase 3 fuzzy control development.
Provides simulated sensor data matching Phase 2 perception output format.

Based on: specs/004-fuzzy-control/data-model.md
Integration Point: Phase 2 perception module (src/perception/)

Usage: Enable Phase 3 development/testing before Phase 2 neural networks are trained.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


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
        raise NotImplementedError("Must be implemented by concrete class")

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
        raise NotImplementedError("Must be implemented by concrete class")

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

        Example:
            # Obstacle 0.4m ahead, green cube 1.2m at -30°
            data = mock.create_custom(
                obstacle_distances=[5.0, 5.0, 5.0, 0.4, 5.0, 5.0, 5.0, 5.0, 5.0],
                cube_color='green',
                cube_distance=1.2,
                cube_angle=-30.0
            )
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def add_noise(self, data: PerceptionData, noise_level: float = 0.1) -> PerceptionData:
        """
        Add realistic sensor noise to perception data

        Args:
            data: Clean PerceptionData
            noise_level: Noise magnitude [0, 1] (0 = no noise, 1 = max noise)

        Returns:
            PerceptionData with added Gaussian noise to distances/angles

        Noise characteristics:
        - Distance: ±5cm at 0.1, ±20cm at 1.0
        - Angle: ±2° at 0.1, ±10° at 1.0
        - Confidence: -0.05 to -0.2 reduction
        """
        raise NotImplementedError("Must be implemented by concrete class")

    def get_state_specific_scenario(self, state: str) -> PerceptionData:
        """
        Get scenario matching robot operational state

        Args:
            state: 'searching' | 'approaching' | 'grasping' | 'navigating' | 'depositing' | 'avoiding'

        Returns:
            PerceptionData typical for that state

        State mappings:
        - 'searching': No cubes visible, open space
        - 'approaching': Cube visible at medium distance (0.5-1.5m)
        - 'grasping': Cube at close range (0.1-0.2m), aligned
        - 'navigating': No cube (holding), path to box
        - 'depositing': At box location, no obstacles nearby
        - 'avoiding': Critical obstacle <0.3m in path
        """
        raise NotImplementedError("Must be implemented by concrete class")

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

        Example:
            # Simulate approach sequence
            sequence = mock.simulate_sequence([
                'cube_center_far',
                'cube_center_near',
                'approaching_cube'
            ], dt=0.1)
        """
        raise NotImplementedError("Must be implemented by concrete class")


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
