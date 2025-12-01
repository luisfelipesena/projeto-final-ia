"""
FuzzyController - Bridge to src/control/fuzzy_*.

Provides unified fuzzy control interface for navigation and manipulation.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


@dataclass
class FuzzyInputs:
    """Input structure for fuzzy controller."""
    distance_to_obstacle: float = 5.0
    angle_to_obstacle: float = 0.0
    distance_to_cube: float = 3.0
    angle_to_cube: float = 0.0
    cube_detected: bool = False
    holding_cube: bool = False
    front_blocked: float = 0.0
    lateral_blocked: float = 0.0


@dataclass
class FuzzyOutput:
    """Output from fuzzy controller."""
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    action: str = "idle"


class FuzzyController:
    """Unified fuzzy controller for navigation and manipulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._navigator = None
        self._manipulator = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize fuzzy systems."""
        try:
            from control.fuzzy_navigator import FuzzyNavigator
            from control.fuzzy_manipulator import FuzzyManipulator

            self._navigator = FuzzyNavigator()
            self._manipulator = FuzzyManipulator()
            self._initialized = True
        except ImportError as e:
            print(f"[FUZZY] Warning: Could not import fuzzy modules: {e}")
            self._initialized = False

    def compute(self, inputs: FuzzyInputs) -> FuzzyOutput:
        """Compute fuzzy output from inputs."""
        if not self._initialized:
            return self._fallback_compute(inputs)

        try:
            if inputs.cube_detected and not inputs.holding_cube:
                # Use manipulator for cube approach
                manip = self._manipulator.compute(
                    inputs.distance_to_cube,
                    inputs.angle_to_cube
                )
                return FuzzyOutput(
                    linear_velocity=manip.approach_speed,
                    angular_velocity=manip.correction_speed,
                    action=manip.action.name.lower()
                )
            else:
                # Use navigator for general movement
                nav = self._navigator.compute(
                    obstacle_left=inputs.distance_to_obstacle if inputs.angle_to_obstacle < 0 else 2.0,
                    obstacle_front=inputs.distance_to_obstacle if abs(inputs.angle_to_obstacle) < 30 else 2.0,
                    obstacle_right=inputs.distance_to_obstacle if inputs.angle_to_obstacle > 0 else 2.0,
                    target_angle=inputs.angle_to_cube if inputs.cube_detected else 0.0
                )
                return FuzzyOutput(
                    linear_velocity=nav.linear_velocity,
                    angular_velocity=nav.angular_velocity,
                    action=nav.action
                )
        except Exception as e:
            print(f"[FUZZY] Compute error: {e}")
            return self._fallback_compute(inputs)

    def _fallback_compute(self, inputs: FuzzyInputs) -> FuzzyOutput:
        """Simple rule-based fallback."""
        linear = 0.0
        angular = 0.0
        action = "idle"

        # Obstacle avoidance priority
        if inputs.front_blocked > 0.5 or inputs.distance_to_obstacle < 0.3:
            linear = -0.05
            angular = 0.3 if inputs.angle_to_obstacle > 0 else -0.3
            action = "avoid"
        elif inputs.cube_detected and not inputs.holding_cube:
            # Approach cube
            if inputs.distance_to_cube < 0.15:
                linear = 0.0
                angular = 0.0
                action = "grasp_ready"
            elif abs(inputs.angle_to_cube) > 10:
                linear = 0.05
                angular = -0.002 * inputs.angle_to_cube
                action = "align"
            else:
                linear = min(0.15, inputs.distance_to_cube * 0.5)
                angular = 0.0
                action = "approach"
        elif inputs.holding_cube:
            linear = 0.1
            angular = 0.0
            action = "transport"
        else:
            # Search
            linear = 0.05
            angular = 0.3
            action = "search"

        return FuzzyOutput(
            linear_velocity=linear,
            angular_velocity=angular,
            action=action
        )

    def compute_navigation(self, obstacle_dists: list, target_angle: float = 0.0) -> FuzzyOutput:
        """Compute navigation output from sector distances."""
        if self._initialized and self._navigator:
            nav = self._navigator.compute_from_sectors(obstacle_dists, target_angle)
            return FuzzyOutput(
                linear_velocity=nav.linear_velocity,
                angular_velocity=nav.angular_velocity,
                action=nav.action
            )
        return FuzzyOutput(linear_velocity=0.1, angular_velocity=0.0, action="forward")
