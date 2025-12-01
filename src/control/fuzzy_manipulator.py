"""
Fuzzy Logic Manipulator for grasp control.

Implements Mamdani Fuzzy Inference System for:
- Approach velocity control based on distance to cube
- Alignment control based on cube angle
- Grasp timing decisions

Fuzzy Variables:
- Input: cube_distance, cube_angle, alignment_error
- Output: approach_speed, correction_angle, grasp_ready
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple
import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


class ManipulationAction(Enum):
    """Manipulation action to take."""
    WAIT = auto()           # Not ready, wait
    APPROACH = auto()       # Move forward toward cube
    ALIGN_LEFT = auto()     # Correct alignment left
    ALIGN_RIGHT = auto()    # Correct alignment right
    FINAL_APPROACH = auto() # Final slow approach
    GRASP = auto()          # Execute grasp


@dataclass
class ManipulationOutput:
    """Output from fuzzy manipulation controller."""
    action: ManipulationAction
    approach_speed: float     # Forward velocity for approach
    correction_speed: float   # Angular velocity for alignment
    grasp_confidence: float   # Confidence that grasp should proceed


class FuzzyManipulator:
    """Fuzzy logic controller for manipulation and grasping."""

    # Thresholds
    GRASP_DISTANCE = 0.12      # Distance to trigger grasp (m)
    ALIGN_THRESHOLD = 3.0      # Max angle error for grasp (degrees)
    APPROACH_DISTANCE = 0.3    # Start slowing down (m)

    def __init__(self):
        """Initialize fuzzy controller."""
        if SKFUZZY_AVAILABLE:
            self._build_fuzzy_system()
        else:
            self._simulation = None

    def _build_fuzzy_system(self) -> None:
        """Build fuzzy inference system."""
        # Input: Distance to cube (0 to 1 meter)
        self.distance = ctrl.Antecedent(np.linspace(0, 1, 100), 'distance')
        self.distance['very_close'] = fuzz.trapmf(self.distance.universe, [0, 0, 0.08, 0.15])
        self.distance['close'] = fuzz.trimf(self.distance.universe, [0.1, 0.2, 0.35])
        self.distance['medium'] = fuzz.trimf(self.distance.universe, [0.25, 0.4, 0.6])
        self.distance['far'] = fuzz.trapmf(self.distance.universe, [0.5, 0.7, 1, 1])

        # Input: Angle to cube (-45 to 45 degrees)
        self.angle = ctrl.Antecedent(np.linspace(-45, 45, 100), 'angle')
        self.angle['far_left'] = fuzz.trapmf(self.angle.universe, [-45, -45, -20, -10])
        self.angle['left'] = fuzz.trimf(self.angle.universe, [-15, -7, 0])
        self.angle['center'] = fuzz.trimf(self.angle.universe, [-5, 0, 5])
        self.angle['right'] = fuzz.trimf(self.angle.universe, [0, 7, 15])
        self.angle['far_right'] = fuzz.trapmf(self.angle.universe, [10, 20, 45, 45])

        # Output: Approach speed (0 to 0.15 m/s)
        self.approach = ctrl.Consequent(np.linspace(0, 0.15, 100), 'approach')
        self.approach['stop'] = fuzz.trimf(self.approach.universe, [0, 0, 0.02])
        self.approach['very_slow'] = fuzz.trimf(self.approach.universe, [0.01, 0.03, 0.05])
        self.approach['slow'] = fuzz.trimf(self.approach.universe, [0.04, 0.07, 0.1])
        self.approach['normal'] = fuzz.trapmf(self.approach.universe, [0.08, 0.12, 0.15, 0.15])

        # Output: Correction angular velocity (-0.5 to 0.5 rad/s)
        self.correction = ctrl.Consequent(np.linspace(-0.5, 0.5, 100), 'correction')
        self.correction['hard_left'] = fuzz.trapmf(self.correction.universe, [-0.5, -0.5, -0.3, -0.15])
        self.correction['left'] = fuzz.trimf(self.correction.universe, [-0.25, -0.12, 0])
        self.correction['none'] = fuzz.trimf(self.correction.universe, [-0.05, 0, 0.05])
        self.correction['right'] = fuzz.trimf(self.correction.universe, [0, 0.12, 0.25])
        self.correction['hard_right'] = fuzz.trapmf(self.correction.universe, [0.15, 0.3, 0.5, 0.5])

        # Output: Grasp readiness (0 to 1)
        self.grasp_ready = ctrl.Consequent(np.linspace(0, 1, 100), 'grasp_ready')
        self.grasp_ready['not_ready'] = fuzz.trapmf(self.grasp_ready.universe, [0, 0, 0.2, 0.4])
        self.grasp_ready['almost'] = fuzz.trimf(self.grasp_ready.universe, [0.3, 0.5, 0.7])
        self.grasp_ready['ready'] = fuzz.trapmf(self.grasp_ready.universe, [0.6, 0.8, 1, 1])

        # Define rules
        rules = self._define_rules()

        # Create control system
        self._ctrl = ctrl.ControlSystem(rules)
        self._simulation = ctrl.ControlSystemSimulation(self._ctrl)

    def _define_rules(self) -> list:
        """Define fuzzy rules for manipulation."""
        rules = []

        # Distance-based approach speed

        # Far - move at normal speed if aligned
        rules.append(ctrl.Rule(
            self.distance['far'] & self.angle['center'],
            (self.approach['normal'], self.correction['none'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['far'] & self.angle['left'],
            (self.approach['slow'], self.correction['left'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['far'] & self.angle['right'],
            (self.approach['slow'], self.correction['right'], self.grasp_ready['not_ready'])
        ))

        # Medium distance - slow down, focus on alignment
        rules.append(ctrl.Rule(
            self.distance['medium'] & self.angle['center'],
            (self.approach['slow'], self.correction['none'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['medium'] & self.angle['left'],
            (self.approach['very_slow'], self.correction['left'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['medium'] & self.angle['right'],
            (self.approach['very_slow'], self.correction['right'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['medium'] & self.angle['far_left'],
            (self.approach['stop'], self.correction['hard_left'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['medium'] & self.angle['far_right'],
            (self.approach['stop'], self.correction['hard_right'], self.grasp_ready['not_ready'])
        ))

        # Close - final alignment
        rules.append(ctrl.Rule(
            self.distance['close'] & self.angle['center'],
            (self.approach['very_slow'], self.correction['none'], self.grasp_ready['almost'])
        ))

        rules.append(ctrl.Rule(
            self.distance['close'] & self.angle['left'],
            (self.approach['stop'], self.correction['left'], self.grasp_ready['not_ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['close'] & self.angle['right'],
            (self.approach['stop'], self.correction['right'], self.grasp_ready['not_ready'])
        ))

        # Very close - grasp decision
        rules.append(ctrl.Rule(
            self.distance['very_close'] & self.angle['center'],
            (self.approach['stop'], self.correction['none'], self.grasp_ready['ready'])
        ))

        rules.append(ctrl.Rule(
            self.distance['very_close'] & (self.angle['left'] | self.angle['right']),
            (self.approach['stop'], self.correction['none'], self.grasp_ready['almost'])
        ))

        rules.append(ctrl.Rule(
            self.distance['very_close'] & (self.angle['far_left'] | self.angle['far_right']),
            (self.approach['stop'], self.correction['none'], self.grasp_ready['not_ready'])
        ))

        return rules

    def compute(
        self,
        cube_distance: float,
        cube_angle: float,
    ) -> ManipulationOutput:
        """Compute manipulation output.

        Args:
            cube_distance: Distance to cube (meters)
            cube_angle: Angle to cube (degrees, + = right)

        Returns:
            ManipulationOutput with action and velocities
        """
        if SKFUZZY_AVAILABLE and self._simulation:
            return self._compute_fuzzy(cube_distance, cube_angle)
        else:
            return self._compute_fallback(cube_distance, cube_angle)

    def _compute_fuzzy(
        self,
        cube_distance: float,
        cube_angle: float,
    ) -> ManipulationOutput:
        """Compute using fuzzy system."""
        # Clamp inputs
        cube_distance = np.clip(cube_distance, 0, 1)
        cube_angle = np.clip(cube_angle, -45, 45)

        # Set inputs
        self._simulation.input['distance'] = cube_distance
        self._simulation.input['angle'] = cube_angle

        # Compute
        try:
            self._simulation.compute()
            approach = self._simulation.output['approach']
            correction = self._simulation.output['correction']
            grasp_conf = self._simulation.output['grasp_ready']
        except Exception:
            approach = 0.0
            correction = 0.0
            grasp_conf = 0.0

        # Determine action
        action = self._determine_action(cube_distance, cube_angle, approach, correction, grasp_conf)

        return ManipulationOutput(
            action=action,
            approach_speed=float(approach),
            correction_speed=float(correction),
            grasp_confidence=float(grasp_conf),
        )

    def _compute_fallback(
        self,
        cube_distance: float,
        cube_angle: float,
    ) -> ManipulationOutput:
        """Rule-based fallback."""
        approach = 0.0
        correction = 0.0
        grasp_conf = 0.0
        action = ManipulationAction.WAIT

        # Alignment first
        if abs(cube_angle) > 20:
            correction = 0.3 * np.sign(-cube_angle)
            approach = 0.0
            action = ManipulationAction.ALIGN_LEFT if cube_angle < 0 else ManipulationAction.ALIGN_RIGHT
        elif abs(cube_angle) > self.ALIGN_THRESHOLD:
            correction = 0.15 * np.sign(-cube_angle)
            approach = 0.03
            action = ManipulationAction.ALIGN_LEFT if cube_angle < 0 else ManipulationAction.ALIGN_RIGHT
        elif cube_distance > self.APPROACH_DISTANCE:
            approach = 0.1
            correction = 0.05 * np.sign(-cube_angle)
            action = ManipulationAction.APPROACH
        elif cube_distance > self.GRASP_DISTANCE:
            approach = 0.05
            correction = 0.0
            grasp_conf = 0.5
            action = ManipulationAction.FINAL_APPROACH
        else:
            approach = 0.0
            correction = 0.0
            grasp_conf = 1.0
            action = ManipulationAction.GRASP

        return ManipulationOutput(
            action=action,
            approach_speed=approach,
            correction_speed=correction,
            grasp_confidence=grasp_conf,
        )

    def _determine_action(
        self,
        distance: float,
        angle: float,
        approach: float,
        correction: float,
        grasp_conf: float,
    ) -> ManipulationAction:
        """Determine action from fuzzy outputs."""
        if grasp_conf > 0.7 and distance < self.GRASP_DISTANCE:
            return ManipulationAction.GRASP

        if distance < 0.15 and abs(angle) < self.ALIGN_THRESHOLD:
            return ManipulationAction.FINAL_APPROACH

        if abs(correction) > 0.1:
            if correction < 0:
                return ManipulationAction.ALIGN_LEFT
            else:
                return ManipulationAction.ALIGN_RIGHT

        if approach > 0.02:
            return ManipulationAction.APPROACH

        return ManipulationAction.WAIT

    def should_grasp(self, cube_distance: float, cube_angle: float) -> Tuple[bool, float]:
        """Simple check if grasp should proceed.

        Args:
            cube_distance: Distance to cube (m)
            cube_angle: Angle to cube (degrees)

        Returns:
            (should_grasp, confidence)
        """
        output = self.compute(cube_distance, cube_angle)
        return (output.action == ManipulationAction.GRASP, output.grasp_confidence)
