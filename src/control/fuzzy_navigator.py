"""
Fuzzy Logic Navigator for obstacle avoidance and target following.

Implements Mamdani Fuzzy Inference System for navigation control.
Uses LIDAR sector data for obstacle avoidance and cube detection
for target following.

Fuzzy Variables:
- Input: obstacle_left, obstacle_front, obstacle_right, target_angle
- Output: linear_velocity, angular_velocity
"""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False
    print("[FUZZY] Warning: scikit-fuzzy not available, using fallback")


@dataclass
class NavigationOutput:
    """Output from fuzzy navigation controller."""
    linear_velocity: float   # Forward velocity (m/s)
    angular_velocity: float  # Angular velocity (rad/s)
    action: str              # Description of action


class FuzzyNavigator:
    """Fuzzy logic controller for navigation and obstacle avoidance."""

    def __init__(self):
        """Initialize fuzzy controller."""
        if SKFUZZY_AVAILABLE:
            self._build_fuzzy_system()
        else:
            self._simulation = None

    def _build_fuzzy_system(self) -> None:
        """Build fuzzy inference system using scikit-fuzzy."""
        # Input variables

        # Obstacle distances (0 to 2 meters)
        self.obstacle_left = ctrl.Antecedent(np.linspace(0, 2, 100), 'obstacle_left')
        self.obstacle_front = ctrl.Antecedent(np.linspace(0, 2, 100), 'obstacle_front')
        self.obstacle_right = ctrl.Antecedent(np.linspace(0, 2, 100), 'obstacle_right')

        # Target angle (-90 to 90 degrees)
        self.target_angle = ctrl.Antecedent(np.linspace(-90, 90, 100), 'target_angle')

        # Output variables
        self.linear_vel = ctrl.Consequent(np.linspace(-0.1, 0.3, 100), 'linear_vel')
        self.angular_vel = ctrl.Consequent(np.linspace(-1.0, 1.0, 100), 'angular_vel')

        # Membership functions for obstacles
        for obs in [self.obstacle_left, self.obstacle_front, self.obstacle_right]:
            obs['very_close'] = fuzz.trapmf(obs.universe, [0, 0, 0.2, 0.35])
            obs['close'] = fuzz.trimf(obs.universe, [0.2, 0.4, 0.6])
            obs['medium'] = fuzz.trimf(obs.universe, [0.5, 0.8, 1.2])
            obs['far'] = fuzz.trapmf(obs.universe, [1.0, 1.5, 2, 2])

        # Membership functions for target angle
        self.target_angle['far_left'] = fuzz.trapmf(self.target_angle.universe, [-90, -90, -40, -20])
        self.target_angle['left'] = fuzz.trimf(self.target_angle.universe, [-30, -15, 0])
        self.target_angle['center'] = fuzz.trimf(self.target_angle.universe, [-10, 0, 10])
        self.target_angle['right'] = fuzz.trimf(self.target_angle.universe, [0, 15, 30])
        self.target_angle['far_right'] = fuzz.trapmf(self.target_angle.universe, [20, 40, 90, 90])

        # Membership functions for linear velocity
        self.linear_vel['backward'] = fuzz.trapmf(self.linear_vel.universe, [-0.1, -0.1, -0.05, 0])
        self.linear_vel['stop'] = fuzz.trimf(self.linear_vel.universe, [-0.02, 0, 0.02])
        self.linear_vel['slow'] = fuzz.trimf(self.linear_vel.universe, [0, 0.08, 0.15])
        self.linear_vel['medium'] = fuzz.trimf(self.linear_vel.universe, [0.1, 0.18, 0.25])
        self.linear_vel['fast'] = fuzz.trapmf(self.linear_vel.universe, [0.2, 0.25, 0.3, 0.3])

        # Membership functions for angular velocity
        self.angular_vel['hard_left'] = fuzz.trapmf(self.angular_vel.universe, [-1, -1, -0.6, -0.3])
        self.angular_vel['left'] = fuzz.trimf(self.angular_vel.universe, [-0.5, -0.25, 0])
        self.angular_vel['straight'] = fuzz.trimf(self.angular_vel.universe, [-0.1, 0, 0.1])
        self.angular_vel['right'] = fuzz.trimf(self.angular_vel.universe, [0, 0.25, 0.5])
        self.angular_vel['hard_right'] = fuzz.trapmf(self.angular_vel.universe, [0.3, 0.6, 1, 1])

        # Define rules
        rules = self._define_rules()

        # Create control system
        self._ctrl = ctrl.ControlSystem(rules)
        self._simulation = ctrl.ControlSystemSimulation(self._ctrl)

    def _define_rules(self) -> List:
        """Define fuzzy rules for navigation."""
        rules = []

        # OBSTACLE AVOIDANCE RULES (high priority)

        # Front obstacle - stop or back up
        rules.append(ctrl.Rule(
            self.obstacle_front['very_close'],
            (self.linear_vel['backward'], self.angular_vel['straight'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['close'] & self.obstacle_left['close'],
            (self.linear_vel['stop'], self.angular_vel['hard_right'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['close'] & self.obstacle_right['close'],
            (self.linear_vel['stop'], self.angular_vel['hard_left'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['close'] & self.obstacle_left['far'] & self.obstacle_right['far'],
            (self.linear_vel['slow'], self.angular_vel['left'])
        ))

        # Side obstacles
        rules.append(ctrl.Rule(
            self.obstacle_left['very_close'] & self.obstacle_front['far'],
            (self.linear_vel['slow'], self.angular_vel['right'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_right['very_close'] & self.obstacle_front['far'],
            (self.linear_vel['slow'], self.angular_vel['left'])
        ))

        # TARGET FOLLOWING RULES (when path is clear)

        rules.append(ctrl.Rule(
            self.obstacle_front['far'] & self.target_angle['center'],
            (self.linear_vel['fast'], self.angular_vel['straight'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['far'] & self.target_angle['left'],
            (self.linear_vel['medium'], self.angular_vel['left'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['far'] & self.target_angle['right'],
            (self.linear_vel['medium'], self.angular_vel['right'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['far'] & self.target_angle['far_left'],
            (self.linear_vel['slow'], self.angular_vel['hard_left'])
        ))

        rules.append(ctrl.Rule(
            self.obstacle_front['far'] & self.target_angle['far_right'],
            (self.linear_vel['slow'], self.angular_vel['hard_right'])
        ))

        # Medium distance - cautious approach
        rules.append(ctrl.Rule(
            self.obstacle_front['medium'] & self.target_angle['center'],
            (self.linear_vel['medium'], self.angular_vel['straight'])
        ))

        # Default - search/wander
        rules.append(ctrl.Rule(
            self.obstacle_front['far'] & self.obstacle_left['far'] & self.obstacle_right['far'],
            (self.linear_vel['medium'], self.angular_vel['straight'])
        ))

        return rules

    def compute(
        self,
        obstacle_left: float,
        obstacle_front: float,
        obstacle_right: float,
        target_angle: float = 0.0,
    ) -> NavigationOutput:
        """Compute navigation output from fuzzy controller.

        Args:
            obstacle_left: Distance to left obstacle (m)
            obstacle_front: Distance to front obstacle (m)
            obstacle_right: Distance to right obstacle (m)
            target_angle: Angle to target (degrees, + = right)

        Returns:
            NavigationOutput with velocities and action
        """
        if SKFUZZY_AVAILABLE and self._simulation:
            return self._compute_fuzzy(
                obstacle_left, obstacle_front, obstacle_right, target_angle
            )
        else:
            return self._compute_fallback(
                obstacle_left, obstacle_front, obstacle_right, target_angle
            )

    def _compute_fuzzy(
        self,
        obstacle_left: float,
        obstacle_front: float,
        obstacle_right: float,
        target_angle: float,
    ) -> NavigationOutput:
        """Compute using fuzzy system."""
        # Clamp inputs
        obstacle_left = np.clip(obstacle_left, 0, 2)
        obstacle_front = np.clip(obstacle_front, 0, 2)
        obstacle_right = np.clip(obstacle_right, 0, 2)
        target_angle = np.clip(target_angle, -90, 90)

        # Set inputs
        self._simulation.input['obstacle_left'] = obstacle_left
        self._simulation.input['obstacle_front'] = obstacle_front
        self._simulation.input['obstacle_right'] = obstacle_right
        self._simulation.input['target_angle'] = target_angle

        # Compute
        try:
            self._simulation.compute()
            linear = self._simulation.output['linear_vel']
            angular = self._simulation.output['angular_vel']
        except Exception:
            # Fallback if computation fails
            linear = 0.0
            angular = 0.0

        # Determine action description
        action = self._describe_action(linear, angular, obstacle_front)

        return NavigationOutput(
            linear_velocity=float(linear),
            angular_velocity=float(angular),
            action=action,
        )

    def _compute_fallback(
        self,
        obstacle_left: float,
        obstacle_front: float,
        obstacle_right: float,
        target_angle: float,
    ) -> NavigationOutput:
        """Simple rule-based fallback when fuzzy not available."""
        linear = 0.15
        angular = 0.0
        action = "forward"

        # Obstacle avoidance (priority)
        if obstacle_front < 0.3:
            linear = -0.05
            angular = 0.0
            action = "backup"
        elif obstacle_front < 0.5:
            linear = 0.0
            if obstacle_left < obstacle_right:
                angular = -0.5  # Turn right
                action = "avoid_left"
            else:
                angular = 0.5   # Turn left
                action = "avoid_right"
        elif obstacle_left < 0.3:
            angular = -0.3
            linear = 0.1
            action = "veer_right"
        elif obstacle_right < 0.3:
            angular = 0.3
            linear = 0.1
            action = "veer_left"
        else:
            # Target following
            if abs(target_angle) > 30:
                angular = 0.5 * np.sign(-target_angle)
                linear = 0.08
                action = "turn_to_target"
            elif abs(target_angle) > 10:
                angular = 0.25 * np.sign(-target_angle)
                linear = 0.12
                action = "adjust_to_target"
            else:
                linear = 0.2
                angular = 0.0
                action = "approach_target"

        return NavigationOutput(
            linear_velocity=linear,
            angular_velocity=angular,
            action=action,
        )

    def _describe_action(
        self, linear: float, angular: float, front_dist: float
    ) -> str:
        """Generate human-readable action description."""
        if linear < -0.02:
            return "backup"
        elif linear < 0.02:
            if abs(angular) > 0.3:
                return "turn_in_place"
            return "stop"
        elif abs(angular) > 0.4:
            return "sharp_turn"
        elif abs(angular) > 0.15:
            return "turning"
        else:
            if front_dist < 0.5:
                return "cautious_forward"
            return "forward"

    def compute_from_sectors(
        self,
        sector_distances: List[float],
        target_angle: float = 0.0,
    ) -> NavigationOutput:
        """Compute from LIDAR sector distances.

        Assumes 9 sectors: [left_back, left, left_front, front_left, front,
                           front_right, right_front, right, right_back]

        Args:
            sector_distances: List of 9 sector minimum distances
            target_angle: Angle to target (degrees)

        Returns:
            NavigationOutput
        """
        if len(sector_distances) < 9:
            sector_distances = sector_distances + [2.0] * (9 - len(sector_distances))

        # Extract relevant sectors
        left = min(sector_distances[1], sector_distances[2])     # left, left_front
        front = min(sector_distances[3], sector_distances[4], sector_distances[5])
        right = min(sector_distances[6], sector_distances[7])    # right_front, right

        return self.compute(left, front, right, target_angle)
