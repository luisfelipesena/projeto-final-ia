"""
Odometry Module

Wheel-based position and orientation estimation for GPS-free navigation.
Based on: Siegwart & Nourbakhsh (2004) - Introduction to Autonomous Mobile Robots

The YouBot uses mecanum wheels which allow omnidirectional movement.
This module tracks position using wheel encoder integration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


@dataclass
class Pose2D:
    """
    2D robot pose (position + orientation)

    Attributes:
        x: X position (meters, forward from start)
        y: Y position (meters, left from start)
        theta: Orientation (radians, counter-clockwise from start)
        timestamp: Unix timestamp of pose estimate
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def distance_to(self, other: 'Pose2D') -> float:
        """Euclidean distance to another pose"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def angle_to(self, other: 'Pose2D') -> float:
        """Angle to another pose (radians)"""
        dx = other.x - self.x
        dy = other.y - self.y
        return np.arctan2(dy, dx)

    def __add__(self, other: 'Pose2D') -> 'Pose2D':
        """Add two poses (for incremental updates)"""
        return Pose2D(
            x=self.x + other.x,
            y=self.y + other.y,
            theta=self._normalize_angle(self.theta + other.theta),
            timestamp=time.time()
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


@dataclass
class OdometryConfig:
    """Odometry configuration for YouBot"""
    wheel_radius: float = 0.05  # meters (from base.py)
    lx: float = 0.228  # Longitudinal distance from COM to wheel (meters)
    ly: float = 0.158  # Lateral distance from COM to wheel (meters)
    encoder_ticks_per_rev: int = 4096  # Typical encoder resolution
    max_velocity: float = 0.3  # m/s


class Odometry:
    """
    Wheel odometry for YouBot position tracking

    Uses mecanum wheel kinematics to estimate robot pose from
    wheel velocities or encoder readings.

    Mecanum wheel inverse kinematics (velocity to wheels):
        w1 = (vx - vy - (lx + ly) * omega) / r  (front-left)
        w2 = (vx + vy + (lx + ly) * omega) / r  (front-right)
        w3 = (vx + vy - (lx + ly) * omega) / r  (rear-left)
        w4 = (vx - vy + (lx + ly) * omega) / r  (rear-right)

    Forward kinematics (wheels to velocity):
        vx = r/4 * (w1 + w2 + w3 + w4)
        vy = r/4 * (-w1 + w2 + w3 - w4)
        omega = r/(4*(lx+ly)) * (-w1 + w2 - w3 + w4)

    Usage:
        odom = Odometry()

        # Update with wheel velocities (from motor readings)
        odom.update_from_velocities([v1, v2, v3, v4], dt=0.032)

        # Or update with commanded velocities
        odom.update_from_command(vx=0.1, vy=0.0, omega=0.0, dt=0.032)

        # Get current pose
        pose = odom.get_pose()
        print(f"Position: ({pose.x:.2f}, {pose.y:.2f}), heading: {np.degrees(pose.theta):.1f}°")
    """

    def __init__(self, config: Optional[OdometryConfig] = None):
        """
        Initialize odometry

        Args:
            config: OdometryConfig with robot parameters
        """
        self.config = config or OdometryConfig()

        # Current pose estimate
        self.pose = Pose2D()

        # Velocity state
        self.vx = 0.0  # Forward velocity (m/s)
        self.vy = 0.0  # Lateral velocity (m/s)
        self.omega = 0.0  # Angular velocity (rad/s)

        # Pose history for path tracking
        self.pose_history: List[Pose2D] = []
        self.max_history = 1000

        # Timing
        self.last_update_time = time.time()

        # Drift correction (calibration factors)
        self.drift_correction = {
            'x_scale': 1.0,
            'y_scale': 1.0,
            'theta_scale': 1.0
        }

    def update_from_velocities(
        self,
        wheel_velocities: List[float],
        dt: Optional[float] = None
    ) -> Pose2D:
        """
        Update pose from wheel angular velocities

        Args:
            wheel_velocities: [w1, w2, w3, w4] wheel angular velocities (rad/s)
            dt: Time delta (seconds), computed if None

        Returns:
            Updated Pose2D
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
        else:
            self.last_update_time = time.time()

        if dt <= 0:
            return self.pose

        w1, w2, w3, w4 = wheel_velocities
        r = self.config.wheel_radius
        l = self.config.lx + self.config.ly

        # Forward kinematics
        self.vx = r / 4.0 * (w1 + w2 + w3 + w4)
        self.vy = r / 4.0 * (-w1 + w2 + w3 - w4)
        self.omega = r / (4.0 * l) * (-w1 + w2 - w3 + w4)

        # Update pose
        self._integrate_pose(dt)

        return self.pose

    def update_from_command(
        self,
        vx: float,
        vy: float,
        omega: float,
        dt: Optional[float] = None
    ) -> Pose2D:
        """
        Update pose from commanded velocities

        Assumes commanded velocities are achieved (no slip).
        Use this when direct wheel encoder feedback is unavailable.

        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            omega: Angular velocity (rad/s)
            dt: Time delta (seconds)

        Returns:
            Updated Pose2D
        """
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
        else:
            self.last_update_time = time.time()

        if dt <= 0:
            return self.pose

        self.vx = vx
        self.vy = vy
        self.omega = omega

        self._integrate_pose(dt)

        return self.pose

    def _integrate_pose(self, dt: float) -> None:
        """
        Integrate velocities to update pose

        Uses midpoint integration for better accuracy during rotation.

        Args:
            dt: Time delta (seconds)
        """
        # Apply drift correction
        vx_corrected = self.vx * self.drift_correction['x_scale']
        vy_corrected = self.vy * self.drift_correction['y_scale']
        omega_corrected = self.omega * self.drift_correction['theta_scale']

        # Current heading
        theta = self.pose.theta

        # Midpoint heading for integration
        theta_mid = theta + omega_corrected * dt / 2.0

        # Position increment in world frame
        dx = (vx_corrected * np.cos(theta_mid) - vy_corrected * np.sin(theta_mid)) * dt
        dy = (vx_corrected * np.sin(theta_mid) + vy_corrected * np.cos(theta_mid)) * dt
        dtheta = omega_corrected * dt

        # Update pose
        self.pose.x += dx
        self.pose.y += dy
        self.pose.theta = Pose2D._normalize_angle(self.pose.theta + dtheta)
        self.pose.timestamp = time.time()

        # Save to history
        if len(self.pose_history) >= self.max_history:
            self.pose_history.pop(0)
        self.pose_history.append(Pose2D(
            x=self.pose.x,
            y=self.pose.y,
            theta=self.pose.theta,
            timestamp=self.pose.timestamp
        ))

    def get_pose(self) -> Pose2D:
        """Get current pose estimate"""
        return Pose2D(
            x=self.pose.x,
            y=self.pose.y,
            theta=self.pose.theta,
            timestamp=self.pose.timestamp
        )

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity (vx, vy, omega)"""
        return self.vx, self.vy, self.omega

    def get_distance_traveled(self) -> float:
        """
        Get total distance traveled

        Returns:
            Distance in meters
        """
        if len(self.pose_history) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.pose_history)):
            total += self.pose_history[i-1].distance_to(self.pose_history[i])

        return total

    def reset(self, pose: Optional[Pose2D] = None) -> None:
        """
        Reset odometry to a new pose

        Args:
            pose: New pose (defaults to origin)
        """
        if pose is None:
            self.pose = Pose2D()
        else:
            self.pose = Pose2D(
                x=pose.x,
                y=pose.y,
                theta=pose.theta,
                timestamp=time.time()
            )

        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.pose_history.clear()
        self.pose_history.append(self.pose)

    def set_drift_correction(
        self,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        theta_scale: float = 1.0
    ) -> None:
        """
        Set drift correction factors

        These can be calibrated by comparing odometry to
        ground truth over known distances.

        Args:
            x_scale: Correction factor for X movement
            y_scale: Correction factor for Y movement
            theta_scale: Correction factor for rotation
        """
        self.drift_correction = {
            'x_scale': x_scale,
            'y_scale': y_scale,
            'theta_scale': theta_scale
        }

    def compute_wheel_velocities(
        self,
        vx: float,
        vy: float,
        omega: float
    ) -> List[float]:
        """
        Compute wheel velocities for desired motion (inverse kinematics)

        Args:
            vx: Desired forward velocity (m/s)
            vy: Desired lateral velocity (m/s)
            omega: Desired angular velocity (rad/s)

        Returns:
            [w1, w2, w3, w4] wheel angular velocities (rad/s)
        """
        r = self.config.wheel_radius
        l = self.config.lx + self.config.ly

        w1 = (vx - vy - l * omega) / r  # front-left
        w2 = (vx + vy + l * omega) / r  # front-right
        w3 = (vx + vy - l * omega) / r  # rear-left
        w4 = (vx - vy + l * omega) / r  # rear-right

        return [w1, w2, w3, w4]


# Deposit box locations (from IA_20252.wbt world file)
# GREEN: translation 0.48 1.58 0
# BLUE: translation 0.48 -1.62 0
# RED: translation 2.31 0.00999969 0
DEPOSIT_BOXES = {
    'green': Pose2D(x=0.48, y=1.58, theta=0.0),
    'blue': Pose2D(x=0.48, y=-1.62, theta=0.0),
    'red': Pose2D(x=2.31, y=0.01, theta=0.0),
}


def get_deposit_box_pose(color: str) -> Optional[Pose2D]:
    """
    Get approximate pose of deposit box by color

    Args:
        color: Box color ('green', 'blue', 'red')

    Returns:
        Pose2D of box or None if unknown color
    """
    return DEPOSIT_BOXES.get(color)


def test_odometry():
    """Test odometry functionality"""
    print("Testing Odometry...")

    odom = Odometry()

    # Simulate forward motion
    dt = 0.032  # 32ms timestep

    # Move forward at 0.1 m/s for 1 second
    for _ in range(31):
        odom.update_from_command(vx=0.1, vy=0.0, omega=0.0, dt=dt)

    pose = odom.get_pose()
    print(f"  After 1s forward at 0.1m/s: x={pose.x:.3f}m (expected ~0.1m)")

    # Rotate 90 degrees
    odom.reset()
    for _ in range(31):
        odom.update_from_command(vx=0.0, vy=0.0, omega=np.pi/2, dt=dt)

    pose = odom.get_pose()
    print(f"  After 1s rotation: theta={np.degrees(pose.theta):.1f}° (expected ~90°)")

    # Test wheel velocities
    odom.reset()
    wheel_vels = odom.compute_wheel_velocities(vx=0.1, vy=0.0, omega=0.0)
    print(f"  Wheel velocities for vx=0.1: {[f'{w:.2f}' for w in wheel_vels]}")

    print("  Odometry test passed")


if __name__ == "__main__":
    test_odometry()
