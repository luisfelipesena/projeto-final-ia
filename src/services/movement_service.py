"""
MovementService - Basic motion commands without perception dependency

Based on: Siegwart & Nourbakhsh (2004) - Introduction to Autonomous Mobile Robots
"""

import math
import time
from typing import Optional, Callable


class MovementService:
    """
    Simple motion commands for YouBot base.

    NO perception dependency - pure dead-reckoning motion.
    Test this FIRST before integrating with vision.

    Usage:
        movement = MovementService(base, robot, time_step)

        # Drive 1 meter forward
        movement.forward(1.0)

        # Turn 90 degrees left
        movement.turn(90)

        # Test square pattern
        movement.test_square()
    """

    # Default speeds
    DEFAULT_LINEAR_SPEED = 0.15   # m/s
    DEFAULT_ANGULAR_SPEED = 0.4  # rad/s
    DEFAULT_STRAFE_SPEED = 0.10  # m/s

    def __init__(self, base, robot, time_step: int):
        """
        Initialize MovementService.

        Args:
            base: Base controller instance (from base.py)
            robot: Webots Robot instance (for step())
            time_step: Simulation time step in ms
        """
        self.base = base
        self.robot = robot
        self.time_step = time_step
        self.is_moving = False

        # Callbacks for external control
        self._step_callback: Optional[Callable] = None

    def set_step_callback(self, callback: Callable) -> None:
        """Set callback to run on each simulation step during motion."""
        self._step_callback = callback

    def _step(self) -> bool:
        """Execute one simulation step. Returns False if simulation ended."""
        result = self.robot.step(self.time_step)
        if self._step_callback:
            self._step_callback()
        return result != -1

    def _execute_motion(self, vx: float, vy: float, omega: float, duration: float) -> bool:
        """
        Execute motion for specified duration.

        Args:
            vx: Forward velocity (m/s)
            vy: Strafe velocity (m/s)
            omega: Angular velocity (rad/s)
            duration: Time to execute (seconds)

        Returns:
            True if completed, False if simulation ended
        """
        print(f"[Movement] Execute: vx={vx:.2f} vy={vy:.2f} omega={omega:.2f} dur={duration:.2f}s")
        self.is_moving = True
        self.base.move(vx, vy, omega)

        steps = int(duration * 1000 / self.time_step)
        for _ in range(steps):
            if not self._step():
                self.is_moving = False
                return False

        self.stop()
        return True

    def forward(self, distance_m: float, speed: float = None) -> bool:
        """
        Drive forward X meters.

        Args:
            distance_m: Distance to travel (meters)
            speed: Linear speed (m/s), default 0.15

        Returns:
            True if completed
        """
        speed = speed or self.DEFAULT_LINEAR_SPEED
        duration = abs(distance_m) / speed
        vx = speed if distance_m > 0 else -speed
        return self._execute_motion(vx=vx, vy=0, omega=0, duration=duration)

    def backward(self, distance_m: float, speed: float = None) -> bool:
        """
        Drive backward X meters.

        Args:
            distance_m: Distance to travel (meters)
            speed: Linear speed (m/s), default 0.10

        Returns:
            True if completed
        """
        speed = speed or (self.DEFAULT_LINEAR_SPEED * 0.67)  # Slower backward
        duration = abs(distance_m) / speed
        return self._execute_motion(vx=-speed, vy=0, omega=0, duration=duration)

    def turn(self, angle_deg: float, speed: float = None) -> bool:
        """
        Turn X degrees (positive=left/CCW, negative=right/CW).

        Args:
            angle_deg: Angle to turn (degrees)
            speed: Angular speed (rad/s), default 0.4

        Returns:
            True if completed
        """
        speed = speed or self.DEFAULT_ANGULAR_SPEED
        angle_rad = math.radians(abs(angle_deg))
        duration = angle_rad / speed
        omega = speed if angle_deg > 0 else -speed
        return self._execute_motion(vx=0, vy=0, omega=omega, duration=duration)

    def strafe(self, distance_m: float, speed: float = None) -> bool:
        """
        Strafe sideways X meters (positive=left, negative=right).

        Args:
            distance_m: Distance to strafe (meters)
            speed: Strafe speed (m/s), default 0.10

        Returns:
            True if completed
        """
        speed = speed or self.DEFAULT_STRAFE_SPEED
        duration = abs(distance_m) / speed
        vy = speed if distance_m > 0 else -speed
        return self._execute_motion(vx=0, vy=vy, omega=0, duration=duration)

    def stop(self) -> None:
        """Immediately stop all motion."""
        self.base.move(vx=0, vy=0, omega=0)
        self.is_moving = False

    def move_continuous(self, vx: float, vy: float, omega: float) -> None:
        """
        Start continuous motion (non-blocking).
        Call stop() to halt.

        Args:
            vx: Forward velocity (m/s)
            vy: Strafe velocity (m/s)
            omega: Angular velocity (rad/s)
        """
        self.base.move(vx, vy, omega)
        self.is_moving = True

    # ==================== TEST METHODS ====================

    def test_square(self, side_length: float = 1.0) -> bool:
        """
        Test: Drive in a 1m x 1m square pattern.

        Expected: Robot returns close to starting position.

        Args:
            side_length: Length of each side (meters)

        Returns:
            True if completed
        """
        print(f"[MovementService] TEST: Square pattern ({side_length}m sides)")

        for i in range(4):
            print(f"  Side {i+1}/4: Forward {side_length}m")
            if not self.forward(side_length):
                return False

            print(f"  Turn 90° left")
            if not self.turn(90):
                return False

        print("[MovementService] TEST COMPLETE: Square pattern")
        return True

    def test_forward_backward(self, distance: float = 0.5) -> bool:
        """
        Test: Drive forward then backward same distance.

        Expected: Robot returns to starting position.
        """
        print(f"[MovementService] TEST: Forward/Backward ({distance}m)")

        print(f"  Forward {distance}m")
        if not self.forward(distance):
            return False

        # Pause
        for _ in range(10):
            if not self._step():
                return False

        print(f"  Backward {distance}m")
        if not self.backward(distance):
            return False

        print("[MovementService] TEST COMPLETE: Forward/Backward")
        return True

    def test_rotation(self, angle: float = 360) -> bool:
        """
        Test: Rotate full circle.

        Expected: Robot faces same direction after completion.
        """
        print(f"[MovementService] TEST: Rotation ({angle}°)")

        if not self.turn(angle):
            return False

        print("[MovementService] TEST COMPLETE: Rotation")
        return True


# ==================== STANDALONE TEST ====================

def test_movement_service():
    """
    Standalone test for MovementService.

    Run: python -m src.services.movement_service --test square
    """
    import sys

    try:
        from controller import Robot
        from IA_20252.controllers.youbot.base import Base
    except ImportError:
        print("ERROR: Must run inside Webots simulation")
        return

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())
    base = Base(robot)

    movement = MovementService(base, robot, time_step)

    # Parse test type
    test_type = "square"
    if len(sys.argv) > 1:
        if "--test" in sys.argv:
            idx = sys.argv.index("--test")
            if idx + 1 < len(sys.argv):
                test_type = sys.argv[idx + 1]

    print(f"[MovementService] Running test: {test_type}")

    # Wait for simulation to start
    for _ in range(10):
        robot.step(time_step)

    if test_type == "square":
        movement.test_square(1.0)
    elif test_type == "forward":
        movement.test_forward_backward(0.5)
    elif test_type == "rotation":
        movement.test_rotation(360)
    else:
        print(f"Unknown test: {test_type}")
        print("Available tests: square, forward, rotation")


if __name__ == "__main__":
    test_movement_service()
