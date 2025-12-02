"""
ArmService - High-level arm and gripper control.
"""


class ArmService:
    """High-level arm and gripper control service."""

    def __init__(self, arm, gripper, robot, time_step: int):
        self.arm = arm
        self.gripper = gripper
        self.robot = robot
        self.time_step = time_step

    def set_height(self, height) -> None:
        """Set arm height preset."""
        self.arm.set_height(height)

    def set_orientation(self, orientation) -> None:
        """Set arm orientation preset."""
        self.arm.set_orientation(orientation)

    def reset_arm(self) -> None:
        """Reset arm to default position."""
        self.arm.reset()

    def reset(self) -> None:
        """Alias for reset_arm()."""
        self.reset_arm()

    def grip(self) -> None:
        """Close gripper."""
        self.gripper.grip()

    def release(self) -> None:
        """Open gripper."""
        self.gripper.release()

    def is_gripping(self) -> bool:
        """Check if gripper is closed."""
        return self.gripper.is_gripping

    def wait_for_arm(self, steps: int = 60) -> bool:
        """Wait for arm movement to complete."""
        for _ in range(steps):
            if self.robot.step(self.time_step) == -1:
                return False
        return True

    def prepare_for_grasp(self) -> bool:
        """Move arm to grasp position."""
        from arm import ArmHeight, ArmOrientation
        self.arm.set_orientation(ArmOrientation.FRONT)
        self.arm.set_height(ArmHeight.FRONT_FLOOR)
        return self.wait_for_arm()

    def lift_object(self) -> bool:
        """Lift arm after grasping."""
        from arm import ArmHeight
        self.arm.set_height(ArmHeight.FRONT_PLATE)
        return self.wait_for_arm()

    def prepare_for_deposit(self) -> bool:
        """Move arm to deposit position."""
        from arm import ArmHeight
        self.arm.set_height(ArmHeight.FRONT_CARDBOARD_BOX)
        return self.wait_for_arm()

    def prepare_deposit(self) -> bool:
        """Alias for prepare_for_deposit()."""
        return self.prepare_for_deposit()

    def prepare_grasp(self) -> bool:
        """Alias for prepare_for_grasp()."""
        return self.prepare_for_grasp()

    def execute_grasp(self, use_ik: bool = False, forward_reach: float = 0.25, lateral_offset: float = 0.0):
        """Execute grasp sequence and return result.

        Args:
            forward_reach: Forward distance in meters
            lateral_offset: Lateral offset in meters (positive = left)
        """
        from dataclasses import dataclass

        @dataclass
        class GraspResult:
            success: bool
            has_object: bool
            error: str = ""

        try:
            # Open gripper first
            self.release()
            self.wait_for_arm(20)

            # Use FRONT_FLOOR preset (proven to work in Webots examples)
            # IK had issues - preset is more reliable for floor-level grasping
            from arm import ArmHeight
            print(f"[GRASP] Using FRONT_FLOOR preset (reach={forward_reach:.3f}m, lateral={lateral_offset:.3f}m)")
            self.arm.set_height(ArmHeight.FRONT_FLOOR)
            self.wait_for_arm(60)  # Wait for arm to reach position

            # Close gripper
            self.grip()
            self.wait_for_arm(40)

            # Check if object was grasped - with debug info
            has_object = self.gripper.has_object()

            # Debug: check gripper sensor state
            sensor_info = "no_sensor"
            if self.gripper.finger_sensor:
                try:
                    pos = self.gripper.finger_sensor.getValue()
                    sensor_info = f"pos={pos:.4f}"
                except:
                    sensor_info = "sensor_error"
            print(f"[GRASP] Gripper check: {sensor_info}, is_gripping={self.gripper.is_gripping}, has_object={has_object}")

            if has_object:
                # Lift the object
                self.lift_object()
                return GraspResult(success=True, has_object=True)
            else:
                return GraspResult(success=False, has_object=False, error="No object detected")

        except Exception as e:
            return GraspResult(success=False, has_object=False, error=str(e))

    def execute_deposit(self) -> bool:
        """Execute deposit sequence - release object."""
        self.release()
        return self.wait_for_arm(30)

    def return_to_rest(self) -> bool:
        """Return arm to rest/reset position."""
        self.reset_arm()
        return self.wait_for_arm(60)
