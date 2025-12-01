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
