"""Sequences arm and gripper actions."""

from __future__ import annotations

from typing import Optional

from youbot.arm import Arm, ArmHeight
from youbot.gripper import Gripper

from .. import config


HEIGHT_PRESETS = {
    "RESET": ArmHeight.RESET,
    "FLOOR": ArmHeight.FRONT_FLOOR,
    "PLATE": ArmHeight.FRONT_PLATE,
}


class ArmService:
    def __init__(self, arm: Arm, gripper: Gripper, robot):
        self._arm = arm
        self._gripper = gripper
        self._robot = robot
        self._time_step = int(robot.getBasicTimeStep())
        self._pending_height: Optional[str] = None
        self._pending_gripper: Optional[str] = None
        self._wait_timer_ms = 0
        self._is_gripping = False

    def queue(self, lift_request: Optional[str], gripper_request: Optional[str]) -> None:
        if lift_request:
            self._pending_height = lift_request
        if gripper_request:
            self._pending_gripper = gripper_request

    def update(self) -> None:
        if self._wait_timer_ms > 0:
            self._wait_timer_ms -= self._time_step
            return

        if self._pending_height:
            self._apply_height(self._pending_height)
            self._wait_timer_ms = int(config.ARM_WAIT_SECONDS.get(self._pending_height, 1.0) * 1000)
            self._pending_height = None
            return

        if self._pending_gripper:
            self._apply_gripper(self._pending_gripper)
            self._wait_timer_ms = 800
            self._pending_gripper = None

    @property
    def is_gripping(self) -> bool:
        return self._is_gripping

    def _apply_height(self, preset: str) -> None:
        target = HEIGHT_PRESETS.get(preset)
        if target is None:
            return
        self._arm.set_height(target)

    def _apply_gripper(self, action: str) -> None:
        if action == "GRIP":
            self._gripper.grip()
            self._is_gripping = True
        elif action == "RELEASE":
            self._gripper.release()
            self._is_gripping = False
