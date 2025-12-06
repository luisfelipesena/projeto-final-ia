"""High-level base control utilities."""

from __future__ import annotations

from youbot.base import Base

import config
from data_types import MotionCommand, clamp


class BaseController:
    def __init__(self, base: Base):
        self._base = base

    def apply(self, command: MotionCommand) -> None:
        vx = clamp(command.vx, -config.BASE_MAX_SPEED, config.BASE_MAX_SPEED)
        vy = clamp(command.vy, -config.BASE_MAX_SPEED, config.BASE_MAX_SPEED)
        # Allow higher omega for faster rotation (1.0 rad/s max)
        omega = clamp(command.omega, -1.0, 1.0)
        self._base.move(vx, vy, omega)
