"""High-level base control utilities."""

from __future__ import annotations

from youbot.base import Base

from .. import config
from ..types import MotionCommand, clamp


class BaseController:
    def __init__(self, base: Base):
        self._base = base

    def apply(self, command: MotionCommand) -> None:
        vx = clamp(command.vx, -config.BASE_MAX_SPEED, config.BASE_MAX_SPEED)
        vy = clamp(command.vy, -config.BASE_MAX_SPEED, config.BASE_MAX_SPEED)
        omega = clamp(command.omega, -config.BASE_MAX_SPEED, config.BASE_MAX_SPEED)
        self._base.move(vx, vy, omega)
