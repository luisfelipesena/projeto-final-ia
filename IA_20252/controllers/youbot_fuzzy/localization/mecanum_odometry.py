"""Odometry estimation for the YouBot Mecanum drive."""

from __future__ import annotations

import math
from typing import List, Optional

from ..types import MotionCommand


class MecanumOdometry:
    """Fuse wheel encoders (when available) with commanded velocities."""

    def __init__(
        self,
        robot,
        wheel_radius: float = 0.02375,
        lx: float = 0.165,
        ly: float = 0.115,
    ):
        self._robot = robot
        self._time_step = int(robot.getBasicTimeStep())
        self._dt = self._time_step / 1000.0
        self._encoder_names = [
            "wheel1_sensor",
            "wheel2_sensor",
            "wheel3_sensor",
            "wheel4_sensor",
        ]
        self._encoders = self._acquire_encoders()
        self._prev_positions = [sensor.getValue() if sensor else 0.0 for sensor in self._encoders]
        self._wheel_radius = wheel_radius
        self._L = lx + ly
        self._pose = [0.0, 0.0, 0.0]
        self._distance_accumulator = 0.0

    def _acquire_encoders(self) -> List[Optional[object]]:
        encoders: List[Optional[object]] = []
        for name in self._encoder_names:
            try:
                sensor = self._robot.getDevice(name)
                sensor.enable(self._time_step)
            except (AttributeError, RuntimeError):
                sensor = None
            encoders.append(sensor)
        return encoders

    @property
    def pose(self) -> tuple[float, float, float]:
        return tuple(self._pose)

    @property
    def distance_since_reset(self) -> float:
        return self._distance_accumulator

    def reset_distance_accumulator(self) -> None:
        self._distance_accumulator = 0.0

    def update(self, command: Optional[MotionCommand] = None) -> tuple[float, float, float]:
        if any(sensor is not None for sensor in self._encoders):
            self._update_from_encoders()
        elif command is not None:
            self._integrate_velocity(command)
        return self.pose

    # ------------------------------------------------------------------
    def _update_from_encoders(self) -> None:
        positions = []
        for sensor in self._encoders:
            if sensor is None:
                positions.append(None)
            else:
                positions.append(sensor.getValue())
        w = [0.0, 0.0, 0.0, 0.0]
        for i, value in enumerate(positions):
            if value is None:
                continue
            delta = value - self._prev_positions[i]
            w[i] = delta / self._dt
            self._prev_positions[i] = value
        vx, vy, omega = self._wheel_speeds_to_body_velocity(w)
        self._integrate(vx, vy, omega)

    def _integrate_velocity(self, command: MotionCommand) -> None:
        vx = command.vx
        vy = command.vy
        omega = command.omega
        self._integrate(vx, vy, omega)

    def _integrate(self, vx: float, vy: float, omega: float) -> None:
        x, y, theta = self._pose
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x += (vx * cos_t - vy * sin_t) * self._dt
        y += (vx * sin_t + vy * cos_t) * self._dt
        theta += omega * self._dt
        theta = math.atan2(math.sin(theta), math.cos(theta))
        distance = math.sqrt((vx * self._dt) ** 2 + (vy * self._dt) ** 2)
        self._distance_accumulator += distance
        self._pose = [x, y, theta]

    def _wheel_speeds_to_body_velocity(self, w: List[float]) -> tuple[float, float, float]:
        r4 = self._wheel_radius / 4.0
        vx = (w[0] + w[1] + w[2] + w[3]) * r4
        vy = (-w[0] + w[1] + w[2] - w[3]) * r4
        omega = (-w[0] + w[1] - w[2] + w[3]) * r4 / self._L
        return vx, vy, omega
