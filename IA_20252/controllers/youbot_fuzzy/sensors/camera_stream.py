"""Camera acquisition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from controller import Camera  # type: ignore[import-not-found]

import config


@dataclass
class CameraFrame:
    image: bytes
    width: int
    height: int
    camera: Camera


class CameraStream:
    """Wraps Webots Camera API with utility helpers."""

    def __init__(self, robot):
        self._camera: Optional[Camera] = None
        try:
            self._camera = robot.getDevice(config.CAMERA_NAME)
        except AttributeError:
            self._camera = None
        if self._camera:
            self._camera.enable(int(robot.getBasicTimeStep()))

    def has_sensor(self) -> bool:
        return self._camera is not None

    def capture(self) -> Optional[CameraFrame]:
        if not self._camera:
            return None
        image = self._camera.getImage()
        return CameraFrame(
            image=image,
            width=self._camera.getWidth(),
            height=self._camera.getHeight(),
            camera=self._camera,
        )
