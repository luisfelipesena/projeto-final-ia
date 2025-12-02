"""Mission state machine for the fuzzy YouBot."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import CubeHypothesis, LidarSnapshot, MotionCommand
from .. import config
from ..control.fuzzy_planner import FuzzyPlanner
from ..manipulation.arm_service import ArmService
from ..motion.base_controller import BaseController
from ..sensors.lidar_adapter import LidarAdapter
from ..sensors.camera_stream import CameraStream
from ..perception.cube_detector import CubeDetector
from ..world.model import WorldModel


@dataclass
class MissionState:
    load_state: bool = False
    collected: int = 0
    target_index: int = 0


class MissionPipeline:
    def __init__(
        self,
        base_controller: BaseController,
        arm_service: ArmService,
        lidar: LidarAdapter,
        camera: CameraStream,
        detector: CubeDetector,
        world: WorldModel,
        planner: FuzzyPlanner,
        logger=None,
    ):
        self.base_controller = base_controller
        self.arm_service = arm_service
        self.lidar = lidar
        self.camera = camera
        self.detector = detector
        self.world = world
        self.planner = planner
        self.state = MissionState()
        self.logger = logger
        self._log_counter = 0

    def step(self) -> None:
        snapshot = self._read_lidar()
        self.world.update_obstacles(snapshot)

        frame = self.camera.capture()
        cube = self.detector.detect(frame)
        self.world.update_cube(cube)

        command = self.planner.plan(snapshot, cube, self.state.load_state)
        self.base_controller.apply(command)
        self.arm_service.queue(command.lift_request, command.gripper_request)
        self.arm_service.update()

        self._update_load_state(command)
        self._log_state(snapshot, cube, command)

    def _read_lidar(self) -> LidarSnapshot:
        if not self.lidar.has_sensor():
            return LidarSnapshot()
        return self.lidar.summarize()

    def _update_load_state(self, command) -> None:
        if not self.state.load_state and self.arm_service.is_gripping:
            self.state.load_state = True
        elif self.state.load_state and not self.arm_service.is_gripping and command.gripper_request == "RELEASE":
            self.state.load_state = False
            self.state.collected += 1
            self.world.advance_goal()

    def _log_state(self, obstacles: LidarSnapshot, cube: CubeHypothesis, command: MotionCommand) -> None:
        if not self.logger or not config.ENABLE_LOGGING:
            return
        self._log_counter += 1
        if self._log_counter % config.LOG_INTERVAL_STEPS != 0:
            return
        self.logger.info(
            "front=%.2f left=%.2f right=%.2f cube=%s conf=%.2f vx=%.2f vy=%.2f omega=%.2f load=%s",
            obstacles.front_distance,
            obstacles.left_distance,
            obstacles.right_distance,
            cube.color or "-",
            cube.confidence or 0.0,
            command.vx,
            command.vy,
            command.omega,
            self.state.load_state,
        )
