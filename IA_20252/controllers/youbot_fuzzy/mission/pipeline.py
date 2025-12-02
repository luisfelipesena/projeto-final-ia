"""Mission state machine for the fuzzy YouBot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

from ..types import CubeHypothesis, LidarSnapshot, MotionCommand
from .. import config
from ..control.fuzzy_planner import FuzzyPlanner
from ..manipulation.arm_service import ArmService
from ..motion.base_controller import BaseController
from ..localization.mecanum_odometry import MecanumOdometry
from ..localization.icp_correction import ICPCorrection
from ..sensors.lidar_adapter import LidarAdapter
from ..sensors.camera_stream import CameraStream
from ..perception.cube_detector import CubeDetector
from ..world.model import WorldModel


@dataclass
class MissionState:
    load_state: bool = False
    collected: int = 0
    target_index: int = 0
    phase: str = "SCAN_GRID"


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
        odometry: MecanumOdometry,
        icp: ICPCorrection,
        logger=None,
    ):
        self.base_controller = base_controller
        self.arm_service = arm_service
        self.lidar = lidar
        self.camera = camera
        self.detector = detector
        self.world = world
        self.planner = planner
        self.odometry = odometry
        self.icp = icp
        self.state = MissionState()
        self.logger = logger
        self._log_counter = 0

    def step(self) -> None:
        snapshot = self._read_lidar()
        self.world.update_obstacles(snapshot)

        frame = self.camera.capture()
        cube = self.detector.detect(frame)
        self.world.update_cube(cube)
        if not self.state.load_state:
            if cube.color:
                self.state.phase = "PICK"
            elif self.state.phase not in {"DELIVER", "RETURN"}:
                self.state.phase = "SCAN_GRID"
        nav_points = tuple(self.lidar.navigation_points())
        cube_points = tuple(self.lidar.cube_candidates())
        current_pose = self.world.pose
        patch_vector_world = self.world.next_patch_vector()
        goal_vector_world = self.world.goal_vector()
        patch_vector_body = self._world_to_robot(patch_vector_world, current_pose[2])
        goal_vector_body = self._world_to_robot(goal_vector_world, current_pose[2])

        command = self.planner.plan(
            snapshot,
            cube,
            self.state.load_state,
            patch_vector=patch_vector_body,
            goal_vector=goal_vector_body,
            cube_candidates=len(cube_points),
        )
        self.base_controller.apply(command)
        self.arm_service.queue(command.lift_request, command.gripper_request)
        self.arm_service.update()
        pose = self.odometry.update(command)
        if self.icp.available() and self.odometry.distance_since_reset >= self.icp.distance_threshold:
            pose = tuple(self.icp.correct(nav_points, pose))
            self.odometry.reset_distance_accumulator()
        self.world.update_pose(pose)
        self.world.integrate_lidar(nav_points, cube_points)

        self._update_load_state(command)
        self._log_state(snapshot, cube, command, pose)

    def _read_lidar(self) -> LidarSnapshot:
        if not self.lidar.has_high():
            return LidarSnapshot()
        return self.lidar.summarize()

    def _update_load_state(self, command) -> None:
        if not self.state.load_state and self.arm_service.is_gripping:
            self.state.load_state = True
            self.state.phase = "DELIVER"
        elif self.state.load_state and not self.arm_service.is_gripping and command.gripper_request == "RELEASE":
            self.state.load_state = False
            self.state.collected += 1
            self.world.advance_goal()
            self.state.phase = "RETURN"
        elif not self.state.load_state and self.state.phase == "RETURN":
            self.state.phase = "SCAN_GRID"

    def _log_state(self, obstacles: LidarSnapshot, cube: CubeHypothesis, command: MotionCommand, pose: Tuple[float, float, float]) -> None:
        if not self.logger or not config.ENABLE_LOGGING:
            return
        self._log_counter += 1
        if self._log_counter % config.LOG_INTERVAL_STEPS != 0:
            return
        self.logger.info(
            "front=%.2f left=%.2f right=%.2f cube=%s conf=%.2f vx=%.2f vy=%.2f omega=%.2f load=%s phase=%s pose=(%.2f, %.2f, %.2f)",
            obstacles.front_distance,
            obstacles.left_distance,
            obstacles.right_distance,
            cube.color or "-",
            cube.confidence or 0.0,
            command.vx,
            command.vy,
            command.omega,
            self.state.load_state,
            self.state.phase,
            pose[0],
            pose[1],
            pose[2],
        )

    @staticmethod
    def _world_to_robot(vector: Tuple[float, float], theta: float) -> Tuple[float, float]:
        dx, dy = vector
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx = dx * cos_t + dy * sin_t
        vy = -dx * sin_t + dy * cos_t
        return vx, vy
