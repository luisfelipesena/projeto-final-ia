"""Mission state machine for the fuzzy YouBot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math

from data_types import CubeHypothesis, LidarSnapshot, MotionCommand
import config
from control.fuzzy_planner import FuzzyPlanner
from manipulation.arm_service import ArmService
from motion.base_controller import BaseController
from localization.mecanum_odometry import MecanumOdometry
from localization.icp_correction import ICPCorrection
from sensors.lidar_adapter import LidarAdapter
from sensors.camera_stream import CameraStream
from perception.cube_detector import CubeDetector
from world.model import WorldModel


@dataclass
class MissionState:
    load_state: bool = False
    collected: int = 0
    target_index: int = 0
    phase: str = "SCAN_GRID"
    pick_sub_phase: str = ""  # Sub-phases: OPEN_GRIPPER, LOWER_ARM, APPROACH, GRIP, LIFT
    pick_timer_ms: int = 0
    approach_steps: int = 0


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

        # Get LIDAR cube candidates (only for logging/aux data; no direct trigger)
        cube_points = tuple(self.lidar.cube_candidates())

        # Phase transitions - only camera triggers PICK; LIDAR no longer triggers
        if not self.state.load_state:
            # Primary: camera detected cube
            if cube.color and (cube.confidence or 0.0) >= 0.5 and self.state.phase not in {"PICK"}:
                self.state.phase = "PICK"
                self.state.pick_sub_phase = "OPEN_GRIPPER"
                self.state.pick_timer_ms = 0
                if config.ENABLE_LOGGING:
                    print(f"PICK_TRIGGER: Camera detected {cube.color} conf={cube.confidence:.2f}")
            elif not cube.color and self.state.phase not in {"DELIVER", "RETURN", "PICK"}:
                self.state.phase = "SCAN_GRID"

        nav_points = tuple(self.lidar.navigation_points())
        current_pose = self.world.pose
        patch_vector_world = self.world.next_patch_vector()
        goal_vector_world = self.world.goal_vector()
        patch_vector_body = self._world_to_robot(patch_vector_world, current_pose[2])
        goal_vector_body = self._world_to_robot(goal_vector_world, current_pose[2])

        # Handle PICK phase with proper grasp sequence
        if self.state.phase == "PICK":
            command = self._handle_pick_sequence(snapshot, cube)
        else:
            command = self.planner.plan(
                snapshot,
                cube,
                self.state.load_state,
                patch_vector=patch_vector_body,
                goal_vector=goal_vector_body,
                cube_candidates=len(cube_points),
                world=self.world,
            )

        self.base_controller.apply(command)
        self.arm_service.queue(command.lift_request, command.gripper_request)
        self.arm_service.update()
        pose = self.odometry.update(command)
        if self.icp.available() and len(nav_points) > 10 and self.odometry.distance_since_reset >= self.icp.distance_threshold:
            pose = tuple(self.icp.correct(nav_points, pose))
            self.odometry.reset_distance_accumulator()
        self.world.update_pose(pose)
        self.world.integrate_lidar(nav_points, cube_points)

        self._update_load_state(command)
        self._log_state(snapshot, cube, command, pose)

    def _handle_pick_sequence(self, snapshot: LidarSnapshot, cube: CubeHypothesis) -> MotionCommand:
        """Handle the PICK phase with proper grasp sequence per GRASP_TEST.md."""
        time_step = int(self.base_controller._base.time_step)

        # Decrement timer
        if self.state.pick_timer_ms > 0:
            self.state.pick_timer_ms -= time_step
            # While waiting, keep robot still
            return MotionCommand(vx=0.0, vy=0.0, omega=0.0)

        sub = self.state.pick_sub_phase

        if sub == "OPEN_GRIPPER":
            # Step 1: Open gripper
            self.state.pick_timer_ms = 1000  # 1 second wait
            self.state.pick_sub_phase = "RESET_ARM"
            return MotionCommand(gripper_request="RELEASE")

        elif sub == "RESET_ARM":
            # Step 2: Reset arm to safe position (per GRASP_TEST.md)
            self.state.pick_timer_ms = 1500  # 1.5 seconds for RESET
            self.state.pick_sub_phase = "LOWER_ARM"
            return MotionCommand(lift_request="RESET")

        elif sub == "LOWER_ARM":
            # Second: Lower arm to floor position
            self.state.pick_timer_ms = 2500  # 2.5 seconds for arm to reach floor
            self.state.pick_sub_phase = "APPROACH"
            self.state.approach_steps = 0
            return MotionCommand(lift_request="FLOOR")

        elif sub == "APPROACH":
            # Third: Move forward slowly towards cube (2 seconds at 0.05 m/s = 10cm)
            self.state.approach_steps += 1
            approach_duration_steps = int(2000 / time_step)  # 2 seconds worth of steps

            if self.state.approach_steps >= approach_duration_steps:
                self.state.pick_sub_phase = "GRIP"
                self.state.pick_timer_ms = 0
                return MotionCommand(vx=0.0)

            # Slow forward approach - align with cube if detected
            alignment = cube.alignment if cube.alignment else 0.0
            vy = -alignment * 0.3  # Lateral correction
            return MotionCommand(vx=0.05, vy=vy, omega=0.0)

        elif sub == "GRIP":
            # Fourth: Close gripper
            self.state.pick_timer_ms = 1500  # 1.5 seconds for grip
            self.state.pick_sub_phase = "LIFT"
            return MotionCommand(gripper_request="GRIP")

        elif sub == "LIFT":
            # Fifth: Lift to plate height
            self.state.pick_timer_ms = 2000  # 2 seconds for lift
            self.state.pick_sub_phase = "DONE"
            return MotionCommand(lift_request="PLATE")

        elif sub == "DONE":
            # Pick complete - transition to DELIVER
            # Note: load_state is set by _update_load_state based on arm_service.is_gripping
            self.state.phase = "DELIVER"
            self.state.pick_sub_phase = ""
            return MotionCommand()

        # Default - shouldn't reach here
        return MotionCommand()

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
        # Reduce log frequency further: 6x slower than base, min every 50 steps
        if self._log_counter % max(config.LOG_INTERVAL_STEPS * 6, 50) != 0:
            return

        # Get known map distances in robot frame for comparison
        walls_robot = self.world.distance_to_walls_robot_frame()
        obs_dist, obs_dx, obs_dy = self.world.distance_to_nearest_obstacle()

        phase_str = self.state.phase
        if self.state.phase == "PICK" and self.state.pick_sub_phase:
            phase_str = f"PICK:{self.state.pick_sub_phase}"

        self.logger.info(
            "LIDAR: f=%.2f l=%.2f r=%.2f | MAP_ROBOT: f=%.2f l=%.2f r=%.2f obs=%.2f(dx=%.1f,dy=%.1f) | cube=%s conf=%.2f | vx=%.2f vy=%.2f ω=%.2f | %s pose=(%.2f,%.2f,%.1f°)",
            obstacles.front_distance,
            obstacles.left_distance,
            obstacles.right_distance,
            walls_robot["front"],
            walls_robot["left"],
            walls_robot["right"],
            obs_dist,
            obs_dx,
            obs_dy,
            cube.color or "-",
            cube.confidence or 0.0,
            command.vx,
            command.vy,
            command.omega,
            phase_str,
            pose[0],
            pose[1],
            math.degrees(pose[2]),
        )

    @staticmethod
    def _world_to_robot(vector: Tuple[float, float], theta: float) -> Tuple[float, float]:
        dx, dy = vector
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx = dx * cos_t + dy * sin_t
        vy = -dx * sin_t + dy * cos_t
        return vx, vy
