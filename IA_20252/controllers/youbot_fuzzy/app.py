"""Main entrypoint for the YouBot fuzzy controller."""

from __future__ import annotations

from controller import Robot  # type: ignore[import-not-found]

from youbot.arm import Arm
from youbot.base import Base
from youbot.gripper import Gripper

import config
from control.fuzzy_planner import FuzzyPlanner
from logger import get_logger
from manipulation.arm_service import ArmService
from mission.pipeline import MissionPipeline
from motion.base_controller import BaseController
from perception.adaboost_classifier import AdaBoostColorClassifier
from perception.color_classifier import ColorClassifier
from perception.cube_detector import CubeDetector
from perception.yolo_detector import YoloDetector
from localization.mecanum_odometry import MecanumOdometry
from localization.icp_correction import ICPCorrection
from sensors.camera_stream import CameraStream
from sensors.lidar_adapter import LidarAdapter
from world.model import WorldModel


class YouBotFuzzyApp:
    """Wire up sensors, world model, planner, and mission state machine."""

    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self.lidar = LidarAdapter(self.robot)
        self.camera = CameraStream(self.robot)
        self.yolo_detector = YoloDetector()
        self.adaboost_classifier = AdaBoostColorClassifier()
        self.color_classifier = ColorClassifier(adaboost=self.adaboost_classifier)
        self.detector = CubeDetector(
            classifier=self.color_classifier,
            yolo=self.yolo_detector,
            adaboost=self.adaboost_classifier,
        )

        self.world = WorldModel()
        self.planner = FuzzyPlanner()
        self.base_controller = BaseController(self.base)
        self.arm_service = ArmService(self.arm, self.gripper, self.robot)
        self.odometry = MecanumOdometry(self.robot)
        self.icp = ICPCorrection()
        self.logger = get_logger() if config.ENABLE_LOGGING else None

        self.mission = MissionPipeline(
            base_controller=self.base_controller,
            arm_service=self.arm_service,
            lidar=self.lidar,
            camera=self.camera,
            detector=self.detector,
            world=self.world,
            planner=self.planner,
            odometry=self.odometry,
            icp=self.icp,
            logger=self.logger,
        )

    def run(self):
        while self.robot.step(self.time_step) != -1:
            self.mission.step()


def main():
    app = YouBotFuzzyApp()
    app.run()


if __name__ == "__main__":
    main()
