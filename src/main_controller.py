"""
Main YouBot Controller - MATA64 Final Project.

Integrates all modules:
- Perception: CubeDetector (HSV), LidarProcessor, LidarMLP (RNA)
- Control: StateMachine, FuzzyNavigator, FuzzyManipulator
- Actuators: BaseController, ArmController, GripperController

The robot autonomously collects 15 colored cubes and deposits them
in corresponding colored boxes using LIDAR + Camera (no GPS).
"""

from controller import Robot, Camera, Lidar
import numpy as np

# Import all modules
from .perception import CubeDetector, LidarProcessor, LidarMLP
from .control import StateMachine, RobotState, FuzzyNavigator, FuzzyManipulator, ManipulationAction
from .actuators import BaseController, ArmController, ArmHeight, GripperController
from .utils.config import DEPOSIT_BOXES, NAVIGATION, GRASP


class YouBotController:
    """Main controller for YouBot cube collection task."""

    def __init__(self, robot: Robot):
        """Initialize all components.

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Initialize actuators
        self.base = BaseController(robot)
        self.arm = ArmController(robot)
        self.gripper = GripperController(robot)

        # Initialize sensors
        self._init_sensors()

        # Initialize perception
        self.cube_detector = CubeDetector()
        self.lidar_processor = LidarProcessor()
        self.lidar_mlp = LidarMLP()

        # Try to load trained LIDAR model
        self.lidar_mlp.load()

        # Initialize control
        self.state_machine = StateMachine()
        self.fuzzy_navigator = FuzzyNavigator()
        self.fuzzy_manipulator = FuzzyManipulator()

        # Register state handlers
        self._register_state_handlers()

        # Internal state
        self._search_direction = 1  # 1 = left, -1 = right
        self._grasp_step = 0
        self._deposit_target = None
        self._steps_without_cube = 0

        print("[YOUBOT] Controller initialized")

    def _init_sensors(self) -> None:
        """Initialize camera and LIDAR sensors."""
        # Camera
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            print(f"[YOUBOT] Camera enabled: {self.camera.getWidth()}x{self.camera.getHeight()}")
        else:
            print("[YOUBOT] Warning: Camera not found")

        # LIDAR
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            self.lidar.enablePointCloud()
            print(f"[YOUBOT] LIDAR enabled: {self.lidar.getNumberOfPoints()} points")
        else:
            print("[YOUBOT] Warning: LIDAR not found")

    def _register_state_handlers(self) -> None:
        """Register handlers for each state."""
        self.state_machine.register_handler(RobotState.SEARCHING, self._handle_searching)
        self.state_machine.register_handler(RobotState.APPROACHING, self._handle_approaching)
        self.state_machine.register_handler(RobotState.ALIGNING, self._handle_aligning)
        self.state_machine.register_handler(RobotState.GRASPING, self._handle_grasping)
        self.state_machine.register_handler(RobotState.TRANSPORTING, self._handle_transporting)
        self.state_machine.register_handler(RobotState.DEPOSITING, self._handle_depositing)
        self.state_machine.register_handler(RobotState.RECOVERING, self._handle_recovering)

        # Enter callbacks
        self.state_machine.register_enter_callback(RobotState.GRASPING, self._on_enter_grasping)
        self.state_machine.register_enter_callback(RobotState.DEPOSITING, self._on_enter_depositing)

    def run(self) -> None:
        """Main control loop."""
        print("[YOUBOT] Starting main loop")

        # Start state machine
        self.state_machine.start()

        # Prepare arm for operation
        self.arm.set_height(ArmHeight.FRONT_FLOOR)
        self.gripper.release()

        while self.robot.step(self.time_step) != -1:
            # Read sensors
            image = self._get_camera_image()
            lidar_data = self._get_lidar_data()

            # Process perception
            cube = self.cube_detector.detect_nearest(image) if image is not None else None
            sectors = self.lidar_processor.process(lidar_data) if lidar_data is not None else []

            # Update state machine context
            self._update_context(cube, sectors)

            # Run state machine
            self.state_machine.update()

            # Check completion
            if self.state_machine.context.cubes_collected >= 15:
                print("[YOUBOT] Task complete! 15 cubes collected.")
                self.base.stop()
                break

    def _get_camera_image(self) -> np.ndarray:
        """Get camera image as numpy array."""
        if not self.camera:
            return None

        image_data = self.camera.getImage()
        if image_data is None:
            return None

        return CubeDetector.webots_image_to_numpy(
            image_data,
            self.camera.getWidth(),
            self.camera.getHeight()
        )

    def _get_lidar_data(self) -> np.ndarray:
        """Get LIDAR range data as numpy array."""
        if not self.lidar:
            return None

        range_image = self.lidar.getRangeImage()
        if range_image is None:
            return None

        return LidarProcessor.webots_lidar_to_numpy(range_image)

    def _update_context(self, cube, sectors) -> None:
        """Update state machine context with sensor data."""
        ctx = self.state_machine.context

        # Cube detection
        if cube:
            ctx.cube_detected = True
            ctx.cube_color = cube.color
            ctx.cube_distance = cube.distance
            ctx.cube_angle = cube.angle
            self._steps_without_cube = 0
        else:
            self._steps_without_cube += 1
            if self._steps_without_cube > 30:  # ~0.5 seconds
                ctx.cube_detected = False

        # Gripper status
        ctx.has_object = self.gripper.has_object()

        # Obstacle detection from sectors
        if sectors:
            sector_dists = [s.min_distance for s in sectors]
            ctx.front_clear = sector_dists[4] > NAVIGATION.SAFE_DISTANCE if len(sector_dists) > 4 else True
            ctx.nearest_obstacle = min(sector_dists) if sector_dists else float('inf')

    # State handlers

    def _handle_searching(self, ctx) -> RobotState:
        """Search for cubes by rotating."""
        if ctx.cube_detected:
            return RobotState.APPROACHING

        # Get LIDAR for obstacle avoidance while searching
        lidar_data = self._get_lidar_data()
        if lidar_data is not None:
            sector_dists = self.lidar_processor.get_sector_distances(lidar_data)
            nav = self.fuzzy_navigator.compute_from_sectors(sector_dists.tolist())

            if nav.linear_velocity < 0:  # Obstacle too close
                self.base.move(nav.linear_velocity, 0, nav.angular_velocity)
                return None

        # Search rotation
        self.base.move(0.05, 0, self._search_direction * NAVIGATION.SEARCH_OMEGA)

        # Change direction periodically
        if ctx.steps_in_state % 200 == 0:
            self._search_direction *= -1

        return None

    def _handle_approaching(self, ctx) -> RobotState:
        """Approach detected cube."""
        if not ctx.cube_detected:
            return RobotState.SEARCHING

        # Use fuzzy manipulator for approach
        manip = self.fuzzy_manipulator.compute(ctx.cube_distance, ctx.cube_angle)

        if manip.action == ManipulationAction.GRASP:
            return RobotState.ALIGNING

        # Get LIDAR for obstacle awareness
        lidar_data = self._get_lidar_data()
        if lidar_data is not None:
            sector_dists = self.lidar_processor.get_sector_distances(lidar_data)
            nav = self.fuzzy_navigator.compute_from_sectors(
                sector_dists.tolist(), ctx.cube_angle
            )

            # Blend navigation and manipulation
            if nav.linear_velocity < 0:
                self.base.move(nav.linear_velocity, 0, nav.angular_velocity)
            else:
                self.base.move(
                    manip.approach_speed,
                    0,
                    manip.correction_speed
                )
        else:
            self.base.move(manip.approach_speed, 0, manip.correction_speed)

        return None

    def _handle_aligning(self, ctx) -> RobotState:
        """Fine alignment before grasp."""
        if not ctx.cube_detected:
            return RobotState.SEARCHING

        manip = self.fuzzy_manipulator.compute(ctx.cube_distance, ctx.cube_angle)

        if manip.action == ManipulationAction.GRASP:
            return RobotState.GRASPING

        # Fine corrections only
        self.base.move(0, 0, manip.correction_speed * 0.5)

        # Timeout in alignment
        if ctx.steps_in_state > GRASP.MAX_ALIGN_ATTEMPTS:
            return RobotState.GRASPING

        return None

    def _on_enter_grasping(self, ctx) -> None:
        """Prepare for grasp."""
        self._grasp_step = 0
        self.base.stop()
        self.arm.prepare_for_grasp()

    def _handle_grasping(self, ctx) -> RobotState:
        """Execute grasp sequence."""
        self._grasp_step += 1

        if self._grasp_step < GRASP.WAIT_STEPS_ARM:
            # Wait for arm to position
            return None

        if self._grasp_step == GRASP.WAIT_STEPS_ARM:
            # Close gripper
            self.gripper.grip()
            return None

        if self._grasp_step < GRASP.WAIT_STEPS_ARM + GRASP.WAIT_STEPS_GRIPPER:
            # Wait for gripper
            return None

        # Check if object was grasped
        if self.gripper.has_object():
            self._deposit_target = DEPOSIT_BOXES.get(ctx.cube_color)
            self.arm.set_height(ArmHeight.RESET)  # Lift arm
            return RobotState.TRANSPORTING
        else:
            # Grasp failed
            return RobotState.RECOVERING

    def _handle_transporting(self, ctx) -> RobotState:
        """Transport to deposit location."""
        if not ctx.has_object:
            return RobotState.SEARCHING

        # TODO: Implement navigation to deposit box
        # For now, simple rotation to find box area

        lidar_data = self._get_lidar_data()
        if lidar_data is not None:
            sector_dists = self.lidar_processor.get_sector_distances(lidar_data)
            nav = self.fuzzy_navigator.compute_from_sectors(sector_dists.tolist())
            self.base.move(nav.linear_velocity, 0, nav.angular_velocity)
        else:
            self.base.move(0.1, 0, 0)

        # Simplified: after some time, deposit
        if ctx.steps_in_state > 300:
            return RobotState.DEPOSITING

        return None

    def _on_enter_depositing(self, ctx) -> None:
        """Prepare for deposit."""
        self.base.stop()
        self.arm.prepare_for_deposit()

    def _handle_depositing(self, ctx) -> RobotState:
        """Deposit cube."""
        if ctx.steps_in_state < GRASP.WAIT_STEPS_ARM:
            return None

        if ctx.steps_in_state == GRASP.WAIT_STEPS_ARM:
            self.gripper.release()
            return None

        if ctx.steps_in_state < GRASP.WAIT_STEPS_ARM + GRASP.WAIT_STEPS_GRIPPER:
            return None

        # Reset arm
        self.arm.reset()

        # Signal complete
        self.state_machine.deposit_complete()
        return RobotState.SEARCHING

    def _handle_recovering(self, ctx) -> RobotState:
        """Recover from failed grasp."""
        # Back up a bit
        if ctx.steps_in_state < 30:
            self.base.move(-0.1, 0, 0)
            return None

        # Open gripper and reset arm
        self.gripper.release()
        self.arm.prepare_for_grasp()

        if ctx.steps_in_state > 60:
            return RobotState.SEARCHING

        return None


def main():
    """Entry point for Webots controller."""
    robot = Robot()
    controller = YouBotController(robot)
    controller.run()


if __name__ == "__main__":
    main()
