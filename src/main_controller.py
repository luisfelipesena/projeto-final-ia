"""
Main Controller Integration

Autonomous YouBot controller integrating all AI components:
- Perception: LIDAR + Camera neural networks
- Control: Fuzzy logic inference
- Navigation: Local mapping + Odometry
- Manipulation: Grasping + Depositing

Based on: MATA64 Final Project Requirements
NO GPS ALLOWED in final demonstration.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import time
import logging

# Add src to path for imports
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules
from perception import PerceptionSystem, PerceptionState
from control.fuzzy_controller import FuzzyController, FuzzyInputs, FuzzyOutputs
from control.state_machine import StateMachine, RobotState, StateTransitionConditions
from navigation import LocalMap, Odometry, Pose2D, get_deposit_box_pose
from manipulation import GraspController, DepositController


class MainController:
    """
    Autonomous cube collection controller for YouBot

    Integrates perception, control, navigation, and manipulation
    into a unified control loop.

    Main Loop Flow:
    1. Read sensors (LIDAR, Camera)
    2. Update perception system
    3. Update state machine
    4. Run fuzzy inference for current state
    5. Execute actions (move, grasp, deposit)
    6. Update odometry and local map

    Usage:
        from controller import Robot
        from base import Base
        from arm import Arm
        from gripper import Gripper

        robot = Robot()
        controller = MainController(
            robot=robot,
            base=Base(robot),
            arm=Arm(robot),
            gripper=Gripper(robot)
        )
        controller.initialize()
        controller.run()
    """

    def __init__(
        self,
        robot,
        base,
        arm,
        gripper,
        lidar_model_path: Optional[str] = None,
        camera_model_path: Optional[str] = None,
        log_enabled: bool = True
    ):
        """
        Initialize main controller

        Args:
            robot: Webots Robot instance
            base: Base controller
            arm: Arm controller
            gripper: Gripper controller
            lidar_model_path: Path to trained LIDAR model
            camera_model_path: Path to trained camera model
            log_enabled: Enable logging
        """
        self.robot = robot
        self.base = base
        self.arm = arm
        self.gripper = gripper

        self.time_step = int(robot.getBasicTimeStep())

        # Get sensors
        self.camera = robot.getDevice("camera")
        self.lidar = robot.getDevice("lidar")

        # Enable sensors
        if self.camera:
            self.camera.enable(self.time_step)
        if self.lidar:
            self.lidar.enable(self.time_step)

        # Model paths
        self.lidar_model_path = lidar_model_path
        self.camera_model_path = camera_model_path

        # Initialize subsystems (done in initialize())
        self.perception: Optional[PerceptionSystem] = None
        self.fuzzy_controller: Optional[FuzzyController] = None
        self.state_machine: Optional[StateMachine] = None
        self.odometry: Optional[Odometry] = None
        self.local_map: Optional[LocalMap] = None
        self.grasp_controller: Optional[GraspController] = None
        self.deposit_controller: Optional[DepositController] = None

        # Task tracking
        self.cubes_collected = 0
        self.cubes_deposited = {
            'green': 0,
            'blue': 0,
            'red': 0
        }
        self.total_cubes = 15
        self.current_cube_color: Optional[str] = None

        # Logging
        self.log_enabled = log_enabled
        if log_enabled:
            # Use absolute path relative to project root
            project_root = Path(__file__).resolve().parent.parent
            log_dir = project_root / 'logs'
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / 'main_controller.log'

            logging.basicConfig(
                level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s',
                handlers=[
                    logging.FileHandler(str(log_file)),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger('main_controller')
        else:
            self.logger = None

        # Performance metrics
        self.loop_count = 0
        self.start_time = 0.0

        # Camera warmup (skip first N frames to stabilize)
        self.warmup_frames = 10
        self.frame_count = 0

        # GPS for training mode (disable before final demo)
        # NOTE: GPS device doesn't exist on default YouBot - skip if not found
        self.gps = robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.time_step)
            print("[MainController] GPS enabled for training mode")
        else:
            print("[MainController] GPS not found - using odometry only")

    def initialize(self) -> None:
        """
        Initialize all subsystems

        Must be called before run().
        """
        print("Initializing Main Controller...")

        # Initialize perception
        self.perception = PerceptionSystem(
            lidar_model_path=self.lidar_model_path,
            camera_model_path=self.camera_model_path
        )
        print("  Perception system initialized")

        # Initialize fuzzy controller
        self.fuzzy_controller = FuzzyController()
        self.fuzzy_controller.initialize()
        print(f"  Fuzzy controller initialized ({len(self.fuzzy_controller.rules)} rules)")

        # Initialize state machine
        self.state_machine = StateMachine(
            initial_state=RobotState.SEARCHING,
            timeout_seconds=120.0
        )
        print("  State machine initialized")

        # Initialize navigation
        self.odometry = Odometry()
        self.local_map = LocalMap()
        print("  Navigation initialized")

        # Initialize manipulation
        self.grasp_controller = GraspController(self.arm, self.gripper)
        self.deposit_controller = DepositController(self.arm, self.gripper)
        print("  Manipulation initialized")

        self.start_time = time.time()
        print("Main Controller ready!")

    def _log(self, message: str, level: str = "info") -> None:
        """Log message if logging enabled"""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)

    def get_sensor_data(self) -> tuple:
        """
        Read current sensor data

        Returns:
            (lidar_ranges, camera_image)
        """
        lidar_ranges = None
        camera_image = None

        if self.lidar:
            lidar_ranges = np.array(self.lidar.getRangeImage())

        if self.camera:
            # Get camera image
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            image_data = self.camera.getImage()

            if image_data:
                # Convert to numpy RGB
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((height, width, 4))  # BGRA
                camera_image = image[:, :, :3][:, :, ::-1]  # Convert to RGB

        return lidar_ranges, camera_image

    def build_fuzzy_inputs(self, perception_state: PerceptionState) -> FuzzyInputs:
        """
        Build fuzzy controller inputs from perception state

        Args:
            perception_state: Current perception state

        Returns:
            FuzzyInputs for fuzzy controller
        """
        # Obstacle info
        obstacle_info = self.perception.get_obstacle_info()

        # Cube info
        cube_info = self.perception.get_cube_info()

        return FuzzyInputs(
            distance_to_obstacle=obstacle_info['distance_to_obstacle'],
            angle_to_obstacle=obstacle_info['angle_to_obstacle'],
            distance_to_cube=cube_info['distance_to_cube'],
            angle_to_cube=cube_info['angle_to_cube'],
            cube_detected=cube_info['cube_detected'],
            holding_cube=(self.state_machine.current_state == RobotState.NAVIGATING_TO_BOX)
        )

    def build_state_conditions(self, perception_state: PerceptionState) -> StateTransitionConditions:
        """
        Build state machine conditions from perception state

        Args:
            perception_state: Current perception state

        Returns:
            StateTransitionConditions for state machine
        """
        cube_info = self.perception.get_cube_info()

        return StateTransitionConditions(
            cube_detected=cube_info['cube_detected'],
            cube_distance=cube_info['distance_to_cube'],
            cube_angle=cube_info['angle_to_cube'],
            obstacle_distance=perception_state.min_obstacle_distance,
            holding_cube=(self.state_machine.current_state == RobotState.NAVIGATING_TO_BOX),
            at_target_box=self._check_at_target_box(),
            grasp_success=self.grasp_controller.is_success() if self.grasp_controller else False,
            deposit_complete=self.deposit_controller.is_success() if self.deposit_controller else False
        )

    def _check_at_target_box(self) -> bool:
        """
        Check if robot is at target deposit box

        Uses odometry position and known box locations.

        Returns:
            True if at correct box
        """
        if not self.current_cube_color:
            return False

        # Get current position from odometry
        pose = self.odometry.get_pose()

        # Known box positions (from IA_20252.wbt world file)
        box_positions = {
            'green': (0.48, 1.58),
            'blue': (0.48, -1.62),
            'red': (2.31, 0.01)
        }

        target = box_positions.get(self.current_cube_color)
        if not target:
            return False

        # Check if close enough (within 0.3m)
        distance = np.sqrt((pose.x - target[0])**2 + (pose.y - target[1])**2)
        return distance < 0.3

    def _compute_navigation_to_box(self) -> Tuple[float, float]:
        """
        Compute velocities to navigate to target deposit box

        Uses proportional controller for heading and distance.

        Returns:
            (vx, omega) velocities for navigation
        """
        target = get_deposit_box_pose(self.current_cube_color)
        if not target:
            return 0.0, 0.0

        pose = self.odometry.get_pose()

        # Distance and angle to target
        dx = target.x - pose.x
        dy = target.y - pose.y
        distance = np.sqrt(dx**2 + dy**2)
        desired_heading = np.arctan2(dy, dx)
        heading_error = desired_heading - pose.theta

        # Normalize angle to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # P-controller gains
        K_angular = 0.5
        K_linear = 0.3

        # Angular velocity
        omega = K_angular * heading_error
        omega = np.clip(omega, -0.4, 0.4)

        # Only move forward when roughly aligned (< 30 degrees)
        if abs(heading_error) < 0.5:
            vx = K_linear * min(distance, 1.0)
            vx = np.clip(vx, 0.0, 0.2)
        else:
            vx = 0.0  # Turn in place first

        # Debug logging every ~1 second (30 frames)
        if self.loop_count % 30 == 0:
            self._log(
                f"NAV: pose=({pose.x:.2f}, {pose.y:.2f}, θ={np.degrees(pose.theta):.1f}°) "
                f"→ target=({target.x:.2f}, {target.y:.2f}) "
                f"dist={distance:.2f}m heading_err={np.degrees(heading_error):.1f}°"
            )

        return vx, omega

    def execute_control(self, fuzzy_outputs: FuzzyOutputs) -> None:
        """
        Execute control outputs

        Args:
            fuzzy_outputs: Outputs from fuzzy controller
        """
        current_state = self.state_machine.current_state

        # Handle manipulation states
        if current_state == RobotState.GRASPING:
            if not self.grasp_controller.is_done():
                self.grasp_controller.update()
            # Robot should be stationary during grasp
            self.base.reset()
            return

        if current_state == RobotState.DEPOSITING:
            if not self.deposit_controller.is_done():
                self.deposit_controller.update()
            # Robot should be stationary during deposit
            self.base.reset()
            return

        # Handle AVOIDING state with emergency behavior
        if current_state == RobotState.AVOIDING:
            # Emergency: stop forward motion, turn away from obstacle
            obstacle_info = self.perception.get_obstacle_info()
            angle = obstacle_info.get('angle_to_obstacle', 0.0)

            # Turn away from obstacle direction
            if angle >= 0:
                omega = -0.4  # Turn left if obstacle on right
            else:
                omega = 0.4   # Turn right if obstacle on left

            self.base.move(vx=0.0, vy=0.0, omega=omega)
            dt = self.time_step / 1000.0
            self.odometry.update_from_command(vx=0.0, vy=0.0, omega=omega, dt=dt)
            return

        # Handle NAVIGATING_TO_BOX with dedicated navigation controller
        if current_state == RobotState.NAVIGATING_TO_BOX:
            vx, omega = self._compute_navigation_to_box()
            self.base.move(vx=vx, vy=0.0, omega=omega)
            dt = self.time_step / 1000.0
            self.odometry.update_from_command(vx=vx, vy=0.0, omega=omega, dt=dt)
            return

        # Normal movement states
        vx = fuzzy_outputs.linear_velocity
        omega = fuzzy_outputs.angular_velocity

        # Apply movement
        self.base.move(vx=vx, vy=0.0, omega=omega)

        # Update odometry with commanded velocities
        dt = self.time_step / 1000.0
        self.odometry.update_from_command(vx=vx, vy=0.0, omega=omega, dt=dt)

    def handle_state_entry(self, new_state: RobotState) -> None:
        """
        Handle actions when entering a new state

        Args:
            new_state: State being entered
        """
        if new_state == RobotState.GRASPING:
            # Start grasp sequence
            cube_info = self.perception.get_cube_info()
            color = cube_info.get('cube_color', 'unknown')
            self.current_cube_color = color
            # Also set in state machine for navigation tracking
            self.state_machine.set_cube_tracking(f"cube_{self.cubes_collected}", color)
            self.grasp_controller.start(color)
            self._log(f"Starting grasp for {color} cube")

        elif new_state == RobotState.NAVIGATING_TO_BOX:
            # Update cube tracking
            result = self.grasp_controller.get_result()
            if result and result.success:
                self.current_cube_color = result.cube_color
                self.cubes_collected += 1
                self._log(f"Cube collected! Total: {self.cubes_collected}/{self.total_cubes}")

        elif new_state == RobotState.DEPOSITING:
            # Start deposit sequence
            self.deposit_controller.start(self.current_cube_color)
            self._log(f"Starting deposit for {self.current_cube_color} cube")

        elif new_state == RobotState.SEARCHING:
            # Check if deposit was completed
            if self.deposit_controller.is_success():
                result = self.deposit_controller.get_result()
                if result and result.cube_color:
                    self.cubes_deposited[result.cube_color] += 1
                    self._log(f"Deposited {result.cube_color}! Stats: {self.cubes_deposited}")

            # Reset manipulation controllers
            self.grasp_controller.reset()
            self.deposit_controller.reset()
            self.current_cube_color = None

    def step(self) -> bool:
        """
        Execute one control loop iteration

        Returns:
            True if should continue, False if task complete or error
        """
        # Camera warmup - skip first N frames to stabilize
        self.frame_count += 1
        if self.frame_count <= self.warmup_frames:
            if self.frame_count == self.warmup_frames:
                self._log(f"Camera warmup complete ({self.warmup_frames} frames)")
            return True  # Skip processing during warmup

        # Check task completion
        total_deposited = sum(self.cubes_deposited.values())
        if total_deposited >= self.total_cubes:
            self._log(f"TASK COMPLETE! All {self.total_cubes} cubes deposited!")
            return False

        # GPS logging for training mode
        if self.gps and self.loop_count % 30 == 0:  # Log every 30 iterations (~1s)
            pos = self.gps.getValues()
            self._log(f"GPS: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

        # Read sensors
        lidar_ranges, camera_image = self.get_sensor_data()

        # Update perception
        perception_state = self.perception.update(lidar_ranges, camera_image)

        # Update local map
        if lidar_ranges is not None:
            pose = self.odometry.get_pose()
            self.local_map.update_from_lidar(lidar_ranges, robot_angle=pose.theta)

        # Build conditions and update state machine
        previous_state = self.state_machine.current_state
        conditions = self.build_state_conditions(perception_state)

        # Debug logging every ~1s during SEARCHING to understand why not finding cubes
        if previous_state == RobotState.SEARCHING and self.loop_count % 30 == 0:
            cube_info = self.perception.get_cube_info()
            obstacle_info = self.perception.get_obstacle_info()
            self._log(
                f"[SEARCH] cube_detected={cube_info.get('cube_detected')}, "
                f"cube_dist={cube_info.get('distance_to_cube', 'N/A'):.2f}m, "
                f"obstacle_dist={obstacle_info.get('distance_to_obstacle', 'N/A'):.2f}m"
            )

        self.state_machine.update(conditions)

        # Handle state transitions
        if self.state_machine.current_state != previous_state:
            self.handle_state_entry(self.state_machine.current_state)

        # Run fuzzy inference
        fuzzy_inputs = self.build_fuzzy_inputs(perception_state)
        fuzzy_outputs = self.fuzzy_controller.infer(fuzzy_inputs)

        # Execute control
        self.execute_control(fuzzy_outputs)

        self.loop_count += 1

        return True

    def run(self) -> None:
        """
        Main control loop

        Runs until task is complete or Webots stops.
        """
        self._log("Starting main control loop")

        while self.robot.step(self.time_step) != -1:
            if not self.step():
                break

        # Final stats
        elapsed = time.time() - self.start_time
        total_deposited = sum(self.cubes_deposited.values())

        self._log(f"Run complete. Time: {elapsed:.1f}s, Cubes: {total_deposited}/{self.total_cubes}")
        self._log(f"Final stats: {self.cubes_deposited}")

        # Stop robot
        self.base.reset()


def create_controller_for_webots():
    """
    Factory function to create controller for Webots

    Returns:
        MainController instance ready to run

    Usage in Webots:
        from main_controller import create_controller_for_webots
        controller = create_controller_for_webots()
        controller.initialize()
        controller.run()
    """
    # Import Webots controller module
    from controller import Robot
    from base import Base
    from arm import Arm
    from gripper import Gripper

    robot = Robot()
    base = Base(robot)
    arm = Arm(robot)
    gripper = Gripper(robot)

    # Model paths (relative to controller directory)
    lidar_model = "models/lidar_net.pt"
    camera_model = "models/camera_net.pt"

    controller = MainController(
        robot=robot,
        base=base,
        arm=arm,
        gripper=gripper,
        lidar_model_path=lidar_model if Path(lidar_model).exists() else None,
        camera_model_path=camera_model if Path(camera_model).exists() else None
    )

    return controller


if __name__ == "__main__":
    print("Main Controller Module")
    print("To use in Webots:")
    print("  from main_controller import create_controller_for_webots")
    print("  controller = create_controller_for_webots()")
    print("  controller.initialize()")
    print("  controller.run()")
