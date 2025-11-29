from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper

class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.time_step)
        
        
    def run(self):
        """
        Main control loop - integrates with autonomous MainController

        DECISÃO 021: Controller Integration Strategy
        - Imports src/main_controller.py via sys.path manipulation
        - Uses factory function create_controller_for_webots()
        - Fallback to heuristic mode if models missing
        """
        import sys
        from pathlib import Path

        # Add src/ to Python path for imports
        src_path = Path(__file__).resolve().parent.parent.parent.parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        try:
            from main_controller import MainController
            from perception import PerceptionSystem
            from control.fuzzy_controller import FuzzyController
            from control.state_machine import StateMachine, RobotState
            from navigation import LocalMap, Odometry
            from manipulation import GraspController, DepositController

            print("[youbot] Initializing MainController...")

            # Create MainController with existing robot components
            controller = MainController(
                robot=self.robot,
                base=self.base,
                arm=self.arm,
                gripper=self.gripper,
                lidar_model_path=None,  # Heuristic mode
                camera_model_path=None,  # HSV fallback
                log_enabled=True
            )

            controller.initialize()
            print("[youbot] MainController ready - starting autonomous operation")
            controller.run()

        except ImportError as e:
            print(f"[youbot] ERROR: Failed to import MainController: {e}")
            print("[youbot] Falling back to basic sensor test mode...")
            self._run_sensor_test()
        except Exception as e:
            print(f"[youbot] ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _run_sensor_test(self):
        """Fallback mode - basic sensor test without autonomous control"""
        import numpy as np

        print("[youbot] Running basic sensor test...")
        step_count = 0

        while self.robot.step(self.time_step) != -1:
            step_count += 1

            if step_count % 100 == 0:
                # Read LIDAR
                lidar_data = self.lidar.getRangeImage()
                min_dist = min(lidar_data) if lidar_data else float('inf')

                # Read Camera
                camera_img = self.camera.getImage()
                img_valid = camera_img is not None

                print(f"[youbot] Step {step_count}: LIDAR min={min_dist:.2f}m, Camera={'OK' if img_valid else 'FAIL'}")

if __name__ == "__main__":
    import sys

    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n[youbot] Running Phase 1.2-1.4 Validation Tests")
        from test_controller import main as run_tests
        run_tests()
    else:
        # Normal operation - autonomous controller
        print("\n[youbot] Starting Autonomous YouBot Controller")
        print("[youbot] MATA64 Final Project - Cube Collection Task")
        print("[youbot] GPS Disabled - Odometry Navigation Only\n")

        controller = YouBotController()
        controller.run()