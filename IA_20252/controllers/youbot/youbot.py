"""
YouBot Controller - Main entry point for Webots simulation

MATA64 Final Project: Autonomous Cube Collection
DECISAO 028: Modular services architecture with main_controller_v2

Usage in Webots:
    - Set controller to "youbot" in robot node
    - Simulation will use MainControllerV2 with RNA + Fuzzy control
"""

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

        self.lidar = self.robot.getDevice("Hokuyo URG-04LX-UG01")
        if self.lidar:
            self.lidar.enable(self.time_step)

    def run(self):
        """
        Main control loop - uses MainControllerV2 (DECISAO 028)

        Features:
        - RNA for LIDAR obstacle detection (MATA64 requirement)
        - Modular services (movement, arm, vision, navigation)
        - Comprehensive logging for debugging
        """
        import sys
        from pathlib import Path

        # Add src/ to Python path for imports
        src_path = Path(__file__).resolve().parent.parent.parent.parent / 'src'
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Also add controllers path for base/arm/gripper
        controller_path = Path(__file__).resolve().parent
        if str(controller_path) not in sys.path:
            sys.path.insert(0, str(controller_path))

        try:
            # Use V2 controller with modular services
            from main_controller_v2 import MainControllerV2

            print("[youbot] Starting MainControllerV2 (DECISAO 028)")
            print("[youbot] RNA + Fuzzy Control enabled")
            print("[youbot] GPS Disabled - Odometry Navigation Only\n")

            controller = MainControllerV2()
            controller.run()

        except ImportError as e:
            print(f"[youbot] ERROR: Failed to import MainControllerV2: {e}")
            print("[youbot] Falling back to basic sensor test mode...")
            import traceback
            traceback.print_exc()
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
                if self.lidar:
                    lidar_data = self.lidar.getRangeImage()
                    min_dist = min(lidar_data) if lidar_data else float('inf')
                else:
                    min_dist = float('inf')

                # Read Camera
                camera_img = self.camera.getImage()
                img_valid = camera_img is not None

                print(f"[youbot] Step {step_count}: LIDAR min={min_dist:.2f}m, "
                      f"Camera={'OK' if img_valid else 'FAIL'}")


if __name__ == "__main__":
    import sys

    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n[youbot] Running service tests...")
        from service_tests import run_all_tests
        run_all_tests()
    else:
        # Normal operation - autonomous controller
        print("\n[youbot] Starting Autonomous YouBot Controller")
        print("[youbot] MATA64 Final Project - Cube Collection Task")

        controller = YouBotController()
        controller.run()
