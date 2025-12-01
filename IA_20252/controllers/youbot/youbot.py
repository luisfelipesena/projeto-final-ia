"""
YouBot Controller - Main entry point for Webots simulation

MATA64 Final Project: Autonomous Cube Collection
DECISAO 028: Modular services architecture with main_controller_v2

Usage in Webots:
    - Set controller to "youbot" in robot node
    - Simulation will use MainControllerV2 with RNA + Fuzzy control
"""

import sys
from pathlib import Path


def main():
    """
    Main entry point - delegates to MainControllerV2.

    IMPORTANT: Only ONE Robot() instance allowed per controller.
    MainControllerV2 creates it internally.
    """
    # Add src/ to Python path for imports
    src_path = Path(__file__).resolve().parent.parent.parent.parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Also add controllers path for base/arm/gripper
    controller_path = Path(__file__).resolve().parent
    if str(controller_path) not in sys.path:
        sys.path.insert(0, str(controller_path))

    print("\n[youbot] Starting Autonomous YouBot Controller")
    print("[youbot] MATA64 Final Project - Cube Collection Task")

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
        _run_sensor_test()
    except Exception as e:
        print(f"[youbot] ERROR: {e}")
        import traceback
        traceback.print_exc()


def _run_sensor_test():
    """Fallback mode - basic sensor test without autonomous control."""
    from controller import Robot

    robot = Robot()
    time_step = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(time_step)

    lidar = robot.getDevice("Lidar")
    if lidar:
        lidar.enable(time_step)

    print("[youbot] Running basic sensor test...")
    step_count = 0

    while robot.step(time_step) != -1:
        step_count += 1

        if step_count % 100 == 0:
            # Read LIDAR
            if lidar:
                lidar_data = lidar.getRangeImage()
                min_dist = min(lidar_data) if lidar_data else float('inf')
            else:
                min_dist = float('inf')

            # Read Camera
            camera_img = camera.getImage()
            img_valid = camera_img is not None

            print(f"[youbot] Step {step_count}: LIDAR min={min_dist:.2f}m, "
                  f"Camera={'OK' if img_valid else 'FAIL'}")


if __name__ == "__main__":
    # Check for test mode argument
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("\n[youbot] Running service tests...")
        from service_tests import run_all_tests
        run_all_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "--mcp":
        # MCP mode - controlled via file-based IPC
        print("\n[youbot] Starting MCP Controller Mode...")
        mcp_path = Path(__file__).resolve().parent.parent.parent.parent / 'youbot_mcp'
        if str(mcp_path) not in sys.path:
            sys.path.insert(0, str(mcp_path))
        from youbot_mcp_controller import YouBotMCPController
        controller = YouBotMCPController()
        controller.run()
    else:
        # Normal operation - autonomous controller
        main()
