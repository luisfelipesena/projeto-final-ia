
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/Users/luisfelipesena/Development/Personal/projeto-final-ia")
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'IA_20252' / 'controllers' / 'youbot'))

print("Verifying imports...")

try:
    from actuators.base_controller import BaseController
    print("BaseController imported successfully")
except ImportError as e:
    print(f"Failed to import BaseController: {e}")

try:
    from actuators.arm_controller import ArmController
    print("ArmController imported successfully")
except ImportError as e:
    print(f"Failed to import ArmController: {e}")

try:
    from actuators.gripper_controller import GripperController
    print("GripperController imported successfully")
except ImportError as e:
    print(f"Failed to import GripperController: {e}")

try:
    from perception.cube_detector import CubeDetector
    print("CubeDetector imported successfully")
except ImportError as e:
    print(f"Failed to import CubeDetector: {e}")

try:
    from utils.config import GRIPPER
    print(f"Config imported. GRIP_THRESHOLD: {GRIPPER.GRIP_THRESHOLD}")
except ImportError as e:
    print(f"Failed to import Config: {e}")

print("\nVerifying Services...")
try:
    from services.movement_service import MovementService
    print("MovementService imported successfully")
except ImportError as e:
    print(f"Failed to import MovementService: {e}")

try:
    from services.arm_service import ArmService
    print("ArmService imported successfully")
except ImportError as e:
    print(f"Failed to import ArmService: {e}")

try:
    from services.vision_service import VisionService
    print("VisionService imported successfully")
except ImportError as e:
    print(f"Failed to import VisionService: {e}")

try:
    from services.navigation_service import NavigationService
    print("NavigationService imported successfully")
except ImportError as e:
    print(f"Failed to import NavigationService: {e}")

print("\nVerifying Controller Syntax...")
try:
    # We can't instantiate the controller because it requires Webots Robot instance
    # But we can import the module to check for syntax errors
    import youbot_mcp.youbot_mcp_controller
    print("youbot_mcp_controller module imported successfully")
except ImportError as e:
    print(f"Failed to import youbot_mcp_controller: {e}")
except Exception as e:
    print(f"Error importing youbot_mcp_controller: {e}")

print("\nVerification Complete.")
