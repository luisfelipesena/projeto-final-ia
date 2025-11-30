"""
YouBot MCP Controller - Webots controller with MCP file-based communication

MATA64 Final Project: Autonomous Cube Collection
Extends MainControllerV2 with MCP command/status IPC.

This controller:
1. Reads commands from commands.json (written by MCP server)
2. Executes commands on robot hardware
3. Writes status to status.json (read by MCP server)
4. Saves camera/LIDAR data to files for MCP access

Usage:
    Set as Webots controller for YouBot robot node.
"""

import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'IA_20252' / 'controllers' / 'youbot'))

from controller import Robot

# Import Webots controllers
from base import Base
from arm import Arm, ArmHeight, ArmOrientation
from gripper import Gripper

# Import services
from services.movement_service import MovementService
from services.arm_service import ArmService
from services.vision_service import VisionService
from services.navigation_service import NavigationService

# Import perception
from perception.cube_detector import CubeDetector

# MCP communication paths
MCP_DIR = PROJECT_ROOT / "youbot_mcp"
DATA_DIR = MCP_DIR / "data" / "youbot"
COMMANDS_FILE = DATA_DIR / "commands.json"
STATUS_FILE = DATA_DIR / "status.json"
CAMERA_IMAGE_FILE = DATA_DIR / "camera_image.jpg"
LIDAR_DATA_FILE = DATA_DIR / "lidar_data.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)


class MCPState(Enum):
    """Controller states for MCP mode"""
    IDLE = auto()
    SEARCHING = auto()
    APPROACHING = auto()
    GRASPING = auto()
    DEPOSITING = auto()
    AVOIDING = auto()
    EXECUTING_COMMAND = auto()


DEPOSIT_BOXES = {
    'green': (0.48, 1.58),
    'blue': (0.48, -1.62),
    'red': (2.31, 0.01),
}


class YouBotMCPController:
    """
    YouBot controller with MCP file-based communication.

    Combines hardware control with MCP IPC for external tool access.
    """

    def __init__(self):
        print("[MCP Controller] Initializing...")

        # Webots setup
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Hardware controllers
        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        # LIDAR
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            print("[MCP Controller] LIDAR enabled")

        # Services
        self.movement = MovementService(self.base, self.robot, self.time_step)
        self.arm_svc = ArmService(self.arm, self.gripper, self.robot, self.time_step)
        self.detector = CubeDetector()
        self.vision = VisionService(self.detector, self.time_step)
        self.navigation = NavigationService(
            self.movement, self.vision, self.robot, self.camera, self.time_step,
            lidar=self.lidar
        )

        # State
        self.state = MCPState.IDLE
        self.autonomous_mode = False
        self.target_cube_color: Optional[str] = None
        self.last_command_id = 0
        self.cubes_collected = 0
        self.grasp_started = False  # Flag to prevent multiple grasp attempts

        # Last detections cache
        self.last_detections = []
        self.current_target = None

        print("[MCP Controller] Initialization complete")

    def _height_to_string(self, height: int) -> str:
        """Convert arm height enum value to string name."""
        names = ["FRONT_FLOOR", "FRONT_PLATE", "FRONT_CARDBOARD_BOX", "RESET",
                 "BACK_PLATE_HIGH", "BACK_PLATE_LOW", "HANOI_PREPARE"]
        return names[height] if 0 <= height < len(names) else "UNKNOWN"

    def _orientation_to_string(self, orientation: int) -> str:
        """Convert arm orientation enum value to string name."""
        names = ["BACK_LEFT", "LEFT", "FRONT_LEFT", "FRONT",
                 "FRONT_RIGHT", "RIGHT", "BACK_RIGHT"]
        return names[orientation] if 0 <= orientation < len(names) else "UNKNOWN"

    def _step(self) -> bool:
        """Execute one simulation step."""
        return self.robot.step(self.time_step) != -1

    def _get_camera_image(self) -> Optional[np.ndarray]:
        """Get current camera image as numpy array."""
        image = self.camera.getImage()
        if image:
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            return image_array[:, :, :3]  # RGB only
        return None

    def _update_vision(self) -> None:
        """Update vision service with current frame."""
        image = self._get_camera_image()
        if image is not None:
            self.vision.update(image)
            # Cache detections - use detector directly since VisionService only tracks single target
            self.last_detections = self.detector.detect(image)
            self.current_target = self.vision.get_target()

    def _get_lidar_data(self) -> Dict[str, Any]:
        """Get processed LIDAR data."""
        if not self.lidar:
            return {"enabled": False}

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return {"enabled": True, "ranges": []}

        valid = [r for r in ranges if 0.01 < r < 10.0]
        min_dist = min(valid) if valid else float('inf')

        # Sector analysis (9 sectors)
        num_sectors = 9
        points_per_sector = len(ranges) // num_sectors
        sectors = {}
        sector_names = ['far_left', 'left', 'front_left', 'front', 'front_right', 'right', 'far_right', 'back_right', 'back_left']

        for i, name in enumerate(sector_names[:num_sectors]):
            start = i * points_per_sector
            end = start + points_per_sector
            sector_ranges = ranges[start:end]
            sector_valid = [r for r in sector_ranges if 0.01 < r < 10.0]
            sectors[name] = {
                "min": min(sector_valid) if sector_valid else float('inf'),
                "avg": sum(sector_valid) / len(sector_valid) if sector_valid else float('inf'),
                "obstacle": min(sector_valid) < 0.5 if sector_valid else False
            }

        return {
            "enabled": True,
            "min_distance": min_dist,
            "sectors": sectors,
            "point_count": len(ranges)
        }

    def _read_command(self) -> Optional[Dict]:
        """Read pending command from commands.json."""
        try:
            if COMMANDS_FILE.exists():
                with open(COMMANDS_FILE, 'r', encoding='utf-8') as f:
                    cmd = json.load(f)

                cmd_id = cmd.get("id", 0)
                if cmd_id > self.last_command_id:
                    self.last_command_id = cmd_id
                    return cmd
        except Exception as e:
            print(f"[MCP] Error reading command: {e}")
        return None

    def _write_status(self) -> None:
        """Write current status to status.json."""
        try:
            lidar_data = self._get_lidar_data()

            # Serialize detections
            detection_list = []
            for det in self.last_detections[:5]:  # Top 5
                detection_list.append({
                    "color": det.color,
                    "confidence": det.confidence,
                    "distance": det.distance,
                    "angle": det.angle,
                    "bbox": det.bbox
                })

            target_info = None
            if self.current_target:
                target_info = {
                    "color": self.current_target.color,
                    "distance": self.current_target.distance,
                    "angle": self.current_target.angle
                }

            status = {
                "last_update": time.time(),
                "current_state": self.state.name,
                "autonomous_mode": self.autonomous_mode,
                "base_velocity": {
                    "vx": self.base.vx,
                    "vy": self.base.vy,
                    "omega": self.base.omega
                },
                "arm_height": self._height_to_string(self.arm.current_height),
                "arm_orientation": self._orientation_to_string(self.arm.current_orientation),
                "gripper_state": "closed" if self.gripper.is_gripping else "open",
                "gripper_has_object": self.gripper.has_object() if hasattr(self.gripper, 'has_object') else False,
                "cubes_collected": self.cubes_collected,
                "cube_detections": detection_list,
                "current_target": target_info,
                "min_obstacle_distance": lidar_data.get("min_distance", float('inf')),
                "obstacle_sectors": lidar_data.get("sectors", {}),
                "lidar_enabled": lidar_data.get("enabled", False)
            }

            with open(STATUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, default=str)

        except Exception as e:
            print(f"[MCP] Error writing status: {e}")

    def _execute_command(self, cmd: Dict) -> None:
        """Execute a command from MCP server."""
        action = cmd.get("action", "")
        params = cmd.get("params", {})

        print(f"[MCP] Executing: {action}")

        if action == "move_base":
            vx = params.get("vx", 0)
            vy = params.get("vy", 0)
            omega = params.get("omega", 0)
            self.base.move(vx, vy, omega)

        elif action == "stop_base":
            self.base.reset()

        elif action == "move_forward":
            distance = params.get("distance_m", 0.1)
            speed = params.get("speed", 0.1)
            self.movement.forward(distance_m=distance, speed=speed)

        elif action == "rotate":
            angle = params.get("angle_deg", 90)
            speed = params.get("speed", 0.5)
            self.movement.turn(angle_deg=angle, speed=speed)

        elif action == "set_arm_height":
            height_str = params.get("height", "RESET")
            height_map = {
                "FRONT_FLOOR": ArmHeight.FRONT_FLOOR,
                "FRONT_PLATE": ArmHeight.FRONT_PLATE,
                "FRONT_CARDBOARD_BOX": ArmHeight.FRONT_CARDBOARD_BOX,
                "RESET": ArmHeight.RESET,
                "BACK_PLATE_HIGH": ArmHeight.BACK_PLATE_HIGH,
                "BACK_PLATE_LOW": ArmHeight.BACK_PLATE_LOW,
                "HANOI_PREPARE": ArmHeight.HANOI_PREPARE
            }
            if height_str in height_map:
                self.arm.set_height(height_map[height_str])

        elif action == "set_arm_orientation":
            orient_str = params.get("orientation", "FRONT")
            orient_map = {
                "BACK_LEFT": ArmOrientation.BACK_LEFT,
                "LEFT": ArmOrientation.LEFT,
                "FRONT_LEFT": ArmOrientation.FRONT_LEFT,
                "FRONT": ArmOrientation.FRONT,
                "FRONT_RIGHT": ArmOrientation.FRONT_RIGHT,
                "RIGHT": ArmOrientation.RIGHT,
                "BACK_RIGHT": ArmOrientation.BACK_RIGHT
            }
            if orient_str in orient_map:
                self.arm.set_orientation(orient_map[orient_str])

        elif action == "set_arm_ik":
            x = params.get("x", 0)
            y = params.get("y", 0.3)
            z = params.get("z", 0.1)
            self.arm.inverse_kinematics(x, y, z)

        elif action == "reset_arm":
            self.arm.reset()

        elif action == "grip":
            self.gripper.grip()

        elif action == "release":
            self.gripper.release()

        elif action == "set_gripper_gap":
            gap = params.get("gap", 0.025)
            self.gripper.set_gap(gap)

        elif action == "capture_camera":
            self._save_camera_image()

        elif action == "detect_cubes":
            self._update_vision()

        elif action == "get_lidar":
            lidar_data = self._get_lidar_data()
            with open(LIDAR_DATA_FILE, 'w') as f:
                json.dump(lidar_data, f, indent=2, default=str)

        elif action == "grasp_sequence":
            self._execute_grasp_sequence()

        elif action == "deposit_cube":
            color = params.get("color", "green")
            self._execute_deposit(color)

        elif action == "set_state":
            state_str = params.get("state", "IDLE")
            state_map = {
                "IDLE": MCPState.IDLE,
                "SEARCHING": MCPState.SEARCHING,
                "APPROACHING": MCPState.APPROACHING,
                "GRASPING": MCPState.GRASPING,
                "DEPOSITING": MCPState.DEPOSITING,
                "AVOIDING": MCPState.AVOIDING
            }
            if state_str in state_map:
                self.state = state_map[state_str]

        elif action == "start_autonomous":
            self.autonomous_mode = True
            self.state = MCPState.SEARCHING

        elif action == "stop_autonomous":
            self.autonomous_mode = False
            self.state = MCPState.IDLE
            self.base.reset()

        else:
            print(f"[MCP] Unknown action: {action}")

    def _save_camera_image(self) -> None:
        """Save current camera image to file."""
        try:
            import cv2
            image = self._get_camera_image()
            if image is not None:
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(CAMERA_IMAGE_FILE), bgr)
                print(f"[MCP] Camera image saved")
        except Exception as e:
            print(f"[MCP] Error saving image: {e}")

    def _execute_grasp_sequence(self) -> None:
        """Execute complete grasp sequence (includes final approach)."""
        target_info = ""
        if self.current_target:
            target_info = f" (target: {self.current_target.color} at {self.current_target.distance:.2f}m)"
        print(f"[MCP] Starting grasp sequence{target_info}")

        # Step 0: Final approach - same as main_controller_v2
        # 12cm forward brings cube within gripper reach
        print("[MCP] Step 0: Final approach (12cm forward)")
        self.movement.forward(distance_m=0.12, speed=0.06)

        print("[MCP] Step 1: Preparing grasp (arm to FRONT_PLATE, gripper open)")
        if not self.arm_svc.prepare_grasp():
            print("[MCP] Grasp prepare failed")
            return

        print("[MCP] Step 2: Executing grasp (lower arm, close gripper, lift)")
        result = self.arm_svc.execute_grasp()

        if result.success:
            self.cubes_collected += 1
            if self.current_target:
                self.target_cube_color = self.current_target.color
            print(f"[MCP] *** GRASP SUCCESS! *** Cubes collected: {self.cubes_collected}/15")
        else:
            print(f"[MCP] Grasp FAILED: {result.error if result.error else 'cube not captured'}")
            print("[MCP] Releasing gripper and resetting arm for retry")
            self.gripper.release()  # Release gripper so we can retry
            self.arm_svc.reset()

    def _execute_deposit(self, color: str) -> None:
        """Execute deposit sequence for given color."""
        print(f"[MCP] Depositing to {color} box")

        box_pos = DEPOSIT_BOXES.get(color)
        if not box_pos:
            print(f"[MCP] Unknown box color: {color}")
            return

        # Simple timed deposit (could be improved with navigation)
        # Turn 180 degrees
        self.movement.turn(angle_deg=180, speed=0.8)

        # Drive forward
        self.movement.forward(distance_m=0.5, speed=0.15)

        # Execute deposit
        self.arm_svc.prepare_deposit()
        self.arm_svc.execute_deposit()
        self.arm_svc.return_to_rest()

        self.target_cube_color = None
        print("[MCP] Deposit complete")

    def _run_autonomous_step(self) -> None:
        """Run one step of autonomous cube collection."""
        if self.state == MCPState.SEARCHING:
            target = self.vision.get_target()
            if target and target.is_reliable:
                self.target_cube_color = target.color
                self.vision.lock_color(target.color)
                self.state = MCPState.APPROACHING
            else:
                # Rotate and search
                self.base.move(0.05, 0, 0.3)

        elif self.state == MCPState.APPROACHING:
            result = self.navigation.approach_target()
            if result.success:
                self.base.reset()
                print(f"[MCP] Approach complete at {result.final_distance:.2f}m, {result.final_angle:.1f}° → GRASPING")
                self.state = MCPState.GRASPING
            elif result.phase.name == "LOST":
                print("[MCP] Target lost during approach → SEARCHING")
                self.state = MCPState.SEARCHING

        elif self.state == MCPState.GRASPING:
            # Only execute grasp sequence ONCE per GRASPING state entry
            if not self.grasp_started:
                self.grasp_started = True
                log_file = DATA_DIR / "grasp_log.txt"

                def log(msg):
                    print(msg)
                    with open(log_file, 'a') as f:
                        f.write(f"{time.time()}: {msg}\n")

                log(f"[MCP] GRASPING: Attempting grasp of {self.target_cube_color} cube")
                log(f"[MCP] GRASPING: Current arm height = {self._height_to_string(self.arm.current_height)}")

                try:
                    import math
                    target = self.vision.get_target()

                    # Step 0: Calculate grasp parameters based on current target
                    if target:
                        log(f"[MCP] GRASPING: Target at {target.distance:.3f}m, {target.angle:.1f}°")

                        # Lateral correction if needed
                        lateral_offset = math.tan(math.radians(target.angle)) * target.distance
                        if abs(lateral_offset) > 0.015:  # >1.5cm offset
                            log(f"[MCP] GRASPING: Step 0a - Lateral correction {lateral_offset*100:.1f}cm")
                            self.movement.strafe(lateral_offset, speed=0.04)

                    # Approach gets us to 22cm. With 15cm camera offset = 37cm from arm base
                    # FRONT_FLOOR reaches ~31cm, so need 6cm forward to close the gap
                    forward_move = 0.06
                    log(f"[MCP] GRASPING: Step 1 - Forward approach ({forward_move*100:.0f}cm)")
                    self.movement.forward(distance_m=forward_move, speed=0.04)

                    # 2. Prepare grasp (opens gripper, moves arm to FRONT_PLATE)
                    log("[MCP] GRASPING: Step 2 - Prepare grasp")
                    prep_result = self.arm_svc.prepare_grasp()
                    log(f"[MCP] GRASPING: Prepare result = {prep_result}, arm = {self._height_to_string(self.arm.current_height)}")

                    if not prep_result:
                        log("[MCP] GRASPING: Prepare failed → SEARCHING")
                        self.vision.unlock()
                        self.state = MCPState.SEARCHING
                        self.grasp_started = False
                        return
                    log("[MCP] GRASPING: Prepare complete")

                    log("[MCP] GRASPING: Step 3 - Execute grasp (FRONT_FLOOR preset)")
                    result = self.arm_svc.execute_grasp(use_ik=False)
                    log(f"[MCP] GRASPING: Execute complete - success={result.success}, has_object={result.has_object}, error={result.error}")
                    log(f"[MCP] GRASPING: Arm after execute = {self._height_to_string(self.arm.current_height)}")

                    if result.success:
                        self.cubes_collected += 1
                        log(f"[MCP] GRASPING: SUCCESS! Total: {self.cubes_collected}")
                        self.state = MCPState.DEPOSITING
                    else:
                        log(f"[MCP] GRASPING: FAILED: {result.error}")
                        self.arm_svc.reset()
                        self.vision.unlock()
                        self.state = MCPState.SEARCHING

                except Exception as e:
                    log(f"[MCP] GRASPING: EXCEPTION: {e}")
                    import traceback
                    traceback.print_exc()
                    self.arm_svc.reset()
                    self.vision.unlock()
                    self.state = MCPState.SEARCHING

                # Reset flag when leaving GRASPING
                self.grasp_started = False
                log(f"[MCP] GRASPING: Complete, new state = {self.state.name}")

        elif self.state == MCPState.DEPOSITING:
            if self.target_cube_color:
                self._execute_deposit(self.target_cube_color)
            self.vision.unlock()
            self.state = MCPState.SEARCHING

    def run(self) -> None:
        """Main control loop."""
        print("[MCP Controller] Starting main loop")

        # Warmup
        for _ in range(10):
            if not self._step():
                return

        self.arm_svc.reset()

        while self._step():
            # Update vision
            self._update_vision()

            # Process MCP commands
            cmd = self._read_command()
            if cmd:
                self._execute_command(cmd)

            # Run autonomous if enabled
            if self.autonomous_mode:
                self._run_autonomous_step()

            # Always write status
            self._write_status()

            # Check completion
            if self.cubes_collected >= 15:
                print(f"[MCP Controller] TASK COMPLETE: {self.cubes_collected} cubes!")
                self.autonomous_mode = False
                self.state = MCPState.IDLE

        print("[MCP Controller] Simulation ended")


def main():
    controller = YouBotMCPController()
    controller.run()


if __name__ == "__main__":
    main()
