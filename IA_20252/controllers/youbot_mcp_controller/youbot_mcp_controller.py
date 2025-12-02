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
import torch
from pathlib import Path
from enum import Enum, auto
from typing import Optional, List, Dict, Any

# Add paths - find project root by looking for src/ directory
_file_path = Path(__file__).resolve()
# When running from youbot_mcp/ -> parent is project root
# When running from IA_20252/controllers/youbot_mcp_controller/ -> need to go up 3 levels
if (_file_path.parent / 'src').exists():
    PROJECT_ROOT = _file_path.parent
elif (_file_path.parent.parent.parent.parent / 'src').exists():
    PROJECT_ROOT = _file_path.parent.parent.parent.parent
else:
    PROJECT_ROOT = _file_path.parent.parent  # fallback
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'IA_20252' / 'controllers' / 'youbot'))

from controller import Robot

# Import Webots controllers
# Import Webots controllers
from actuators.base_controller import BaseController
from actuators.arm_controller import ArmController, ArmHeight, ArmOrientation
from actuators.gripper_controller import GripperController

# Import services
from services.movement_service import MovementService
from services.arm_service import ArmService
from services.vision_service import VisionService
from services.navigation_service import NavigationService

# Import perception
from perception.cube_detector import CubeDetector

# Import Fuzzy Controller (MATA64 requirement)
from control.fuzzy_navigator import FuzzyNavigator, NavigationOutput

# Import RNA model for LIDAR (MATA64 requirement)
from perception.lidar_mlp import LidarMLP

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

    # Class-level version for cache detection
    VERSION = "2025-12-01-V5"

    def __init__(self):
        print(f"[MCP Controller] Initializing... VERSION: {self.VERSION}")

        # Webots setup
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        # Hardware controllers
        # Hardware controllers
        self.base = BaseController(self.robot)
        self.arm = ArmController(self.robot)
        self.gripper = GripperController(self.robot)

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.time_step)

        # LIDAR
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            print("[MCP Controller] LIDAR enabled")

        # Load RNA model for LIDAR (MATA64 requirement)
        self.lidar_model = self._load_lidar_model()

        # Services
        self.movement = MovementService(self.base, self.robot, self.time_step)
        self.arm_svc = ArmService(self.arm, self.gripper, self.robot, self.time_step)
        self.detector = CubeDetector()
        self.vision = VisionService(self.detector, self.time_step)
        self.navigation = NavigationService(
            self.movement, self.vision, self.robot, self.camera, self.time_step,
            lidar=self.lidar
        )

        # Initialize Fuzzy Controller (MATA64 requirement)
        self.fuzzy = FuzzyNavigator()
        print("[MCP Controller] Fuzzy controller initialized")

        # State
        self.state = MCPState.IDLE
        self.autonomous_mode = False
        self.target_cube_color: Optional[str] = None
        self.last_command_id = 0
        self.cubes_collected = 0
        self.grasp_started = False  # Flag to prevent multiple grasp attempts

        # Search pattern state (prevents constant rotation hitting walls)
        self._search_angle_covered = 0.0
        self._search_direction = True  # True=left (positive omega)
        self._search_phase = 'scan'    # 'scan' or 'move'
        self._search_move_time = 0.0

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

    def _load_lidar_model(self) -> Optional[LidarMLP]:
        """Load RNA model for LIDAR obstacle detection (MATA64 requirement)."""
        try:
            model = LidarMLP(input_size=512, num_sectors=9)
            model_path = PROJECT_ROOT / "models" / "lidar_mlp.pth"

            if model.load(str(model_path)):
                return model
            else:
                print(f"[RNA] AVISO: Modelo nao encontrado em {model_path}, usando heuristico")
                return None
        except Exception as e:
            print(f"[RNA] Erro ao carregar modelo: {e}")
            return None

    def _get_obstacle_map_rna(self, ranges: List[float]) -> List[float]:
        """Use RNA to detect obstacles per sector (MATA64 requirement)."""
        if self.lidar_model is None:
            return self._heuristic_obstacle_detection(ranges)

        try:
            # LidarMLP handles normalization internally
            obstacles = self.lidar_model.predict(np.array(ranges))
            return obstacles.tolist()
        except Exception as e:
            print(f"[RNA] Erro na inferencia: {e}")
            return self._heuristic_obstacle_detection(ranges)

    def _heuristic_obstacle_detection(self, ranges: List[float]) -> List[float]:
        """Fallback heuristic obstacle detection."""
        num_sectors = 9
        points_per_sector = len(ranges) // num_sectors if ranges else 1
        obstacles = []

        for i in range(num_sectors):
            start = i * points_per_sector
            end = start + points_per_sector
            sector = ranges[start:end] if ranges else []
            valid = [r for r in sector if 0.01 < r < 5.0]
            if valid:
                min_dist = min(valid)
                obstacles.append(1.0 if min_dist < 0.5 else 0.0)
            else:
                obstacles.append(0.0)

        return obstacles



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

    def _save_grasp_screenshot(self, stage: str) -> None:
        """Save screenshot during grasp for debugging."""
        import cv2
        image = self._get_camera_image()
        if image is not None:
            timestamp = int(time.time())
            filename = DATA_DIR / f"grasp_{stage}_{timestamp}.jpg"
            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filename), bgr)
            print(f"[MCP] Screenshot saved: {filename.name}")

    def _update_vision(self) -> None:
        """Update vision service with current frame."""
        image = self._get_camera_image()
        if image is not None:
            self.vision.update(image)
            # Cache detections - use detector directly since VisionService only tracks single target
            self.last_detections = self.detector.detect(image)
            self.current_target = self.vision.get_target()

            # Debug: log raw detections during approach
            if self.state == MCPState.APPROACHING and self.last_detections:
                from pathlib import Path
                with open(Path(__file__).parent / "data" / "youbot" / "nav_debug.log", 'a') as f:
                    det = self.last_detections[0] if self.last_detections else None
                    if det:
                        f.write(f"RAW_DET: {det.color} angle={det.angle:.1f} dist={det.distance:.2f}\n")
                    else:
                        f.write("RAW_DET: NONE angle=0.0 dist=0.00\n")

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
                "version": self.VERSION,
                "last_update": time.time(),
                "current_state": self.state.name,
                "autonomous_mode": self.autonomous_mode,
                "base_velocity": {
                    "vx": self.base.velocity[0],
                    "vy": self.base.velocity[1],
                    "omega": self.base.velocity[2]
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
            self.base.stop()

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
            self.base.stop()

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
                print("[MCP] Camera image saved")
        except Exception as e:
            print(f"[MCP] Error saving image: {e}")

    def _execute_grasp_sequence(self) -> None:
        """Execute complete grasp sequence (includes final approach)."""
        target_info = ""
        target_distance = 0.22  # Default if no target
        if self.current_target:
            target_distance = self.current_target.distance
            target_info = f" (target: {self.current_target.color} at {target_distance:.2f}m)"
        print(f"[MCP] Starting grasp sequence{target_info}")

        # Step 0: Final approach - 6cm forward (theory says 6cm from 22cm approach)
        forward_move = 0.06
        print(f"[MCP] Step 0: Final approach ({forward_move*100:.0f}cm forward)")
        self.movement.forward(distance_m=forward_move, speed=0.06)

        print("[MCP] Step 1: Preparing grasp (arm to FRONT_PLATE, gripper open)")
        if not self.arm_svc.prepare_grasp():
            print("[MCP] Grasp prepare failed")
            return

        # Calculate forward reach for IK
        # Camera is ~15cm ahead of arm base, so arm needs to reach FURTHER
        CAMERA_ARM_OFFSET = 0.15
        camera_dist = target_distance - forward_move
        forward_reach = camera_dist + CAMERA_ARM_OFFSET
        forward_reach = max(0.18, min(0.32, forward_reach))
        print(f"[MCP] Step 2: Executing grasp with IK (camera_dist={camera_dist:.2f}m + offset={CAMERA_ARM_OFFSET:.2f}m = reach={forward_reach:.2f}m)")
        result = self.arm_svc.execute_grasp(use_ik=True, forward_reach=forward_reach)

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

            # Accept targets with confidence >= 0.07 (matches cube_detector defaults)
            # Lower threshold allows acquiring distant cubes
            if target and target.confidence >= 0.07:
                # CRITICAL: Stop rotation IMMEDIATELY before state change
                self.base.stop()
                self.target_cube_color = target.color
                self.vision.lock_color(target.color)
                # Reset search state for next time
                self._search_angle_covered = 0.0
                self._search_phase = 'scan'
                self.state = MCPState.APPROACHING
                print(f"[MCP Search] Target acquired: {target.color} at {target.distance:.2f}m, {target.angle:.1f}° → APPROACHING")
                return  # Don't do anything else this frame
            else:
                # Get LIDAR data for obstacle avoidance
                lidar_data = self._get_lidar_data()
                sectors = lidar_data.get("sectors", {})
                front_dist = min(
                    sectors.get("front", {}).get("min", float('inf')),
                    sectors.get("front_left", {}).get("min", float('inf')),
                    sectors.get("front_right", {}).get("min", float('inf'))
                )
                left_dist = min(
                    sectors.get("left", {}).get("min", float('inf')),
                    sectors.get("front_left", {}).get("min", float('inf'))
                )
                right_dist = min(
                    sectors.get("right", {}).get("min", float('inf')),
                    sectors.get("front_right", {}).get("min", float('inf'))
                )

                # Use fuzzy controller for obstacle-aware navigation (MATA64 requirement)
                sector_names = ['far_left', 'left', 'front_left', 'front', 'front_right', 'right', 'far_right', 'back_right', 'back_left']
                sector_mins = [sectors.get(name, {}).get("min", float('inf')) for name in sector_names]

                try:
                    outputs = self.fuzzy.compute_from_sectors(sector_mins, target_angle=0.0)
                except Exception:
                    outputs = None

                # Controlled search pattern: scan ~360° then move forward
                if self._search_phase == 'scan':
                    # PURE ROTATION during scan - no forward movement
                    omega = 0.35 if self._search_direction else -0.35
                    self._search_angle_covered += abs(omega) * self.time_step / 1000.0

                    # After ~360° (6.28 rad), switch to move phase
                    if self._search_angle_covered > 6.28:
                        self._search_phase = 'move'
                        self._search_move_time = 0.0
                        self._search_direction = not self._search_direction
                        print("[MCP Search] Full scan complete, moving forward")

                    # Pure rotation - no forward drift
                    self.base.move(0, 0, omega)
                else:
                    # Check obstacles before moving forward
                    if front_dist < 0.40:
                        # Front obstacle - switch to scan phase
                        self.base.stop()
                        self._search_phase = 'scan'
                        self._search_angle_covered = 0.0
                        self._search_direction = not self._search_direction
                        print(f"[MCP Search] Front obstacle at {front_dist:.2f}m, scanning")
                        return

                    # Check side obstacles
                    if left_dist < 0.30:
                        # Veer right
                        self.base.move(0.05, 0, -0.3)
                        return
                    elif right_dist < 0.30:
                        # Veer left
                        self.base.move(0.05, 0, 0.3)
                        return

                    # Move forward for ~1.5 seconds (shorter bursts)
                    self._search_move_time += self.time_step / 1000.0
                    if self._search_move_time > 1.5:
                        self._search_phase = 'scan'
                        self._search_angle_covered = 0.0
                        print("[MCP Search] Move complete, scanning again")

                    # Slower forward speed during search
                    vx = 0.08
                    if outputs and outputs.linear_velocity > 0:
                        vx = min(outputs.linear_velocity, 0.10)
                    self.base.move(vx, 0, 0)

        elif self.state == MCPState.APPROACHING:
            # Non-blocking approach: one step per iteration
            import math
            target = self.vision.get_target()

            if not target:
                # Don't immediately give up - wait for re-acquisition
                if not hasattr(self, '_approach_lost_frames'):
                    self._approach_lost_frames = 0
                    self._last_known_angle = 0.0
                self._approach_lost_frames += 1

                # REDUCED persistence: 45 frames (~1.5s) instead of 90
                if self._approach_lost_frames < 45:
                    # Search in direction of last known position
                    search_omega = 0.30 if self._last_known_angle < 0 else -0.30
                    self.base.move(0, 0, search_omega)  # Pure rotation to find target
                    if self._approach_lost_frames % 15 == 0:
                        print(f"[MCP] Approach: searching for target... ({self._approach_lost_frames}/45)")
                    return

                # Lost - go back to SEARCHING
                print(f"[MCP] Approach: target lost, re-searching")
                self._approach_lost_frames = 0
                self.vision.unlock()
                self.target_cube_color = None
                self.state = MCPState.SEARCHING
                self.base.stop()
                return

            # Reset lost frame counter and remember angle
            self._approach_lost_frames = 0
            self._last_known_angle = target.angle

            # Check for front obstacles during approach
            lidar_data = self._get_lidar_data()
            sectors = lidar_data.get("sectors", {})
            front_dist = min(
                sectors.get("front", {}).get("min", float('inf')),
                sectors.get("front_left", {}).get("min", float('inf')),
                sectors.get("front_right", {}).get("min", float('inf'))
            )

            # Obstacle too close (but not the cube) - stop and return to search
            if front_dist < 0.25 and target.distance > 0.35:
                self.base.stop()
                print(f"[MCP] Approach: obstacle at {front_dist:.2f}m, returning to search")
                self.vision.unlock()
                self.target_cube_color = None
                self.state = MCPState.SEARCHING
                return

            # Check if close enough to grasp
            # Distance <= 0.25m and angle < 10° for reliable grasp
            if target.distance <= 0.25 and abs(target.angle) < 10:
                self.base.stop()
                self._grasp_target_distance = target.distance
                self._grasp_target_angle = target.angle
                print(f"[MCP] Approach complete at {target.distance:.2f}m, {target.angle:.1f}° → GRASPING")
                self.state = MCPState.GRASPING
                return

            # Use fuzzy controller for approach decisions (MATA64 requirement)
            sector_names = ['far_left', 'left', 'front_left', 'front', 'front_right', 'right', 'far_right', 'back_right', 'back_left']
            sector_mins = [sectors.get(name, {}).get("min", float('inf')) for name in sector_names]

            try:
                fuzzy_out = self.fuzzy.compute_from_sectors(sector_mins, target_angle=target.angle)
                fuzzy_vx = fuzzy_out.linear_velocity
            except Exception:
                fuzzy_vx = 0.08

            # Proportional control for alignment and approach
            angle_rad = math.radians(target.angle)

            if abs(target.angle) > 20:
                # Large angle: PURE ROTATION to align
                omega = -0.30 if target.angle > 0 else 0.30
                self.base.move(0, 0, omega)
                return
            elif abs(target.angle) > 10:
                # Medium angle: slow forward + rotation
                omega = -angle_rad * 0.8
                omega = max(-0.35, min(0.35, omega))
                self.base.move(0.03, 0, omega)
                return
            else:
                # Aligned - move forward with minor correction
                omega = -angle_rad * 0.6
                # Slower approach for better tracking
                vx = min(0.08, max(0.04, target.distance * 0.3))
                if fuzzy_vx > 0.03:
                    vx = min(vx, fuzzy_vx)

            self.base.move(vx, 0, omega)

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
                self._save_grasp_screenshot("before")

                try:
                    # Step 1: Short forward approach (8cm instead of 25cm)
                    # Robot is already at ~25cm, need to get to ~17cm for arm reach
                    forward_move = 0.08
                    log(f"[MCP] GRASPING: Step 1 - Forward approach ({forward_move*100:.0f}cm)")
                    self.movement.forward(distance_m=forward_move, speed=0.02)

                    # Wait for robot to settle
                    for _ in range(20):
                        self._step()

                    self._save_grasp_screenshot("after_forward")

                    # Step 2: Open gripper
                    log("[MCP] GRASPING: Step 2 - Opening gripper")
                    self.gripper.release()
                    for _ in range(30):
                        self._step()

                    # Step 3: Lower arm to FRONT_FLOOR preset (no IK)
                    log("[MCP] GRASPING: Step 3 - Lowering arm to FRONT_FLOOR")
                    self.arm.set_height(ArmHeight.FRONT_FLOOR)
                    for _ in range(60):
                        self._step()

                    self._save_grasp_screenshot("arm_lowered")

                    # Step 4: Close gripper
                    log("[MCP] GRASPING: Step 4 - Closing gripper")
                    self.gripper.grip()
                    for _ in range(40):
                        self._step()

                    # Step 5: Check if object was grasped
                    has_object = self.gripper.has_object()
                    finger_pos = self.gripper.get_finger_position()
                    log(f"[MCP] GRASPING: Gripper check - has_object={has_object}, finger_pos={finger_pos}")

                    self._save_grasp_screenshot("after_grip")

                    if has_object:
                        # Step 6: Lift the cube
                        log("[MCP] GRASPING: Step 6 - Lifting cube")
                        self.arm.set_height(ArmHeight.FRONT_PLATE)
                        for _ in range(60):
                            self._step()

                        self.cubes_collected += 1
                        log(f"[MCP] GRASPING: SUCCESS! Total: {self.cubes_collected}")
                        self.state = MCPState.DEPOSITING
                    else:
                        log("[MCP] GRASPING: FAILED - No object detected")
                        # Release and reset
                        self.gripper.release()
                        for _ in range(20):
                            self._step()
                        self.arm.reset()
                        self.vision.unlock()
                        self.state = MCPState.SEARCHING

                except Exception as e:
                    log(f"[MCP] GRASPING: EXCEPTION: {e}")
                    import traceback
                    traceback.print_exc()
                    self.arm.reset()
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
        # Version check - timestamp helps detect cached code
        VERSION_TS = "2025-12-01-V3"
        print(f"[MCP Controller] Starting main loop - VERSION: {VERSION_TS}")
        print(f"[MCP Controller] Build timestamp: {int(time.time())}")

        # Clear old debug logs
        nav_log = DATA_DIR / "nav_debug.log"
        if nav_log.exists():
            nav_log.unlink()

        # Warmup
        for _ in range(10):
            if not self._step():
                return

        self.arm_svc.reset()

        while self._step():
            # Update vision
            self._update_vision()

            # EMERGENCY OBSTACLE AVOIDANCE - Check BEFORE any state processing
            lidar_data = self._get_lidar_data()
            front_dist = lidar_data.get("min_distance", float('inf'))
            sectors = lidar_data.get("sectors", {})

            # Get front sector distances
            front_sector = min(
                sectors.get("front", {}).get("min", float('inf')),
                sectors.get("front_left", {}).get("min", float('inf')),
                sectors.get("front_right", {}).get("min", float('inf'))
            )

            # Emergency stop if too close to obstacle
            if front_sector < 0.20 and self.state != MCPState.GRASPING:
                self.base.stop()
                # Back up slowly
                self.base.move(-0.05, 0, 0)
                self._write_status()
                continue  # Skip state machine this frame

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
