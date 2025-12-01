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

# Import Fuzzy Controller (MATA64 requirement)
from control.fuzzy_controller import FuzzyController, FuzzyInputs

# Import RNA model for LIDAR (MATA64 requirement)
from perception.models.simple_lidar_mlp import SimpleLIDARMLP

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
    VERSION = "2025-12-01-V3"

    def __init__(self):
        print(f"[MCP Controller] Initializing... VERSION: {self.VERSION}")

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
        self.fuzzy = FuzzyController(config={'logging': True})
        try:
            self.fuzzy.initialize()
            print("[MCP Controller] Fuzzy controller initialized")
        except Exception as e:
            print(f"[MCP Controller] WARNING: Fuzzy init failed: {e}")

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

    def _load_lidar_model(self) -> Optional[SimpleLIDARMLP]:
        """Load RNA model for LIDAR obstacle detection (MATA64 requirement)."""
        try:
            model = SimpleLIDARMLP(input_size=512, num_sectors=9)
            model_path = PROJECT_ROOT / "models" / "lidar_mlp.pth"

            if model_path.exists():
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                print(f"[RNA] Modelo LIDAR carregado: {model_path}")
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
            input_tensor = SimpleLIDARMLP.preprocess_lidar(ranges, max_range=5.0, target_size=512)
            with torch.no_grad():
                obstacles = self.lidar_model(input_tensor)
            return obstacles.squeeze().tolist()
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

    def _compute_fuzzy_inputs(self) -> FuzzyInputs:
        """Convert sensor data to FuzzyInputs for controller."""
        # Get LIDAR data
        lidar_data = self._get_lidar_data()
        sectors = lidar_data.get("sectors", {})

        # Calculate obstacle distance and angle
        obstacle_dist = lidar_data.get("min_distance", 5.0)
        if obstacle_dist == float('inf'):
            obstacle_dist = 5.0

        # Compute obstacle angle from sectors
        left_blocked = sectors.get('left', {}).get('obstacle', False) or sectors.get('front_left', {}).get('obstacle', False)
        right_blocked = sectors.get('right', {}).get('obstacle', False) or sectors.get('front_right', {}).get('obstacle', False)
        front_blocked_flag = sectors.get('front', {}).get('obstacle', False)

        if left_blocked and not right_blocked:
            obstacle_angle = -45.0
        elif right_blocked and not left_blocked:
            obstacle_angle = 45.0
        elif front_blocked_flag:
            obstacle_angle = 0.0
        else:
            obstacle_angle = 0.0

        # Get cube data from vision
        target = self.current_target
        if target:
            cube_dist = target.distance
            cube_angle = target.angle
            cube_detected = True
        else:
            cube_dist = 3.0
            cube_angle = 0.0
            cube_detected = False

        # Holding state
        holding = self.state == MCPState.DEPOSITING

        # Calculate blocked scores
        front_dist = sectors.get('front', {}).get('min', float('inf'))
        left_dist = min(sectors.get('left', {}).get('min', float('inf')),
                       sectors.get('front_left', {}).get('min', float('inf')))
        right_dist = min(sectors.get('right', {}).get('min', float('inf')),
                        sectors.get('front_right', {}).get('min', float('inf')))

        front_blocked_score = max(0.0, min(1.0, (0.35 - front_dist) / 0.35)) if front_dist < 0.35 else 0.0
        lateral_blocked_score = max(
            max(0.0, min(1.0, (0.30 - left_dist) / 0.30)) if left_dist < 0.30 else 0.0,
            max(0.0, min(1.0, (0.30 - right_dist) / 0.30)) if right_dist < 0.30 else 0.0
        )

        return FuzzyInputs(
            distance_to_obstacle=min(obstacle_dist, 5.0),
            angle_to_obstacle=max(-135, min(135, obstacle_angle)),
            distance_to_cube=min(cube_dist, 3.0),
            angle_to_cube=max(-135, min(135, cube_angle)),
            cube_detected=cube_detected,
            holding_cube=holding,
            front_blocked=front_blocked_score,
            lateral_blocked=lateral_blocked_score
        )

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

            # During search, accept targets with 2+ frames (rotation makes 3 frames difficult)
            # Also check confidence >= 0.60 for reliability
            if target and target.frames_tracked >= 2 and target.confidence >= 0.60:
                # CRITICAL: Stop rotation IMMEDIATELY before state change
                self.base.reset()
                self.target_cube_color = target.color
                self._target_side = 'right' if target.angle > 0 else 'left'  # Remember which side
                self.vision.lock_color(target.color)
                # Reset search state for next time
                self._search_angle_covered = 0.0
                self._search_phase = 'scan'
                self.state = MCPState.APPROACHING
                print(f"[MCP Search] Target acquired: {target.color} at {target.distance:.2f}m, {target.angle:.1f}° (frames={target.frames_tracked}) → APPROACHING")
                return  # Don't do anything else this frame
            elif target and target.frames_tracked == 1 and target.confidence >= 0.60:
                # NEW: Promising detection found - PAUSE rotation to let tracking stabilize
                # This gives VisionService one more frame to confirm (reach frames_tracked=2)
                self.base.reset()
                return  # Wait one frame for tracking to confirm
            else:
                # Get LIDAR data - only consider FRONT sectors for obstacle avoidance
                lidar_data = self._get_lidar_data()
                sectors = lidar_data.get("sectors", {})
                front_dist = min(
                    sectors.get("front", {}).get("min", float('inf')),
                    sectors.get("front_left", {}).get("min", float('inf')),
                    sectors.get("front_right", {}).get("min", float('inf'))
                )

                # Use fuzzy controller for obstacle-aware navigation (MATA64 requirement)
                inputs = self._compute_fuzzy_inputs()
                try:
                    outputs = self.fuzzy.infer(inputs)
                except Exception:
                    outputs = None

                # Controlled search pattern: scan ~270° then move forward
                if self._search_phase == 'scan':
                    # Rotate to scan (alternating direction)
                    omega = 0.4 if self._search_direction else -0.4
                    self._search_angle_covered += abs(omega) * self.time_step / 1000.0

                    # After ~270° (4.7 rad), switch to move phase
                    if self._search_angle_covered > 4.7:
                        self._search_phase = 'move'
                        self._search_move_time = 0.0
                        self._search_direction = not self._search_direction
                        print("[MCP Search] Scan complete, moving forward")

                    self.base.move(0, 0, omega)
                else:
                    # Check FRONT obstacles before moving forward
                    if front_dist < 0.35:
                        # Front obstacle - switch to scan phase and change direction
                        self.base.reset()
                        self._search_phase = 'scan'
                        self._search_angle_covered = 0.0
                        self._search_direction = not self._search_direction
                        print(f"[MCP Search] Front obstacle at {front_dist:.2f}m, scanning")
                        return

                    # Move forward for ~2 seconds
                    self._search_move_time += self.time_step / 1000.0
                    if self._search_move_time > 2.0:
                        self._search_phase = 'scan'
                        self._search_angle_covered = 0.0
                        print("[MCP Search] Move complete, scanning again")

                    # Use fuzzy output for velocity if available (MATA64 requirement)
                    vx = 0.12
                    if outputs and outputs.linear_velocity > 0.05:
                        vx = min(outputs.linear_velocity, 0.15)
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

                # Keep rotating in direction of last known target
                if self._approach_lost_frames < 90:  # ~3s persistence (was 60)
                    # Search in direction of last known position
                    # Negative last_angle (cube was LEFT) → positive omega (turn LEFT)
                    search_omega = 0.25 if self._last_known_angle < 0 else -0.25
                    self.base.move(0.02, 0, search_omega)  # Slight forward + rotate
                    if self._approach_lost_frames % 30 == 0:
                        print(f"[MCP] Approach: target lost, searching... ({self._approach_lost_frames}/90)")
                    return

                # Truly lost - go back to SEARCHING but KEEP color lock
                # This forces the robot to find the SAME color again
                print(f"[MCP] Approach: target lost 90 frames, re-searching for {self.target_cube_color}")
                self._approach_lost_frames = 0
                # DON'T unlock color - keep looking for same color
                self.state = MCPState.SEARCHING
                self.base.reset()
                return

            # Reset lost frame counter and remember angle
            self._approach_lost_frames = 0
            self._last_known_angle = target.angle

            # Check if target is on the expected side - if not, IGNORE and keep rotating
            target_side = getattr(self, '_target_side', None)
            if target_side:
                current_side = 'right' if target.angle > 0 else 'left'
                if current_side != target_side and abs(target.angle) > 5:  # Allow small drift
                    # Wrong side detection - IGNORE it and keep rotating toward expected side
                    # Right side (+angle) → turn right → negative omega
                    # Left side (-angle) → turn left → positive omega
                    omega = -0.20 if target_side == 'right' else 0.20
                    self.base.move(0, 0, omega)  # Pure rotation toward expected side
                    return  # Skip the rest of this frame

            # Check for front obstacles during approach
            lidar_data = self._get_lidar_data()
            sectors = lidar_data.get("sectors", {})
            front_dist = min(
                sectors.get("front", {}).get("min", float('inf')),
                sectors.get("front_left", {}).get("min", float('inf')),
                sectors.get("front_right", {}).get("min", float('inf'))
            )

            # Obstacle too close - stop and return to search
            if front_dist < 0.25 and target.distance > 0.30:
                self.base.reset()
                print(f"[MCP] Approach: obstacle at {front_dist:.2f}m, returning to search")
                self.vision.unlock()
                self.target_cube_color = None
                self.state = MCPState.SEARCHING
                return

            # Check if close enough to grasp
            # MUST be well-aligned frontally (angle < 12°) to ensure accurate grasp
            # Lateral correction during grasp is imprecise at larger angles
            if target.distance <= 0.32 and abs(target.angle) < 12:
                self.base.reset()
                # SAVE target parameters BEFORE transitioning (VisionService may lose tracking)
                self._grasp_target_distance = target.distance
                self._grasp_target_angle = target.angle
                print(f"[MCP] Approach complete at {target.distance:.2f}m, {target.angle:.1f}° → GRASPING")
                self.state = MCPState.GRASPING
                return

            # Use fuzzy controller for approach decisions (MATA64 requirement)
            inputs = self._compute_fuzzy_inputs()
            try:
                fuzzy_out = self.fuzzy.infer(inputs)
                fuzzy_vx = fuzzy_out.linear_velocity
            except Exception:
                fuzzy_vx = 0.1

            # Proportional control for alignment and approach
            # OMEGA SIGN: positive angle (cube RIGHT) → turn RIGHT → negative omega
            # Negative angle (cube LEFT) → turn LEFT → positive omega
            angle_rad = math.radians(target.angle)

            if abs(target.angle) > 15:
                # Large angle: ROTATE IN PLACE first (no forward motion)
                # This prevents oscillation between two same-color cubes
                omega = -angle_rad * 0.5  # Gentle rotation
                omega = max(-0.25, min(0.25, omega))
                self.base.move(0, 0, omega)  # Pure rotation, no forward
                return
            elif abs(target.angle) > 8:
                # Medium angle: rotate with slow forward motion
                omega = -angle_rad * 0.6
                omega = max(-0.30, min(0.30, omega))
                vx = 0.05
                action = "ALIGN"
                self.base.move(vx, 0, omega)
                return
            else:
                # Aligned - move forward with minor correction
                omega = -angle_rad * 0.8  # Gentle correction while moving
                # Speed proportional to distance (slow down as we get close)
                vx = min(0.15, max(0.06, target.distance * 0.5))
                if fuzzy_vx > 0.05:
                    vx = min(vx, fuzzy_vx)
                action = "FORWARD"

            # Apply movement
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
                log(f"[MCP] GRASPING: Current arm height = {self._height_to_string(self.arm.current_height)}")
                self._save_grasp_screenshot("before")

                try:
                    import math

                    # VERIFY cube is actually visible before grasp
                    self._update_vision()
                    current_target = self.vision.get_target()
                    if not current_target:
                        log("[MCP] GRASPING: ABORT - No cube visible!")
                        self._save_grasp_screenshot("abort_no_cube")
                        self.vision.unlock()
                        self.state = MCPState.SEARCHING
                        self.grasp_started = False
                        return

                    # Use CURRENT target parameters (not saved)
                    target_distance = current_target.distance
                    target_angle = current_target.angle
                    log(f"[MCP] GRASPING: Current target at {target_distance:.3f}m, {target_angle:.1f}° (color={current_target.color})")

                    # Step 0: PRE-ALIGNMENT - rotate to center cube in view
                    # Cube at negative angle (left) -> turn by same angle (CW/right) to center
                    # Cube at positive angle (right) -> turn by same angle (CCW/left) to center
                    if abs(target_angle) > 5:
                        turn_angle = target_angle  # Same sign as target angle
                        log(f"[MCP] GRASPING: Step 0 - Pre-alignment (turning {turn_angle:.1f}° to center cube at {target_angle:.1f}°)")
                        self.movement.turn(angle_deg=turn_angle, speed=0.3)
                        # Wait for robot to settle after rotation (30 steps ~ 0.5s)
                        for _ in range(30):
                            self._step()
                        self._update_vision()
                        current_target = self.vision.get_target()
                        if current_target:
                            target_angle = current_target.angle
                            target_distance = current_target.distance
                            log(f"[MCP] GRASPING: After alignment: {target_distance:.3f}m, {target_angle:.1f}°")
                        self._save_grasp_screenshot("after_align")

                    # With FRONT_FLOOR preset, gripper position is fixed.
                    # Need to drive robot VERY close so cube is directly under gripper.
                    # Forward move should put the cube at the gripper's reach position.
                    # For target at ~0.30m, need to close ~25cm to reach FRONT_FLOOR position.
                    forward_move = 0.25
                    log(f"[MCP] GRASPING: Step 1 - Forward approach ({forward_move*100:.0f}cm)")
                    self.movement.forward(distance_m=forward_move, speed=0.03)
                    self._save_grasp_screenshot("after_forward")

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

                    # Use IK for precise positioning to 3cm cube
                    # After forward_move, remaining distance from camera = target_distance - forward_move
                    # forward_reach = remaining_distance + CAMERA_ARM_OFFSET
                    CAMERA_ARM_OFFSET = 0.15
                    remaining_dist = max(0.03, target_distance - forward_move)
                    forward_reach = remaining_dist + CAMERA_ARM_OFFSET
                    forward_reach = max(0.18, min(0.32, forward_reach))
                    log(f"[MCP] GRASPING: Step 3 - Execute grasp with IK (reach={forward_reach:.3f}m)")
                    result = self.arm_svc.execute_grasp(use_ik=True, forward_reach=forward_reach)
                    self._save_grasp_screenshot("after_grasp")
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
