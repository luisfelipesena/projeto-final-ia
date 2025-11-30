"""
MainControllerV2 - Clean controller using modular services

Replaces monolithic main_controller.py with service-based architecture.
Based on: Brooks (1986) - Subsumption Architecture

DECISÃO 028: Modular restructure for testable components.
"""

import sys
import time
import numpy as np
from enum import Enum, auto
from typing import Optional, List
from dataclasses import dataclass
from pathlib import Path

import torch

# Add project paths
sys.path.insert(0, '/Users/luisfelipesena/Development/Personal/projeto-final-ia/src')
sys.path.insert(0, '/Users/luisfelipesena/Development/Personal/projeto-final-ia/IA_20252/controllers/youbot')

from controller import Robot

# Import Webots controllers
from base import Base
from arm import Arm
from gripper import Gripper

# Import our services
from services.movement_service import MovementService
from services.arm_service import ArmService
from services.vision_service import VisionService
from services.navigation_service import NavigationService

# Import cube detector
from perception.cube_detector import CubeDetector

# Import RNA model for LIDAR (MATA64 requirement)
from perception.models.simple_lidar_mlp import SimpleLIDARMLP

# Import Fuzzy Controller (MATA64 requirement)
from control.fuzzy_controller import FuzzyController, FuzzyInputs


class RobotStateV2(Enum):
    """Simplified robot states"""
    SEARCHING = auto()      # Looking for cubes
    APPROACHING = auto()    # Moving toward detected cube
    GRASPING = auto()       # Executing grasp sequence
    DEPOSITING = auto()     # Moving to box and depositing
    AVOIDING = auto()       # Obstacle avoidance (emergency)


@dataclass
class ControllerStats:
    """Statistics for debugging"""
    cubes_collected: int = 0
    cubes_failed: int = 0
    state_changes: int = 0
    time_started: float = 0
    current_state_time: float = 0


# Deposit box coordinates (from odometry.py)
DEPOSIT_BOXES = {
    'green': (0.48, 1.58),
    'blue': (0.48, -1.62),
    'red': (2.31, 0.01),
}


class MainControllerV2:
    """
    Clean controller using modular services.

    State Machine:
        SEARCHING → APPROACHING → GRASPING → DEPOSITING → SEARCHING

    Each state uses specific services:
        SEARCHING: VisionService + MovementService (rotate scan)
        APPROACHING: NavigationService (align + approach)
        GRASPING: ArmService (grasp sequence)
        DEPOSITING: NavigationService + ArmService

    Usage:
        controller = MainControllerV2()
        controller.run()  # Main loop
    """

    # State timeouts (seconds)
    SEARCH_ROTATE_INTERVAL = 2.0  # Rotate every 2s while searching
    STATE_TIMEOUT = 60.0          # Max time per state before reset

    def __init__(self):
        """Initialize controller and all services."""
        print("[MainControllerV2] Initializing...")

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

        # LIDAR (for obstacle detection)
        # Webots uses lowercase device names by default
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            print(f"[LIDAR] Device enabled")

        # Load RNA model for LIDAR (MATA64 requirement)
        self.lidar_model = self._load_lidar_model()

        # Initialize services
        self.movement = MovementService(self.base, self.robot, self.time_step)
        self.arm_svc = ArmService(self.arm, self.gripper, self.robot, self.time_step)
        self.detector = CubeDetector()
        self.vision = VisionService(self.detector, self.time_step)
        self.navigation = NavigationService(
            self.movement, self.vision, self.robot, self.camera, self.time_step,
            lidar=self.lidar, lidar_model=self.lidar_model
        )

        # Initialize Fuzzy Controller (MATA64 requirement)
        self.fuzzy = FuzzyController(config={'logging': True})
        self.fuzzy.initialize()
        print("[MainControllerV2] Fuzzy controller initialized")

        # State
        self.state = RobotStateV2.SEARCHING
        self.state_start_time = time.time()
        self.last_rotate_time = time.time()
        self.target_cube_color: Optional[str] = None

        # Stats
        self.stats = ControllerStats(time_started=time.time())

        print("[MainControllerV2] Initialization complete")

    def _load_lidar_model(self) -> Optional[SimpleLIDARMLP]:
        """Load RNA model for LIDAR obstacle detection (MATA64 requirement)."""
        try:
            model = SimpleLIDARMLP(input_size=512, num_sectors=9)
            model_path = Path(__file__).parent.parent / "models" / "lidar_mlp.pth"

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
        points_per_sector = len(ranges) // num_sectors
        obstacles = []

        for i in range(num_sectors):
            start = i * points_per_sector
            end = start + points_per_sector
            sector = ranges[start:end]
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

    def _update_vision(self) -> None:
        """Update vision with current camera frame."""
        image = self.camera.getImage()
        if image:
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            image_rgb = image_array[:, :, :3]
            self.vision.update(image_rgb)

    def _get_min_obstacle_distance(self) -> float:
        """Get minimum obstacle distance from LIDAR using RNA when available."""
        if not self.lidar:
            return float('inf')

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return float('inf')

        # Use RNA to get obstacle map (MATA64 requirement)
        obstacle_map = self._get_obstacle_map_rna(ranges)

        # Find minimum distance in sectors with obstacles
        valid = [r for r in ranges if r > 0.01]
        if not valid:
            return float('inf')

        return min(valid)

    def _get_obstacle_sectors(self) -> dict:
        """Get obstacle presence per sector from RNA."""
        if not self.lidar:
            return {'left': False, 'front': False, 'right': False}

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return {'left': False, 'front': False, 'right': False}

        obstacles = self._get_obstacle_map_rna(ranges)
        # Sectors: 0-2 left, 3-5 front, 6-8 right
        return {
            'left': any(o > 0.5 for o in obstacles[0:3]),
            'front': any(o > 0.5 for o in obstacles[3:6]),
            'right': any(o > 0.5 for o in obstacles[6:9]),
        }

    def _compute_fuzzy_inputs(self) -> FuzzyInputs:
        """Convert sensor data to FuzzyInputs for controller."""
        # Get obstacle data from RNA
        obstacle_dist = self._get_min_obstacle_distance()
        sectors = self._get_obstacle_sectors()

        # Compute obstacle angle from sectors
        if sectors['left'] and not sectors['right']:
            obstacle_angle = -45.0
        elif sectors['right'] and not sectors['left']:
            obstacle_angle = 45.0
        elif sectors['front']:
            obstacle_angle = 0.0
        else:
            obstacle_angle = 0.0

        # Get cube data from vision
        target = self.vision.get_target()
        if target:
            cube_dist = target.distance
            cube_angle = target.angle
            cube_detected = True
        else:
            cube_dist = 3.0
            cube_angle = 0.0
            cube_detected = False

        # Holding state
        holding = self.state == RobotStateV2.DEPOSITING

        return FuzzyInputs(
            distance_to_obstacle=min(obstacle_dist, 5.0),
            angle_to_obstacle=max(-135, min(135, obstacle_angle)),
            distance_to_cube=min(cube_dist, 3.0),
            angle_to_cube=max(-135, min(135, cube_angle)),
            cube_detected=cube_detected,
            holding_cube=holding
        )

    def _transition_to(self, new_state: RobotStateV2, reason: str = "") -> None:
        """Transition to new state."""
        if new_state != self.state:
            print(f"[State] {self.state.name} → {new_state.name} ({reason})")
            self.state = new_state
            self.state_start_time = time.time()
            self.stats.state_changes += 1

    def run(self) -> None:
        """
        Main control loop.

        Runs until simulation ends or 15 cubes collected.
        """
        print("[MainControllerV2] Starting main loop")
        print(f"  Time step: {self.time_step}ms")

        # Warmup
        for _ in range(10):
            if not self._step():
                return

        # Ensure arm is reset
        self.arm_svc.reset()

        while self._step():
            # Update vision every frame
            self._update_vision()

            # Check state timeout
            state_duration = time.time() - self.state_start_time
            if state_duration > self.STATE_TIMEOUT:
                print(f"[WARNING] State timeout in {self.state.name}")
                self._transition_to(RobotStateV2.SEARCHING, "timeout")
                continue

            # Check for obstacle emergency (except during manipulation & approaching)
            # During APPROACHING, NavigationService handles obstacle-aware movement
            if self.state not in (RobotStateV2.GRASPING, RobotStateV2.DEPOSITING, RobotStateV2.APPROACHING):
                obstacle_dist = self._get_min_obstacle_distance()
                if obstacle_dist < 0.25:
                    self._handle_obstacle(obstacle_dist)
                    continue

            # State-specific behavior
            if self.state == RobotStateV2.SEARCHING:
                self._do_searching()
            elif self.state == RobotStateV2.APPROACHING:
                self._do_approaching()
            elif self.state == RobotStateV2.GRASPING:
                self._do_grasping()
            elif self.state == RobotStateV2.DEPOSITING:
                self._do_depositing()
            elif self.state == RobotStateV2.AVOIDING:
                self._do_avoiding()

            # Check completion
            if self.stats.cubes_collected >= 15:
                print(f"[MainControllerV2] TASK COMPLETE: {self.stats.cubes_collected} cubes!")
                break

        self._print_final_stats()

    def _do_searching(self) -> None:
        """SEARCHING: Use fuzzy controller for intelligent search."""
        target = self.vision.get_target()

        if target and target.is_reliable:
            # Found a cube!
            self.target_cube_color = target.color
            self.vision.lock_color(target.color)
            self._transition_to(RobotStateV2.APPROACHING, f"found {target.color}")
            return

        # Use fuzzy controller for search behavior
        inputs = self._compute_fuzzy_inputs()
        outputs = self.fuzzy.infer(inputs)

        # Fuzzy outputs guide movement (with search-appropriate defaults)
        vx = outputs.linear_velocity if outputs.linear_velocity > 0.01 else 0.05
        omega = outputs.angular_velocity if abs(outputs.angular_velocity) > 0.05 else 0.3

        self.movement.move_continuous(vx=vx, vy=0, omega=omega)

    def _do_approaching(self) -> None:
        """APPROACHING: Use NavigationService to approach target."""
        result = self.navigation.approach_target()

        if result.success:
            self.movement.stop()
            self._transition_to(RobotStateV2.GRASPING, f"dist={result.final_distance:.2f}m")
        elif result.phase.name == "LOST":
            # Don't unlock - let LOST_THRESHOLD persistence handle brief detection failures
            # If target is truly lost, we'll re-acquire in SEARCHING
            self._transition_to(RobotStateV2.SEARCHING, "target_lost")

    def _do_grasping(self) -> None:
        """GRASPING: Use ArmService for grasp sequence."""
        print(f"[Grasping] Attempting grasp of {self.target_cube_color} cube")

        # Final approach: move forward to close gap between camera distance and arm reach
        # 15cm brings cube within gripper reach (~27cm from robot center)
        print("[Grasping] Final approach")
        self.movement.forward(distance_m=0.15, speed=0.06)

        # Prepare and execute grasp
        if not self.arm_svc.prepare_grasp():
            self._transition_to(RobotStateV2.SEARCHING, "prepare_failed")
            return

        result = self.arm_svc.execute_grasp()

        if result.success:
            self.stats.cubes_collected += 1
            print(f"[Grasping] SUCCESS! Total: {self.stats.cubes_collected}")
            self._transition_to(RobotStateV2.DEPOSITING, "grasp_success")
        else:
            self.stats.cubes_failed += 1
            print(f"[Grasping] FAILED: {result.error}")
            self.arm_svc.reset()
            self.vision.unlock()
            self._transition_to(RobotStateV2.SEARCHING, "grasp_failed")

    def _do_depositing(self) -> None:
        """DEPOSITING: Navigate to correct box and deposit cube."""
        if not self.target_cube_color:
            self._transition_to(RobotStateV2.SEARCHING, "no_color")
            return

        # Get target box position
        box_pos = DEPOSIT_BOXES.get(self.target_cube_color)
        if not box_pos:
            self._transition_to(RobotStateV2.SEARCHING, "unknown_color")
            return

        # Timed navigation phases: turn → drive → drop
        if not hasattr(self, '_deposit_phase'):
            self._deposit_phase = 'turn'
            self._deposit_start = time.time()
            print(f"[Deposit] Starting turn toward {self.target_cube_color} box at {box_pos}")

        if self._deposit_phase == 'turn':
            # Turn 180° (arm is at back) for ~2 seconds
            if time.time() - self._deposit_start < 2.0:
                self.movement.move_continuous(vx=0, vy=0, omega=0.8)
                return
            self.movement.stop()
            self._deposit_phase = 'drive'
            self._deposit_start = time.time()
            print("[Deposit] Driving toward box")

        if self._deposit_phase == 'drive':
            # Drive for ~3 seconds
            if time.time() - self._deposit_start < 3.0:
                self.movement.move_continuous(vx=0.15, vy=0, omega=0)
                return
            self.movement.stop()
            self._deposit_phase = 'drop'
            print("[Deposit] Executing drop")

        if self._deposit_phase == 'drop':
            # Execute deposit sequence
            self.arm_svc.prepare_deposit()
            self.arm_svc.execute_deposit()
            self.arm_svc.return_to_rest()

            # Cleanup state
            del self._deposit_phase
            del self._deposit_start
            self.vision.unlock()
            self.target_cube_color = None

            print(f"[Deposit] Complete! Cubes: {self.stats.cubes_collected}")
            self._transition_to(RobotStateV2.SEARCHING, "deposit_complete")

    def _do_avoiding(self) -> None:
        """AVOIDING: Back away from obstacle."""
        obstacle_dist = self._get_min_obstacle_distance()

        if obstacle_dist > 0.5:
            # Safe - return to previous activity
            self._transition_to(RobotStateV2.SEARCHING, "obstacle_cleared")
            return

        # Back up slowly
        self.movement.move_continuous(vx=-0.10, vy=0, omega=0)

    def _handle_obstacle(self, distance: float) -> None:
        """Handle obstacle emergency."""
        if self.state != RobotStateV2.AVOIDING:
            self.movement.stop()
            self._transition_to(RobotStateV2.AVOIDING, f"obstacle at {distance:.2f}m")

    def _print_final_stats(self) -> None:
        """Print final statistics."""
        elapsed = time.time() - self.stats.time_started
        print("\n" + "=" * 50)
        print("[MainControllerV2] FINAL STATISTICS")
        print("=" * 50)
        print(f"  Cubes collected: {self.stats.cubes_collected}")
        print(f"  Cubes failed: {self.stats.cubes_failed}")
        print(f"  State changes: {self.stats.state_changes}")
        print(f"  Total time: {elapsed:.1f}s")
        print("=" * 50)


# ==================== ENTRY POINT ====================

def main():
    """Main entry point."""
    controller = MainControllerV2()
    controller.run()


if __name__ == "__main__":
    main()
