"""
MainControllerV2 - Clean controller using modular services

Replaces monolithic main_controller.py with service-based architecture.
Based on: Brooks (1986) - Subsumption Architecture

DECISÃO 028: Modular restructure for testable components.
"""

import sys
import time
import math
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
from perception.lidar_processor import LIDARProcessor, ObstacleMap


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
            self.lidar_fov = self.lidar.getFov()
            self.lidar_resolution = self.lidar.getHorizontalResolution()
            self.lidar_max_range = self.lidar.getMaxRange()
            print(f"[LIDAR] Device enabled (FoV={self.lidar_fov:.2f} rad, res={self.lidar_resolution})")
        else:
            self.lidar_fov = None
            self.lidar_resolution = None
            self.lidar_max_range = 5.0

        # Load RNA model for LIDAR (MATA64 requirement)
        self.lidar_model = self._load_lidar_model()
        self.lidar_processor = self._load_lidar_processor()

        # Initialize services
        self.movement = MovementService(self.base, self.robot, self.time_step)
        self.arm_svc = ArmService(self.arm, self.gripper, self.robot, self.time_step)
        self.detector = CubeDetector()
        self.vision = VisionService(
            self.detector,
            self.time_step,
            lidar_probe=self._lidar_distance_at
        )
        self.navigation = NavigationService(
            self.movement, self.vision, self.robot, self.camera, self.time_step,
            lidar=self.lidar, lidar_model=self.lidar_model,
            obstacle_map_provider=self._compute_obstacle_map
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

        # Search pattern state (prevents constant 360° spin)
        self._search_angle_covered = 0.0  # radians covered in current scan
        self._search_direction = True     # True=left, False=right
        self._search_phase = 'scan'       # 'scan' or 'move'
        self._search_move_time = 0.0      # time spent moving forward

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

    def _load_lidar_processor(self) -> Optional[LIDARProcessor]:
        """Attempt to load TorchScript-based LIDAR processor."""
        model_path = Path(__file__).parent.parent / "models" / "lidar_processor.ts"
        if not model_path.exists():
            return None
        try:
            processor = LIDARProcessor(str(model_path))
            print(f"[LIDAR] Processor loaded: {model_path}")
            return processor
        except Exception as exc:
            print(f"[LIDAR] Failed to load processor: {exc}")
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

    def _compute_obstacle_map(self) -> Optional[ObstacleMap]:
        """Build ObstacleMap using available processor/MLP/heuristics."""
        if not self.lidar:
            return None

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return None

        ranges_np = np.array(ranges, dtype=np.float32)

        if self.lidar_processor:
            try:
                return self.lidar_processor.process(ranges_np)
            except Exception as exc:
                print(f"[LIDAR] Processor inference failed: {exc}")

        if self.lidar_model:
            probabilities = self._get_obstacle_map_rna(ranges)
            if probabilities:
                return self._obstacle_map_from_probabilities(probabilities, ranges_np)

        return self._heuristic_obstacle_map(ranges_np)

    def _obstacle_map_from_probabilities(
        self,
        probabilities: List[float],
        ranges: np.ndarray
    ) -> ObstacleMap:
        probs = np.array(probabilities, dtype=np.float32)
        probs = np.pad(probs, (0, max(0, 9 - probs.size)), constant_values=0)[:9]
        sectors = (probs > 0.5).astype(np.float32)
        min_distances = self._sector_min_distances(ranges)
        return ObstacleMap(
            sectors=sectors,
            probabilities=probs,
            min_distances=min_distances
        )

    def _heuristic_obstacle_map(self, ranges: np.ndarray) -> ObstacleMap:
        min_distances = self._sector_min_distances(ranges)
        probabilities = np.where(min_distances < 0.5, 1.0, 0.0).astype(np.float32)
        sectors = (probabilities > 0.5).astype(np.float32)
        return ObstacleMap(
            sectors=sectors,
            probabilities=probabilities,
            min_distances=min_distances
        )

    def _sector_min_distances(self, ranges: np.ndarray) -> np.ndarray:
        if ranges.size == 0:
            return np.full(9, np.inf, dtype=np.float32)

        points_per_sector = max(1, ranges.size // 9)
        min_distances = np.full(9, np.inf, dtype=np.float32)
        max_range = self.lidar_max_range if hasattr(self, 'lidar_max_range') else 5.0

        for i in range(9):
            start = i * points_per_sector
            end = start + points_per_sector
            sector = ranges[start:end]
            valid = sector[(sector > 0.01) & (sector < max_range)]
            if valid.size > 0:
                min_distances[i] = float(valid.min())

        return min_distances

    def _get_min_obstacle_distance(self, obstacle_map: Optional[ObstacleMap] = None) -> float:
        """Get minimum obstacle distance from computed obstacle map."""
        obstacle_map = obstacle_map or self._compute_obstacle_map()
        if obstacle_map is None:
            return float('inf')

        valid = obstacle_map.min_distances[np.isfinite(obstacle_map.min_distances)]
        if valid.size == 0:
            return float('inf')
        return float(valid.min())

    def _get_obstacle_sectors(self, obstacle_map: Optional[ObstacleMap] = None) -> dict:
        """Get obstacle presence per sector from ObstacleMap."""
        obstacle_map = obstacle_map or self._compute_obstacle_map()
        if obstacle_map is None:
            return {'left': False, 'front': False, 'right': False}

        return {
            'left': any(obstacle_map.is_sector_occupied(idx) for idx in range(0, 3)),
            'front': any(obstacle_map.is_sector_occupied(idx) for idx in range(3, 6)),
            'right': any(obstacle_map.is_sector_occupied(idx) for idx in range(6, 9)),
        }

    def _lidar_distance_at(self, angle_deg: float) -> Optional[float]:
        """Return LIDAR distance sample at given camera-relative angle (degrees)."""
        if not self.lidar or self.lidar_resolution in (None, 0) or not self.lidar_fov:
            return None

        ranges = self.lidar.getRangeImage()
        if not ranges:
            return None

        angle_rad = math.radians(angle_deg)
        half_fov = self.lidar_fov / 2.0
        if angle_rad < -half_fov or angle_rad > half_fov:
            return None

        normalized = (angle_rad + half_fov) / self.lidar_fov
        index = int(round(normalized * (self.lidar_resolution - 1)))
        index = max(0, min(self.lidar_resolution - 1, index))
        value = ranges[index]

        if value <= 0.01 or math.isinf(value):
            return None
        return value

    def _compute_fuzzy_inputs(self) -> FuzzyInputs:
        """Convert sensor data to FuzzyInputs for controller."""
        obstacle_map = self._compute_obstacle_map()

        # Get obstacle data from RNA / processor
        obstacle_dist = self._get_min_obstacle_distance(obstacle_map)
        sectors = self._get_obstacle_sectors(obstacle_map)

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

        front_blocked = 0.0
        lateral_blocked = 0.0
        if obstacle_map:
            front_clear = obstacle_map.get_clearance([3, 4, 5])
            left_clear = obstacle_map.get_clearance([0, 1, 2])
            right_clear = obstacle_map.get_clearance([6, 7, 8])

            front_blocked = self._blocked_score(front_clear, 0.35)
            lateral_blocked = max(
                self._blocked_score(left_clear, 0.30),
                self._blocked_score(right_clear, 0.30)
            )

        return FuzzyInputs(
            distance_to_obstacle=min(obstacle_dist, 5.0),
            angle_to_obstacle=max(-135, min(135, obstacle_angle)),
            distance_to_cube=min(cube_dist, 3.0),
            angle_to_cube=max(-135, min(135, cube_angle)),
            cube_detected=cube_detected,
            holding_cube=holding,
            front_blocked=front_blocked,
            lateral_blocked=lateral_blocked
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

        # Avoid initial wall collisions: back away if front clearance < 0.4m
        obstacle_map = self._compute_obstacle_map()
        init_clearance = obstacle_map.get_clearance([3, 4, 5]) if obstacle_map else float('inf')
        if init_clearance < 0.40:
            print(f"[Startup] Front clearance {init_clearance:.2f}m < 0.40m, backing up")
            self.movement.backward(distance_m=0.30, speed=0.10)
            self.movement.turn(angle_deg=90)

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
                if obstacle_dist < 0.35:
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
        """SEARCHING: Use controlled scan-then-move pattern."""
        target = self.vision.get_target()

        if target and target.is_reliable:
            # Found a cube!
            self.target_cube_color = target.color
            self.vision.lock_color(target.color)
            self._transition_to(RobotStateV2.APPROACHING, f"found {target.color}")
            # Reset search state for next time
            self._search_angle_covered = 0.0
            self._search_phase = 'scan'
            return

        # Use fuzzy controller for obstacle awareness
        inputs = self._compute_fuzzy_inputs()
        outputs = self.fuzzy.infer(inputs)

        # Check for obstacles - stop if too close
        if inputs.distance_to_obstacle < 0.4:
            self.movement.stop()
            self._search_angle_covered = 0.0  # Reset scan after obstacle
            self._search_phase = 'scan'
            self._search_direction = not self._search_direction  # Try other direction
            return

        # Controlled search pattern: scan ~270° then move forward
        if self._search_phase == 'scan':
            # Rotate to scan for cubes (alternating direction)
            omega = 0.3 if self._search_direction else -0.3
            self._search_angle_covered += abs(omega) * self.time_step / 1000.0

            # After ~270° (4.7 rad), switch to move phase
            if self._search_angle_covered > 4.7:
                self._search_phase = 'move'
                self._search_move_time = 0.0
                self._search_direction = not self._search_direction  # Alternate next time
                print("[Search] Scan complete, moving forward")

            self.movement.move_continuous(vx=0, vy=0, omega=omega)
        else:
            # Move forward for ~2 seconds, then scan again
            self._search_move_time += self.time_step / 1000.0
            if self._search_move_time > 2.0:
                self._search_phase = 'scan'
                self._search_angle_covered = 0.0
                print("[Search] Move complete, scanning again")

            # Move forward at moderate speed
            vx = min(outputs.linear_velocity, 0.12) if outputs.linear_velocity > 0.05 else 0.10
            self.movement.move_continuous(vx=vx, vy=0, omega=0)

    def _do_approaching(self) -> None:
        """APPROACHING: Use NavigationService to approach target."""
        result = self.navigation.approach_target()

        if result.success:
            self.movement.stop()
            self._transition_to(RobotStateV2.GRASPING, f"dist={result.final_distance:.2f}m")
        else:
            # Any failure (LOST, max_attempts, etc) - unlock and search again
            self.vision.unlock()
            self.target_cube_color = None
            self._transition_to(RobotStateV2.SEARCHING, f"approach_failed:{result.reason}")

    def _do_grasping(self) -> None:
        """GRASPING: Use ArmService for grasp sequence."""
        print(f"[Grasping] Attempting grasp of {self.target_cube_color} cube")

        # Get target info before final approach
        target = self.vision.get_target()
        if not target or not target.is_reliable:
            print("[Grasping] Target lost before grasp, re-approaching")
            self.movement.stop()
            self._transition_to(RobotStateV2.APPROACHING, "target_unstable")
            return

        if not self._vision_lidar_consistent(target):
            print("[Grasping] Vision/LIDAR mismatch (>5cm). Re-approaching.")
            self.movement.stop()
            self._transition_to(RobotStateV2.APPROACHING, "distance_mismatch")
            return

        # Final approach: move forward to close gap (theory: 6cm from 22cm approach distance)
        print("[Grasping] Final approach (6cm)")
        self.movement.forward(distance_m=0.06, speed=0.06)

        # Prepare and execute grasp
        if not self.arm_svc.prepare_grasp():
            self._transition_to(RobotStateV2.SEARCHING, "prepare_failed")
            return

        # Calculate forward reach for IK
        # Camera is ~15cm ahead of arm base, so arm needs to reach FURTHER than camera sees
        # After 6cm approach: camera_dist = target.distance - 0.06
        # Arm reach needed: camera_dist + CAMERA_ARM_OFFSET (camera ahead of arm base)
        CAMERA_ARM_OFFSET = 0.15  # Camera is ~15cm ahead of arm base
        camera_dist = (target.distance - 0.06) if target else 0.16
        forward_reach = camera_dist + CAMERA_ARM_OFFSET
        forward_reach = max(0.18, min(0.32, forward_reach))  # Clamp to IK working range
        print(f"[Grasping] Using IK: camera_dist={camera_dist:.2f}m + offset={CAMERA_ARM_OFFSET:.2f}m = reach={forward_reach:.2f}m")
        result = self.arm_svc.execute_grasp(use_ik=True, forward_reach=forward_reach)

        if result.success:
            if not self._confirm_grasp_clearance():
                print("[Grasping] Clearance check failed, cube likely still blocking front.")
                self.stats.cubes_failed += 1
                self.arm_svc.reset()
                self.vision.unlock()
                self._transition_to(RobotStateV2.SEARCHING, "clearance_failed")
                return

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

    def _vision_lidar_consistent(self, target) -> bool:
        """Check if vision and LIDAR distances agree within 5cm."""
        if not target:
            return False
        lidar_distance = self._lidar_distance_at(target.angle)
        if lidar_distance is None:
            return True
        return abs(lidar_distance - target.distance) <= 0.05

    def _confirm_grasp_clearance(self) -> bool:
        """Ensure front LIDAR sectors clear after grasp to avoid dragging cubes."""
        obstacle_map = self._compute_obstacle_map()
        if not obstacle_map:
            return True
        clearance = obstacle_map.get_clearance([3, 4, 5])
        return clearance > 0.20 or not math.isfinite(clearance)

    def _blocked_score(self, clearance: float, limit: float) -> float:
        if clearance is None or not math.isfinite(clearance):
            return 0.0
        return float(np.clip((limit - clearance) / limit, 0.0, 1.0))

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
