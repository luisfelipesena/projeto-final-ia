"""
YouBot Controller - Main FSM logic for cube collection task.
Handles search, approach, grasp, navigation to box, drop, and return.
"""

import math
import sys
from controller import Supervisor

from base import Base
from arm import Arm
from gripper import Gripper
from color_classifier import ColorClassifier
from constants import (
    ARENA_CENTER, ARENA_SIZE, KNOWN_OBSTACLES, 
    BOX_POSITIONS, SPAWN_POSITION
)
from routes import get_route_to_box, get_return_route, wrap_angle, color_from_rgb
from occupancy_grid import OccupancyGrid
from fuzzy_navigator import FuzzyNavigator

# MCP Bridge for Claude Code integration (optional)
MCP_BRIDGE_PATH = "/Users/luisfelipesena/Development/Personal/webots-youbot-mcp"
sys.path.insert(0, MCP_BRIDGE_PATH)
try:
    from mcp_bridge import MCPBridge
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[MCP] Bridge not available - running without MCP integration")


class YouBotController:
    """Main controller for YouBot cube collection task."""

    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self._log_times = {}

        # Camera with Recognition
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            if self.camera.hasRecognition():
                self.camera.recognitionEnable(self.time_step)
                print("[INIT] Camera recognition enabled")
            else:
                print("[INIT] Camera has no recognition capability")

        # Main LiDAR
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            self.lidar.enablePointCloud()

        # Lateral and Rear LiDARs
        self.lidar_rear = self.robot.getDevice("lidar_rear")
        if self.lidar_rear:
            self.lidar_rear.enable(self.time_step)
            self.lidar_rear.enablePointCloud()

        self.lidar_left = self.robot.getDevice("lidar_left")
        if self.lidar_left:
            self.lidar_left.enable(self.time_step)
            self.lidar_left.enablePointCloud()

        self.lidar_right = self.robot.getDevice("lidar_right")
        if self.lidar_right:
            self.lidar_right.enable(self.time_step)
            self.lidar_right.enablePointCloud()

        # Distance sensors (rear)
        self.ds_rear = self.robot.getDevice("ds_rear")
        self.ds_rear_left = self.robot.getDevice("ds_rear_left")
        self.ds_rear_right = self.robot.getDevice("ds_rear_right")

        # Distance sensors (lateral)
        self.ds_left = self.robot.getDevice("ds_left")
        self.ds_right = self.robot.getDevice("ds_right")

        # Distance sensors (front)
        self.ds_front = self.robot.getDevice("ds_front")
        self.ds_front_left = self.robot.getDevice("ds_front_left")
        self.ds_front_right = self.robot.getDevice("ds_front_right")

        # Enable all distance sensors
        for ds in [self.ds_rear, self.ds_rear_left, self.ds_rear_right,
                   self.ds_left, self.ds_right,
                   self.ds_front, self.ds_front_left, self.ds_front_right]:
            if ds:
                ds.enable(self.time_step)

        self.navigator = FuzzyNavigator()

        # Color classifier (Neural Network)
        self.color_classifier = ColorClassifier("model/color_model.onnx")

        self.pose = self._initial_pose()
        self.current_target = None
        self.current_color = None
        self.mode = "search"
        self.stage = 0
        self.stage_timer = 0.0
        self.collected = 0
        self.max_cubes = 15

        # Lawnmower search state
        self.search_state = "forward"
        self.search_direction = 1
        self.turn_progress = 0.0

        # Cube tracking
        self.lost_cube_timer = 0.0
        self.locked_cube_angle = None
        self.locked_cube_distance = None

        self.box_positions = BOX_POSITIONS

        # Grid / path
        self.grid = OccupancyGrid(ARENA_CENTER, ARENA_SIZE, cell_size=0.12)
        self._seed_static_map()
        self._waypoints = []
        self._path_dirty = True
        self.active_goal = None
        self._max_cmd = 0.25

        # MCP Bridge for Claude Code real-time monitoring
        self.mcp = None
        self.delivered = {"red": 0, "green": 0, "blue": 0}
        if MCP_AVAILABLE:
            try:
                self.mcp = MCPBridge(self.robot)
                print("[MCP] Bridge initialized successfully")
            except Exception as e:
                print(f"[MCP] Bridge init failed: {e}")

    def _log_throttled(self, key, msg, interval=1.5):
        """Log message with throttling to avoid spam."""
        try:
            now = self.robot.getTime()
        except Exception:
            now = 0.0
        last = self._log_times.get(key, -1e9)
        if now - last >= interval:
            print(msg)
            self._log_times[key] = now

    # ===== Grid helpers =====
    def _seed_static_map(self):
        """Initialize grid with known obstacles."""
        self.grid.fill_border(OccupancyGrid.OBSTACLE, static=True)
        wall_inflate = 0.25
        for ox, oy, radius in KNOWN_OBSTACLES:
            self.grid.fill_disk(ox, oy, radius + wall_inflate, OccupancyGrid.OBSTACLE, static=True)
        for pos in self.box_positions.values():
            self.grid.fill_disk(pos[0], pos[1], 0.20, OccupancyGrid.BOX, static=True)
        print(
            f"[GRID] size=({self.grid.width}x{self.grid.height}) cell={self.grid.cell_size:.2f} "
            f"bounds=({self.grid.min_x:.2f},{self.grid.min_y:.2f})-({self.grid.max_x:.2f},{self.grid.max_y:.2f})"
        )

    def _set_goal(self, target):
        """Set navigation goal."""
        self.current_target = target
        self.active_goal = target
        self._waypoints = []
        self._path_dirty = True

    def _wall_clearances(self):
        """Get distances to arena walls."""
        x, y, _ = self.pose
        return (
            x - self.grid.min_x,
            self.grid.max_x - x,
            y - self.grid.min_y,
            self.grid.max_y - y,
        )

    def _enforce_boundary_safety(self, vx_cmd, vy_cmd, margin=0.25):
        """Prevent movement toward walls."""
        yaw = self.pose[2]
        dx_world = math.cos(yaw) * vx_cmd - math.sin(yaw) * vy_cmd
        dy_world = math.sin(yaw) * vx_cmd + math.cos(yaw) * vy_cmd

        left, right, bottom, top = self._wall_clearances()
        if left < margin and dx_world < 0:
            dx_world = 0.0
        if right < margin and dx_world > 0:
            dx_world = 0.0
        if bottom < margin and dy_world < 0:
            dy_world = 0.0
        if top < margin and dy_world > 0:
            dy_world = 0.0

        vx_adj = math.cos(yaw) * dx_world + math.sin(yaw) * dy_world
        vy_adj = -math.sin(yaw) * dx_world + math.cos(yaw) * dy_world
        return vx_adj, vy_adj

    def _safe_move(self, vx, vy, omega):
        """Execute movement with safety limits."""
        vals = [vx, vy, omega]
        safe = []
        for v in vals:
            if v is None or not math.isfinite(v):
                safe.append(0.0)
            else:
                safe.append(max(-self._max_cmd, min(self._max_cmd, v)))
        self.base.move(safe[0], safe[1], safe[2])

    def _clamp_cmds(self, vx, vy, omega):
        """Clamp velocity commands."""
        omega = max(-0.5, min(0.5, omega))
        vx = max(-0.18, min(0.18, vx))
        vy = max(-0.14, min(0.14, vy))
        return vx, vy, omega

    def _distance_to_point(self, point):
        """Calculate distance and angle to a point."""
        dx = point[0] - self.pose[0]
        dy = point[1] - self.pose[1]
        distance = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - self.pose[2])
        return distance, angle

    def _skip_passed_waypoints(self):
        """
        Skip waypoints that are behind the robot or already within threshold.
        This prevents getting stuck trying to reach waypoints we've passed.
        """
        if not hasattr(self, '_return_waypoints') or not self._return_waypoints:
            return
        
        wp_threshold = 0.45
        max_skip = min(5, len(self._return_waypoints) - 1)  # Don't skip more than 5
        skipped = 0
        
        while self._return_waypoint_idx < len(self._return_waypoints) and skipped < max_skip:
            wp = self._return_waypoints[self._return_waypoint_idx]
            dist, angle = self._distance_to_point(wp)
            
            # Skip if within threshold OR if behind us (angle > 120°) and close
            should_skip = False
            if dist < wp_threshold:
                should_skip = True
            elif abs(angle) > math.radians(120) and dist < 1.0:
                # Waypoint is behind us and relatively close - skip it
                should_skip = True
            
            if should_skip:
                self._return_waypoint_idx += 1
                skipped += 1
            else:
                break
        
        if skipped > 0:
            print(f"[RETURN] Skipped {skipped} passed waypoints, now at WP {self._return_waypoint_idx + 1}")

    def _replan_path(self):
        """Recompute A* path to goal."""
        if not self.active_goal:
            self._waypoints = []
            self._path_dirty = False
            return
        path = self.grid.plan_path((self.pose[0], self.pose[1]), self.active_goal)
        self._waypoints = path
        self._path_dirty = False

    def _update_grid_from_lidar(self, lidar_info):
        """Update occupancy grid from LIDAR readings."""
        if not lidar_info["points"]:
            return
        if any(math.isnan(v) for v in self.pose):
            return
            
        robot_x, robot_y, robot_yaw = self.pose
        origin = (robot_x, robot_y)
        
        hit_changed = False
        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)
        
        for lx, ly in lidar_info["points"]:
            wx = robot_x + lx * cy - ly * sy
            wy = robot_y + lx * sy + ly * cy
            
            if math.isnan(wx) or math.isnan(wy):
                continue
                
            if self.grid.raycast(origin, (wx, wy), OccupancyGrid.OBSTACLE):
                hit_changed = True
                
        if hit_changed:
            self._path_dirty = True

    def _check_rear_safety(self, rear_info, margin=0.30):
        """Check if reverse is safe."""
        if not rear_info:
            return True
        
        r = rear_info.get("rear", 2.0)
        rl = rear_info.get("rear_left", 2.0)
        rr = rear_info.get("rear_right", 2.0)
        
        if r < margin * 0.8:
            return False
        if rl < margin * 0.6 or rr < margin * 0.6:
            return False
        return True

    def _initial_pose(self):
        """Get initial pose from Webots."""
        try:
            self_node = self.robot.getSelf()
            if self_node:
                pos = self_node.getPosition()
                orient = self_node.getOrientation()
                yaw = math.atan2(orient[3], orient[0])
                return [pos[0], pos[1], yaw]
        except Exception as e:
            print(f"[POSE] Error getting initial pose: {e}")
        return [0.0, 0.0, 0.0]

    def _get_ground_truth_pose(self):
        """Get ground truth pose from Webots (for odometry correction)."""
        try:
            self_node = self.robot.getSelf()
            if self_node:
                pos = self_node.getPosition()
                orient = self_node.getOrientation()
                yaw = math.atan2(orient[3], orient[0])
                return [pos[0], pos[1], yaw]
        except Exception:
            pass
        return None

    def _integrate_pose(self, vx, vy, omega, dt):
        """Integrate odometry with NaN protection."""
        if math.isnan(vx) or math.isnan(vy) or math.isnan(omega):
            return

        if any(math.isnan(v) for v in self.pose):
            gt = self._get_ground_truth_pose()
            if gt:
                self.pose = gt
                print(f"[POSE] Recovered from ground truth: ({gt[0]:.2f}, {gt[1]:.2f})")
            else:
                self.pose = [0.0, 0.0, 0.0]
            return

        yaw = self.pose[2]
        dx_world = math.cos(yaw) * vx - math.sin(yaw) * vy
        dy_world = math.sin(yaw) * vx + math.cos(yaw) * vy

        new_x = self.pose[0] + dx_world * dt
        new_y = self.pose[1] + dy_world * dt
        new_yaw = wrap_angle(self.pose[2] + omega * dt)

        if not (math.isnan(new_x) or math.isnan(new_y) or math.isnan(new_yaw)):
            self.pose[0] = new_x
            self.pose[1] = new_y
            self.pose[2] = new_yaw
        else:
            gt = self._get_ground_truth_pose()
            if gt:
                self.pose = gt

    def _process_recognition(self, lock_color=None, lock_angle=None):
        """Process camera recognition for cube detection."""
        if not self.camera or not self.camera.hasRecognition():
            return None

        objects = self.camera.getRecognitionObjects()
        if not objects:
            return None

        best = None
        best_score = float('inf')

        for obj in objects:
            pos = obj.getPosition()
            colors = obj.getColors()

            dist = math.sqrt(pos[0]**2 + pos[1]**2)
            angle = math.atan2(pos[1], pos[0]) if pos[0] != 0 else 0

            color = None
            confidence = 0.0
            if colors:
                try:
                    r, g, b = colors[0], colors[1], colors[2]
                    color, confidence = self.color_classifier.predict_from_rgb(r, g, b)
                    if confidence < 0.5:
                        color = color_from_rgb(r, g, b)
                except Exception:
                    color = None

            if lock_color and color != lock_color:
                continue

            score = dist
            if lock_angle is not None:
                angle_diff = abs(wrap_angle(angle - lock_angle))
                if angle_diff < math.radians(10):
                    score -= 10.0
                else:
                    score += 5.0

            if score < best_score and dist < 2.5:
                best_score = score
                best = {
                    "color": color,
                    "distance": dist,
                    "angle": angle,
                    "position": pos,
                }

        return best

    def _process_lidar(self):
        """Process all LIDARs and fuse into single point cloud."""
        fused_points = []
        
        def process_sensor(sensor, dx, dy, dtheta):
            if not sensor:
                return 2.0, []
            ranges = sensor.getRangeImage()
            if not ranges:
                return 2.0, []
            
            res = sensor.getHorizontalResolution()
            fov = sensor.getFov()
            angle_step = fov / max(1, res - 1)
            min_dist = 2.0
            sensor_points = []
            
            center_idx = len(ranges) // 2
            fov_deg = math.degrees(fov)
            if fov_deg > 0:
                window = int(len(ranges) * (40.0 / fov_deg) / 2)
            else:
                window = 5
            window = max(1, window)
            
            start = max(0, center_idx - window)
            end = min(len(ranges), center_idx + window)
            
            center_vals = [r for r in ranges[start:end] if not (math.isinf(r) or math.isnan(r))]
            if center_vals:
                min_dist = min(center_vals)
            
            for i, r in enumerate(ranges):
                if math.isinf(r) or math.isnan(r) or r <= 0.1:
                    continue
                
                alpha = -fov / 2.0 + i * angle_step
                sx = r * math.cos(alpha)
                sy = r * math.sin(alpha)
                rx = dx + sx * math.cos(dtheta) - sy * math.sin(dtheta)
                ry = dy + sx * math.sin(dtheta) + sy * math.cos(dtheta)
                sensor_points.append((rx, ry))
                
            return min_dist, sensor_points

        front_dist, p_front = process_sensor(self.lidar, 0.28, 0.0, 0.0)
        fused_points.extend(p_front)
        
        rear_dist, p_rear = process_sensor(self.lidar_rear, -0.29, 0.0, math.pi)
        fused_points.extend(p_rear)
        
        left_dist, p_left = process_sensor(self.lidar_left, 0.0, 0.22, math.pi/2)
        fused_points.extend(p_left)
        
        right_dist, p_right = process_sensor(self.lidar_right, 0.0, -0.22, -math.pi/2)
        fused_points.extend(p_right)
        
        return {
            "front": front_dist,
            "rear": rear_dist,
            "left": left_dist,
            "right": right_dist,
            "points": fused_points
        }

    def _process_rear_sensors(self):
        """Process rear distance sensors."""
        def raw_to_meters(raw_value):
            if raw_value is None or math.isnan(raw_value):
                return 2.0
            return max(0.05, min(2.0, raw_value / 1000.0))

        rear = 2.0
        rear_left = 2.0
        rear_right = 2.0
        
        if self.ds_rear:
            rear = raw_to_meters(self.ds_rear.getValue())
        if self.ds_rear_left:
            rear_left = raw_to_meters(self.ds_rear_left.getValue())
        if self.ds_rear_right:
            rear_right = raw_to_meters(self.ds_rear_right.getValue())

        return {"rear": rear, "rear_left": rear_left, "rear_right": rear_right}

    def _process_lateral_sensors(self):
        """Process lateral distance sensors."""
        def raw_to_meters(raw_value):
            if raw_value is None or math.isnan(raw_value):
                return 2.0
            return max(0.05, min(2.0, raw_value / 1000.0))

        left = 2.0
        right = 2.0
        
        if self.ds_left:
            left = raw_to_meters(self.ds_left.getValue())
        if self.ds_right:
            right = raw_to_meters(self.ds_right.getValue())

        return {"left": left, "right": right}

    def _process_front_sensors(self):
        """Process front distance sensors."""
        def raw_to_meters(raw_value):
            if raw_value is None or math.isnan(raw_value):
                return 2.0
            return max(0.05, min(2.0, raw_value / 1000.0))

        front = 2.0
        front_left = 2.0
        front_right = 2.0

        if self.ds_front:
            front = raw_to_meters(self.ds_front.getValue())
        if self.ds_front_left:
            front_left = raw_to_meters(self.ds_front_left.getValue())
        if self.ds_front_right:
            front_right = raw_to_meters(self.ds_front_right.getValue())

        return {"front": front, "front_left": front_left, "front_right": front_right}

    def _check_obstacle_ahead(self, lidar_front, threshold=0.5):
        """Check for obstacles ahead using multiple sources."""
        if lidar_front < threshold:
            return True

        left, right, bottom, top = self._wall_clearances()
        if min(left, right, bottom, top) < threshold:
            return True

        x, y, yaw = self.pose
        for ox, oy, radius in KNOWN_OBSTACLES:
            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy)
            obs_angle = wrap_angle(math.atan2(dy, dx) - yaw)

            if abs(obs_angle) < math.radians(40) and dist < (threshold + radius + 0.1):
                return True

        return False

    def _search_navigation(self, lidar_info, dt):
        """Lawnmower search navigation."""
        front_dist = lidar_info["front"]
        left_dist = lidar_info["left"]
        right_dist = lidar_info["right"]

        OBSTACLE_THRESHOLD = 0.50
        FORWARD_SPEED = 0.15
        
        left_wall, right_wall, bottom_wall, top_wall = self._wall_clearances()
        min_wall = min(left_wall, right_wall, bottom_wall, top_wall)
        
        obstacle_ahead = self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD)

        if min_wall < 0.25:
            return -0.08, 0.0, 0.0

        if self.search_state == "forward":
            if obstacle_ahead:
                if left_dist > right_dist:
                    self.search_direction = 1
                else:
                    self.search_direction = -1
                self.search_state = "turn"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            else:
                omega_correct = 0.0
                if left_dist < 0.40:
                    omega_correct = -0.15
                elif right_dist < 0.40:
                    omega_correct = 0.15
                
                return FORWARD_SPEED, 0.0, omega_correct

        elif self.search_state == "turn":
            self.turn_progress += dt
            if self.turn_progress >= 2.5:
                self.search_state = "forward"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            
            if not self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD + 0.2) and self.turn_progress > 1.0:
                self.search_state = "forward"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0

            return 0.10, 0.0, 0.4 * self.search_direction

        self.search_state = "forward"
        return 0.0, 0.0, 0.0

    def _start_grasp(self):
        """Begin grasp sequence."""
        self.mode = "grasp"
        self.stage = 0
        self.stage_timer = 0.0
        self.base.reset()
        print(f"[GRASP] Starting capture - color={self.current_color}")

    def _start_drop(self):
        """Begin drop sequence."""
        self.mode = "drop"
        self.stage = 0
        self.stage_timer = 0.0
        self.base.reset()

    def _handle_grasp(self, dt):
        """Handle grasp state machine."""
        self.stage_timer += dt

        if not hasattr(self, '_grasp_forward_time'):
            dist = self.locked_cube_distance if self.locked_cube_distance else 0.25
            forward_needed = dist - 0.12
            forward_needed = max(0.05, min(0.25, forward_needed))
            self._grasp_forward_time = forward_needed / 0.04
            self._grasp_samples = []
            print(f"[GRASP] Forward distance: {forward_needed:.2f}m ({self._grasp_forward_time:.1f}s) (cube at {dist:.2f}m)")

        if self.stage == 0:
            self.gripper.release()
            self.arm.set_height(Arm.RESET)
            if self.stage_timer >= 1.2:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 1:
            self.arm.set_height(Arm.FRONT_FLOOR)
            if self.stage_timer >= 2.0:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 2:
            omega_cmd = 0.0
            locked_recognition = self._process_recognition(
                lock_color=self.current_color,
                lock_angle=self.locked_cube_angle
            )
            if locked_recognition:
                cam_angle = locked_recognition["angle"]
                omega_cmd = -cam_angle * 0.8
                omega_cmd = max(-0.3, min(0.3, omega_cmd))

            self._safe_move(0.04, 0.0, omega_cmd)
            if self.stage_timer >= self._grasp_forward_time:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0
                print("[GRASP] Stopped, closing gripper...")

        elif self.stage == 3:
            self.gripper.grip()
            if self.stage_timer >= 1.2:
                self.stage += 1
                self.stage_timer = 0.0
                self._grasp_samples = []

        elif self.stage == 4:
            self.arm.set_height(Arm.FRONT_PLATE)
            self.gripper.grip()

            left, right = self.gripper.finger_positions()
            if left is not None:
                self._grasp_samples.append(left)
            if right is not None:
                self._grasp_samples.append(right)

            if self.stage_timer >= 1.5:
                has_obj = self.gripper.has_object(threshold=0.003)

                if self._grasp_samples:
                    avg_pos = sum(self._grasp_samples) / len(self._grasp_samples)
                    max_pos = max(self._grasp_samples) if self._grasp_samples else 0
                    print(f"[GRASP] Verification: avg={avg_pos:.4f}, max={max_pos:.4f}, has_object={has_obj}")
                else:
                    print(f"[GRASP] No samples, has_object={has_obj}")

                self.base.reset()
                self._grasp_verified = has_obj
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 5:
            has_obj = getattr(self, '_grasp_verified', True)

            if hasattr(self, '_grasp_forward_time'):
                del self._grasp_forward_time
            if hasattr(self, '_grasp_samples'):
                del self._grasp_samples
            if hasattr(self, '_grasp_verified'):
                del self._grasp_verified

            if has_obj:
                self.collected += 1
                print(f"[GRASP] Cube captured! Total: {self.collected}/{self.max_cubes}")

                self.locked_cube_distance = None
                if self.active_goal:
                    cell = self.grid.world_to_cell(*self.active_goal)
                    if cell:
                        self.grid.set(cell[0], cell[1], OccupancyGrid.FREE, overwrite_static=False)
                        self._path_dirty = True

                gt = self._get_ground_truth_pose()
                if gt:
                    self.pose = gt
                    print(f"[TO_BOX] Ground truth sync: ({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.1f}°)")

                self.mode = "to_box"
                self.stage = 0
                self.stage_timer = 0.0
                color_key = (self.current_color or "").lower()
                target_box = self.box_positions.get(color_key, None)
                if target_box:
                    dist_to_box = math.hypot(target_box[0] - self.pose[0], target_box[1] - self.pose[1])
                    angle_to_box = wrap_angle(math.atan2(target_box[1] - self.pose[1], target_box[0] - self.pose[0]) - self.pose[2])
                    print(f"[TO_BOX] Going to box {color_key.upper()} at {target_box}")
                    print(f"[TO_BOX] Current pose: ({self.pose[0]:.2f}, {self.pose[1]:.2f}, {math.degrees(self.pose[2]):.1f}°) dist={dist_to_box:.2f}m ang={math.degrees(angle_to_box):.1f}°")
                    self._set_goal(target_box)
                else:
                    fallback = list(self.box_positions.values())[0]
                    print(f"[TO_BOX] Color '{self.current_color}' not found, using fallback {fallback}")
                    self._set_goal(fallback)
            else:
                print("[GRASP] FAILED - empty gripper, reversing to retry")
                self.gripper.release()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 6:
            self.arm.set_height(Arm.FRONT_PLATE)
            self._safe_move(-0.06, 0.0, 0.0)
            if self.stage_timer >= 2.0:
                self.base.reset()
                print("[GRASP] Reverse complete, returning to search")
                self.mode = "search"
                self.stage = 0
                self.stage_timer = 0.0
                self.current_color = None
                self.locked_cube_angle = None
                self.locked_cube_distance = None
                self.current_target = None
                self.active_goal = None
                self._waypoints = []
                self._path_dirty = True

    def _handle_drop(self, dt):
        """Handle drop state machine."""
        self.stage_timer += dt

        if self.stage == 0:
            self.arm.set_height(Arm.FRONT_FLOOR)
            self.gripper.grip()
            if self.stage_timer >= 2.0:
                self.stage += 1
                self.stage_timer = 0.0
                print("[DROP] Arm lowered, advancing over box...")

        elif self.stage == 1:
            self._safe_move(0.03, 0.0, 0.0)
            self.gripper.grip()
            if self.stage_timer >= 1.5:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0
                print("[DROP] Over box, releasing cube...")

        elif self.stage == 2:
            self.gripper.release()
            if self.stage_timer >= 1.0:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 3:
            self._safe_move(-0.06, 0.0, 0.0)
            if self.stage_timer >= 1.0:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 4:
            self.arm.set_height(Arm.RESET)
            if self.stage_timer >= 1.5:
                self.stage += 1
                self.stage_timer = 0.0
                print("[DROP] Retreating...")

        elif self.stage == 5:
            # Short retreat from box - just enough to clear the box edge
            self._safe_move(-0.10, 0.0, 0.0)
            if self.stage_timer >= 1.5:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0
                print(f"[DROP] Retreating done. Preparing for return navigation...")

        elif self.stage == 6:
            color_key = (self.current_color or "").lower()
            if color_key in self.delivered:
                self.delivered[color_key] += 1

            # Sync ground truth after deposit
            gt = self._get_ground_truth_pose()
            if gt:
                self.pose = gt

            print(f"[DROP] Cube deposited in {self.current_color.upper()} box! (delivered: {self.delivered})")
            print(f"[DROP] Current position: ({self.pose[0]:.2f}, {self.pose[1]:.2f})")

            # CRITICAL FIX: Instead of going directly to search (which fails near obstacles),
            # use return_to_spawn mode which has robust waypoint-based navigation.
            # This ensures the robot safely navigates back to spawn area before searching.
            print(f"[DROP] Navigating back to spawn for next search...")
            
            # Save the box color for return route calculation
            self._last_box_color = color_key
            
            # Transition to return_to_spawn mode (uses waypoint navigation)
            self.mode = "return_to_spawn"
            
            # Clear navigation state
            self.current_target = None
            self.active_goal = None
            self._waypoints = []
            self._path_dirty = True
            self.locked_cube_angle = None
            self.locked_cube_distance = None
            
            # Keep current_color for route calculation, will be cleared when reaching spawn
            self.base.reset()
            self.stage = 0

    def _handle_return_to_spawn(self, dt, lidar_info, rear_info, lateral_info, front_info):
        """
        Handle return to spawn navigation using strategic waypoints.
        
        Key fix: Use waypoint-following navigation similar to TO_BOX mode,
        with car-like constraints to prevent 180-degree spins.
        """
        self.arm.set_height(Arm.RESET)
        self.gripper.release()

        current_time = self.robot.getTime()

        # Initialization
        if not hasattr(self, '_return_phase'):
            self._return_phase = 0  # 0=retreat, 1=turn-in-place, 2=navigate
            self._return_start = current_time
            self._return_log_time = 0.0
            self._return_turn_start = None
            self._return_last_angle = None

            print("[RETURN] Phase 0: Retreating from box...")

        # Sensor consolidation
        rear_obs = min(lidar_info["rear"], rear_info.get("rear", 2.0))
        rl_obs = rear_info.get("rear_left", 2.0)
        rr_obs = rear_info.get("rear_right", 2.0)
        left_obs = min(lidar_info["left"], lateral_info.get("left", 2.0))
        right_obs = min(lidar_info["right"], lateral_info.get("right", 2.0))
        front_obs = min(lidar_info["front"], front_info["front"])
        fl_obs = front_info.get("front_left", 2.0)
        fr_obs = front_info.get("front_right", 2.0)

        min_rear = min(rear_obs, rl_obs, rr_obs)
        min_front = min(front_obs, fl_obs, fr_obs)

        dist_to_spawn, angle_to_spawn = self._distance_to_point(SPAWN_POSITION)

        # ===== PHASE 0: RETREAT FROM BOX (smart - check if already clear) =====
        if self._return_phase == 0:
            phase_elapsed = current_time - self._return_start

            if not hasattr(self, '_retreat_start_pos'):
                self._retreat_start_pos = (self.pose[0], self.pose[1])
                self._retreat_substage = 0  # 0=check/reverse, 1=turn+forward, 2=done
                
                # SMART CHECK: If already far from box, skip retreat entirely
                color_key = getattr(self, '_last_box_color', None)
                if color_key:
                    box_pos = BOX_POSITIONS.get(color_key)
                    if box_pos:
                        dist_to_box = math.hypot(self.pose[0] - box_pos[0], self.pose[1] - box_pos[1])
                        if dist_to_box > 0.50:
                            # Already retreated during DROP, skip to route calculation
                            print(f"[RETURN] Already {dist_to_box:.2f}m from box, skipping retreat")
                            self._retreat_substage = 2

            retreat_dist = math.hypot(
                self.pose[0] - self._retreat_start_pos[0],
                self.pose[1] - self._retreat_start_pos[1]
            )

            rear_clear = min_rear > 0.25 and rl_obs > 0.25 and rr_obs > 0.25
            front_clear = min_front > 0.35

            # Substage 0: Try to reverse (if needed)
            if self._retreat_substage == 0:
                if retreat_dist < 0.30 and phase_elapsed < 2.0 and rear_clear:
                    self._safe_move(-0.12, 0.0, 0.0)
                    return
                else:
                    # Rear blocked or retreated enough, try turning + forward
                    if retreat_dist < 0.15 and front_clear:
                        # Didn't retreat enough, try turning and moving forward
                        self._retreat_substage = 1
                        self._retreat_turn_start = current_time
                        color_key = getattr(self, '_last_box_color', 'red')
                        # Turn right for red (go south), left for others
                        self._retreat_turn_dir = -0.4 if color_key == 'red' else 0.4
                        print("[RETURN] Rear blocked, turning to escape...")
                    else:
                        self._retreat_substage = 2  # Done

            # Substage 1: Turn and move forward to escape
            if self._retreat_substage == 1:
                turn_elapsed = current_time - self._retreat_turn_start
                if turn_elapsed < 2.0 and front_clear:
                    # Turn while moving forward slightly
                    self._safe_move(0.06, 0.0, self._retreat_turn_dir)
                    return
                else:
                    self._retreat_substage = 2  # Done

            # Substage 2: Finish retreat phase
            if self._retreat_substage == 2:
                # Cleanup retreat state
                for attr in ['_retreat_start_pos', '_retreat_substage', 
                             '_retreat_turn_start', '_retreat_turn_dir']:
                    if hasattr(self, attr):
                        delattr(self, attr)

                # Sync ground truth AFTER retreat (critical fix!)
                gt = self._get_ground_truth_pose()
                if gt:
                    self.pose = gt

                # NOW calculate the route from actual position
                color_key = getattr(self, '_last_box_color', 'red')
                self._return_waypoints = get_return_route((self.pose[0], self.pose[1]), color_key)
                self._return_waypoint_idx = 0
                
                # Skip waypoints that are already behind or too close
                self._skip_passed_waypoints()
                
                print(f"[RETURN] Retreated {retreat_dist:.2f}m. Current pos: ({self.pose[0]:.2f}, {self.pose[1]:.2f})")
                print(f"[RETURN] Strategic route calculated: {len(self._return_waypoints)} waypoints")
                for i, wp in enumerate(self._return_waypoints):
                    marker = " <--" if i == self._return_waypoint_idx else ""
                    print(f"  [{i+1}] ({wp[0]:.2f}, {wp[1]:.2f}){marker}")
                
                print("[RETURN] Phase 1: Turn-in-place check...")
                self._return_phase = 1
                self._return_start = current_time
            return

        # ===== PHASE 1: TURN-IN-PLACE (if needed) =====
        if self._return_phase == 1:
            # Get current waypoint
            if self._return_waypoint_idx < len(self._return_waypoints):
                target_wp = self._return_waypoints[self._return_waypoint_idx]
            else:
                target_wp = SPAWN_POSITION

            dist_to_wp, angle_to_wp = self._distance_to_point(target_wp)

            # Only need to turn if angle > 80 degrees
            if abs(angle_to_wp) > math.radians(80):
                # Initialize on first iteration - CHOOSE DIRECTION ONCE AND COMMIT
                if self._return_turn_start is None:
                    self._return_turn_start = current_time
                    self._return_last_angle = angle_to_wp
                    self._return_turn_attempts = 0

                    # For near-180° turns, the sign is ambiguous. 
                    # COMMIT to a direction based on which way is shorter OR use box position.
                    # For RED box (east), robot faces east (~0°), waypoint is west (~180°)
                    # Turning RIGHT (going through south) is often clearer path
                    color_key = getattr(self, '_last_box_color', 'unknown')
                    
                    if abs(angle_to_wp) > math.radians(150):
                        # Near 180° - pick direction based on box color (known clear path)
                        if color_key == 'red':
                            # From RED box, turn RIGHT (south then west)
                            self._return_committed_dir = -1
                        elif color_key == 'green':
                            # From GREEN box, turn RIGHT (south then west)  
                            self._return_committed_dir = -1
                        elif color_key == 'blue':
                            # From BLUE box, turn LEFT (north then west)
                            self._return_committed_dir = 1
                        else:
                            # Default: turn right
                            self._return_committed_dir = -1
                    else:
                        # Normal case: turn toward the waypoint
                        self._return_committed_dir = 1 if angle_to_wp > 0 else -1

                    print(f"[RETURN] Starting turn {'RIGHT' if self._return_committed_dir < 0 else 'LEFT'} (angle={math.degrees(angle_to_wp):.0f}°, from {color_key})")
                
                turn_elapsed = current_time - self._return_turn_start

                # CRITICAL FIX: Do NOT switch directions for near-180° turns!
                # The angle will cross ±180° boundary, but we must keep turning the same way.
                # Only allow direction re-evaluation when angle has become small (< 90°)
                if abs(angle_to_wp) < math.radians(90):
                    # Now angle is small enough that sign is reliable
                    current_dir = 1 if angle_to_wp > 0 else -1
                    if current_dir != self._return_committed_dir:
                        # We overshot, switch direction
                        self._return_committed_dir = current_dir
                        print(f"[RETURN] Fine-tuning: now turning {'LEFT' if current_dir > 0 else 'RIGHT'} (angle={math.degrees(angle_to_wp):.0f}°)")

                # Stuck detection: if turning for >8s without progress
                if turn_elapsed > 8.0:
                    self._return_turn_attempts += 1

                    if self._return_turn_attempts >= 2:
                        # Give up on this waypoint, skip it
                        print(f"[RETURN] Turn timeout! Skipping waypoint {self._return_waypoint_idx+1}")
                        self._return_waypoint_idx += 1
                        self._return_turn_start = None
                        self._return_last_angle = None
                        self._return_turn_attempts = 0
                        if hasattr(self, '_return_committed_dir'):
                            delattr(self, '_return_committed_dir')
                        self._skip_passed_waypoints()
                        return
                    else:
                        # Reset timer but KEEP the same direction (don't re-evaluate for 180° turns)
                        print(f"[RETURN] Turn attempt {self._return_turn_attempts}, continuing same direction...")
                        self._return_turn_start = current_time
                
                # Use committed direction with obstacle checking
                omega_speed = 0.50  # Slightly faster
                omega = self._return_committed_dir * omega_speed
                
                # Check if direction is blocked - slow down but don't reverse
                if omega > 0 and left_obs < 0.20:
                    omega = omega_speed * 0.4
                elif omega < 0 and right_obs < 0.20:
                    omega = -omega_speed * 0.4
                
                # Move forward slightly while turning to help clear obstacles
                fwd = 0.05 if min_front > 0.40 else (-0.03 if min_rear > 0.30 else 0.0)
                
                self._safe_move(fwd, 0.0, omega)
                
                # Log progress (less frequently to reduce spam)
                if current_time - self._return_log_time > 1.5:
                    dir_str = "L" if omega > 0 else "R"
                    print(f"[RETURN] Turning {dir_str}: angle={math.degrees(angle_to_wp):.0f}° (elapsed {turn_elapsed:.1f}s)")
                    self._return_log_time = current_time
                return
            else:
                # Angle is now small enough, proceed to navigation
                self._return_turn_start = None
                self._return_last_angle = None
                for attr in ['_return_turn_attempts', '_return_committed_dir']:
                    if hasattr(self, attr):
                        delattr(self, attr)
                print("[RETURN] Phase 2: Navigating to spawn...")
                self._return_phase = 2
                self._return_start = current_time

        # ===== PHASE 2: WAYPOINT NAVIGATION (similar to TO_BOX) =====
        if self._return_phase == 2:
            # Arrived at spawn?
            if dist_to_spawn < 0.60:
                print(f"[RETURN] Arrived at spawn! Pose: ({self.pose[0]:.2f}, {self.pose[1]:.2f})")
                print("[RETURN] ========== STARTING NEW SEARCH ==========")

                gt = self._get_ground_truth_pose()
                if gt:
                    self.pose = gt

                # Cleanup all return state
                for attr in ['_return_phase', '_return_start', '_return_log_time',
                             '_return_waypoints', '_return_waypoint_idx', '_last_box_color',
                             '_return_turn_start', '_return_last_angle', '_return_nav_last_pos',
                             '_return_nav_stuck_time', '_return_turn_attempts', '_return_committed_dir']:
                    if hasattr(self, attr):
                        delattr(self, attr)

                # Clear navigation state for fresh search
                self.current_color = None
                self.current_target = None
                self.active_goal = None
                self._waypoints = []
                self._path_dirty = True
                self.search_state = "forward"
                self.turn_progress = 0.0

                self.mode = "search"
                return

            # Current waypoint
            if self._return_waypoint_idx < len(self._return_waypoints):
                target_wp = self._return_waypoints[self._return_waypoint_idx]
            else:
                target_wp = SPAWN_POSITION

            dist_to_wp, angle_to_wp = self._distance_to_point(target_wp)

            # Advance waypoint if reached
            wp_threshold = 0.45
            if dist_to_wp < wp_threshold and self._return_waypoint_idx < len(self._return_waypoints):
                self._return_waypoint_idx += 1
                # Skip any other passed waypoints
                self._skip_passed_waypoints()
                if self._return_waypoint_idx < len(self._return_waypoints):
                    next_wp = self._return_waypoints[self._return_waypoint_idx]
                    print(f"[RETURN] Waypoint {self._return_waypoint_idx}/{len(self._return_waypoints)} reached -> next: ({next_wp[0]:.2f}, {next_wp[1]:.2f})")
                else:
                    print("[RETURN] All waypoints reached, heading to spawn")
                # Reset stuck tracking
                if hasattr(self, '_return_nav_last_pos'):
                    delattr(self, '_return_nav_last_pos')
                if hasattr(self, '_return_nav_stuck_time'):
                    delattr(self, '_return_nav_stuck_time')
                return

            # Stuck detection in navigation
            if not hasattr(self, '_return_nav_last_pos'):
                self._return_nav_last_pos = (self.pose[0], self.pose[1])
                self._return_nav_stuck_time = current_time
            else:
                moved = math.hypot(
                    self.pose[0] - self._return_nav_last_pos[0],
                    self.pose[1] - self._return_nav_last_pos[1]
                )
                if moved > 0.15:  # Moved more than 15cm
                    self._return_nav_last_pos = (self.pose[0], self.pose[1])
                    self._return_nav_stuck_time = current_time
                elif current_time - self._return_nav_stuck_time > 8.0:
                    # Stuck for 8 seconds, skip to next waypoint
                    print(f"[RETURN] Navigation stuck! Skipping waypoint {self._return_waypoint_idx + 1}")
                    self._return_waypoint_idx += 1
                    self._skip_passed_waypoints()
                    self._return_nav_last_pos = (self.pose[0], self.pose[1])
                    self._return_nav_stuck_time = current_time
                    return

            # Periodic logging
            if current_time - self._return_log_time > 2.0:
                wp_info = f"WP {self._return_waypoint_idx+1}/{len(self._return_waypoints)}" if self._return_waypoint_idx < len(self._return_waypoints) else "SPAWN"
                print(f"[RETURN] Pos: ({self.pose[0]:.2f}, {self.pose[1]:.2f}) -> {wp_info} dist={dist_to_wp:.2f}m ang={math.degrees(angle_to_wp):.0f}°")
                self._return_log_time = current_time

            # ===== CAR-LIKE NAVIGATION (same as TO_BOX) =====
            
            # If angle > 100 degrees, go back to Phase 1 (turn-in-place)
            # This handles cases where we skipped waypoints and now face wrong direction
            if abs(angle_to_wp) > math.radians(100):
                print(f"[RETURN] Waypoint behind ({math.degrees(angle_to_wp):.0f}°), returning to turn-in-place")
                self._return_phase = 1
                self._return_turn_start = None
                self._return_last_angle = None
                return
            
            # If angle still significant (> 70 degrees), rotate while moving slowly
            if abs(angle_to_wp) > math.radians(70):
                omega = 0.40 if angle_to_wp > 0 else -0.40
                fwd = 0.03 if min_front > 0.40 else 0.0
                self._safe_move(fwd, 0.0, omega)
                return

            left_blocked = left_obs < 0.20
            right_blocked = right_obs < 0.20

            cmd_speed = 0.0
            cmd_omega = 0.0

            # Priority 1: Emergency stop
            if min_front < 0.22:
                rear_all_clear = min_rear > 0.30
                if rear_all_clear:
                    cmd_speed = -0.08
                    if not left_blocked and (right_blocked or left_obs > right_obs):
                        cmd_omega = 0.4
                    elif not right_blocked:
                        cmd_omega = -0.4
                else:
                    cmd_speed = 0.0
                    if not left_blocked:
                        cmd_omega = 0.5
                    elif not right_blocked:
                        cmd_omega = -0.5

            # Priority 2: Front danger
            elif min_front < 0.35:
                rear_all_clear = min_rear > 0.30
                if rear_all_clear:
                    cmd_speed = -0.06
                    if not left_blocked and (right_blocked or left_obs > right_obs):
                        cmd_omega = 0.4
                    elif not right_blocked:
                        cmd_omega = -0.4
                else:
                    cmd_speed = 0.02
                    if not left_blocked and (right_blocked or left_obs > right_obs):
                        cmd_omega = 0.4
                    elif not right_blocked:
                        cmd_omega = -0.4

            # Priority 3: Front warning
            elif min_front < 0.55:
                if left_blocked:
                    cmd_omega = -0.35
                elif right_blocked:
                    cmd_omega = 0.35
                elif left_obs > right_obs + 0.1:
                    cmd_omega = 0.25
                elif right_obs > left_obs + 0.1:
                    cmd_omega = -0.25
                else:
                    cmd_omega = -angle_to_wp * 0.8
                    cmd_omega = max(-0.35, min(0.35, cmd_omega))
                
                cmd_speed = 0.06

            # Priority 4: Normal navigation
            else:
                cmd_omega = -angle_to_wp * 1.2
                cmd_omega = max(-0.40, min(0.40, cmd_omega))
                
                # Lateral repulsion
                if left_obs < 0.35:
                    cmd_omega -= 0.15
                if right_obs < 0.35:
                    cmd_omega += 0.15
                cmd_omega = max(-0.45, min(0.45, cmd_omega))
                
                cmd_speed = 0.12
                turn_penalty = 1.0 - min(0.3, abs(cmd_omega) / 0.45)
                cmd_speed *= turn_penalty

            # Final safety check: never turn into blocked side
            if left_blocked and cmd_omega > 0:
                cmd_omega = -0.35
                cmd_speed = min(cmd_speed, 0.03)
            if right_blocked and cmd_omega < 0:
                cmd_omega = 0.35
                cmd_speed = min(cmd_speed, 0.03)

            # Both sides blocked
            if left_blocked and right_blocked:
                cmd_omega = 0.0
                if min_front > 0.35:
                    cmd_speed = 0.04
                elif min_rear > 0.30:
                    cmd_speed = -0.06
                else:
                    cmd_speed = 0.0

            self._safe_move(cmd_speed, 0.0, cmd_omega)

    def run(self):
        """Main control loop."""
        print("[RUN] YouBot controller started")

        # Move forward at spawn to clear wall
        print("[INIT] Moving forward to clear spawn wall...")
        for _ in range(50):
            if self.robot.step(self.time_step) == -1:
                return
            self.base.move(0.12, 0.0, 0.0)
        self.base.reset()

        gt = self._get_ground_truth_pose()
        if gt:
            self.pose = gt
            print(f"[INIT] Spawn complete. Pose: ({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.1f}°)")
        else:
            print("[INIT] Spawn complete (no ground truth)")

        while self.robot.step(self.time_step) != -1:
            dt = self.time_step / 1000.0

            if self.collected >= self.max_cubes:
                print("[DONE] All cubes collected!")
                self.base.reset()
                break

            # Odometry with periodic sync
            vx_odo, vy_odo, omega_odo = self.base.compute_odometry(dt)
            self._integrate_pose(vx_odo, vy_odo, omega_odo, dt)

            if not hasattr(self, '_gt_sync_timer'):
                self._gt_sync_timer = 0.0
            self._gt_sync_timer += dt

            sync_interval = 0.5 if self.mode in ("to_box", "return_to_spawn") else 2.0
            pose_invalid = any(math.isnan(v) for v in self.pose)
            if pose_invalid or self._gt_sync_timer >= sync_interval:
                gt = self._get_ground_truth_pose()
                if gt:
                    if pose_invalid:
                        print(f"[POSE] Invalid pose, syncing: ({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.1f}°)")
                    self.pose = gt
                self._gt_sync_timer = 0.0

            # Sensors
            lidar_info = self._process_lidar()
            rear_info = self._process_rear_sensors()
            lateral_info = self._process_lateral_sensors()
            front_info = self._process_front_sensors()
            
            if "rear" in lidar_info:
                rear_info["rear"] = min(rear_info["rear"], lidar_info["rear"])
            
            self._update_grid_from_lidar(lidar_info)

            lock_color = self.current_color if self.mode == "approach" else None
            lock_angle = self.locked_cube_angle if self.mode == "approach" else None
            recognition = self._process_recognition(lock_color=lock_color, lock_angle=lock_angle)

            # MCP Bridge
            if self.mcp:
                left_f, right_f = self.gripper.finger_positions()
                gripper_closed = (left_f or 0) < 0.005 and (right_f or 0) < 0.005
                has_cube = self.gripper.has_object(threshold=0.003) if not gripper_closed else False

                recog_info = None
                if recognition:
                    recog_info = {"color": recognition["color"], "distance": round(recognition["distance"], 3), "angle": round(recognition["angle"], 2)}

                self.mcp.publish({
                    "pose": self.pose,
                    "mode": self.mode,
                    "collected": self.collected,
                    "max_cubes": self.max_cubes,
                    "delivered": self.delivered,
                    "current_target": self.current_color,
                    "lidar": {"front": lidar_info["front"], "rear": lidar_info["rear"], "left": lidar_info["left"], "right": lidar_info["right"]},
                    "distance_sensors": front_info,
                    "gripper": {"left": left_f, "right": right_f, "closed": gripper_closed, "has_cube": has_cube},
                    "recognition": recog_info,
                    "arm_state": getattr(self, 'arm_state', 'unknown'),
                })
                self.mcp.get_command()

                if hasattr(self, '_mcp_frame_counter'):
                    self._mcp_frame_counter += 1
                else:
                    self._mcp_frame_counter = 0
                if self._mcp_frame_counter % 50 == 0:
                    self.mcp.save_camera_frame(self.camera)

            # ===== MODE: SEARCH =====
            if self.mode == "search":
                left, right, bottom, top = self._wall_clearances()
                near_wall = min(left, right, bottom, top) < 0.35

                if recognition:
                    self.current_color = recognition["color"]
                    self.locked_cube_angle = recognition["angle"]
                    self.mode = "approach"
                    self.lost_cube_timer = 0.0
                    print(f"[SEARCH] Cube detected (color={self.current_color}) dist={recognition['distance']:.2f}m, starting approach")
                    continue

                vx_cmd, vy_cmd, omega_cmd = self._search_navigation(lidar_info, dt)
                vx_cmd, vy_cmd, omega_cmd = self._clamp_cmds(vx_cmd, vy_cmd, omega_cmd if not near_wall else 0.0)
                vx_cmd, vy_cmd = self._enforce_boundary_safety(vx_cmd, vy_cmd)
                self._safe_move(vx_cmd, vy_cmd, omega_cmd)
                continue

            # ===== MODE: APPROACH =====
            if self.mode == "approach":
                front_obstacle = min(
                    front_info["front"],
                    front_info["front_left"],
                    front_info["front_right"],
                    lidar_info["front"]
                )

                if front_obstacle < 0.20:
                    print(f"[APPROACH] FRONT OBSTACLE! dist={front_obstacle:.2f}m, avoiding...")

                    if front_info["front_left"] > front_info["front_right"]:
                        vy_cmd = 0.08
                        omega_cmd = 0.2
                    else:
                        vy_cmd = -0.08
                        omega_cmd = -0.2

                    self._safe_move(-0.04, vy_cmd, omega_cmd)

                    if not hasattr(self, '_approach_obstacle_count'):
                        self._approach_obstacle_count = 0
                    self._approach_obstacle_count += 1

                    if self._approach_obstacle_count > 30:
                        print("[APPROACH] Could not avoid, returning to search")
                        self.mode = "search"
                        self.locked_cube_angle = None
                        self.locked_cube_distance = None
                        self.current_color = None
                        self._approach_obstacle_count = 0
                    continue
                else:
                    if hasattr(self, '_approach_obstacle_count'):
                        self._approach_obstacle_count = 0

                if recognition:
                    self.lost_cube_timer = 0.0
                    cam_dist = recognition["distance"]
                    cam_angle = recognition["angle"]

                    self.locked_cube_distance = cam_dist
                    self.locked_cube_angle = cam_angle

                    grasp_distance = 0.32
                    grasp_angle_max = math.radians(10)

                    if cam_dist < grasp_distance and abs(cam_angle) < grasp_angle_max:
                        print(f"[APPROACH] Cube at {cam_dist:.2f}m, angle={math.degrees(cam_angle):.1f}°, starting grasp")
                        self._start_grasp()
                        continue
                    elif cam_dist < grasp_distance and abs(cam_angle) >= grasp_angle_max:
                        omega_cmd = -cam_angle * 1.0
                        omega_cmd = max(-0.4, min(0.4, omega_cmd))
                        self._safe_move(0.0, 0.0, omega_cmd)
                        self._log_throttled("approach_align_close", f"[APPROACH] Close but misaligned: angle={math.degrees(cam_angle):.1f}°", 1.0)
                        continue

                    angle_threshold = math.radians(5)

                    if abs(cam_angle) > angle_threshold:
                        omega_cmd = -cam_angle * 0.8
                        omega_cmd = max(-0.3, min(0.3, omega_cmd))
                        vx_cmd = 0.0
                        self._log_throttled("approach_align", f"[APPROACH] Aligning: angle={math.degrees(cam_angle):.1f}°, dist={cam_dist:.2f}m", 1.0)
                    else:
                        vx_cmd = 0.08
                        omega_cmd = -cam_angle * 1.2
                        omega_cmd = max(-0.2, min(0.2, omega_cmd))
                        self._log_throttled("approach_advance", f"[APPROACH] Advancing: dist={cam_dist:.2f}m, angle={math.degrees(cam_angle):.1f}°", 1.0)

                    self._safe_move(vx_cmd, 0.0, omega_cmd)
                    continue
                else:
                    self.lost_cube_timer += dt

                    if self.locked_cube_distance is not None and self.locked_cube_distance < 0.35:
                        if self.lost_cube_timer > 0.3:
                            print(f"[APPROACH] Cube lost at {self.locked_cube_distance:.2f}m, starting grasp")
                            self._start_grasp()
                            continue

                    if self.lost_cube_timer > 4.0:
                        print("[APPROACH] Timeout - returning to search")
                        self.mode = "search"
                        self.locked_cube_angle = None
                        self.locked_cube_distance = None
                        self.current_color = None
                        self.lost_cube_timer = 0.0
                        continue

                    if self.locked_cube_angle is not None:
                        omega_recovery = -self.locked_cube_angle * 0.3
                        omega_recovery = max(-0.2, min(0.2, omega_recovery))
                    else:
                        omega_recovery = 0.0
                    self._safe_move(0.08, 0.0, omega_recovery)
                    continue

            # ===== MODE: GRASP =====
            if self.mode == "grasp":
                self._handle_grasp(dt)
                continue

            # ===== MODE: TO_BOX =====
            if self.mode == "to_box":
                self.gripper.grip()
                self.arm.set_height(Arm.RESET)
                
                if not hasattr(self, 'tobox_state'):
                    self.tobox_state = 0
                    self._tobox_maneuver_timer = 0.0
                    self._route_waypoints = None
                    self._current_waypoint_idx = 0

                if not self.active_goal:
                    print("[TO_BOX] No goal defined, returning to search")
                    self.mode = "search"
                    continue

                if self._route_waypoints is None:
                    self._route_waypoints = get_route_to_box(
                        (self.pose[0], self.pose[1]), 
                        self.current_color
                    )
                    self._current_waypoint_idx = 0
                    print("[TO_BOX] ===== ROUTE CALCULATED =====")
                    print(f"[TO_BOX] Color: {self.current_color} | Destination: {BOX_POSITIONS.get(self.current_color)}")
                    print(f"[TO_BOX] Waypoints ({len(self._route_waypoints)}):")
                    for i, wp in enumerate(self._route_waypoints):
                        print(f"  [{i+1}] ({wp[0]:.2f}, {wp[1]:.2f})")
                    print("[TO_BOX] =============================")
                
                if not hasattr(self, '_last_pos_log_time'):
                    self._last_pos_log_time = 0.0
                current_time = self.robot.getTime()
                if current_time - self._last_pos_log_time > 2.0:
                    wp_idx = self._current_waypoint_idx
                    wp_total = len(self._route_waypoints)
                    wp_name = f"WP{wp_idx+1}/{wp_total}" if wp_idx < wp_total else "BOX"
                    print(f"[TO_BOX] Pos: ({self.pose[0]:.2f}, {self.pose[1]:.2f}, {math.degrees(self.pose[2]):.0f}°) -> {wp_name}")
                    self._last_pos_log_time = current_time

                front_obs = min(lidar_info["front"], front_info["front"])
                fl_obs = min(lidar_info["front"], front_info.get("front_left", 2.0))
                fr_obs = min(lidar_info["front"], front_info.get("front_right", 2.0))
                rear_obs = min(lidar_info["rear"], rear_info.get("rear", 2.0))
                rear_min = min(rear_obs, rear_info.get("rear_left", 2.0), rear_info.get("rear_right", 2.0))
                left_obs = min(lidar_info["left"], lateral_info.get("left", 2.0))
                right_obs = min(lidar_info["right"], lateral_info.get("right", 2.0))

                if self._current_waypoint_idx < len(self._route_waypoints):
                    current_target = self._route_waypoints[self._current_waypoint_idx]
                else:
                    current_target = self.active_goal
                
                dist_to_waypoint, angle_to_waypoint = self._distance_to_point(current_target)
                dist_to_box, angle_to_box = self._distance_to_point(self.active_goal)
                
                is_final_waypoint = (self._current_waypoint_idx >= len(self._route_waypoints) - 1)
                is_pre_final = (self._current_waypoint_idx == len(self._route_waypoints) - 2)
                
                if is_final_waypoint:
                    waypoint_threshold = 0.50
                elif is_pre_final:
                    waypoint_threshold = 0.30
                else:
                    waypoint_threshold = 0.40
                
                if dist_to_waypoint < waypoint_threshold and not is_final_waypoint:
                    self._current_waypoint_idx += 1
                    next_wp = self._route_waypoints[self._current_waypoint_idx] if self._current_waypoint_idx < len(self._route_waypoints) else self.active_goal
                    print(f"[TO_BOX] Waypoint {self._current_waypoint_idx}/{len(self._route_waypoints)} reached -> next: ({next_wp[0]:.2f}, {next_wp[1]:.2f})")
                    continue

                if self.tobox_state == 0:
                    min_front = min(front_obs, fl_obs, fr_obs)
                    left_blocked = left_obs < 0.25  # Increased from 0.20
                    right_blocked = right_obs < 0.25
                    rear_clear = rear_min > 0.30

                    # ===== STUCK DETECTION =====
                    if not hasattr(self, '_tobox_stuck_pos'):
                        self._tobox_stuck_pos = (self.pose[0], self.pose[1])
                        self._tobox_stuck_time = current_time
                        self._tobox_escape_mode = False

                    moved_dist = math.hypot(
                        self.pose[0] - self._tobox_stuck_pos[0],
                        self.pose[1] - self._tobox_stuck_pos[1]
                    )

                    if moved_dist > 0.12:  # Moved 12cm, reset stuck tracking
                        self._tobox_stuck_pos = (self.pose[0], self.pose[1])
                        self._tobox_stuck_time = current_time
                        self._tobox_escape_mode = False

                    stuck_duration = current_time - self._tobox_stuck_time

                    # If stuck for 3+ seconds, enter escape mode
                    if stuck_duration > 3.0 and not self._tobox_escape_mode:
                        self._tobox_escape_mode = True
                        self._tobox_escape_start = current_time
                        # LOCK escape direction based on current sensor readings
                        self._escape_go_left = left_obs > right_obs + 0.03
                        # Track skipped waypoints for re-routing
                        if not hasattr(self, '_escape_skips'):
                            self._escape_skips = 0
                        print(f"[TO_BOX] STUCK! Escape {'LEFT' if self._escape_go_left else 'RIGHT'} (L={left_obs:.2f}, R={right_obs:.2f})")

                    # ===== ESCAPE MODE =====
                    if self._tobox_escape_mode:
                        escape_elapsed = current_time - self._tobox_escape_start
                        go_left = getattr(self, '_escape_go_left', left_obs > right_obs)

                        # Phase 1 (0-1.5s): Aggressive strafe + reverse
                        if escape_elapsed < 1.5:
                            vy_escape = 0.15 if go_left else -0.15
                            vx_escape = -0.08 if rear_clear else 0.0
                            omega_escape = 0.35 if go_left else -0.35
                            self._safe_move(vx_escape, vy_escape, omega_escape)
                            if int(escape_elapsed * 4) % 4 == 0:
                                print(f"[TO_BOX] Escape P1: strafe {'LEFT' if go_left else 'RIGHT'}")
                            continue

                        # Phase 2 (1.5-3s): Reverse + strong turn
                        elif escape_elapsed < 3.0:
                            if rear_clear:
                                self._safe_move(-0.15, 0.0, 0.6 if go_left else -0.6)
                            else:
                                # Rear blocked - forward strafe
                                self._safe_move(0.08, 0.12 if go_left else -0.12, 0.4 if go_left else -0.4)
                            if int(escape_elapsed * 4) % 4 == 0:
                                print(f"[TO_BOX] Escape P2: reverse+turn {'LEFT' if go_left else 'RIGHT'}")
                            continue

                        # Phase 3: Skip waypoint
                        else:
                            self._escape_skips = getattr(self, '_escape_skips', 0) + 1
                            print(f"[TO_BOX] Skip WP{self._current_waypoint_idx + 1} (total skips: {self._escape_skips})")
                            self._current_waypoint_idx += 1

                            # If skipped 3+ waypoints, re-generate route
                            if self._escape_skips >= 3:
                                print("[TO_BOX] Too many skips! Re-routing from current position.")
                                self._route_waypoints = get_route_to_box(
                                    (self.pose[0], self.pose[1]), self.current_color
                                )
                                self._current_waypoint_idx = 0
                                self._escape_skips = 0

                            self._tobox_escape_mode = False
                            self._tobox_stuck_pos = (self.pose[0], self.pose[1])
                            self._tobox_stuck_time = current_time
                            continue

                    waypoints_remaining = len(self._route_waypoints) - self._current_waypoint_idx
                    approaching_box = waypoints_remaining <= 3 or dist_to_box < 1.0

                    if approaching_box:
                        if dist_to_box < 0.55 or min_front < 0.18:
                            print(f"[TO_BOX] Arrived at box (dist={dist_to_box:.2f}m). Starting alignment.")
                            self._safe_move(0.0, 0.0, 0.0)
                            self.tobox_state = 1
                            self._tobox_maneuver_timer = 0.0
                            self._align_start_time = None
                            # Cleanup stuck tracking
                            for attr in ['_tobox_stuck_pos', '_tobox_stuck_time', '_tobox_escape_mode',
                                         '_tobox_escape_start', '_escape_go_left', '_escape_skips']:
                                if hasattr(self, attr):
                                    delattr(self, attr)
                            continue

                        cmd_omega = -angle_to_waypoint * 1.2
                        cmd_omega = max(-0.35, min(0.35, cmd_omega))
                        cmd_speed = 0.12

                        if left_blocked:
                            cmd_omega = min(cmd_omega - 0.15, -0.20)
                        if right_blocked:
                            cmd_omega = max(cmd_omega + 0.15, 0.20)

                        self._safe_move(cmd_speed, 0.0, cmd_omega)
                        continue

                    EMERGENCY_STOP = 0.22
                    FRONT_DANGER = 0.38  # Increased from 0.35
                    FRONT_WARN = 0.55
                    LATERAL_WARN = 0.35

                    front_left_close = fl_obs < FRONT_WARN
                    front_right_close = fr_obs < FRONT_WARN

                    cmd_speed = 0.0
                    cmd_omega = 0.0
                    cmd_vy = 0.0  # Add lateral velocity

                    if min_front < EMERGENCY_STOP:
                        self._tobox_maneuver_timer += dt

                        # Use strafe to escape!
                        if left_obs > right_obs + 0.05:
                            cmd_vy = 0.10
                        elif right_obs > left_obs + 0.05:
                            cmd_vy = -0.10

                        if self._tobox_maneuver_timer > 0.2 and rear_clear:
                            cmd_speed = -0.10
                            if not left_blocked and (right_blocked or left_obs > right_obs):
                                cmd_omega = 0.5
                            elif not right_blocked:
                                cmd_omega = -0.5

                    elif min_front < FRONT_DANGER:
                        self._tobox_maneuver_timer += dt

                        # Use strafe + reverse + turn
                        if left_obs > right_obs + 0.05:
                            cmd_vy = 0.08
                        elif right_obs > left_obs + 0.05:
                            cmd_vy = -0.08

                        if rear_clear:
                            cmd_speed = -0.12
                            if not left_blocked and (right_blocked or left_obs > right_obs):
                                cmd_omega = 0.5
                            elif not right_blocked:
                                cmd_omega = -0.5
                        else:
                            cmd_speed = 0.0
                            if not left_blocked:
                                cmd_omega = 0.5
                            elif not right_blocked:
                                cmd_omega = -0.5

                    elif min_front < FRONT_WARN or front_left_close or front_right_close:
                        self._tobox_maneuver_timer = 0.0

                        # Add strafe for obstacle avoidance
                        if front_left_close and not front_right_close:
                            cmd_vy = -0.06
                        elif front_right_close and not front_left_close:
                            cmd_vy = 0.06

                        if left_blocked:
                            cmd_omega = -0.4
                            cmd_vy = -0.08
                        elif right_blocked:
                            cmd_omega = 0.4
                            cmd_vy = 0.08
                        elif front_left_close and not front_right_close:
                            cmd_omega = -0.35
                        elif front_right_close and not front_left_close:
                            cmd_omega = 0.35
                        elif left_obs > right_obs + 0.1:
                            cmd_omega = 0.3
                        elif right_obs > left_obs + 0.1:
                            cmd_omega = -0.3
                        else:
                            cmd_omega = -angle_to_waypoint * 0.6
                            cmd_omega = max(-0.35, min(0.35, cmd_omega))

                        cmd_speed = 0.06 + 0.06 * (min_front / FRONT_WARN)

                    else:
                        self._tobox_maneuver_timer = 0.0

                        cmd_omega = -angle_to_waypoint * 1.5
                        MAX_TURN = 0.45
                        cmd_omega = max(-MAX_TURN, min(MAX_TURN, cmd_omega))

                        if abs(angle_to_waypoint) > math.pi/2:
                            cmd_speed = 0.05
                            cmd_omega = -angle_to_waypoint * 0.5
                            cmd_omega = max(-0.30, min(0.30, cmd_omega))
                        else:
                            if left_obs < LATERAL_WARN:
                                cmd_omega -= 0.15 * (LATERAL_WARN - left_obs) / LATERAL_WARN
                            if right_obs < LATERAL_WARN:
                                cmd_omega += 0.15 * (LATERAL_WARN - right_obs) / LATERAL_WARN
                            cmd_omega = max(-MAX_TURN, min(MAX_TURN, cmd_omega))

                            cmd_speed = 0.14
                            turn_penalty = 1.0 - min(0.25, abs(cmd_omega) / MAX_TURN)
                            cmd_speed *= turn_penalty

                    if left_blocked and cmd_omega > 0:
                        cmd_omega = -0.45
                        cmd_vy = -0.08
                        cmd_speed = min(cmd_speed, 0.02)

                    if right_blocked and cmd_omega < 0:
                        cmd_omega = 0.45
                        cmd_vy = 0.08
                        cmd_speed = min(cmd_speed, 0.02)

                    if left_blocked and right_blocked:
                        cmd_omega = 0.0
                        cmd_vy = 0.0
                        if min_front > 0.35:
                            cmd_speed = 0.05
                        elif rear_clear:
                            cmd_speed = -0.10
                        else:
                            cmd_speed = 0.0

                    self._safe_move(cmd_speed, cmd_vy, cmd_omega)

                elif self.tobox_state == 1:
                    if self._align_start_time is None:
                        self._align_start_time = self.robot.getTime()
                    
                    align_elapsed = self.robot.getTime() - self._align_start_time
                    
                    if self.current_color == "green":
                        target_heading = math.pi / 2
                    elif self.current_color == "blue":
                        target_heading = -math.pi / 2
                    else:
                        target_heading = 0.0
                    
                    heading_error = wrap_angle(target_heading - self.pose[2])
                    
                    if align_elapsed < 0.1:
                        print(f"[TO_BOX] Alignment: error={math.degrees(heading_error):.0f}°")
                    
                    ALIGN_TOLERANCE = 0.44
                    ALIGN_TIMEOUT = 2.0
                    
                    if abs(heading_error) < ALIGN_TOLERANCE:
                        print(f"[TO_BOX] Alignment OK (error={math.degrees(heading_error):.0f}°). Approaching.")
                        self.tobox_state = 2
                        self._approach_start_time = None
                        continue
                    
                    if align_elapsed > ALIGN_TIMEOUT:
                        print(f"[TO_BOX] Alignment timeout. Proceeding (error={math.degrees(heading_error):.0f}°).")
                        self.tobox_state = 2
                        self._approach_start_time = None
                        continue

                    omega = heading_error * 0.5
                    omega = max(-0.25, min(0.25, omega))
                    
                    self._safe_move(0.0, 0.0, omega)

                elif self.tobox_state == 2:
                    ds_val = front_info["front"]
                    
                    if self._approach_start_time is None:
                        self._approach_start_time = self.robot.getTime()
                        print(f"[TO_BOX] Starting approach. Front sensor: {ds_val:.3f}m")
                    
                    approach_elapsed = self.robot.getTime() - self._approach_start_time
                    
                    if int(approach_elapsed * 2) % 2 == 0 and approach_elapsed > 0.5:
                        if not hasattr(self, '_last_approach_log') or self._last_approach_log != int(approach_elapsed):
                            self._last_approach_log = int(approach_elapsed)
                            print(f"[TO_BOX] Approaching: ds={ds_val:.3f}m, dist_box={dist_to_box:.2f}m")
                    
                    drop_ready = (0.35 < ds_val < 0.60)
                    timeout_drop = (approach_elapsed > 4.0 and dist_to_box < 0.65)
                    
                    if drop_ready or timeout_drop:
                        if timeout_drop and not drop_ready:
                            print(f"[TO_BOX] Approach timeout (ds={ds_val:.3f}m). DROP.")
                        else:
                            print(f"[TO_BOX] Final position (ds={ds_val:.3f}m). Executing DROP.")
                        self._safe_move(0.0, 0.0, 0.0)
                        self._start_drop()
                        for attr in ['tobox_state', '_tobox_maneuver_timer', '_align_start_time', 
                                     '_approach_start_time', '_route_waypoints', '_current_waypoint_idx',
                                     '_last_pos_log_time', '_last_approach_log']:
                            if hasattr(self, attr):
                                delattr(self, attr)
                        continue

                    if ds_val < 0.35:
                        print(f"[TO_BOX] Too close ({ds_val:.3f}m). Safety DROP.")
                        self._safe_move(0.0, 0.0, 0.0)
                        self._start_drop()
                        for attr in ['tobox_state', '_tobox_maneuver_timer', '_route_waypoints', 
                                     '_current_waypoint_idx', '_last_pos_log_time']:
                            if hasattr(self, attr):
                                delattr(self, attr)
                        continue

                    if ds_val > 0.50:
                        speed = 0.08
                    elif ds_val > 0.35:
                        speed = 0.05
                    else:
                        speed = 0.03
                    
                    omega = -angle_to_box * 1.0
                    omega = max(-0.20, min(0.20, omega))
                    
                    self._safe_move(speed, 0.0, omega)
                
                continue

            # ===== MODE: DROP =====
            if self.mode == "drop":
                self._handle_drop(dt)
                continue

            # ===== MODE: RETURN_TO_SPAWN =====
            if self.mode == "return_to_spawn":
                self._handle_return_to_spawn(dt, lidar_info, rear_info, lateral_info, front_info)
                continue

