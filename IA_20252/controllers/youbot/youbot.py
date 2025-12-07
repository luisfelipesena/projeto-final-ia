import math
import time
from collections import deque

from controller import Robot
from base import Base
from arm import Arm
from gripper import Gripper

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

CUBE_SIZE = 0.03
ARENA_CENTER = (-0.79, 0.0)
ARENA_SIZE = (7.0, 4.0)


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class ColorClassifier:
    def __init__(self):
        self.hsv_ranges = {
            "red": [
                ((0, 100, 100), (10, 255, 255)),
                ((170, 100, 100), (180, 255, 255)),
            ],
            "green": [((35, 100, 100), (85, 255, 255))],
            "blue": [((100, 100, 100), (130, 255, 255))],
        }

    def classify(self, hsv_roi):
        if hsv_roi is None or cv2 is None or np is None:
            return None

        scores = {}
        for color, ranges in self.hsv_ranges.items():
            mask = None
            for low, high in ranges:
                m = cv2.inRange(
                    hsv_roi,
                    np.array(low, dtype=np.uint8),
                    np.array(high, dtype=np.uint8),
                )
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            scores[color] = cv2.countNonZero(mask) if mask is not None else 0

        if not scores:
            return None
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return None
        total = sum(scores.values()) or 1
        return best, scores[best] / total


class OccupancyGrid:
    def __init__(self, center, size, resolution=0.1):
        self.resolution = resolution
        self.center = center
        self.size = size
        self.origin = (
            center[0] - size[0] / 2.0,
            center[1] - size[1] / 2.0,
        )
        width = int(math.ceil(size[0] / resolution))
        height = int(math.ceil(size[1] / resolution))
        if np is not None:
            self.grid = -1 * np.ones((height, width), dtype=np.int8)
        else:
            self.grid = [[-1 for _ in range(width)] for _ in range(height)]

    def world_to_grid(self, x, y):
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return gx, gy

    def grid_in_bounds(self, gx, gy):
        if np is not None:
            return (
                0 <= gy < self.grid.shape[0]
                and 0 <= gx < self.grid.shape[1]
            )
        return (
            0 <= gy < len(self.grid)
            and 0 <= gx < len(self.grid[0])
        )

    def mark_obstacle(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if not self.grid_in_bounds(gx, gy):
            return
        if np is not None:
            self.grid[gy, gx] = 1
        else:
            self.grid[gy][gx] = 1

    def mark_free(self, x, y):
        gx, gy = self.world_to_grid(x, y)
        if not self.grid_in_bounds(gx, gy):
            return
        if np is not None:
            if self.grid[gy, gx] == -1:
                self.grid[gy, gx] = 0
        else:
            if self.grid[gy][gx] == -1:
                self.grid[gy][gx] = 0


class FuzzyNavigator:
    def __init__(self, max_speed=0.2):
        self.max_speed = max_speed

    @staticmethod
    def _mu_close(x, threshold=0.35):
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_far(x, start=0.35, end=1.2):
        if x <= start:
            return 0.0
        if x >= end:
            return 1.0
        return (x - start) / (end - start)

    @staticmethod
    def _mu_small_angle(angle_rad):
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(5):
            return 1.0
        if a > math.radians(30):
            return 0.0
        return (math.radians(30) - a) / math.radians(25)

    @staticmethod
    def _mu_big_angle(angle_rad):
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(10):
            return 0.0
        if a > math.radians(70):
            return 1.0
        return (a - math.radians(10)) / math.radians(60)

    def compute(
        self,
        has_target,
        target_distance,
        target_angle,
        obs_front,
        obs_left,
        obs_right,
    ):
        if not has_target:
            return 0.0, 0.0, 0.35

        mu_front_close = self._mu_close(obs_front)
        mu_left_close = self._mu_close(obs_left)
        mu_right_close = self._mu_close(obs_right)

        mu_far = self._mu_far(target_distance)
        mu_small = self._mu_small_angle(target_angle)
        mu_big = self._mu_big_angle(target_angle)

        # Rule aggregation (simple weighted blend)
        vx = (0.12 * mu_far) * (1.0 - mu_front_close)
        # Strafe away from the closest side obstacle
        vy = 0.08 * (mu_right_close - mu_left_close)

        # Heading correction
        omega = (
            0.6 * mu_big * (1 if target_angle > 0 else -1)
            + 0.25 * (mu_right_close - mu_left_close)
        )
        omega += 0.25 * (0.5 - mu_small) * (1 if target_angle > 0 else -1)

        vx = max(-self.max_speed, min(self.max_speed, vx))
        vy = max(-self.max_speed, min(self.max_speed, vy))
        omega = max(-0.8, min(0.8, omega))
        return vx, vy, omega


class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            self.lidar.enablePointCloud()

        self.color_classifier = ColorClassifier()
        self.navigator = FuzzyNavigator()
        self.grid = OccupancyGrid(ARENA_CENTER, ARENA_SIZE, resolution=0.1)

        self.pose = self._initial_pose()
        self.current_target = None
        self.current_color = None
        self.mode = "search"
        self.stage = 0
        self.stage_timer = 0.0
        self.collected = 0
        self.max_cubes = 15
        self.search_spin_dir = 1

        # Navegação em padrão lawnmower (varredura)
        self.search_state = "forward"  # forward, turn_start, turn_mid, turn_end
        self.search_direction = 1  # 1 = direita, -1 = esquerda
        self.turn_progress = 0.0
        self.forward_timer = 0.0

        self.box_positions = {
            "green": (0.48, 1.58),
            "blue": (0.48, -1.62),
            "red": (2.31, 0.01),
        }

    def _initial_pose(self):
        try:
            self_node = self.robot.getSelf()
            translation = self_node.getField("translation").getSFVec3f()
            rotation = self_node.getField("rotation").getSFRotation()
            yaw = rotation[3] * rotation[2] if abs(rotation[2]) > 0.5 else rotation[3]
            return [translation[0], translation[1], yaw]
        except Exception:
            return [0.0, 0.0, 0.0]

    def _integrate_pose(self, vx, vy, omega, dt):
        yaw = self.pose[2]
        dx_world = math.cos(yaw) * vx - math.sin(yaw) * vy
        dy_world = math.sin(yaw) * vx + math.cos(yaw) * vy
        self.pose[0] += dx_world * dt
        self.pose[1] += dy_world * dt
        self.pose[2] = wrap_angle(self.pose[2] + omega * dt)

    def _process_lidar(self):
        if not self.lidar:
            return {"front": 1.0, "left": 1.0, "right": 1.0, "points": []}

        ranges = self.lidar.getRangeImage()
        res = self.lidar.getHorizontalResolution()
        fov = self.lidar.getFov()
        angle_step = fov / max(1, res - 1)

        points = []
        front_window = []
        left_window = []
        right_window = []

        for i, r in enumerate(ranges):
            angle = -fov / 2.0 + i * angle_step
            if math.isinf(r) or math.isnan(r) or r <= 0:
                continue
            if i % 2 == 0:
                points.append((r, angle))
            # windows
            deg = math.degrees(angle)
            if -20 <= deg <= 20:
                front_window.append(r)
            elif 20 < deg <= 90:
                left_window.append(r)
            elif -90 <= deg < -20:
                right_window.append(r)

        front = min(front_window) if front_window else 2.0
        left = min(left_window) if left_window else 2.0
        right = min(right_window) if right_window else 2.0

        return {"front": front, "left": left, "right": right, "points": points, "fov": fov}

    def _update_grid_with_lidar(self, lidar_points):
        yaw = self.pose[2]
        for r, angle in lidar_points:
            if math.isnan(r) or math.isinf(r) or r <= 0:
                continue
            x_r = r * math.cos(angle)
            y_r = r * math.sin(angle)
            x_w = self.pose[0] + math.cos(yaw) * x_r - math.sin(yaw) * y_r
            y_w = self.pose[1] + math.sin(yaw) * x_r + math.cos(yaw) * y_r
            if math.isnan(x_w) or math.isnan(y_w):
                continue
            self.grid.mark_obstacle(x_w, y_w)

    def _process_camera(self):
        if not self.camera or cv2 is None or np is None:
            return None
        image = self.camera.getImageArray()
        if image is None:
            return None

        img = np.array(image, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w, _ = img.shape

        best_detection = None
        for color, ranges in self.color_classifier.hsv_ranges.items():
            mask = None
            for low, high in ranges:
                m = cv2.inRange(
                    hsv,
                    np.array(low, dtype=np.uint8),
                    np.array(high, dtype=np.uint8),
                )
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            if mask is None:
                continue
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if area < 30:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            cx = x + bw / 2.0
            angle = ((cx - (w / 2.0)) / w) * self.camera.getFov()
            distance = self._estimate_distance(bh, h)
            if (
                best_detection is None
                or area > best_detection["area"]
            ):
                best_detection = {
                    "color": color,
                    "angle": angle,
                    "distance": distance,
                    "area": area,
                    "bbox": (x, y, bw, bh),
                }

        return best_detection

    def _estimate_distance(self, bbox_h, img_h):
        try:
            f = (img_h / 2.0) / math.tan(self.camera.getFov() / 2.0)
            distance = (CUBE_SIZE * f) / max(1.0, bbox_h)
            return distance
        except Exception:
            return 0.4

    def _project_target_world(self, detection):
        yaw = self.pose[2]
        dist = detection["distance"]
        ang = detection["angle"]
        x_r = dist * math.cos(ang)
        y_r = dist * math.sin(ang)
        x_w = self.pose[0] + math.cos(yaw) * x_r - math.sin(yaw) * y_r
        y_w = self.pose[1] + math.sin(yaw) * x_r + math.cos(yaw) * y_r
        return (x_w, y_w)

    def _search_navigation(self, lidar_info, dt):
        """Navegação lawnmower: anda reto, gira 90° ao encontrar obstáculo/parede."""
        front_dist = lidar_info["front"]
        left_dist = lidar_info["left"]
        right_dist = lidar_info["right"]

        OBSTACLE_THRESHOLD = 0.5  # metros
        TURN_SPEED = 0.6  # rad/s
        FORWARD_SPEED = 0.15  # m/s
        TURN_DURATION = math.pi / 2 / TURN_SPEED  # tempo para girar 90°

        if self.search_state == "forward":
            # Andar reto até encontrar obstáculo na frente
            if front_dist < OBSTACLE_THRESHOLD:
                # Iniciar manobra de curva
                self.search_state = "turn_start"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            else:
                # Pequena correção para desviar de obstáculos laterais
                vy_correct = 0.0
                if left_dist < 0.3:
                    vy_correct = -0.05
                elif right_dist < 0.3:
                    vy_correct = 0.05
                return FORWARD_SPEED, vy_correct, 0.0

        elif self.search_state == "turn_start":
            # Primeira curva de 90° na direção atual
            self.turn_progress += dt
            if self.turn_progress >= TURN_DURATION:
                self.search_state = "turn_mid"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, TURN_SPEED * self.search_direction

        elif self.search_state == "turn_mid":
            # Andar reto um pouco (trocar de faixa)
            self.turn_progress += dt
            if self.turn_progress >= 1.0:  # 1 segundo andando
                self.search_state = "turn_end"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            # Se encontrar obstáculo, voltar a girar
            if front_dist < OBSTACLE_THRESHOLD:
                self.search_state = "turn_end"
                self.turn_progress = 0.0
            return FORWARD_SPEED, 0.0, 0.0

        elif self.search_state == "turn_end":
            # Segunda curva de 90° na mesma direção
            self.turn_progress += dt
            if self.turn_progress >= TURN_DURATION:
                self.search_state = "forward"
                self.turn_progress = 0.0
                self.search_direction *= -1  # Inverter direção para próxima vez
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, TURN_SPEED * self.search_direction

        return 0.0, 0.0, 0.0

    def _start_grasp(self):
        self.mode = "grasp"
        self.stage = 0
        self.stage_timer = 0.0
        self.base.reset()

    def _start_drop(self):
        self.mode = "drop"
        self.stage = 0
        self.stage_timer = 0.0
        self.base.reset()

    def _handle_grasp(self, dt):
        self.stage_timer += dt

        if self.stage == 0:
            self.gripper.release()
            self.arm.set_height(Arm.RESET)
            if self.stage_timer >= 1.5:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 1:
            self.arm.set_height(Arm.FRONT_FLOOR)
            if self.stage_timer >= 2.5:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 2:
            # Slow approach
            self.base.move(0.05, 0.0, 0.0)
            if self.stage_timer >= 2.0:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 3:
            self.gripper.grip()
            if self.stage_timer >= 1.5:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 4:
            self.arm.set_height(Arm.FRONT_PLATE)
            if self.stage_timer >= 2.0:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 5:
            self.collected += 1
            self.mode = "to_box"
            self.stage = 0
            self.stage_timer = 0.0
            target_box = self.box_positions.get(
                (self.current_color or "").lower(), None
            )
            if target_box:
                self.current_target = target_box
            else:
                self.current_target = list(self.box_positions.values())[0]
            return

    def _handle_drop(self, dt):
        self.stage_timer += dt
        if self.stage == 0:
            self.arm.set_height(Arm.FRONT_FLOOR)
            if self.stage_timer >= 1.5:
                self.stage += 1
                self.stage_timer = 0.0
        elif self.stage == 1:
            self.base.move(0.03, 0.0, 0.0)
            if self.stage_timer >= 1.2:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0
        elif self.stage == 2:
            self.gripper.release()
            if self.stage_timer >= 1.0:
                self.stage += 1
                self.stage_timer = 0.0
        elif self.stage == 3:
            self.arm.set_height(Arm.FRONT_PLATE)
            if self.stage_timer >= 1.0:
                self.stage += 1
                self.stage_timer = 0.0
        elif self.stage == 4:
            self.mode = "search"
            self.current_target = None
            self.current_color = None
            self.base.reset()
            self.stage = 0

    def _distance_to_target(self):
        if not self.current_target:
            return None, None
        dx = self.current_target[0] - self.pose[0]
        dy = self.current_target[1] - self.pose[1]
        distance = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - self.pose[2])
        return distance, angle

    def run(self):
        while self.robot.step(self.time_step) != -1:
            dt = self.time_step / 1000.0

            if self.collected >= self.max_cubes:
                self.base.reset()
                break

            # Odometry update
            vx_odo, vy_odo, omega_odo = self.base.compute_odometry(dt)
            self._integrate_pose(vx_odo, vy_odo, omega_odo, dt)

            # Sensor processing
            lidar_info = self._process_lidar()
            self._update_grid_with_lidar(lidar_info["points"])
            detection = self._process_camera()

            if self.mode == "search":
                if detection:
                    self.current_color = detection["color"]
                    self.current_target = self._project_target_world(detection)
                    self.mode = "approach"
                else:
                    # Navegação em padrão lawnmower: andar reto, girar ao encontrar obstáculo/parede
                    vx_cmd, vy_cmd, omega_cmd = self._search_navigation(lidar_info, dt)
                    self.base.move(vx_cmd, vy_cmd, omega_cmd)
                    continue

            if self.mode == "approach":
                if detection:
                    self.current_color = detection["color"]
                    self.current_target = self._project_target_world(detection)
                distance, angle = self._distance_to_target()
                if distance is not None and distance < 0.18:
                    self._start_grasp()
                    continue
                vx_cmd, vy_cmd, omega_cmd = self.navigator.compute(
                    True,
                    distance or 0.5,
                    angle or 0.0,
                    lidar_info["front"],
                    lidar_info["left"],
                    lidar_info["right"],
                )
                self.base.move(vx_cmd, vy_cmd, omega_cmd)
                continue

            if self.mode == "grasp":
                self._handle_grasp(dt)
                continue

            if self.mode == "to_box":
                distance, angle = self._distance_to_target()
                if distance is not None and distance < 0.25:
                    self._start_drop()
                    continue
                vx_cmd, vy_cmd, omega_cmd = self.navigator.compute(
                    True,
                    distance or 0.5,
                    angle or 0.0,
                    lidar_info["front"],
                    lidar_info["left"],
                    lidar_info["right"],
                )
                self.base.move(vx_cmd, vy_cmd, omega_cmd)
                continue

            if self.mode == "drop":
                self._handle_drop(dt)
                continue

            if self.collected >= self.max_cubes:
                self.base.reset()
                break


if __name__ == "__main__":
    controller = YouBotController()
    controller.run()