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

# Obstáculos conhecidos (WoodenBoxes do mundo) - posição (x, y) e raio de segurança
KNOWN_OBSTACLES = [
    (0.6, 0.0, 0.25),      # A
    (1.96, -1.24, 0.25),   # B
    (1.95, 1.25, 0.25),    # C
    (-2.28, 1.5, 0.25),    # D
    (-1.02, 0.75, 0.25),   # E
    (-1.02, -0.74, 0.25),  # F
    (-2.27, -1.51, 0.25),  # G
]

# Depósitos (PlasticFruitBox) - evitar colisão mas são destinos
BOX_POSITIONS = {
    "green": (0.48, 1.58),
    "blue": (0.48, -1.62),
    "red": (2.31, 0.01),
}


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
    """Controlador Fuzzy para navegação com desvio de obstáculos."""

    def __init__(self, max_speed=0.2):
        self.max_speed = max_speed
        self.safety_margin = 0.4  # Margem de segurança dos obstáculos

    @staticmethod
    def _mu_close(x, threshold=0.45):
        """Pertinência para 'perto' - threshold aumentado para maior segurança."""
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_very_close(x, threshold=0.25):
        """Pertinência para 'muito perto' - emergência."""
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_far(x, start=0.4, end=1.5):
        """Pertinência para 'longe'."""
        if x <= start:
            return 0.0
        if x >= end:
            return 1.0
        return (x - start) / (end - start)

    @staticmethod
    def _mu_small_angle(angle_rad):
        """Pertinência para ângulo pequeno (alinhado)."""
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(5):
            return 1.0
        if a > math.radians(25):
            return 0.0
        return (math.radians(25) - a) / math.radians(20)

    @staticmethod
    def _mu_medium_angle(angle_rad):
        """Pertinência para ângulo médio."""
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(10) or a > math.radians(60):
            return 0.0
        if a < math.radians(35):
            return (a - math.radians(10)) / math.radians(25)
        return (math.radians(60) - a) / math.radians(25)

    @staticmethod
    def _mu_big_angle(angle_rad):
        """Pertinência para ângulo grande (precisa girar muito)."""
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(40):
            return 0.0
        if a > math.radians(90):
            return 1.0
        return (a - math.radians(40)) / math.radians(50)

    def check_known_obstacles(self, pose):
        """Verifica proximidade de obstáculos conhecidos e retorna distâncias por setor."""
        x, y, yaw = pose
        min_dist_left = 2.0
        min_dist_right = 2.0
        min_dist_front = 2.0

        for ox, oy, radius in KNOWN_OBSTACLES:
            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy) - radius

            # Ângulo do obstáculo relativo ao robô
            obs_angle = wrap_angle(math.atan2(dy, dx) - yaw)

            # Classificar por setor
            if abs(obs_angle) < math.radians(30):
                min_dist_front = min(min_dist_front, dist)
            elif obs_angle > 0 and obs_angle < math.radians(120):
                min_dist_left = min(min_dist_left, dist)
            elif obs_angle < 0 and obs_angle > math.radians(-120):
                min_dist_right = min(min_dist_right, dist)

        return min_dist_front, min_dist_left, min_dist_right

    def compute(
        self,
        has_target,
        target_distance,
        target_angle,
        obs_front,
        obs_left,
        obs_right,
        pose=None,
    ):
        """Calcula comandos de velocidade usando lógica fuzzy."""
        if not has_target:
            return 0.0, 0.0, 0.3

        # Combinar LiDAR com obstáculos conhecidos
        if pose is not None:
            known_front, known_left, known_right = self.check_known_obstacles(pose)
            obs_front = min(obs_front, known_front)
            obs_left = min(obs_left, known_left)
            obs_right = min(obs_right, known_right)

        # Fuzzificação
        mu_front_close = self._mu_close(obs_front)
        mu_front_very_close = self._mu_very_close(obs_front)
        mu_left_close = self._mu_close(obs_left)
        mu_right_close = self._mu_close(obs_right)

        mu_target_far = self._mu_far(target_distance)
        mu_target_close = self._mu_close(target_distance, threshold=0.3)
        mu_angle_small = self._mu_small_angle(target_angle)
        mu_angle_medium = self._mu_medium_angle(target_angle)
        mu_angle_big = self._mu_big_angle(target_angle)

        # ========== REGRAS FUZZY ==========

        # Regra 1: Se obstáculo muito perto na frente -> PARAR e girar
        if mu_front_very_close > 0.5:
            vx = -0.05  # Recuar levemente
            vy = 0.1 * (1 if obs_left < obs_right else -1)  # Strafe para lado livre
            omega = 0.5 * (1 if obs_left > obs_right else -1)  # Girar para lado livre
            return vx, vy, omega

        # Regra 2: Se obstáculo perto na frente -> reduzir velocidade, desviar
        speed_reduction = 1.0 - (0.8 * mu_front_close)

        # Regra 3: Velocidade frontal baseada em distância e alinhamento
        vx = 0.15 * mu_target_far * speed_reduction * (1.0 - 0.5 * mu_angle_big)

        # Regra 4: Se perto do alvo e alinhado -> aproximar devagar
        if mu_target_close > 0.3 and mu_angle_small > 0.5:
            vx = 0.08 * speed_reduction

        # Regra 5: Strafe para desviar de obstáculos laterais
        vy = 0.12 * (mu_right_close - mu_left_close)

        # Regra 6: Rotação para alinhar com alvo
        angle_sign = 1 if target_angle > 0 else -1
        omega = (
            0.8 * mu_angle_big * angle_sign +
            0.4 * mu_angle_medium * angle_sign +
            0.1 * (1.0 - mu_angle_small) * angle_sign
        )

        # Regra 7: Ajuste de rotação por obstáculos laterais
        omega += 0.3 * (mu_right_close - mu_left_close)

        # Limitação de velocidades
        vx = max(-self.max_speed, min(self.max_speed, vx))
        vy = max(-self.max_speed * 0.8, min(self.max_speed * 0.8, vy))
        omega = max(-1.0, min(1.0, omega))

        return vx, vy, omega


class YouBotController:
    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        # Log throttling
        self._log_times = {}

        # Sensores
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            # Tentar habilitar recognition se disponível
            try:
                self.camera.recognitionEnable(self.time_step)
            except:
                pass

        # LiDAR principal (altura média - detecta obstáculos e paredes)
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            self.lidar.enablePointCloud()

        # LiDAR baixo (altura do chão - detecta cubos pequenos)
        self.lidar_low = self.robot.getDevice("lidar_low")
        if self.lidar_low:
            self.lidar_low.enable(self.time_step)
            self.lidar_low.enablePointCloud()

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

        self.box_positions = BOX_POSITIONS

    def _log_throttled(self, key, msg, interval=1.5):
        """Log com rate limit por chave."""
        try:
            now = self.robot.getTime()
        except Exception:
            now = 0.0
        last = self._log_times.get(key, -1e9)
        if now - last >= interval:
            print(msg)
            self._log_times[key] = now

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

    def _process_lidar_low(self):
        """Processa LiDAR baixo para detectar cubos no chão."""
        if not self.lidar_low:
            return {"cubes": [], "min_front": 2.0}

        ranges = self.lidar_low.getRangeImage()
        if not ranges:
            return {"cubes": [], "min_front": 2.0}

        res = self.lidar_low.getHorizontalResolution()
        fov = self.lidar_low.getFov()
        angle_step = fov / max(1, res - 1)

        # Detectar objetos próximos (potenciais cubos)
        cube_candidates = []
        front_readings = []

        for i, r in enumerate(ranges):
            if math.isinf(r) or math.isnan(r) or r <= 0:
                continue
            angle = -fov / 2.0 + i * angle_step

            # Cubos estão tipicamente entre 0.1m e 1.0m
            if 0.08 < r < 1.0:
                cube_candidates.append((r, angle))

            # Leituras frontais
            if abs(angle) < math.radians(20):
                front_readings.append(r)

        min_front = min(front_readings) if front_readings else 2.0
        return {"cubes": cube_candidates, "min_front": min_front}

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
            if area < 10:  # aceitar cubos pequenos distantes
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

        if best_detection:
            self._log_throttled(
                "camera_detect",
                f"[CAM] cor={best_detection['color']} area={best_detection['area']:.1f} dist≈{best_detection['distance']:.2f} angle={best_detection['angle']:.2f}",
                interval=1.0,
            )
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

    def _project_lidar_point_world(self, r, angle):
        """Converte medida do LiDAR baixo para coordenada no mundo."""
        yaw = self.pose[2]
        x_r = r * math.cos(angle)
        y_r = r * math.sin(angle)
        x_w = self.pose[0] + math.cos(yaw) * x_r - math.sin(yaw) * y_r
        y_w = self.pose[1] + math.sin(yaw) * x_r + math.cos(yaw) * y_r
        return (x_w, y_w)

    def _check_obstacle_ahead(self, lidar_front, threshold=0.5):
        """Verifica se há obstáculo à frente (LiDAR + conhecidos)."""
        # Verificar LiDAR
        if lidar_front < threshold:
            return True

        # Verificar obstáculos conhecidos
        x, y, yaw = self.pose
        for ox, oy, radius in KNOWN_OBSTACLES:
            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy)
            obs_angle = wrap_angle(math.atan2(dy, dx) - yaw)

            # Se obstáculo está na frente e próximo
            if abs(obs_angle) < math.radians(40) and dist < (threshold + radius + 0.1):
                return True

        return False

    def _search_navigation(self, lidar_info, dt):
        """Navegação lawnmower: anda reto, gira ao encontrar obstáculo/parede."""
        front_dist = lidar_info["front"]
        left_dist = lidar_info["left"]
        right_dist = lidar_info["right"]

        OBSTACLE_THRESHOLD = 0.55  # metros - aumentado para segurança
        TURN_SPEED = 0.5  # rad/s
        FORWARD_SPEED = 0.12  # m/s - reduzido para mais controle
        TURN_DURATION = math.pi / 2 / TURN_SPEED  # tempo para girar 90°

        # Verificar obstáculos conhecidos também
        obstacle_ahead = self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD)

        if self.search_state == "forward":
            if obstacle_ahead:
                # Decidir direção da curva baseado no espaço lateral
                if left_dist > right_dist:
                    self.search_direction = 1  # Girar para esquerda
                else:
                    self.search_direction = -1  # Girar para direita
                self.search_state = "turn_start"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            else:
                # Correção lateral para desviar de obstáculos próximos
                vy_correct = 0.0
                omega_correct = 0.0

                if left_dist < 0.35:
                    vy_correct = -0.06
                    omega_correct = -0.1
                elif right_dist < 0.35:
                    vy_correct = 0.06
                    omega_correct = 0.1

                return FORWARD_SPEED, vy_correct, omega_correct

        elif self.search_state == "turn_start":
            self.turn_progress += dt
            if self.turn_progress >= TURN_DURATION:
                self.search_state = "turn_mid"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, TURN_SPEED * self.search_direction

        elif self.search_state == "turn_mid":
            self.turn_progress += dt
            # Andar um pouco para mudar de faixa
            if self.turn_progress >= 0.8:
                self.search_state = "turn_end"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            # Se encontrar obstáculo, encurtar
            if self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD):
                self.search_state = "turn_end"
                self.turn_progress = 0.0
            return FORWARD_SPEED, 0.0, 0.0

        elif self.search_state == "turn_end":
            self.turn_progress += dt
            if self.turn_progress >= TURN_DURATION:
                self.search_state = "forward"
                self.turn_progress = 0.0
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
                    # Tentar usar LiDAR baixo para encontrar cubo à frente
                    lidar_low_info = self._process_lidar_low()
                    best = None
                    for r, ang in lidar_low_info["cubes"]:
                        if r < 1.2 and abs(ang) < math.radians(35):
                            best = (r, ang)
                            break
                    if best:
                        if lidar_info["front"] > best[0]:
                            self._log_throttled(
                                "lidar_low_detect",
                                f"[LIDAR_LOW] alvo r={best[0]:.2f} ang={best[1]:.2f} (alto não marcou)",
                                interval=1.0,
                            )
                        self.current_color = None  # cor será determinada quando a câmera ver
                        self.current_target = self._project_lidar_point_world(best[0], best[1])
                        self.mode = "approach"
                        continue

                    # Navegação em padrão lawnmower: andar reto, girar ao encontrar obstáculo/parede
                    vx_cmd, vy_cmd, omega_cmd = self._search_navigation(lidar_info, dt)
                    self.base.move(vx_cmd, vy_cmd, omega_cmd)
                    continue

            if self.mode == "approach":
                if detection:
                    self.current_color = detection["color"]
                    self.current_target = self._project_target_world(detection)
                distance, angle = self._distance_to_target()

                # Usar LiDAR baixo para alinhamento fino quando perto
                lidar_low_info = self._process_lidar_low()
                if lidar_low_info["min_front"] < 0.25 and distance and distance < 0.3:
                    # Muito perto - iniciar grasp
                    self._start_grasp()
                    continue

                if distance is not None and distance < 0.22:
                    self._start_grasp()
                    continue

                vx_cmd, vy_cmd, omega_cmd = self.navigator.compute(
                    True,
                    distance or 0.5,
                    angle or 0.0,
                    lidar_info["front"],
                    lidar_info["left"],
                    lidar_info["right"],
                    pose=self.pose,  # Passar pose para verificar obstáculos conhecidos
                )
                self.base.move(vx_cmd, vy_cmd, omega_cmd)
                continue

            if self.mode == "grasp":
                self._handle_grasp(dt)
                continue

            if self.mode == "to_box":
                distance, angle = self._distance_to_target()
                if distance is not None and distance < 0.30:
                    self._start_drop()
                    continue
                vx_cmd, vy_cmd, omega_cmd = self.navigator.compute(
                    True,
                    distance or 0.5,
                    angle or 0.0,
                    lidar_info["front"],
                    lidar_info["left"],
                    lidar_info["right"],
                    pose=self.pose,  # Passar pose para verificar obstáculos conhecidos
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