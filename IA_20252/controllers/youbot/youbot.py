import heapq
import math
from controller import Supervisor
from base import Base
from arm import Arm
from gripper import Gripper
from color_classifier import ColorClassifier

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


# Grade de ocupação para navegação baseada em odometria
class OccupancyGrid:
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    BOX = 3
    CUBE = 4

    def __init__(self, arena_center, arena_size, cell_size=0.12):
        self.cell_size = cell_size
        self.min_x = arena_center[0] - arena_size[0] / 2.0
        self.min_y = arena_center[1] - arena_size[1] / 2.0
        self.max_x = arena_center[0] + arena_size[0] / 2.0
        self.max_y = arena_center[1] + arena_size[1] / 2.0
        self.width = int(math.ceil(arena_size[0] / cell_size))
        self.height = int(math.ceil(arena_size[1] / cell_size))
        self.grid = [
            [self.UNKNOWN for _ in range(self.width)] for _ in range(self.height)
        ]
        self.static_mask = [
            [False for _ in range(self.width)] for _ in range(self.height)
        ]

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height

    def world_to_cell(self, x, y):
        if math.isnan(x) or math.isnan(y):
            return None
        gx = int((x - self.min_x) / self.cell_size)
        gy = int((y - self.min_y) / self.cell_size)
        if not self.in_bounds(gx, gy):
            return None
        return gx, gy

    def cell_to_world(self, gx, gy):
        wx = self.min_x + (gx + 0.5) * self.cell_size
        wy = self.min_y + (gy + 0.5) * self.cell_size
        return wx, wy

    def get(self, gx, gy):
        if not self.in_bounds(gx, gy):
            return self.UNKNOWN
        return self.grid[gy][gx]

    def set(self, gx, gy, value, static=False, overwrite_static=False):
        if not self.in_bounds(gx, gy):
            return False
        if self.static_mask[gy][gx] and not overwrite_static:
            return False
        if self.grid[gy][gx] == value:
            if static and not self.static_mask[gy][gx]:
                self.static_mask[gy][gx] = True
            return False
        self.grid[gy][gx] = value
        if static:
            self.static_mask[gy][gx] = True
        return True

    def fill_disk(self, x, y, radius, value, static=False):
        min_x = x - radius
        max_x = x + radius
        min_y = y - radius
        max_y = y + radius
        cell_min = self.world_to_cell(min_x, min_y)
        cell_max = self.world_to_cell(max_x, max_y)
        if cell_min is None or cell_max is None:
            return
        gx_min, gy_min = cell_min
        gx_max, gy_max = cell_max
        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                wx, wy = self.cell_to_world(gx, gy)
                if math.hypot(wx - x, wy - y) <= radius:
                    self.set(gx, gy, value, static=static)

    def fill_border(self, value, static=True):
        for gx in range(self.width):
            self.set(gx, 0, value, static=static)
            self.set(gx, self.height - 1, value, static=static)
        for gy in range(self.height):
            self.set(0, gy, value, static=static)
            self.set(self.width - 1, gy, value, static=static)

    def _bresenham(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        points = []
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def raycast(self, start_world, end_world, hit_state, free_state=FREE):
        start_cell = self.world_to_cell(*start_world)
        end_cell = self.world_to_cell(*end_world)
        if start_cell is None or end_cell is None:
            return False
        line = self._bresenham(start_cell, end_cell)
        # marcar livres até penúltimo
        for gx, gy in line[:-1]:
            self.set(gx, gy, free_state)
        gx_hit, gy_hit = line[-1]
        return self.set(gx_hit, gy_hit, hit_state)

    def plan_path(self, start_world, goal_world):
        start_cell = self.world_to_cell(*start_world)
        goal_cell = self.world_to_cell(*goal_world)
        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return []

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        came_from = {}
        g_score = {start_cell: 0}
        goal = goal_cell

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
            cx, cy = current
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if not self.in_bounds(nx, ny):
                    continue
                if self.get(nx, ny) == self.OBSTACLE:
                    continue
                tentative_g = g_score[current] + 1
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, 1e9):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        if goal not in came_from and goal != start_cell:
            return []

        path_cells = []
        node = goal
        while node != start_cell:
            path_cells.append(node)
            node = came_from.get(node, start_cell)
            if node == start_cell:
                break
        path_cells.reverse()
        return [self.cell_to_world(gx, gy) for gx, gy in path_cells]


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def color_from_rgb(r, g, b):
    """Determina cor do cubo baseado em RGB (0-1)."""
    if r > 0.5 and g < 0.5 and b < 0.5:
        return "red"
    elif g > 0.5 and r < 0.5 and b < 0.5:
        return "green"
    elif b > 0.5 and r < 0.5 and g < 0.5:
        return "blue"
    return None


class FuzzyNavigator:
    """Controlador Fuzzy para navegação com desvio de obstáculos."""

    def __init__(self, max_speed=0.2):
        self.max_speed = max_speed
        self.safety_margin = 0.4

    @staticmethod
    def _mu_close(x, threshold=0.45):
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_very_close(x, threshold=0.25):
        return max(0.0, min(1.0, (threshold - x) / threshold))

    @staticmethod
    def _mu_far(x, start=0.4, end=1.5):
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
        if a > math.radians(25):
            return 0.0
        return (math.radians(25) - a) / math.radians(20)

    @staticmethod
    def _mu_medium_angle(angle_rad):
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(10) or a > math.radians(60):
            return 0.0
        if a < math.radians(35):
            return (a - math.radians(10)) / math.radians(25)
        return (math.radians(60) - a) / math.radians(25)

    @staticmethod
    def _mu_big_angle(angle_rad):
        a = abs(wrap_angle(angle_rad))
        if a < math.radians(40):
            return 0.0
        if a > math.radians(90):
            return 1.0
        return (a - math.radians(40)) / math.radians(50)

    def check_known_obstacles(self, pose):
        """Verifica proximidade de obstáculos conhecidos."""
        x, y, yaw = pose
        min_dist_left = 2.0
        min_dist_right = 2.0
        min_dist_front = 2.0

        for ox, oy, radius in KNOWN_OBSTACLES:
            dx = ox - x
            dy = oy - y
            dist = math.hypot(dx, dy) - radius
            obs_angle = wrap_angle(math.atan2(dy, dx) - yaw)

            if abs(obs_angle) < math.radians(30):
                min_dist_front = min(min_dist_front, dist)
            elif obs_angle > 0 and obs_angle < math.radians(120):
                min_dist_left = min(min_dist_left, dist)
            elif obs_angle < 0 and obs_angle > math.radians(-120):
                min_dist_right = min(min_dist_right, dist)

        return min_dist_front, min_dist_left, min_dist_right

    def check_rear_clearance(self, omega, obs_front, obs_left, obs_right,
                               rear_sensors=None):
        """Check if rotation will cause rear collision using real rear sensors.

        Args:
            omega: Rotação desejada (positivo = esquerda, negativo = direita)
            obs_front, obs_left, obs_right: Distâncias do LIDAR frontal
            rear_sensors: Dict com 'rear', 'rear_left', 'rear_right' em metros (opcional)
        """
        MIN_REAR_CLEARANCE = 0.35  # 35cm mínimo atrás para girar
        MIN_SIDE_CLEARANCE = 0.40  # 40cm mínimo nas diagonais traseiras

        # Se temos sensores traseiros reais, usar eles
        if rear_sensors:
            rear = rear_sensors.get("rear", 2.0)
            rear_left = rear_sensors.get("rear_left", 2.0)
            rear_right = rear_sensors.get("rear_right", 2.0)

            # Obstáculo direto atrás - não pode dar ré
            if rear < MIN_REAR_CLEARANCE:
                return False, "rear_blocked"

            # Rotação para esquerda (omega > 0) -> traseira vai para direita
            if omega > 0.05:
                if rear_right < MIN_SIDE_CLEARANCE:
                    return False, "rear_right"
            # Rotação para direita (omega < 0) -> traseira vai para esquerda
            elif omega < -0.05:
                if rear_left < MIN_SIDE_CLEARANCE:
                    return False, "rear_left"
        else:
            # Fallback: usar LIDAR lateral como estimativa (comportamento antigo)
            ROBOT_HALF_LENGTH = 0.30
            SAFETY_MARGIN = 0.20
            MIN_CLEARANCE = ROBOT_HALF_LENGTH + SAFETY_MARGIN

            if omega > 0.05 and obs_right < MIN_CLEARANCE:
                return False, "rear_right"
            elif omega < -0.05 and obs_left < MIN_CLEARANCE:
                return False, "rear_left"

        # Frente muito perto para qualquer rotação
        if obs_front < 0.35:
            return False, "front_blocked"

        return True, None

    def compute(self, has_target, target_distance, target_angle,
                obs_front, obs_left, obs_right, pose=None):
        """Calcula comandos de velocidade usando lógica fuzzy."""
        if not has_target:
            return 0.0, 0.0, 0.25

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

        # Regra 1: Obstáculo muito perto -> PARAR e recuar/strafe (SEM rotação excessiva)
        if mu_front_very_close > 0.5:
            vx = -0.04
            vy = 0.10 * (1 if obs_left < obs_right else -1)  # Strafe prioritário
            omega = 0.0  # NÃO ROTACIONAR - apenas strafe para evitar spinning
            return vx, vy, omega

        # Regra 2: Obstáculo perto -> reduzir velocidade
        speed_reduction = 1.0 - (0.7 * mu_front_close)

        # Regra 3: Velocidade frontal - PRIORIZAR MOVIMENTO PARA FRENTE
        # Aumentado de 0.15 para 0.20, reduzido penalidade por ângulo
        vx = 0.20 * max(0.4, mu_target_far) * speed_reduction * (1.0 - 0.25 * mu_angle_big)

        # Regra 4: Perto do alvo e alinhado -> aproximação final
        if mu_target_close > 0.3 and mu_angle_small > 0.5:
            vx = 0.10 * speed_reduction

        # Regra 4b: Quando alinhado (ângulo pequeno), garantir movimento frontal
        if mu_angle_small > 0.6:
            vx = max(vx, 0.12 * speed_reduction)

        # Regra 5: Strafe para desvio de obstáculos
        vy = 0.10 * (mu_right_close - mu_left_close)

        # Regra 6: Rotação para alinhar - CONTROLE PROPORCIONAL
        # Usar ângulo normalizado ao invés de sinal binário para evitar overshooting
        angle_norm = target_angle / math.radians(90)  # normalizado entre -1 e 1
        angle_norm = max(-1.0, min(1.0, angle_norm))  # clamp

        # Ganhos reduzidos: 0.35, 0.18, 0.05 (antes: 0.8, 0.4, 0.1)
        omega_fuzzy = (
            0.35 * mu_angle_big +
            0.18 * mu_angle_medium +
            0.05 * (1.0 - mu_angle_small)
        )
        omega = omega_fuzzy * angle_norm

        # Regra 7: Ajuste por obstáculos laterais (reduzido)
        omega += 0.2 * (mu_right_close - mu_left_close)

        # Limitar velocidades
        vx = max(-self.max_speed, min(self.max_speed, vx))
        vy = max(-self.max_speed * 0.8, min(self.max_speed * 0.8, vy))
        omega = max(-0.6, min(0.6, omega))  # Reduzido de 1.0 para 0.6

        return vx, vy, omega


class YouBotController:
    def __init__(self):
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        self.base = Base(self.robot)
        self.arm = Arm(self.robot)
        self.gripper = Gripper(self.robot)

        self._log_times = {}

        # Camera com Recognition
        self.camera = self.robot.getDevice("camera")
        if self.camera:
            self.camera.enable(self.time_step)
            if self.camera.hasRecognition():
                self.camera.recognitionEnable(self.time_step)
                print("[INIT] Camera recognition enabled")
            else:
                print("[INIT] Camera has no recognition capability")

        # LiDAR principal
        self.lidar = self.robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            self.lidar.enablePointCloud()

        # Sensores de distância traseiros para detecção de colisão
        self.ds_rear = self.robot.getDevice("ds_rear")
        self.ds_rear_left = self.robot.getDevice("ds_rear_left")
        self.ds_rear_right = self.robot.getDevice("ds_rear_right")

        # Sensores de distância frontais para detecção durante APPROACH
        self.ds_front = self.robot.getDevice("ds_front")
        self.ds_front_left = self.robot.getDevice("ds_front_left")
        self.ds_front_right = self.robot.getDevice("ds_front_right")

        for ds in [self.ds_rear, self.ds_rear_left, self.ds_rear_right,
                   self.ds_front, self.ds_front_left, self.ds_front_right]:
            if ds:
                ds.enable(self.time_step)

        self.navigator = FuzzyNavigator()

        # Classificador de cores (RNA)
        self.color_classifier = ColorClassifier("model/color_model.onnx")

        self.pose = self._initial_pose()
        self.current_target = None
        self.current_color = None
        self.mode = "search"
        self.stage = 0
        self.stage_timer = 0.0
        self.collected = 0
        self.max_cubes = 15

        # Navegação lawnmower
        self.search_state = "forward"
        self.search_direction = 1
        self.turn_progress = 0.0

        # Timeout para cubo perdido durante approach
        self.lost_cube_timer = 0.0

        # Lock no cubo específico (ângulo e distância)
        self.locked_cube_angle = None
        self.locked_cube_distance = None  # Última distância conhecida do cubo

        self.box_positions = BOX_POSITIONS

        # Grid / caminho
        self.grid = OccupancyGrid(ARENA_CENTER, ARENA_SIZE, cell_size=0.12)
        self._seed_static_map()
        self._waypoints = []
        self._path_dirty = True
        self.active_goal = None
        self._max_cmd = 0.25

    def _log_throttled(self, key, msg, interval=1.5):
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
        self.grid.fill_border(OccupancyGrid.OBSTACLE, static=True)
        wall_inflate = 0.30  # Robot half-width (~0.29m) + safety margin for full body clearance
        # Obstáculos fixos
        for ox, oy, radius in KNOWN_OBSTACLES:
            self.grid.fill_disk(ox, oy, radius + wall_inflate, OccupancyGrid.OBSTACLE, static=True)
        # Caixas de depósito
        for pos in self.box_positions.values():
            self.grid.fill_disk(pos[0], pos[1], 0.22, OccupancyGrid.BOX, static=True)
        print(
            f"[GRID] size=({self.grid.width}x{self.grid.height}) cell={self.grid.cell_size:.2f} "
            f"bounds=({self.grid.min_x:.2f},{self.grid.min_y:.2f})-({self.grid.max_x:.2f},{self.grid.max_y:.2f})"
        )

    def _mark_cube_candidate(self, x, y):
        cell = self.grid.world_to_cell(x, y)
        if cell:
            self.grid.set(cell[0], cell[1], OccupancyGrid.CUBE, static=False)
            self._path_dirty = True

    def _set_goal(self, target):
        self.current_target = target
        self.active_goal = target
        self._waypoints = []
        self._path_dirty = True

    def _wall_clearances(self):
        x, y, _ = self.pose
        return (
            x - self.grid.min_x,
            self.grid.max_x - x,
            y - self.grid.min_y,
            self.grid.max_y - y,
        )

    def _enforce_boundary_safety(self, vx_cmd, vy_cmd, margin=0.25):
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
        vals = [vx, vy, omega]
        safe = []
        for v in vals:
            if v is None or not math.isfinite(v):
                safe.append(0.0)
            else:
                safe.append(max(-self._max_cmd, min(self._max_cmd, v)))
        self.base.move(safe[0], safe[1], safe[2])

    def _clamp_cmds(self, vx, vy, omega):
        omega = max(-0.5, min(0.5, omega))
        vx = max(-0.18, min(0.18, vx))
        vy = max(-0.14, min(0.14, vy))
        return vx, vy, omega

    def _distance_to_point(self, point):
        dx = point[0] - self.pose[0]
        dy = point[1] - self.pose[1]
        distance = math.hypot(dx, dy)
        angle = wrap_angle(math.atan2(dy, dx) - self.pose[2])
        return distance, angle

    def _replan_path(self):
        if not self.active_goal:
            self._waypoints = []
            self._path_dirty = False
            return
        path = self.grid.plan_path((self.pose[0], self.pose[1]), self.active_goal)
        self._waypoints = path
        self._path_dirty = False

    def _next_nav_point(self):
        if self._path_dirty:
            self._replan_path()
        if self._waypoints:
            return self._waypoints[0]
        return self.active_goal

    def _update_grid_from_lidar(self, lidar_info):
        if not lidar_info["points"]:
            return
        if any(math.isnan(v) for v in self.pose):
            return
        yaw = self.pose[2]
        origin = (self.pose[0], self.pose[1])
        hit_changed = False
        for r, angle in lidar_info["points"]:
            local_x = r * math.cos(angle)
            local_y = r * math.sin(angle)
            wx = origin[0] + math.cos(yaw) * local_x - math.sin(yaw) * local_y
            wy = origin[1] + math.sin(yaw) * local_x + math.cos(yaw) * local_y
            if math.isnan(wx) or math.isnan(wy):
                continue
            if self.grid.raycast(origin, (wx, wy), OccupancyGrid.OBSTACLE):
                hit_changed = True
        if hit_changed:
            self._path_dirty = True

    def _initial_pose(self):
        """Obtém pose inicial do nó do robô no Webots."""
        try:
            self_node = self.robot.getSelf()
            if self_node:
                # Usar getPosition() para coordenadas precisas
                pos = self_node.getPosition()
                # Extrair yaw da matriz de orientação
                orient = self_node.getOrientation()
                # orient é matriz 3x3 [R00,R01,R02,R10,R11,R12,R20,R21,R22]
                # yaw = atan2(R10, R00)
                yaw = math.atan2(orient[3], orient[0])
                return [pos[0], pos[1], yaw]
        except Exception as e:
            print(f"[POSE] Erro ao obter pose inicial: {e}")
        return [0.0, 0.0, 0.0]

    def _get_ground_truth_pose(self):
        """Obtém pose real do Webots (ground truth)."""
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
        """Integra odometria com proteção contra NaN."""
        # Proteção contra valores NaN na entrada
        if math.isnan(vx) or math.isnan(vy) or math.isnan(omega):
            return

        # Verificar se pose atual está corrompida
        if any(math.isnan(v) for v in self.pose):
            # Recuperar do ground truth
            gt = self._get_ground_truth_pose()
            if gt:
                self.pose = gt
                print(f"[POSE] Recuperado do ground truth: ({gt[0]:.2f}, {gt[1]:.2f})")
            else:
                self.pose = [0.0, 0.0, 0.0]
            return

        yaw = self.pose[2]
        dx_world = math.cos(yaw) * vx - math.sin(yaw) * vy
        dy_world = math.sin(yaw) * vx + math.cos(yaw) * vy

        new_x = self.pose[0] + dx_world * dt
        new_y = self.pose[1] + dy_world * dt
        new_yaw = wrap_angle(self.pose[2] + omega * dt)

        # Validar resultado antes de aplicar
        if not (math.isnan(new_x) or math.isnan(new_y) or math.isnan(new_yaw)):
            self.pose[0] = new_x
            self.pose[1] = new_y
            self.pose[2] = new_yaw
        else:
            # Recuperar do ground truth se resultado é NaN
            gt = self._get_ground_truth_pose()
            if gt:
                self.pose = gt

    def _process_recognition(self, lock_color=None, lock_angle=None):
        """Usa Camera Recognition API para detectar cubos.

        Args:
            lock_color: Se definido, só retorna cubos dessa cor
            lock_angle: Se definido (rad), prioriza cubo mais próximo desse ângulo
        """
        if not self.camera or not self.camera.hasRecognition():
            return None

        objects = self.camera.getRecognitionObjects()
        if not objects:
            return None

        best = None
        best_score = float('inf')

        for obj in objects:
            pos = obj.getPosition()  # posição relativa à câmera
            colors = obj.getColors()

            dist = math.sqrt(pos[0]**2 + pos[1]**2)
            angle = math.atan2(pos[1], pos[0]) if pos[0] != 0 else 0

            # Determinar cor
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

            # Filtrar por cor
            if lock_color and color != lock_color:
                continue

            # Score: prioriza MAIS PERTO (distância é o principal)
            score = dist

            # Se lock_angle ativo, forte preferência por cubo naquele ângulo
            if lock_angle is not None:
                angle_diff = abs(wrap_angle(angle - lock_angle))
                if angle_diff < math.radians(10):  # mesmo cubo (tolerância 10°)
                    score -= 10.0  # bônus forte
                else:
                    score += 5.0  # penalidade forte - ignora outros cubos

            if score < best_score and dist < 2.5:  # max 2.5m
                best_score = score
                best = {
                    "color": color,
                    "distance": dist,
                    "angle": angle,
                    "position": pos,
                }

        if best:
            self._log_throttled(
                "recognition",
                f"[RECOG] cor={best['color']} dist={best['distance']:.2f} ang={math.degrees(best['angle']):.1f}°",
                interval=0.5,
            )

        return best

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

        return {"front": front, "left": left, "right": right, "points": points}

    def _process_rear_sensors(self):
        """Processa sensores de distância traseiros.

        Converte valor bruto do sensor (0-1000) para distância em metros.
        Lookup table: 0->0m, 50->0.05m, 1000->1.0m (linear)

        Returns:
            dict com rear, rear_left, rear_right em metros
        """
        def raw_to_meters(raw_value):
            if raw_value is None or math.isnan(raw_value):
                return 2.0  # Default seguro
            # Inverso da lookup table: raw 0-1000 -> dist 0-1.0m
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

    def _process_front_sensors(self):
        """Processa sensores de distância frontais.

        Returns:
            dict com front, front_left, front_right em metros
        """
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
        """Navegação lawnmower com proteção de parede."""
        front_dist = lidar_info["front"]
        left_dist = lidar_info["left"]
        right_dist = lidar_info["right"]

        OBSTACLE_THRESHOLD = 0.40
        WALL_PROXIMITY = 0.35
        TURN_SPEED = 0.30  # Reduzido de 0.35 para mais controle
        FORWARD_SPEED = 0.10
        TURN_DURATION = math.pi / 2.5 / TURN_SPEED  # Turn ~72° ao invés de 90°

        # Verificar proximidade de parede
        left_wall, right_wall, bottom_wall, top_wall = self._wall_clearances()
        near_wall = min(left_wall, right_wall, bottom_wall, top_wall) < WALL_PROXIMITY
        very_near_wall = min(left_wall, right_wall, bottom_wall, top_wall) < 0.25

        obstacle_ahead = self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD)

        # Se muito perto da parede, usar strafe ao invés de rotação
        if very_near_wall:
            # Determinar direção de escape
            if left_wall < right_wall:
                return 0.0, -0.08, 0.0  # Strafe para direita
            elif right_wall < left_wall:
                return 0.0, 0.08, 0.0   # Strafe para esquerda
            elif bottom_wall < top_wall:
                return 0.08, 0.0, 0.0   # Avançar
            else:
                return -0.08, 0.0, 0.0  # Recuar

        # Velocidade de rotação reduzida perto de paredes
        actual_turn_speed = TURN_SPEED * (0.6 if near_wall else 1.0)

        if self.search_state == "forward":
            if obstacle_ahead:
                if left_dist > right_dist:
                    self.search_direction = 1
                else:
                    self.search_direction = -1
                self.search_state = "turn_start"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            else:
                vy_correct = 0.0
                omega_correct = 0.0
                # Correção lateral mais suave
                if left_dist < 0.30:
                    vy_correct = -0.05
                    omega_correct = -0.08
                elif right_dist < 0.30:
                    vy_correct = 0.05
                    omega_correct = 0.08
                return FORWARD_SPEED, vy_correct, omega_correct

        elif self.search_state == "turn_start":
            # Verificar se ainda é seguro girar
            if very_near_wall:
                self.search_state = "forward"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0

            self.turn_progress += dt
            if self.turn_progress >= TURN_DURATION:
                self.search_state = "turn_mid"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, actual_turn_speed * self.search_direction

        elif self.search_state == "turn_mid":
            self.turn_progress += dt
            if self.turn_progress >= 0.6:  # Reduzido de 0.8
                self.search_state = "turn_end"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            if self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD):
                self.search_state = "turn_end"
                self.turn_progress = 0.0
            return FORWARD_SPEED * 0.8, 0.0, 0.0  # Velocidade reduzida

        elif self.search_state == "turn_end":
            # Verificar se ainda é seguro girar
            if very_near_wall:
                self.search_state = "forward"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0

            self.turn_progress += dt
            if self.turn_progress >= TURN_DURATION:
                self.search_state = "forward"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            return 0.0, 0.0, actual_turn_speed * self.search_direction

        return 0.0, 0.0, 0.0

    def _start_grasp(self):
        self.mode = "grasp"
        self.stage = 0
        self.stage_timer = 0.0
        self.base.reset()
        print(f"[GRASP] Iniciando captura - cor={self.current_color}")

    def _start_drop(self):
        self.mode = "drop"
        self.stage = 0
        self.stage_timer = 0.0
        self.base.reset()

    def _handle_grasp(self, dt):
        self.stage_timer += dt

        # Calcular distância de avanço baseado na última distância conhecida
        # Fórmula: forward = cam_dist - 0.12 (deixar ~12cm para o gripper alcançar)
        # O gripper tem alcance de 5-10cm quando aberto
        if not hasattr(self, '_grasp_forward_time'):
            dist = self.locked_cube_distance if self.locked_cube_distance else 0.25
            # Buffer GRANDE: gripper estava passando o cubo consistentemente
            # Camera está 0.27m do centro, gripper em FRONT_FLOOR está BEM à frente
            # Parar com margem de 0.12m para gripper alcançar cubo sem ultrapassar
            forward_needed = dist - 0.12  # Buffer grande = para bem antes do cubo
            forward_needed = max(0.05, min(0.25, forward_needed))  # Clamp entre 5-25cm
            self._grasp_forward_time = forward_needed / 0.04  # Tempo a 4cm/s (velocidade real)
            self._grasp_samples = []  # Para verificar gripper
            print(f"[GRASP] Distância para avanço: {forward_needed:.2f}m ({self._grasp_forward_time:.1f}s) (cubo a {dist:.2f}m)")

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
            # Avançar LENTAMENTE pelo tempo calculado
            # Com correção de ângulo em tempo real para rastrear o cubo CORRETO
            omega_cmd = 0.0

            # CRÍTICO: Usar _process_recognition com filtros para rastrear cubo locked
            # Evita rastrear cubos errados durante avanço
            locked_recognition = self._process_recognition(
                lock_color=self.current_color,
                lock_angle=self.locked_cube_angle
            )
            if locked_recognition:
                cam_angle = locked_recognition["angle"]
                # Correção suave: P-controller no ângulo
                omega_cmd = -cam_angle * 0.8
                omega_cmd = max(-0.3, min(0.3, omega_cmd))

            self._safe_move(0.04, 0.0, omega_cmd)
            if self.stage_timer >= self._grasp_forward_time:
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0
                print("[GRASP] Parado, fechando garra...")

        elif self.stage == 3:
            # Fechar gripper e esperar
            self.gripper.grip()
            if self.stage_timer >= 1.2:
                self.stage += 1
                self.stage_timer = 0.0
                self._grasp_samples = []  # Reset samples

        elif self.stage == 4:
            # Levantar e coletar amostras do sensor para verificação
            self.arm.set_height(Arm.FRONT_PLATE)
            self.gripper.grip()  # Manter fechado

            # Coletar múltiplas amostras do sensor de posição dos dedos
            left, right = self.gripper.finger_positions()
            if left is not None:
                self._grasp_samples.append(left)
            if right is not None:
                self._grasp_samples.append(right)

            if self.stage_timer >= 1.5:
                # Usar método has_object do gripper (threshold=0.003 para 3cm cube)
                has_obj = self.gripper.has_object(threshold=0.003)

                # Log detalhado
                if self._grasp_samples:
                    avg_pos = sum(self._grasp_samples) / len(self._grasp_samples)
                    max_pos = max(self._grasp_samples) if self._grasp_samples else 0
                    print(f"[GRASP] Verificação: avg={avg_pos:.4f}, max={max_pos:.4f}, has_object={has_obj}")
                else:
                    print(f"[GRASP] Sem amostras, has_object={has_obj}")

                self.base.reset()
                self._grasp_verified = has_obj
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 5:
            # Verificar resultado e decidir próximo passo
            has_obj = getattr(self, '_grasp_verified', True)

            # Cleanup
            if hasattr(self, '_grasp_forward_time'):
                del self._grasp_forward_time
            if hasattr(self, '_grasp_samples'):
                del self._grasp_samples
            if hasattr(self, '_grasp_verified'):
                del self._grasp_verified

            if has_obj:
                # SUCESSO - ir para caixa
                self.collected += 1
                print(f"[GRASP] Cubo capturado! Total: {self.collected}/{self.max_cubes}")

                self.locked_cube_distance = None
                if self.active_goal:
                    cell = self.grid.world_to_cell(*self.active_goal)
                    if cell:
                        self.grid.set(cell[0], cell[1], OccupancyGrid.FREE, overwrite_static=False)
                        self._path_dirty = True

                # CRÍTICO: Sincronizar pose com ground truth ANTES de calcular rota
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
                    print(f"[TO_BOX] Indo para caixa {color_key.upper()} em {target_box}")
                    print(f"[TO_BOX] Pose atual: ({self.pose[0]:.2f}, {self.pose[1]:.2f}, {math.degrees(self.pose[2]):.1f}°) dist={dist_to_box:.2f}m ang={math.degrees(angle_to_box):.1f}°")
                    self._set_goal(target_box)
                else:
                    fallback = list(self.box_positions.values())[0]
                    print(f"[TO_BOX] Cor '{self.current_color}' não encontrada, usando fallback {fallback}")
                    self._set_goal(fallback)
            else:
                # FALHA - dar RÉ antes de tentar novamente
                print("[GRASP] FALHA - garra vazia, dando ré para tentar novamente")
                self.gripper.release()
                self.stage += 1  # Ir para stage 6 (reverse)
                self.stage_timer = 0.0

        elif self.stage == 6:
            # Stage de RÉ - recuar para tentar novamente
            self.arm.set_height(Arm.FRONT_PLATE)
            self._safe_move(-0.06, 0.0, 0.0)  # Ré lenta
            if self.stage_timer >= 2.0:  # Recuar por 2s = ~12cm
                self.base.reset()
                print("[GRASP] Ré completa, voltando ao search")
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
        """Sequência de depósito do cubo no box.

        Stage 0: Baixar braço de RESET para FRONT_FLOOR (posição sobre box)
        Stage 1: Avançar até ficar DENTRO do box
        Stage 2: Soltar cubo (abrir gripper)
        Stage 3: Recuar um pouco ainda com braço baixo
        Stage 4: Levantar braço para RESET
        Stage 5: Recuar mais para longe do box
        Stage 6: Girar para evitar re-detectar cubo
        Stage 7: Finalizar e voltar ao search
        """
        self.stage_timer += dt

        if self.stage == 0:
            # Baixar braço para posição de depósito (FRONT_FLOOR)
            # Esta posição coloca o gripper na altura do box
            self.arm.set_height(Arm.FRONT_FLOOR)
            self.gripper.grip()
            if self.stage_timer >= 2.0:
                self.stage += 1
                self.stage_timer = 0.0
                print("[DROP] Braço baixo, avançando sobre box...")

        elif self.stage == 1:
            # Avançar até ficar BEM sobre o box
            # Já estamos a ~0.20m (lidar), precisamos avançar mais ~10cm
            self._safe_move(0.04, 0.0, 0.0)
            self.gripper.grip()
            if self.stage_timer >= 2.5:  # 0.04 m/s × 2.5s = 10cm
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0
                print("[DROP] Sobre o box, soltando cubo...")

        elif self.stage == 2:
            # Soltar o cubo - já estamos sobre o box
            self.gripper.release()
            if self.stage_timer >= 1.0:
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 3:
            # Recuar um pouco ainda com braço baixo
            self._safe_move(-0.06, 0.0, 0.0)
            if self.stage_timer >= 1.0:  # ~6cm de ré
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 4:
            # Levantar braço para não bater no box ao recuar mais
            self.arm.set_height(Arm.RESET)
            if self.stage_timer >= 1.5:
                self.stage += 1
                self.stage_timer = 0.0
                print("[DROP] Recuando...")

        elif self.stage == 5:
            # Recuar para longe do box
            self._safe_move(-0.10, 0.0, 0.0)
            if self.stage_timer >= 2.0:  # -0.10 m/s × 2.0s = -20cm
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 6:
            # Girar para evitar re-detectar o cubo depositado
            self._safe_move(0.0, 0.0, 0.4)
            if self.stage_timer >= 1.5:  # ~60° de rotação
                self.base.reset()
                self.stage += 1
                self.stage_timer = 0.0

        elif self.stage == 7:
            print(f"[DROP] Cubo depositado na caixa {self.current_color}")
            self.mode = "search"
            self.current_target = None
            self.active_goal = None
            self._waypoints = []
            self._path_dirty = True
            self.current_color = None
            self.locked_cube_angle = None
            self.locked_cube_distance = None
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
        print("[RUN] YouBot controller started")

        # Move forward at spawn to clear wall (robot spawns at x=-3.99 near left wall)
        print("[INIT] Moving forward to clear spawn wall...")
        for _ in range(50):  # ~0.8s at 16ms timestep
            if self.robot.step(self.time_step) == -1:
                return
            self.base.move(0.12, 0.0, 0.0)
        self.base.reset()

        # Sincronizar pose com ground truth após movimento inicial
        gt = self._get_ground_truth_pose()
        if gt:
            self.pose = gt
            print(f"[INIT] Spawn complete. Pose: ({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.1f}°)")
        else:
            print("[INIT] Spawn complete (sem ground truth)")

        while self.robot.step(self.time_step) != -1:
            dt = self.time_step / 1000.0

            if self.collected >= self.max_cubes:
                print("[DONE] Todos os cubos coletados!")
                self.base.reset()
                break

            # Odometry com sincronização periódica
            vx_odo, vy_odo, omega_odo = self.base.compute_odometry(dt)
            self._integrate_pose(vx_odo, vy_odo, omega_odo, dt)

            # Sincronizar com ground truth periodicamente
            # TO_BOX precisa de sync mais frequente (0.5s), outros modos 2s
            if not hasattr(self, '_gt_sync_timer'):
                self._gt_sync_timer = 0.0
            self._gt_sync_timer += dt

            sync_interval = 0.5 if self.mode == "to_box" else 2.0
            pose_invalid = any(math.isnan(v) for v in self.pose)
            if pose_invalid or self._gt_sync_timer >= sync_interval:
                gt = self._get_ground_truth_pose()
                if gt:
                    if pose_invalid:
                        print(f"[POSE] Pose inválida, sincronizando: ({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.1f}°)")
                    self.pose = gt
                self._gt_sync_timer = 0.0

            # Sensores
            lidar_info = self._process_lidar()
            rear_info = self._process_rear_sensors()
            front_info = self._process_front_sensors()
            self._update_grid_from_lidar(lidar_info)

            # Lock na cor e ângulo durante approach
            lock_color = self.current_color if self.mode == "approach" else None
            lock_angle = self.locked_cube_angle if self.mode == "approach" else None
            recognition = self._process_recognition(lock_color=lock_color, lock_angle=lock_angle)

            # ===== MODO SEARCH =====
            if self.mode == "search":
                # Travar omega se perto de parede para evitar 360 batendo
                left, right, bottom, top = self._wall_clearances()
                near_wall = min(left, right, bottom, top) < 0.35

                if recognition:
                    self.current_color = recognition["color"]
                    self.locked_cube_angle = recognition["angle"]  # LOCK no ângulo
                    self.mode = "approach"
                    self.lost_cube_timer = 0.0
                    print(f"[SEARCH] Cubo detectado (cor={self.current_color}) dist={recognition['distance']:.2f}m, iniciando approach")
                    continue

                # Navegação lawnmower
                vx_cmd, vy_cmd, omega_cmd = self._search_navigation(lidar_info, dt)
                vx_cmd, vy_cmd, omega_cmd = self._clamp_cmds(vx_cmd, vy_cmd, omega_cmd if not near_wall else 0.0)
                vx_cmd, vy_cmd = self._enforce_boundary_safety(vx_cmd, vy_cmd)
                self._safe_move(vx_cmd, vy_cmd, omega_cmd)
                continue

            # ===== MODO APPROACH =====
            if self.mode == "approach":
                # PRIMEIRO: Verificar obstáculo frontal com sensores de distância
                # Isso detecta obstáculos que LIDAR pode não ver bem de perto
                front_obstacle = min(
                    front_info["front"],
                    front_info["front_left"],
                    front_info["front_right"],
                    lidar_info["front"]
                )

                # Obstáculo detectado - PARAR e desviar
                if front_obstacle < 0.20:
                    # Obstáculo muito perto - recuar e tentar contornar
                    print(f"[APPROACH] OBSTÁCULO FRONTAL! dist={front_obstacle:.2f}m, desviando...")

                    # Determinar direção de escape
                    if front_info["front_left"] > front_info["front_right"]:
                        vy_cmd = 0.08  # Strafe esquerda
                        omega_cmd = 0.2
                    else:
                        vy_cmd = -0.08  # Strafe direita
                        omega_cmd = -0.2

                    self._safe_move(-0.04, vy_cmd, omega_cmd)

                    # Após alguns ciclos, abortar approach se não conseguir
                    if not hasattr(self, '_approach_obstacle_count'):
                        self._approach_obstacle_count = 0
                    self._approach_obstacle_count += 1

                    if self._approach_obstacle_count > 30:  # ~0.5s
                        print("[APPROACH] Não conseguiu desviar, voltando ao search")
                        self.mode = "search"
                        self.locked_cube_angle = None
                        self.locked_cube_distance = None
                        self.current_color = None
                        self._approach_obstacle_count = 0
                    continue
                else:
                    # Reset contador quando não há obstáculo
                    if hasattr(self, '_approach_obstacle_count'):
                        self._approach_obstacle_count = 0

                if recognition:
                    self.lost_cube_timer = 0.0
                    cam_dist = recognition["distance"]
                    cam_angle = recognition["angle"]

                    # Atualizar distância e ângulo conhecidos
                    self.locked_cube_distance = cam_dist
                    self.locked_cube_angle = cam_angle

                    # Chegou perto o suficiente E alinhado -> grasp
                    grasp_distance = 0.32
                    grasp_angle_max = math.radians(10)

                    if cam_dist < grasp_distance and abs(cam_angle) < grasp_angle_max:
                        print(f"[APPROACH] Cubo a {cam_dist:.2f}m, ângulo={math.degrees(cam_angle):.1f}°, iniciando grasp")
                        self._start_grasp()
                        continue
                    elif cam_dist < grasp_distance and abs(cam_angle) >= grasp_angle_max:
                        omega_cmd = -cam_angle * 1.0
                        omega_cmd = max(-0.4, min(0.4, omega_cmd))
                        self._safe_move(0.0, 0.0, omega_cmd)
                        self._log_throttled("approach_align_close", f"[APPROACH] Perto mas desalinhado: ângulo={math.degrees(cam_angle):.1f}°", 1.0)
                        continue

                    # ESTRATÉGIA: Primeiro ALINHAR, depois AVANÇAR
                    angle_threshold = math.radians(5)

                    if abs(cam_angle) > angle_threshold:
                        omega_cmd = -cam_angle * 0.8
                        omega_cmd = max(-0.3, min(0.3, omega_cmd))
                        vx_cmd = 0.0
                        self._log_throttled("approach_align", f"[APPROACH] Alinhando: ângulo={math.degrees(cam_angle):.1f}°, dist={cam_dist:.2f}m", 1.0)
                    else:
                        vx_cmd = 0.08
                        omega_cmd = -cam_angle * 1.2
                        omega_cmd = max(-0.2, min(0.2, omega_cmd))
                        self._log_throttled("approach_advance", f"[APPROACH] Avançando: dist={cam_dist:.2f}m, ângulo={math.degrees(cam_angle):.1f}°", 1.0)

                    self._safe_move(vx_cmd, 0.0, omega_cmd)
                    continue
                else:
                    self.lost_cube_timer += dt

                    if self.locked_cube_distance is not None and self.locked_cube_distance < 0.35:
                        if self.lost_cube_timer > 0.3:
                            print(f"[APPROACH] Cubo perdido a {self.locked_cube_distance:.2f}m, iniciando grasp")
                            self._start_grasp()
                            continue

                    if self.lost_cube_timer > 4.0:
                        print("[APPROACH] Timeout - voltando ao search")
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

            # ===== MODO GRASP =====
            if self.mode == "grasp":
                self._handle_grasp(dt)
                continue

            # ===== MODO TO_BOX =====
            if self.mode == "to_box":
                # MANTER gripper fechado e braço RECOLHIDO (RESET = posição alta)
                # Isso evita colisões do braço com obstáculos durante navegação
                self.gripper.grip()
                self.arm.set_height(Arm.RESET)

                # Navegação com A* waypoints + LIDAR para desvio de obstáculos
                if self.active_goal:
                    # === A* WAYPOINT NAVIGATION ===
                    # Use waypoints from A* path planner instead of direct-to-goal
                    waypoint = self._next_nav_point()
                    if waypoint:
                        distance, angle = self._distance_to_point(waypoint)
                        # Pop waypoint when reached (within 0.25m)
                        if distance < 0.25 and self._waypoints:
                            self._waypoints.pop(0)
                            waypoint = self._next_nav_point()
                            if waypoint:
                                distance, angle = self._distance_to_point(waypoint)
                    else:
                        # No waypoints, go direct to goal
                        distance, angle = self._distance_to_point(self.active_goal)

                    # Always check final goal distance for DROP trigger
                    final_distance, final_angle = self._distance_to_point(self.active_goal)
                    lidar_front = lidar_info["front"]

                    # === FINAL APPROACH DETECTION ===
                    # When close to goal with box detected by LIDAR, switch to final approach mode
                    # Removed angle requirement - robot can approach from any angle
                    in_final_approach = (
                        final_distance < 1.2 and
                        lidar_front < 0.80
                    )

                    # Log periódico de navegação
                    if not hasattr(self, '_tobox_log_timer'):
                        self._tobox_log_timer = 0.0
                    self._tobox_log_timer += dt
                    if self._tobox_log_timer >= 2.0:
                        wp_info = f"wp={len(self._waypoints)}" if self._waypoints else "direct"
                        fa_info = " [FINAL]" if in_final_approach else ""
                        print(f"[TO_BOX] pose=({self.pose[0]:.2f},{self.pose[1]:.2f}) goal={self.active_goal} dist={final_distance:.2f}m ang={math.degrees(final_angle):.1f}° [{wp_info}]{fa_info}")
                        self._tobox_log_timer = 0.0

                    # === DROP TRIGGER ===
                    # Aproximar BEM PERTO do box antes de depositar
                    # lidar < 0.25m garante que estamos quase encostando no box
                    # PlasticFruitBox é ~0.60m x 0.40m x 0.22m de altura
                    if in_final_approach and 0.10 < lidar_front < 0.25:
                        print(f"[TO_BOX] DROP TRIGGER! lidar={lidar_front:.2f}m, goal_dist={final_distance:.2f}m, ang={math.degrees(final_angle):.1f}°")
                        if hasattr(self, '_tobox_log_timer'):
                            del self._tobox_log_timer
                        self._start_drop()
                        continue

                    # === FINAL APPROACH MODE ===
                    # Continuar aproximando até LIDAR < 0.25m
                    if in_final_approach:
                        # Alinhar com o goal e aproximar devagar
                        omega_cmd = -final_angle * 0.8
                        omega_cmd = max(-0.3, min(0.3, omega_cmd))

                        # Velocidade proporcional à distância do box
                        # Mais perto = mais devagar
                        if lidar_front < 0.40:
                            vx_cmd = 0.03  # Bem devagar quando perto
                        else:
                            vx_cmd = 0.06

                        vy_cmd = 0.0
                        self._log_throttled("tobox_final", f"[TO_BOX] Final approach: lidar={lidar_front:.2f}m, goal_dist={final_distance:.2f}m", 0.8)
                        vx_cmd, vy_cmd = self._enforce_boundary_safety(vx_cmd, vy_cmd)
                        self._safe_move(vx_cmd, vy_cmd, omega_cmd)
                        continue

                    # ===== DESVIO DE OBSTÁCULOS - ROTAÇÃO + STRAFE =====
                    obs_front = lidar_info["front"]
                    obs_left = lidar_info["left"]
                    obs_right = lidar_info["right"]

                    # Verificar obstáculos conhecidos também
                    known_front, known_left, known_right = self.navigator.check_known_obstacles(self.pose)
                    obs_front = min(obs_front, known_front)
                    obs_left = min(obs_left, known_left)
                    obs_right = min(obs_right, known_right)

                    # Determinar lado de escape baseado no espaço disponível
                    # Prioriza o lado com MAIS espaço para contornar o obstáculo
                    escape_left = obs_left > obs_right

                    # Stuck detection e clearing timer
                    if not hasattr(self, '_emergency_counter'):
                        self._emergency_counter = 0
                    if not hasattr(self, '_clearing_timer'):
                        self._clearing_timer = 0.0

                    # PARADA DE EMERGÊNCIA: obstáculo muito perto (< 0.30m)
                    # Now with rear clearance check to prevent rear collisions during escape
                    if obs_front < 0.30:
                        self._emergency_counter += 1
                        self._clearing_timer = 1.5  # Continuar evitando por 1.5s após sair da emergência

                        # Calcular rotação desejada para escape
                        desired_omega = 0.45 if escape_left else -0.45

                        # Verificar rear clearance com sensores traseiros reais
                        rear_safe, rear_issue = self.navigator.check_rear_clearance(
                            desired_omega, obs_front, obs_left, obs_right, rear_info
                        )

                        # Se preso por muito tempo (>25 ciclos), manobra agressiva de escape
                        if self._emergency_counter > 25:
                            vx_cmd = -0.10  # Ré forte
                            if rear_safe:
                                omega_cmd = 0.6 if escape_left else -0.6
                                vy_cmd = 0.15 if escape_left else -0.15
                            else:
                                # Priorizar strafe quando rear em risco
                                omega_cmd = 0.0
                                vy_cmd = 0.20 if escape_left else -0.20
                            self._log_throttled("tobox_stuck", f"[TO_BOX] PRESO! Escape: rear_safe={rear_safe}", 0.5)
                        else:
                            # Escape normal: ré + rotation/strafe baseado em rear clearance
                            vx_cmd = -0.06  # Recuar
                            if rear_safe:
                                omega_cmd = desired_omega
                                vy_cmd = 0.12 if escape_left else -0.12
                            else:
                                # Usar strafe predominante quando rear em risco
                                omega_cmd = 0.0
                                vy_cmd = 0.18 if escape_left else -0.18
                            self._log_throttled("tobox_emergency", f"[TO_BOX] Emergência! obs_front={obs_front:.2f}m, rear_safe={rear_safe}", 1.0)

                        vx_cmd, vy_cmd = self._enforce_boundary_safety(vx_cmd, vy_cmd)
                        self._safe_move(vx_cmd, vy_cmd, omega_cmd)
                        continue
                    else:
                        # Reset contador quando não está em emergência
                        self._emergency_counter = 0

                    # DESVIO PREVENTIVO: obstáculo próximo (< 0.50m) OU ainda em clearing
                    # Now includes rear clearance check to prevent rear wheel collisions
                    if obs_front < 0.50 or self._clearing_timer > 0:
                        if self._clearing_timer > 0:
                            self._clearing_timer -= dt

                        # Calcular rotação desejada
                        desired_omega = 0.35 if escape_left else -0.35

                        # CRÍTICO: Verificar clearance traseira com sensores reais
                        rear_safe, rear_issue = self.navigator.check_rear_clearance(
                            desired_omega, obs_front, obs_left, obs_right, rear_info
                        )

                        if rear_safe:
                            # Seguro rotacionar - movimento normal de contorno
                            omega_cmd = desired_omega
                            vy_cmd = 0.12 if escape_left else -0.12
                            vx_cmd = 0.04
                        else:
                            # Risco de colisão traseira - usar strafe prioritário
                            if rear_issue == "rear_right":
                                # Traseira bateria à direita - strafe esquerda, não girar à esquerda
                                vy_cmd = 0.15
                                omega_cmd = min(0.0, -0.15)  # Só permite giro à direita (leve)
                            elif rear_issue == "rear_left":
                                # Traseira bateria à esquerda - strafe direita, não girar à direita
                                vy_cmd = -0.15
                                omega_cmd = max(0.0, 0.15)  # Só permite giro à esquerda (leve)
                            else:
                                # Frente bloqueada - ré + strafe
                                vy_cmd = 0.12 if escape_left else -0.12
                                omega_cmd = 0.0
                            vx_cmd = 0.02  # Avançar bem devagar quando traseira em risco
                            self._log_throttled("tobox_rear_risk", f"[TO_BOX] Rear collision risk: {rear_issue}", 1.0)

                        self._log_throttled("tobox_preventive", f"[TO_BOX] Contornando: obs_front={obs_front:.2f}m, lado={'L' if escape_left else 'R'}, rear_safe={rear_safe}", 1.5)
                        vx_cmd, vy_cmd = self._enforce_boundary_safety(vx_cmd, vy_cmd)
                        self._safe_move(vx_cmd, vy_cmd, omega_cmd)
                        continue

                    # NAVEGAÇÃO NORMAL: Car-like motion with proportional control
                    # Allow forward motion even with larger angles to prevent oscillation
                    angle_threshold = math.radians(25)  # Increased from 12° to 25°

                    # Calculate rotation command
                    omega_cmd = -angle * 0.6
                    omega_cmd = max(-0.40, min(0.40, omega_cmd))

                    # Check rear clearance with real sensors before rotating
                    rear_safe, rear_issue = self.navigator.check_rear_clearance(
                        omega_cmd, obs_front, obs_left, obs_right, rear_info
                    )

                    if not rear_safe:
                        # Rear collision risk - prefer forward + strafe instead of rotation
                        if rear_issue == "rear_right":
                            # Rear hitting right - strafe left and reduce right rotation
                            vy_cmd = 0.10
                            omega_cmd = max(0.0, omega_cmd)  # Only allow left rotation
                            self._log_throttled("tobox_rear_right", "[TO_BOX] Rear clearance risk RIGHT, strafe left", 1.0)
                        elif rear_issue == "rear_left":
                            # Rear hitting left - strafe right and reduce left rotation
                            vy_cmd = -0.10
                            omega_cmd = min(0.0, omega_cmd)  # Only allow right rotation
                            self._log_throttled("tobox_rear_left", "[TO_BOX] Rear clearance risk LEFT, strafe right", 1.0)
                        else:
                            # Front blocked - use strafe primarily
                            vy_cmd = 0.08 if obs_left > obs_right else -0.08
                            omega_cmd *= 0.5  # Reduce rotation
                            self._log_throttled("tobox_front_blocked", "[TO_BOX] Front blocked, using strafe", 1.0)
                        vx_cmd = 0.03  # Reduced forward speed
                    else:
                        # Car-like arc motion: always move forward while rotating
                        # Scale vx by alignment quality (1.0 when aligned, 0.4 at 25°)
                        alignment_quality = 1.0 - (abs(angle) / angle_threshold)
                        alignment_quality = max(0.4, min(1.0, alignment_quality))

                        # Base speed scaled by alignment and distance
                        base_speed = min(0.12, distance * 0.15)
                        vx_cmd = base_speed * alignment_quality
                        vx_cmd = max(0.04, vx_cmd)  # Minimum 4cm/s to maintain progress

                        # Lateral correction for side obstacles
                        if obs_left < 0.35:
                            vy_cmd = -0.05
                        elif obs_right < 0.35:
                            vy_cmd = 0.05
                        else:
                            vy_cmd = 0.0

                    vx_cmd, vy_cmd = self._enforce_boundary_safety(vx_cmd, vy_cmd)
                    self._safe_move(vx_cmd, vy_cmd, omega_cmd)
                else:
                    # Sem goal definido - voltar ao search
                    print("[TO_BOX] Sem goal definido, voltando ao search")
                    self.mode = "search"
                continue

            # ===== MODO DROP =====
            if self.mode == "drop":
                self._handle_drop(dt)
                continue


if __name__ == "__main__":
    controller = YouBotController()
    controller.run()
