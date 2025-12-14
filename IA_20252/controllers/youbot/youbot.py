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

# ===== ROTAS BASE PARA NAVEGAÇÃO =====
# Waypoints estratégicos para navegação segura
# O robô deve seguir esses corredores para evitar obstáculos

# Ponto inicial do robô (spawn)
SPAWN_POSITION = (-3.91, 0.0)

# Posições dos obstáculos para referência:
# A = (0.6, 0.0)   - CENTRO (perigo principal!)
# E = (-1.02, 0.75)  - Superior esquerdo
# F = (-1.02, -0.74) - Inferior esquerdo
# Corredor central entre E e F: y entre -0.5 e 0.5

def get_route_to_box(current_pos, destination_color):
    """
    Retorna lista de waypoints para ir de current_pos até o box do destination_color.
    
    Estratégia CAR-LIKE com passagem COMPLETA de obstáculos:
    - Ir RETO pelo corredor central até passar E/F
    - INCLINAR para desviar
    - MANTER inclinação até a RODA TRASEIRA passar o obstáculo (~30cm extra)
    - Só então ALINHAR com o box
    
    Robô: ~58cm comprimento, ~38cm largura, roda traseira ~29cm do centro.
    
    Obstáculos:
    - E: (-1.02, 0.75) - superior esquerdo
    - F: (-1.02, -0.74) - inferior esquerdo  
    - A: (0.6, 0.0) - central, raio ~0.25m
    """
    x, y = current_pos[0], current_pos[1]
    waypoints = []
    
    box_pos = BOX_POSITIONS.get(destination_color)
    if not box_pos:
        return [current_pos]
    
    if destination_color == "blue":
        # BLUE em (0.48, -1.62)
        # Obstáculo F está em (-1.02, -0.74), borda direita em ~(-0.77, -0.74)
        
        # 1. Ir RETO pelo corredor central até PASSAR F (x > -0.5)
        if x < -0.5:
            waypoints.append((-0.45, 0.0))  # Passar borda de F
        
        # 2. Continuar RETO mais um pouco para roda traseira passar
        waypoints.append((-0.15, 0.0))  # +30cm para roda traseira
        
        # 3. INCLINAR para SUL - início suave
        waypoints.append((0.0, -0.35))
        
        # 4. MANTER inclinação descendo - andar RETO inclinado
        waypoints.append((0.15, -0.70))
        waypoints.append((0.25, -1.00))  # Continuar reto inclinado
        
        # 5. Agora alinhar com X do box
        waypoints.append((0.35, -1.25))
        waypoints.append((0.48, -1.35))  # Mesmo X do box
        
        # 6. Destino final
        waypoints.append(box_pos)
        
    elif destination_color == "green":
        # GREEN em (0.48, 1.58)
        # Obstáculo E está em (-1.02, 0.75), borda direita em ~(-0.77, 0.75)
        
        # 1. Ir RETO pelo corredor central até PASSAR E (x > -0.5)
        if x < -0.5:
            waypoints.append((-0.45, 0.0))
        
        # 2. Continuar RETO mais um pouco para roda traseira passar
        waypoints.append((-0.15, 0.0))
        
        # 3. INCLINAR para NORTE - início suave
        waypoints.append((0.0, 0.35))
        
        # 4. MANTER inclinação subindo - andar RETO inclinado
        waypoints.append((0.15, 0.70))
        waypoints.append((0.25, 1.00))
        
        # 5. Agora alinhar com X do box
        waypoints.append((0.35, 1.25))
        waypoints.append((0.48, 1.35))
        
        # 6. Destino final
        waypoints.append(box_pos)
        
    elif destination_color == "red":
        # RED em (2.31, 0.01)
        # Obstáculo A está em (0.6, 0.0), raio ~0.25m
        # Borda: x de 0.35 a 0.85, y de -0.25 a 0.25
        # Precisa passar x > 1.15 para roda traseira limpar
        
        # 1. Ir RETO pelo corredor central
        if x < -0.5:
            waypoints.append((-0.45, 0.0))
        
        # 2. Continuar RETO até perto de A
        waypoints.append((0.0, 0.0))
        
        # 3. INCLINAR para desviar de A - escolher lado baseado em y
        if y >= 0:
            # Desviar por CIMA de A (y positivo)
            waypoints.append((0.25, 0.45))   # Início da inclinação
            waypoints.append((0.50, 0.55))   # MANTER inclinado passando A
            waypoints.append((0.75, 0.55))   # Continuar RETO inclinado
            waypoints.append((1.00, 0.50))   # Ainda inclinado
            waypoints.append((1.25, 0.35))   # Roda traseira passou A!
        else:
            # Desviar por BAIXO de A (y negativo)
            waypoints.append((0.25, -0.45))
            waypoints.append((0.50, -0.55))
            waypoints.append((0.75, -0.55))
            waypoints.append((1.00, -0.50))
            waypoints.append((1.25, -0.35))
        
        # 4. AGORA voltar ao centro gradualmente
        waypoints.append((1.55, 0.15 if y >= 0 else -0.15))
        waypoints.append((1.85, 0.01))
        
        # 5. Alinhar reto com o box
        waypoints.append((2.05, 0.01))
        
        # 6. Destino final
        waypoints.append(box_pos)
    
    return waypoints


def get_return_route(current_pos, from_color):
    """
    Retorna waypoints SIMPLES para voltar ao spawn.
    O robô vai para FRENTE (não ré) - é mais seguro e simples.
    
    Estratégia: Sair do box → Centro → Spawn
    """
    _ = current_pos  # Não usado, mas mantido para assinatura
    waypoints = []
    
    # 1. Primeiro, afastar-se do box (ir para o centro do mapa)
    if from_color == "blue":
        # Estava no BLUE (0.48, -1.62) - ir para norte
        waypoints.append((0.30, -1.00))
        waypoints.append((0.15, -0.50))
        waypoints.append((0.0, 0.0))
    elif from_color == "green":
        # Estava no GREEN (0.48, 1.58) - ir para sul
        waypoints.append((0.30, 1.00))
        waypoints.append((0.15, 0.50))
        waypoints.append((0.0, 0.0))
    elif from_color == "red":
        # Estava no RED (2.31, 0.01) - ir para oeste
        waypoints.append((1.50, 0.0))
        waypoints.append((1.00, 0.0))
        waypoints.append((0.50, 0.0))
        waypoints.append((0.0, 0.0))
    
    # 2. Do centro, ir para o spawn (reto para oeste)
    waypoints.append((-0.50, 0.0))
    waypoints.append((-1.50, 0.0))
    waypoints.append((-2.50, 0.0))
    waypoints.append(SPAWN_POSITION)
    
    return waypoints


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
        """A* pathfinding com fallback para células bloqueadas."""
        start_cell = self.world_to_cell(*start_world)
        goal_cell = self.world_to_cell(*goal_world)
        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return []

        # IMPORTANTE: Se a célula inicial está bloqueada, encontrar célula livre mais próxima
        if self.get(*start_cell) == self.OBSTACLE:
            start_cell = self._find_nearest_free_cell(start_cell)
            if start_cell is None:
                return []  # Não há célula livre próxima
        
        # Se a célula final está bloqueada (exceto BOX que é o destino), encontrar adjacente
        goal_val = self.get(*goal_cell)
        if goal_val == self.OBSTACLE:
            goal_cell = self._find_nearest_free_cell(goal_cell)
            if goal_cell is None:
                return []

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])  # Euclidean para 8-dir

        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        came_from = {}
        g_score = {start_cell: 0}
        goal = goal_cell

        visited = set()
        max_iterations = self.width * self.height  # Evitar loops infinitos
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                break
            cx, cy = current
            # 8-directional movement for smoother paths
            neighbors = [
                (cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1),
                (cx + 1, cy + 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx - 1, cy - 1)
            ]
            for nx, ny in neighbors:
                if not self.in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                cell_val = self.get(nx, ny)
                # Evitar OBSTACLE e BOX (exceto se for o goal)
                if cell_val == self.OBSTACLE:
                    continue
                if cell_val == self.BOX and (nx, ny) != goal:
                    continue
                # Custo: 1.0 para ortogonal, 1.414 para diagonal
                is_diag = abs(nx - cx) + abs(ny - cy) == 2
                step_cost = 1.414 if is_diag else 1.0
                tentative_g = g_score[current] + step_cost
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

    def _find_nearest_free_cell(self, blocked_cell, max_radius=5):
        """Encontra a célula FREE mais próxima de uma célula bloqueada."""
        bx, by = blocked_cell
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # Só borda do quadrado
                    nx, ny = bx + dx, by + dy
                    if self.in_bounds(nx, ny):
                        val = self.get(nx, ny)
                        if val != self.OBSTACLE and val != self.BOX:
                            return (nx, ny)
        return None


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

        # Lateral and Rear LiDARS
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

        # Sensores de distância traseiros
        self.ds_rear = self.robot.getDevice("ds_rear")
        self.ds_rear_left = self.robot.getDevice("ds_rear_left")
        self.ds_rear_right = self.robot.getDevice("ds_rear_right")

        # Sensores de distância laterais
        self.ds_left = self.robot.getDevice("ds_left")
        self.ds_right = self.robot.getDevice("ds_right")

        # Sensores de distância frontais
        self.ds_front = self.robot.getDevice("ds_front")
        self.ds_front_left = self.robot.getDevice("ds_front_left")
        self.ds_front_right = self.robot.getDevice("ds_front_right")

        # Habilitar todos os sensores de distância
        for ds in [self.ds_rear, self.ds_rear_left, self.ds_rear_right,
                   self.ds_left, self.ds_right,
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
        # Inflação conservadora - confiamos no APF para desvio fino
        # Robot ~0.58m x 0.38m, half-width ~0.19m, half-length ~0.29m
        wall_inflate = 0.25  # Menor inflação = mais caminhos, APF cuida do resto
        # Obstáculos fixos (WoodenBoxes ~0.5m x 0.5m, radius 0.25m)
        for ox, oy, radius in KNOWN_OBSTACLES:
            self.grid.fill_disk(ox, oy, radius + wall_inflate, OccupancyGrid.OBSTACLE, static=True)
        # Caixas de depósito - marcar pequeno para A* chegar bem perto
        for pos in self.box_positions.values():
            self.grid.fill_disk(pos[0], pos[1], 0.20, OccupancyGrid.BOX, static=True)
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

    def _move_ackermann(self, speed, steering_angle):
        """
        Executes an Ackermann-like movement using the omnidirectional base.
        Constraint: vy = 0 (no strafing/skidding)
        omega = (speed * tan(steering_angle)) / L
        """
        L = 0.4  # Approximate wheelbase
        
        # Limit steering angle
        max_steer = math.radians(45)
        steering_angle = max(-max_steer, min(max_steer, steering_angle))
        
        if abs(speed) < 0.001:
            omega = 0.0
        else:
            omega = (speed * math.tan(steering_angle)) / L
            
        self.base.move(speed, 0.0, omega)

    def _safe_move(self, vx, vy, omega):
        # Forward to ackermann if strafe is zero or negligible
        if abs(vy) < 0.001:
            # Reverse engineer steering angle from omega
            # omega = speed * tan(delta) / L  => tan(delta) = omega * L / speed
            L = 0.4
            if abs(vx) > 0.001:
                tan_delta = (omega * L) / vx
                delta = math.atan(tan_delta)
                self._move_ackermann(vx, delta)
                return

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
            
        robot_x, robot_y, robot_yaw = self.pose
        origin = (robot_x, robot_y)
        
        hit_changed = False
        
        # Pre-calculate cos/sin for rotation
        cy = math.cos(robot_yaw)
        sy = math.sin(robot_yaw)
        
        for lx, ly in lidar_info["points"]:
            # Transform from Robot Frame to World Frame
            wx = robot_x + lx * cy - ly * sy
            wy = robot_y + lx * sy + ly * cy
            
            if math.isnan(wx) or math.isnan(wy):
                continue
                
            # Raycast from robot center to hit point
            if self.grid.raycast(origin, (wx, wy), OccupancyGrid.OBSTACLE):
                hit_changed = True
                
        if hit_changed:
            self._path_dirty = True

    def _get_pure_pursuit_point(self, lookahead=0.6):
        """Encontra ponto no caminho atual para Pure Pursuit."""
        if not self._waypoints:
            return self.active_goal
            
        rx, ry, _ = self.pose
        
        # Encontrar ponto mais próximo no caminho
        closest_dist = float('inf')
        closest_idx = 0
        for i, (wx, wy) in enumerate(self._waypoints):
            d = math.hypot(wx - rx, wy - ry)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i
                
        # Procurar ponto à frente pela distância de lookahead
        target = self._waypoints[-1] # Default: final
        
        for i in range(closest_idx, len(self._waypoints)):
            wx, wy = self._waypoints[i]
            d = math.hypot(wx - rx, wy - ry)
            if d >= lookahead:
                target = (wx, wy)
                break
                
        return target

    def _check_rear_safety(self, rear_info, margin=0.30):
        """Verifica se a ré é segura usando LIDAR e sensores de distância."""
        if not rear_info:
            return True
        
        # Combinar leituras do LIDAR traseiro e sensores de distância
        r = rear_info.get("rear", 2.0)
        rl = rear_info.get("rear_left", 2.0)
        rr = rear_info.get("rear_right", 2.0)
        
        # Margem mais tolerante para evitar falsos positivos
        if r < margin * 0.8:  # Centro traseiro
            return False
        if rl < margin * 0.6 or rr < margin * 0.6:  # Cantos
            return False
        return True

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
            # Reduced log frequency to avoid spam (interval 0.5 -> 10.0 or remove)
            # Only log if specifically debugging
            pass 
            # self._log_throttled(
            #    "recognition",
            #    f"[RECOG] cor={best['color']} dist={best['distance']:.2f} ang={math.degrees(best['angle']):.1f}°",
            #    interval=5.0,
            # )

        return best

    def _process_lidar(self):
        """Process data from all LIDARs and fuse into a single point cloud in robot frame."""
        fused_points = []
        
        # Helper to process single lidar
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
            
            # Find minimum distance in the center of the FOV (approx 40 degrees window)
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
                
                # Angle relative to sensor
                alpha = -fov / 2.0 + i * angle_step
                
                # Sensor frame: x forward, y left
                sx = r * math.cos(alpha)
                sy = r * math.sin(alpha)
                
                # Robot frame transformation
                rx = dx + sx * math.cos(dtheta) - sy * math.sin(dtheta)
                ry = dy + sx * math.sin(dtheta) + sy * math.cos(dtheta)
                
                sensor_points.append((rx, ry))
                
            return min_dist, sensor_points

        # 1. Front Lidar (0.28, 0, 0)
        front_dist, p_front = process_sensor(self.lidar, 0.28, 0.0, 0.0)
        fused_points.extend(p_front)
        
        # 2. Rear Lidar (-0.29, 0, PI)
        rear_dist, p_rear = process_sensor(self.lidar_rear, -0.29, 0.0, math.pi)
        fused_points.extend(p_rear)
        
        # 3. Left Lidar (0, 0.22, PI/2)
        left_dist, p_left = process_sensor(self.lidar_left, 0.0, 0.22, math.pi/2)
        fused_points.extend(p_left)
        
        # 4. Right Lidar (0, -0.22, -PI/2)
        right_dist, p_right = process_sensor(self.lidar_right, 0.0, -0.22, -math.pi/2)
        fused_points.extend(p_right)
        
        return {
            "front": front_dist,
            "rear": rear_dist,
            "left": left_dist,
            "right": right_dist,
            "points": fused_points # List of (x, y) in robot frame
        }

    def _process_rear_sensors(self):
        """Processa sensores de distância traseiros."""
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
        """Processa sensores de distância laterais."""
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
        """Navegação lawnmower com comportamento Car-like."""
        front_dist = lidar_info["front"]
        left_dist = lidar_info["left"]
        right_dist = lidar_info["right"]

        OBSTACLE_THRESHOLD = 0.50
        FORWARD_SPEED = 0.15
        
        # Verificar proximidade de parede
        left_wall, right_wall, bottom_wall, top_wall = self._wall_clearances()
        min_wall = min(left_wall, right_wall, bottom_wall, top_wall)
        
        obstacle_ahead = self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD)

        # Se muito perto da parede, dar ré
        if min_wall < 0.25:
            return -0.08, 0.0, 0.0

        if self.search_state == "forward":
            if obstacle_ahead:
                if left_dist > right_dist:
                    self.search_direction = 1  # Left
                else:
                    self.search_direction = -1 # Right
                self.search_state = "turn"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            else:
                # Wall following / Centering
                omega_correct = 0.0
                if left_dist < 0.40:
                    omega_correct = -0.15 # Steer Right
                elif right_dist < 0.40:
                    omega_correct = 0.15 # Steer Left
                
                return FORWARD_SPEED, 0.0, omega_correct

        elif self.search_state == "turn":
            self.turn_progress += dt
            # Car turn: Move forward while steering
            if self.turn_progress >= 2.5: # Turn duration
                self.search_state = "forward"
                self.turn_progress = 0.0
                return 0.0, 0.0, 0.0
            
            # Check if cleared
            if not self._check_obstacle_ahead(front_dist, OBSTACLE_THRESHOLD + 0.2) and self.turn_progress > 1.0:
                 self.search_state = "forward"
                 self.turn_progress = 0.0
                 return 0.0, 0.0, 0.0

            return 0.10, 0.0, 0.4 * self.search_direction

        # Fallback
        self.search_state = "forward"
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
            print(f"[DROP] Cubo depositado na caixa {self.current_color}. Iniciando retorno ao spawn.")
            
            # Usar os waypoints percorridos na ida, invertidos (RÉ PURA)
            if hasattr(self, '_last_route_taken') and self._last_route_taken:
                # Inverter a rota (excluir o box, que é o último)
                self._return_waypoints = list(reversed(self._last_route_taken[:-1]))
                self._return_waypoints.append(SPAWN_POSITION)
            else:
                # Fallback: rota simples
                self._return_waypoints = [SPAWN_POSITION]
            
            self._return_waypoint_idx = 0
            self.mode = "return_to_spawn"
            print(f"[RETURN] Rota: {len(self._return_waypoints)} waypoints (RÉ PURA)")
            
            # Limpar estado do drop
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
            lateral_info = self._process_lateral_sensors()
            front_info = self._process_front_sensors()
            
            # Consolidar informação de sensores com LIDAR (usar o menor valor)
            if "rear" in lidar_info:
                rear_info["rear"] = min(rear_info["rear"], lidar_info["rear"])
            
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
                # MANTER gripper fechado e braço RECOLHIDO
                self.gripper.grip()
                self.arm.set_height(Arm.RESET)
                
                # Inicialização do estado
                if not hasattr(self, 'tobox_state'):
                    self.tobox_state = 0
                    self._tobox_maneuver_timer = 0.0
                    self._route_waypoints = None
                    self._current_waypoint_idx = 0

                if not self.active_goal:
                    print("[TO_BOX] Sem goal definido, voltando ao search")
                    self.mode = "search"
                    continue

                # ===== CALCULAR ROTA BASE =====
                if self._route_waypoints is None:
                    self._route_waypoints = get_route_to_box(
                        (self.pose[0], self.pose[1]), 
                        self.current_color
                    )
                    self._current_waypoint_idx = 0
                    self._last_route_taken = self._route_waypoints.copy()  # Salvar para retorno
                    print("[TO_BOX] ===== ROTA CALCULADA =====")
                    print(f"[TO_BOX] Cor: {self.current_color} | Destino: {BOX_POSITIONS.get(self.current_color)}")
                    print(f"[TO_BOX] Waypoints ({len(self._route_waypoints)}):")
                    for i, wp in enumerate(self._route_waypoints):
                        print(f"  [{i+1}] ({wp[0]:.2f}, {wp[1]:.2f})")
                    print("[TO_BOX] =============================")
                
                # Log periódico da posição (a cada ~2 segundos)
                if not hasattr(self, '_last_pos_log_time'):
                    self._last_pos_log_time = 0.0
                current_time = self.robot.getTime()
                if current_time - self._last_pos_log_time > 2.0:
                    wp_idx = self._current_waypoint_idx
                    wp_total = len(self._route_waypoints)
                    wp_name = f"WP{wp_idx+1}/{wp_total}" if wp_idx < wp_total else "BOX"
                    print(f"[TO_BOX] Pos: ({self.pose[0]:.2f}, {self.pose[1]:.2f}, {math.degrees(self.pose[2]):.0f}°) → {wp_name}")
                    self._last_pos_log_time = current_time

                # ===== SENSORES 360° =====
                front_obs = min(lidar_info["front"], front_info["front"])
                fl_obs = min(lidar_info["front"], front_info.get("front_left", 2.0))
                fr_obs = min(lidar_info["front"], front_info.get("front_right", 2.0))
                rear_obs = min(lidar_info["rear"], rear_info.get("rear", 2.0))
                rear_min = min(rear_obs, rear_info.get("rear_left", 2.0), rear_info.get("rear_right", 2.0))
                left_obs = min(lidar_info["left"], lateral_info.get("left", 2.0))
                right_obs = min(lidar_info["right"], lateral_info.get("right", 2.0))

                # ===== WAYPOINT ATUAL =====
                if self._current_waypoint_idx < len(self._route_waypoints):
                    current_target = self._route_waypoints[self._current_waypoint_idx]
                else:
                    current_target = self.active_goal
                
                dist_to_waypoint, angle_to_waypoint = self._distance_to_point(current_target)
                dist_to_box, angle_to_box = self._distance_to_point(self.active_goal)
                
                # Verificar se chegou no waypoint atual
                is_final_waypoint = (self._current_waypoint_idx >= len(self._route_waypoints) - 1)
                is_pre_final = (self._current_waypoint_idx == len(self._route_waypoints) - 2)
                
                # Threshold menor para waypoints intermediários, maior para pré-final e final
                if is_final_waypoint:
                    waypoint_threshold = 0.50  # Box - transição para alinhamento
                elif is_pre_final:
                    waypoint_threshold = 0.30  # Penúltimo - precisão maior
                else:
                    waypoint_threshold = 0.40  # Intermediários
                
                if dist_to_waypoint < waypoint_threshold and not is_final_waypoint:
                    self._current_waypoint_idx += 1
                    next_wp = self._route_waypoints[self._current_waypoint_idx] if self._current_waypoint_idx < len(self._route_waypoints) else self.active_goal
                    print(f"[TO_BOX] ✓ Waypoint {self._current_waypoint_idx}/{len(self._route_waypoints)} alcançado → próximo: ({next_wp[0]:.2f}, {next_wp[1]:.2f})")
                    continue

                # State 0: Navegação por waypoints
                if self.tobox_state == 0:
                    # Sensores
                    min_front = min(front_obs, fl_obs, fr_obs)
                    left_blocked = left_obs < 0.20
                    right_blocked = right_obs < 0.20
                    rear_clear = rear_min > 0.35
                    
                    # ===== MODO APROXIMAÇÃO DO BOX =====
                    # Ativar quando: últimos 3 waypoints OU dist_to_box < 1.0m
                    # Isso garante que o robô NÃO trata o box como obstáculo
                    waypoints_remaining = len(self._route_waypoints) - self._current_waypoint_idx
                    approaching_box = waypoints_remaining <= 3 or dist_to_box < 1.0
                    
                    # ===== SE APROXIMANDO DO BOX: IGNORAR OBSTÁCULO FRONTAL =====
                    if approaching_box:
                        # Transição para alinhamento
                        if dist_to_box < 0.55 or min_front < 0.18:
                            print(f"[TO_BOX] ✓ Chegou ao box (dist={dist_to_box:.2f}m). Iniciando alinhamento.")
                            self._safe_move(0.0, 0.0, 0.0)
                            self.tobox_state = 1
                            self._tobox_maneuver_timer = 0.0
                            self._align_start_time = None
                            continue
                        
                        # Seguir waypoint - IGNORAR sensor frontal (é o box!)
                        cmd_omega = -angle_to_waypoint * 1.2
                        cmd_omega = max(-0.35, min(0.35, cmd_omega))
                        cmd_speed = 0.12
                        
                        # Só reagir a laterais (outros objetos, não o box)
                        if left_blocked:
                            cmd_omega = min(cmd_omega - 0.15, -0.20)
                        if right_blocked:
                            cmd_omega = max(cmd_omega + 0.15, 0.20)
                        
                        self._safe_move(cmd_speed, 0.0, cmd_omega)
                        continue

                    # ===== NAVEGAÇÃO NORMAL (longe do box) =====
                    EMERGENCY_STOP = 0.22
                    FRONT_DANGER = 0.35
                    FRONT_WARN = 0.55
                    LATERAL_WARN = 0.35
                    
                    front_left_close = fl_obs < FRONT_WARN
                    front_right_close = fr_obs < FRONT_WARN
                    
                    cmd_speed = 0.0
                    cmd_omega = 0.0
                    
                    # ===== PRIORIDADE 1: EMERGÊNCIA FRONTAL =====
                    if min_front < EMERGENCY_STOP:
                        self._tobox_maneuver_timer += dt
                        print(f"[TO_BOX] 🛑 EMERGÊNCIA! Frente a {min_front:.2f}m!")
                        
                        if self._tobox_maneuver_timer > 0.2 and rear_clear:
                            cmd_speed = -0.08
                            # Virar para o lado LIVRE
                            if not left_blocked and (right_blocked or left_obs > right_obs):
                                cmd_omega = 0.4
                            elif not right_blocked:
                                cmd_omega = -0.4
                        else:
                            cmd_speed = 0.0
                    
                    # ===== PRIORIDADE 2: PERIGO FRONTAL =====
                    elif min_front < FRONT_DANGER:
                        self._tobox_maneuver_timer += dt
                        print(f"[TO_BOX] ⚠ Perigo frontal a {min_front:.2f}m!")
                        
                        if rear_clear and self._tobox_maneuver_timer < 2.0:
                            cmd_speed = -0.10
                            # Virar para o lado LIVRE
                            if not left_blocked and (right_blocked or left_obs > right_obs + 0.1):
                                cmd_omega = 0.45
                            elif not right_blocked:
                                cmd_omega = -0.45
                            else:
                                cmd_omega = 0.0  # Ambos bloqueados - só ré
                        else:
                            self._tobox_maneuver_timer = 0.0
                            cmd_speed = 0.0
                            if not left_blocked and (right_blocked or left_obs > right_obs):
                                cmd_omega = 0.5
                            elif not right_blocked:
                                cmd_omega = -0.5
                    
                    # ===== PRIORIDADE 3: DESVIO PREVENTIVO =====
                    elif min_front < FRONT_WARN or front_left_close or front_right_close:
                        self._tobox_maneuver_timer = 0.0
                        
                        # Decidir direção baseado em espaço E bloqueios
                        if left_blocked:
                            # Esquerda bloqueada - SÓ pode virar direita
                            cmd_omega = -0.4
                        elif right_blocked:
                            # Direita bloqueada - SÓ pode virar esquerda
                            cmd_omega = 0.4
                        elif front_left_close and not front_right_close:
                            cmd_omega = -0.35
                        elif front_right_close and not front_left_close:
                            cmd_omega = 0.35
                        elif left_obs > right_obs + 0.1:
                            cmd_omega = 0.3
                        elif right_obs > left_obs + 0.1:
                            cmd_omega = -0.3
                        else:
                            # Espaços similares - usar waypoint
                            cmd_omega = -angle_to_waypoint * 0.6
                            cmd_omega = max(-0.35, min(0.35, cmd_omega))
                        
                        # Velocidade proporcional ao espaço
                        cmd_speed = 0.03 + 0.06 * (min_front / FRONT_WARN)
                    
                    # ===== PRIORIDADE 4: NAVEGAÇÃO NORMAL =====
                    else:
                        self._tobox_maneuver_timer = 0.0
                        
                        # Seguir waypoint
                        cmd_omega = -angle_to_waypoint * 1.5
                        MAX_TURN = 0.45
                        cmd_omega = max(-MAX_TURN, min(MAX_TURN, cmd_omega))
                        
                        if abs(angle_to_waypoint) > math.pi/2:
                            cmd_speed = 0.03
                            cmd_omega = -angle_to_waypoint * 0.5
                            cmd_omega = max(-0.30, min(0.30, cmd_omega))
                        else:
                            # Repulsão lateral suave
                            if left_obs < LATERAL_WARN:
                                cmd_omega -= 0.15 * (LATERAL_WARN - left_obs) / LATERAL_WARN
                            if right_obs < LATERAL_WARN:
                                cmd_omega += 0.15 * (LATERAL_WARN - right_obs) / LATERAL_WARN
                            cmd_omega = max(-MAX_TURN, min(MAX_TURN, cmd_omega))
                            
                            cmd_speed = 0.13
                            turn_penalty = 1.0 - min(0.25, abs(cmd_omega) / MAX_TURN)
                            cmd_speed *= turn_penalty
                    
                    # ===== VERIFICAÇÃO FINAL: NUNCA VIRAR PARA LADO BLOQUEADO =====
                    # Esta verificação é ABSOLUTA e sobrescreve qualquer decisão anterior
                    if left_blocked and cmd_omega > 0:
                        # Quer virar esquerda mas esquerda bloqueada!
                        print(f"[TO_BOX] ⛔ BLOQUEIO ESQ a {left_obs:.2f}m! Forçando direita.")
                        cmd_omega = -0.4
                        cmd_speed = min(cmd_speed, 0.02)
                    
                    if right_blocked and cmd_omega < 0:
                        # Quer virar direita mas direita bloqueada!
                        print(f"[TO_BOX] ⛔ BLOQUEIO DIR a {right_obs:.2f}m! Forçando esquerda.")
                        cmd_omega = 0.4
                        cmd_speed = min(cmd_speed, 0.02)
                    
                    # Se AMBOS bloqueados, só andar reto devagar ou ré
                    if left_blocked and right_blocked:
                        print(f"[TO_BOX] ⛔ AMBOS LADOS bloqueados! L={left_obs:.2f} R={right_obs:.2f}")
                        cmd_omega = 0.0
                        if min_front > 0.35:
                            cmd_speed = 0.03  # Reto devagar
                        elif rear_clear:
                            cmd_speed = -0.08  # Ré
                        else:
                            cmd_speed = 0.0  # Parado
                    
                    self._safe_move(cmd_speed, 0.0, cmd_omega)

                # State 1: Alinhamento PERMISSIVO com a Caixa
                # Não precisa ser perfeito - só precisa estar razoavelmente de frente
                elif self.tobox_state == 1:
                    if self._align_start_time is None:
                        self._align_start_time = self.robot.getTime()
                    
                    align_elapsed = self.robot.getTime() - self._align_start_time
                    
                    # Ângulo alvo para cada box
                    if self.current_color == "green":
                        target_heading = math.pi / 2  # 90°
                    elif self.current_color == "blue":
                        target_heading = -math.pi / 2  # -90°
                    else:  # red
                        target_heading = 0.0
                    
                    heading_error = wrap_angle(target_heading - self.pose[2])
                    
                    # Log inicial
                    if align_elapsed < 0.1:
                        print(f"[TO_BOX] Alinhamento: erro={math.degrees(heading_error):.0f}°")
                    
                    # TOLERÂNCIA PERMISSIVA: 25° ou 2s timeout
                    # Se já está razoavelmente alinhado, ir direto para aproximação
                    ALIGN_TOLERANCE = 0.44  # ~25°
                    ALIGN_TIMEOUT = 2.0     # 2 segundos
                    
                    if abs(heading_error) < ALIGN_TOLERANCE:
                        print(f"[TO_BOX] ✓ Alinhamento OK (erro={math.degrees(heading_error):.0f}°). Aproximação.")
                        self.tobox_state = 2
                        self._approach_start_time = None
                        continue
                    
                    if align_elapsed > ALIGN_TIMEOUT:
                        print(f"[TO_BOX] Timeout alinhamento. Prosseguindo (erro={math.degrees(heading_error):.0f}°).")
                        self.tobox_state = 2
                        self._approach_start_time = None
                        continue

                    # Rotação suave para alinhar
                    omega = heading_error * 0.5
                    omega = max(-0.25, min(0.25, omega))
                    
                    self._safe_move(0.0, 0.0, omega)

                # State 2: Aproximação Final até o BOX
                # IMPORTANTE: Usar sensor frontal para detectar distância ao box
                elif self.tobox_state == 2:
                    ds_val = front_info["front"]
                    
                    # Timeout para evitar loop infinito
                    if self._approach_start_time is None:
                        self._approach_start_time = self.robot.getTime()
                        print(f"[TO_BOX] Iniciando aproximação. Sensor frontal: {ds_val:.3f}m")
                    
                    approach_elapsed = self.robot.getTime() - self._approach_start_time
                    
                    # Log periódico da aproximação
                    if int(approach_elapsed * 2) % 2 == 0 and approach_elapsed > 0.5:
                        if not hasattr(self, '_last_approach_log') or self._last_approach_log != int(approach_elapsed):
                            self._last_approach_log = int(approach_elapsed)
                            print(f"[TO_BOX] Aproximando: ds={ds_val:.3f}m, dist_box={dist_to_box:.2f}m")
                    
                    # Condição de DROP:
                    # - Sensor frontal detecta box perto (35-60cm) - zona segura (NÃO bater!)
                    # - OU timeout (4s) e estamos razoavelmente perto
                    drop_ready = (0.35 < ds_val < 0.60)
                    timeout_drop = (approach_elapsed > 4.0 and dist_to_box < 0.65)
                    
                    if drop_ready or timeout_drop:
                        if timeout_drop and not drop_ready:
                            print(f"[TO_BOX] Timeout aproximação (ds={ds_val:.3f}m). DROP.")
                        else:
                            print(f"[TO_BOX] ✓ Posição final (ds={ds_val:.3f}m). Executando DROP.")
                        self._safe_move(0.0, 0.0, 0.0)  # PARAR antes do drop
                        self._start_drop()
                        # Cleanup de atributos TO_BOX (mantém _last_route_taken para retorno)
                        for attr in ['tobox_state', '_tobox_maneuver_timer', '_align_start_time', 
                                     '_approach_start_time', '_route_waypoints', '_current_waypoint_idx',
                                     '_last_pos_log_time', '_last_approach_log']:
                            if hasattr(self, attr):
                                delattr(self, attr)
                        continue

                    # PROTEÇÃO: Se muito perto, PARAR e fazer drop
                    if ds_val < 0.35:
                        print(f"[TO_BOX] ⚠ Muito perto ({ds_val:.3f}m). DROP de segurança.")
                        self._safe_move(0.0, 0.0, 0.0)
                        self._start_drop()
                        for attr in ['tobox_state', '_tobox_maneuver_timer', '_route_waypoints', 
                                     '_current_waypoint_idx', '_last_pos_log_time']:
                            if hasattr(self, attr):
                                delattr(self, attr)
                        continue

                    # Aproximar lentamente mantendo alinhamento
                    # Velocidade proporcional à distância
                    if ds_val > 0.50:
                        speed = 0.08
                    elif ds_val > 0.35:
                        speed = 0.05
                    else:
                        speed = 0.03
                    
                    # Correção de ângulo suave
                    omega = -angle_to_box * 1.0
                    omega = max(-0.20, min(0.20, omega))
                    
                    self._safe_move(speed, 0.0, omega)
                
                continue

            # ===== MODO DROP =====
            if self.mode == "drop":
                self._handle_drop(dt)
                continue

            # ===== MODO RETURN_TO_SPAWN =====
            # Estratégia SIMPLES:
            # Fase 0: Ré RETA por 4 segundos para se afastar do box
            # Fase 1: Girar até apontar para o spawn
            # Fase 2: Ir PARA FRENTE até o spawn
            if self.mode == "return_to_spawn":
                self.arm.set_height(Arm.RESET)
                self.gripper.release()
                
                current_time = self.robot.getTime()
                
                # Inicialização
                if not hasattr(self, '_return_phase'):
                    self._return_phase = 0  # 0=ré, 1=giro, 2=navegação
                    self._return_start = current_time
                    self._return_log_time = 0.0
                    print("[RETURN] Fase 0: Recuando do box (4s)...")
                
                # Sensores
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
                
                # ===== FASE 0: RÉ RETA =====
                if self._return_phase == 0:
                    phase_elapsed = current_time - self._return_start
                    
                    if phase_elapsed < 4.0 and min_rear > 0.25:
                        # Ré reta, sem rotação
                        self._safe_move(-0.12, 0.0, 0.0)
                    else:
                        # Ir para próxima fase
                        if phase_elapsed >= 4.0:
                            print(f"[RETURN] Fase 1: Girando para spawn (ang={math.degrees(angle_to_spawn):.0f}°)...")
                        else:
                            print("[RETURN] Traseira bloqueada. Fase 1: Girando...")
                        self._return_phase = 1
                        self._return_start = current_time
                    continue
                
                # ===== FASE 1: GIRO NO LUGAR =====
                elif self._return_phase == 1:
                    phase_elapsed = current_time - self._return_start
                    
                    # Se alinhado (< 45°), ir para navegação
                    if abs(angle_to_spawn) < 0.78:  # ~45°
                        print("[RETURN] ✓ Alinhado! Fase 2: Navegando para spawn...")
                        self._return_phase = 2
                        self._return_start = current_time
                        continue
                    
                    # Timeout de 10 segundos
                    if phase_elapsed > 10.0:
                        print(f"[RETURN] Timeout giro. Prosseguindo (ang={math.degrees(angle_to_spawn):.0f}°)...")
                        self._return_phase = 2
                        self._return_start = current_time
                        continue
                    
                    # Girar na direção do spawn
                    omega = 0.45 if angle_to_spawn > 0 else -0.45
                    
                    # Se frente bloqueada, dar ré enquanto gira
                    if min_front < 0.35:
                        self._safe_move(-0.06, 0.0, omega)
                    else:
                        self._safe_move(0.0, 0.0, omega)
                    continue
                
                # ===== FASE 2: NAVEGAÇÃO PARA FRENTE =====
                elif self._return_phase == 2:
                    # Log periódico
                    if current_time - self._return_log_time > 2.0:
                        print(f"[RETURN] Pos: ({self.pose[0]:.2f}, {self.pose[1]:.2f}) → SPAWN dist={dist_to_spawn:.2f}m ang={math.degrees(angle_to_spawn):.0f}°")
                        self._return_log_time = current_time
                    
                    # Chegou no spawn?
                    if dist_to_spawn < 0.60:
                        print(f"[RETURN] ✓ Chegou ao spawn! Pose: ({self.pose[0]:.2f}, {self.pose[1]:.2f})")
                        print("[RETURN] ========== INICIANDO NOVA BUSCA ==========")
                        
                        gt = self._get_ground_truth_pose()
                        if gt:
                            self.pose = gt
                        
                        for attr in ['_return_phase', '_return_start', '_return_log_time', 
                                     '_return_waypoints', '_return_waypoint_idx', '_last_route_taken']:
                            if hasattr(self, attr):
                                delattr(self, attr)
                        
                        self.mode = "search"
                        continue
                    
                    # Navegação simples para o spawn
                    cmd_speed = 0.0
                    cmd_omega = 0.0
                    
                    # Se ângulo grande (> 90°), parar e girar
                    if abs(angle_to_spawn) > math.pi / 2:
                        if min_front < 0.35:
                            cmd_speed = -0.06
                        else:
                            cmd_speed = 0.02
                        cmd_omega = 0.4 if angle_to_spawn > 0 else -0.4
                    
                    # Emergência frontal
                    elif min_front < 0.30:
                        if min_rear > 0.30:
                            cmd_speed = -0.08
                            cmd_omega = 0.35 if left_obs > right_obs else -0.35
                        else:
                            cmd_omega = 0.4 if left_obs > right_obs else -0.4
                    
                    # Lateral bloqueado
                    elif left_obs < 0.25 or right_obs < 0.25:
                        cmd_speed = 0.04
                        cmd_omega = -0.35 if left_obs < right_obs else 0.35
                    
                    # Navegação normal
                    else:
                        cmd_omega = -angle_to_spawn * 1.5
                        cmd_omega = max(-0.5, min(0.5, cmd_omega))
                        
                        if min_front < 0.50:
                            cmd_speed = 0.08
                        else:
                            cmd_speed = 0.14
                        
                        cmd_speed *= (1.0 - min(0.3, abs(cmd_omega)))
                    
                    self._safe_move(cmd_speed, 0.0, cmd_omega)
                    continue


if __name__ == "__main__":
    controller = YouBotController()
    controller.run()
