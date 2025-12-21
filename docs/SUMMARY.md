# YouBot Cube Collection — Resumo Técnico (focado em implementação)

## Visão geral
- **Objetivo**: coletar 15 cubos (red/green/blue) e depositar nas caixas da cor correta no mundo `IA_20252.wbt`.
- **Plataforma**: Webots (controlador Python, nó Supervisor), KUKA YouBot com base mecanum, braço 5-DOF e garra paralela.
- **Sensores principais**: câmera RGB com Recognition ativo, LIDAR 180° frontal, 8 sensores infravermelhos de distância (6 front/rear + 2 laterais novos).
- **Estados principais (FSM)**: `search` → `approach` → `grasp` → `to_box` → `drop` → `return_to_spawn` → volta a `search`.

## Técnicas de IA Utilizadas

| Técnica | Aplicação | Algoritmo/Arquitetura |
|---------|-----------|----------------------|
| **Redes Neurais** | Classificação de cor dos cubos | MobileNetV3-Small (transfer learning) |
| **Lógica Fuzzy** | Navegação reativa e desvio de obstáculos | Regras SE-ENTÃO com funções trapezoidais |
| **Busca A*** | Planejamento de trajetória global | A* com grade de ocupação 12×12cm |

## Arquitetura Modular do Código

O controlador foi dividido em módulos para melhor manutenibilidade:

```
controllers/youbot/
├── constants.py          # Constantes e configuração da arena (~60 linhas)
├── routes.py             # Planejamento de rotas estratégicas (~240 linhas)
├── occupancy_grid.py     # Grade de ocupação e A* (~280 linhas)
├── fuzzy_navigator.py    # Navegação fuzzy (~230 linhas)
├── youbot_controller.py  # Controlador principal FSM (~1600 linhas)
├── youbot.py             # Entry point (~12 linhas)
├── base.py               # Controle da base mecanum
├── arm.py                # Controle do braço 5-DOF
├── gripper.py            # Controle da garra
└── color_classifier.py   # CNN para classificação de cores
```

### Módulos Principais

| Módulo | Responsabilidade |
|--------|------------------|
| `constants.py` | `CUBE_SIZE`, `ARENA_*`, `BOX_POSITIONS`, `SPAWN_POSITION`, `KNOWN_OBSTACLES` |
| `routes.py` | `get_route_to_box()`, `get_return_route()`, `wrap_angle()`, `color_from_rgb()` |
| `occupancy_grid.py` | Classe `OccupancyGrid` com A*, raycast, inflação de obstáculos |
| `fuzzy_navigator.py` | Classe `FuzzyNavigator` com funções de pertinência e regras |
| `youbot_controller.py` | Classe `YouBotController` com FSM e handlers de cada estado |

## Configuração do mundo e hardware
- Arena: `RectangleArena` 7.0 m × 4.0 m, centro em (-0.79, 0.0), parede 0.30 m. Robô spawna em x≈-4.0 m.
- Caixas (PlasticFruitBox): 
  - **GREEN**: (0.48, 1.58) - norte
  - **BLUE**: (0.48, -1.62) - sul  
  - **RED**: (2.31, 0.01) - leste, rotacionada 90°
- Obstáculos fixos (WoodenBox, raio seguro ~0.25 m): A-G nas posições do mundo e em `KNOWN_OBSTACLES` do controlador.
- Sensores no slot do robô:
  - **4 LIDARs 360°**: front (0°), rear (180°), left (90°), right (-90°), cada um com FOV 90°, 64 raios, range 0.1–3.5 m
  - **Câmera**: 128×128, offset (0.27, 0, -0.06), Recognition ativado com `maxRange=3 m`
  - **IR distance sensors**: traseiros `ds_rear`, `ds_rear_left` (135°), `ds_rear_right` (-135°); frontais `ds_front` (0°), `ds_front_left` (~23°), `ds_front_right` (~-23°)
- Base mecanum: parâmetros `WHEEL_RADIUS=0.05`, `LX=0.228`, `LY=0.158`.

## Arquitetura de Navegação Car-Like

### Conceito
O YouBot utiliza navegação **car-like** onde:
- `Vy = 0` (sem strafe lateral por padrão)
- Apenas `Vx` (frente/ré) e `ω` (rotação) são usados
- Simula um veículo com "rodas frontais que esterçam" e "rodas traseiras de tração"

### Rotas Pré-definidas (Base Routes)
O robô segue **corredores fixos** para cada caixa, garantindo:
1. Passar reto pelos primeiros obstáculos (E, F)
2. Inclinação gradual para desviar do obstáculo central (A)
3. Garantir que as **rodas traseiras** passem completamente o obstáculo antes de virar
4. Alinhamento perpendicular final com a caixa

```
SPAWN (-3.91, 0)
    │
    │  Reto até X=-0.45 (passa E e F)
    ▼
    ├───────────► GREEN (0.48, 1.58) - inclina para +Y
    │
    ├───────────► BLUE (0.48, -1.62) - inclina para -Y
    │
    └───────────► RED (2.31, 0.01) - desvia A, volta para Y=0
```

### Waypoints por Cor

**BLUE** (8 waypoints):
```
(-0.45, 0.00) → (-0.15, 0.00) → (0.00, -0.35) → (0.15, -0.70)
→ (0.25, -1.00) → (0.35, -1.25) → (0.48, -1.35) → (0.48, -1.62)
```

**GREEN** (8 waypoints):
```
(-0.45, 0.00) → (-0.15, 0.00) → (0.00, 0.35) → (0.15, 0.70)
→ (0.25, 1.00) → (0.35, 1.25) → (0.48, 1.35) → (0.48, 1.58)
```

**RED** (11 waypoints):
```
(-0.45, 0.00) → (0.00, 0.00) → (0.25, -0.45) → (0.50, -0.55)
→ (0.75, -0.55) → (1.00, -0.50) → (1.25, -0.35) → (1.55, -0.15)
→ (1.85, 0.01) → (2.05, 0.01) → (2.31, 0.01)
```

## Sistema de Navegação TO_BOX

### Estados da Máquina de Estados (tobox_state)
- **State 0**: Navegação por waypoints com obstacle avoidance
- **State 1**: Alinhamento perpendicular com a caixa
- **State 2**: Aproximação final e trigger de DROP

### Hierarquia de Prioridades (State 0)

```
PRIORIDADE 1: EMERGÊNCIA (min_front < 0.22m)
├── Parar imediatamente
├── Ré se traseira livre
└── Rotação para lado mais livre

PRIORIDADE 2: PERIGO FRONTAL (min_front < 0.35m)
├── Ré lenta + rotação
└── Se traseira bloqueada: rotação pura

PRIORIDADE 3: ALERTA FRONTAL (min_front < 0.50m)
├── Velocidade reduzida (0.04 m/s)
└── Rotação para waypoint

PRIORIDADE 4: BLOQUEIO LATERAL (left/right < 0.20m)
├── Forçar rotação OPOSTA ao lado bloqueado
└── Log: "⛔ BLOQUEIO ESQ/DIR"

PRIORIDADE 5: NAVEGAÇÃO NORMAL
├── Seguir waypoint com omega proporcional ao ângulo
└── Velocidade 0.10–0.12 m/s
```

### Override Absoluto de Colisão
```python
# NUNCA virar para lado bloqueado (verificação final)
if left_blocked and cmd_omega > 0:
    cmd_omega = -0.4  # Forçar direita
if right_blocked and cmd_omega < 0:
    cmd_omega = 0.4   # Forçar esquerda
```

### Box Approach Mode
Quando `waypoints_remaining <= 3` ou `dist_to_box < 1.0m`:
- **Ignora** detecção de obstáculo frontal (o box é o destino!)
- Reage apenas a obstáculos **laterais**
- Transição para alinhamento quando `dist_to_box < 0.55m` ou `min_front < 0.18m`

### Alinhamento Perpendicular (State 1)
```python
target_heading = {
    "green": +90°,   # Apontar para norte
    "blue":  -90°,   # Apontar para sul
    "red":   0°      # Apontar para leste
}

ALIGN_TOLERANCE = 25°  # Tolerância permissiva
ALIGN_TIMEOUT = 2.0s   # Não gastar muito tempo
omega = heading_error * 0.5  # Rotação suave
```

## Sistema de Retorno (RETURN_TO_SPAWN)

### Estratégia: Waypoints Adaptativos (3 fases)

O sistema de retorno foi redesenhado para evitar travamentos em giros de 180°:

```
┌─────────────────────────────────────────────────────────────┐
│  FASE 0: RETREAT (recuo da caixa)                           │
│  - Recua 60cm ou até obstáculo traseiro                     │
│  - Sincroniza ground truth após recuo                       │
│  - Calcula rota ADAPTATIVA baseada na posição REAL          │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 1: TURN-IN-PLACE (se necessário)                      │
│  - Se ângulo ao waypoint > 100°: rotação pura               │
│  - Detecção de travamento: se < 30° progresso em 5s, skip   │
│  - Velocidade aumentada (0.5) para ângulos > 150°           │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  FASE 2: WAYPOINT NAVIGATION                                │
│  - Navegação car-like idêntica ao TO_BOX                    │
│  - Skip automático de waypoints passados                    │
│  - Detecção de stuck: se < 15cm movimento em 8s, skip       │
│  - Retorna à Fase 1 se ângulo > 100°                        │
└─────────────────────────────────────────────────────────────┘
```

### Waypoints Adaptativos por Cor

A função `get_return_route(current_pos, from_color)` gera waypoints **baseados na posição atual**, não assumindo posição fixa após depósito:

**RED** (posição típica após recuo ~0.9, 0.5):
```python
# Só adiciona waypoints OESTE da posição atual
if x > 1.0: waypoints.append((0.90, 0.45))
if x > 0.6: waypoints.append((0.50, 0.45))
if x > 0.2: waypoints.append((0.15, 0.30))
waypoints.append((-0.20, 0.15))  # Entrada corredor central
```

**GREEN** (posição típica após recuo ~0.5, 1.0):
```python
if y > 0.80: waypoints.append((0.50, 0.60))
if y > 0.40: waypoints.append((0.35, 0.30))
waypoints.append((0.10, 0.10))
waypoints.append((-0.25, 0.0))  # Corredor central
```

**BLUE** (posição típica após recuo ~0.5, -1.0):
```python
if y < -0.80: waypoints.append((0.50, -0.60))
if y < -0.40: waypoints.append((0.35, -0.30))
waypoints.append((0.10, -0.10))
waypoints.append((-0.25, 0.0))  # Corredor central
```

**Corredor comum** (evita obstáculos E e F):
```python
waypoints.extend([
    (-0.60, 0.0),   # Antes de E/F
    (-1.30, 0.0),   # Entre E e F
    (-1.80, 0.0),   # Após E/F
    (-2.50, 0.0),   # Área livre
    (-3.20, 0.0),   # Aproximando spawn
    SPAWN_POSITION  # (-3.91, 0.0)
])
```

### Mecanismos de Recuperação

```python
# Skip de waypoints já passados
def _skip_passed_waypoints(self):
    while waypoint_idx < len(waypoints):
        dist, angle = distance_to_point(waypoint)
        if dist < 0.45:  # Muito perto
            skip()
        elif abs(angle) > 120° and dist < 1.0:  # Atrás e perto
            skip()
        else:
            break

# Detecção de travamento em rotação (Fase 1)
if turn_elapsed > 5.0 and angle_progress < 30°:
    skip_waypoint()

# Detecção de travamento em navegação (Fase 2)
if moved < 0.15m in 8.0s:
    skip_waypoint()
```

### Controle
- Segue waypoint-a-waypoint com `cmd_omega = -angle_to_wp * k` (car-like) e velocidade reduzida em curvas.
- Evitação de colisão **apenas quando necessário** (emergência/perigo frontal + repulsão lateral suave).
- **Transição automática** entre fases conforme necessário (Fase 2 pode voltar à Fase 1 se waypoint ficar atrás).

### Saída do Spawn (reorientação)
Ao chegar no spawn (`dist_to_spawn < 0.60m`):
- Sincroniza ground truth
- Limpa todos os estados de retorno
- Transiciona para modo `search`

## Pilha de Sensores 360°

### LIDARs (4 unidades)
```
        FRONT (0°)
           │
    ┌──────┴──────┐
    │             │
LEFT│   YOUBOT    │RIGHT
(90°)│             │(-90°)
    └──────┬──────┘
           │
        REAR (180°)
```

Cada LIDAR:
- FOV: 90° (π/2)
- Resolução: 64 raios
- Range: 0.1 – 3.5 m
- Atualização: 32 ms

### Processamento LIDAR
```python
def _process_lidar_360(self, lidar, name):
    ranges = lidar.getRangeImage()
    valid = [r for r in ranges if 0.1 < r < 3.5]
    return {
        "min": min(valid),
        "avg": mean(valid),
        "count": len([r for r in valid if r < 0.5])
    }
```

### Fusão de Sensores
```python
min_front = min(
    lidar_info["front"]["min"],  # LIDAR frontal
    front_info["front"],         # IR ds_front
    front_info["front_left"],    # IR ds_front_left
    front_info["front_right"]    # IR ds_front_right
)
```

## Fluxo Completo (Loop) - v2 com Early Search

```
┌─────────────────────────────────────────────────────────────┐
│                        SEARCH                               │
│  Lawnmower + Recognition → detecta cubo                     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                       APPROACH                              │
│  Alinha → Avança → cubo a 0.32m                             │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        GRASP                                │
│  Braço baixo → Avança → Fecha garra → Verifica             │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                       TO_BOX                                │
│  Waypoints → Obstacle Avoidance → Alinhamento → Aproximação │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        DROP                                 │
│  Braço baixo → Avança sobre box → Solta → Recua            │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   RETURN_TO_SPAWN                           │
│  Waypoints estratégicos → Escapa área do box                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ EARLY EXIT: x<-1.15 OU timeout 30s → SEARCH        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
                   (LOOP)
```

## Constantes Críticas

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `EMERGENCY_STOP` | 0.22 m | Parada imediata |
| `FRONT_DANGER` | 0.35 m | Ré obrigatória |
| `FRONT_WARN` | 0.50 m | Velocidade reduzida |
| `LATERAL_DANGER` | 0.20 m | Bloqueio lateral |
| `WP_ARRIVAL` | 0.35 m | Chegada ao waypoint |
| `ALIGN_TOLERANCE` | 25° | Alinhamento permissivo |
| `BALIZA_DURATION` | 1.5 s | Duração de cada fase |

## Depósito (DROP)

### Thresholds de Segurança
```python
# Trigger do DROP (no TO_BOX state 2)
drop_ready = 0.45 < ds_front < 0.65  # Distância segura (não bater no box)
emergency = ds_front < 0.45          # Muito perto, DROP de segurança
timeout = elapsed > 4.0s AND dist_box < 0.65m
```

### Sequência
1. Braço FRONT_FLOOR (2.0 s)
2. Avança 0.04 m/s até `ds_front <= 0.60m` (ou timeout 2.5 s)
3. Abre garra (1.0 s)
4. Recuo curto (1.0 s)
5. Braço RESET (1.5 s)
6. Recuo longo (2.0 s)

## Localização e Odometria

### Cinemática Mecanum
```python
vx = R/4 * (ω1 + ω2 + ω3 + ω4)
vy = R/4 * (-ω1 + ω2 + ω3 - ω4)
omega = R / (4*(LX+LY)) * (-ω1 + ω2 - ω3 + ω4)
```

### Sincronização Ground Truth
- `search/approach`: a cada 2.0 s
- `to_box`: a cada 0.5 s (alta precisão)
- `return_to_spawn`: a cada 0.5 s (alta precisão)

## Rede Neural Convolucional (CNN)

### Arquitetura: MobileNetV3-Small
- **Backbone**: MobileNetV3-Small pré-treinado no ImageNet
- **Classificador**: Dense(256) → Dense(3, softmax)
- **Input**: 64×64×3 RGB normalizado
- **Output**: probabilidades para {red, green, blue}

### Convolução Depthwise Separável
```
Convolução Tradicional: O(k² · Ci · Co · H · W)
Depthwise Separável:    O(k² · Ci · H · W + Ci · Co · H · W)
                        ≈ 8-9x menos operações
```

### Transfer Learning (2 fases)
1. **Backbone congelado**: treina apenas classificador (10 epochs)
2. **Fine-tuning**: descongelado, learning rate 0.0001 (5 epochs)

### Métricas
- **Acurácia**: 99.4%
- **Fallback HSV**: usado quando confiança CNN < 0.5

```python
# Fallback HSV para iluminação adversa
if confidence < 0.5:
    h, s, v = rgb_to_hsv(r, g, b)
    if 0 <= h < 10 or h > 340: return "red"
    if 80 <= h < 160: return "green"
    if 200 <= h < 260: return "blue"
```

## Sistema Fuzzy de Navegação

### Variáveis de Entrada
| Variável | Universo | Termos Linguísticos |
|----------|----------|---------------------|
| `distância` | [0, 2] m | muito_perto, perto, longe |
| `ângulo` | [-π, π] rad | grande_esq, pequeno_esq, alinhado, pequeno_dir, grande_dir |

### Funções de Pertinência (Trapezoidais)
```
μ_muito_perto(d): 1.0 se d < 0.25m, 0 se d > 0.45m
μ_perto(d):       triângulo centrado em 0.45m
μ_longe(d):       1.0 se d > 0.65m
```

### Regras Fuzzy Principais
```
SE distância=muito_perto ENTÃO velocidade=reverso, strafe=lateral
SE distância=perto ENTÃO velocidade=lento
SE distância=longe E ângulo=alinhado ENTÃO velocidade=rápido
SE ângulo=grande ENTÃO velocidade=parar, omega=máximo
SE lateral_esq=bloqueado ENTÃO strafe=direita
SE lateral_dir=bloqueado ENTÃO strafe=esquerda
```

### Defuzzificação
- **Método**: Centróide ponderado
- **Saídas**: `Vx` (linear), `Vy` (strafe), `ω` (angular)

## Algoritmo A* com Grade de Ocupação

### Grade de Ocupação
- **Resolução**: 12×12 cm por célula
- **Dimensões**: 58×33 células (arena 7m×4m)
- **Estados**: LIVRE (0), OCUPADO (1), INFLADO (2), DESCONHECIDO (3)

### Atualização por Raycasting
```python
def update_from_lidar(self, robot_pos, robot_heading, ranges):
    for i, r in enumerate(ranges):
        angle = robot_heading + ray_angles[i]
        # Bresenham: células no caminho = LIVRE
        for cell in bresenham(robot_pos, endpoint):
            grid[cell] = LIVRE
        # Célula final (obstáculo) = OCUPADO
        if r < max_range:
            grid[endpoint] = OCUPADO
```

### Inflação de Obstáculos
- **Raio de inflação**: 30 cm (meia-diagonal do robô + margem)
- **Propósito**: garantir que caminhos gerados tenham espaço suficiente

### Heurística A*
```python
def heuristic(a, b):
    # Distância Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def f(n):
    return g[n] + heuristic(n, goal)
```

### Suavização de Caminho
```python
def smooth_path(path):
    # Remove waypoints colineares
    smoothed = [path[0]]
    for i in range(1, len(path)-1):
        if not collinear(smoothed[-1], path[i], path[i+1]):
            smoothed.append(path[i])
    smoothed.append(path[-1])
    return smoothed
```

## Sistema de Escape (TO_BOX)

### Detecção de Travamento
```python
if tobox_time > 3.0 and distance_moved < 0.15:
    enter_escape_mode()
```

### Fases do Escape
1. **Fase 1 (1.5s)**: Reverso + strafe na direção menos obstruída
2. **Fase 2 (1.5s)**: Rotação para waypoint
3. **Após 3 escapes**: Re-roteia com A*

### Lock de Direção
```python
# Evita oscilação (bug corrigido)
if entering_escape:
    self._escape_go_left = (left_dist > right_dist + 0.03)
# Direção permanece fixa durante todo o escape
```

## Pós-Depósito: Escape Estratégico (Fix v2)

### Problema Anterior
Após depositar o cubo, o robô transitava diretamente para `search` mode enquanto ainda estava em "zona morta" perto das caixas, cercado por obstáculos. O lawnmower search sem awareness de obstáculos causava colisões imediatas.

### Solução: Usar return_to_spawn com Early Exit
Em vez de ir direto para `search`, o robô agora:
1. Transita para `return_to_spawn` mode
2. Segue waypoints estratégicos para escapar da área do box
3. Quando atinge "zona segura" (corredor central) OU timeout 30s → `search` mode

### Sequência de Drop (8 estágios)
| Stage | Duração | Ação |
|-------|---------|------|
| 0-5 | ~7s | Aproximação, drop, recuo inicial |
| 6 | 2.5s | Recuo longo (0.12 m/s) |
| 7 | ~2.0-2.5s | Rotação para evitar box |
| 8 | - | **Transição para return_to_spawn** (não search!) |

### Early Search Transition (Zona Segura)
```python
# Em _handle_return_to_spawn phase 2:
in_safe_zone = self.pose[0] < -1.15 and abs(self.pose[1]) < 0.8
timeout_exceeded = return_elapsed > 30.0

if in_safe_zone or timeout_exceeded:
    self.mode = "search"  # Agora seguro para lawnmower
```

**Zona segura**: x < -1.15 (13cm oeste de E/F) e |y| < 0.8 (corredor entre E e F)

### Direção de Rotação por Cor
```python
if color == "red":
    turn_direction = LEFT   # 140° (virar para norte)
elif color == "green":
    turn_direction = RIGHT  # 115° (virar para centro)
elif color == "blue":
    turn_direction = LEFT   # 115° (virar para centro)
```

## Materiais de Apresentação

### Estrutura dos Slides (20 slides, apenas figuras TikZ)
1. Título
2. Agenda (timeline visual)
3. Problema (arena diagram)
4. Restrições (obrigatório/proibido)
5. Sensores do YouBot (top view detalhado)
6. Recognition API (flow diagram)
7. CNN MobileNetV3 (arquitetura)
8. Fuzzy - Funções de Pertinência
9. Fuzzy - Regras (cenários visuais)
10. A* - Fórmula e Grid
11. Inflação de Obstáculos
12. Arquitetura Modular
13. Máquina de Estados (FSM)
14. Pipeline Completo
15. Sequência de Coleta (6 passos)
16. Navegação A* em Ação
17. Demonstração
18. Limitações dos Algoritmos
19. Conclusão
20. Referências e Perguntas

### Falas (15 minutos)
- Arquivo: `slides-template/falas.txt`
- Tempo médio por slide: 45 segundos
- Total: ~320 linhas de script

## Referências Técnicas

### Algoritmos
- Hart, Nilsson, Raphael (1968) - A* Search Algorithm
- Zadeh (1965) - Fuzzy Sets and Systems
- Howard et al. (2019) - MobileNetV3: Searching for Efficient CNNs

### Documentação
- Webots Documentation: https://cyberbotics.com/doc/guide/index
- Webots Recognition API: https://cyberbotics.com/doc/reference/recognition
- LIDAR Sensor: https://cyberbotics.com/doc/reference/lidar
- Camera Recognition: https://cyberbotics.com/doc/reference/camera#wb_camera_recognition_get_objects

### Conceitos
- Car-like kinematics: Ackermann steering geometry principles
- Three-Point Turn: manobra para inversão em espaços confinados
- Mecanum wheel kinematics: omnidirectional movement
- UMBMark: Universal Mobile Robot Benchmark (calibração odometria)
