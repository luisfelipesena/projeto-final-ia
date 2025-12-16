# YouBot Cube Collection — Resumo Técnico (focado em implementação)

## Visão geral
- Objetivo: coletar 15 cubos (red/green/blue) e depositar nas caixas da cor correta no mundo `IA_20252.wbt`.
- Plataforma: Webots (controlador Python, nó Supervisor), KUKA YouBot com base mecanum, braço 5-DOF e garra paralela.
- Sensores principais: câmera RGB com Recognition ativo, 4 LIDARs 360° (front/rear/left/right), 6 sensores infravermelhos de distância (front/rear com diagonais).
- Estados principais (FSM): `search` → `approach` → `grasp` → `to_box` → `drop` → `return_to_spawn` → volta a `search`.

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

## Fluxo Completo (Loop)

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
│  Waypoints simples → Baliza se preso → Volta ao spawn      │
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

## Referências Técnicas
- Webots docs: https://cyberbotics.com/doc/guide/index
- Webots Recognition API: https://cyberbotics.com/doc/reference/recognition
- Three-Point Turn: manobra padrão de veículos para inversão em espaços confinados
- Car-like kinematics: Ackermann steering geometry principles
- MobileNetV3 (Howard et al., 2019) para classificação de cores
