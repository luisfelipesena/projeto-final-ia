# YouBot MCP - Documentação de Testes e Debugging

**Atualizado:** 2025-12-01 (Sessão Intensiva de Debug)
**Status:** Detecção corrigida, grasp em refinamento

---

## 1. Resumo dos Problemas Encontrados e Soluções

### 1.1 PROBLEMA CRÍTICO: Detecção de Obstáculos como Cubos

**Sintoma:** Robot detectava caixas de madeira azuis (WoodenBox) como "blue cubes"

**Evidência via Screenshots:**
- `grasp_before_*.jpg`: Mostra caixa de madeira azul grande no centro-esquerda
- Cubo azul real visível como pequeno quadrado na direita da imagem
- Robot se aproximava da caixa de madeira pensando ser um cubo

**Causa Raiz:** Saturação HSV muito baixa (S=70) permitia cores "lavadas" dos obstáculos

**Solução Aplicada:**
```python
# cube_detector.py - COLOR_RANGES
# ANTES: S=70 (detectava obstáculos)
'blue': {'lower': np.array([100, 70, 40]), ...}

# DEPOIS: S=120 (filtra obstáculos, detecta apenas cubos puros)
'blue': {'lower': np.array([100, 120, 60]), ...}
```

**Resultado:** Após correção, robot passou a detectar cubo real a +27° (lado direito)

---

### 1.2 Convenções de Sinais (CRÍTICO para navegação)

#### Ângulo de Visão (CubeDetector)
```python
angle = (cx - 0.5) * 60.0  # FOV 60°
# cx=0.0 (esquerda) → angle=-30°
# cx=0.5 (centro)   → angle=0°
# cx=1.0 (direita)  → angle=+30°
```

#### Rotação do Robot (MovementService)
```python
def turn(angle_deg):
    # positive → CCW (esquerda)
    # negative → CW (direita)
```

#### Para Alinhar com Cubo
```python
# Cubo à ESQUERDA (angle=-10°) → girar ESQUERDA → turn(+10°)
# Cubo à DIREITA (angle=+10°) → girar DIREITA → turn(-10°)
# CORRETO: turn_angle = -target_angle (NEGATIVO do ângulo)
```

---

### 1.3 Sequência de Grasping Atual

```
GRASPING State:
1. Verificar cubo visível (abort se não)
2. Screenshot "before"
3. Pre-alignment se |angle| > 5°
4. Screenshot "after_align"
5. Forward approach (25cm)
6. Screenshot "after_forward"
7. Prepare grasp (arm to FRONT_PLATE, gripper open)
8. Execute grasp (FRONT_FLOOR preset)
9. Screenshot "after_grasp"
10. Verificar has_object()
```

---

## 2. Parâmetros Calibrados

### 2.1 Detecção de Cubos (cube_detector.py)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| HSV Saturation min | 120 | Filtra caixas de madeira (S~80-100) |
| HSV Value min | 60 | Tolera sombras leves |
| MAX_BBOX_FRACTION | 0.30 | Rejeita objetos muito grandes |
| MAX_PROJECTED_WIDTH | 0.30 | Limite de largura no frame |
| MIN_BBOX_ASPECT | 0.70 | Cubos são ~quadrados |
| MIN_SOLIDITY | 0.80 | Rejeita formas côncavas |

### 2.2 Tracking (vision_service.py)

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| LOST_THRESHOLD | 60 frames | ~2s para realinhar |
| ANGLE_TOLERANCE | 30° | Permite rotação durante approach |
| POSITION_TOLERANCE | 0.40m | Permite movimento |
| MIN_CONFIDENCE | 0.55 | Aceita detecções menos certas |

### 2.3 Grasping

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| forward_move | 0.25m | Aproximar bem do cubo |
| CAMERA_ARM_OFFSET | 0.15m | Câmera à frente do braço |
| grasp_threshold_dist | 0.32m | Trigger GRASPING |
| grasp_threshold_angle | 12° | Alinhamento mínimo |

---

## 3. Arquitetura LIDAR + Fuzzy para Navegação

### 3.1 Setores LIDAR

```
      front_left(2)   front(3)   front_right(4)
              \         |         /
               \        |        /
    left(1) ----[     ROBOT     ]---- right(5)
               /        |        \
              /         |         \
      far_left(0)              far_right(6)
            \                      /
             back_left(8)  back_right(7)
```

### 3.2 Uso Proposto do LIDAR para Validação de Cubos

```python
def validate_cube_detection(detection, lidar_data):
    """
    Usar LIDAR para confirmar se detecção é cubo real ou obstáculo.

    Cubos: 5cm - LIDAR não detecta (muito pequeno)
    Obstáculos: 30-50cm - LIDAR detecta como parede

    Se LIDAR mostra obstáculo na direção do "cubo" detectado,
    provavelmente é um falso positivo (caixa de madeira).
    """
    cube_angle = detection.angle  # graus
    cube_distance = detection.distance  # metros

    # Mapear ângulo para setor LIDAR
    if cube_angle < -30:
        sector = 'far_left'
    elif cube_angle < -15:
        sector = 'left'
    elif cube_angle < -5:
        sector = 'front_left'
    elif cube_angle < 5:
        sector = 'front'
    elif cube_angle < 15:
        sector = 'front_right'
    elif cube_angle < 30:
        sector = 'right'
    else:
        sector = 'far_right'

    lidar_dist = lidar_data['obstacle_sectors'][sector]['min']

    # Se LIDAR mostra obstáculo PRÓXIMO da distância do cubo detectado
    # E obstáculo está PERTO → provavelmente é falso positivo
    if lidar_dist < cube_distance + 0.10:
        return False  # Rejeitar - provavelmente obstáculo

    return True  # Aceitar - parece ser cubo real
```

### 3.3 Integração Fuzzy Controller

```python
# Inputs do Fuzzy (já implementado em fuzzy_controller.py)
class FuzzyInputs:
    obstacle_front: float    # 0-1, maior = mais longe
    obstacle_left: float
    obstacle_right: float
    cube_distance: float     # 0-1, maior = mais perto
    cube_angle: float        # -1 a +1, 0 = centrado
    lateral_blocked: float   # 0-1

# Outputs do Fuzzy
class FuzzyOutputs:
    forward_speed: float     # 0-1
    rotation_speed: float    # -1 a +1
    action: str              # 'search', 'approach', 'avoid', 'grasp'
```

### 3.4 Lógica de Navegação Proposta

```python
def navigate_to_cube(self):
    # 1. Obter dados de sensores
    lidar = self._get_lidar_data()
    cube = self.vision.get_target()

    # 2. Validar cubo com LIDAR
    if cube and not validate_cube_detection(cube, lidar):
        print("[NAV] Detecção rejeitada - parece obstáculo")
        return 'search'

    # 3. Computar inputs fuzzy
    inputs = self._compute_fuzzy_inputs()

    # 4. Obter decisão fuzzy
    outputs = self.fuzzy.compute(inputs)

    # 5. Executar ação
    if outputs.action == 'avoid':
        # Obstáculo detectado - desviar
        self.base.move(0, 0, outputs.rotation_speed)
    elif outputs.action == 'approach':
        # Caminho livre - aproximar do cubo
        self.base.move(outputs.forward_speed * 0.1, 0,
                       outputs.rotation_speed * 0.3)
    elif outputs.action == 'grasp':
        # Perto o suficiente - iniciar grasp
        return 'grasp'
    else:
        # Procurar
        return 'search'
```

---

## 4. Screenshots de Debug (Funcionalidade Adicionada)

### 4.1 Localização
```
youbot_mcp/data/youbot/
├── grasp_before_*.jpg       # Visão antes de iniciar grasp
├── grasp_after_align_*.jpg  # Após pre-alinhamento
├── grasp_after_forward_*.jpg # Após aproximação
├── grasp_after_grasp_*.jpg  # Após tentativa de grasp
└── grasp_abort_no_cube_*.jpg # Se cubo não visível
```

### 4.2 Como Interpretar

| Screenshot | O que verificar |
|------------|-----------------|
| before | Cubo está visível? Na posição correta? |
| after_align | Cubo está mais centralizado? |
| after_forward | Cubo está próximo? Gripper pode alcançar? |
| after_grasp | Cubo foi capturado? Está no gripper? |

---

## 5. Comandos Úteis para Debug

### 5.1 Monitoramento em Tempo Real

```bash
# Monitor simples (recomendado)
python3 /tmp/monitor.py

# Conteúdo do monitor.py:
import json, re, time
for i in range(20):
    try:
        with open('youbot_mcp/data/youbot/status.json') as f:
            txt = f.read()
        txt = re.sub(r': Infinity', ': 999', txt)
        d = json.loads(txt)
        t = d.get('current_target') or {}
        state = d['current_state']
        color = t.get('color', '-')
        dist = t.get('distance', 0)
        angle = t.get('angle', 0)
        cubes = d['cubes_collected']
        print(f"[{i:2d}] {state:12} {color:5} d={dist:.2f} a={angle:+6.1f} c={cubes}")
    except Exception as e:
        print(f"[{i:2d}] ERR")
    time.sleep(1.5)
```

### 5.2 Reinício Limpo

```bash
pkill -9 -f webots
sleep 2
rm -f youbot_mcp/data/youbot/grasp_*.jpg
echo "" > youbot_mcp/data/youbot/grasp_log.txt
open -a "Webots" "IA_20252/worlds/IA_20252.wbt"
```

### 5.3 Ver Screenshots

```bash
# Listar screenshots recentes
ls -la youbot_mcp/data/youbot/grasp_*.jpg

# Abrir no preview (macOS)
open youbot_mcp/data/youbot/grasp_before_*.jpg
```

---

## 6. Checklist de Validação

### 6.1 Detecção
- [x] HSV saturation aumentada para 120 (filtra obstáculos)
- [x] Tamanho máximo de bbox reduzido para 0.30
- [ ] Validação com LIDAR (a implementar)

### 6.2 Tracking
- [x] LOST_THRESHOLD aumentado para 60 frames
- [x] ANGLE_TOLERANCE aumentada para 30°
- [ ] Estabilidade durante rotação

### 6.3 Aproximação
- [x] Pre-alinhamento adicionado (rotate to center)
- [x] Sign convention documentada
- [ ] Validar rotação funciona corretamente

### 6.4 Grasping
- [x] Forward approach calibrado (25cm)
- [x] Screenshots em cada etapa
- [x] Verificação de cubo visível antes de grasp
- [ ] Captura física de cubo (em teste)
- [ ] has_object() retorna True

### 6.5 Depósito
- [ ] Navegação até caixa de cor
- [ ] Release do cubo
- [ ] Retorno a SEARCHING

---

## 7. Problemas Conhecidos e Próximos Passos

### 7.1 Problema: Grasp Mecânico Falha

**Observação:** finger_pos_after_close=0.00000 (gripper fecha no vazio)

**Hipóteses:**
1. Forward approach não está movendo robot suficiente
2. FRONT_FLOOR não posiciona gripper na altura correta
3. Cubo é empurrado durante aproximação

**Próximos testes:**
1. Verificar se movimento forward realmente ocorre
2. Testar IK vs FRONT_FLOOR preset
3. Reduzir velocidade de aproximação

### 7.2 Problema: Target Perdido Frequentemente

**Observação:** Oscila SEARCHING ↔ APPROACHING

**Hipóteses:**
1. Detecção instável (filtros muito agressivos?)
2. Rotação move cubo para fora do FOV
3. Frames insuficientes para tracking estável

**Próximos testes:**
1. Relaxar filtros ligeiramente se necessário
2. Rotacionar mais devagar durante approach
3. Aumentar MIN_FRAMES_RELIABLE

### 7.3 Integração LIDAR-Visão (A Fazer)

```python
# Adicionar em _run_autonomous_step():
if self.state == MCPState.APPROACHING:
    cube = self.vision.get_target()
    if cube:
        # Validar com LIDAR
        if not self._validate_with_lidar(cube):
            print("[NAV] False positive - parece obstáculo")
            self.vision.unlock()
            self.state = MCPState.SEARCHING
            return
```

---

## 8. Histórico de Versões

### V4 - Sessão de Debug Intensivo (2025-12-01)
- HSV saturation: 70 → 120 (filtra caixas de madeira)
- Screenshots automáticos em GRASPING
- Pre-alinhamento antes de grasp
- Documentação de sign conventions
- Análise via imagens identificou problema de detecção

### V3 - Correções Anteriores (2025-12-01)
- SEARCHING: só considera setores frontais do LIDAR
- VisionService: LOST_THRESHOLD=60, tolerâncias flexíveis
- APPROACHING: controle proporcional de omega
- Logs aprimorados em nav_debug.log

### V2 - Integração RNA/Fuzzy
- MLP LIDAR treinado (97.8% accuracy)
- FuzzyController integrado
- Debug logging

---

## 9. Arquivos Principais

| Arquivo | Responsabilidade |
|---------|------------------|
| `youbot_mcp/youbot_mcp_controller.py` | State machine, lógica autônoma |
| `src/perception/cube_detector.py` | HSV segmentation, detecção |
| `src/services/vision_service.py` | Tracking de cubo |
| `src/services/arm_service.py` | Controle do braço/gripper |
| `src/services/movement_service.py` | Movimentação da base |
| `src/control/fuzzy_controller.py` | Lógica fuzzy |
| `IA_20252/controllers/youbot/base.py` | Cinemática Mecanum |
| `IA_20252/controllers/youbot/arm.py` | IK do braço |
| `IA_20252/controllers/youbot/gripper.py` | Controle gripper |

---

*Última atualização: 2025-12-01*
*Próximo objetivo: Capturar 1 cubo com sucesso*
