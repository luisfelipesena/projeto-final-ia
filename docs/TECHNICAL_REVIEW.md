# Revisão Técnica - YouBot Autônomo para Coleta de Cubos

**Aluno:** Luis Felipe Cordeiro Sena
**Disciplina:** MATA64 - Inteligência Artificial - UFBA
**Professor:** Luciano Oliveira
**Data:** 2025-11-30
**Status:** Em desenvolvimento (sistema ainda não funcional end-to-end)

---

## 1. Definição do Problema

### 1.1 Objetivo Principal

Desenvolver um sistema autônomo para o robô YouBot no simulador Webots que execute coleta e organização de 15 cubos coloridos (verde, azul, vermelho) em caixas correspondentes, navegando em arena com obstáculos fixos (caixotes de madeira).

### 1.2 Restrições Obrigatórias

| Restrição | Especificação |
|-----------|---------------|
| **Sensores** | LIDAR (detecção de obstáculos) + Câmera RGB (identificação de cores) |
| **GPS** | PROIBIDO na demonstração final |
| **RNA** | Obrigatório: MLP ou CNN para detecção de obstáculos/mapeamento |
| **Lógica Fuzzy** | Obrigatório: controle das ações do robô |
| **supervisor.py** | Inalterável (spawna 15 cubos aleatoriamente) |

### 1.3 Ciclo de Tarefa

```
Para cada cubo (15 total):
    1. BUSCAR cubo na arena
    2. DETECTAR cor via câmera RGB
    3. APROXIMAR-SE do cubo
    4. AGARRAR com gripper
    5. NAVEGAR até caixa correspondente
    6. DEPOSITAR na caixa
    7. Repetir
```

---

## 2. Fundamentação Teórica

### 2.1 Arquitetura de Controle

**Abordagem:** Subsumption Architecture (Brooks, 1986)

A arquitetura escolhida organiza comportamentos em camadas de prioridade, onde camadas inferiores (reativas) podem suprimir camadas superiores (deliberativas).

```
Camada 3: TASK (buscar cubo, depositar)
    ↓ suprimida por
Camada 2: NAVIGATION (aproximar, evitar obstáculo próximo)
    ↓ suprimida por
Camada 1: SAFETY (parar se obstáculo muito próximo)
```

**Justificativa teórica:**
- Brooks (1986): Comportamento inteligente emerge de camadas simples interagindo
- Arkin (1998): Behavior-based robotics valida abordagem para navegação reativa

### 2.2 Redes Neurais Artificiais (RNA)

#### 2.2.1 SimpleLIDARMLP - Detecção de Obstáculos

**Arquitetura implementada:**

```
Input: 512 pontos LIDAR normalizados [0,1]
    ↓
Hidden1: Linear(512→128) + ReLU + Dropout(0.2)
    ↓
Hidden2: Linear(128→64) + ReLU + Dropout(0.2)
    ↓
Output: Linear(64→9) + Sigmoid
    ↓
Saída: 9 setores de ocupação [0,1]
```

| Característica | Valor |
|----------------|-------|
| Parâmetros treináveis | 74,505 |
| Entrada | 512 pontos (resampled de 667 original) |
| Saída | 9 setores (40° cada, cobrindo 360°) |
| Função de ativação | ReLU (hidden), Sigmoid (output) |
| Regularização | Dropout 0.2 por camada |

**Base teórica:**
- Goodfellow et al. (2016): MLPs como aproximadores universais
- Thrun et al. (2005): Processamento de dados LIDAR em robótica

**Treinamento (auto-labeling):**
```python
# Threshold-based labeling: setor com min_distance < 0.5m = obstáculo
labels[sector] = 1.0 if min(sector_distances) < threshold else 0.0
```

**Limitação atual:** Modelo treinado com dados sintéticos. Necessita validação com dados reais do Webots.

#### 2.2.2 Visão Computacional - Detecção de Cubos

**Abordagem híbrida:** HSV segmentation (primário) + CNN classificação (opcional)

**Pipeline HSV implementado:**

```python
COLOR_RANGES = {
    'green': H=[35,85], S=[80,255], V=[50,255],
    'blue':  H=[100,130], S=[80,255], V=[50,255],
    'red':   H=[0,10]∪[160,180], S=[80,255], V=[50,255]
}
```

| Filtro | Valor | Propósito |
|--------|-------|-----------|
| MIN_CONTOUR_AREA | 100 px² | Detectar cubos distantes (~2m) |
| MAX_CONTOUR_AREA | 3000 px² | Excluir caixas de depósito |
| MAX_BBOX_FRACTION | 0.20 | Rejeitar objetos muito grandes |
| MIN_BBOX_ASPECT | 0.5 | Rejeitar contornos alongados |

**Estimativa de distância (modelo pinhole simplificado):**

```python
distance = CUBE_SIZE / (apparent_size × tan(FOV/2))
# CUBE_SIZE = 0.05m, FOV = 60°
# tan(30°) ≈ 0.577
distance = 0.05 / (apparent_size × 0.577)
```

**Base teórica:**
- Gonzalez & Woods (2018): Digital Image Processing - segmentação por cor
- Bradski & Kaehler (2008): Learning OpenCV - HSV color spaces

**Limitação atual:** Classificação de cor sensível a iluminação. HSV ranges calibrados empiricamente para Webots, podem falhar com sombras ou reflexos.

### 2.3 Lógica Fuzzy (Planejada mas NÃO Integrada)

**Design especificado (DECISÃO 018):**

```
Entradas fuzzy:
- distance_to_obstacle: {very_close, close, medium, far}
- angle_to_obstacle: {left, front_left, front, front_right, right}
- distance_to_cube: {very_close, close, medium, far}
- angle_to_cube: {left, center, right}

Saídas fuzzy:
- linear_velocity: {stop, slow, medium, fast}
- angular_velocity: {hard_left, left, straight, right, hard_right}
- action: {avoid, search, approach, grasp}

Regras: 25 (planejadas, 15 safety + 5 task + 5 exploration)
Defuzzificação: Centróide (Mamdani)
```

**Base teórica:**
- Zadeh (1965): Fuzzy Sets - fundamentos matemáticos
- Mamdani & Assilian (1975): Controlador fuzzy para sistemas complexos
- Saffiotti (1997): Fuzzy logic in autonomous robot navigation

**STATUS:** Controlador fuzzy especificado em `src/control/fuzzy_controller.py` mas **NÃO INTEGRADO** no MainControllerV2 atual. O controle atual usa lógica determinística.

---

## 3. Arquitetura Implementada

### 3.1 Visão Geral do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    MainControllerV2                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Máquina de Estados (5 estados)              ││
│  │  SEARCHING → APPROACHING → GRASPING → DEPOSITING         ││
│  │       ↑____________↓_____________↓___________↓           ││
│  │                        AVOIDING (emergência)             ││
│  └─────────────────────────────────────────────────────────┘│
│                              ↓                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐│
│  │VisionSvc │ │Movement  │ │   Arm    │ │NavigationService ││
│  │(tracking)│ │Service   │ │ Service  │ │(approach coord)  ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘│
│       ↓            ↓            ↓                ↓           │
│  CubeDetector    Base.py    Arm.py/Gripper   LIDAR RNA     │
│  (HSV/CNN)     (omnidir)     (5-DOF)        (SimpleMLP)    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Serviços Modulares (DECISÃO 028)

**Motivação:** Arquitetura anterior monolítica apresentava:
- State oscillation: SEARCHING↔APPROACHING ~50 transições/min
- Múltiplos cubos confundidos (tracking só por cor)
- GRASPING nunca alcançado (condições resetavam)
- Impossível testar componentes isoladamente

**Serviços extraídos:**

| Serviço | Responsabilidade | Base Teórica |
|---------|------------------|--------------|
| **MovementService** | Comandos dead-reckoning (forward, turn, strafe) | Siegwart & Nourbakhsh (2004) |
| **ArmService** | Sequências de grasp/deposit sem movimento | Craig (2005) |
| **VisionService** | Tracking estável com persistência por posição | Bradski & Kaehler (2008) |
| **NavigationService** | Coordenação movimento + visão para approach | Latombe (1991) |

### 3.3 VisionService - Tracking Estável

**Problema resolvido:** Oscilação entre múltiplos cubos da mesma cor.

**Solução implementada:**

```python
class TrackedCube:
    track_id: int           # Identificador único
    color: str              # 'green', 'blue', 'red'
    distance: float         # metros
    angle: float            # graus (+ = direita)
    frames_tracked: int     # estabilidade
    last_position: (x, y)   # para matching por posição

# Matching por POSIÇÃO, não apenas cor
def _find_match(detections, target):
    same_color = filter(d.color == target.color)
    best = argmin(angle_diff + distance_diff)
    return best if score < threshold else None
```

| Parâmetro | Valor | Propósito |
|-----------|-------|-----------|
| LOST_THRESHOLD | 60 frames (~2s) | Evitar perda prematura |
| MIN_CONFIDENCE | 0.60 | Filtrar detecções ruidosas |
| POSITION_TOLERANCE | 0.30m | Matching espacial |
| ANGLE_TOLERANCE | 20° | Matching angular |
| MIN_FRAMES_RELIABLE | 10 | Hysteresis para estabilidade |

### 3.4 NavigationService - Approach em Duas Fases

**Algoritmo implementado:**

```
Fase 1: ALIGN
    while |angle| > 4°:
        omega = angle × 0.5 (P-controller)
        move_continuous(vx=0, omega=omega)

Fase 2: APPROACH
    while distance > 0.30m:
        if |angle| > 16°: goto ALIGN  # histerese
        if obstacle_front < 0.30m:
            lateral_dodge()
        else:
            move_continuous(vx=0.08, omega=angle×0.02)
```

**Parâmetros de approach:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| ALIGN_THRESHOLD_EXIT | 4° | Precisão necessária para grasp |
| ALIGN_THRESHOLD_ENTER | 8° | Histerese para evitar flip-flop |
| APPROACH_DISTANCE | 0.30m | Offset câmera→gripper |
| APPROACH_SPEED | 0.08 m/s | Precisão sobre velocidade |

### 3.5 ArmService - Sequências de Manipulação

**Grasp sequence:**

```
1. prepare_grasp():
   - gripper.release()
   - arm.set_height(FRONT_PLATE)  # braço levantado
   - wait(2.0s)

2. execute_grasp():
   - arm.set_height(FRONT_FLOOR)  # abaixar para cubo
   - wait(1.5s)
   - gripper.grip()
   - wait(1.0s)
   - verify has_object()  # sensor do gripper
   - arm.set_height(FRONT_PLATE)  # levantar com cubo
   - wait(2.0s)
```

**Base teórica:**
- Craig (2005): Introduction to Robotics - cinemática de manipuladores
- Siciliano et al. (2009): Robotics - controle de grippers

---

## 4. Estado Atual e Problemas Identificados

### 4.1 O Que Está Funcionando

| Componente | Status | Evidência |
|------------|--------|-----------|
| Detecção HSV de cubos | ✅ Funcional | Detecta cubos a 0.3-3.0m |
| SimpleLIDARMLP | ✅ Treinado | 97.7% accuracy em dados sintéticos |
| Serviços isolados | ✅ Testáveis | Testes unitários passando |
| MovementService | ✅ Funcional | Square pattern executa corretamente |
| ArmService | ✅ Parcial | Posições corretas, grasp a validar |

### 4.2 O Que NÃO Está Funcionando

| Problema | Causa Provável | Impacto |
|----------|----------------|---------|
| **Integração end-to-end** | State machine não alcança GRASPING consistentemente | Sistema não coleta cubos |
| **Approach-to-grasp gap** | Distância final ~0.30m, gripper alcança ~0.25m | Cubo fora do alcance |
| **Deposit navigation** | Coordenadas hardcoded, sem navegação real | Deposita no local errado |
| **Fuzzy controller** | Não integrado no MainControllerV2 | Controle determinístico apenas |
| **RNA para navegação** | Usado apenas para obstacle flag, não para controle | Subutilizado |

### 4.3 Análise de Root Causes

**RC-1: Gap de distância approach→grasp**

```
Camera detecta cubo: OK
Approach para até 0.30m: OK (APPROACH_DISTANCE)
Final approach: 0.08m (hardcoded em _do_grasping)
Gripper reach: ~0.25m do centro do robô
Camera offset: ~0.15m à frente do centro

Problema: 0.30 - 0.08 = 0.22m final, mas gripper alcança 0.25m
          → margem de apenas 3cm, muito sensível a erros de odometria
```

**RC-2: Fuzzy não integrado**

O requisito obrigatório de lógica fuzzy foi ESPECIFICADO (25 regras, DECISÃO 018) mas o MainControllerV2 atual usa controle determinístico:

```python
# main_controller_v2.py atual (determinístico)
if target.distance <= APPROACH_DISTANCE:
    transition_to(GRASPING)

# Deveria ser (fuzzy)
fuzzy_action = fuzzy_controller.evaluate(distance, angle, obstacle)
if fuzzy_action == "grasp":
    transition_to(GRASPING)
```

**RC-3: Deposit sem navegação**

```python
# main_controller_v2.py:335 - TODO ainda presente
def _do_depositing(self):
    print(f"[Depositing] Moving to {self.target_cube_color} box")
    # TODO: Navigate to correct box based on color
    # For now, just deposit in place (proof of concept)
```

---

## 5. Compliance com Requisitos

### 5.1 Requisitos Obrigatórios

| Requisito | Status | Detalhes |
|-----------|--------|----------|
| **RNA para detecção de obstáculos** | ⚠️ Parcial | SimpleLIDARMLP implementado e treinado, mas usado apenas como flag, não para controle |
| **Lógica Fuzzy para controle** | ❌ Não integrado | Especificado (25 regras) mas MainControllerV2 usa lógica determinística |
| **GPS proibido** | ✅ Compliant | Código não usa GPS na demonstração |
| **supervisor.py inalterado** | ✅ Compliant | Zero modificações |
| **15 cubos coloridos** | ⚠️ Sistema não funcional | Supervisor spawna 15, mas robô não coleta consistentemente |

### 5.2 Gap de Implementação

**Para compliance completo, falta:**

1. **Integrar fuzzy_controller.py no MainControllerV2**
   - Substituir lógica determinística por inferência fuzzy
   - Usar outputs fuzzy para velocidades e decisões

2. **Usar RNA para controle real**
   - SimpleLIDARMLP informa apenas obstacle_distance
   - Deveria alimentar entradas do fuzzy controller

3. **Implementar navegação para deposit**
   - Atualmente deposita "in place"
   - Necessita odometria ou visual servoing para caixas

---

## 6. Referências Científicas Utilizadas

### 6.1 Fundamentação Primária

| Referência | Uso no Projeto |
|------------|----------------|
| Brooks (1986) - Subsumption Architecture | Organização hierárquica de comportamentos |
| Zadeh (1965) - Fuzzy Sets | Base matemática para controlador fuzzy |
| Mamdani & Assilian (1975) | Inferência fuzzy tipo Mamdani |
| Goodfellow et al. (2016) - Deep Learning | Arquitetura MLP para LIDAR |
| Thrun et al. (2005) - Probabilistic Robotics | Processamento de sensores, odometria |

### 6.2 Fundamentação Secundária

| Referência | Uso no Projeto |
|------------|----------------|
| Craig (2005) - Robotics | Cinemática do braço YouBot |
| Siciliano et al. (2009) | Controle de manipuladores |
| Latombe (1991) - Robot Motion Planning | NavigationService approach |
| Bradski & Kaehler (2008) - Learning OpenCV | VisionService tracking |
| Saffiotti (1997) - Fuzzy Navigation | Design do controlador fuzzy |

---

## 7. Recomendações para Ajustes

### 7.1 Prioridade Alta (Requisitos Obrigatórios)

1. **Integrar FuzzyController no MainControllerV2**
   ```python
   # Proposta de integração
   fuzzy_inputs = {
       'distance_to_obstacle': self._get_min_obstacle_distance(),
       'angle_to_obstacle': lidar_obstacle_angle,
       'distance_to_cube': target.distance if target else 3.0,
       'angle_to_cube': target.angle if target else 0.0
   }
   fuzzy_output = self.fuzzy_controller.evaluate(fuzzy_inputs)
   self.movement.move_continuous(
       vx=fuzzy_output.linear_velocity,
       omega=fuzzy_output.angular_velocity
   )
   ```

2. **Usar RNA no pipeline de controle**
   - SimpleLIDARMLP → 9 setores de ocupação → entradas fuzzy
   - Não apenas min_distance, mas mapa espacial de obstáculos

### 7.2 Prioridade Média (Funcionalidade)

3. **Ajustar geometria approach→grasp**
   - APPROACH_DISTANCE: 0.30m → 0.25m
   - Final approach: 0.08m → 0.05m
   - Ou: recalibrar com medições reais no Webots

4. **Implementar navegação para deposit**
   - Opção A: Odometria incremental (relativa ao ponto de grasp)
   - Opção B: Visual servoing para caixas coloridas
   - Opção C: Posições fixas conhecidas das caixas

### 7.3 Prioridade Baixa (Robustez)

5. **Treinar CNN para classificação de cor**
   - HSV sensível a iluminação
   - CNN mais robusto a variações

6. **Validar SimpleLIDARMLP com dados reais**
   - Coletar scans do Webots em diferentes posições
   - Re-treinar ou fine-tune

---

## 8. Conclusão

O projeto apresenta uma **arquitetura bem fundamentada teoricamente** com serviços modulares, tracking estável, e especificações de RNA e fuzzy. Entretanto, a **integração end-to-end não está funcional**:

- **RNA (SimpleLIDARMLP):** Implementada mas subutilizada
- **Fuzzy Controller:** Especificado mas não integrado
- **Navegação:** Approach funcional, deposit não implementado

Para compliance com os requisitos obrigatórios de MATA64, é necessário:
1. Integrar o FuzzyController existente no loop de controle
2. Usar os outputs da RNA como entradas do fuzzy
3. Implementar navegação básica para deposit

O esforço de modularização (DECISÃO 028) facilita essas integrações, pois cada serviço pode ser testado isoladamente antes da integração final.

---

**Documento preparado para revisão do Professor Luciano Oliveira**
**Última atualização:** 2025-11-30
