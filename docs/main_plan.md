# Plano Principal — YouBot Fuzzy Controller

Este documento descreve o plano completo para implementar o controlador inteligente do YouBot que coleta cubos coloridos e os deposita nas caixas correspondentes usando Lógica Fuzzy, LIDAR duplo, visão computacional (YOLO + AdaBoost) e odometria com correção ICP.

## 1. Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MISSION PIPELINE                                   │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │SCAN_GRID│ → │   PICK   │ → │ DELIVER  │ → │  RETURN  │ → │SCAN_GRID│   │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
        ↑                              ↑                              ↑
   ┌────┴────┐                    ┌────┴────┐                    ┌────┴────┐
   │  FUZZY  │                    │  WORLD  │                    │ SENSORS │
   │ PLANNER │←───────────────────│  MODEL  │←───────────────────│ FUSION  │
   └─────────┘                    └─────────┘                    └─────────┘
        │                              ↑                              ↑
        ↓                         ┌────┴────┐                    ┌────┴────┐
   ┌─────────┐                    │GRID MAP │                    │DUAL-LIDAR│
   │  BASE   │                    │ODOMETRY │                    │ CAMERA  │
   │CONTROLLER│                   │  ICP    │                    │  YOLO   │
   └─────────┘                    └─────────┘                    └─────────┘
```

## 2. Fases de Implementação
 
### Fase 1: Sensores e Mundo (✅ Concluído)

**Objetivo:** Configurar os sensores LIDAR duplo e câmera no mundo Webots.

1. **Modificar `IA_20252/worlds/IA_20252.wbt`:**
   - [x] Adicionar `lidar_low` (altura 5cm, FOV 180°, tilt -0.05 rad, range 0.03-2.5m)
   - [x] Adicionar `lidar_high` (altura 25cm, FOV 360°, 2 camadas, range 0.1-7.0m)
   - [x] Manter câmera existente (128x128 px, frontal)

2. **Criar `sensors/lidar_adapter.py`:**
   - [x] Inicializar ambos LIDARs com parâmetros de `config.py`
   - [x] Método `summarize()` → `LidarSnapshot` (front/left/right/density)
   - [x] Método `cube_candidates()` → detectar objetos baixos (cubos)
   - [x] Método `navigation_points()` → pontos para ICP

3. **Criar `sensors/camera_stream.py`:**
   - [x] Wrapper para captura de frames (`CameraFrame` dataclass)
   - [x] Método `capture()` retorna buffer RGB

### Fase 2: Percepção Neural e Visão (✅ Estrutura Pronta)

**Objetivo:** Implementar pipeline de detecção YOLO → AdaBoost → HSV.

1. **Criar `perception/yolo_detector.py`:**
   - [x] Carregar modelo YOLOv8n de `models/yolov8n-cubes.pt`
   - [x] Método `detect(image)` → lista de `Detection` (bbox, confidence, label)
   - [x] Graceful degradation se `ultralytics` não disponível

2. **Criar `perception/adaboost_classifier.py`:**
   - [x] Carregar modelo `models/adaboost_color.pkl` via `joblib`
   - [x] Extrair features: HSV histogram + HOG via OpenCV
   - [x] Método `predict(patch)` → cor classificada

3. **Criar `perception/color_classifier.py`:**
   - [x] Pipeline híbrido: AdaBoost → HSV heuristic
   - [x] Método `classify_patch(image)` com fallback

4. **Criar `perception/cube_detector.py`:**
   - [x] Orquestrar: YOLO detection → color classification
   - [x] Método `detect(frame, lidar)` → `CubeHypothesis`
   - [x] Calcular bearing, alignment, distance, confidence

5. **Treinar modelos (✅ AdaBoost Pronto):**
   - [x] Executar `tools/run_dataset_capture.py` para gerar imagens (300 imagens: 100/cor)
   - [ ] Treinar YOLOv8n com dataset sintético (opcional - HSV fallback funciona)
   - [x] Treinar AdaBoost com patches coloridos (`models/adaboost_color.pkl` - 4KB)
   - [x] Colocar pesos em `models/`

### Fase 3: Localização e Mapeamento (✅ Estrutura Pronta)

**Objetivo:** Implementar odometria Mecanum + ICP + Grid Map.

1. **Criar `localization/mecanum_odometry.py`:**
   - [x] Integrar velocidades das rodas via encoders
   - [x] Fallback: integrar comandos de velocidade
   - [x] Método `update(dt)` → (x, y, theta)
   - [x] Acumular `distance_since_reset`

2. **Criar `localization/icp_correction.py`:**
   - [x] Usar Open3D para scan matching
   - [x] Método `correct(odometry_pose, lidar_points)`
   - [x] Aplicar correção quando drift > 0.15m

3. **Criar `mapping/grid_map.py`:**
   - [x] Grid 70x40 células (10cm resolução, arena 7x4m)
   - [x] Estados: unknown, free, obstacle, cube
   - [x] Método `mark_pose()` → células visitadas
   - [x] Método `mark_obstacles(points)`
   - [x] Método `next_unvisited_patch()` → exploração

### Fase 4: Controle Fuzzy (✅ Concluído)

**Objetivo:** Implementar regras fuzzy para navegação e decisão.

1. **Criar `control/fuzzy_planner.py`:**
   - [x] Variáveis linguísticas: `front_distance`, `obstacle_density`, `cube_alignment`
   - [x] Funções de pertinência triangulares/trapezoidais
   - [x] Regras de navegação (evitar obstáculos)
   - [x] Regras de aproximação (alinhar com cubo)
   - [x] Regras de busca (explorar patches não visitados)
   - [x] Defuzzificação: centróide → `MotionCommand(vx, vy, omega)`

2. **Regras implementadas:**
   ```
   IF front_distance=VERY_CLOSE AND density=HIGH → RETREAT + ROTATE
   IF cube_detected AND alignment=CENTER AND distance=GOOD → STOP + GRIP
   IF load_state=LOADED AND heading_error=SMALL → DRIVE_TO_BOX
   IF no_cube AND patch_available → EXPLORE_PATCH
   ```

### Fase 5: Manipulação (✅ Concluído)

**Objetivo:** Sequenciar braço e garra para coleta/depósito.

1. **Criar `manipulation/arm_service.py`:**
   - [x] Fila de comandos com timers
   - [x] Presets: RESET (1.5s), FLOOR (2.5s), PLATE (2.0s)
   - [x] Método `queue(lift_request, gripper_request)`
   - [x] Método `update(dt)` → processar fila
   - [x] Flag `is_gripping` para coordenação

### Fase 6: Orquestração da Missão (✅ Concluído)

**Objetivo:** State machine que coordena todo o sistema.

1. **Criar `mission/pipeline.py`:**
   - [x] Estados: `SCAN_GRID`, `PICK`, `DELIVER`, `RETURN`
   - [x] Integrar todos os módulos no `step()`
   - [x] Fluxo:
     ```
     SCAN_GRID: explorar arena buscando cubos
         ↓ (cubo detectado)
     PICK: aproximar, alinhar, baixar braço, fechar garra
         ↓ (cubo coletado)
     DELIVER: navegar até caixa da cor correspondente
         ↓ (chegou na caixa)
     RETURN: soltar cubo, recuar
         ↓ (cubo depositado)
     SCAN_GRID: continuar buscando
     ```

2. **Criar `app.py` (entrypoint):**
   - [x] Instanciar todos os módulos
   - [x] Wiring de dependências
   - [x] Loop principal: `while robot.step() != -1: mission.step()`

## 3. Configuração e Parâmetros

### 3.1 Constantes Críticas (`config.py`)

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `ARENA_SIZE` | (7.0, 4.0) | Dimensões da arena em metros |
| `CUBE_SIZE` | 0.03 | Tamanho do cubo (3cm) |
| `DANGER_ZONE` | 0.25 | Distância de perigo (metros) |
| `CUBE_DETECTION_MIN_DISTANCE` | 0.05 | Distância mínima para detectar cubo |
| `CUBE_DETECTION_MAX_DISTANCE` | 1.5 | Distância máxima para detectar cubo |
| `CUBE_HEIGHT_DIFFERENCE_THRESHOLD` | 0.2 | Diferença de altura para identificar cubo |
| `YOLO_CONFIDENCE_THRESHOLD` | 0.35 | Confiança mínima para detecção YOLO |
| `BASE_MAX_SPEED` | 0.3 | Velocidade máxima da base (m/s) |

### 3.2 Caixas de Depósito

| Cor | Posição (x, y) |
|-----|----------------|
| GREEN | (0.48, 1.58) |
| BLUE | (0.48, -1.62) |
| RED | (2.31, 0.01) |

### 3.3 Sensores LIDAR

| Sensor | Posição | FOV | Range | Uso |
|--------|---------|-----|-------|-----|
| `lidar_low` | (0.15, 0.05, 0) | 180° | 0.03-2.5m | Detectar cubos |
| `lidar_high` | (0, 0.25, 0) | 360° | 0.1-7.0m | Navegação |

**Configuração de Setores (config.py) - Rotation 0 (sem rotação):**
```python
# Mapeamento com rotation 0: Índice 180=FRENTE, 90=ESQUERDA(bloqueado), 0=TRÁS, 270=DIREITA
# Dead zones: (70-110) corpo bloqueia lado esquerdo (índice 90 sempre ~0.11m)
front_sector=(170, 190)  # Em torno de 180 = frente
left_sector=(120, 150)   # Front-left diagonal (evita índice 90 bloqueado)
right_sector=(210, 240)  # Front-right diagonal
LIDAR_DEAD_ZONES = [(70, 110)]
```

**Debug LIDAR:** Console mostra `LIDAR[F=180]: 180:X.XX | 90:X.XX | ...` para calibração.

## 4. Dependências de Software

### 4.1 Python Packages (opcionais para ML)

```bash
pip install numpy opencv-python ultralytics open3d joblib scikit-learn
```

### 4.2 Modelos Pré-treinados

| Arquivo | Localização | Status | Descrição |
|---------|-------------|--------|-----------|
| `yolov8n-cubes.pt` | `models/` | ⚠️ BASE | Modelo YOLOv8n base (6.5MB) - fallback HSV ativo |
| `adaboost_color.pkl` | `models/` | ✅ TREINADO | Classificador AdaBoost (4KB, 300 imgs) |

**Dataset:** 300 imagens sintéticas em `datasets/cubes/train/` (100 por cor: red, green, blue).
Sistema usa pipeline: YOLO → AdaBoost → HSV heuristic (fallback).

## 5. Próximos Passos

### 5.1 Treinamento de Modelos (✅ Concluído)

Dataset e AdaBoost já treinados:
- **Dataset:** 300 imagens em `datasets/cubes/train/` (100 por cor)
- **AdaBoost:** `models/adaboost_color.pkl` (4KB)

**YOLO (Opcional):** Se quiser treinar modelo específico:
```bash
cd IA_20252/controllers/youbot_fuzzy
yolo train model=yolov8n.pt data=datasets/cubes/data.yaml epochs=50 imgsz=128
cp runs/detect/train/weights/best.pt models/yolov8n-cubes.pt
```

**NOTA:** Sistema funciona com HSV heuristic como fallback. AdaBoost melhora precisão.

### 5.2 Calibração e Ajustes (✅ LIDAR + Grasp Calibrados)

1. **LIDAR:** ✅ Concluído
   - Rotação corrigida: -90° Z (índice 0=frente, alinhado com braço/câmera)
   - Setores recalculados para nova orientação
   - Dead zones: corpo traseiro + zona do braço

2. **Fuzzy:** ✅ Concluído
   - Approach rule usa média ponderada (mais responsivo)
   - Reverse threshold ajustado para 0.18m (alinhado com arm reach)
   - vx aumentado para 0.08-0.15 quando caminho livre

3. **Manipulação:** ✅ Concluído
   - Sequência completa: OPEN(1s) → RESET(1.5s) → FLOOR(2.5s) → APPROACH(2s) → GRIP(1.5s) → LIFT(2s)
   - Tempo total: ~11 segundos
   - load_state transição via arm_service.is_gripping (evita race condition)

### 5.3 Testes Finais

1. Executar cenário completo: coletar 15 cubos
2. Verificar tempo total < 10 minutos
3. Confirmar zero colisões
4. Gravar vídeo para entrega

## 6. Estrutura de Diretórios Final

```
IA_20252/controllers/youbot_fuzzy/
├── youbot_fuzzy.py          # Entrypoint Webots
├── app.py                   # Wiring principal
├── config.py                # Constantes
├── data_types.py            # Dataclasses
├── logger.py                # Logging
├── sensors/
│   ├── lidar_adapter.py     # Dual-LIDAR
│   └── camera_stream.py     # Camera wrapper
├── perception/
│   ├── utils.py             # frame_to_bgr
│   ├── yolo_detector.py     # YOLOv8
│   ├── adaboost_classifier.py
│   ├── color_classifier.py
│   └── cube_detector.py
├── localization/
│   ├── mecanum_odometry.py
│   └── icp_correction.py
├── mapping/
│   └── grid_map.py
├── world/
│   └── model.py
├── control/
│   └── fuzzy_planner.py
├── motion/
│   └── base_controller.py
├── manipulation/
│   └── arm_service.py
├── mission/
│   └── pipeline.py
├── models/                   # Pesos ML
│   ├── yolov8n-cubes.pt      # ⚠️ Placeholder (precisa treinar)
│   └── adaboost_color.pkl    # ❌ Criar via train_adaboost.py
└── tools/
    ├── run_dataset_capture.py  # Gera imagens no Webots
    └── train_adaboost.py       # Treina classificador de cor
```

## 7. Checklist de Entrega

- [ ] Código funcional no Webots R2025a
- [ ] Vídeo demonstrativo (≤ 15 min)
- [ ] Explicação conceitual (sem mostrar código)
- [ ] Robô coleta e deposita cubos corretamente
- [ ] Sem uso de GPS (apenas LIDAR + câmera)
- [ ] Supervisor não modificado

---

**Última atualização:** 06/12/2025
**Referência:** `CLAUDE.md`, `docs/TESTES.md`

### Changelog Recente:
- ✅ **FIX CRÍTICO:** Rotação LIDAR corrigida no world file (rotation 0 - sem rotação)
- ✅ **FIX CRÍTICO:** HSV hue scale corrigido (hue*30 para range 0-180)
- ✅ Setores LIDAR recalculados: front=(170,190), left=(120,150), right=(210,240)
- ✅ Dead zones atualizados: [(70,110)] - corpo bloqueia índice 90
- ✅ Color detection: agora escaneia TODAS as cores via HSV masks
- ✅ Sequência de grasp completa: OPEN → RESET → FLOOR → APPROACH → GRIP → LIFT
- ✅ Fuzzy planner: approach rule usa média ponderada (não min)
- ✅ Grid exploration: nearest-first patch selection
- ✅ ICP validation: verifica nav_points > 10 antes de aplicar
- ✅ AdaBoost treinado: 300 imagens, modelo de 4KB
- ✅ Odometria: parâmetros corrigidos (wheel_radius=0.05, lx=0.228, ly=0.158)
