# CLAUDE — Diretrizes Para o Projeto YouBot Fuzzy

## 1. Missão e Objetivos
- **Tarefa Principal:** coletar 15 cubos (3 cm) distribuídos aleatoriamente, identificar cor (verde, azul, vermelho) e depositar no `PlasticFruitBox` correspondente sem uso de GPS (`Final Project Requirements/Trabalhofinal.html`).
- **Restrições:** navegação somente com LIDAR + câmera (sensores adicionais opcionais); rotina de spawn de cubos não pode ser alterada; apresentação final deve explicar conceitos sem mostrar código.
- **Entrega:** código + vídeo ≤ 15 min demonstrando robô em `IA_20252/worlds/IA_20252.wbt`.

## 2. Ambiente e Métricas Físicas
### 2.1 Arena e Objetos Fixos (`IA_20252/worlds/IA_20252.wbt`)
*Fonte direta: campos `RectangleArena`, `PlasticFruitBox` e `WoodenBox` definidos no mundo oficial.*
| Item | Valor |
| --- | --- |
| Piso | `RectangleArena` 7 m (x) × 4 m (y), paredes 0,3 m
| Textura | `../textures/Material.001_baseColor.png`
| Posição inicial do YouBot | (-3.9997, ~0, 0.102)

### 2.2 Caixas de Depósito (`PlasticFruitBox`)
*Coordenadas vindas de `IA_20252/worlds/IA_20252.wbt`, linhas `PlasticFruitBox`.*
| Caixa | Posição (x, y) [m] | Cor |
| --- | --- | --- |
| GREEN | (0.48, 1.58) | (0, 1, 0)
| BLUE | (0.48, -1.62) | (0, 0, 1)
| RED | (2.31, 0.01) — rot. 90° | (1, 0, 0)

### 2.3 Obstáculos de Madeira (`WoodenBox`, tamanho 0.3 m³)
*Coordenadas extraídas dos nós `WoodenBox` do mesmo arquivo de mundo.*
| Nome | x | y |
| --- | --- | --- |
| A | 0.60 | 0.00 |
| B | 1.96 | -1.24 |
| C | 1.95 | 1.25 |
| D | -2.28 | 1.50 |
| E | -1.02 | 0.75 |
| F | -1.02 | -0.74 |
| G | -2.27 | -1.51 |

### 2.4 Spawn dos Cubos (`controllers/supervisor/supervisor.py`)
*Parâmetros confirmados no script supervisor oficial.*
- Quantidade padrão: 15.
- Intervalo X: [-3.0, 1.75], Y: [-1.0, 1.0], Z ≈ 0.0155 m.
- Raio de exclusão: `min_dist = 2.5 * size = 0.075 m` + amortecimento aos obstáculos (raio obstáculo + 0.03 m).
- `recognitionColors` iguais à cor do cubo → usar para visão.

### 2.5 Cinemática do YouBot
Fonte: `IA_20252/controllers/youbot/*.py` + `draft.md`.
- **Base:** rodas Mecanum (raio 0.05 m, LX 0.228 m, LY 0.158 m). `Base.move(vx, vy, ω)` já converte para velocidades de roda.
- **Braço:** presets `ArmHeight`/`ArmOrientation`; segmentos: `[0.253, 0.155, 0.135, 0.081, 0.105]` m; alcance prático no `FRONT_FLOOR` ≈ 0.25 m frontal, altura 0.016 m.
- **Garra:** abertura útil 0–25 mm; usar os dois motores (`finger::left/right`) conforme observação do `draft.md` para garantir simetria.

### 2.6 Sensores montados no corpo (`IA_20252/worlds/IA_20252.wbt`)
| Sensor | Posição/Parâmetros | Uso |
| --- | --- | --- |
| `lidar_low` | `translation 0.15 0.05 0`; `tiltAngle -0.05 rad`; `horizontalResolution 180`; `fieldOfView π (180°)`; `range [0.03, 2.5]`; `type "fixed"` | Detecção de cubos rente ao solo (3 cm) — FOV frontal |
| `lidar_high` | `translation 0 0.25 0`; 2 camadas; `horizontalResolution 360`; `fieldOfView 2π (360°)`; `range [0.1, 7.0]`; `type "rotating"`; `defaultFrequency 10` | Mapeamento de paredes/obstáculos, navegação 360° |
| `camera` | `translation 0.27 0 -0.06`; 128×128 px | Classificação visual (YOLO → AdaBoost → HSV) |

## 3. Teoria e Referências Obrigatórias
### 3.1 Lógica Fuzzy (DataCamp — *Fuzzy Logic in AI*)
- Estrutura oficial: `fuzzification → knowledge base → inference engine → defuzzification`.
- **Fuzzification:** converter leituras (distância, ângulo, cor) em graus de pertinência via funções triangulares/trapezoidais.
- **Knowledge Base:** regras linguísticas do tipo `SE ... ENTÃO ...`, cada uma ponderada por especialistas.
- **Inference Engine:** agrega as regras usando operadores lógicos (min/max) conforme artigo.
- **Defuzzification:** método do centróide, conforme recomendado, para gerar `vx`, `vy`, `ω` contínuos.

### 3.2 Tutoriais Webots (Guia Oficial)
- **Tutorial 1:** configurar primeira simulação (garante que o novo controlador use a estrutura correta).
- **Tutorial 2:** alterar ambiente; útil para validar obstáculos/caixas.
- **Tutorial 3:** ajustes de aparência — alinhado com requisito de manter cores dos cubos/caixas.
- **Tutorial 4:** controllers em detalhes; mostra padrão para comunicação com a API.
- **Tutorial 8 (Supervisor):** reforça regra de não alterar lógica de spawn e explica supervisores.

### 3.3 Sensores e Objetos
- **Lidar Reference** (`https://cyberbotics.com/doc/reference/lidar`):
  - `horizontalResolution`: número de raios por camada (balanço entre cobertura e custo).
  - `fieldOfView`: limite ≤ π rad para projeção planar; configuraremos ~180°.
  - `minRange`/`maxRange`: limites de leitura que definem pertinência `Muito Perto`/`Longo`.
  - Funções `enable`, `enablePointCloud`, `get_range_image` e `get_point_cloud` são obrigatórias no adaptador.
- **Solid Reference** (`https://cyberbotics.com/doc/reference/solid?version=R2023a`):
  - Regras para `boundingObject`, `physics`, nomes exclusivos e escala uniforme — necessárias ao adicionar sensores extras.
- **Object Factory (PlasticFruitBox/WoodenBox)** (`https://cyberbotics.com/doc/guide/object-factory?version=R2023a`):
  - Confirma dimensões/cores padrões que serão usadas pelo classificador de cor.
- **Repositório `cyberbotics/webots` (branch `released`)**:
  - Exemplo de controladores avançados, especialmente integração com ROS2 e pipelines de percepção; referência para estilo.

### 3.4 Frameworks de Visão Computacional
- **OpenCV** ([opencv.org](https://opencv.org/)): biblioteca para pré-processamento (ROI crops, HSV, HOG) e carregamento do classificador AdaBoost.
- **Darknet / YOLO** ([github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)): base de dados e modelos a serem usados via Ultralytics YOLOv8 para detectar cubos/caixas.

## 4. Arquitetura de Software (Domain Driven)
### 4.1 Contextos Delimitados
| Contexto | Responsabilidades | Módulos sugeridos |
| --- | --- | --- |
| **Aquisição de Sensores** | Gerenciar LIDAR/câmera/IMU; normalizar timestamps | `sensors/lidar_adapter.py`, `sensors/camera_stream.py`
| **Percepção** | Segmentação de cor, CNN/MLP para mapeamento (obstáculos/cubos), leitura de `recognitionColors` | `perception/color_classifier.py`, `perception/cube_detector.py`
| **Modelagem do Mundo** | Construir grid polar e mapa topológico (caixas-alvo, obstáculos) | `world/model.py`
| **Decisão Fuzzy** | Avaliar variáveis linguísticas e aplicar regras | `control/fuzzy_planner.py`
| **Execução de Movimento** | Enfileirar comandos para `Base`, `Arm`, `Gripper` mantendo limites de velocidade | `motion/base_controller.py`, `manipulation/arm_service.py`
| **Orquestração da Missão** | Loop de alto nível: buscar → coletar → classificar → depositar | `mission/pipeline.py`
| **Localização** | Odometria Mecanum + correção ICP | `localization/mecanum_odometry.py`, `localization/icp_correction.py`
| **Mapeamento** | Grid 10 cm, patches visitados, contagem de cubos | `mapping/grid_map.py`

### 4.2 Fluxo de Dados
1. **Sensores** produzem `RangeScan`, `PointCloud`, `RGBFrame` sincronizados.
2. **Percepção** gera `ObstacleMap` (distâncias agregadas) + `CubeHypothesis` (posição relativa, cor).
3. **Modelagem** alinha hipóteses com coordenadas da arena (usar odometria inferida + detecção de paredes via LIDAR).
4. **Decisão Fuzzy** recebe variáveis (distância frontal, densidade de obstáculos, alinhamento ao cubo, estado de carga) e devolve comandos de velocidade + estados do braço.
5. **Execução** valida contra limites (por exemplo, `Base.MAX_SPEED = 0.3 m/s`) e aplica smoothing temporal.
6. **Máquina de estados** (`MissionState.phase`): {`SCAN_GRID`, `PICK`, `DELIVER`, `RETURN`} orientam o fluxo de busca → coleta → entrega → reposicionamento.

### 4.3 Layout de Pacotes do Controlador
```
IA_20252/controllers/youbot_fuzzy/
├── youbot_fuzzy.py          # entrypoint Webots (adiciona sys.path e chama app.main)
├── app.py                   # wiring: instancia Robot, sensores, percepção, missão
├── config.py                # constantes (arena, sensores, ML, thresholds)
├── types.py                 # dataclasses compartilhados (LidarSnapshot, CubeHypothesis, etc.)
├── logger.py                # wrapper de logging configurável
├── sensors/
│   ├── lidar_adapter.py     # Dual-LIDAR: summarize() + cube_candidates() + navigation_points()
│   └── camera_stream.py     # CameraFrame com buffer RGB
├── perception/
│   ├── utils.py             # frame_to_bgr (conversão Webots → OpenCV)
│   ├── yolo_detector.py     # YOLOv8 via ultralytics (Detection dataclass)
│   ├── adaboost_classifier.py  # AdaBoost + HOG/HSV via joblib/OpenCV
│   ├── color_classifier.py  # híbrido: AdaBoost → HSV heuristic
│   └── cube_detector.py     # pipeline YOLO → AdaBoost → HSV → CubeHypothesis
├── localization/
│   ├── mecanum_odometry.py  # odometria via encoders ou velocidades
│   └── icp_correction.py    # correção de drift via Open3D ICP
├── mapping/
│   └── grid_map.py          # occupancy grid 10cm, patches visitados, cube_counts
├── world/
│   └── model.py             # WorldModel: box targets, obstacle map, goal vectors
├── control/
│   └── fuzzy_planner.py     # regras fuzzy → MotionCommand (vx, vy, omega)
├── motion/
│   └── base_controller.py   # aplica MotionCommand respeitando limites
├── manipulation/
│   └── arm_service.py       # sequência de lift/grip com timers
├── mission/
│   └── pipeline.py          # MissionPipeline: state machine SCAN_GRID→PICK→DELIVER→RETURN
└── tools/
    └── run_dataset_capture.py  # script para gerar dataset sintético
```
*Cada módulo interage apenas via objetos de domínio (ex.: `CubeHypothesis`, `MotionCommand`, `LidarSnapshot`), evitando importações cruzadas diretas.*

## 5. Stack de Controle (ANN + Fuzzy)
### 5.1 Variáveis Linguísticas (baseadas em métricas reais)
| Variável | Domínio (m ou rad) | Termos sugeridos |
| --- | --- | --- |
| `front_distance` | 0–5 | {Muito Perto (≤0.25), Perto (0.25–0.75), Médio (0.75–1.5), Longo (>1.5)}
| `lateral_clearance` | 0–3 | {Fechado, Parcial, Livre}
| `heading_error` | -π..π | {Grande Esquerda (<-0.6), Ajuste (~0), Grande Direita (>0.6)}
| `cube_alignment` (offset do braço) | -0.3..0.3 | {Esquerda, Central, Direita}
| `load_state` | discreto | {Vazio, Carregado}
| `goal_priority` | n/a | {Verde, Azul, Vermelho} via lógica determinística (contagem por cor).

### 5.2 Funções de Pertinência
- **Triangulares** para distâncias (e.g., `Muito Perto` centrado em 0.15 m, base até 0.35 m) respeitando alcance do braço.
- **Trapezoidais** para `heading_error` (miolo linear para estabilidade do controle diferencial).
- **Singletons** para estados discretos (`load_state`).

### 5.3 Exemplo de Regras
1. IF `front_distance` é `Muito Perto` AND `lateral_clearance` é `Fechado` → `retroceder` + `ω = ±0.3` para escapar.
2. IF `cube_detected` AND `cube_alignment` é `Central` AND `front_distance` em [0.35,0.45] → `parar base`, `baixar braço`, `fechar garra`.
3. IF `load_state` = `Carregado` AND `goal_color` = `Verde` AND `heading_error` pequeno → `dirigir` alvo GREEN.
4. IF `front_distance` = `Longo` AND `goal_priority` = `Vermelho` → `planejar` trajetória longa (ANN define referência) + `fuzzy` refina micro-ajustes.
- **Defuzzificação:** usar método do centroide para `vx`, `vy`, `ω`; comandos discretos do braço podem usar winner-take-all.

### 5.4 YOLO (Detecção)
- Modelo leve (YOLOv8n ou Darknet tiny) com pesos em `models/yolov8n-cubes.pt`.
- Responsável por retornar `bbox`, confiança e rótulo (cor) via `perception/yolo_detector.py`.
- Dataset sintético gerado no Webots (`tools/run_dataset_capture.py`) com domain randomization.

### 5.5 AdaBoost + OpenCV
- Classificador `AdaBoostColorClassifier` (`models/adaboost_color.pkl`) usando histogramas HSV + HOG (OpenCV).
- Atua como fallback/enriquecimento da cor quando YOLO não atinge confiança mínima.
- Integração direta no `ColorClassifier.classify_patch` para reutilizar pipeline.

### 5.6 Navegação/Odometria
- `MecanumOdometry` consome encoders `wheel*_sensor`; na ausência deles integra velocidades comandadas (`MotionCommand`).
- `ICPCorrection` (Open3D) aplica ICP quando deslocamento acumulado ≥ 0.15 m; mistura com odometria via peso 0.6.
- `GridMap` (0.1 m) mantém patches visitados, obstáculos e contagem de cubos detectados pelo `lidar_low`.
- Máquina de estados da missão (`MissionState.phase`): `SCAN_GRID → PICK → DELIVER → RETURN`.

## 6. Sensores e Calibração
### 6.1 LIDAR (Dual-LIDAR Strategy)
- **`lidar_high` (25 cm altura):**
  - 2 camadas, `horizontalResolution = 360`, `fieldOfView = 2π (360°)`, `range = [0.1, 7.0]`
  - Tipo: `rotating`, frequência 10 Hz
  - Setores definidos em `config.py`: front (150-210), left (210-270), right (90-150)
  - Uso: navegação, desvio de obstáculos, densidade frontal, ICP scan matching
- **`lidar_low` (5 cm altura, frontal):**
  - 1 camada, `horizontalResolution = 180`, `fieldOfView = π (180° frontal)`, `range = [0.03, 2.5]`
  - Tipo: `fixed`, `tiltAngle = -0.05 rad` (inclinado para baixo)
  - Uso: detectar objetos baixos (cubos de 3 cm) invisíveis ao LIDAR alto
- **Fusão de dados (`LidarAdapter.cube_candidates`):**
  - Compara leituras `lidar_low` com `lidar_high` no mesmo ângulo
  - Se `dist_high - dist_low > CUBE_HEIGHT_DIFFERENCE_THRESHOLD (0.2m)` → provável cubo
  - Filtra por `CUBE_DETECTION_MIN_DISTANCE (0.05m)` e `MAX_DISTANCE (1.5m)`
- `samplingPeriod`: 32 ms (múltiplo do `basicTimeStep` 16 ms)
- Point cloud habilitado apenas no `lidar_high` para ICP

### 6.2 Câmera
- Resolução default 128×128; manter `camera.enable(time_step)` já presente.
- Pipeline: YOLO (detecção) → AdaBoost (cor) → heurística HSV (fallback) via `CubeDetector`.
- Registrar `recognitionColors` do supervisor para comparar com média de pixels.

### 6.3 Outros Sensores
- Opcional adicionar `InertialUnit` ou `Gyro` (seguir `Solid` node rules) para reduzir deriva ao planificar trajetórias longas.

## 7. Manipulação e Sequência de Grasp (baseada no `draft.md`)
1. `Gripper.release()` → aguardar ≥1 s.
2. `Arm.set_height(RESET)` → aguardar 1.5 s.
3. `Arm.set_height(FRONT_FLOOR)` → 2.5 s.
4. `Base.move(0.05, 0, 0)` por 2 s (≈10 cm) – respeitar `MAX_SPEED`.
5. `Gripper.grip()` + feedback de posição.
6. `Arm.set_height(FRONT_PLATE)` para transporte.
- Sequência deve ser encapsulada em `manipulation/pick_and_place.py` com timers dependentes do `time_step`.

## 8. Estratégia de Testes e Validação
- **Unitários:** validar módulos de percepção e funções fuzzy (membership e defuzzificação) com dados sintéticos.
- **Simulação:** executar cenários com seeds diferentes (Supervisor aceita argumentos para alterar faixa ou quantidade de objetos).
- **Regressão:** comparar logs de `front_distance`, `heading_error`, `cube_alignment` vs. comandos aplicados.
- **Stress tests:** aumentar `n_objects` e verificar se colisões com `WoodenBox` permanecem nulas.
- **Demonstração final:** script que percorre pipeline completo 3× (uma por cor).

## 9. Padrões de Código
- Nomear arquivos com função explícita (`arm_service.py`, não `service.py`).
- Seguir estilo Python 3.10+, tipagem gradual (`from __future__ import annotations`).
- Domínios separados por pacotes; evitar dependências cruzadas diretas (usar interfaces/protocolos).
- Logar estados relevantes (sem excesso) usando wrapper leve (`infrastructure/logger.py`).
- Não modificar `supervisor.py` original; qualquer lógica extra deve residir no controlador do robô.

## 10. Checklists Operacionais
1. **Comissionamento do LIDAR:** verificar `enable`, `getSamplingPeriod`, conversão polar → cartesiano.
2. **Calibração de Cor:** capturar imagens das três caixas e ajustar HSV/ANN.
3. **Validação de Regras:** varrer `front_distance` em bancada virtual e confirmar saída defuzzificada.
4. **Entrega:** gravar vídeo, coletar logs, preparar relatório verbal.

## 11. Referências
- `Final Project Requirements/Trabalhofinal.html` — requisitos do professor
- `IA_20252/worlds/IA_20252.wbt` — definição do mundo e sensores
- `IA_20252/controllers/youbot/{base.py, arm.py, gripper.py}` — APIs do robô
- `IA_20252/controllers/supervisor/supervisor.py` — spawn de cubos (não modificar)
- `docs/main_plan.md` — plano de implementação passo a passo
- `docs/TESTES.md` — roteiro de validação no Webots
- DataCamp — *Fuzzy Logic in AI* (https://www.datacamp.com/pt/tutorial/fuzzy-logic-in-ai)
- Cyberbotics Docs — Tutorials, Lidar, Solid, Object Factory
- GitHub `cyberbotics/webots` (branch `released`)
- Ultralytics YOLOv8 (https://docs.ultralytics.com/)
- Open3D ICP (http://www.open3d.org/)
