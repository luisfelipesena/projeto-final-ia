# YouBot Cube Collection — Resumo Técnico (focado em implementação)

## Visão geral
- Objetivo: coletar 15 cubos (red/green/blue) e depositar nas caixas da cor correta no mundo `IA_20252.wbt`.
- Plataforma: Webots (controlador Python, nó Supervisor), KUKA YouBot com base mecanum, braço 5-DOF e garra paralela.
- Sensores principais: câmera RGB com Recognition ativo, LIDAR planar, 6 sensores infravermelhos de distância (frente/traseira).
- Estados principais (FSM): `search` → `approach` → `grasp` → `to_box` → `drop` → volta a `search`.
- Bugs conhecidos (em aberto): (1) risco de travar lateralmente ao contornar obstáculos largos; (2) ocasionalmente o cubo não é depositado no box (overshoot lateral/longitudinal).

## Configuração do mundo e hardware
- Arena: `RectangleArena` 7.0 m × 4.0 m, centro em (-0.79, 0.0), parede 0.30 m. Robô spawna em x≈-4.0 m.
- Caixas (PlasticFruitBox): green (0.48, 1.58), blue (0.48, -1.62), red (2.31, 0.01, rotacionada 90°).
- Obstáculos fixos (WoodenBox, raio seguro ~0.25 m): A-G nas posições do mundo e em `KNOWN_OBSTACLES` do controlador.
- Sensores no slot do robô (vide `IA_20252/worlds/IA_20252.wbt`):
  - LIDAR: `horizontalResolution=180`, `fieldOfView=π` (180°), `minRange=0.1 m`, `maxRange=5.0 m`, `numberOfLayers=1`, `pointCloud` habilitado.
  - Câmera: 128×128, offset (0.27, 0, -0.06), Recognition ativado com `maxRange=3 m`.
  - IR distance sensors (lookup linear 0→0 m, 50→0.05 m, 1000→1.0 m): traseiros `ds_rear`, `ds_rear_left` (135°), `ds_rear_right` (-135°); frontais `ds_front` (0°), `ds_front_left` (~23°), `ds_front_right` (~-23°).
- Base mecanum: parâmetros `WHEEL_RADIUS=0.05`, `LX=0.228`, `LY=0.158`.

## Pilha de percepção
### Câmera + Recognition API (Webots)
- Ativação: `camera.enable()` + `camera.recognitionEnable(time_step)`.
- API nativa (Webots): `getRecognitionObjects()` retorna lista de objetos com `getPosition()` (coordenadas relativas ao frame da câmera; x alinhado ao eixo óptico, y lateral), `getColors()` (RGB normalizado 0–1, média da textura), `getSize()`, `getModel()`, `getId()`. A detecção é feita internamente pelo Webots no host, não pelo nosso classificador.
- Uso no código (`_process_recognition`):
  - Filtra por cor opcional (`lock_color`) e prioriza ângulo opcional (`lock_angle`) para manter tracking do mesmo cubo.
  - Distância = √(x² + y²); ângulo = atan2(y, x); limite de uso 2.5 m.
  - Score prioriza menor distância e forte viés para o ângulo bloqueado (evita trocar de cubo).
  - Saída alimenta os modos `search/approach` e o alinhamento fino no grasp.

### Classificação de cor
- RNA: MobileNetV3-Small fine-tuned (cabeça 256→3), entrada 64×64, normalização ImageNet, saída softmax (red/green/blue). Formato ONNX carregado por `onnxruntime`.
- Fallback: heurística HSV/RGB (`color_from_rgb` e `_fallback_hsv`) quando `confidence < 0.5` ou modelo indisponível. Decisão final é feita na câmera (usando `getColors()` do Recognition).

### LIDAR
- Varredura planar 180 amostras, FOV 180°, alcance 0.1–5.0 m.
- Pré-processamento: ignora `inf/NaN/≤0`; downsample a cada 2 pontos para `points` (r, θ). Janelas:
  - front: -20°..20°, left: 20°..90°, right: -90°..-20° (usa mínimo da janela).
- Usa para: (1) detecção frontal/lateral (`obs_front/left/right`), (2) raycast no grid, (3) gatilho de aproximação ao box (`lidar_front` < 0.25 m).

### Sensores IR (distance sensors)
- Conversão `raw_to_meters`: `max(0.05, min(2.0, raw/1000.0))`.
- Traseiros: proteção de rotação e ré (mínimo 0.35 m atrás, 0.40 m nas diagonais) em `check_rear_clearance`.
- Frontais: redundância de proximidade no `approach` (parada/strafe se <0.20 m).

## Mapeamento e planejamento
- Grade (`OccupancyGrid`): célula 0.12 m, 58×33, bounds do mundo; estados UNKNOWN/FREE/OBSTACLE/BOX/CUBE.
- Semeadura estática: bordas como OBSTACLE; inflar obstáculos fixos com `wall_inflate=0.30` (meia-largura do robô + margem); caixas marcadas como BOX.
- Raycasting (Bresenham): cada ponto LIDAR faz `grid.raycast(origin, hit, OBSTACLE, free=FREE)`, marcando caminho livre e célula atingida como obstáculo; qualquer mudança marca `_path_dirty`.
- A* (`plan_path`): vizinhos 4-conexos, custo unitário, heurística Manhattan; retorna waypoints em mundo; replanning sempre que `_path_dirty` ou novo objetivo.
- Waypoints: consumidos quando `dist < 0.25 m`; se vazio, navega direto para o goal.

## Localização e odometria
- Encoders das 4 rodas → `compute_odometry`:
  - `vx = R/4 * Σωi`
  - `vy = R/4 * (-ω1 + ω2 + ω3 - ω4)`
  - `omega = R / (4*(LX+LY)) * (-ω1 + ω2 - ω3 + ω4)`
- Integração (`_integrate_pose`): transforma para mundo com yaw, valida NaN, wrap de ângulo.
- Correção de drift via ground truth (`_get_ground_truth_pose`):
  - `search/approach`: a cada 2.0 s ou NaN.
  - `to_box`: a cada 0.5 s (alta precisão para depósito).
- Recuperação de pose: se NaN, restaura do ground truth imediatamente.

## Navegação e controle
### FuzzyNavigator (desvio e perseguição ao alvo)
- Memberships: `_mu_close(0–0.45)`, `_mu_very_close(0–0.25)`, `_mu_far(0.4–1.5)`, `_mu_small_angle(0–25°)`, `_mu_medium_angle(10–60°)`, `_mu_big_angle(40–90°)`.
- Regras chave:
  - Muito perto: `vx=-0.04`, strafe longe do lado mais obstruído, `omega=0`.
  - Perto: reduz velocidade frontal; mantém avanço mínimo se bem alinhado.
  - Strafe lateral = 0.10*(μ_right_close - μ_left_close).
  - Rotação proporcional ao ângulo normalizado (limite |ω|≤0.6) com leve viés de obstáculo lateral.
- Limites finais: `vx∈[-0.18,0.18]`, `vy∈[-0.14,0.14]`, `omega∈[-0.6,0.6]`.

### Padrão de busca (SEARCH)
- Lawnmower adaptativo com proteção de parede: velocidade fwd 0.10 m/s; giros ~72° (`TURN_DURATION = π/2.5 / TURN_SPEED`), turn speed 0.30 rad/s (reduz perto de parede).
- Se parede <0.25 m, prioriza strafe/avanço/recúo ao invés de girar.
- Transição para `approach` ao detectar cubo via Recognition.

### Aproximação ao cubo (APPROACH)
- Usa câmera + Recognition com lock de cor/ângulo. Alinha primeiro, avança depois.
- Gatilho de grasp: `dist < 0.32 m` e `|ang| < 10°`. Se muito perto e desalinhado, gira sem avançar.
- Segurança frontal: se sensor/LIDAR <0.20 m, recua + strafe e aborta após 0.5 s de tentativas.
- Timeout de perda de cubo: 4 s; se o cubo some perto (<0.35 m) por 0.3 s, força grasp para não perder o alvo.

### Grasp
- Sequência (estágios):
  0) Garra abre; braço RESET (1.2 s).
  1) Braço FRONT_FLOOR (2.0 s).
  2) Avanço lento 0.04 m/s por tempo calculado: `forward = clamp(dist_cam - 0.12, 0.05..0.25)`, com correção de yaw usando Recognition travado no mesmo cubo.
  3) Fecha garra (1.2 s).
  4) Sobe para FRONT_PLATE e verifica posse: `has_object(threshold=0.003)` com múltiplas amostras dos sensores dos dedos.
  5) Se sucesso → `to_box`; se falha → recuo 2 s e volta a `search`.

### Navegação até a caixa (TO_BOX)
- Goal = centro da caixa da cor; `A*` gera waypoints.
- Modo final de aproximação quando `dist_goal < 1.2 m` e `lidar_front < 0.80 m`.
- Gatilho de DROP: `lidar_front` entre 0.10–0.25 m (quase encostado).
- Desvio reativo:
  - Emergência <0.30 m: ré + rotação/strafe respeitando clearance traseiro; se preso >25 ciclos, manobra agressiva (ré 0.10, |ω|=0.6, |vy|=0.18).
  - Preventivo <0.50 m ou clearing timer: rotação/strafe priorizando lado mais livre; verifica traseiros para não colidir.
- Movimento “car-like”: sempre que possível combina avanço proporcional ao alinhamento (mínimo 0.04 m/s) com rotação limitada a |0.40|.

### Depósito (DROP)
- 0) Braço FRONT_FLOOR, garra fechada (2.0 s).
- 1) Avança 0.04 m/s por 2.5 s (~10 cm) para ficar sobre o box.
- 2) Abre garra (1.0 s).
- 3) Recuo curto 0.06 m/s por 1.0 s.
- 4) Braço RESET (1.5 s).
- 5) Recuo 0.10 m/s por 2.0 s (~20 cm).
- 6) Giro 0.4 rad/s por 1.5 s (~60°) para não redetectar o cubo.

## Reconhecimento de cor e pipeline de percepção (detalhe)
- Fonte do sinal de cor: `Camera Recognition` (média da textura do objeto) → `color_classifier.predict_from_rgb(r,g,b)` → se `confidence<0.5`, heurística RGB simples.
- Rede: MobileNetV3-Small fine-tuned, ImageNet init, ONNX. Entrada 64×64, normalização (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Saída em Python via `onnxruntime` CPU.
- Falha de modelo ou ausência de onnxruntime: loga fallback HSV e continua.

## Layout de sensores (resumo)
- Câmera: (0.27, 0, -0.06), Recognition `maxRange=3 m`.
- LIDAR: (0.28, 0, 0), FOV 180°, 180 raios, range 0.1–5 m.
- IR frontais: (0.29,0), (0.25,±0.12), todos a 0.05 m de altura, abertura 0.3 rad, 2 raios.
- IR traseiros: (-0.29,0), (-0.25,±0.15), mesmas características.

## Local de bugs conhecidos
- Travamento lateral em obstáculos largos: ocorre em `to_box` quando o LIDAR frontal fica <0.30 m e a folga traseira inibe rotação; pode entrar em loop de strafe curto + rotação limitada.
- Depósito incorreto: em alguns casos o avanço +10 cm (DROP estágio 1) não centraliza no box, levando a soltar o cubo na borda; sensibilidade alta a pequenas derivações de pose mesmo com sync de 0.5 s.

## Referências técnicas
- Webots docs: https://cyberbotics.com/doc/guide/index
- Webots Recognition API: https://cyberbotics.com/doc/reference/recognition
- Webots Camera/LIDAR: https://cyberbotics.com/doc/reference/camera, https://cyberbotics.com/doc/reference/lidar
- DeepWiki Webots (overview): https://deepwiki.com/cyberbotics/webots
- Mapeamento por grade ocupação (Elfes, 1989); A* (Hart et al., 1968); Fuzzy Sets (Zadeh, 1965); MobileNetV3 (Howard et al., 2019).
