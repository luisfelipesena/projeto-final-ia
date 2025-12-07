# TESTES — Validação do YouBot Fuzzy Controller

Roteiro passo a passo para validar o controlador no Webots.

## 1. Pré-requisitos

### 1.1 Ambiente

| Item | Requisito |
|------|-----------|
| Webots | R2025a (ou R2023b com warnings) |
| Python | 3.10+ (`/usr/bin/python3` ou configurar em Preferences) |
| Comando Python | Webots → Preferences → Python command: `/usr/bin/python3` |

### 1.2 Dependências Python (opcionais)

```bash
# No terminal, na raiz do projeto:
pip install numpy opencv-python

# Para ML completo (opcional):
pip install ultralytics open3d joblib scikit-learn
```

> **Nota:** O controlador funciona sem estas dependências, usando fallbacks.

### 1.3 Modelos ML (opcional)

Se quiser usar detecção neural:
```
IA_20252/controllers/youbot_fuzzy/models/
├── yolov8n-cubes.pt      # Modelo YOLO treinado
└── adaboost_color.pkl    # Classificador AdaBoost
```

### 1.4 Treinamento de Modelos ML

**Passo 1: Gerar dataset de imagens (dentro do Webots)**
```bash
cd IA_20252/controllers/youbot_fuzzy
# Definir diretório de saída
export YOUBOT_DATASET_DIR=datasets/cubes/train
# Executar no Webots como controlador supervisor temporário
python tools/run_dataset_capture.py
```

**Passo 2: Treinar classificador AdaBoost**
```bash
cd IA_20252/controllers/youbot_fuzzy
python tools/train_adaboost.py --dataset datasets/cubes/train --output models/adaboost_color.pkl
```

**Passo 3: Treinar YOLO (opcional)**
```bash
# Requer ultralytics instalado
cd IA_20252/controllers/youbot_fuzzy
yolo train model=yolov8n.pt data=datasets/cubes/data.yaml epochs=50 imgsz=128
# Copiar melhor modelo para:
cp runs/detect/train/weights/best.pt models/yolov8n-cubes.pt
```

## 2. Preparação do Ambiente

### 2.1 Abrir o Webots

1. Abrir Webots R2025a
2. `File > Open World...`
3. Selecionar: `IA_20252/worlds/IA_20252.wbt`

### 2.2 Verificar Estrutura do Mundo

Na árvore de cena, confirmar:

- [x] `RectangleArena` (7x4 m)
- [x] `PlasticFruitBox` GREEN, BLUE, RED
- [x] `WoodenBox` A, B, C, D, E, F, G
- [x] `Youbot` com:
  - `lidar_low` (no bodySlot)
  - `lidar_high` (no bodySlot)
  - `Camera` (no bodySlot)
- [x] `Robot (Supervisor)` separado

### 2.3 Configurar Controlador

1. Selecionar nó `Youbot` na árvore
2. Em `controller`, verificar se está `youbot_fuzzy`
3. Se não estiver:
   - Clicar no campo `controller`
   - Selecionar `youbot_fuzzy`
   - Salvar o mundo (`Ctrl+S`)

### 2.4 Configurar Supervisor (spawn de cubos)

1. Selecionar nó `Robot (Supervisor)`
2. Em `controller`, deve estar `supervisor`
3. **NÃO ALTERAR** - responsável por spawnar os 15 cubos

## 3. Execução e Validação

### 3.1 Iniciar Simulação

1. Definir velocidade: `1.00x` (realtime)
2. Clicar em **Run** (play)
3. Observar console para logs

### 3.2 Verificar Spawn dos Cubos

**Esperado no console:**
```
INFO: supervisor: Starting controller: /usr/bin/python3 -u supervisor.py
Spawn complete. The supervisor has spawned 15/15 objects (0 failed).
INFO: 'supervisor' controller exited successfully.
```

**Validar visualmente:**
- 15 cubos distribuídos na área de spawn (x: -3 a 1.75, y: -1 a 1)
- Cores: verde, azul, vermelho (aleatório)

### 3.3 Verificar Controlador YouBot

**Esperado no console:**
```
INFO: youbot: Starting controller: /usr/bin/python3 -u youbot_fuzzy.py
```

**Se houver erro de importação:**
```bash
# Verificar se youbot_fuzzy.py adiciona sys.path corretamente
# O arquivo deve ter no início:
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
```

## 4. Validação por Componente

### 4.1 Dual-LIDAR

**Teste 1: Verificar presença dos sensores**
```
# Nos logs, procurar por:
# (Se ENABLE_LOGGING=True em config.py)
lidar_high enabled
lidar_low enabled
```

**Teste 2: Visualização LIDAR**
1. Menu: `View > Optional Rendering > Lidar Rays`
2. Deve mostrar raios do `lidar_high` (360°) e `lidar_low` (180° frontal)

**Teste 3: Calibração dos Setores LIDAR (Debug)**
```
# Nos logs, output de debug mostra distâncias em índices-chave:
LIDAR[F=0]: 0:X.XX | 10:X.XX | 90:X.XX | 180:X.XX | 270:X.XX | 350:X.XX
```

**Mapeamento LIDAR (rotation 0, sem rotação):**
- Índice 0 = TRÁS do robô (~0.96m para arena aberta)
- Índice 90 = ESQUERDA (BLOQUEADO pelo corpo! ~0.11m)
- Índice 180 = FRENTE do robô (direção do braço/câmera, ~1.1m)
- Índice 270 = DIREITA (~1.0m)
- Dead zones: (70-110) lado esquerdo bloqueado pelo corpo

**Como verificar mapeamento:**
1. No console, observar linha `LIDAR[F=180]: 180:X.XX | 90:X.XX | ...`
2. Índice 180 deve mostrar distância maior quando frente está livre
3. Índice 90 sempre mostra ~0.11m (corpo bloqueia)
4. Raios no Webots devem apontar para fora do robô

**Configuração atual (config.py):**
```python
front_sector=(170, 190)  # Em torno de 180 = frente
left_sector=(120, 150)   # Front-left diagonal (evita índice 90 bloqueado)
right_sector=(210, 240)  # Front-right diagonal
LIDAR_DEAD_ZONES = [(70, 110)]  # Lado esquerdo bloqueado pelo corpo
```

**Teste 4: Valores de distância**
```
# Nos logs, verificar:
front=X.XX, left=X.XX, right=X.XX, density=X.XX
```
- `front` deve diminuir ao aproximar de obstáculo na frente
- `density` deve aumentar perto de paredes

### 4.2 Câmera e Percepção

**Teste 1: Visualização**
1. Menu: `View > Optional Rendering > Camera...`
2. Selecionar a câmera do Youbot

**Teste 2: Detecção de Cubos**
```
# Nos logs, quando cubo visível:
cube=GREEN conf=0.XX bearing=X.XX alignment=X.XX
# ou
cube=BLUE conf=0.XX ...
# ou
cube=RED conf=0.XX ...
```

**Se não detectar:**
- Verificar se há luz suficiente no mundo
- Ajustar `HSV_RANGES` em `config.py`
- Se usando YOLO: verificar se modelo existe em `models/`

### 4.3 Odometria e Localização

**Teste 1: Movimento da base**
```
# Nos logs:
pose=(X.XX, Y.XX, θ=X.XX)
```
- Valores devem mudar conforme robô se move
- `x` inicial ~-4.0, `y` inicial ~0

**Teste 2: Correção ICP**
```
# Quando deslocamento > 0.15m:
ICP correction applied
```

### 4.4 Fuzzy Planner

**Teste 1: Comandos de movimento**
```
# Nos logs:
vx=X.XX, vy=X.XX, omega=X.XX
```
- `vx` esperado: 0.08-0.15 quando caminho livre (era 0.01-0.03 antes do fix)
- `omega` esperado: <0.3 quando explorando (não deve dominar o movimento)

**Teste 2: Comportamento de evasão**
1. Posicionar robô próximo a obstáculo (<0.25m = DANGER_ZONE)
2. Observar se `omega` muda de sinal (rotação para escapar)
3. `vx` deve ficar negativo quando <0.18m (reverso)

### 4.5 Máquina de Estados

**Verificar transições de fase:**
```
phase=SCAN_GRID    # Explorando, buscando cubos
phase=PICK         # Cubo detectado, aproximando
phase=DELIVER      # Cubo coletado, indo para caixa
phase=RETURN       # Depositando cubo
```

### 4.6 Manipulação (Braço/Garra)

**Sequência de coleta (conforme GRASP_TEST.md):**
1. `PICK:OPEN_GRIPPER` - Garra abre (`RELEASE`, 1.0s)
2. `PICK:RESET_ARM` - Braço posição segura (`RESET`, 1.5s)
3. `PICK:LOWER_ARM` - Braço desce (`FLOOR`, 2.5s)
4. `PICK:APPROACH` - Robô avança (0.05 m/s, 2.0s)
5. `PICK:GRIP` - Garra fecha (`GRIP`, 1.5s)
6. `PICK:LIFT` - Braço sobe (`PLATE`, 2.0s)
7. `PICK:DONE` - Transição para `DELIVER`
8. `load_state=True` via arm_service.is_gripping

**Tempo total esperado:** ~11 segundos

**Sequência de depósito:**
1. Braço posiciona (`PLATE`)
2. Garra abre (`RELEASE`)
3. `load_state=False` nos logs

## 5. Cenários de Teste

### 5.1 Teste Básico (1 cubo)

1. Iniciar simulação
2. Aguardar robô detectar um cubo
3. Observar sequência completa:
   - Aproximação
   - Coleta
   - Navegação até caixa
   - Depósito
4. **Sucesso:** cubo dentro da caixa correta

### 5.2 Teste Completo (15 cubos)

1. Deixar simulação rodar
2. Monitorar contagem de cubos coletados
3. Tempo esperado: < 10 minutos
4. **Sucesso:** todos os 15 cubos nas caixas corretas

### 5.3 Teste de Colisão

1. Observar comportamento perto de `WoodenBox`
2. Robô deve desviar, não colidir
3. **Sucesso:** zero colisões durante execução

### 5.4 Teste de Robustez

1. Alterar seed do supervisor (se possível)
2. Executar múltiplas vezes
3. **Sucesso:** comportamento consistente

## 6. Troubleshooting

### 6.1 Erro: Python não encontrado

```
WARNING: Python was not found
```

**Solução:**
1. Webots → Preferences
2. Python command: `/usr/bin/python3`
3. Reiniciar Webots

### 6.2 Erro: ImportError

```
ImportError: attempted relative import with no known parent package
```

**Solução:**
Verificar `youbot_fuzzy.py`:
```python
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
from app import main
```

### 6.3 LIDAR não detecta cubos

**Possíveis causas:**
1. `tiltAngle` incorreto no `lidar_low`
2. Cubo fora do range (0.03-2.5m)
3. Thresholds muito restritivos

**Solução:**
- Verificar `IA_20252.wbt`: `lidar_low` deve ter `tiltAngle -0.05`
- Ajustar `CUBE_HEIGHT_DIFFERENCE_THRESHOLD` em `config.py`

### 6.4 Câmera não classifica cor

**Possíveis causas:**
1. Iluminação inadequada
2. HSV ranges incorretos
3. Modelo ML não encontrado

**Solução:**
- Verificar luz no mundo (Tutorial 3 Webots)
- Ajustar `HSV_RANGES` em `config.py`
- Verificar se fallback para heurísticas está funcionando

### 6.5 Robô não se move

**Possíveis causas:**
1. `Base` não inicializada
2. `fuzzy_planner` retornando zeros
3. Erro silencioso em módulos

**Solução:**
- Ativar `ENABLE_LOGGING = True` em `config.py`
- Verificar logs para identificar módulo problemático

## 7. Preparação para Entrega

### 7.1 Gravar Vídeo

1. Webots → `Tools > Movie > Start Recording`
2. Executar cenário completo
3. Parar gravação
4. Editar se necessário (≤ 15 min)

### 7.2 Salvar Logs

```bash
# Copiar output do console para arquivo
# ou redirecionar:
python youbot_fuzzy.py > logs/execution.log 2>&1
```

### 7.3 Checklist Final

- [ ] Vídeo mostra spawn dos 15 cubos
- [ ] Vídeo mostra pelo menos 1 ciclo completo por cor
- [ ] Explicação conceitual (fuzzy, LIDAR, visão)
- [ ] Sem mostrar código no vídeo
- [ ] Código funciona sem GPS
- [ ] Supervisor não foi modificado

## 8. Debug Mode e Output Esperado

### 8.1 Ativar Debug Logging

Em `config.py`:
```python
ENABLE_LOGGING = True
LOG_INTERVAL_STEPS = 10  # Log a cada 10 steps (reduzir para mais detalhes)
```

### 8.2 Output Esperado (Comportamento Correto)

**Inicialização:**
```
INFO: youbot: Starting controller: /usr/bin/python3 -u youbot_fuzzy.py
```

**Durante exploração (SCAN_GRID):**
```
LIDAR[F=180]: 180:1.10 | 170:1.05 | 190:1.08 | 90:0.11 | 270:1.00 | 0:0.96
FUZZY[EXPLORE]: f=1.02 l=1.15 r=1.05 → vx=0.12 ω=0.05
LIDAR: f=1.02 l=1.15 r=1.05 | MAP_ROBOT: f=1.00 l=1.20 r=1.10 obs=0.80
phase=SCAN_GRID load=False cube=- conf=0.00 vx=0.12 vy=0.00 omega=0.05
```

**Quando detecta cubo:**
```
CAMERA_DEBUG: mean_BGR=(35.2,180.3,42.1)
phase=PICK load=False cube=GREEN conf=0.65 bear=-0.12 align=0.03
front=0.85 left=2.10 right=1.80 density=0.01
vx=0.08 vy=-0.02 omega=0.18
```

**Durante coleta:**
```
phase=PICK load=False lift=FLOOR gripper=GRIP
```

**Carregando cubo:**
```
phase=DELIVER load=True cube=GREEN goal=(0.48, 1.58)
vx=0.10 vy=0.00 omega=0.25
```

### 8.3 Sinais de Problema

**LIDAR com setores incorretos (robô gira sem parar):**
```
# Se front_sector usa índice 90 (bloqueado pelo corpo):
LIDAR[F=180]: 90:0.11 | 180:1.10 ...  # Índice 90 sempre 0.11m (corpo)!
FUZZY[EXPLORE]: f=0.11 ... → vx=-0.10 ω=0.50  # Escape infinito
```
**Solução:** Verificar config.py usa front_sector=(170,190) e left_sector evita índice 90

**Câmera não detecta cores:**
```
CAMERA_DEBUG: mean_BGR=(128.0,128.0,128.0)  # Cinza = sem cor dominante
cube=- conf=0.00
```

**Robô preso em escape loop:**
```
# Mesmos valores repetindo, escape_counter aumentando
vx=-0.15 vy=0.10 omega=0.80
vx=-0.15 vy=-0.10 omega=-0.80
# Alternando indefinidamente
```

---

**Referências:** `CLAUDE.md`, `docs/main_plan.md`
