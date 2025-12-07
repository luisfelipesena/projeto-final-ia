# YouBot Cube Collector - Projeto Final IA (MATA64)

Sistema autônomo de coleta e classificação de cubos coloridos usando robô KUKA YouBot no simulador Webots.

**Universidade Federal da Bahia** | Semestre 2025.2 | Prof. Luciano Oliveira

## Objetivo

Robô YouBot coleta **15 cubos** (verde, azul, vermelho) distribuídos aleatoriamente na arena e deposita cada um na caixa de cor correspondente, navegando sem GPS e evitando obstáculos.

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        YouBot Controller                        │
├─────────────────────────────────────────────────────────────────┤
│  Sensores              │  Processamento         │  Atuadores   │
│  ├─ LiDAR (180°)       │  ├─ Lógica Fuzzy       │  ├─ Base     │
│  ├─ LiDAR_low (90°)    │  ├─ CNN (cor) [WIP]    │  ├─ Arm      │
│  └─ Camera RGB 128x128 │  └─ OccupancyGrid      │  └─ Gripper  │
├─────────────────────────────────────────────────────────────────┤
│  Navegação: Odometria via encoders (sem GPS)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Requisitos Obrigatórios

| Requisito | Implementação |
|-----------|---------------|
| **Rede Neural (RNA/CNN)** | Classificador de cor (MobileNetV3 → ONNX) - [Em desenvolvimento](docs/rna_plan.md) |
| **Lógica Fuzzy** | `FuzzyNavigator` - controle de velocidade e desvio de obstáculos |
| **Sem GPS** | Odometria por encoders das rodas (cinemática Mecanum) |

## Estrutura do Projeto

```
IA_20252/
├── controllers/youbot/
│   ├── youbot.py      # Controller principal
│   ├── base.py        # Controle base omnidirecional + odometria
│   ├── arm.py         # Braço 5-DOF com IK
│   └── gripper.py     # Garra paralela
├── worlds/
│   └── IA_20252.wbt   # Mundo Webots (arena 7x4m)
└── textures/          # Texturas do piso
docs/
├── rna_plan.md        # Plano da rede neural
└── Final Project.pdf  # Especificação do projeto
```

## Componentes Implementados

### 1. Lógica Fuzzy (`FuzzyNavigator`)

Controlador fuzzy para navegação reativa com 7 regras principais:

| Variável Fuzzy | Conjuntos | Uso |
|----------------|-----------|-----|
| `obs_front/left/right` | muito_perto, perto, longe | Distância obstáculos (LiDAR + conhecidos) |
| `target_distance` | perto, longe | Distância ao alvo |
| `target_angle` | pequeno, médio, grande | Ângulo para alvo |

**Saídas:** `vx` (frontal), `vy` (strafe), `omega` (rotação)

Regras principais:
- R1: Se obs_front muito_perto → recuar + girar para lado livre
- R2: Se obs_front perto → reduzir velocidade
- R3-R6: Controle proporcional baseado em distância/ângulo ao alvo
- R7: Ajuste rotação por obstáculos laterais

### 2. Odometria (Mecanum Wheels)

Cinemática inversa para rodas Mecanum sem GPS:

```python
# Velocidades das rodas → velocidade do chassi
vx = (R/4) * (ω1 + ω2 + ω3 + ω4)
vy = (R/4) * (-ω1 + ω2 + ω3 - ω4)
ω  = (R/4(Lx+Ly)) * (-ω1 + ω2 - ω3 + ω4)
```

- `R = 0.05m` (raio roda)
- `Lx = 0.228m`, `Ly = 0.158m` (distâncias COM-rodas)

### 3. Visão Computacional (Atual: HSV)

Classificação por ranges HSV:
- **Vermelho:** H ∈ [0,10] ∪ [170,180], S,V ∈ [100,255]
- **Verde:** H ∈ [35,85]
- **Azul:** H ∈ [100,130]

### 4. Máquina de Estados

```
search → approach → grasp → to_box → drop → search
   ↑___________________________________________|
```

## Arena

| Elemento | Posição | Descrição |
|----------|---------|-----------|
| Arena | Centro (-0.79, 0) | 7m × 4m |
| Caixa Verde | (0.48, 1.58) | Depósito cubos verdes |
| Caixa Azul | (0.48, -1.62) | Depósito cubos azuis |
| Caixa Vermelha | (2.31, 0.01) | Depósito cubos vermelhos |
| Obstáculos A-G | Vários | WoodenBox 0.3×0.3×0.3m |

## Sensores Configurados

| Sensor | FOV | Resolução | Range | Uso |
|--------|-----|-----------|-------|-----|
| LiDAR principal | 180° | 180 pts | 0.1-5m | Obstáculos/paredes |
| LiDAR baixo | 90° | 90 pts | 0.05-1.5m | Detecção de cubos |
| Camera RGB | - | 128×128 | - | Classificação cor |

## Execução

```bash
# 1. Abrir Webots e carregar o mundo
webots IA_20252/worlds/IA_20252.wbt

# 2. O controller inicia automaticamente
# Ou manualmente: python IA_20252/controllers/youbot/youbot.py
```

**Dependências Python:**
```bash
pip install numpy opencv-python-headless
# Futuro (RNA): pip install onnxruntime
```

## Trabalho Futuro (RNA)

Conforme [docs/rna_plan.md](docs/rna_plan.md):

1. **Dataset:** Captura ROIs 64×64 da câmera durante simulação
2. **Modelo:** MobileNetV3-Small fine-tuned (3 classes)
3. **Exportação:** PyTorch → ONNX
4. **Inferência:** onnxruntime no controller (< 5ms/frame)

## Referências

- [Webots Documentation](https://cyberbotics.com/doc/guide/index)
- [Webots DeepWiki](https://deepwiki.com/cyberbotics/webots) - Robot Control APIs, LIDAR
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [Webots Python IDE Setup](https://cyberbotics.com/doc/guide/using-your-ide?tab-language=python)
- [AdaBoost - Adaptive Boosting](https://pedroazambuja.medium.com/adaboost-adaptive-boosting-dbbec150fced)
- [Webots LIDAR Reference](https://cyberbotics.com/doc/reference/lidar)
- Cândido, R.P. et al. - "Refinamento de modelos de navegação de robôs autônomos através da calibração do sistema de odometria" (UMBMark method)
- Borenstein, J. and Feng, L. (1994) - "UMBMark: A method for measuring, comparing, and correcting dead-reckoning errors in mobile robots"

## Licença

Projeto acadêmico - UFBA 2025.2
