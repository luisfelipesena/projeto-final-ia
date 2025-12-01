# YouBot MATA64 - Project Rules

## Projeto
- **Disciplina**: MATA64 - Inteligência Artificial - UFBA
- **Objetivo**: Robô YouBot coleta 15 cubos coloridos e deposita nas caixas correspondentes
- **Requisitos**: RNA + Lógica Fuzzy, sem GPS

## Arquitetura
```
src/
├── perception/     # Sensores: CubeDetector, LidarProcessor, LidarMLP
├── control/        # Decisão: StateMachine, FuzzyNavigator, FuzzyManipulator
├── actuators/      # Atuadores: BaseController, ArmController, GripperController
└── utils/          # Config e Logger
```

## Sensores
- **LIDAR**: 512 pontos, range 0.01-5.0m, FOV 360°
- **Camera**: 128x128 RGB, FOV ~57°
- **Gripper**: PositionSensor para detecção de objeto

## Atuadores
- **Base**: 4 rodas Mecanum, vmax=0.3m/s
- **Arm**: 5-DOF, usar presets + IK validado
- **Gripper**: MIN_POS=0.0, MAX_POS=0.025

## Limites do Braço (CRÍTICO)
```python
ARM_LIMITS = {
    'arm1': (-2.949, 2.949),    # Base rotation
    'arm2': (-1.13, 1.57),      # Shoulder - MAIS RESTRITIVO
    'arm3': (-2.635, 2.548),    # Elbow
    'arm4': (-1.78, 1.78),      # Wrist pitch
    'arm5': (-2.949, 2.949),    # Wrist roll
}
```

## Cubos e Caixas
- **Cubos**: 3cm x 3cm x 3cm, cores: verde/azul/vermelho
- **GREEN box**: (0.48, 1.58)
- **BLUE box**: (0.48, -1.62)
- **RED box**: (2.31, 0.01)

## Regras de Implementação

### OBRIGATÓRIO
1. SEMPRE validar IK antes de aplicar (respeitar ARM_LIMITS)
2. SEMPRE verificar `has_object()` após `grip()`
3. NUNCA usar GPS - apenas LIDAR + Camera
4. Fuzzy para TODAS decisões de navegação
5. RNA para classificação de obstáculos LIDAR
6. Timeouts em TODAS operações

### Grasp Sequence
1. Approach até 15cm
2. Align (centralizar cubo, |angle| < 5°)
3. Lower arm (FRONT_FLOOR)
4. Forward 5cm
5. Grip + Verify (has_object)
6. Lift (FRONT_PLATE)

### State Machine
```
IDLE → SEARCHING → APPROACHING → ALIGNING → GRASPING →
VERIFYING → TRANSPORTING → DEPOSITING → (loop)
```

## HSV Ranges (Webots)
```python
HSV_RANGES = {
    'red':   [(0, 100, 100), (10, 255, 255)],    # + (170, 100, 100), (180, 255, 255)
    'green': [(35, 100, 100), (85, 255, 255)],
    'blue':  [(100, 100, 100), (130, 255, 255)],
}
```

## Debug
- Screenshots em cada estágio do grasp
- Logs com timestamps: `[MODULE] message`
- Salvar LIDAR data para análise

## Referências Teóricas (para vídeo)
- Zadeh (1965): Fuzzy Sets
- Mamdani & Assilian (1975): Fuzzy Logic Controller
- Habermann et al. (2013): MLP para LIDAR
- Saffiotti (1997): Fuzzy Navigation
