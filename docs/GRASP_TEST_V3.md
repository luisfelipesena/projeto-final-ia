# YouBot Grasp Test V3 - Angular Approach Validation

## Overview
V3 valida grasp em ângulos diferentes (30°, 60°) usando **rotação proporcional** durante approach.

## Test Results

| Ângulo | Status | Finger Position | Notas |
|--------|--------|-----------------|-------|
| 0° | ✅ SUCCESS | ~0.0032 | Baseline (V2) |
| +30° | ✅ SUCCESS | 0.00068 | Cubo à esquerda, rotação CCW |
| -30° | 🔄 Pending | - | Cubo à direita |
| +60° | 🔄 Pending | - | |
| -60° | 🔄 Pending | - | |

## Problemas Identificados e Soluções

### Problema 1: Deposit Box vs Cubo
**Sintoma**: Robô detectava deposit box verde (grande, distante) em vez do cubo verde.

**Causa**: Filtro de tamanho muito permissivo + sem filtro de distância.

**Solução**:
```python
MAX_INITIAL_DISTANCE = 0.7  # Deposit boxes estão a >1.5m
MAX_SCAN_SIZE = 25          # Cubos a 0.5m aparecem ~19-20px
```

### Problema 2: Forward Approach Insuficiente
**Sintoma**: Gripper fechava antes de alcançar o cubo.

**Evolução**:
| Tentativa | Distância | Resultado |
|-----------|-----------|-----------|
| 1 | 6cm | Muito curto |
| 2 | 10cm | Cubo escapou |
| 3 | 12cm | Quase pegou |
| 4 | **13cm** | ✅ SUCCESS |

**Solução Final**:
```python
self.base.move(0.05, 0, 0)   # 5cm/s
self.wait_seconds(2.6)        # 2.6s = 13cm
```

## Matemática das Distâncias

### Geometria do Setup
```
Robot center: (-1.5, 0.0)
Cube 30°:     (-1.0, 0.29)  → atan2(0.29, 0.5) ≈ 30°

Distância inicial: sqrt(0.5² + 0.29²) = 0.578m ≈ 58cm
```

### Arm Reach (FRONT_FLOOR)
```
Gripper tip from robot center: ~25cm (forward)
```

### Cálculo do Forward Approach
```
Distância inicial:     ~58cm (em ângulo)
Após approach angular: ~38cm (agora alinhado)
Arm reach:             ~25cm
Forward necessário:    38 - 25 = 13cm ✓
```

### Por que 13cm funciona:
1. Robot alinha com cubo durante APPROACH (rotação)
2. GRASP_READY_SIZE=20px → cubo a ~18-20cm do gripper
3. Arm reach FRONT_FLOOR: ~25cm do centro do robô
4. Gripper tip: ~5cm além do arm base
5. 13cm forward coloca gripper exatamente no cubo

## Parâmetros Validados (V3)

```python
# Constantes de detecção
MIN_CUBE_SIZE = 5           # Filtrar ruído
MAX_CUBE_SIZE = 22          # Filtrar obstáculos grandes
MAX_SCAN_SIZE = 25          # Para fase de scan
MAX_INITIAL_DISTANCE = 0.7  # Filtrar deposit boxes (>1.5m)
GRASP_READY_SIZE = 20       # px - iniciar grasp
GRASP_READY_ANGLE = 2.0     # graus - alinhamento necessário

# Approach
APPROACH_SPEED = 0.06       # m/s durante approach fino

# Grasp sequence
FORWARD_SPEED = 0.05        # m/s (5cm/s)
FORWARD_TIME = 2.6          # s (= 13cm)
OBJECT_THRESHOLD = 0.0003   # finger_pos > threshold = objeto
```

## Grasp Sequence Timing (V3)

| Step | Ação | Duração |
|------|------|---------|
| 0 | Backup se muito perto | 1.3s (condicional) |
| 1 | Open gripper | 1.0s |
| 2 | Reset arm | 1.5s |
| 3 | Lower to FRONT_FLOOR | 2.5s |
| 4 | Forward 13cm | 2.6s |
| 5 | Close gripper | 2.0s |
| 6 | Check object | - |
| 7 | Lift to FRONT_PLATE | 2.0s |
| **Total** | | **~12.6s** |

## Rotation Approach Logic

### Convenção de Sinais
```
Mecanum wheels: POSITIVE omega = rotate CW (clockwise)

Cubo à DIREITA (angle > 0) → rotate RIGHT → omega POSITIVO
Cubo à ESQUERDA (angle < 0) → rotate LEFT → omega NEGATIVO

Portanto: omega = angle * k (MESMO SINAL)
```

### Implementação
```python
omega = angle * 0.02  # Proporcional ao ângulo
omega = max(-0.4, min(0.4, omega))  # Clamp

# Forward speed baseado no alinhamento
if abs(angle) > 20:
    vx = 0.03  # Slow
elif abs(angle) > 10:
    vx = 0.05  # Medium
else:
    vx = 0.06  # Fast when aligned
    if abs(angle) < 1.5:
        omega = 0  # Stop rotation
```

## Filtros de Detecção

### Por que deposit boxes eram detectadas:
1. **Cor**: Deposit boxes são verde/azul/vermelho (mesmas cores dos cubos)
2. **Tamanho**: A distância faz parecerem pequenas (~19px)
3. **Posição Y**: Aparecem na parte inferior da imagem

### Solução: Filtro de Distância
```python
# Distância estimada via tamanho aparente
# Cubo 3cm a 50cm = ~19px
# Deposit box a 150cm com tamanho ~19px = FALSO POSITIVO

if det.distance > MAX_INITIAL_DISTANCE:  # 0.7m
    continue  # Ignorar - muito longe para ser cubo
```

## Logs de Sucesso (+30°)

```
[SCAN] Valid GREEN: dist=0.58m, size=19px, angle=-30.1°
[SCAN] Selected NEAREST: dist=0.58m
[APPROACH] Size: 19px/20, Angle: -30.1°/±2.0°
[MOVE] angle=-30.1°, vx=0.03, omega=-0.40
...
[MOVE] angle=-1.8°, vx=0.06, omega=0.00
[APPROACH] *** READY TO GRASP ***
           Size: 20px (threshold: 20)
           Angle: -1.8° (threshold: ±2.0°)
[GRASP 4] Forward approach (13cm)...
[GRASP 5] Closing gripper...
         Finger BEFORE: 0.0250
         Finger AFTER: 0.0007
[GRASP 6] Checking object...
         has_object(): True
RESULT: *** SUCCESS ***
```

## Arquivos Modificados

| Arquivo | Mudanças |
|---------|----------|
| `youbot_grasp_test_v3.py` | Filtro distância, forward 13cm, rotação proporcional |
| `supervisor_test_v3.py` | Teste individual por ângulo |

## Próximos Passos

1. ✅ Testar +30° - **VALIDADO**
2. [ ] Testar -30° (cubo à direita)
3. [ ] Testar +60°
4. [ ] Testar -60°
5. [ ] Integrar com estado autônomo completo

## Referências

- `docs/GRASP_TEST.md` - Mecânica original do grasp
- `docs/GRASP_TEST_V2.md` - Integração com detecção
- `src/perception/cube_detector.py` - HSV detection
