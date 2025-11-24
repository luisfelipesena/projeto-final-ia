# Guia de Testes no Webots

Este documento descreve passo a passo como testar o sistema autônomo do YouBot no simulador Webots.

## Requisitos

- Webots R2023b ou superior instalado
- Python 3.8+ com dependências instaladas
- Arena IA_20252 disponível

## 1. Configuração Inicial

### 1.1 Instalação de Dependências

```bash
# No diretório do projeto
cd projeto-final-ia
pip install -r requirements.txt
```

### 1.2 Verificar Estrutura do Projeto

```
projeto-final-ia/
├── IA_20252/
│   ├── controllers/
│   │   ├── youbot/
│   │   │   ├── youbot.py
│   │   │   ├── base.py
│   │   │   ├── arm.py
│   │   │   └── gripper.py
│   │   └── supervisor/
│   │       └── supervisor.py  # NÃO MODIFICAR!
│   └── worlds/
│       └── IA_20252.wbt
├── src/
│   ├── perception/
│   ├── control/
│   ├── navigation/
│   └── manipulation/
└── models/  # Modelos treinados (se disponíveis)
```

---

## 2. Teste de Componentes Individuais

### 2.1 Teste da Base Móvel

**Objetivo:** Verificar que o robô se move corretamente em todas as direções.

**Passos:**

1. Abrir Webots e carregar `IA_20252/worlds/IA_20252.wbt`
2. Aguardar o supervisor spawnar os 15 cubos
3. No console Python do controlador, executar:

```python
# Teste de movimento para frente (5 segundos)
base.move(vx=0.2, vy=0.0, omega=0.0)
# Aguardar movimento...
base.reset()

# Teste de strafe para esquerda
base.move(vx=0.0, vy=0.2, omega=0.0)
# Aguardar movimento...
base.reset()

# Teste de rotação
base.move(vx=0.0, vy=0.0, omega=0.3)
# Aguardar movimento...
base.reset()
```

**Critérios de Sucesso:**
- [ ] Robô move para frente sem drift lateral significativo
- [ ] Robô faz strafe sem avançar/recuar
- [ ] Robô rotaciona no próprio eixo

### 2.2 Teste do Braço

**Objetivo:** Verificar posicionamento do braço em diferentes alturas e orientações.

**Passos:**

```python
# Testar alturas
arm.set_height(arm.FRONT_FLOOR)    # Altura do chão
# Aguardar 2s
arm.set_height(arm.FRONT_PLATE)    # Altura de prato
# Aguardar 2s
arm.set_height(arm.RESET)          # Posição inicial

# Testar orientações
arm.set_orientation(arm.LEFT)       # Virar para esquerda
# Aguardar 2s
arm.set_orientation(arm.RIGHT)      # Virar para direita
# Aguardar 2s
arm.set_orientation(arm.FRONT)      # Voltar para frente
```

**Critérios de Sucesso:**
- [ ] Braço alcança todas as posições sem colisão com base
- [ ] Movimentos são suaves e estáveis

### 2.3 Teste da Garra

**Objetivo:** Verificar abertura e fechamento da garra.

**Passos:**

```python
# Abrir garra
gripper.release()
# Aguardar 1s

# Fechar garra
gripper.grip()
# Verificar estado
print(f"Garra fechada: {gripper.is_gripping}")
```

**Critérios de Sucesso:**
- [ ] Garra abre completamente
- [ ] Garra fecha com força suficiente
- [ ] Estado `is_gripping` reflete posição correta

---

## 3. Teste de Sensores

### 3.1 Teste do LIDAR

**Objetivo:** Verificar leitura de obstáculos pelo LIDAR.

**Passos:**

1. Posicionar robô próximo a um obstáculo (caixa de madeira)
2. Executar:

```python
# Obter leitura do LIDAR
lidar_ranges = lidar.getRangeImage()
print(f"Pontos LIDAR: {len(lidar_ranges)}")
print(f"Distância mínima: {min(lidar_ranges):.2f}m")
print(f"Distância máxima: {max(lidar_ranges):.2f}m")
```

**Critérios de Sucesso:**
- [ ] LIDAR retorna 667 pontos
- [ ] Distâncias refletem obstáculos visíveis
- [ ] Range válido: 0.1m - 5.0m

### 3.2 Teste da Câmera

**Objetivo:** Verificar detecção de cubos coloridos.

**Passos:**

1. Posicionar robô de frente para um cubo colorido
2. Executar:

```python
# Capturar imagem
width = camera.getWidth()
height = camera.getHeight()
image = camera.getImage()

print(f"Resolução: {width}x{height}")
print(f"Bytes: {len(image) if image else 0}")

# Verificar detecção de cores (via CubeDetector)
from src.perception import CubeDetector
detector = CubeDetector()

import numpy as np
img_array = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
img_rgb = img_array[:, :, :3][:, :, ::-1]

detections = detector.detect(img_rgb)
for det in detections:
    print(f"Cubo {det.color}: distância={det.distance:.2f}m, ângulo={det.angle:.1f}°")
```

**Critérios de Sucesso:**
- [ ] Câmera retorna imagem 512x512
- [ ] Cubos são detectados com cor correta
- [ ] Distância estimada aproximadamente correta

---

## 4. Teste do Sistema de Controle Fuzzy

### 4.1 Teste de Inferência

**Objetivo:** Verificar que o controlador fuzzy produz saídas válidas.

**Passos:**

```python
from src.control import FuzzyController, FuzzyInputs

controller = FuzzyController({'logging': False})
controller.initialize()

# Cenário: obstáculo à frente
inputs = FuzzyInputs(
    distance_to_obstacle=0.5,
    angle_to_obstacle=0.0,
    distance_to_cube=3.0,
    angle_to_cube=0.0,
    cube_detected=False,
    holding_cube=False
)

outputs = controller.infer(inputs)
print(f"Velocidade linear: {outputs.linear_velocity:.3f} m/s")
print(f"Velocidade angular: {outputs.angular_velocity:.3f} rad/s")
print(f"Ação: {outputs.action}")
```

**Critérios de Sucesso:**
- [ ] Controlador inicializa sem erros
- [ ] Saídas estão em ranges válidos (vx: [-0.3, 0.3], omega: [-0.5, 0.5])
- [ ] Ação corresponde ao cenário

### 4.2 Teste de Desvio de Obstáculos

**Passos:**

1. Posicionar robô a 0.5m de um obstáculo
2. Executar controlador e observar comportamento

**Critérios de Sucesso:**
- [ ] Robô reduz velocidade ao se aproximar
- [ ] Robô desvia do obstáculo (gira)
- [ ] Nenhuma colisão ocorre

---

## 5. Teste de Integração Completa

### 5.1 Ciclo de Coleta de Cubo

**Objetivo:** Testar ciclo completo: buscar → aproximar → pegar → depositar.

**Passos:**

1. Resetar simulação (reiniciar mundo)
2. Aguardar spawn dos 15 cubos
3. Iniciar controlador principal:

```python
from src.main_controller import create_controller_for_webots

controller = create_controller_for_webots()
controller.initialize()
controller.run()
```

4. Observar comportamento por 5 minutos

**Critérios de Sucesso:**
- [ ] Robô navega sem colidir com obstáculos
- [ ] Robô detecta e se aproxima de cubos
- [ ] Robô pega cubos com a garra
- [ ] Robô identifica cor corretamente
- [ ] Robô navega até caixa correspondente
- [ ] Robô deposita cubo na caixa correta

### 5.2 Métricas de Performance

| Métrica | Meta | Valor Obtido |
|---------|------|--------------|
| Cubos coletados | 15/15 | |
| Identificação de cor | >95% | |
| Colisões | 0 | |
| Tempo total | <5 min | |
| FPS médio | >10 | |

---

## 6. Checklist Final para Apresentação

### Antes da Gravação

- [ ] Simulação funciona consistentemente (testar 3x)
- [ ] Nenhuma mensagem de erro no console
- [ ] Câmera do Webots posicionada para boa visualização
- [ ] Logs habilitados para demonstrar funcionamento

### Durante a Gravação

1. **Início:** Mostrar arena vazia, robô em posição inicial
2. **Spawn:** Mostrar 15 cubos aparecendo aleatoriamente
3. **Navegação:** Demonstrar desvio de obstáculos
4. **Detecção:** Mostrar robô identificando cubo
5. **Coleta:** Mostrar sequência de grasp
6. **Identificação:** Destacar classificação de cor
7. **Deposição:** Mostrar robô depositando na caixa correta
8. **Final:** Mostrar estatísticas (cubos coletados, tempo)

### Verificações Críticas

- [ ] GPS está **DESABILITADO** no robô
- [ ] supervisor.py **NÃO FOI MODIFICADO**
- [ ] Código-fonte **NÃO APARECE** no vídeo
- [ ] Sistema funciona com spawn aleatório

---

## 7. Troubleshooting

### Problema: LIDAR não retorna dados

**Solução:** Verificar se `lidar.enable(time_step)` foi chamado.

### Problema: Câmera retorna imagem preta

**Solução:** Aguardar pelo menos 2 time_steps após enable.

### Problema: Robô não se move

**Solução:** Verificar se wheels estão em modo velocidade:
```python
for wheel in wheels:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)
```

### Problema: Garra não fecha

**Solução:** Verificar nome correto do motor: `"finger::left"`

### Problema: Controlador fuzzy lento

**Solução:** Desabilitar logging:
```python
controller = FuzzyController({'logging': False})
```

---

## 8. Comandos Úteis

```bash
# Executar testes unitários
python -m pytest tests/control/ -v

# Verificar imports
python -c "from src.perception import CubeDetector; print('OK')"
python -c "from src.control import FuzzyController; print('OK')"

# Testar módulos individualmente
python -m src.perception.lidar_processor
python -m src.perception.cube_detector
python -m src.control.fuzzy_controller
python -m src.navigation.odometry
python -m src.manipulation.grasping
```

---

**Última Atualização:** 2025-11-24
**Versão:** 0.3.0
