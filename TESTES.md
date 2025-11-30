# Guia de Testes - Serviços Modulares YouBot

**DECISÃO 028:** Arquitetura modular com serviços testáveis isoladamente.

---

## 🚀 Quick Start: Como Testar no Webots

### Passo 1: Abrir Webots

```bash
open -a Webots
# Ou: /Applications/Webots.app/Contents/MacOS/webots
```

### Passo 2: Carregar o World

1. `File → Open World...`
2. Selecionar: `IA_20252/worlds/IA_20252.wbt`
3. **NÃO** clique Play ainda!

### Passo 3: Configurar o Controlador de Teste

1. Pausar simulação (botão `||`)
2. Na árvore à esquerda, expandir `youBot`
3. Clicar em `controller "youbot"`
4. No painel direito, mudar `controller` de `"youbot"` para `"<extern>"`
5. **OU** editar `IA_20252/controllers/youbot/youbot.py`:

```python
# ANTES (linha ~100):
from src.main_controller import MainController

# DEPOIS (para testes isolados):
from service_tests import run_all_tests
```

### Passo 4: Escolher o Teste

Editar `IA_20252/controllers/youbot/service_tests.py` linha 350:

```python
# Opções: arm_positions, arm_grasp, movement, vision
TEST_TO_RUN = "arm_positions"  # Mudar conforme necessário
```

### Passo 5: Rodar

1. Salvar arquivos
2. Clicar Play (`▶`)
3. Observar console para output

---

## 📋 Ordem de Testes (Validação Incremental)

### TESTE 1: ARM POSITIONS (Primeiro - Sem Setup)

**Objetivo:** Verificar se o braço se move corretamente entre posições.

**Setup:** Nenhum. Não precisa de cubo.

**O que faz:**
1. Move braço para RESET (tucked)
2. Move para FRONT_PLATE (raised)
3. Move para FRONT_FLOOR (lowered)
4. Retorna para FRONT_PLATE
5. Retorna para RESET

**Configurar:**
```python
TEST_TO_RUN = "arm_positions"
```

**Sucesso esperado:**
```
=================================================
TESTE: ARM POSITIONS
=================================================
  → Movendo para: RESET (tucked)
    ✓ Chegou em RESET (tucked)
  → Movendo para: FRONT_PLATE (raised)
    ✓ Chegou em FRONT_PLATE (raised)
  ...
TESTE ARM POSITIONS: COMPLETO
=================================================
```

**Checklist:**
- [ ] Braço move suavemente entre posições?
- [ ] Nenhum erro de motor?
- [ ] Posições finais parecem corretas?

---

### TESTE 2: MOVEMENT SQUARE (Segundo - Sem Setup)

**Objetivo:** Verificar se a base móvel funciona corretamente.

**Setup:** Nenhum. Certifique que área à frente está livre.

**O que faz:**
1. Move 0.5m para frente
2. Gira 90° esquerda
3. Repete 4x (quadrado completo)
4. Deve retornar ~posição inicial

**Configurar:**
```python
TEST_TO_RUN = "movement"
```

**Sucesso esperado:**
```
=================================================
TESTE: MOVEMENT SQUARE
=================================================
  Lado 1/4:
    → Frente 0.5m...
    → Girando 90°...
  Lado 2/4:
    ...
TESTE MOVEMENT SQUARE: COMPLETO
Verificar: robot voltou ao ponto inicial?
=================================================
```

**Checklist:**
- [ ] Robot move para frente corretamente?
- [ ] Giros são ~90°?
- [ ] Retorna aproximadamente ao ponto inicial?
- [ ] Movimento é suave (sem tremores)?

---

### TESTE 3: ARM GRASP (Terceiro - REQUER Setup Manual)

**Objetivo:** Verificar ciclo completo de grasp.

**⚠️ SETUP OBRIGATÓRIO:**

1. **ANTES de dar Play**, pausar simulação (`||`)
2. Na árvore à esquerda, encontrar um cubo (ex: `DEF GREEN_CUBE_0 WoodenCube`)
3. No painel direito, editar `translation`:
   - X: `0` (centro frente do robot)
   - Y: `0.025` (altura do cubo no chão)
   - Z: `-0.25` (25cm à frente do robot)
4. Dar Play (`▶`)

**Configurar:**
```python
TEST_TO_RUN = "arm_grasp"
```

**O que faz:**
1. Abre gripper
2. Move braço para FRONT_PLATE (raised)
3. Abaixa braço para FRONT_FLOOR
4. Fecha gripper
5. Verifica sensor `has_object()`
6. Levanta braço
7. Abre gripper (deposita)
8. Retorna para RESET

**Sucesso esperado:**
```
=================================================
TESTE: ARM GRASP CYCLE
=================================================
  [1/7] Abrindo gripper...
    ✓ Gripper aberto
  [2/7] Movendo braço para frente (raised)...
    ✓ Braço em FRONT_PLATE
  [3/7] Abaixando braço para o chão...
    ✓ Braço em FRONT_FLOOR
  [4/7] Fechando gripper...
    ✓ Gripper fechado
  [5/7] Verificando sensor...
    → has_object() = True
    ✓✓✓ CUBO DETECTADO! Grasp funcionou!
  [6/7] Levantando braço...
    ✓ Braço levantado
  [7/7] Abrindo gripper (depositar)...
    ✓ Gripper aberto
TESTE ARM GRASP: SUCESSO!
=================================================
```

**Checklist:**
- [ ] `has_object() = True`? Se False, cubo mal posicionado
- [ ] Cubo foi fisicamente agarrado?
- [ ] Cubo levantou junto com o braço?
- [ ] Cubo caiu ao abrir gripper?

**Troubleshooting se `has_object() = False`:**
1. Cubo muito longe (>30cm)
2. Cubo muito perto (<15cm)
3. Cubo desalinhado lateralmente
4. Gripper não fechou completamente

---

### TESTE 4: VISION TRACKING (Quarto - Setup: Cubos Visíveis)

**Objetivo:** Verificar estabilidade do tracking de cubos.

**Setup:** Ter cubos visíveis na frente do robot (spawned pelo supervisor).

**O que faz:**
1. Processa 100 frames de câmera
2. Registra quantas vezes o tracking "pulou" entre cubos diferentes
3. Reporta switches (oscilações)

**Configurar:**
```python
TEST_TO_RUN = "vision"
```

**Sucesso esperado:**
```
=================================================
TESTE: VISION TRACKING
=================================================
  Frame 0: Primeiro target: green (id=1)
  Frame 20: green id=1 dist=1.45m angle=-3.2°
  Frame 40: green id=1 dist=1.45m angle=-3.1°
  ...
TESTE VISION TRACKING: 0 switches
  ✓ ESTÁVEL - Tracking não oscilou
=================================================
```

**Checklist:**
- [ ] `switches = 0`? Tracking estável
- [ ] Se switches > 0, verificar se há múltiplos cubos próximos
- [ ] Distâncias e ângulos parecem realistas?

---

## 🔧 Validação do Controller Principal (main_controller_v2)

Após validar serviços isolados, testar integração:

### Configurar youbot.py:

```python
# IA_20252/controllers/youbot/youbot.py

# Comentar:
# from src.main_controller import MainController

# Descomentar/adicionar:
from src.main_controller_v2 import MainControllerV2 as MainController
```

### Comportamento Esperado:

```
[MainControllerV2] Initializing...
[MainControllerV2] Initialization complete
[MainControllerV2] Starting main loop
  Time step: 32ms
[State] SEARCHING → APPROACHING (found green)
[Navigation] ALIGNED: angle=2.3° → APPROACH
[Navigation] APPROACH: dist=1.20m angle=1.5°
[Navigation] APPROACH: dist=0.85m angle=0.8°
[Navigation] COMPLETE: dist=0.28m angle=0.5°
[State] APPROACHING → GRASPING (dist=0.28m)
[Grasping] Attempting grasp of green cube
[Grasping] SUCCESS! Total: 1
[State] GRASPING → DEPOSITING (grasp_success)
[Depositing] Moving to green box
[Depositing] Complete! Cubes: 1
[State] DEPOSITING → SEARCHING (deposit_complete)
...
```

---

## 📊 Matriz de Validação

| Teste | Precisa Setup? | Duração | Valida |
|-------|---------------|---------|--------|
| ARM_POSITIONS | Não | ~15s | Motores do braço |
| MOVEMENT | Não | ~30s | Base omnidirecional |
| ARM_GRASP | Sim (cubo) | ~20s | Grasp físico + sensor |
| VISION | Não | ~5s | Tracking estável |
| MAIN_V2 | Não | ~5min | Integração completa |

---

## ❌ Problemas Comuns

### "ImportError: No module named 'services'"

Path não configurado. Verificar se `service_tests.py` tem:
```python
src_path = Path(__file__).resolve().parent.parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))
```

### Braço não move

1. Verificar se `Arm` foi importado corretamente
2. Recarregar world: `Ctrl+Shift+L`

### `has_object() = False` sempre

1. Posicionar cubo mais próximo (~20cm)
2. Verificar alinhamento lateral
3. Sensor pode precisar de mais frames após fechar gripper

### Tracking oscila muito

Múltiplos cubos da mesma cor muito próximos. VisionService usa posição para distinguir, mas se distâncias são muito similares pode confundir.

### Robot não se move (MOVEMENT test)

1. Simulação pausada?
2. Área à frente bloqueada?
3. Verificar console para erros

---

## 📁 Arquivos de Teste

| Arquivo | Função |
|---------|--------|
| `IA_20252/controllers/youbot/service_tests.py` | Testes isolados (ARM, MOVEMENT, VISION) |
| `src/services/movement_service.py` | MovementService + test_square() |
| `src/services/arm_service.py` | ArmService + test_grasp_cycle() |
| `src/services/vision_service.py` | VisionService |
| `src/services/navigation_service.py` | NavigationService + test_approach() |
| `src/main_controller_v2.py` | Controller integrado usando serviços |

---

## ✅ Checklist Final de Validação

### Fase 1: Serviços Isolados
- [ ] ARM_POSITIONS passou (braço move)
- [ ] MOVEMENT passou (base move em quadrado)
- [ ] ARM_GRASP passou (cubo detectado e pegou)
- [ ] VISION passou (0 switches)

### Fase 2: Integração
- [ ] main_controller_v2.py roda sem erros
- [ ] Estado transita SEARCHING → APPROACHING corretamente
- [ ] Estado transita APPROACHING → GRASPING corretamente
- [ ] Grasp físico funciona (cubo levanta)
- [ ] Depositing funciona

### Fase 3: Ciclo Completo
- [ ] Pelo menos 1 cubo coletado e depositado
- [ ] Sem oscilação de estados excessiva
- [ ] Performance aceitável (15 cubos em <10min)

---

**Última atualização:** 2024-11-30 (DECISÃO 028 - Arquitetura Modular)
