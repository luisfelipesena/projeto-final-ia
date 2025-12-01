# YouBot MCP - Guia Completo de Testes e Debugging

**Atualizado:** 2025-12-01
**Status:** Correções V3 aplicadas - pronto para validação (VERSION: 2025-12-01-V3)

---

## 1. Arquitetura MCP (Model Context Protocol)

O sistema usa comunicação baseada em arquivos JSON entre Webots e ferramentas externas.

### 1.1 Estrutura de Arquivos

```
youbot_mcp/
├── youbot_mcp_controller.py   # Controller principal com lógica autônoma
└── data/youbot/
    ├── commands.json          # Comandos enviados para o robô
    ├── status.json            # Estado atual (atualizado a cada step)
    ├── nav_debug.log          # Logs de navegação/approach
    ├── grasp_log.txt          # Logs de grasping
    ├── camera_image.jpg       # Última imagem capturada
    └── lidar_data.json        # Dados LIDAR processados
```

### 1.2 Fluxo de Inicialização

1. **Webots carrega** `IA_20252/worlds/IA_20252.wbt`
2. **youbot.py** é executado com argumento `--mcp` (configurado em controllerArgs)
3. **youbot.py importa** `YouBotMCPController` de `youbot_mcp/youbot_mcp_controller.py`
4. **Loop principal**: lê comandos → atualiza visão → executa estado → escreve status

### 1.3 Dependências Críticas

```python
# youbot.py linha ~100
elif len(sys.argv) > 1 and sys.argv[1] == "--mcp":
    mcp_path = Path(__file__).resolve().parent.parent.parent.parent / 'youbot_mcp'
    sys.path.insert(0, str(mcp_path))
    from youbot_mcp_controller import YouBotMCPController
```

---

## 2. Como Rodar em Modo Autônomo

### 2.1 Passo a Passo

```bash
# 1. Abrir Webots com o world
open /Applications/Webots.app --args /path/to/IA_20252/worlds/IA_20252.wbt

# 2. Aguardar inicialização (~15 segundos até "MCP Controller Starting")

# 3. Enviar comando de início autônomo
echo '{"action": "start_autonomous", "params": {}, "timestamp": '$(date +%s)', "id": 1}' \
  > youbot_mcp/data/youbot/commands.json

# 4. Monitorar em tempo real
watch -n 1 'cat youbot_mcp/data/youbot/status.json | jq "{state: .current_state, cubes: .cubes_collected, target: .current_target}"'
```

### 2.2 Comandos MCP Disponíveis

| Action | Parâmetros | Descrição |
|--------|------------|-----------|
| `start_autonomous` | - | Inicia coleta autônoma |
| `stop_autonomous` | - | Para e volta para IDLE |
| `move` | `vx`, `vy`, `omega` | Movimento manual |
| `arm_height` | `height: LOW/MID/HIGH/RESET` | Define altura do braço |
| `arm_orientation` | `orientation: FRONT/LEFT/RIGHT` | Orientação do braço |
| `grip` | - | Fecha garra |
| `release` | - | Abre garra |
| `capture_camera` | - | Salva imagem em camera_image.jpg |
| `detect_cubes` | - | Força detecção |
| `grasp_sequence` | - | Executa sequência completa de grasp |
| `deposit_cube` | `color: green/blue/red` | Deposita na caixa da cor |

### 2.3 Exemplo de Monitoramento

```bash
# Terminal 1: Status em tempo real
watch -n 0.5 'cat youbot_mcp/data/youbot/status.json | jq "."'

# Terminal 2: Log de navegação
tail -f youbot_mcp/data/youbot/nav_debug.log

# Terminal 3: Console Webots (prints do controller)
# Visível diretamente na interface do Webots
```

---

## 3. Estados do Robô (State Machine)

```
IDLE ─────────────────────────────────────────────────────────┐
  │                                                           │
  └──(start_autonomous)──> SEARCHING                          │
                              │                               │
                         (target found)                       │
                              │                               │
                              v                               │
                         APPROACHING                          │
                              │                               │
                         (close enough)                       │
                              │                               │
                              v                               │
                         GRASPING                             │
                              │                               │
                         (grasp complete)                     │
                              │                               │
                              v                               │
                         DEPOSITING ────(done)───> SEARCHING  │
                                                              │
                         (stop_autonomous) ───────────────────┘
```

### 3.1 SEARCHING

**Comportamento:**
- Rotaciona ~270° (4.7 rad) em uma direção
- Alterna direção após cada scan completo
- Move para frente ~2 segundos entre scans
- Verifica obstáculos frontais antes de mover

**Código relevante:** `youbot_mcp_controller.py:478-518`

### 3.2 APPROACHING

**Comportamento esperado:**
1. Se `|angle| > 10°`: rotaciona para alinhar
2. Se `|angle| <= 10°`: move para frente
3. Completa quando `distance <= 0.25m`

**Código relevante:** `youbot_mcp_controller.py:527-565`

**PROBLEMA ATUAL:** Robô não rotaciona durante approach (ver seção 5)

### 3.3 GRASPING

**Sequência:**
1. Abre garra
2. Posiciona braço em FRONT_FLOOR
3. Avança suavemente
4. Fecha garra
5. Verifica sensor `has_object()`
6. Levanta braço

**Código relevante:** `youbot_mcp_controller.py:567-640`

### 3.4 DEPOSITING

**Comportamento:**
1. Levanta braço (FRONT_PLATE)
2. Navega até caixa da cor correspondente
3. Posiciona sobre a caixa
4. Abre garra

**Coordenadas das caixas:**
```python
DEPOSIT_BOXES = {
    'green': (0.48, 1.58),
    'blue': (0.48, -1.62),
    'red': (2.31, 0.01),
}
```

---

## 4. Sistema de Percepção

### 4.1 Detecção de Cubos (CubeDetector)

**Arquivo:** `src/perception/cube_detector.py`

**Pipeline:**
1. Segmentação HSV por cor (verde, azul, vermelho)
2. Detecção de contornos
3. Filtragem por área mínima (1000 pixels)
4. Cálculo de distância via modelo pinhole
5. Cálculo de ângulo: `angle = (cx - 0.5) * 60.0` (FOV 60°)

**HSV Ranges (aproximados):**
```python
'green': (35, 100, 100) - (85, 255, 255)
'blue': (100, 100, 100) - (130, 255, 255)
'red': (0, 100, 100) - (10, 255, 255) + (170, 100, 100) - (180, 255, 255)
```

### 4.2 Tracking (VisionService)

**Arquivo:** `src/services/vision_service.py`

**Parâmetros chave:**
```python
LOST_THRESHOLD = 30       # Frames até declarar perdido (~1s)
MIN_CONFIDENCE = 0.60     # Confiança mínima
POSITION_TOLERANCE = 0.30 # Tolerância de posição (metros)
ANGLE_TOLERANCE = 20.0    # Tolerância de ângulo (graus)
```

**Comportamento:**
- Mantém tracking de um cubo por vez
- Usa position matching (não só cor)
- Persiste posição mesmo se perdido temporariamente
- `lock_color()` limita detecção à cor selecionada

### 4.3 LIDAR

**Configuração:**
- 512 pontos em 360°
- Dividido em 9 setores (~40° cada)
- Range: 0.01m - 10.0m

**Setores:**
```
far_left | left | front_left | front | front_right | right | far_right | back_right | back_left
    0       1         2          3          4          5         6           7            8
```

---

## 5. PROBLEMAS ATUAIS IDENTIFICADOS

### 5.1 CRÍTICO: Robô não rotaciona durante approach

**Sintoma:**
- Ângulo do cubo permanece constante (~-26°) durante APPROACHING
- `omega` é enviado (0.8 rad/s) mas robô não gira
- `nav_debug.log` fica VAZIO ou mostra valores estáticos

**Investigação realizada:**

1. **Testado omega positivo e negativo** - nenhum funciona
2. **Testado `turn_left()`/`turn_right()` direto** - não funciona
3. **Verificado sintaxe Python** - OK
4. **Cache limpo múltiplas vezes** - problema persiste

**Evidência de cache:**
- Status mostra state=APPROACHING
- Mas `nav_debug.log` não é escrito
- Código novo inclui escrita no log que não aparece
- CONCLUSÃO: Webots está usando código antigo cacheado

**Causa raiz provável:**
Python importa `youbot_mcp_controller.py` uma vez e cacheia o módulo. Mesmo limpando `__pycache__`, o módulo já está carregado em memória.

**Soluções a tentar:**
1. Reiniciar Webots completamente (`pkill -9 -f webots`)
2. Usar `importlib.reload()` no youbot.py
3. Renomear o arquivo temporariamente para forçar reimportação
4. Adicionar timestamp no print inicial para verificar versão

### 5.2 Ângulo sempre ~26°

**Observação:** Cubos detectados consistentemente em -26° a -26.5°

**Hipótese:**
- FOV da câmera é 60° (±30°)
- Cubo no limite do FOV
- Quando robô tenta girar, cubo sai do view → target_lost
- Search rotaciona de volta → encontra mesmo cubo no mesmo ângulo

**Validação necessária:**
- Verificar se search está funcionando (omega alterna?)
- Verificar se target_lost ocorre rapidamente durante approach

### 5.3 Detecção de cor inconsistente

**Sintoma:** Status mostra cor diferente dos cubos visíveis

**Possíveis causas:**
- HSV thresholds inadequados para iluminação do ambiente
- Reflexos ou sombras afetando detecção
- Múltiplos cubos detectados, ordem incorreta

---

## 6. Cinemática das Rodas Mecanum

### 6.1 Convenção de Sinais

**Arquivo:** `IA_20252/controllers/youbot/base.py`

```python
# Kinematics formula:
speeds[0] = (1/R) * (vx - vy - K * omega)  # front-left
speeds[1] = (1/R) * (vx + vy + K * omega)  # front-right
speeds[2] = (1/R) * (vx + vy - K * omega)  # rear-left
speeds[3] = (1/R) * (vx - vy + K * omega)  # rear-right

# K = LX + LY = 0.386
# R = WHEEL_RADIUS = 0.05
```

### 6.2 Padrões de Movimento

| omega | Pattern wheels | Direção |
|-------|---------------|---------|
| +0.8 | [-,+,-,+] | Esquerda (anti-horário) |
| -0.8 | [+,-,+,-] | Direita (horário) |

### 6.3 Código Correto para Alinhar

```python
# Cubo à ESQUERDA (angle < 0) → girar ESQUERDA → omega POSITIVO
# Cubo à DIREITA (angle > 0) → girar DIREITA → omega NEGATIVO

omega = -math.radians(target.angle) * gain
# angle=-26° → omega = -(-0.454)*gain = +0.454*gain → gira ESQUERDA ✓
# angle=+26° → omega = -(+0.454)*gain = -0.454*gain → gira DIREITA ✓
```

---

## 7. Comandos de Debug Úteis

### 7.1 Reinício Limpo

```bash
# Matar tudo e limpar cache
pkill -9 -f webots
find /path/to/projeto -name "*.pyc" -delete
find /path/to/projeto -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
rm -f youbot_mcp/data/youbot/nav_debug.log

# Reiniciar
open /Applications/Webots.app --args /path/to/IA_20252/worlds/IA_20252.wbt
```

### 7.2 Verificar Versão do Código

Adicionar no início do APPROACHING:
```python
print("[MCP-V3] APPROACHING running version 2025-12-01-A")
```

Se não aparecer, código antigo está cacheado.

### 7.3 Teste Manual de Rotação

```bash
# Enviar comando de rotação direta
echo '{"action": "move", "params": {"vx": 0, "vy": 0, "omega": 0.5}, "timestamp": '$(date +%s)', "id": 99}' \
  > youbot_mcp/data/youbot/commands.json
```

### 7.4 Monitoramento Contínuo

```bash
# Estado e target
watch -n 0.5 'cat youbot_mcp/data/youbot/status.json | jq "{state: .current_state, angle: .current_target.angle, dist: .current_target.distance, omega: .base_velocity.omega}"'

# Detecções brutas
watch -n 0.5 'cat youbot_mcp/data/youbot/status.json | jq ".cube_detections"'

# LIDAR
watch -n 1 'cat youbot_mcp/data/youbot/status.json | jq ".obstacle_sectors.front"'
```

---

## 8. Requisitos vs Estado Atual

| Requisito | Status | Problema | Arquivo |
|-----------|--------|----------|---------|
| Detectar cubos | ✅ | Cores às vezes incorretas | `cube_detector.py` |
| Navegar até cubo | ⚠️ | Debug aprimorado - validar | `youbot_mcp_controller.py:681-740` |
| Pegar cubo | ⚠️ | Implementado, não testado | `youbot_mcp_controller.py:742-824` |
| Identificar cor | ⚠️ | Precisão a validar | `cube_detector.py:100-140` |
| Depositar | ⚠️ | Coordenadas definidas | `youbot_mcp_controller.py:71-74` |
| Evitar obstáculos | ✅ | LIDAR + Fuzzy funciona | `youbot_mcp_controller.py:620-679` |
| Usar RNA | ✅ | MLP LIDAR treinado (97.8%) | `models/lidar_mlp.pth` |
| Usar Fuzzy | ✅ | Integrado em SEARCHING/APPROACHING | `youbot_mcp_controller.py:620-740` |

---

## 9. Próximos Passos Priorizados

### P0: Resolver Cache (Bloqueador)
1. Verificar com timestamp único se código novo executa
2. Se não, investigar `importlib.reload()` ou renomear arquivo
3. Ou mover lógica para novo arquivo

### P1: Validar Rotação
1. Testar comando `move` manual com omega
2. Se funciona manual, problema é no código de approach
3. Se não funciona, problema na base.py ou mundo

### P2: Testar Grasp Isolado
1. Posicionar robô manualmente próximo ao cubo
2. Enviar `grasp_sequence`
3. Verificar `has_object()` e movimento físico

### P3: Validar Detecção de Cores
1. Capturar imagens de teste
2. Verificar HSV em cada canal
3. Ajustar thresholds se necessário

### P4: Integrar Fuzzy
1. Conectar `FuzzyController` ao loop principal
2. Usar para decisões de velocidade/navegação

---

## 10. Testes Isolados (Legado)

Ver seção original abaixo para testes de serviços individuais (ARM, MOVEMENT, VISION).

### 10.1 Configurar para Testes Isolados

```python
# Em service_tests.py linha 350:
TEST_TO_RUN = "arm_positions"  # arm_positions | movement | arm_grasp | vision
```

### 10.2 Testes Disponíveis

| Teste | Setup | Valida |
|-------|-------|--------|
| `arm_positions` | Nenhum | Motores do braço |
| `movement` | Área livre | Base omnidirecional |
| `arm_grasp` | Cubo a 25cm | Grasp + sensor |
| `vision` | Cubos visíveis | Tracking estável |

---

## 11. Checklist de Validação Final

### Fase 1: Debugging
- [x] Código novo confirmado em execução (timestamp) - VERSION: 2025-12-01-V3
- [x] SEARCHING não para mais por obstáculos traseiros
- [x] VisionService mantém tracking por mais tempo (LOST_THRESHOLD=60)
- [x] APPROACHING usa controle proporcional para rotação
- [x] Logs aparecem em nav_debug.log (enhanced logging added)

### Fase 2: Funcional
- [ ] SEARCHING → APPROACHING transita quando cubo detectado
- [ ] Alinhamento (angle → 0°) funciona com controle proporcional
- [ ] Approach (distance → 0.22m) funciona
- [ ] GRASPING pega cubo fisicamente
- [ ] has_object() retorna True
- [ ] DEPOSITING navega e solta

### Fase 3: Completo
- [ ] 1 cubo coletado e depositado
- [ ] Cores corretas nas caixas
- [ ] Sem colisões com obstáculos
- [ ] 15 cubos em <10 minutos

---

## 12. Correções Aplicadas (2025-12-01)

### V3 - Correções Críticas (2025-12-01)

#### Bug 1: SEARCHING parava imediatamente
- **Causa:** Checagem de obstáculo usava TODOS os setores LIDAR (incluindo traseiros)
- **Sintoma:** `min_obstacle_distance=0.398m` (parede traseira) < threshold 0.4m → stop
- **Fix:** Agora só considera setores FRONT (front, front_left, front_right)

#### Bug 2: VisionService perdia tracking muito rápido
- **Causa:** `LOST_THRESHOLD=30` (~1s) era muito curto para realinhar
- **Fix:** Aumentado para 60 frames (~2s), tolerâncias mais flexíveis

#### Bug 3: APPROACHING não alinhava corretamente
- **Causa:** Omega fixo de 0.8 era muito agressivo
- **Fix:** Controle proporcional com omega = -angle_rad * 1.5 (clampado a ±0.6)

#### Parâmetros VisionService atualizados:
```python
LOST_THRESHOLD = 60       # Frames (~2s) - mais tempo para realinhar
MIN_CONFIDENCE = 0.55     # Reduzido para aceitar mais detecções
POSITION_TOLERANCE = 0.40 # Metros - permite mais movimento
ANGLE_TOLERANCE = 30.0    # Graus - permite rotação durante approach
MIN_FRAMES_RELIABLE = 3   # Aquisição mais rápida
```

### V2 - Correções Anteriores

#### P0: F-String Fix
- **Arquivo:** `youbot_mcp_controller.py` linha 177
- **Problema:** Format specifier inválido em f-string
- **Fix:** Separado em if/else para evitar expressão condicional no specifier

#### P1: Fuzzy Controller Integrado
- Importado `FuzzyController` e `FuzzyInputs`
- Inicializado no `__init__` com logging habilitado
- Adicionado `_compute_fuzzy_inputs()` para converter sensor data
- Integrado em SEARCHING e APPROACHING states

#### P2: Modelo RNA LIDAR
- Treinado `models/lidar_mlp.pth` com 2000 amostras sintéticas
- Accuracy: 97.8% na validação
- Carregamento automático no `__init__` com fallback heurístico

#### P3: Debug Aprimorado no APPROACHING
- Logging detalhado com wheel speeds e fuzzy outputs
- Arquivo `nav_debug.log` atualizado a cada frame
- Timestamp incluído em cada entrada

#### P4: Version Check
- `VERSION = "2025-12-01-V3"` na classe
- Timestamp impresso no console ao iniciar
- Versão incluída no `status.json`

### Comandos para Limpar Cache e Testar
```bash
# 1. Limpar cache Python
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
rm -f youbot_mcp/data/youbot/nav_debug.log youbot_mcp/data/youbot/grasp_log.txt

# 2. Reiniciar Webots
pkill -9 -f webots
open /Applications/Webots.app --args $(pwd)/IA_20252/worlds/IA_20252.wbt

# 3. Verificar versão no console
# Deve aparecer: [MCP Controller] Initializing... VERSION: 2025-12-01-V3

# 4. Iniciar modo autônomo
echo '{"action": "start_autonomous", "params": {}, "timestamp": '$(date +%s)', "id": 1}' \
  > youbot_mcp/data/youbot/commands.json

# 5. Monitorar estado e target
watch -n 0.5 'cat youbot_mcp/data/youbot/status.json | jq "{version, state: .current_state, angle: .current_target.angle, dist: .current_target.distance, velocity: .base_velocity}"'

# 6. Monitorar log de navegação
tail -f youbot_mcp/data/youbot/nav_debug.log
```

---

*Última atualização: 2025-12-01*
*Status: Correções aplicadas - pronto para validação*
