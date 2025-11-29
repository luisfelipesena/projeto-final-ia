# Guia de Testes - YouBot Autônomo

## Status Atual

**GPS:** ⚠️ Dispositivo não existe no YouBot padrão - usando apenas odometria

**Fixes aplicados (DECISÃO 022):**
- Camera warmup (10 frames)
- Stable detection (3 frames consecutivos)
- Min approaching time (2s)
- AVOIDING thresholds ajustados (0.4m/0.6m)
- Output smoothing (EMA 0.3)
- Distance estimation calibrada
- **MAX_CONTOUR_AREA = 15000** - Filtra deposit boxes (objetos grandes)
- **Grasp timing** - 8.5s total (aumentado para dar tempo ao braço)
- **Print spam fix** - Logs de grasp só aparecem 1x por estado
- **AVOIDING excluído durante GRASPING/DEPOSITING** - Cubo próximo não triggera AVOIDING
- **Log spam reduzido** - "Waiting for min approach time" aparece apenas 1x/segundo

---

## FASE 1: Teste com GPS (Treinamento)

### 1.1 Executar no Webots

```bash
# Abrir Webots
open -a Webots

# Ou via linha de comando
/Applications/Webots.app/Contents/MacOS/webots
```

**Carregar World:**
- `File → Open World...`
- Selecionar: `IA_20252/worlds/IA_20252.wbt`
- Clicar Play (▶)

### 1.2 O que observar no Console

**Inicialização esperada (SUCESSO):**
```
[youbot] Starting Autonomous YouBot Controller
[youbot] MATA64 Final Project - Cube Collection Task
[youbot] GPS Disabled - Odometry Navigation Only

[youbot] Initializing MainController...
Device "gps" was not found on robot "youBot"
[MainController] GPS not found - using odometry only
Initializing Main Controller...
  Perception system initialized
  Fuzzy controller initialized (26 rules)
  State machine initialized
  Navigation initialized
  Manipulation initialized
Main Controller ready!
[youbot] MainController ready - starting autonomous operation
Spawn complete. The supervisor has spawned 15/15 objects
[INFO] Camera warmup complete (10 frames)          ← warmup OK
```

### 1.3 Comportamento esperado APÓS fixes

| Antes | Depois |
|-------|--------|
| SEARCHING → GRASPING em <1s | Espera 3 detecções + 2s approaching |
| cube_distance: 0.1 sempre | Distâncias realistas (0.15-3.0m) |
| Oscilação +15°/-15° | Movimento suave (smoothing) |
| AVOIDING lock 2min | Exit rápido (0.6m threshold) |
| Detecta deposit box | Filtra objetos >15000px |
| Print spam GRASP | 1 print por estado |
| GRASPING → AVOIDING (cubo = obstáculo) | AVOIDING desativado durante GRASPING/DEPOSITING |
| Log spam "Waiting..." ~50x/s | 1 log por segundo |

### 1.4 Observações sobre GPS

**NOTA:** GPS não existe no YouBot padrão do world file.
O sistema usa apenas **odometria + LIDAR + câmera** (como requerido na demo final).

---

## FASE 2: Checklist de Validação

### Teste A: Inicialização (30s)

- [ ] Console mostra "GPS not found - using odometry only"?
- [ ] "Camera warmup complete (10 frames)"?
- [ ] 15 cubos visíveis na arena?
- [ ] Robô começa em SEARCHING (não GRASPING imediato)?

**Me envie:** Screenshot do console nos primeiros 30s

### Teste B: Detecção Estável (2min)

Observe transições de estado:

- [ ] SEARCHING por alguns segundos antes de APPROACHING?
- [ ] "Waiting for min approach time" aparece?
- [ ] Distâncias reportadas são realistas (não 0.1m constante)?

**Me envie:** Console log mostrando transições

### Teste C: Movimento Suave (2min)

- [ ] Robô move sem oscilar +15°/-15°?
- [ ] Velocidades parecem suaves?
- [ ] AVOIDING não trava por 2min?

**Me envie:** Descrição do comportamento ou vídeo curto

### Teste D: Ciclo Completo (5min)

Fluxo esperado:
```
SEARCHING (alguns segundos)
    ↓ cube_detected (3 frames estáveis)
APPROACHING (mín 2s)
    ↓ distance < 0.25m + 2s elapsed
GRASPING (sequência de 5s)
    ↓ grasp_success
NAVIGATING_TO_BOX
    ↓ at_target_box
DEPOSITING
    ↓ deposit_complete
SEARCHING (reinicia)
```

- [ ] Completou ao menos 1 ciclo?
- [ ] Se parou, em qual estado?

**Me envie:** Logs do ciclo ou estado onde travou

---

## FASE 3: Desabilitar GPS para Demo Final

### 3.1 Como desabilitar

Editar `src/main_controller.py`:

```python
# Linha ~157-161: COMENTAR estas linhas
# self.gps = robot.getDevice("gps")
# if self.gps:
#     self.gps.enable(self.time_step)
#     print("[MainController] GPS enabled for training mode")

# Ou simplesmente:
self.gps = None  # Desabilita GPS
```

E comentar o log de GPS no step():
```python
# Linha ~432-434: COMENTAR
# if self.gps and self.loop_count % 30 == 0:
#     pos = self.gps.getValues()
#     self._log(f"GPS: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
```

### 3.2 Validação End-to-End sem GPS

**Objetivo:** Confirmar que sistema funciona APENAS com odometria + LIDAR + câmera

**Teste crítico:**
1. Desabilitar GPS (passo 3.1)
2. Recarregar world no Webots
3. Observar que NÃO aparece "GPS enabled"
4. Robô deve completar ciclo usando apenas odometria

**Checklist final:**
- [ ] Sem "GPS enabled" no console
- [ ] Robô navega para caixa correta
- [ ] Deposita cubo na cor certa
- [ ] Sistema completa ≥1 ciclo

---

## O que me enviar após testes

### Obrigatório (Fase 2)

1. **Screenshot console inicialização** - mostrando warmup + GPS
2. **Logs de transição de estados** - SEARCHING→APPROACHING→etc
3. **Comportamento:** suave ou ainda oscila?
4. **Ciclo completo?** SIM/PARCIAL/NÃO + estado onde parou

### Se houver problemas

5. **Texto completo de erros/tracebacks**
6. **Distâncias reportadas** (ainda 0.1m?)
7. **Tempo em cada estado** antes de transição

### Validação Final (Fase 3)

8. **Confirmação:** sistema funciona SEM GPS?
9. **Screenshot:** console sem "GPS enabled"
10. **Ciclo completo sem GPS?** SIM/NÃO

---

## Troubleshooting

### "GPS: x=nan, y=nan"
GPS não está habilitado no world file. Verificar se sensor GPS existe no YouBot.

### Ainda detecta cube_distance: 0.1
Verificar se `cube_detector.py` foi atualizado (MIN_CONTOUR_AREA = 1500).

### Detecta deposit box como cubo
Verificar se `cube_detector.py` tem MAX_CONTOUR_AREA = 15000 para filtrar objetos grandes.

### Ainda transita SEARCHING→GRASPING instantâneo
Verificar se `state_machine.py` foi atualizado (CUBE_DETECTED_THRESHOLD = 3).

### Robot não se move
1. Simulação pausada?
2. Erros no console?
3. Recarregar world: `Ctrl+Shift+L`

### AVOIDING ainda trava
Verificar thresholds em `state_machine.py`: entry=0.4m, exit=0.6m

### GRASPING → AVOIDING interrompe grasp (CORRIGIDO)
**Problema anterior:** Durante GRASPING, o cubo (a ~0.13m) era detectado como obstáculo, causando transição para AVOIDING e rotação infinita.
**Solução:** `state_machine.py` agora exclui GRASPING e DEPOSITING do trigger de AVOIDING.

---

## Métricas de Sucesso

| Métrica | Meta | Status |
|---------|------|--------|
| Camera warmup | 10 frames | Implementado |
| Stable detection | 3 frames | Implementado |
| Min approach time | 2s | Implementado |
| AVOIDING exit | <30s | Threshold 0.6m |
| Output smoothing | EMA 0.3 | Implementado |
| GPS training mode | Ativo | N/A (GPS não existe) |
| Deposit box filter | MAX_CONTOUR_AREA | Implementado (15000px) |
| Grasp timing | 8.5s | Implementado |
| Print spam fix | 1x/estado | Implementado |
| AVOIDING excluído manipulação | GRASPING/DEPOSITING | Implementado |
| Log spam approach | 1x/segundo | Implementado |
| Ciclo completo | ≥1 | A validar |
| GPS desabilitado | Demo final | N/A (sempre odometria) |
