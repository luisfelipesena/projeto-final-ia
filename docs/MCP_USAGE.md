# YouBot MCP - Guia de Uso

## Arquitetura

```
Claude/LLM  <-->  Arquivos JSON  <-->  Webots Controller
                      |                       |
               commands.json          youbot_grasp_test.py
               status.json            (ou youbot_mcp_controller.py)
               grasp_*.jpg
```

## CRITICAL: Reiniciar Webots para Novos Controllers

Webots **cacheia** Python controllers. `Cmd+Shift+R` NÃO recarrega código Python!

### Para recarregar controller:
```bash
# Opção 1: Kill e reabrir (RECOMENDADO)
osascript -e 'tell application "Webots" to quit'
sleep 2
open -a Webots "/path/to/world.wbt"

# Opção 2: Via Claude Code
osascript << 'EOF'
tell application "Webots"
    quit
end tell
EOF
sleep 2
open -a Webots "/path/to/world.wbt" &
sleep 40  # Aguardar teste completar
cat youbot_mcp/data/youbot/status.json
```

### Verificar se código foi recarregado:
- Checar timestamp dos screenshots:
  ```bash
  ls -la youbot_mcp/data/youbot/v3_*.jpg | head -3
  ```
- Se timestamps antigos, controller NÃO foi recarregado

## Arquivos de Comunicação

```
youbot_mcp/data/youbot/
├── commands.json      # Comandos → Controller
├── status.json        # Status ← Controller
├── grasp_*.jpg        # Screenshots durante grasp
```

## YOLO Mode - Teste de Grasp (Step 1)

### Setup Rápido

1. **Abrir Webots**: `IA_20252/worlds/IA_20252.wbt`
2. **Reload** (Cmd+Shift+R) - spawna cubos, carrega controller
3. **Play** - auto-teste roda após 3s

### Controller Atual: `youbot_grasp_test`

O robô automaticamente:
1. Espera 3s (warmup)
2. Abre gripper
3. Baixa braço (FRONT_FLOOR)
4. Move 8cm para frente
5. Fecha gripper
6. Verifica objeto (sensor dedo > 0.005)
7. Levanta (FRONT_PLATE)
8. Salva screenshots

### Verificar Resultado

**Status** (`youbot_mcp/data/youbot/status.json`):
```json
{
  "version": "GRASP_TEST_V1",
  "grasp_result": {
    "success": true/false,
    "finger_before": 0.0,
    "finger_after": 0.012,
    "has_object": true/false
  }
}
```

**Screenshots**:
- `grasp_01_start.jpg` - Inicial
- `grasp_02_arm_lowered.jpg` - Braço baixo
- `grasp_03_after_forward.jpg` - Após mover
- `grasp_04_after_grip.jpg` - Gripper fechado
- `grasp_05_lifted.jpg` - Cubo levantado
- `grasp_06_final.jpg` - Final

### Comandos MCP (via commands.json)

```json
{"id": 1, "action": "run_grasp_test", "params": {}}
{"id": 2, "action": "grip", "params": {}}
{"id": 3, "action": "release", "params": {}}
{"id": 4, "action": "set_arm_height", "params": {"height": "FRONT_FLOOR"}}
{"id": 5, "action": "move_forward", "params": {"distance_m": 0.08}}
{"id": 6, "action": "check_grasp", "params": {}}
```

## Troubleshooting

### Erro "No module named 'perception.lidar_mlp'"
- Controller errado. Verificar que é `youbot_grasp_test` no world file.

### Grasp falha (has_object=false)
1. Robô muito longe do cubo → Reposicionar no Webots
2. Cubo não centralizado → Ajustar posição Y
3. Threshold errado → Verificar finger_position no status

### Não há cubos
- Reload (Cmd+Shift+R) para spawnar cubos

## Próximos Passos (Após Step 1 OK)

1. **Step 2**: Detecção de cubo (câmera + HSV)
2. **Step 3**: Lógica de approach
3. **Step 4**: Integrar com estado autônomo completo
