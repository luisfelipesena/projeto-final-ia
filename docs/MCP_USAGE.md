# YouBot MCP - Guia de Uso

## Visão Geral

O YouBot MCP (Model Context Protocol) permite controlar o robô YouBot no Webots através de um servidor MCP que se comunica via arquivos JSON.

## Arquitetura

```
Claude/LLM  <-->  MCP Server  <-->  JSON files  <-->  Webots Controller
                     |                    |                    |
            youbot_mcp_server.py    commands.json      youbot_mcp_controller.py
                                    status.json
                                    camera_image.jpg
```

## Pré-requisitos

### 1. Instalar Dependências

```bash
cd /path/to/projeto-final-ia
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configurar Webots

1. Abra o projeto no Webots: `IA_20252/worlds/IA_20252.wbt`
2. Defina o controller do YouBot para `youbot_mcp_controller`
3. Copie o controller para a pasta correta:
   ```bash
   cp youbot_mcp/youbot_mcp_controller.py IA_20252/controllers/youbot_mcp_controller/
   ```

## Uso

### Iniciar o MCP Server

```bash
cd youbot_mcp
python youbot_mcp_server.py
```

### Conectar via Claude Desktop

Adicione ao `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "youbot": {
      "command": "python",
      "args": ["/path/to/youbot_mcp/youbot_mcp_server.py"],
      "env": {}
    }
  }
}
```

## Comandos MCP Disponíveis

### Movimento Base
- `youbot_move` - Movimento omnidirecional (vx, vy, omega)
- `youbot_stop` - Parar movimento
- `youbot_forward/backward/left/right` - Movimentos simples

### Braço
- `youbot_arm_height` - Definir altura (FRONT_FLOOR, RESET, etc.)
- `youbot_arm_orientation` - Definir orientação

### Gripper
- `youbot_grip` - Fechar garra
- `youbot_release` - Abrir garra

### Percepção
- `youbot_camera` - Capturar imagem da câmera
- `youbot_lidar` - Dados LIDAR processados
- `youbot_detect_cubes` - Detectar cubos coloridos

### Alto Nível
- `youbot_grasp` - Sequência completa de pegar cubo
- `youbot_deposit` - Depositar cubo na caixa
- `youbot_autonomous` - Modo autônomo (liga/desliga)
- `youbot_status` - Status atual do robô

## Arquivos de Comunicação

```
youbot_mcp/data/youbot/
├── commands.json      # Comandos do MCP → Controller
├── status.json        # Status do Controller → MCP
├── camera_image.jpg   # Última imagem da câmera
└── grasp_*.jpg        # Screenshots durante grasp
```

## Exemplo de Uso

```python
# Via Claude ou outro cliente MCP:

# 1. Ver status
youbot_status()

# 2. Detectar cubos
youbot_detect_cubes()

# 3. Pegar cubo verde
youbot_grasp(color="green")

# 4. Verificar se pegou
youbot_status()  # check has_object

# 5. Depositar
youbot_deposit(color="green")
```

## Modo Autônomo

```python
# Ativar modo autônomo (coleta todos os 15 cubos)
youbot_autonomous(enabled=True)

# Monitorar progresso
youbot_status()  # cubes_collected

# Desativar
youbot_autonomous(enabled=False)
```

## Troubleshooting

### MCP não conecta
- Verifique se Webots está rodando
- Verifique se o controller está carregado
- Check logs em `youbot_mcp/youbot_mcp.log`

### Robô não responde
- Verifique `youbot_mcp/data/youbot/status.json` está sendo atualizado
- Reinicie a simulação no Webots (Cmd+Shift+R)

### Grasp falha
- Verifique `youbot_mcp/data/youbot/grasp_*.jpg` para debug
- Ajuste distância de approach se necessário

## Integração com /src

O MCP controller usa os módulos em `/src`:
- `src/perception/` - CubeDetector, LidarProcessor, LidarMLP
- `src/control/` - FuzzyNavigator, FuzzyManipulator
- `src/actuators/` - BaseController, ArmController, GripperController

## Permissões macOS

O reload do Webots usa AppleScript. Garanta permissão em:
System Preferences → Security & Privacy → Privacy → Accessibility
