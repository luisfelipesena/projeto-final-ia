# YouBot MCP - Publication Guide

## Overview

This guide explains how to publish the YouBot MCP Server as a **standalone repository** for community use. The MCP enables any LLM (Claude, GPT, etc.) to control the KUKA YouBot in Webots simulator.

## Why Separate Repository?

- **Cleaner package**: Only MCP-related code, no academic project files
- **Easy installation**: `pip install webots-youbot-mcp`
- **MCP Registry compatible**: Follows official publishing standards
- **Community contributions**: Independent issue tracking and PRs

## Step 1: Create New Repository via GitHub CLI

```bash
# Install gh if needed
brew install gh

# Authenticate
gh auth login

# Create new public repository
gh repo create webots-youbot-mcp \
  --public \
  --description "MCP server for KUKA YouBot control in Webots simulator" \
  --license MIT

# Clone it
cd ~/Development
gh repo clone webots-youbot-mcp
cd webots-youbot-mcp
```

## Step 2: Copy MCP Files

```bash
# From projeto-final-ia
cp ../projeto-final-ia/youbot_mcp/youbot_mcp_server.py ./src/
cp ../projeto-final-ia/youbot_mcp/youbot_mcp_controller.py ./src/

# Copy required services (only what MCP needs)
mkdir -p src/services
cp ../projeto-final-ia/src/services/movement_service.py ./src/services/
cp ../projeto-final-ia/src/services/arm_service.py ./src/services/
cp ../projeto-final-ia/src/services/vision_service.py ./src/services/
cp ../projeto-final-ia/src/perception/cube_detector.py ./src/perception/
```

## Step 3: Repository Structure

```
webots-youbot-mcp/
├── README.md                  # Main documentation
├── LICENSE                    # MIT License
├── pyproject.toml             # Python package config
├── server.json                # MCP Registry metadata
├── src/
│   ├── __init__.py
│   ├── youbot_mcp_server.py   # MCP server (FastMCP)
│   ├── youbot_mcp_controller.py # Webots controller
│   ├── services/              # Robot services
│   │   ├── movement_service.py
│   │   ├── arm_service.py
│   │   └── vision_service.py
│   └── perception/
│       └── cube_detector.py
├── worlds/                    # Example Webots worlds
│   └── youbot_demo.wbt
└── docs/
    ├── SETUP.md              # Installation guide
    └── API.md                # Tool reference
```

## Step 4: Create pyproject.toml

```toml
[project]
name = "webots-youbot-mcp"
version = "1.0.0"
description = "MCP server for KUKA YouBot control in Webots simulator"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Luis Felipe Sena"}
]
keywords = ["mcp", "webots", "youbot", "robotics", "simulation", "claude"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "mcp>=1.0.0",
    "pydantic>=2.0.0",
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
]

[project.optional-dependencies]
dev = ["pytest", "black", "mypy"]

[project.urls]
Homepage = "https://github.com/luisfelipesena/webots-youbot-mcp"
Repository = "https://github.com/luisfelipesena/webots-youbot-mcp"
Documentation = "https://github.com/luisfelipesena/webots-youbot-mcp#readme"

[project.scripts]
youbot-mcp = "src.youbot_mcp_server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Step 5: Create server.json (MCP Registry)

```json
{
  "name": "webots-youbot-mcp",
  "version": "1.0.0",
  "description": "Control KUKA YouBot robot in Webots simulator via MCP",
  "publisher": {
    "id": "io.github.luisfelipesena",
    "name": "Luis Felipe Sena",
    "url": "https://github.com/luisfelipesena"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/luisfelipesena/webots-youbot-mcp"
  },
  "license": "MIT",
  "runtime": "python",
  "capabilities": {
    "tools": true,
    "resources": false,
    "prompts": false
  },
  "tools": [
    {"name": "youbot_check_connection", "description": "Verify Webots controller is running"},
    {"name": "youbot_get_status", "description": "Get complete robot state"},
    {"name": "youbot_move_base", "description": "Set omnidirectional velocities"},
    {"name": "youbot_stop_base", "description": "Stop all movement"},
    {"name": "youbot_move_forward", "description": "Move forward by distance"},
    {"name": "youbot_rotate", "description": "Rotate in place"},
    {"name": "youbot_set_arm_height", "description": "Set arm height preset"},
    {"name": "youbot_set_arm_orientation", "description": "Set arm rotation"},
    {"name": "youbot_set_arm_position", "description": "IK positioning"},
    {"name": "youbot_reset_arm", "description": "Return arm to rest"},
    {"name": "youbot_grip", "description": "Close gripper"},
    {"name": "youbot_release", "description": "Open gripper"},
    {"name": "youbot_get_camera_image", "description": "Capture RGB image"},
    {"name": "youbot_detect_cubes", "description": "Find colored cubes"},
    {"name": "youbot_get_lidar_data", "description": "Get obstacle ranges"},
    {"name": "youbot_grasp_cube", "description": "Execute grasp sequence"},
    {"name": "youbot_deposit_cube", "description": "Navigate and deposit"},
    {"name": "youbot_start_autonomous", "description": "Start task automation"},
    {"name": "youbot_stop_autonomous", "description": "Stop automation"},
    {"name": "youbot_reload_world", "description": "Reload Webots world (macOS)"}
  ]
}
```

## Step 6: Create README.md

```markdown
# YouBot MCP Server

Control KUKA YouBot robot in Webots simulator using Model Context Protocol.

## Features

- **22 MCP Tools** for complete robot control
- **Omnidirectional movement** via mecanum wheels
- **5-DOF arm** with inverse kinematics
- **RGB camera** with HSV color detection
- **360° LIDAR** for obstacle avoidance
- **Autonomous mode** for cube collection tasks
- **macOS Integration** - Reload Webots via AppleScript

## Quick Start

### 1. Install
```bash
pip install webots-youbot-mcp
```

### 2. Configure Claude Desktop
Add to `~/.config/claude/mcp.json`:
```json
{
  "mcpServers": {
    "youbot": {
      "command": "youbot-mcp"
    }
  }
}
```

### 3. Setup Webots
- Open Webots R2023b+
- Load a world with YouBot robot
- Set controller to `youbot_mcp_controller`
- Start simulation

### 4. Use with Claude
"Check if the robot is connected, then start autonomous mode"

## Requirements

- Webots R2023b or later
- Python 3.10+
- macOS/Linux (Windows partial support)

## Tool Categories

| Category | Tools |
|----------|-------|
| System | check_connection, get_status, reload_world |
| Movement | move_base, stop_base, move_forward, rotate |
| Arm | set_arm_height, set_arm_orientation, set_arm_position, reset_arm |
| Gripper | grip, release, set_gripper_gap |
| Perception | get_camera_image, detect_cubes, get_lidar_data |
| Tasks | grasp_cube, deposit_cube, start_autonomous, stop_autonomous |

## License

MIT - See LICENSE file
```

## Step 7: Publish

### To PyPI:
```bash
pip install build twine
python -m build
twine upload dist/*
```

### To MCP Registry:
```bash
# Install publisher
curl -L https://github.com/modelcontextprotocol/registry/releases/latest/download/mcp-publisher_darwin_arm64.tar.gz | tar -xz
sudo mv mcp-publisher /usr/local/bin/

# Login and publish
mcp-publisher login
mcp-publisher publish
```

### Push to GitHub:
```bash
git add .
git commit -m "Initial release v1.0.0"
git push origin main
git tag v1.0.0
git push --tags
```

## Verification Checklist

- [ ] Repository created on GitHub
- [ ] All files copied and tested
- [ ] pyproject.toml valid
- [ ] server.json matches tool list
- [ ] README has installation instructions
- [ ] Package installs via pip
- [ ] MCP server starts correctly
- [ ] Tools work with Claude Desktop
- [ ] Published to PyPI
- [ ] Published to MCP Registry

## Links

- [MCP Protocol](https://modelcontextprotocol.io)
- [MCP Registry](https://registry.modelcontextprotocol.io)
- [Webots Documentation](https://cyberbotics.com/doc)
- [YouBot Reference](https://www.kuka.com/youbot)
