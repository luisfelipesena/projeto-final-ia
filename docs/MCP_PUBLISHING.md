# YouBot MCP - Publishing Guide

## Overview

The YouBot MCP server enables LLM control of the KUKA YouBot robot in Webots simulation. This guide covers how to publish the MCP server for public use.

## MCP Server Structure

```
youbot_mcp/
├── youbot_mcp_server.py      # Main MCP server (FastMCP)
├── youbot_mcp_controller.py  # Webots controller (copy to IA_20252/controllers/)
├── test_runner.py            # Autonomous test runner
└── data/youbot/              # IPC files (commands.json, status.json)
```

## Publishing Checklist

### 1. Package Structure

Create a proper Python package:

```
youbot-mcp/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── youbot_mcp/
│       ├── __init__.py
│       ├── server.py           # youbot_mcp_server.py
│       ├── controller.py       # youbot_mcp_controller.py
│       └── test_runner.py
└── examples/
    └── claude_desktop_config.json
```

### 2. pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "youbot-mcp"
version = "1.0.0"
description = "MCP server for KUKA YouBot control in Webots simulation"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "UFBA MATA64", email = "example@ufba.br" }
]
keywords = ["mcp", "webots", "robotics", "youbot", "simulation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "mcp>=1.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
controller = [
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "torch>=2.0.0",
    "scikit-fuzzy>=0.4.2",
]
dev = [
    "pytest>=7.0.0",
    "mypy>=1.0.0",
]

[project.scripts]
youbot-mcp = "youbot_mcp.server:main"

[project.urls]
Homepage = "https://github.com/yourusername/youbot-mcp"
Documentation = "https://github.com/yourusername/youbot-mcp#readme"
Repository = "https://github.com/yourusername/youbot-mcp"
```

### 3. README.md for Publishing

```markdown
# YouBot MCP

MCP (Model Context Protocol) server for controlling KUKA YouBot in Webots simulation.

## Features

- Omnidirectional base control (mecanum wheels)
- 5-DOF arm control with inverse kinematics
- Gripper manipulation with object detection
- Camera and LIDAR perception
- High-level task execution (grasp, deposit, autonomous mode)

## Installation

```bash
pip install youbot-mcp
```

## Usage with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "youbot": {
      "command": "youbot-mcp",
      "env": {}
    }
  }
}
```

## Requirements

- Webots R2023b or later
- Python 3.10+
- macOS/Linux (Windows untested)

## Available Tools

| Tool | Description |
|------|-------------|
| youbot_check_connection | Verify Webots is running |
| youbot_get_status | Get robot state |
| youbot_move_base | Set velocities |
| youbot_stop_base | Stop movement |
| youbot_set_arm_height | Arm preset positions |
| youbot_grip/release | Gripper control |
| youbot_detect_cubes | HSV cube detection |
| youbot_grasp_cube | Complete grasp sequence |
| youbot_start_autonomous | Begin autonomous task |
```

### 4. Publishing to PyPI

```bash
# Build
pip install build
python -m build

# Upload to TestPyPI first
pip install twine
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ youbot-mcp

# Upload to PyPI
twine upload dist/*
```

### 5. Publishing to npm (Alternative)

For Node.js ecosystem:

```bash
# Create wrapper
npm init
# Add bin entry pointing to Python script
npm publish
```

## MCP Server Quality Checklist

Based on MCP best practices:

### Strategic Design
- [x] Tools enable complete workflows (grasp_cube, deposit_cube)
- [x] Tool names reflect natural task subdivisions
- [x] Response formats optimize for agent context (markdown/json)
- [x] Human-readable identifiers used
- [x] Error messages guide agents toward correct usage

### Implementation Quality
- [x] All tools have descriptive names and documentation
- [x] Server name follows format: `youbot_mcp`
- [x] All network operations use async/await
- [x] Common functionality extracted into reusable functions
- [x] Error messages are clear and actionable

### Tool Configuration
- [x] All tools implement 'name' and 'annotations'
- [x] Annotations correctly set (readOnlyHint, destructiveHint, etc.)
- [x] All tools use Pydantic BaseModel for input validation
- [x] All Pydantic Fields have descriptions and constraints
- [x] Comprehensive docstrings with input/output types

### Code Quality
- [x] Proper imports (Pydantic, FastMCP)
- [x] CHARACTER_LIMIT defined (25000)
- [x] Async functions properly defined
- [x] Type hints throughout
- [x] Constants at module level (UPPER_CASE)

## Integration with Webots

The MCP server communicates with Webots via JSON files:

```
MCP Server <---> commands.json <---> Webots Controller
            <--- status.json   <---
            <--- camera_image.jpg <--
```

### Webots Controller Setup

1. Copy controller to Webots:
   ```bash
   cp youbot_mcp_controller.py IA_20252/controllers/youbot_mcp_controller/
   ```

2. In Webots, set YouBot controller to `youbot_mcp_controller`

3. Start simulation

### macOS Permissions

For `youbot_reload_world` to work:
- System Preferences → Privacy & Security → Accessibility
- Add Terminal/iTerm

## License

MIT License - see LICENSE file
