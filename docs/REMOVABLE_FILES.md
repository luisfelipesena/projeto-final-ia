# Removable Files After MCP Testing

## Overview

This document lists files that can be safely removed after MCP testing is complete. These are temporary files, test artifacts, and development-only resources.

## Removable After Testing

### MCP Test Artifacts
```
youbot_mcp/data/youbot/
├── camera_image.jpg          # Captured during tests
├── commands.json             # Last command sent
├── status.json               # Last status received
├── grasp_*.jpg               # Debug screenshots from grasp sequence
├── grasp_log.txt             # Grasp attempt logs
├── test_report_*.json        # Test run reports
├── image.png                 # Debug images
└── image copy.png            # Debug images
```

### Training Data (After Model Trained)
```
data/lidar/
├── synthetic_training.json   # 7.82 MB - synthetic training data
└── synthetic_large.json      # 18.84 MB - extended training data
```

### Development Artifacts
```
youbot_mcp/
├── youbot_mcp.log            # Server logs
└── __pycache__/              # Python cache
```

## Keep (Required for Operation)

### Models
```
models/
├── lidar_mlp.pth             # Trained LIDAR model (REQUIRED)
└── lidar_mlp_history.json    # Training history (optional, for reference)
```

### Controllers
```
IA_20252/controllers/youbot_mcp_controller/
└── youbot_mcp_controller.py  # Webots controller (REQUIRED)
```

### MCP Server
```
youbot_mcp/
├── youbot_mcp_server.py      # MCP server (REQUIRED)
├── youbot_mcp_controller.py  # Source controller
└── test_runner.py            # Test utility (optional)
```

### Documentation
```
docs/
├── MCP_USAGE.md              # Usage guide (REQUIRED)
├── MCP_PUBLISHING.md         # Publishing guide
├── Final Project.pdf         # Project requirements
├── new_research.md           # Research references
└── REFERENCIAS.md            # Bibliography
```

## Cleanup Commands

### Remove Test Artifacts Only
```bash
rm -f youbot_mcp/data/youbot/camera_image.jpg
rm -f youbot_mcp/data/youbot/commands.json
rm -f youbot_mcp/data/youbot/status.json
rm -f youbot_mcp/data/youbot/grasp_*.jpg
rm -f youbot_mcp/data/youbot/grasp_log.txt
rm -f youbot_mcp/data/youbot/test_report_*.json
rm -f youbot_mcp/data/youbot/image*.png
rm -f youbot_mcp/youbot_mcp.log
rm -rf youbot_mcp/__pycache__
```

### Remove Training Data (After Model Verified)
```bash
rm -f data/lidar/synthetic_training.json
rm -f data/lidar/synthetic_large.json
```

### Full Cleanup Script
```bash
#!/bin/bash
# cleanup_test_artifacts.sh

echo "Cleaning MCP test artifacts..."

# MCP data files
rm -f youbot_mcp/data/youbot/camera_image.jpg
rm -f youbot_mcp/data/youbot/commands.json
rm -f youbot_mcp/data/youbot/status.json
rm -f youbot_mcp/data/youbot/grasp_*.jpg
rm -f youbot_mcp/data/youbot/grasp_log.txt
rm -f youbot_mcp/data/youbot/test_report_*.json
rm -f youbot_mcp/data/youbot/image*.png
rm -f youbot_mcp/data/youbot/nav_debug.log

# Logs and cache
rm -f youbot_mcp/youbot_mcp.log
rm -rf youbot_mcp/__pycache__
rm -rf IA_20252/controllers/youbot_mcp_controller/__pycache__

# Python cache in all subdirectories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "Done. Kept: models/lidar_mlp.pth, MCP server files, documentation"
```

## Size Summary

| Category | Approx Size | Removable? |
|----------|-------------|------------|
| Test images (grasp_*.jpg) | ~50 KB | Yes |
| Training data (JSON) | ~27 MB | After training |
| Model weights (.pth) | ~500 KB | NO |
| Log files | ~10 KB | Yes |
| __pycache__ | ~100 KB | Yes |

## Notes

1. **Never remove** `models/lidar_mlp.pth` - this is the trained neural network
2. **Keep** `youbot_mcp_server.py` for MCP operation
3. Training data can be regenerated with:
   ```bash
   python scripts/generate_synthetic_lidar.py --samples 2000
   ```
4. Test artifacts are regenerated each time tests run
