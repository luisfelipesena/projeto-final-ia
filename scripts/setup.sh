#!/bin/bash
# Setup script for YouBot MATA64 Project
# Run: chmod +x scripts/setup.sh && ./scripts/setup.sh

set -e

echo "=== YouBot MATA64 Setup ==="

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Project root: $PROJECT_ROOT"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/lidar
mkdir -p data/camera
mkdir -p models
mkdir -p youbot_mcp/data/youbot

# Check Webots installation
if command -v webots &> /dev/null; then
    echo "✓ Webots found: $(which webots)"
else
    echo "⚠ Webots not found in PATH"
    echo "  Install from: https://cyberbotics.com/"
fi

# Check PYTHONPATH for Webots controller
WEBOTS_HOME=${WEBOTS_HOME:-/Applications/Webots.app}
if [ -d "$WEBOTS_HOME" ]; then
    echo "✓ WEBOTS_HOME: $WEBOTS_HOME"
    export PYTHONPATH="$WEBOTS_HOME/lib/controller/python:$PYTHONPATH"
else
    echo "⚠ WEBOTS_HOME not set or invalid"
fi

# Verify key packages
echo ""
echo "=== Package Verification ==="
python3 -c "import numpy; print(f'✓ numpy {numpy.__version__}')" 2>/dev/null || echo "✗ numpy missing"
python3 -c "import cv2; print(f'✓ opencv {cv2.__version__}')" 2>/dev/null || echo "✗ opencv-python missing"
python3 -c "import torch; print(f'✓ torch {torch.__version__}')" 2>/dev/null || echo "✗ torch missing"
python3 -c "import skfuzzy; print('✓ scikit-fuzzy')" 2>/dev/null || echo "✗ scikit-fuzzy missing"
python3 -c "import mcp; print('✓ mcp')" 2>/dev/null || echo "✗ mcp missing"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run data collection (in Webots):"
echo "  Set youbot controller to: collect_lidar_data"
echo ""
echo "To train LIDAR model:"
echo "  python scripts/train_lidar_mlp.py"
echo ""
echo "To start MCP server:"
echo "  python youbot_mcp/youbot_mcp_server.py"
