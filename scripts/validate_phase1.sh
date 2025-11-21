#!/bin/bash
# Phase 1.1 Validation Script
# Validates Webots R2023b installation and environment setup

set -e  # Exit on error

PROJECT_ROOT="/Users/luisfelipesena/Development/Personal/projeto-final-ia"
WORLD_FILE="$PROJECT_ROOT/IA_20252/worlds/IA_20252.wbt"

echo "========================================="
echo "Phase 1.1 Validation - Webots Setup"
echo "========================================="
echo ""

# Check 1: Webots Installation
echo "✓ Check 1: Webots Installation"
if command -v webots &> /dev/null; then
    WEBOTS_VERSION=$(webots --version 2>&1 | head -1)
    echo "  ✅ Webots found: $WEBOTS_VERSION"

    if [[ "$WEBOTS_VERSION" == *"R2023b"* ]]; then
        echo "  ✅ Correct version: R2023b"
    else
        echo "  ❌ Wrong version! Expected R2023b, got: $WEBOTS_VERSION"
        exit 1
    fi
else
    # Try macOS path
    if [ -f "/Applications/Webots.app/Contents/MacOS/webots" ]; then
        WEBOTS_VERSION=$(/Applications/Webots.app/Contents/MacOS/webots --version 2>&1 | head -1)
        echo "  ✅ Webots found at /Applications/Webots.app"
        echo "  ⚠️  Not in PATH, add: export PATH=\"/Applications/Webots.app:\$PATH\""
    else
        echo "  ❌ Webots not found!"
        echo "     Install from: https://github.com/cyberbotics/webots/releases/tag/R2023b"
        exit 1
    fi
fi
echo ""

# Check 2: Python Version
echo "✓ Check 2: Python Version"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "  Found: Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    echo "  ✅ Python 3.8+ requirement met"
else
    echo "  ❌ Python 3.8+ required, found: $PYTHON_VERSION"
    exit 1
fi
echo ""

# Check 3: Project Structure
echo "✓ Check 3: Project Structure"
REQUIRED_DIRS=(
    "$PROJECT_ROOT/IA_20252"
    "$PROJECT_ROOT/IA_20252/controllers"
    "$PROJECT_ROOT/IA_20252/worlds"
    "$PROJECT_ROOT/tests"
    "$PROJECT_ROOT/docs"
    "$PROJECT_ROOT/specs/001-webots-setup"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $(basename $dir)/ exists"
    else
        echo "  ❌ Missing: $dir"
        exit 1
    fi
done
echo ""

# Check 4: World File
echo "✓ Check 4: World File"
if [ -f "$WORLD_FILE" ]; then
    FILE_SIZE=$(du -h "$WORLD_FILE" | awk '{print $1}')
    echo "  ✅ IA_20252.wbt exists ($FILE_SIZE)"
else
    echo "  ❌ World file not found: $WORLD_FILE"
    exit 1
fi
echo ""

# Check 5: Supervisor (immutability check)
echo "✓ Check 5: Supervisor File"
SUPERVISOR="$PROJECT_ROOT/IA_20252/controllers/supervisor/supervisor.py"
if [ -f "$SUPERVISOR" ]; then
    if grep -q "spawn" "$SUPERVISOR"; then
        echo "  ✅ supervisor.py exists with spawn logic"
    else
        echo "  ⚠️  supervisor.py may be modified (no 'spawn' found)"
    fi
else
    echo "  ❌ supervisor.py not found!"
    exit 1
fi
echo ""

# Check 6: Documentation
echo "✓ Check 6: Documentation"
REQUIRED_DOCS=(
    "$PROJECT_ROOT/README.md"
    "$PROJECT_ROOT/DECISIONS.md"
    "$PROJECT_ROOT/specs/001-webots-setup/spec.md"
    "$PROJECT_ROOT/specs/001-webots-setup/quickstart.md"
)

for doc in "${REQUIRED_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✅ $(basename $doc) exists"
    else
        echo "  ❌ Missing: $doc"
        exit 1
    fi
done
echo ""

# Check 7: Virtual Environment
echo "✓ Check 7: Virtual Environment"
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "  ✅ venv/ exists"

    if [ -f "$PROJECT_ROOT/venv/bin/pytest" ]; then
        echo "  ✅ pytest installed in venv"
    else
        echo "  ⚠️  pytest not found in venv"
        echo "     Run: source venv/bin/activate && pip install -r requirements.txt"
    fi
else
    echo "  ⚠️  venv/ not created"
    echo "     Run: python3 -m venv venv"
fi
echo ""

# Check 8: Test Suite
echo "✓ Check 8: Test Suite"
if [ -f "$PROJECT_ROOT/tests/test_webots_setup.py" ]; then
    echo "  ✅ test_webots_setup.py exists"

    # Count test functions
    TEST_COUNT=$(grep -c "def test_" "$PROJECT_ROOT/tests/test_webots_setup.py")
    echo "  ✅ $TEST_COUNT test functions defined"
else
    echo "  ❌ test_webots_setup.py not found!"
    exit 1
fi
echo ""

# Summary
echo "========================================="
echo "✅ Phase 1.1 Validation PASSED"
echo "========================================="
echo ""
echo "Next Steps:"
echo "1. Create venv (if not exists): python3 -m venv venv"
echo "2. Activate venv: source venv/bin/activate"
echo "3. Install deps: pip install -r requirements.txt"
echo "4. Run tests: pytest tests/test_webots_setup.py -v"
echo "5. Test world: webots IA_20252/worlds/IA_20252.wbt"
echo ""
echo "After validation: Fill docs/environment.md with actual config"
