# Quick Start: Webots Environment Setup

**Feature**: 001-webots-setup
**Audience**: Developers setting up the YouBot simulation environment
**Time Required**: ~30 minutes (excluding downloads)

---

## Prerequisites

Before starting, ensure you have:
- [ ] macOS (Intel/Apple Silicon) OR Linux Ubuntu 22.04+
- [ ] Internet connection for downloads
- [ ] ~5GB free disk space
- [ ] Admin/sudo privileges for installation
- [ ] Basic terminal/command-line familiarity

---

## Quick Setup (5 Steps)

### Step 1: Install Webots R2023b (10 min)

**macOS**:
```bash
# Download from: https://github.com/cyberbotics/webots/releases/tag/R2023b
# File: webots-R2023b.dmg

# Install:
# 1. Open webots-R2023b.dmg
# 2. Drag Webots to Applications folder
# 3. First launch: Right-click → Open (bypass Gatekeeper)

# Verify installation:
/Applications/Webots.app/webots --version
# Expected output: Webots R2023b
```

**Linux Ubuntu 22.04+**:
```bash
# Download Debian package
wget https://github.com/cyberbotics/webots/releases/download/R2023b/webots_2023b_amd64.deb

# Install (requires sudo)
sudo apt install ./webots_2023b_amd64.deb

# Verify installation
webots --version
# Expected output: Webots R2023b

# Verify graphics drivers
glxinfo | grep "OpenGL version"
# Should show hardware-accelerated OpenGL
```

---

### Step 2: Configure Python 3.8+ (5 min)

```bash
# Check system Python version
python3 --version
# Required: Python 3.8.0 or higher

# If Python 3.8+ not installed:
# macOS: Install from python.org or use Homebrew
# Linux: sudo apt install python3 python3-pip python3-venv

# Create project virtual environment
cd /path/to/projeto-final-ia
python3 -m venv venv

# Activate venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows (if applicable)

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify pytest installed
pytest --version
```

---

### Step 3: Configure PYTHONPATH (2 min)

**macOS**:
```bash
# Add to ~/.zshrc or ~/.bash_profile
echo 'export PYTHONPATH="/Applications/Webots.app/lib/controller/python38:$PYTHONPATH"' >> ~/.zshrc

# Reload shell config
source ~/.zshrc

# Verify
echo $PYTHONPATH
# Should include /Applications/Webots.app/lib/controller/python38
```

**Linux**:
```bash
# Add to ~/.bashrc
echo 'export PYTHONPATH="/usr/local/webots/lib/controller/python38:$PYTHONPATH"' >> ~/.bashrc

# Reload shell config
source ~/.bashrc

# Verify
echo $PYTHONPATH
# Should include /usr/local/webots/lib/controller/python38
```

**Note**: Adjust `python38` to match your system Python version (e.g., `python39`, `python310`, `python311`).

---

### Step 4: Validate World File (2 min)

```bash
# Verify world file exists
ls -lh IA_20252/worlds/IA_20252.wbt
# Expected: File should exist, ~100-500KB

# Verify supervisor exists (DO NOT MODIFY THIS FILE!)
ls -lh IA_20252/controllers/supervisor/supervisor.py
# Expected: File should exist

# Test load world in Webots
# Option A: GUI
webots IA_20252/worlds/IA_20252.wbt

# Option B: Headless (for CI/CD)
webots --batch --mode=fast IA_20252/worlds/IA_20252.wbt &
sleep 10
pkill webots
```

**Expected Result**:
- Webots loads world in <30 seconds
- Arena displays with wooden boxes (obstacles)
- Three colored deposit boxes visible (green, blue, red)
- YouBot robot at starting position
- 15 colored cubes spawn (random colors: green/blue/red)
- No errors in Webots console

---

### Step 5: Run Validation Tests (5 min)

```bash
# Ensure venv is activated
source venv/bin/activate

# Run test suite
pytest tests/test_webots_setup.py -v

# Expected output:
# ===================== test session starts ======================
# collected 4 items
#
# tests/test_webots_setup.py::TestWebotsInstallation::test_webots_executable_exists PASSED
# tests/test_webots_setup.py::TestWebotsInstallation::test_python_version_compatible PASSED
# tests/test_webots_setup.py::TestWebotsInstallation::test_world_file_exists PASSED
# tests/test_webots_setup.py::TestWebotsInstallation::test_virtual_environment_configured PASSED
#
# ===================== 4 passed in 3.24s =======================

# Generate validation report (optional)
pytest tests/test_webots_setup.py --json-report --json-report-file=logs/validation_report.json
```

---

## Verification Checklist

After completing all steps, verify:

**Installation**:
- [ ] `webots --version` returns "Webots R2023b"
- [ ] `python3 --version` returns 3.8.0 or higher
- [ ] `venv/` directory exists with activated environment
- [ ] `pip list` shows pytest, numpy, scipy installed

**World File**:
- [ ] IA_20252.wbt loads without errors
- [ ] 15 cubes visible in simulation
- [ ] YouBot robot visible at start position
- [ ] No console errors during load

**Validation Tests**:
- [ ] All 4 tests pass (100% pass rate)
- [ ] Test execution completes in <5 minutes
- [ ] No import errors or missing modules

---

## Common Issues & Solutions

### Issue: "webots: command not found"

**Cause**: Webots not in system PATH

**Solution**:
```bash
# macOS
alias webots="/Applications/Webots.app/webots"

# Linux - should work automatically after apt install
# If not, add to ~/.bashrc:
export PATH="/usr/local/webots:$PATH"
```

---

### Issue: "No module named 'controller'"

**Cause**: PYTHONPATH not configured correctly

**Solution**:
```bash
# Verify PYTHONPATH includes Webots controller library
echo $PYTHONPATH

# Manually set for current session:
# macOS
export PYTHONPATH="/Applications/Webots.app/lib/controller/python38:$PYTHONPATH"

# Linux
export PYTHONPATH="/usr/local/webots/lib/controller/python38:$PYTHONPATH"

# Add to shell profile for persistence (see Step 3)
```

---

### Issue: "ImportError: cannot import name X"

**Cause**: Dependency version mismatch or missing package

**Solution**:
```bash
# Recreate venv from scratch
rm -rf venv/
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify all packages installed
pip list
```

---

### Issue: Webots crashes or simulation slow

**Cause**: Graphics driver issues or insufficient hardware

**Solution (Linux)**:
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"

# Install NVIDIA drivers (if NVIDIA GPU)
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

# Install AMD drivers (if AMD GPU)
sudo apt install mesa-utils
```

**Solution (macOS)**:
- Older Intel Macs (pre-2015): May have OpenGL compatibility issues
- Solution: Lower graphics quality in Webots preferences
- Preferences → OpenGL → Disable shadows, lower texture quality

---

### Issue: World file fails to load

**Cause**: Corrupted file or wrong Webots version

**Solution**:
```bash
# Verify file integrity
file IA_20252/worlds/IA_20252.wbt
# Should output: "ASCII text" or "XML document"

# Check Webots version matches R2023b
webots --version

# If wrong version, reinstall R2023b specifically
# DO NOT use latest version (R2024a+) - API incompatible
```

---

### Issue: Virtual environment activated but Webots still can't import modules

**Cause**: Webots R2021b+ ignores venv when launched from within it

**Solution**:
```bash
# Launch Webots from OUTSIDE venv:
deactivate  # Exit venv
webots IA_20252/worlds/IA_20252.wbt

# For testing, activate venv separately:
# Terminal 1: Run Webots from system
webots

# Terminal 2: Run tests with venv
source venv/bin/activate
pytest tests/
```

---

## Next Steps

After completing setup:

1. **Document your configuration**:
   ```bash
   # Save environment details
   python3 scripts/capture_environment.py > docs/environment.md
   ```

2. **Update DECISIONS.md**:
   - Add DECISÃO 005: Webots R2023b installation method chosen
   - Add DECISÃO 006: Python-Webots integration strategy
   - Add DECISÃO 007: pytest testing framework
   - Add DECISÃO 008: Sensor validation approach

3. **Mark TODO.md Phase 1.1 complete**:
   - [x] Webots R2023b instalado
   - [x] Python 3.8+ configurado
   - [x] Ambiente validado (testes passando)
   - [x] Sensores funcionando (LIDAR + Camera)

4. **Proceed to Phase 2**: Perception with Neural Networks
   - Specification: `/speckit.specify` for "LIDAR processing with MLP/CNN"
   - Branch: `002-lidar-perception` or similar

---

## Support & Resources

**Official Documentation**:
- Webots R2023b Docs: https://cyberbotics.com/doc/guide/index
- Python Controller API: https://cyberbotics.com/doc/reference/robot

**Project Documentation**:
- Full specification: `specs/001-webots-setup/spec.md`
- Research findings: `specs/001-webots-setup/research.md`
- Data model: `specs/001-webots-setup/data-model.md`
- Implementation tasks: `specs/001-webots-setup/tasks.md` (generated by `/speckit.tasks`)

**Troubleshooting**:
- Constitution: `.specify/memory/constitution.md` (project principles)
- References: `REFERENCIAS.md` (80+ scientific papers)
- Decisions: `DECISIONS.md` (technical decision log)

**Community**:
- Webots GitHub: https://github.com/cyberbotics/webots/issues
- Webots Discord: https://discord.gg/nTWbN9m

---

**Setup Complete! 🎉**

Your Webots environment is now configured and validated. You can begin developing perception and control algorithms for the YouBot autonomous system.
