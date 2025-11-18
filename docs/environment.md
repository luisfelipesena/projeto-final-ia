# Environment Configuration

**Feature:** 001-webots-setup
**Last Updated:** 2025-11-18
**Status:** Template - to be filled after manual Webots installation

## System Information

### Operating System
```bash
# Run this command and record output
uname -a
```

**Output:**
```
[TO BE FILLED AFTER INSTALLATION]
```

### Hardware Specifications
- **CPU:** [TO BE FILLED]
- **RAM:** [TO BE FILLED]
- **GPU:** [TO BE FILLED]
- **Architecture:** [TO BE FILLED] (Intel/Apple Silicon/x86_64/ARM)

## Software Versions

### Webots Installation

**Expected Version:** R2023b

```bash
# Verify Webots version
webots --version
```

**Output:**
```
[TO BE FILLED AFTER INSTALLATION]
```

**Installation Path:**
- **macOS:** `/Applications/Webots.app/`
- **Linux:** `/usr/local/webots/` or `/opt/webots/`

**Actual Path:** [TO BE FILLED]

### Python Environment

**Expected Version:** Python 3.8 or higher

```bash
# Verify Python version
python3 --version

# Verify pip version
pip3 --version
```

**Output:**
```
[TO BE FILLED]
```

### Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Verify virtual environment Python
which python
python --version
```

**Virtual Environment Path:** [TO BE FILLED]

**Activation Confirmed:** [ ] Yes / [ ] No

### Python Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# List installed packages
pip list
```

**Key Dependencies (verify versions):**
- numpy: [TO BE FILLED]
- scipy: [TO BE FILLED]
- pytest: [TO BE FILLED]
- matplotlib: [TO BE FILLED]
- scikit-fuzzy: [TO BE FILLED]
- torch: [TO BE FILLED]
- opencv-python: [TO BE FILLED]

## Webots Configuration

### World File Validation

```bash
# Check world file exists
ls -lh IA_20252/worlds/IA_20252.wbt
```

**File Size:** [TO BE FILLED]
**Last Modified:** [TO BE FILLED]

### Controller Path Configuration

**PYTHONPATH Setup:**

```bash
# macOS/Linux - Add to ~/.bashrc or ~/.zshrc
export WEBOTS_HOME="/Applications/Webots.app"
export PYTHONPATH="${PYTHONPATH}:${WEBOTS_HOME}/lib/controller/python"

# Verify PYTHONPATH includes Webots controller module
echo $PYTHONPATH
```

**PYTHONPATH Value:** [TO BE FILLED]

**Controller Module Accessible:** [ ] Yes / [ ] No

Test Python can import controller:
```python
python3 -c "from controller import Robot; print('Controller module OK')"
```

**Import Test Result:** [TO BE FILLED]

## Simulation Validation

### World File Loading

1. **Launch Webots:**
   ```bash
   webots IA_20252/worlds/IA_20252.wbt
   ```

2. **Observe Startup:**
   - Webots window opens: [ ] Yes / [ ] No
   - World loads within 30 seconds: [ ] Yes / [ ] No
   - No error messages in console: [ ] Yes / [ ] No

3. **Console Output:**
   ```
   [PASTE WEBOTS CONSOLE OUTPUT HERE]
   ```

### Cube Spawning Verification

**Supervisor Execution:**
- Supervisor script starts: [ ] Yes / [ ] No
- Cubes spawn successfully: [ ] Yes / [ ] No
- Number of cubes spawned: [TO BE FILLED] (expected: 15)

**Cube Distribution:**
- Green cubes: [TO BE FILLED]
- Blue cubes: [TO BE FILLED]
- Red cubes: [TO BE FILLED]
- Total: [TO BE FILLED]

### Arena Inspection

**YouBot Robot:**
- Robot visible in scene: [ ] Yes / [ ] No
- Robot at starting position: [ ] Yes / [ ] No

**Deposit Boxes:**
- Green box visible: [ ] Yes / [ ] No
- Blue box visible: [ ] Yes / [ ] No
- Red box visible: [ ] Yes / [ ] No

**Obstacles:**
- Wooden boxes present: [ ] Yes / [ ] No
- Arena boundaries visible: [ ] Yes / [ ] No

## Automated Test Results

```bash
# Run validation tests
pytest tests/test_webots_setup.py -v
```

**Test Results:**
```
[PASTE PYTEST OUTPUT HERE AFTER RUNNING]
```

**Expected:** 4/4 tests pass when environment is correctly configured

**Tests Passed:** [TO BE FILLED] / 4

## Performance Metrics

### Simulation Load Time

**Measurement:**
1. Close Webots completely
2. Start timer
3. Launch: `webots IA_20252/worlds/IA_20252.wbt`
4. Stop timer when world is fully loaded and interactive

**Load Time:** [TO BE FILLED] seconds (target: <30 seconds)

### Frame Rate

**Measurement:**
- Run simulation for 60 seconds
- Record FPS from Webots status bar

**Average FPS:** [TO BE FILLED] (target: >30 FPS for smooth operation)

### Memory Usage

```bash
# Check Webots memory usage after world loads
ps aux | grep webots
```

**Webots Memory Usage:** [TO BE FILLED] MB

## Troubleshooting Log

**Issues Encountered:** [TO BE FILLED]

**Solutions Applied:** [TO BE FILLED]

**References to Webots Documentation:** [TO BE FILLED]

## Reproducibility Validation

**Second Machine Test:** [ ] Pending / [ ] Pass / [ ] Fail

If testing on a second machine:
- **Machine 2 OS:** [TO BE FILLED]
- **Machine 2 Hardware:** [TO BE FILLED]
- **Setup Time:** [TO BE FILLED] minutes
- **Issues Encountered:** [TO BE FILLED]

## Configuration Files

**Key Files to Backup:**
- `IA_20252/worlds/IA_20252.wbt` (original world file)
- `IA_20252/controllers/supervisor/supervisor.py` (IMMUTABLE - backup to verify no changes)
- `requirements.txt` (Python dependencies)
- `.gitignore` (version control configuration)

## Next Steps (Phase 1.2)

After environment validation is complete:
- [ ] Proceed to Phase 1.2: Exploration of basic controls
- [ ] Test YouBot movements (forward, backward, strafe, rotate)
- [ ] Test arm commands (set_height, set_orientation)
- [ ] Test gripper (grip, release)
- [ ] Create test script: `tests/test_basic_controls.py`

## References

- **DECISÃO 005:** Webots R2023b installation method (DECISIONS.md)
- **DECISÃO 006:** Python integration strategy (DECISIONS.md)
- **Setup Guide:** specs/001-webots-setup/quickstart.md
- **Test Specifications:** specs/001-webots-setup/contracts/test_specifications.md

---

**Template Status:** Ready for manual completion
**Next Action:** Install Webots R2023b following quickstart.md, then fill this template
