# Environment Configuration

**Feature:** 001-webots-setup
**Last Updated:** 2025-11-18
**Status:** ✅ Validated - Webots R2023b installed and tested

## System Information

### Operating System
```bash
# Run this command and record output
uname -a
```

**Output:**
```
Darwin Luiss-MacBook-Pro-2.local 25.1.0 Darwin Kernel Version 25.1.0: Mon Oct 20 19:34:05 PDT 2025; root:xnu-12377.41.6~2/RELEASE_ARM64_T6041 arm64
```

**OS:** macOS 15.1 (Sequoia)
**Kernel:** Darwin 25.1.0

### Hardware Specifications
- **CPU:** Apple M4 Pro
- **RAM:** 48 GB
- **GPU:** Apple M4 Pro (integrated)
- **Architecture:** ARM64 (Apple Silicon)

## Software Versions

### Webots Installation

**Expected Version:** R2023b ✅

```bash
# Verify Webots version
webots --version
```

**Output:**
```
Webots version: R2023b
File not found: '/Applications/Webots.app/resources/qt_warning_filters.conf'.
```

**Note:** Qt warning is benign - Webots functional.

**Installation Path:**
- **macOS:** `/Applications/Webots.app/`
- **Linux:** `/usr/local/webots/` or `/opt/webots/`

**Actual Path:** `/Applications/Webots.app/` ✅

### Python Environment

**Expected Version:** Python 3.8 or higher ✅

```bash
# Verify Python version
python3 --version

# Verify pip version
pip3 --version
```

**Output:**
```
Python 3.14.0
pip 25.3 from /Users/luisfelipesena/Development/Personal/projeto-final-ia/venv/lib/python3.14/site-packages/pip (python 3.14)
```

**Note:** Python 3.14 exceeds minimum requirement (3.8+)

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

**Virtual Environment Path:** `/Users/luisfelipesena/Development/Personal/projeto-final-ia/venv/`

**Activation Confirmed:** [✅] Yes

### Python Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# List installed packages
pip list
```

**Key Dependencies (verified versions):** ✅
- numpy: 1.26.4
- scipy: 1.16.3
- pytest: 7.4.4
- matplotlib: 3.10.7
- scikit-fuzzy: 0.5.0
- torch: 2.9.1
- opencv-python: 4.11.0.86

**Total Packages Installed:** 130+ (including transitive dependencies)

## Webots Configuration

### World File Validation

```bash
# Check world file exists
ls -lh IA_20252/worlds/IA_20252.wbt
```

**File Size:** 4.3K ✅
**Last Modified:** 2025-11-18 11:19
**Status:** Present and ready for loading

### Controller Path Configuration

**PYTHONPATH Setup:**

```bash
# macOS/Linux - Add to ~/.bashrc or ~/.zshrc
export WEBOTS_HOME="/Applications/Webots.app"
export PYTHONPATH="${PYTHONPATH}:${WEBOTS_HOME}/lib/controller/python"

# Verify PYTHONPATH includes Webots controller module
echo $PYTHONPATH
```

**PYTHONPATH Value:** (empty - not required for Webots R2023b)

**Controller Module Accessible:** [N/A] - Module only available within Webots runtime

**Note:** Webots R2023b automatically provides controller module to Python scripts launched from within the simulator. PYTHONPATH configuration not required for basic operation.

Test Python can import controller:
```python
python3 -c "from controller import Robot; print('Controller module OK')"
```

**Import Test Result:** ⚠️ Not accessible outside Webots (expected behavior)

## Simulation Validation

### World File Loading

1. **Launch Webots:**
   ```bash
   /Applications/Webots.app/Contents/MacOS/webots IA_20252/worlds/IA_20252.wbt
   ```

2. **Observe Startup:**
   - Webots window opens: [✅] Yes - Loaded successfully
   - World loads within 30 seconds: [✅] Yes - ~5 seconds
   - No error messages in console: [⚠️] Warnings only (R2025a/R2023b compatibility - non-critical)

3. **Console Output:**
   ```
   INFO: youbot: Starting controller: /Users/luisfelipesena/.../venv/bin/python3 -u youbot.py
   INFO: supervisor: Starting controller: /Users/luisfelipesena/.../venv/bin/python3 -u supervisor.py
   INFO: 'youbot' controller exited successfully.
   Spawn complete. The supervisor has spawned 15/15 objects (0 failed).
   INFO: 'supervisor' controller exited successfully.
   ```

### Cube Spawning Verification

**Supervisor Execution:**
- Supervisor script starts: [✅] Yes
- Cubes spawn successfully: [✅] Yes - 15/15 (0 failed)
- Number of cubes spawned: **15/15** ✅

**Cube Distribution:**
- Green cubes: [✅] Present (visible in scene)
- Blue cubes: [✅] Present (visible in scene)
- Red cubes: [✅] Present (visible in scene)
- Total: **15** ✅

**Console Confirmation:** `"Spawn complete. The supervisor has spawned 15/15 objects (0 failed)."`

### Arena Inspection

**YouBot Robot:**
- Robot visible in scene: [✅] Yes - Bottom left corner
- Robot at starting position: [✅] Yes

**Deposit Boxes:**
- Green box visible: [✅] Yes - Top center
- Blue box visible: [✅] Yes - Bottom center
- Red box visible: [✅] Yes - Right side

**Obstacles:**
- Wooden boxes present: [✅] Yes - 9 wooden boxes distributed in arena
- Arena boundaries visible: [✅] Yes - Gray grid floor with walls

**Status:** ✅ Manual GUI testing COMPLETE | All validation passed | Issue R2025a/R2023b documented in DECISÃO 010

## Automated Test Results

```bash
# Run validation tests
pytest tests/test_webots_setup.py -v
```

**Test Results:**
```
============================= test session starts ==============================
tests/test_webots_setup.py::TestPythonEnvironment::test_python_version PASSED
tests/test_webots_setup.py::TestPythonEnvironment::test_project_structure PASSED
tests/test_webots_setup.py::TestWebotsInstallation::test_webots_executable_exists SKIPPED
tests/test_webots_setup.py::TestWebotsInstallation::test_webots_version SKIPPED
tests/test_webots_setup.py::TestWorldFileConfiguration::test_world_file_exists PASSED
tests/test_webots_setup.py::TestWorldFileConfiguration::test_supervisor_file_not_modified PASSED
tests/test_webots_setup.py::TestDocumentation::test_setup_documentation_exists PASSED
tests/test_webots_setup.py::TestDocumentation::test_decisions_documented PASSED

=================== 6 passed, 2 skipped, 1 warning in 0.03s ====================
```

**Tests Passed:** 6/8 ✅
**Tests Skipped:** 2/8 (Webots not in PATH - non-critical)
**Tests Failed:** 0/8 ✅

**Conclusion:** All critical validation tests passing. Webots installed and functional.

## Performance Metrics

### Simulation Load Time

**Measurement:**
1. Close Webots completely
2. Start timer
3. Launch: `webots IA_20252/worlds/IA_20252.wbt`
4. Stop timer when world is fully loaded and interactive

**Load Time:** [⏳] Pending manual test (target: <30 seconds)

### Frame Rate

**Measurement:**
- Run simulation for 60 seconds
- Record FPS from Webots status bar

**Average FPS:** [⏳] Pending manual test (target: >30 FPS for smooth operation)

### Memory Usage

```bash
# Check Webots memory usage after world loads
ps aux | grep webots
```

**Webots Memory Usage:** [⏳] Pending manual test

## Troubleshooting Log

**Issues Encountered:**
- Qt warning on Webots startup (`qt_warning_filters.conf` not found) - benign, does not affect functionality

**Solutions Applied:**
- No action required - warning does not impact Webots operation

**References to Webots Documentation:**
- Installation guide: INSTALACAO_WEBOTS.md
- Validation script: scripts/validate_phase1.sh

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

## Next Steps

### Immediate (Complete Phase 1.1)
- [⏳] Test world file in Webots GUI: `/Applications/Webots.app/Contents/MacOS/webots IA_20252/worlds/IA_20252.wbt`
- [⏳] Verify 15 cubes spawn correctly (console message: "Supervisor: spawned 15 cubes")
- [⏳] Record performance metrics (load time, FPS, memory)
- [ ] Commit final environment.md updates
- [ ] Merge PR 001-webots-setup → main

### Future (Phase 1.2)
After manual validation complete:
- [ ] Proceed to Phase 2: Sensor exploration (LIDAR, camera)
- [ ] Use SpecKit: `/speckit.specify` for 002-sensor-exploration
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

**Document Status:** ✅ Automated validation complete | ⏳ Manual GUI testing pending
**Last Validation:** 2025-11-18, 21:12 UTC
**Next Action:** Test world file in Webots GUI to verify cube spawning and arena setup
