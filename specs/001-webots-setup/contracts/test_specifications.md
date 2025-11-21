# Test Specifications: Webots Environment Setup

**Feature**: 001-webots-setup
**Date**: 2025-11-18
**Phase**: Phase 1 - Test Contracts

---

## Overview

This document defines the test contracts for Webots environment setup validation. Each test specification includes preconditions, execution steps, expected outcomes, and acceptance criteria mapped to functional requirements.

---

## Test Suite Structure

```
tests/
├── test_webots_setup.py          # Main validation suite (this document)
├── fixtures/
│   ├── conftest.py                # pytest fixtures
│   └── mock_data.py               # Sample sensor data for unit tests
└── integration/                   # Future: Full simulation tests (Phase 2+)
```

---

## Test Category 1: Installation Validation

### Test 1.1: test_webots_executable_exists

**Functional Requirement**: FR-001 (System MUST install Webots R2023b)

**Category**: Installation (fast)

**Markers**: `@pytest.mark.fast`, `@pytest.mark.requires_webots`

**Preconditions**:
- None (this is the first test)

**Test Steps**:
1. Execute shell command: `webots --version`
2. Capture stdout and return code

**Expected Outcomes**:
- Return code: 0 (success)
- stdout contains: "Webots R2023b" or "R2023b"
- Command completes in <2 seconds

**Acceptance Criteria**:
- [x] Webots executable is in system PATH
- [x] Version string matches R2023b exactly
- [x] No error messages in stderr

**Error Scenarios**:
- **Webots not installed**: Return code 127 ("command not found")
- **Wrong version**: stdout contains R2024a or other version
- **Corrupted installation**: Return code non-zero with error message

**Implementation Signature**:
```python
def test_webots_executable_exists() -> None:
    """Verify Webots R2023b is installed and accessible."""
```

---

### Test 1.2: test_python_version_compatible

**Functional Requirement**: FR-002 (System MUST verify Python 3.8+)

**Category**: Environment (fast)

**Markers**: `@pytest.mark.fast`

**Preconditions**:
- System Python interpreter available

**Test Steps**:
1. Get system Python version via `sys.version_info`
2. Compare major.minor version to (3, 8)

**Expected Outcomes**:
- `sys.version_info.major` >= 3
- `sys.version_info.minor` >= 8 (if major == 3)
- OR major >= 4 (future-proof for Python 4+)

**Acceptance Criteria**:
- [x] Python version is 3.8.0 or higher
- [x] Version check completes in <0.1 seconds
- [x] Clear error message if version too old

**Error Scenarios**:
- **Python 2.x**: Major version 2 → fail with "Python 3.8+ required"
- **Python 3.7 or older**: Minor < 8 → fail with version number
- **No Python**: ImportError (should not occur if pytest runs)

**Implementation Signature**:
```python
def test_python_version_compatible() -> None:
    """Verify Python version is 3.8 or higher."""
```

---

### Test 1.3: test_world_file_exists

**Functional Requirement**: FR-005 (System MUST open IA_20252.wbt)

**Category**: Installation (fast)

**Markers**: `@pytest.mark.fast`

**Preconditions**:
- Project directory structure exists

**Test Steps**:
1. Check file existence: `IA_20252/worlds/IA_20252.wbt`
2. Verify file is readable
3. Check file size (should be >10KB)

**Expected Outcomes**:
- File exists at specified path
- File has read permissions
- File size between 10KB and 10MB (plausible for .wbt)

**Acceptance Criteria**:
- [x] World file exists at IA_20252/worlds/IA_20252.wbt
- [x] File is readable by current user
- [x] File size indicates valid world file (not empty or corrupted)

**Error Scenarios**:
- **File missing**: FileNotFoundError with helpful message
- **No permissions**: PermissionError
- **Empty file**: File size 0 bytes → fail

**Implementation Signature**:
```python
def test_world_file_exists() -> None:
    """Verify IA_20252.wbt world file is present."""
```

---

### Test 1.4: test_virtual_environment_configured

**Functional Requirement**: FR-003, FR-004 (System MUST create venv and install dependencies)

**Category**: Environment (medium speed, depends on pip)

**Markers**: `@pytest.mark.medium`

**Preconditions**:
- Python 3.8+ available
- requirements.txt exists in project root

**Test Steps**:
1. Check venv directory exists: `venv/`
2. Verify venv Python executable: `venv/bin/python`
3. Run `venv/bin/pip list` and capture output
4. Check for required packages: pytest, numpy, scipy

**Expected Outcomes**:
- `venv/` directory exists
- `venv/bin/python` is executable
- `pip list` includes pytest, numpy, scipy (minimum)

**Acceptance Criteria**:
- [x] Virtual environment created successfully
- [x] Dependencies from requirements.txt installed
- [x] pytest is importable from venv

**Error Scenarios**:
- **No venv**: Directory doesn't exist → fail with setup instructions
- **Empty venv**: No packages installed → fail
- **Partial install**: Some packages missing → list missing packages

**Implementation Signature**:
```python
def test_virtual_environment_configured() -> None:
    """Verify venv exists with dependencies installed."""
```

---

## Test Category 2: Supervisor Immutability (Constitution Validation)

### Test 2.1: test_supervisor_file_not_modified

**Functional Requirement**: FR-011 (System MUST execute supervisor WITHOUT modifications)

**Category**: Constitution compliance (fast)

**Markers**: `@pytest.mark.fast`, `@pytest.mark.constitution`

**Preconditions**:
- Baseline hash of supervisor.py exists (or use git)

**Test Steps**:
1. Compute SHA256 hash of `IA_20252/controllers/supervisor/supervisor.py`
2. Compare to known baseline hash (from git)
3. OR: Check git status for modifications

**Expected Outcomes**:
- File hash matches original
- OR: `git diff` shows no changes to supervisor.py

**Acceptance Criteria**:
- [x] supervisor.py has not been modified
- [x] Test fails if ANY changes detected (blocks accidental edits)

**Error Scenarios**:
- **Modified file**: Hash mismatch → FAIL with error message "CONSTITUTION VIOLATION: supervisor.py modified"
- **File missing**: FileNotFoundError → FAIL

**Implementation Signature**:
```python
@pytest.mark.constitution
def test_supervisor_file_not_modified() -> None:
    """Verify supervisor.py has not been modified (Principle V)."""
```

---

## Test Category 3: Simulation Behavior (Integration Tests)

### Test 3.1: test_simulation_loads_without_errors

**Functional Requirement**: FR-005 (World MUST load within 30 seconds)

**Category**: Integration (slow, requires Webots)

**Markers**: `@pytest.mark.slow`, `@pytest.mark.requires_webots`, `@pytest.mark.integration`

**Preconditions**:
- Webots R2023b installed
- World file exists
- Headless/batch mode available (for CI/CD)

**Test Steps**:
1. Launch Webots in batch mode: `webots --batch --mode=fast IA_20252/worlds/IA_20252.wbt`
2. Monitor process for 30 seconds
3. Check for crashes (process exit)
4. Capture stderr for errors

**Expected Outcomes**:
- Process stays alive for 30+ seconds
- No error messages in stderr
- No segmentation faults or crashes

**Acceptance Criteria**:
- [x] World loads successfully in batch mode
- [x] Load time < 30 seconds (SC-002)
- [x] No console errors during load

**Error Scenarios**:
- **Crash on load**: Process exits with non-zero code → FAIL
- **Timeout**: Load takes >30s → FAIL (performance issue)
- **Errors in console**: stderr contains "ERROR" → FAIL

**Implementation Signature**:
```python
@pytest.mark.slow
@pytest.mark.requires_webots
def test_simulation_loads_without_errors() -> None:
    """Verify world loads successfully in under 30 seconds."""
```

---

### Test 3.2: test_cube_spawn_count (Future: Phase 2)

**Functional Requirement**: FR-006 (System MUST spawn 15 cubes)

**Category**: Integration (slow, requires simulation run)

**Markers**: `@pytest.mark.slow`, `@pytest.mark.requires_webots`, `@pytest.mark.phase2`

**Preconditions**:
- Simulation running
- Supervisor executed successfully
- Controller can access supervisor data (via receivers/emitters)

**Test Steps**:
1. Launch simulation with external controller
2. Wait for supervisor to complete spawn (10 simulation seconds)
3. Query supervisor for cube count
4. OR: Use vision to count cubes in scene

**Expected Outcomes**:
- Cube count >= 14 (allowing for rare collision failures per edge case)
- Ideally: cube count == 15 (95% of runs per SC-004)

**Acceptance Criteria**:
- [x] At least 14 cubes spawned in 95% of runs
- [x] Cube colors are green, blue, or red (no other colors)

**Error Scenarios**:
- **< 14 cubes**: Supervisor failed to spawn → FAIL
- **Wrong colors**: Non-RGB cubes → FAIL (supervisor modified?)

**Implementation Signature**:
```python
@pytest.mark.slow
@pytest.mark.requires_webots
@pytest.mark.skip(reason="Phase 2: Requires controller implementation")
def test_cube_spawn_count() -> None:
    """Verify supervisor spawns 14-15 cubes successfully."""
```

---

## Test Category 4: Sensor Validation (Phase 2 - Deferred)

### Test 4.1: test_lidar_sensor_available

**Functional Requirement**: FR-009 (System MUST provide LIDAR data)

**Category**: Functional (medium, requires controller)

**Markers**: `@pytest.mark.medium`, `@pytest.mark.requires_webots`, `@pytest.mark.phase2`

**Preconditions**:
- Simulation running
- Controller with LIDAR access implemented

**Test Steps**:
1. Enable LIDAR sensor: `lidar.enable(timestep)`
2. Wait 10 simulation steps
3. Query range data: `lidar.getRangeImage()`
4. Validate array size and content

**Expected Outcomes**:
- Returns array of exactly 512 floats
- Finite values in range [0.01, 10.0] meters
- At least 10% of rays hit obstacles (not all infinite)

**Acceptance Criteria**:
- [x] LIDAR returns valid 512-point array (SC-005)
- [x] Data available within 1 second of enable
- [x] Range values physically plausible

**Implementation Signature**:
```python
@pytest.mark.medium
@pytest.mark.requires_webots
@pytest.mark.skip(reason="Phase 2: Requires sensor controller implementation")
def test_lidar_sensor_available() -> None:
    """Verify LIDAR sensor returns valid 512-point data."""
```

---

### Test 4.2: test_camera_sensor_available

**Functional Requirement**: FR-010 (System MUST provide camera data)

**Category**: Functional (medium, requires controller)

**Markers**: `@pytest.mark.medium`, `@pytest.mark.requires_webots`, `@pytest.mark.phase2`

**Preconditions**:
- Simulation running
- Controller with camera access implemented

**Test Steps**:
1. Enable camera: `camera.enable(timestep)`
2. Wait 20 simulation steps (cameras need more time)
3. Query image: `camera.getImage()`
4. Validate format and content

**Expected Outcomes**:
- Returns bytes of size 128*128*4 = 65536 (BGRA)
- Pixel values in range [0, 255]
- Image not all black (simulation rendering)

**Acceptance Criteria**:
- [x] Camera returns valid 128x128 BGRA image (SC-006)
- [x] Data available within 1 second of enable
- [x] Colored cubes visible in image

**Implementation Signature**:
```python
@pytest.mark.medium
@pytest.mark.requires_webots
@pytest.mark.skip(reason="Phase 2: Requires sensor controller implementation")
def test_camera_sensor_available() -> None:
    """Verify camera sensor returns valid 128x128 BGRA data."""
```

---

## Test Fixtures

### Fixture: webots_process

**Purpose**: Manage Webots batch mode lifecycle for integration tests

**Scope**: Function (per-test)

**Setup**:
```python
@pytest.fixture
def webots_process():
    """Launch Webots in headless batch mode."""
    proc = subprocess.Popen([
        'webots',
        '--batch',
        '--mode=fast',
        'IA_20252/worlds/IA_20252.wbt'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for startup
    time.sleep(5)

    yield proc

    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)
```

**Teardown**: Kills Webots process, ensures no orphans

---

### Fixture: temp_venv

**Purpose**: Create temporary venv for testing venv creation

**Scope**: Function

**Setup**:
```python
@pytest.fixture
def temp_venv(tmp_path):
    """Create temporary virtual environment for testing."""
    venv_path = tmp_path / "test_venv"
    subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
    yield venv_path
    # Auto cleanup via tmp_path
```

---

## Test Execution Strategy

### Local Development

```bash
# Fast tests only (installation checks)
pytest tests/test_webots_setup.py -m fast -v

# All Phase 1.1 tests (exclude Phase 2)
pytest tests/test_webots_setup.py -m "not phase2" -v

# Full suite (including slow integration tests)
pytest tests/test_webots_setup.py -v

# With coverage
pytest tests/test_webots_setup.py --cov=controllers --cov-report=html
```

### CI/CD Pipeline

```bash
# GitHub Actions workflow
pytest tests/test_webots_setup.py -m "fast and not requires_webots" -v  # Stage 1
pytest tests/test_webots_setup.py -m "requires_webots and not slow" -v  # Stage 2
pytest tests/test_webots_setup.py -m slow -v                             # Stage 3
```

---

## Test Coverage Requirements

**Constitution Principle IV (Qualidade Senior)**: Target >80% coverage

**Coverage Scope**:
- ✅ Installation validation: 100% (all tests implemented)
- ⏸️ Sensor validation: 0% (Phase 2 implementation)
- ⏸️ Controller logic: 0% (Phase 2+ implementation)

**Phase 1.1 Coverage Target**: 100% of setup code (currently: validation scripts only)

---

## Test Maintenance

**Adding New Tests**:
1. Follow naming convention: `test_<feature>_<behavior>`
2. Add appropriate markers (`@pytest.mark.fast`, `@pytest.mark.slow`, etc.)
3. Update this document with test specification
4. Map to functional requirement (FR-XXX)

**Updating Tests**:
1. Document reason for change in git commit
2. Update test specification in this document
3. Verify all dependent tests still pass
4. Update DECISIONS.md if test strategy changes

---

## Success Criteria Mapping

| Success Criterion | Test Coverage |
|------------------|---------------|
| SC-001: Setup <30 min | Manual (time user experience) |
| SC-002: Load <30s | test_simulation_loads_without_errors |
| SC-003: 100% pass rate | pytest summary report |
| SC-004: 14+/15 cubes | test_cube_spawn_count (Phase 2) |
| SC-005: LIDAR <1s | test_lidar_sensor_available (Phase 2) |
| SC-006: Camera <1s | test_camera_sensor_available (Phase 2) |
| SC-007: Documentation | Manual review of README.md |
| SC-008: Reproducibility | Run on 2+ machines (manual) |

---

**Compliance Notes**:
- ✅ All tests map to functional requirements
- ✅ Constitution compliance test included (supervisor immutability)
- ✅ Phase 2 tests marked with `@pytest.mark.skip` (deferred)
- ✅ Performance targets encoded as test assertions (<30s load)
