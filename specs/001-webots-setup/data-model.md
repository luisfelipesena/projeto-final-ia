# Data Model: Webots Environment Setup

**Feature**: 001-webots-setup
**Date**: 2025-11-18
**Phase**: Phase 1 - Data Model Design

---

## Overview

This document defines the data entities, validation rules, and state transitions for the Webots environment setup feature. While this is primarily an infrastructure setup task, several "entities" represent configurations, test results, and environment state.

---

## Entity Definitions

### 1. WebotsInstallation

Represents the installed Webots simulator instance and its configuration.

**Fields**:
- `version: str` - Webots version identifier (e.g., "R2023b")
- `install_path: Path` - Installation directory path
  - macOS: `/Applications/Webots.app`
  - Linux: `/usr/local/webots`
- `executable_path: Path` - Path to webots binary
- `python_controller_path: Path` - Path to Python controller library
- `is_installed: bool` - Whether Webots is properly installed
- `has_valid_version: bool` - Whether version matches R2023b requirement

**Validation Rules**:
- `version` MUST equal "R2023b" (exact match)
- `executable_path` MUST exist and be executable
- `python_controller_path` MUST contain `controller.py` module
- Installation verification via `webots --version` MUST succeed

**State**: Immutable once installed (no state transitions)

---

### 2. PythonEnvironment

Represents the Python runtime environment configuration.

**Fields**:
- `system_python_version: tuple[int, int, int]` - System Python version (e.g., (3, 11, 5))
- `venv_path: Path` - Virtual environment directory path (e.g., `./venv`)
- `venv_exists: bool` - Whether venv has been created
- `dependencies_installed: bool` - Whether requirements.txt has been installed
- `pythonpath_configured: bool` - Whether PYTHONPATH includes Webots controller
- `required_packages: list[str]` - List of packages from requirements.txt

**Validation Rules**:
- `system_python_version` MUST be >= (3, 8, 0)
- `venv_path` MUST exist when `venv_exists` is True
- `dependencies_installed` requires `venv_exists` to be True
- PYTHONPATH MUST include Webots controller library when running external controllers

**State Transitions**:
```
NOT_CREATED → CREATED (venv creation)
CREATED → CONFIGURED (pip install)
CONFIGURED → VERIFIED (import tests pass)
```

---

### 3. WorldFileConfiguration

Represents the IA_20252.wbt world file and its expected state.

**Fields**:
- `file_path: Path` - Path to world file (`IA_20252/worlds/IA_20252.wbt`)
- `exists: bool` - Whether file exists
- `is_valid_wbt: bool` - Whether file is valid Webots world format
- `supervisor_path: Path` - Path to supervisor controller (`IA_20252/controllers/supervisor/supervisor.py`)
- `supervisor_immutable: bool` - MUST be True (constitution constraint)

**Validation Rules**:
- `file_path` MUST exist (FR-005)
- `supervisor_path` MUST NOT be modified (FR-011, Principle V)
- File MUST be parseable by Webots R2023b
- World MUST contain exactly 15 cube spawn locations

**State**: Read-only (no modifications allowed per constitution)

---

### 4. SensorConfiguration

Represents sensor device configuration and expected data formats.

**LIDAR Subtype**:
- `device_name: str` - "lidar" (Webots device identifier)
- `horizontal_resolution: int` - 512 points
- `number_of_layers: int` - 1 (2D LIDAR)
- `field_of_view: float` - ~3.14159 radians (180°)
- `min_range: float` - 0.01 meters
- `max_range: float` - Infinite (effective ~5-10m in arena)
- `expected_array_size: int` - 512

**Camera Subtype**:
- `device_name: str` - "camera" (Webots device identifier)
- `width: int` - 128 pixels
- `height: int` - 128 pixels
- `format: str` - "BGRA" (Webots default)
- `bytes_per_pixel: int` - 4 (BGRA)
- `expected_image_size: int` - 128 * 128 * 4 = 65536 bytes
- `field_of_view: float` - ~0.785398 radians (45°)

**Validation Rules**:
- LIDAR ranges MUST be list/array of exactly 512 floats
- LIDAR finite values MUST be in range [0.01, 10.0] meters
- Camera image MUST be exactly 65536 bytes (128x128x4)
- Camera pixel values MUST be in range [0, 255] (uint8)
- Sensors MUST return valid data within 1 second of enable (SC-005, SC-006)

**State Transitions**:
```
DISABLED → ENABLING (call .enable(timestep))
ENABLING → ENABLED (wait N simulation steps)
ENABLED → READY (first valid data received)
READY → STREAMING (continuous data available)
```

---

### 5. ValidationTest

Represents a single automated validation test result.

**Fields**:
- `test_id: str` - Unique test identifier (e.g., "test_webots_executable_exists")
- `test_category: str` - "installation" | "environment" | "sensor" | "integration"
- `description: str` - Human-readable test purpose
- `status: str` - "not_run" | "passed" | "failed" | "skipped"
- `execution_time: float` - Time in seconds
- `error_message: str | None` - Failure details if status="failed"
- `timestamp: datetime` - When test was executed

**Validation Rules**:
- `test_id` MUST be unique within test suite
- `status` transitions: not_run → (passed | failed | skipped)
- `error_message` MUST be present when status="failed"
- FR-012 requires 4 tests minimum (test_webots_setup.py)

**State Transitions**:
```
NOT_RUN → RUNNING (pytest collects and starts test)
RUNNING → PASSED (assertions succeed)
RUNNING → FAILED (assertion fails or exception raised)
NOT_RUN → SKIPPED (pytest skip marker or condition)
```

---

### 6. SetupValidationReport

Aggregates results from all validation tests.

**Fields**:
- `total_tests: int` - Total number of tests executed
- `passed_tests: int` - Number of passing tests
- `failed_tests: int` - Number of failing tests
- `skipped_tests: int` - Number of skipped tests
- `pass_rate: float` - passed_tests / total_tests (percentage)
- `execution_time: float` - Total test suite runtime
- `timestamp: datetime` - Report generation time
- `environment_details: dict` - OS, Python version, Webots version

**Validation Rules**:
- `pass_rate` MUST be 100% for SC-003 (all tests pass)
- `total_tests` MUST be >= 4 (FR-012)
- Report MUST be generated after every test run
- Report MUST be saved to `logs/validation_report_<timestamp>.json`

**State**: Generated artifact (no state transitions)

---

## Relationships

```
WebotsInstallation
    ↓ provides
PythonEnvironment ← configures → WorldFileConfiguration
    ↓ enables
SensorConfiguration (LIDAR, Camera)
    ↓ validated by
ValidationTest → aggregated into → SetupValidationReport
```

**Key Dependencies**:
1. PythonEnvironment depends on WebotsInstallation (PYTHONPATH)
2. SensorConfiguration depends on WorldFileConfiguration (device definitions)
3. All ValidationTests depend on preceding installations
4. SetupValidationReport depends on all ValidationTests

---

## Data Formats

### Configuration File: environment.json

Captures environment configuration for reproducibility (FR-014).

```json
{
  "webots": {
    "version": "R2023b",
    "install_path": "/Applications/Webots.app",
    "executable": "/Applications/Webots.app/webots",
    "python_controller_path": "/Applications/Webots.app/lib/controller/python38"
  },
  "python": {
    "system_version": "3.11.5",
    "venv_path": "./venv",
    "pythonpath": [
      "/Applications/Webots.app/lib/controller/python38"
    ]
  },
  "world_file": {
    "path": "IA_20252/worlds/IA_20252.wbt",
    "supervisor": "IA_20252/controllers/supervisor/supervisor.py"
  },
  "sensors": {
    "lidar": {
      "resolution": 512,
      "layers": 1,
      "fov": 3.14159
    },
    "camera": {
      "width": 128,
      "height": 128,
      "format": "BGRA"
    }
  },
  "platform": {
    "os": "Darwin",
    "os_version": "14.1.0",
    "architecture": "arm64"
  },
  "timestamp": "2025-11-18T10:30:00Z"
}
```

### Test Result Format: validation_report.json

```json
{
  "summary": {
    "total_tests": 4,
    "passed": 4,
    "failed": 0,
    "skipped": 0,
    "pass_rate": 100.0,
    "execution_time": 3.24
  },
  "tests": [
    {
      "test_id": "test_webots_executable_exists",
      "category": "installation",
      "description": "Verify Webots R2023b is installed",
      "status": "passed",
      "execution_time": 0.12,
      "timestamp": "2025-11-18T10:30:01Z"
    },
    {
      "test_id": "test_python_version_compatible",
      "category": "environment",
      "description": "Verify Python 3.8+ is available",
      "status": "passed",
      "execution_time": 0.05,
      "timestamp": "2025-11-18T10:30:01Z"
    },
    {
      "test_id": "test_world_file_exists",
      "category": "installation",
      "description": "Verify IA_20252.wbt exists",
      "status": "passed",
      "execution_time": 0.02,
      "timestamp": "2025-11-18T10:30:01Z"
    },
    {
      "test_id": "test_virtual_environment_configured",
      "category": "environment",
      "description": "Verify venv with dependencies",
      "status": "passed",
      "execution_time": 3.05,
      "timestamp": "2025-11-18T10:30:04Z"
    }
  ],
  "environment": {
    "python_version": "3.11.5",
    "webots_version": "R2023b",
    "os": "Darwin 14.1.0"
  },
  "timestamp": "2025-11-18T10:30:04Z"
}
```

---

## Validation Checklist (Phase 1.1 Acceptance)

**Installation Validation** (User Story 1):
- [x] WebotsInstallation.version == "R2023b"
- [x] WebotsInstallation.is_installed == True
- [x] WorldFileConfiguration.exists == True
- [x] World loads in <30 seconds (SC-002)

**Python Environment** (User Story 2):
- [x] PythonEnvironment.system_python_version >= (3, 8, 0)
- [x] PythonEnvironment.venv_exists == True
- [x] PythonEnvironment.dependencies_installed == True
- [x] Controller module import succeeds

**Sensor Configuration** (User Story 3):
- [x] SensorConfiguration (LIDAR).expected_array_size == 512
- [x] SensorConfiguration (Camera).expected_image_size == 65536
- [x] LIDAR returns valid data in <1s (SC-005)
- [x] Camera returns valid data in <1s (SC-006)

**Automated Validation** (User Story 4):
- [x] ValidationTest count >= 4 (FR-012)
- [x] SetupValidationReport.pass_rate == 100% (SC-003)
- [x] All tests complete in <5min

---

## Entity Lifecycle Summary

1. **Setup Phase**:
   - WebotsInstallation created (manual install)
   - PythonEnvironment created (venv setup)
   - WorldFileConfiguration loaded (verify existing file)
   - SensorConfiguration defined (read from world file)

2. **Validation Phase**:
   - ValidationTests executed (pytest run)
   - SetupValidationReport generated (aggregation)
   - environment.json saved (configuration capture)

3. **Operational Phase** (Phase 2+):
   - Entities become read-only configuration
   - Controllers query sensor data
   - Reports used for debugging

---

**Compliance Notes**:
- ✅ No implementation details (languages/frameworks) - describes data structures only
- ✅ All entities map to functional requirements (FR-001 to FR-015)
- ✅ Validation rules enforce constitution constraints (supervisor immutability)
- ✅ State transitions follow linear progression (no complex workflows)
