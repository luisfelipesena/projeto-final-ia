# Test Specifications: Control Validation

**Feature**: 002-sensor-exploration
**Contract**: Test API for YouBot control validation
**Date**: 2025-11-21

## Overview

This contract defines the test functions and validation criteria for FR-001 through FR-013 (control validation requirements). Implementation file: `tests/test_basic_controls.py`.

---

## Test Suite: Base Movement Control

### TEST-001: Forward Movement Validation

**Requirement**: FR-001 - System MUST successfully execute forward movement commands

**Function Signature:**
```python
def test_base_forward_movement() -> None
```

**Test Steps:**
1. Initialize YouBot in Webots simulation
2. Get initial robot position (x0, y0)
3. Command: `base.set_velocity(vx=0.2, vy=0.0, omega=0.0)`
4. Execute for 5 seconds simulation time
5. Stop: `base.set_velocity(0, 0, 0)`
6. Get final robot position (x1, y1)

**Success Criteria:**
- Robot moves forward: `x1 > x0` (positive X displacement)
- Lateral drift minimal: `|y1 - y0| < 0.1m`
- No rotation: robot heading unchanged (±5°)

---

### TEST-002: Backward Movement Validation

**Requirement**: FR-002 - System MUST successfully execute backward movement commands

**Function Signature:**
```python
def test_base_backward_movement() -> None
```

**Test Steps:**
1. Initialize YouBot
2. Get initial position
3. Command: `base.set_velocity(vx=-0.2, vy=0.0, omega=0.0)`
4. Execute for 5 seconds
5. Stop and measure final position

**Success Criteria:**
- Robot moves backward: `x1 < x0` (negative X displacement)
- Lateral drift minimal: `|y1 - y0| < 0.1m`
- Displacement magnitude: `|x1 - x0| > 0.5m` (reasonable distance traveled)

---

### TEST-003: Strafe Left Movement Validation

**Requirement**: FR-003 - System MUST successfully execute strafe left commands

**Function Signature:**
```python
def test_base_strafe_left() -> None
```

**Test Steps:**
1. Initialize YouBot
2. Get initial position
3. Command: `base.set_velocity(vx=0.0, vy=0.2, omega=0.0)`
4. Execute for 5 seconds
5. Stop and measure final position

**Success Criteria:**
- Robot moves left: `y1 > y0` (positive Y displacement)
- Forward drift minimal: `|x1 - x0| < 0.1m`
- Displacement magnitude: `|y1 - y0| > 0.5m`

---

### TEST-004: Strafe Right Movement Validation

**Requirement**: FR-003 - System MUST successfully execute strafe right commands

**Function Signature:**
```python
def test_base_strafe_right() -> None
```

**Test Steps:**
1. Initialize YouBot
2. Get initial position
3. Command: `base.set_velocity(vx=0.0, vy=-0.2, omega=0.0)`
4. Execute for 5 seconds
5. Stop and measure final position

**Success Criteria:**
- Robot moves right: `y1 < y0` (negative Y displacement)
- Forward drift minimal: `|x1 - x0| < 0.1m`
- Displacement magnitude: `|y1 - y0| > 0.5m`

---

### TEST-005: Clockwise Rotation Validation

**Requirement**: FR-004 - System MUST successfully execute clockwise rotation

**Function Signature:**
```python
def test_base_rotate_clockwise() -> None
```

**Test Steps:**
1. Initialize YouBot
2. Get initial heading angle θ0
3. Command: `base.set_velocity(vx=0.0, vy=0.0, omega=-0.3)`
4. Execute for 5 seconds
5. Stop and measure final heading θ1

**Success Criteria:**
- Robot rotates clockwise: `θ1 < θ0` (negative angular displacement)
- Position drift minimal: `sqrt((x1-x0)² + (y1-y0)²) < 0.2m`
- Rotation magnitude: `|θ1 - θ0| > 0.5 rad` (~30°)

---

### TEST-006: Counterclockwise Rotation Validation

**Requirement**: FR-004 - System MUST successfully execute counterclockwise rotation

**Function Signature:**
```python
def test_base_rotate_counterclockwise() -> None
```

**Test Steps:**
1. Initialize YouBot
2. Get initial heading
3. Command: `base.set_velocity(vx=0.0, vy=0.0, omega=0.3)`
4. Execute for 5 seconds
5. Stop and measure final heading

**Success Criteria:**
- Robot rotates CCW: `θ1 > θ0` (positive angular displacement)
- Position drift minimal: `<0.2m`
- Rotation magnitude: `|θ1 - θ0| > 0.5 rad`

---

### TEST-007: Stop Command Validation

**Requirement**: FR-005 - System MUST successfully execute stop commands

**Function Signature:**
```python
def test_base_stop_command() -> None
```

**Test Steps:**
1. Initialize YouBot
2. Command movement: `base.set_velocity(0.2, 0.1, 0.2)` (mixed motion)
3. Execute for 2 seconds (robot should be moving)
4. Command stop: `base.set_velocity(0, 0, 0)`
5. Wait 1 second
6. Measure position change over next 2 seconds

**Success Criteria:**
- Position drift after stop: `< 0.05m` (robot settles quickly)
- Heading drift after stop: `< 0.05 rad` (~3°)

---

### TEST-008: Velocity Limits Documentation

**Requirement**: FR-006 - System MUST document velocity limits

**Function Signature:**
```python
def test_base_velocity_limits() -> dict
```

**Test Steps:**
1. Test increasing vx: 0.1, 0.2, 0.3, ..., until robot stops responding
2. Document maximum forward velocity
3. Repeat for vy (strafe)
4. Repeat for omega (rotation)
5. Test negative limits (backward, opposite rotation)

**Success Criteria:**
- Returns dict with measured limits:
  ```python
  {
      'vx_max': float,  # m/s
      'vx_min': float,
      'vy_max': float,
      'vy_min': float,
      'omega_max': float,  # rad/s
      'omega_min': float,
  }
  ```
- Limits match YouBot specifications (±10%)

---

## Test Suite: Arm and Gripper Control

### TEST-009: Arm Height Positioning

**Requirement**: FR-008 - System MUST execute set_height commands

**Function Signature:**
```python
def test_arm_height_positions() -> None
```

**Test Steps:**
For each height preset (FLOOR, FRONT, HIGH):
1. Command: `arm.set_height(preset)`
2. Wait for movement to complete (timeout: 5s)
3. Measure final arm position (joint angles or end-effector height)
4. Verify position matches preset

**Success Criteria:**
- Arm reaches each preset within timeout
- Final position consistent across repeated commands (±5%)
- No collisions with base or arena

---

### TEST-010: Arm Orientation Positioning

**Requirement**: FR-009 - System MUST execute set_orientation commands

**Function Signature:**
```python
def test_arm_orientation_positions() -> None
```

**Test Steps:**
For each orientation preset (FRONT, DOWN):
1. Set arm to mid-height first
2. Command: `arm.set_orientation(preset)`
3. Wait for completion
4. Measure final orientation (wrist joint angles)
5. Verify matches preset

**Success Criteria:**
- Arm reaches each orientation within 5s timeout
- Final orientation consistent (±5° per joint)
- Gripper orientation visually matches preset (FRONT=horizontal, DOWN=vertical)

---

### TEST-011: Gripper Close Command

**Requirement**: FR-010 - System MUST execute grip commands

**Function Signature:**
```python
def test_gripper_close() -> None
```

**Test Steps:**
1. Ensure gripper starts open
2. Command: `gripper.grip()`
3. Wait for completion (timeout: 2s)
4. Measure gripper jaw width

**Success Criteria:**
- Gripper closes: jaw_width < 10mm
- Motion completes within timeout
- Gripper state visually closed in simulation

---

### TEST-012: Gripper Open Command

**Requirement**: FR-011 - System MUST execute release commands

**Function Signature:**
```python
def test_gripper_open() -> None
```

**Test Steps:**
1. Close gripper first
2. Command: `gripper.release()`
3. Wait for completion
4. Measure gripper jaw width

**Success Criteria:**
- Gripper opens: jaw_width > 40mm
- Motion completes within 2s
- Gripper state visually open

---

### TEST-013: Arm Joint Limits Documentation

**Requirement**: FR-012 - System MUST document arm joint limits

**Function Signature:**
```python
def test_arm_joint_limits() -> dict
```

**Test Steps:**
1. For each of 5 arm joints:
   - Command incremental position changes
   - Record maximum and minimum achievable angles
2. Document range of motion per joint

**Success Criteria:**
- Returns dict with joint limits:
  ```python
  {
      'joint1': (min_rad, max_rad),
      'joint2': (min_rad, max_rad),
      'joint3': (min_rad, max_rad),
      'joint4': (min_rad, max_rad),
      'joint5': (min_rad, max_rad),
      'gripper': (min_width_mm, max_width_mm),
  }
  ```

---

## Test Execution Requirements

### Pytest Configuration

**Markers:**
```python
# tests/test_basic_controls.py
import pytest

pytestmark = pytest.mark.slow  # All tests require Webots simulation
```

**Fixtures:**
```python
@pytest.fixture(scope="module")
def youbot():
    """Initialize YouBot robot for all tests in module."""
    robot = YouBot()  # From IA_20252/controllers/youbot/youbot.py
    yield robot
    robot.cleanup()  # Ensure clean shutdown

@pytest.fixture(autouse=True)
def reset_robot(youbot):
    """Reset robot to initial state before each test."""
    youbot.base.set_velocity(0, 0, 0)
    youbot.arm.set_height("REST")
    youbot.gripper.release()
    youbot.step(100)  # Wait for reset
```

### Execution Command

```bash
# Run all control tests
pytest tests/test_basic_controls.py -v

# Run specific test
pytest tests/test_basic_controls.py::test_base_forward_movement -v

# Run with detailed output
pytest tests/test_basic_controls.py -v -s
```

### Expected Output

```
tests/test_basic_controls.py::test_base_forward_movement PASSED       [ 7%]
tests/test_basic_controls.py::test_base_backward_movement PASSED      [14%]
tests/test_basic_controls.py::test_base_strafe_left PASSED            [21%]
tests/test_basic_controls.py::test_base_strafe_right PASSED           [28%]
tests/test_basic_controls.py::test_base_rotate_clockwise PASSED       [35%]
tests/test_basic_controls.py::test_base_rotate_counterclockwise PASSED [42%]
tests/test_basic_controls.py::test_base_stop_command PASSED           [50%]
tests/test_basic_controls.py::test_base_velocity_limits PASSED        [57%]
tests/test_basic_controls.py::test_arm_height_positions PASSED        [64%]
tests/test_basic_controls.py::test_arm_orientation_positions PASSED   [71%]
tests/test_basic_controls.py::test_gripper_close PASSED               [78%]
tests/test_basic_controls.py::test_gripper_open PASSED                [85%]
tests/test_basic_controls.py::test_arm_joint_limits PASSED            [92%]

==================== 13 passed in 180.50s (3m 0s) ====================
```

**Performance Target**: SC-004 requires 100% test pass rate.

---

## Test Data Outputs

### Logs

**Format**: JSON lines for structured logging
```json
{"timestamp": "2025-11-21T10:30:45", "test": "test_base_forward_movement", "status": "PASS", "duration": 6.2, "details": {"displacement_x": 0.95, "displacement_y": 0.02}}
{"timestamp": "2025-11-21T10:30:52", "test": "test_base_backward_movement", "status": "PASS", "duration": 6.1, "details": {"displacement_x": -0.93, "displacement_y": 0.03}}
```

### Measurement Files

- `logs/velocity_limits.json` - Documented velocity limits (FR-006)
- `logs/joint_limits.json` - Documented arm joint limits (FR-012)

---

## Integration with CI/CD

**Future Enhancement** (Phase 6 Integration):
- Automate Webots headless execution
- Integrate with GitHub Actions
- Generate test report artifacts

**Current Scope (Phase 1):**
- Manual execution with Webots GUI
- pytest HTML report: `pytest --html=test_report.html`

---

**Contract Status**: ✅ COMPLETE
**All test specifications defined. Ready for implementation.**
