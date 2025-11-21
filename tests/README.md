# YouBot Control Validation Tests

**Phase**: 1.2 - Sensor Exploration and Control Validation
**Purpose**: Validate YouBot control interfaces (base, arm, gripper) through automated pytest suite

## Test Execution Requirements

### Prerequisites

1. **Webots R2023b** must be running with `IA_20252.wbt` world loaded
2. **Python venv** activated: `source venv/bin/activate`
3. **Dependencies** installed: `numpy`, `pytest`, `matplotlib`, `scipy`, `opencv-python`

### Webots Configuration

**IMPORTANT**: Tests use the `controller` module which is **only available** within Webots runtime environment. Standard pytest execution will fail with `ModuleNotFoundError`.

**Correct Test Execution**:
1. Open Webots → Load `IA_20252/worlds/IA_20252.wbt`
2. Set YouBot controller to test script (modify robot node → controller field)
3. Run simulation to execute tests

**Alternative Approach** (recommended for this phase):
- Tests are designed to be embedded in Webots controller scripts
- Can be executed via `pytest` if Webots Python is configured
- See quickstart.md for detailed instructions

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all control tests
pytest tests/test_basic_controls.py -v

# Run specific test class
pytest tests/test_basic_controls.py::TestBaseMovement -v

# Run specific test function
pytest tests/test_basic_controls.py::TestBaseMovement::test_base_forward_movement -v

# Generate HTML report
pytest tests/test_basic_controls.py --html=test_report.html --self-contained-html
```

## Test Files

### test_basic_controls.py
**Coverage**: FR-001 through FR-013 (base, arm, gripper control validation)

**Test Classes**:
- `TestBaseMovement` - Tests FR-001 to FR-007 (forward, backward, strafe, rotate, stop, velocity limits)
- `TestArmGripper` - Tests FR-008 to FR-013 (arm height, orientation, gripper grip/release, joint limits)

**Expected Duration**: ~3 minutes (13 tests with 5-second motion periods each)

**Success Criteria**: 100% pass rate (SC-004: 13/13 tests passing)

### conftest.py
**Purpose**: Pytest configuration and shared fixtures

**Fixtures**:
- `robot` - Webots Robot instance (module-scoped)
- `youbot` - YouBot controller with base, arm, gripper modules
- `reset_robot` - Auto-reset before each test (autouse)
- `velocity_limits` - Storage for velocity measurements (session-scoped)
- `joint_limits` - Storage for joint measurements (session-scoped)

### test_helpers.py
**Purpose**: Utility functions for position/angle calculations

**Functions**:
- `calculate_distance(pos1, pos2)` - Euclidean distance
- `normalize_angle(angle)` - Angle normalization to [-π, π]
- `get_robot_position(robot)` - GPS or supervisor field position
- `get_robot_heading(robot)` - Compass or rotation field heading
- `wait_for_motion(robot, duration_ms)` - Execute simulation steps
- `is_position_stable(positions, tolerance)` - Check if robot stopped

## Test Outputs

### Logs
- **Location**: `logs/test_basic_controls.log` (created during test run)
- **Format**: JSON lines for structured logging
- **Content**: Test results, duration, position measurements

### Measurements
- **Velocity Limits**: `logs/velocity_limits.json` (FR-006)
- **Joint Limits**: `logs/joint_limits.json` (FR-012)

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'controller'

**Cause**: `controller` module only available inside Webots runtime.

**Solution**:
1. Webots → Preferences → General
2. Set "Python command" to: `/Users/luisfelipesena/Development/Personal/projeto-final-ia/venv/bin/python3`
3. Reload simulation

### Issue: Tests fail with "Robot not initialized"

**Cause**: Webots simulation not running or robot not spawned.

**Solution**: Ensure `IA_20252.wbt` world is loaded and simulation is running before executing tests.

### Issue: Timeouts during arm/gripper tests

**Cause**: Arm movements take time to complete (preset positions).

**Solution**: Tests include wait periods (`wait_for_motion`). If still timing out, increase duration in test functions.

## Next Steps

After completing control validation tests:
1. Run sensor analysis notebook (`notebooks/01_sensor_exploration.ipynb`)
2. Generate arena map (`scripts/parse_arena.py`)
3. Document decisions in DECISIONS.md (entries 011-015)
4. Commit results with comprehensive message

See `specs/002-sensor-exploration/quickstart.md` for detailed workflow.
