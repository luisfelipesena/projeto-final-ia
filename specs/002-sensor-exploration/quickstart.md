# Quickstart Guide: Sensor Exploration and Control Validation

**Feature**: 002-sensor-exploration
**Branch**: `002-sensor-exploration`
**Date**: 2025-11-21

## Goal

Validate YouBot control interfaces and analyze sensor data to establish foundational knowledge for Phase 2 (neural network implementation).

---

## Prerequisites

✅ **Phase 1.1 Complete**: Webots R2023b installed, world file validated
✅ **Python Environment**: venv active with dependencies installed
✅ **Project Structure**: Tests and notebooks directories exist

**Verify Setup:**
```bash
# Check Python environment
which python  # Should show venv/bin/python3
python --version  # Should be 3.14.0

# Check dependencies
pip list | grep -E "(numpy|matplotlib|opencv|pytest)"

# Check Webots
/Applications/Webots.app/Contents/MacOS/webots --version  # R2023b
```

---

## Phase Overview

### **Phase 1.2**: Control Validation (Priority P1)
- Test base movements (forward, backward, strafe, rotate, stop)
- Test arm positioning (height, orientation)
- Test gripper actions (grip, release)
- Document movement limits
- **Deliverable**: `tests/test_basic_controls.py` (passing 100%)

### **Phase 1.3**: Sensor Analysis (Priority P2)
- Analyze LIDAR data (format, range, FOV)
- Analyze camera data (resolution, FPS, color detection)
- Create visualizations (polar plots, example images)
- **Deliverable**: `notebooks/01_sensor_exploration.ipynb` (complete analysis)

### **Phase 1.4**: Arena Mapping (Priority P3)
- Parse world file for dimensions
- Document deposit box locations
- Map obstacle positions
- **Deliverable**: `docs/arena_map.md` (schematic diagram)

---

## Step 1: Run Control Validation Tests

### 1.1 Create Test File

**Location**: `tests/test_basic_controls.py`

**Minimal Test Template:**
```python
"""
YouBot Control Validation Tests
Tests FR-001 through FR-013 from spec.md
"""

import pytest
from controller import Robot

class TestBaseMovement:
    """Test base movement controls (FR-001 to FR-007)."""

    @pytest.fixture(scope="class")
    def robot(self):
        """Initialize Webots robot."""
        robot = Robot()
        yield robot
        # Cleanup if needed

    def test_forward_movement(self, robot):
        """FR-001: Forward movement validation."""
        # TODO: Implement test
        pass

    # Add remaining base tests...

class TestArmGripper:
    """Test arm and gripper controls (FR-008 to FR-013)."""

    def test_arm_height(self):
        """FR-008: Arm height positioning."""
        # TODO: Implement test
        pass

    # Add remaining arm/gripper tests...
```

### 1.2 Run Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all control tests
pytest tests/test_basic_controls.py -v

# Run specific test
pytest tests/test_basic_controls.py::TestBaseMovement::test_forward_movement -v

# Generate HTML report
pytest tests/test_basic_controls.py --html=test_report.html --self-contained-html
```

**Expected Result**: All tests passing (SC-004 target: 100%)

### 1.3 Document Limits

Tests should output measured limits to:
- `logs/velocity_limits.json` (FR-006)
- `logs/joint_limits.json` (FR-012)

**Verify:**
```bash
cat logs/velocity_limits.json
cat logs/joint_limits.json
```

---

## Step 2: Run Sensor Analysis Notebook

### 2.1 Create Notebook

**Location**: `notebooks/01_sensor_exploration.ipynb`

**Notebook Outline:**
```markdown
# Sensor Exploration - Phase 1.3

## 1. Setup
- Initialize Webots robot
- Enable LIDAR and camera sensors

## 2. LIDAR Analysis (FR-014 to FR-019)
### 2.1 Specifications
- Document sensor specs (FR-015)
- Measure detection ranges (FR-016)

### 2.2 Visualization
- Create polar plots (FR-017)
- Identify obstacles (FR-018)

## 3. Camera Analysis (FR-020 to FR-026)
### 3.1 Specifications
- Document resolution (FR-021)
- Measure FPS (FR-022)

### 3.2 Color Detection
- Capture example images (FR-023)
- Implement threshold detection (FR-024)
- Evaluate accuracy (FR-025)

## 4. Results Summary
- Key findings
- Recommendations for Phase 2
```

### 2.2 Execute Notebook

```bash
# Activate environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/01_sensor_exploration.ipynb
```

**Execution Steps:**
1. Run all cells sequentially
2. Verify visualizations generate
3. Check accuracy meets >80% target (SC-008)
4. Save notebook with outputs

### 2.3 Verify Outputs

**Check generated files:**
```bash
# LIDAR outputs (FR-017, FR-018)
ls media/lidar_scan_*.png

# Camera example images (FR-023)
ls media/cube_examples/*.png

# Saved data
ls *.npy  # lidar_sample.npy, camera_sample.npy
```

---

## Step 3: Create Arena Map

### 3.1 Parse World File

**Create script**: `scripts/parse_arena.py`

```python
"""
Parse IA_20252.wbt world file to extract arena layout.
Generates docs/arena_map.md
"""

import re

def parse_arena_dimensions(wbt_path):
    """Extract arena size from RectangleArena node."""
    # TODO: Implement parsing logic
    pass

def parse_deposit_boxes(wbt_path):
    """Extract deposit box positions and colors."""
    # TODO: Implement parsing logic
    pass

def generate_arena_map_md(arena_data, output_path):
    """Generate markdown documentation."""
    # TODO: Generate docs/arena_map.md
    pass

if __name__ == "__main__":
    arena = parse_arena_dimensions("IA_20252/worlds/IA_20252.wbt")
    boxes = parse_deposit_boxes("IA_20252/worlds/IA_20252.wbt")
    generate_arena_map_md({'arena': arena, 'boxes': boxes},
                          "docs/arena_map.md")
```

### 3.2 Run Parser

```bash
python scripts/parse_arena.py
```

### 3.3 Verify Output

**Check**: `docs/arena_map.md` exists with:
- Arena dimensions (FR-027)
- Deposit box coordinates (FR-028)
- Obstacle positions (FR-029)
- Schematic diagram (FR-030)

```bash
cat docs/arena_map.md
```

---

## Step 4: Validate Success Criteria

### Checklist

- [ ] **SC-001**: All base commands execute successfully
- [ ] **SC-002**: Arm positioning within 5% tolerance
- [ ] **SC-003**: Gripper commands execute successfully
- [ ] **SC-004**: Test script passes 100% (13/13 tests)
- [ ] **SC-005**: LIDAR specs documented (points, FOV, range)
- [ ] **SC-006**: LIDAR visualizations show obstacles clearly
- [ ] **SC-007**: Camera specs documented (resolution, FPS)
- [ ] **SC-008**: Color detection >80% accuracy
- [ ] **SC-009**: Jupyter notebook complete with visualizations
- [ ] **SC-010**: Arena map contains all required elements
- [ ] **SC-011**: All measurements documented with sufficient detail

**Verify Command:**
```bash
# Check test results
pytest tests/test_basic_controls.py -v | grep -E "PASSED|FAILED"

# Check notebook executed
jupyter nbconvert --execute notebooks/01_sensor_exploration.ipynb --to html

# Check arena map
test -f docs/arena_map.md && echo "✓ Arena map exists"
```

---

## Step 5: Document Decisions

### 5.1 Create DECISÃO Entries

For each technical choice during implementation:
1. Open `DECISIONS.md`
2. Add new DECISÃO entry (011, 012, etc.)
3. Follow template:
   - O que foi decidido
   - Por que foi decidido
   - Base teórica (citations from REFERENCIAS.md)
   - Alternativas consideradas
   - Impacto esperado

**Example:**
```markdown
## DECISÃO 011: LIDAR Visualization Method

**Data:** 2025-11-21
**Fase:** Fase 1.3 - Sensor Analysis
**Status:** ✅ Implementado

### O que foi decidido
Usar matplotlib polar plots para visualização de LIDAR scans.

### Por que foi decidido
Polar coordinates natural representation...

### Base Teórica
- Thrun et al. (2005): Probabilistic Robotics, Cap. 6...

### Alternativas Consideradas
1. Cartesian (XY) plots...

### Impacto Esperado
Facilita identificação visual de obstáculos...
```

### 5.2 Update TODO.md

Mark Phase 1.2-1.3 tasks as complete:
```markdown
#### 1.2 Exploração dos Controles Base ✅ CONCLUÍDO
- [x] Testar movimentos da base
- [x] Testar comandos do braço
...
```

---

## Step 6: Commit and Document

### 6.1 Commit Strategy

```bash
# Stage files
git add tests/test_basic_controls.py
git add notebooks/01_sensor_exploration.ipynb
git add docs/arena_map.md
git add DECISIONS.md TODO.md

# Commit with descriptive message
git commit -m "feat(sensor-exploration): complete Phase 1.2-1.3 validation

- Tests: 13/13 passing for base, arm, gripper controls (SC-004)
- Sensor analysis: LIDAR + camera documented in notebook (SC-009)
- Arena map: dimensions and deposit boxes documented (SC-010)
- Decisions: DECISÃO 011-0XX documented for technical choices

Deliverables:
- tests/test_basic_controls.py (100-200 LOC, 100% passing)
- notebooks/01_sensor_exploration.ipynb (200-300 LOC + visualizations)
- docs/arena_map.md (schematic with measurements)
- logs/velocity_limits.json, logs/joint_limits.json

References:
- Bischoff et al. (2011): YouBot specifications
- Michel (2004): Webots API
- Thrun et al. (2005): LIDAR visualization
- Bradski & Kaehler (2008): HSV color detection

Phase 1.2-1.3 Status: ✅ COMPLETE
Next: /speckit.tasks for granular task breakdown"

# Push to remote
git push origin 002-sensor-exploration
```

---

## Troubleshooting

### Issue: Webots Python Not Found

**Symptom**: `WARNING: Python was not found`

**Solution**:
1. Webots → Preferences → General
2. Set "Python command" to: `/Users/luisfelipesena/Development/Personal/projeto-final-ia/venv/bin/python3`
3. Reload simulation

### Issue: Tests Fail with Import Error

**Symptom**: `ModuleNotFoundError: No module named 'controller'`

**Solution**: `controller` module only available within Webots runtime. Tests must be executed with Webots simulation running.

### Issue: Jupyter Kernel Not Found

**Symptom**: Notebook can't find venv kernel

**Solution**:
```bash
# Install ipykernel in venv
pip install ipykernel

# Register venv as Jupyter kernel
python -m ipykernel install --user --name=projeto-final-ia --display-name="YouBot (Python 3.14)"

# Launch Jupyter and select "YouBot (Python 3.14)" kernel
jupyter notebook
```

### Issue: matplotlib Plots Not Displaying

**Symptom**: `visualize_lidar_polar()` runs but no plot appears

**Solution**:
```python
# Add to notebook cell
%matplotlib inline
import matplotlib.pyplot as plt

# After generating figure
plt.show()  # Force display
```

---

## Next Steps

After completing Phase 1.2-1.3:

1. **Generate Tasks**: Run `/speckit.tasks` to create granular task breakdown
2. **Implementation**: Run `/speckit.implement` to execute tasks sequentially
3. **Validation**: Ensure all success criteria met (SC-001 through SC-011)
4. **Merge**: Create PR to merge `002-sensor-exploration` → `main`
5. **Phase 2**: Begin neural network implementation (RNA for LIDAR/camera)

**Documentation References:**
- Spec: `specs/002-sensor-exploration/spec.md`
- Plan: `specs/002-sensor-exploration/plan.md`
- Research: `specs/002-sensor-exploration/research.md`
- Data Model: `specs/002-sensor-exploration/data-model.md`
- Contracts: `specs/002-sensor-exploration/contracts/*.md`

---

**Quickstart Status**: ✅ COMPLETE
**Ready for /speckit.tasks command.**
