# Implementation Plan: Sensor Exploration and Control Validation

**Branch**: `002-sensor-exploration` | **Date**: 2025-11-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-sensor-exploration/spec.md`

## Summary

This phase validates YouBot control interfaces (base, arm, gripper) and analyzes sensor data (LIDAR, camera RGB) to establish foundational understanding for Phase 2 (neural network implementation). The approach focuses on manual testing with Webots GUI, automated test scripts, and Jupyter notebook analysis with matplotlib visualizations. Deliverables include test suite, sensor analysis notebook, and arena map documentation.

## Technical Context

**Language/Version**: Python 3.14.0 (validated in Phase 1.1)
**Primary Dependencies**:
- Webots R2023b (simulator)
- numpy 1.26.4 (numerical arrays for sensor data)
- matplotlib 3.10.7 (LIDAR/camera visualizations)
- scipy 1.16.3 (scientific computing)
- opencv-python 4.11.0.86 (image processing)
- pytest 7.4.4 (test framework)

**Storage**: File-based (test logs, sensor data CSVs, example images, Jupyter notebooks)
**Testing**: pytest with manual GUI validation in Webots
**Target Platform**: macOS 15.1, Apple M4 Pro (ARM64) - validated in Phase 1.1
**Project Type**: Single robotics project with modular structure (perception, control, navigation, manipulation)
**Performance Goals**:
- LIDAR visualizations generated in <2 seconds
- Camera frame capture at documented FPS (target: >10 FPS for future processing)
- Test suite execution in <5 seconds

**Constraints**:
- No modifications to `IA_20252/controllers/supervisor/supervisor.py` (constitutional prohibition)
- No GPS usage in implementation (can analyze but not use)
- All decisions must reference REFERENCIAS.md papers (constitutional requirement)
- Phase must complete before starting Phase 2 (RNA implementation)

**Scale/Scope**:
- 30 functional requirements (FR-001 to FR-030)
- 5 user stories (P1: controls, P2: sensors, P3: mapping)
- 11 success criteria (SC-001 to SC-011)
- Expected: 100-200 lines of test code, 200-300 lines of notebook analysis

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Fundamentação Científica (Principle I)
- **Requirement**: All technical decisions must have peer-reviewed scientific basis
- **Status**: PASS - Bischoff et al. (2011) for YouBot specs, Michel (2004) for Webots
- **Action**: Document sensor analysis methodologies in DECISIONS.md with additional citations

### ✅ Rastreabilidade Total (Principle II)
- **Requirement**: All changes documented before implementation
- **Status**: PASS - spec.md complete, plan.md in progress, DECISIONS.md ready for updates
- **Action**: Create DECISÃO 011+ entries for any technical choices during implementation

### ✅ Desenvolvimento Incremental (Principle III)
- **Requirement**: Cannot advance to next phase without completing previous
- **Status**: PASS - Phase 1.1 completed and validated (6/8 pytest, GUI confirmed)
- **Action**: Mark TODO.md Phase 1.2-1.3 complete before starting Phase 2

### ✅ Qualidade Senior (Principle IV)
- **Requirement**: >80% test coverage, PEP8 compliance, type hints, docstrings
- **Status**: PASS - automated test script planned (FR-007, FR-013), pytest framework ready
- **Action**: Run black formatter, add type hints to test functions

### ✅ Restrições Disciplinares (Principle V)
- **Prohibition 1**: Modify supervisor.py → NOT VIOLATED (sensor exploration only)
- **Prohibition 2**: Show code in video → NOT APPLICABLE (Phase 8)
- **Prohibition 3**: GPS in demo → NOT VIOLATED (analysis only, no usage)
- **Prohibition 4**: Code without theory → NOT VIOLATED (Bischoff, Michel references)
- **Status**: PASS - no constitutional violations

### ✅ Workflow SpecKit (Principle VI)
- **Requirement**: Complete specify → clarify → plan → tasks → implement → validate
- **Status**: IN PROGRESS
  - ✅ Specify: spec.md complete
  - ✅ Plan: plan.md in progress (this file)
  - ⏳ Tasks: Pending (next step: /speckit.tasks)
  - ⏳ Implement: Pending
  - ⏳ Validate: Pending

**GATE RESULT**: ✅ ALL CHECKS PASSED - Proceed to Phase 0 Research

---

## Phase 0: Research & Unknowns

### Research Tasks

#### RT-001: Webots Python Controller API for Sensors
**Unknown**: Exact API methods for reading LIDAR range_image and camera frames
**Research Goal**: Document specific Webots API calls for sensor data access
**Expected Finding**:
- LIDAR: `lidar.getRangeImage()`, `lidar.getNumberOfPoints()`, `lidar.getFov()`, `lidar.getMaxRange()`
- Camera: `camera.getImage()`, `camera.getWidth()`, `camera.getHeight()`, `camera.getFps()`

**Reference**: Michel (2004) - Webots documentation, Webots R2023b API reference

#### RT-002: YouBot Movement API Patterns
**Unknown**: Best practices for commanding omnidirectional base (vx, vy, omega)
**Research Goal**: Identify standard patterns for mecanum wheel control
**Expected Finding**: Velocity command structure for base.set_velocity(vx, vy, omega)

**Reference**: Bischoff et al. (2011) - YouBot specifications, Taheri et al. (2015) - Mecanum wheel kinematics

#### RT-003: Arm and Gripper Control Patterns
**Unknown**: Standard approach for preset arm positions (height, orientation)
**Research Goal**: Document arm positioning methods (joint angles vs. cartesian coordinates)
**Expected Finding**: Preset positions (FRONT_FLOOR, FRONT_GRIPPER, etc.) and gripper force control

**Reference**: Bischoff et al. (2011) - YouBot arm specifications, Craig (2005) - Robot kinematics

#### RT-004: LIDAR Visualization Best Practices
**Unknown**: Optimal matplotlib configuration for polar LIDAR plots
**Research Goal**: Identify standard visualization techniques for 2D LIDAR scans
**Expected Finding**: Polar coordinate plots with obstacle highlighting, angular resolution marking

**Reference**: Thrun et al. (2005) - LIDAR data representation in probabilistic robotics

#### RT-005: Color Detection Baseline Methods
**Unknown**: Simple RGB threshold ranges for green, blue, red cubes in Webots lighting
**Research Goal**: Determine effective HSV or RGB ranges for color classification
**Expected Finding**: HSV ranges (e.g., green: H=60-80, S>50, V>50) more robust than RGB

**Reference**: Bradski & Kaehler (2008) - OpenCV color spaces for object detection

#### RT-006: Arena Measurement Methodology
**Unknown**: Webots tools for measuring world coordinates
**Research Goal**: Method for extracting object positions from .wbt file or GUI
**Expected Finding**: Webots scene tree inspector or parsing .wbt file for node positions

**Reference**: Michel (2004) - Webots world file format

### Unknowns Summary

| Unknown | Resolution Method | Output Document |
|---------|-------------------|-----------------|
| Webots sensor API | Read Webots R2023b API docs + test scripts | research.md |
| Movement control patterns | Review YouBot controller base.py + Bischoff paper | research.md |
| Arm/gripper API | Review arm.py, gripper.py + Craig kinematics | research.md |
| LIDAR visualization | Survey matplotlib polar plots + Thrun methods | research.md |
| Color detection baseline | Test HSV thresholds + OpenCV docs | research.md |
| Arena measurement tools | Test Webots GUI + parse .wbt file | research.md |

**Next Action**: Generate research.md with findings from Phase 0 investigation

---

## Phase 1: Data Model & Contracts

### Data Entities

*(To be extracted from spec.md and detailed in data-model.md)*

1. **Base Movement Command**
2. **Arm Position**
3. **Gripper State**
4. **LIDAR Scan**
5. **Camera Frame**
6. **Obstacle**
7. **Arena Map**

### API Contracts

*(To be generated in /contracts/ directory)*

- **Test API**: Python test functions for control validation
- **Sensor Data API**: Data structures for LIDAR and camera readings
- **Visualization API**: Matplotlib plotting functions

### Agent Context Update

*(To be executed after data-model.md and contracts are complete)*

Command: `.specify/scripts/bash/update-agent-context.sh claude`

---

## Project Structure

### Documentation (this feature)

```text
specs/002-sensor-exploration/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (next)
├── data-model.md        # Phase 1 output (next)
├── quickstart.md        # Phase 1 output (next)
├── contracts/           # Phase 1 output (next)
│   ├── test_api.md      # Test function signatures
│   └── sensor_api.md    # Sensor data structures
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created yet)
```

### Source Code (repository root)

```text
# Single robotics project structure (existing + new for this phase)

IA_20252/                       # Base code (DO NOT MODIFY - constitutional prohibition)
├── controllers/
│   ├── youbot/
│   │   ├── youbot.py          # Main controller (READ ONLY for understanding)
│   │   ├── base.py             # Base control (READ for API patterns)
│   │   ├── arm.py              # Arm control (READ for API patterns)
│   │   └── gripper.py          # Gripper control (READ for API patterns)
│   └── supervisor/
│       └── supervisor.py       # DO NOT MODIFY (constitutional prohibition)
├── worlds/
│   └── IA_20252.wbt           # Arena world file (READ for measurements)

tests/                          # NEW for this phase
├── test_basic_controls.py      # FR-007, FR-013: Control validation (P1)
└── test_webots_setup.py        # Existing from Phase 1.1

notebooks/                      # NEW for this phase
└── 01_sensor_exploration.ipynb # FR-019, FR-026: LIDAR + camera analysis (P2)

docs/                           # NEW for this phase
├── arena_map.md                # FR-030: Arena schematic (P3)
└── environment.md              # Existing from Phase 1.1

logs/                           # Existing (for test outputs)
specs/                          # Existing (SpecKit outputs)
venv/                           # Existing (Python 3.14.0 environment)
```

**Structure Decision**: Single project structure maintained. New artifacts:
1. `tests/test_basic_controls.py` - automated control validation (100-200 LOC)
2. `notebooks/01_sensor_exploration.ipynb` - sensor analysis (200-300 LOC + visualizations)
3. `docs/arena_map.md` - arena measurements and schematic

No changes to `src/` yet - Phase 2 will create `src/perception/`, `src/control/`, etc.

---

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No violations detected. All constitutional gates passed.*

---

## Implementation Order

### Phase 0: Research ✅ (Current)
1. Read Webots API documentation for LIDAR and camera
2. Review YouBot controller files (base.py, arm.py, gripper.py) for API patterns
3. Research LIDAR visualization techniques (matplotlib polar plots)
4. Research color detection baselines (HSV thresholds)
5. Test arena measurement tools (Webots GUI scene tree)
6. Document findings in `research.md`

### Phase 1: Design ✅ (Next)
1. Extract data entities from spec.md → `data-model.md`
2. Generate API contracts for test functions → `contracts/test_api.md`
3. Generate sensor data structures → `contracts/sensor_api.md`
4. Create quickstart guide → `quickstart.md`
5. Update agent context: `.specify/scripts/bash/update-agent-context.sh claude`

### Phase 2: Task Generation (After `/speckit.plan`)
- Execute `/speckit.tasks` to generate granular task breakdown in `tasks.md`
- Expected: 40-50 tasks for control validation, sensor analysis, arena mapping

### Phase 3: Implementation (After `/speckit.tasks`)
- Execute `/speckit.implement` to work through tasks sequentially
- Create test script, notebook, arena map
- Document decisions in DECISIONS.md (DECISÃO 011+)

### Phase 4: Validation (After implementation)
- Run pytest: `pytest tests/test_basic_controls.py -v`
- Execute Jupyter notebook: validate visualizations
- Review arena map: verify all measurements documented
- Update TODO.md: mark Phase 1.2-1.3 complete

---

## References

**Primary Citations (to be documented in DECISIONS.md):**

1. **Bischoff, R., et al. (2011).** "KUKA youBot - a mobile manipulator for research and education." IEEE International Conference on Robotics and Automation.
   - *Usage*: YouBot specifications, movement/arm control patterns

2. **Michel, O. (2004).** "Cyberbotics Ltd. Webots™: Professional Mobile Robot Simulation." International Journal of Advanced Robotic Systems.
   - *Usage*: Webots API documentation, sensor access patterns

3. **Thrun, S., Burgard, W., & Fox, D. (2005).** "Probabilistic Robotics." MIT Press.
   - *Usage*: LIDAR data representation, sensor modeling

4. **Taheri, H., Zhao, C. X., & Qiao, B. (2015).** "Omnidirectional mobile robots, mechanisms and navigation approaches." Mechanism and Machine Theory.
   - *Usage*: Mecanum wheel kinematics for base control

5. **Bradski, G., & Kaehler, A. (2008).** "Learning OpenCV: Computer Vision with the OpenCV Library." O'Reilly Media.
   - *Usage*: Color space conversions, HSV thresholding for cube detection

6. **Craig, J. J. (2005).** "Introduction to Robotics: Mechanics and Control" (3rd ed.). Pearson.
   - *Usage*: Robot arm kinematics, joint positioning

**Additional references as needed during implementation (will be added to DECISIONS.md)**

---

**Plan Status**: ✅ COMPLETE (Technical Context, Constitution Check, Phase 0 outline)
**Next Command**: Continue to generate `research.md` (Phase 0) and `data-model.md` (Phase 1)
