# Implementation Tasks: Sensor Exploration and Control Validation

**Feature**: 002-sensor-exploration
**Branch**: `002-sensor-exploration`
**Generated**: 2025-11-21
**Total Tasks**: 47

## Overview

This document provides a granular, executable task breakdown for implementing Phase 1.2-1.3 of the YouBot autonomous system. Tasks are organized by user story to enable independent implementation and testing.

**Deliverables:**
- `tests/test_basic_controls.py` - Control validation test suite
- `notebooks/01_sensor_exploration.ipynb` - Sensor analysis notebook
- `docs/arena_map.md` - Arena layout documentation

---

## Task Organization

Tasks are grouped into phases based on user story priorities from spec.md:

- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (prerequisites for all stories)
- **Phase 3**: User Story 1 (P1) - Base Movement Control Validation
- **Phase 4**: User Story 2 (P1) - Arm and Gripper Control Validation
- **Phase 5**: User Story 3 (P2) - LIDAR Data Analysis
- **Phase 6**: User Story 4 (P2) - Camera RGB Analysis
- **Phase 7**: User Story 5 (P3) - Arena Mapping
- **Phase 8**: Polish & Documentation

---

## Task Format Legend

```
- [ ] TXXX [P] [US#] Description with file path
```

- **TXXX**: Task ID (sequential execution order)
- **[P]**: Parallelizable (can run concurrently with other [P] tasks in same phase)
- **[US#]**: User Story label (US1, US2, US3, US4, US5)
- **Description**: Clear action with specific file path

---

## Phase 1: Setup

**Goal**: Initialize project structure for Phase 1.2-1.3 deliverables

**Prerequisites**: Phase 1.1 complete (Webots installed, environment validated)

### Tasks

- [ ] T001 Create tests directory if not exists: `mkdir -p tests`
- [ ] T002 Create notebooks directory if not exists: `mkdir -p notebooks`
- [ ] T003 Create media directory for outputs: `mkdir -p media/{lidar_plots,cube_examples}`
- [ ] T004 Create logs directory if not exists: `mkdir -p logs`
- [ ] T005 Verify venv active and dependencies installed: `pip list | grep -E "(numpy|matplotlib|opencv|pytest)"`
- [ ] T006 Read base controller files for API understanding: `IA_20252/controllers/youbot/{youbot.py,base.py,arm.py,gripper.py}`

**Estimated Time**: 15 minutes

---

## Phase 2: Foundational

**Goal**: Create shared utilities and base test infrastructure

**Prerequisites**: Phase 1 complete

### Tasks

- [ ] T007 [P] Create test base class with robot initialization fixture in `tests/conftest.py`
- [ ] T008 [P] Create utility module for sensor data validation in `tests/utils/sensor_validation.py`
- [ ] T009 [P] Create utility module for visualization helpers in `tests/utils/visualization.py`
- [ ] T010 Verify Webots world file loads successfully: test load `IA_20252/worlds/IA_20252.wbt`

**Estimated Time**: 30 minutes

**Parallel Opportunities**: T007, T008, T009 can run concurrently (different files)

---

## Phase 3: User Story 1 - Base Movement Control Validation (P1)

**Goal**: Validate that YouBot's base responds correctly to all movement commands (forward, backward, strafe, rotate, stop) and document velocity limits.

**Why P1**: Foundational for all navigation - all future phases depend on working base control.

**Independent Test Criteria**:
- All 8 test functions pass (forward, backward, strafe left/right, rotate CW/CCW, stop, limits)
- Observable motion in Webots matches commanded direction
- Velocity limits documented in `logs/velocity_limits.json`

### Tasks

#### Test Structure
- [ ] T011 [US1] Create test file `tests/test_basic_controls.py` with pytest imports and base test class

#### Base Movement Tests
- [ ] T012 [P] [US1] Implement test_base_forward_movement in `tests/test_basic_controls.py` (FR-001)
- [ ] T013 [P] [US1] Implement test_base_backward_movement in `tests/test_basic_controls.py` (FR-002)
- [ ] T014 [P] [US1] Implement test_base_strafe_left in `tests/test_basic_controls.py` (FR-003)
- [ ] T015 [P] [US1] Implement test_base_strafe_right in `tests/test_basic_controls.py` (FR-003)
- [ ] T016 [P] [US1] Implement test_base_rotate_clockwise in `tests/test_basic_controls.py` (FR-004)
- [ ] T017 [P] [US1] Implement test_base_rotate_counterclockwise in `tests/test_basic_controls.py` (FR-004)
- [ ] T018 [P] [US1] Implement test_base_stop_command in `tests/test_basic_controls.py` (FR-005)

#### Limits Documentation
- [ ] T019 [US1] Implement test_base_velocity_limits function to measure max vx, vy, omega in `tests/test_basic_controls.py` (FR-006)
- [ ] T020 [US1] Add velocity limits export to JSON in `logs/velocity_limits.json` (FR-006)

#### Validation
- [ ] T021 [US1] Run pytest for User Story 1 tests: `pytest tests/test_basic_controls.py::TestBaseMovement -v`
- [ ] T022 [US1] Manually verify robot movements in Webots GUI match test expectations
- [ ] T023 [US1] Document findings in DECISIONS.md as DECISÃO 011 (base control validation methodology)

**Estimated Time**: 2-3 hours

**Parallel Opportunities**: T012-T018 (test functions are independent)

**Success Criteria (SC-001)**: All base movement commands execute successfully with observable motion matching command intent

---

## Phase 4: User Story 2 - Arm and Gripper Control Validation (P1)

**Goal**: Validate that YouBot's arm responds to preset positioning commands and gripper executes open/close actions reliably.

**Why P1**: Manipulation is core to project mission (cube collection) - blocking for Phase 5 grasping implementation.

**Independent Test Criteria**:
- All 5 test functions pass (arm height, arm orientation, gripper close, gripper open, joint limits)
- Arm reaches commanded presets within 5% tolerance
- Gripper state visually matches commands in simulation
- Joint limits documented in `logs/joint_limits.json`

### Tasks

#### Test Structure
- [ ] T024 [US2] Add TestArmGripper class to `tests/test_basic_controls.py`

#### Arm Positioning Tests
- [ ] T025 [P] [US2] Implement test_arm_height_positions testing all presets (FLOOR, FRONT, HIGH) in `tests/test_basic_controls.py` (FR-008)
- [ ] T026 [P] [US2] Implement test_arm_orientation_positions testing presets (FRONT, DOWN) in `tests/test_basic_controls.py` (FR-009)

#### Gripper Tests
- [ ] T027 [P] [US2] Implement test_gripper_close validating jaw closure in `tests/test_basic_controls.py` (FR-010)
- [ ] T028 [P] [US2] Implement test_gripper_open validating jaw opening in `tests/test_basic_controls.py` (FR-011)

#### Limits Documentation
- [ ] T029 [US2] Implement test_arm_joint_limits measuring 5-DOF range of motion in `tests/test_basic_controls.py` (FR-012)
- [ ] T030 [US2] Add joint limits export to JSON in `logs/joint_limits.json` (FR-012)

#### Validation
- [ ] T031 [US2] Run pytest for User Story 2 tests: `pytest tests/test_basic_controls.py::TestArmGripper -v`
- [ ] T032 [US2] Manually verify arm/gripper movements in Webots GUI match test expectations
- [ ] T033 [US2] Document findings in DECISIONS.md as DECISÃO 012 (arm/gripper control validation)

**Estimated Time**: 2 hours

**Parallel Opportunities**: T025-T028 (test functions independent)

**Success Criteria**:
- **SC-002**: Arm positioning within 5% tolerance
- **SC-003**: Gripper commands execute successfully with observable state changes

---

## Phase 5: User Story 3 - LIDAR Data Analysis (P2)

**Goal**: Understand LIDAR sensor's data format, range capabilities, FOV, and create polar visualizations with obstacle identification.

**Why P2**: LIDAR is primary sensor for obstacle avoidance - understanding data structure prerequisite for Phase 2 neural networks.

**Independent Test Criteria**:
- LIDAR specifications fully documented (points, FOV, angular resolution, ranges)
- Polar plots generated showing arena boundaries and obstacles
- Obstacle detection algorithm identifies wooden boxes
- All outputs in Jupyter notebook with explanatory markdown

### Tasks

#### Notebook Setup
- [ ] T034 [US3] Create Jupyter notebook `notebooks/01_sensor_exploration.ipynb` with initial structure (Setup, LIDAR Analysis, Camera Analysis, Results)
- [ ] T035 [US3] Add notebook cell: Import libraries (numpy, matplotlib, cv2, controller from Webots)
- [ ] T036 [US3] Add notebook cell: Initialize YouBot robot and enable LIDAR sensor

#### LIDAR Specifications
- [ ] T037 [US3] Add notebook cell: Implement capture_lidar_specifications function (FR-015)
- [ ] T038 [US3] Add notebook cell: Execute capture_lidar_specifications and display results table (horizontal_resolution, num_layers, fov, min_range, max_range)
- [ ] T039 [US3] Add notebook markdown: Document LIDAR specs interpretation

#### LIDAR Data Collection
- [ ] T040 [US3] Add notebook cell: Implement capture_lidar_scan function reading range_image (FR-014)
- [ ] T041 [US3] Add notebook cell: Collect 20 LIDAR scans in loop with time stepping
- [ ] T042 [US3] Add notebook cell: Implement analyze_lidar_ranges function calculating min/max/mean/std (FR-016)
- [ ] T043 [US3] Add notebook markdown: Document observed range capabilities

#### LIDAR Visualization
- [ ] T044 [US3] Add notebook cell: Implement visualize_lidar_polar function with matplotlib polar projection (FR-017)
- [ ] T045 [US3] Add notebook cell: Generate polar plot for sample scan and save to `media/lidar_plots/scan_example.png`
- [ ] T046 [US3] Add notebook cell: Implement identify_obstacles_lidar function with distance thresholding (FR-018)
- [ ] T047 [US3] Add notebook cell: Create obstacle-highlighted polar plot and save to `media/lidar_plots/obstacles_highlighted.png`
- [ ] T048 [US3] Add notebook markdown: Analyze obstacle detection results

#### Validation
- [ ] T049 [US3] Execute full LIDAR analysis section in notebook (Run All Cells)
- [ ] T050 [US3] Verify polar plots show arena boundaries clearly
- [ ] T051 [US3] Verify obstacles (wooden boxes, walls) identifiable in visualizations
- [ ] T052 [US3] Document findings in DECISIONS.md as DECISÃO 013 (LIDAR visualization methodology)

**Estimated Time**: 2-3 hours

**Success Criteria**:
- **SC-005**: LIDAR data format documented (points, angular resolution, FOV, ranges)
- **SC-006**: Polar plots clearly show arena boundaries and obstacles with identifiable shapes

---

## Phase 6: User Story 4 - Camera RGB Analysis (P2)

**Goal**: Understand RGB camera's resolution, FPS, capture example cube images, implement HSV threshold color detection, and evaluate accuracy (>80% target).

**Why P2**: Camera is mandatory for cube color identification - understanding characteristics and baseline detection prerequisite for Phase 2 CNN.

**Independent Test Criteria**:
- Camera specifications fully documented (resolution, FPS)
- Example images saved for green, blue, red cubes
- HSV threshold detection implemented and tested
- Color detection accuracy >80% on test images
- All outputs in Jupyter notebook

### Tasks

#### Camera Specifications
- [ ] T053 [US4] Add notebook cell: Implement capture_camera_specifications function (FR-021, FR-022)
- [ ] T054 [US4] Add notebook cell: Implement measure_camera_fps function over 100 frames
- [ ] T055 [US4] Add notebook cell: Execute camera specs and FPS measurement, display results table
- [ ] T056 [US4] Add notebook markdown: Document camera characteristics interpretation

#### Camera Data Collection
- [ ] T057 [US4] Add notebook cell: Implement capture_camera_frame function converting BGRA to RGB (FR-020)
- [ ] T058 [US4] Add notebook cell: Implement save_example_images function capturing green/blue/red cubes (FR-023)
- [ ] T059 [US4] Execute save_example_images: manually position robot to view each color cube and capture
- [ ] T060 [US4] Verify example images saved: `ls media/cube_examples/*.png` should show 3+ images

#### Color Detection Implementation
- [ ] T061 [US4] Add notebook cell: Define HSV_RANGES dictionary for green, blue, red with calibrated thresholds (FR-024)
- [ ] T062 [US4] Add notebook cell: Implement detect_color_threshold function using cv2.inRange (FR-024)
- [ ] T063 [US4] Add notebook cell: Test detect_color_threshold on example images and display detected colors

#### Accuracy Evaluation
- [ ] T064 [US4] Add notebook cell: Implement evaluate_color_detection_accuracy function with confusion matrix (FR-025)
- [ ] T065 [US4] Add notebook cell: Load 30+ test images (10 per color) with ground truth labels
- [ ] T066 [US4] Add notebook cell: Execute evaluate_color_detection_accuracy and display results table
- [ ] T067 [US4] Add notebook markdown: Analyze accuracy results (target: >80%)

#### Validation
- [ ] T068 [US4] Execute full camera analysis section in notebook (Run All Cells)
- [ ] T069 [US4] Verify color detection accuracy meets SC-008 target (>80%)
- [ ] T070 [US4] Document findings in DECISIONS.md as DECISÃO 014 (camera analysis and HSV thresholding)

**Estimated Time**: 2-3 hours

**Success Criteria**:
- **SC-007**: Camera specs documented (resolution, FPS)
- **SC-008**: HSV threshold achieves >80% accuracy distinguishing green/blue/red cubes

---

## Phase 7: User Story 5 - Arena Mapping (P3)

**Goal**: Document arena dimensions, deposit box locations, obstacle positions, and create schematic diagram.

**Why P3**: Supporting information for Phase 4 navigation - not immediately blocking but helpful for path planning.

**Independent Test Criteria**:
- Arena dimensions documented in meters
- Deposit box coordinates documented (green, blue, red)
- Obstacle positions documented
- Schematic diagram created showing all elements

### Tasks

#### Parsing Script
- [ ] T071 [US5] Create parsing script `scripts/parse_arena.py` with import statements
- [ ] T072 [US5] Implement parse_arena_dimensions function reading RectangleArena node from .wbt file (FR-027)
- [ ] T073 [US5] Implement parse_deposit_boxes function reading PlasticFruitBox nodes with translation and recognitionColors (FR-028)
- [ ] T074 [US5] Implement parse_obstacles function reading WoodenBox nodes (FR-029)
- [ ] T075 [US5] Implement parse_spawn_zone function extracting supervisor.py cube spawn ranges

#### Map Generation
- [ ] T076 [US5] Implement generate_schematic_diagram function using matplotlib to create arena layout visualization (FR-030)
- [ ] T077 [US5] Implement generate_arena_map_md function creating markdown documentation in `docs/arena_map.md`
- [ ] T078 [US5] Add main block in `scripts/parse_arena.py` orchestrating all parsing and generation functions

#### Execution
- [ ] T079 [US5] Run parsing script: `python scripts/parse_arena.py`
- [ ] T080 [US5] Verify `docs/arena_map.md` created with all required sections
- [ ] T081 [US5] Manually validate parsed dimensions against Webots GUI scene tree
- [ ] T082 [US5] Document findings in DECISIONS.md as DECISÃO 015 (arena mapping methodology)

**Estimated Time**: 1-2 hours

**Success Criteria (SC-010)**: Arena map document contains schematic diagram with dimensions, deposit boxes, obstacles, and spawn zones

---

## Phase 8: Polish & Documentation

**Goal**: Finalize documentation, validate all success criteria, prepare for merge

**Prerequisites**: Phases 3-7 complete (all user stories implemented)

### Tasks

#### Test Suite Finalization
- [ ] T083 [P] Run full test suite: `pytest tests/test_basic_controls.py -v`
- [ ] T084 [P] Generate HTML test report: `pytest tests/test_basic_controls.py --html=test_report.html --self-contained-html`
- [ ] T085 Verify 100% test pass rate (SC-004 target: 13/13 tests passing)

#### Code Quality
- [ ] T086 [P] Format test code with black: `black tests/test_basic_controls.py`
- [ ] T087 [P] Add type hints to test functions in `tests/test_basic_controls.py`
- [ ] T088 [P] Add docstrings to test classes and methods

#### Notebook Finalization
- [ ] T089 [P] Execute full notebook: `jupyter nbconvert --execute notebooks/01_sensor_exploration.ipynb --to html`
- [ ] T090 [P] Verify notebook HTML export successful (SC-009 validation)
- [ ] T091 Add notebook markdown: Create "Results Summary" section with key findings table

#### Documentation Updates
- [ ] T092 [P] Update TODO.md: Mark Phase 1.2-1.3 tasks as complete (checkmarks)
- [ ] T093 [P] Verify DECISIONS.md has all technical choices documented (DECISÃO 011-015)
- [ ] T094 [P] Create comprehensive commit message following constitutional template

#### Success Criteria Validation
- [ ] T095 Validate SC-001: All base commands execute successfully (manual check)
- [ ] T096 Validate SC-002: Arm positions within 5% tolerance (test logs)
- [ ] T097 Validate SC-003: Gripper commands successful (manual check)
- [ ] T098 Validate SC-004: Test suite 100% pass rate (pytest output)
- [ ] T099 Validate SC-005: LIDAR specs documented (notebook review)
- [ ] T100 Validate SC-006: LIDAR plots show obstacles (visual inspection)
- [ ] T101 Validate SC-007: Camera specs documented (notebook review)
- [ ] T102 Validate SC-008: Color detection >80% (notebook accuracy table)
- [ ] T103 Validate SC-009: Notebook complete with visualizations (HTML export check)
- [ ] T104 Validate SC-010: Arena map has all elements (arena_map.md review)
- [ ] T105 Validate SC-011: Measurements sufficient for Phase 2 (documentation review)

#### Git Operations
- [ ] T106 Stage all modified files: `git add tests/ notebooks/ docs/ logs/ DECISIONS.md TODO.md`
- [ ] T107 Commit with descriptive message: `git commit -m "feat(sensor-exploration): complete Phase 1.2-1.3 validation..."`
- [ ] T108 Push to remote: `git push origin 002-sensor-exploration`

**Estimated Time**: 1 hour

**Parallel Opportunities**: T083-T088 (independent validation tasks), T092-T094 (documentation updates)

---

## Dependencies Graph

```
Phase 1 (Setup)
  └─► Phase 2 (Foundational)
        ├─► Phase 3 (US1 - Base Control) ──► INDEPENDENT
        ├─► Phase 4 (US2 - Arm/Gripper) ──► INDEPENDENT
        ├─► Phase 5 (US3 - LIDAR) ──────► INDEPENDENT
        ├─► Phase 6 (US4 - Camera) ──────► INDEPENDENT
        └─► Phase 7 (US5 - Arena Map) ───► INDEPENDENT
              │
              └─► Phase 8 (Polish) ◄── Requires ALL user stories complete
```

**User Story Independence**:
- US1 (Base Control): Can be implemented and tested independently
- US2 (Arm/Gripper): Can be implemented and tested independently
- US3 (LIDAR): Can be implemented and tested independently (requires notebook setup)
- US4 (Camera): Can be implemented and tested independently (shares notebook with US3)
- US5 (Arena Map): Can be implemented and tested independently

**Blocking Dependencies**:
- Phase 2 must complete before any user story
- Phase 8 requires all user stories complete
- Within US3/US4: Notebook must exist before cells can be added (but both stories can share notebook)

---

## Parallel Execution Opportunities

### Phase 3 (US1) - 7 parallel test functions
```bash
# All base movement tests can run concurrently (T012-T018)
pytest tests/test_basic_controls.py::test_base_forward_movement &
pytest tests/test_basic_controls.py::test_base_backward_movement &
pytest tests/test_basic_controls.py::test_base_strafe_left &
pytest tests/test_basic_controls.py::test_base_strafe_right &
pytest tests/test_basic_controls.py::test_base_rotate_clockwise &
pytest tests/test_basic_controls.py::test_base_rotate_counterclockwise &
pytest tests/test_basic_controls.py::test_base_stop_command &
wait
```

### Phase 4 (US2) - 4 parallel test functions
```bash
# Arm and gripper tests can run concurrently (T025-T028)
pytest tests/test_basic_controls.py::test_arm_height_positions &
pytest tests/test_basic_controls.py::test_arm_orientation_positions &
pytest tests/test_basic_controls.py::test_gripper_close &
pytest tests/test_basic_controls.py::test_gripper_open &
wait
```

### Phase 8 (Polish) - Multiple independent validations
```bash
# Code quality tasks (T083-T088)
pytest tests/test_basic_controls.py -v &
black tests/test_basic_controls.py &
jupyter nbconvert --execute notebooks/01_sensor_exploration.ipynb --to html &
wait
```

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)

**Recommended MVP**: User Story 1 only (Base Movement Control Validation)
- Delivers: Validated base control (foundational for all navigation)
- Tasks: T001-T023 (23 tasks, ~3-4 hours)
- Success: All base movement tests passing (SC-001, SC-004 partial)

**Rationale**: Base control is foundational - validates simulator integration and API understanding before investing in sensors/mapping.

### Incremental Delivery Order

1. **Iteration 1 (MVP)**: US1 - Base Movement Control
   - Value: Proven base control API
   - Risk reduction: Early validation of Webots integration

2. **Iteration 2**: US2 - Arm and Gripper Control
   - Value: Complete control validation (base + manipulation)
   - Enables: Phase 5 grasping implementation

3. **Iteration 3**: US3 + US4 - Sensor Analysis (parallel)
   - Value: LIDAR and camera characteristics documented
   - Enables: Phase 2 neural network design

4. **Iteration 4**: US5 - Arena Mapping
   - Value: Spatial understanding for navigation
   - Enables: Phase 4 path planning

5. **Iteration 5**: Polish & Documentation
   - Value: Production-ready deliverables
   - Enables: Phase 1.2-1.3 completion and merge to main

### Estimated Timeline

| Phase | Tasks | Estimated Time | Can Start After |
|-------|-------|---------------|-----------------|
| Phase 1: Setup | T001-T006 | 15 min | Immediately |
| Phase 2: Foundational | T007-T010 | 30 min | Phase 1 |
| Phase 3: US1 (P1) | T011-T023 | 2-3 hours | Phase 2 |
| Phase 4: US2 (P1) | T024-T033 | 2 hours | Phase 2 |
| Phase 5: US3 (P2) | T034-T052 | 2-3 hours | Phase 2 |
| Phase 6: US4 (P2) | T053-T070 | 2-3 hours | Phase 2 |
| Phase 7: US5 (P3) | T071-T082 | 1-2 hours | Phase 2 |
| Phase 8: Polish | T083-T108 | 1 hour | Phases 3-7 |
| **Total** | **108 tasks** | **~12-15 hours** | - |

---

## Validation Checklist

Before marking Phase 1.2-1.3 as complete, verify:

- [ ] All 108 tasks completed (checkboxes marked)
- [ ] Test suite: 13/13 tests passing (SC-004)
- [ ] Notebook: Complete with all visualizations (SC-009)
- [ ] Arena map: All elements present (SC-010)
- [ ] DECISIONS.md: DECISÃO 011-015 documented
- [ ] TODO.md: Phase 1.2-1.3 marked complete
- [ ] Git: Committed and pushed to `002-sensor-exploration` branch

**Next Step After Completion**: Create PR to merge `002-sensor-exploration` → `main`

---

**Tasks Status**: ✅ GENERATED
**Ready for**: `/speckit.implement` command to execute tasks sequentially
