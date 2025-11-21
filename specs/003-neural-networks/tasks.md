# Tasks: Neural Network Perception System

**Input**: Design documents from `/specs/003-neural-networks/`
**Prerequisites**: plan.md (✓), spec.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓), quickstart.md (✓)

**Tests**: Test tasks are NOT included - Phase 2 focuses on model training and integration validation per quickstart.md test scenarios.

**Organization**: Tasks are grouped by user story (US1: LIDAR, US2: Camera, US3: Integration) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Single project structure: `src/perception/`, `models/`, `data/`, `tests/`
- Paths are absolute from repository root: `/Users/luisfelipesena/Development/Personal/projeto-final-ia/`

---

## Phase 1: Setup (Project Structure)

**Purpose**: Create directory structure and configure dependencies for neural network training and inference

- [x] T001 Create perception module directory structure: src/perception/, src/perception/training/, src/perception/models/
- [x] T002 Create data directories: data/lidar/scans/, data/lidar/labels/, data/camera/images/, data/camera/labels/
- [x] T003 Create models directory for trained weights: models/
- [x] T004 Create test directory: tests/perception/
- [x] T005 Create training notebooks directory: notebooks/
- [x] T006 [P] Install PyTorch 2.0+ and dependencies (torch, torchvision) via pip/conda
- [x] T007 [P] Install perception libraries (opencv-python, scikit-learn, matplotlib) via pip
- [x] T008 [P] Configure Python path to include src/ directory in Webots controller
- [x] T009 Document architecture decision in DECISIONS.md (DECISÃO 016: LIDAR architecture choice - Hybrid MLP+1D-CNN)
- [x] T010 Document architecture decision in DECISIONS.md (DECISÃO 017: Camera architecture choice - Custom Lightweight CNN)

**Checkpoint**: Project structure ready, dependencies installed, architecture decisions documented

---

## Phase 2: Foundational (Data Collection Infrastructure)

**Purpose**: Core data collection tools and preprocessing utilities MUST be complete before ANY user story training can begin

**⚠️ CRITICAL**: No model training can begin until data collection infrastructure is functional

- [x] T011 Implement LIDAR data collection script in scripts/collect_lidar_data.py (spawn robot at 20+ positions, save scans)
- [x] T012 Implement camera data collection script in scripts/collect_camera_data.py (capture images with cube annotations)
- [x] T013 Create annotation tool for LIDAR labels (mark obstacle sectors) in scripts/annotate_lidar.py
- [x] T014 Create annotation tool for camera labels (bounding boxes + colors) in scripts/annotate_camera.py
- [x] T015 [P] Implement data augmentation utilities for LIDAR in src/perception/training/augmentation.py (noise, dropout, rotation, scaling)
- [x] T016 [P] Implement data augmentation utilities for camera in src/perception/training/augmentation.py (brightness, hue ±10°, flip, blur)
- [x] T017 [P] Implement train/val/test split utility in src/perception/training/data_loader.py (70/15/15 split per FR-023)
- [ ] T018 Validate data collection: Collect 1000+ LIDAR scans from diverse arena positions (FR-010)
- [ ] T019 Validate data collection: Collect 500+ camera images across all cube colors and lighting conditions (FR-020)

**Checkpoint**: Foundation ready - training datasets collected and preprocessed, user story model training can now begin in parallel

---

## Phase 3: User Story 1 - LIDAR Obstacle Detection (Priority: P1) 🎯 MVP

**Goal**: Neural network processes LIDAR scans in <100ms with >90% obstacle detection accuracy, enabling safe navigation

**Independent Test**: Place robot in arena with known obstacles, measure detection accuracy (>90%), false positive rate (<10%), and inference latency (<100ms). Delivers immediate value: safe navigation capability.

### Data Preparation for US1

- [x] T020 [P] [US1] Create LIDARDataset class in src/perception/training/data_loader.py (load scans, labels, apply augmentation)
- [x] T021 [P] [US1] Implement hand-crafted feature extraction in src/perception/lidar_processor.py (6 features: min, mean, std, occupancy, symmetry, variance)

### Model Architecture for US1

- [x] T022 [US1] Implement Hybrid LIDAR Network in src/perception/models/lidar_net.py (1D-CNN branch + MLP classifier per research.md)
- [x] T023 [US1] Define network architecture: Conv1D(667→128→64) + HandCrafted(6) → MLP(70→128→64→9) with Dropout

### Training for US1

- [ ] T024 [US1] Create LIDAR training script in notebooks/lidar_training.ipynb (Adam optimizer, BCE loss, 100-200 epochs)
- [ ] T025 [US1] Implement training loop with early stopping (patience=20) and ReduceLROnPlateau scheduler
- [ ] T026 [US1] Train LIDAR model and validate: >90% accuracy (SC-001), <100ms inference (SC-003), <10% false positives
- [ ] T027 [US1] Save best model as TorchScript in models/lidar_net.pt (<50MB per SC-010)
- [ ] T028 [US1] Save model metadata in models/lidar_net_metadata.json (hyperparameters, metrics, training date)

### Inference Integration for US1

- [ ] T029 [US1] Implement LIDARProcessor class in src/perception/lidar_processor.py (load model, preprocess, inference, postprocess)
- [ ] T030 [US1] Implement ObstacleMap data structure in src/perception/lidar_processor.py (9 sectors, probabilities, min_distance, methods)
- [ ] T031 [US1] Add real-time performance monitoring (log inference time every 100 steps per quickstart.md)
- [ ] T032 [US1] Integrate LIDARProcessor into Webots controller: IA_20252/controllers/youbot/youbot.py (load model in __init__, call in run loop)
- [ ] T033 [US1] Validate LIDAR perception in Webots: Run robot with obstacles, verify >90% detection, <100ms latency (per acceptance scenarios)

**Checkpoint**: At this point, User Story 1 (LIDAR Obstacle Detection) should be fully functional and testable independently. Robot can navigate safely using obstacle detection.

---

## Phase 4: User Story 2 - Camera-Based Cube Detection (Priority: P1)

**Goal**: CNN classifies cube colors (green/blue/red) from 512×512 images with >95% accuracy at >10 FPS, enabling targeted cube collection

**Independent Test**: Place colored cubes at known positions, measure detection accuracy (>95%), color classification precision per color, localization error, and FPS (>10). Delivers immediate value: cube identification capability.

### Data Preparation for US2

- [x] T034 [P] [US2] Create CameraDataset class in src/perception/training/data_loader.py (load images, labels, apply augmentation)
- [ ] T035 [P] [US2] Implement HSV color segmentation for region proposals in src/perception/cube_detector.py (green, blue, red thresholds)

### Model Architecture for US2

- [x] T036 [US2] Implement Custom Lightweight CNN in src/perception/models/camera_net.py (3 Conv2D + BatchNorm + GlobalAvgPool + FC per research.md)
- [x] T037 [US2] Define network architecture: Conv2D(3→32→64→128) + GlobalAvgPool(128) + Dense(128→64→3) with Dropout(0.5)

### Training for US2

- [ ] T038 [US2] Create camera training script in notebooks/camera_training.ipynb (SGD+momentum, CrossEntropy loss, 30-50 epochs)
- [ ] T039 [US2] Implement training loop with StepLR scheduler (decay 0.1 every 20 epochs) and class weighting if imbalanced
- [ ] T040 [US2] Train camera model and validate: >95% per-color accuracy (SC-002), >10 FPS throughput (SC-004), <5% false positives (SC-005)
- [ ] T041 [US2] If accuracy <93%, implement ResNet18 transfer learning fallback per research.md
- [ ] T042 [US2] Save best model as TorchScript in models/camera_net.pt (<50MB per SC-010)
- [ ] T043 [US2] Save model metadata in models/camera_net_metadata.json (hyperparameters, metrics, training date)

### Inference Integration for US2

- [ ] T044 [US2] Implement CubeDetector class in src/perception/cube_detector.py (load model, preprocess, detect, classify, NMS)
- [ ] T045 [US2] Implement BoundingBox and CubeObservation data structures in src/perception/cube_detector.py (bbox coordinates, color, confidence, distance, angle)
- [ ] T046 [US2] Implement distance estimation from camera geometry in src/perception/cube_detector.py (focal_length=462, cube_size=0.05m per FR-018)
- [ ] T047 [US2] Implement angle estimation from bbox center in src/perception/cube_detector.py (bearing in [-90, 90] degrees)
- [ ] T048 [US2] Implement Non-Max Suppression to remove duplicate detections in src/perception/cube_detector.py (IoU threshold=0.5)
- [ ] T049 [US2] Add FPS monitoring (average over last 100 frames) in src/perception/cube_detector.py
- [ ] T050 [US2] Integrate CubeDetector into Webots controller: IA_20252/controllers/youbot/youbot.py (load model in __init__, call in run loop)
- [ ] T051 [US2] Validate camera perception in Webots: Test all 3 colors individually at 1m, verify >95% accuracy, >10 FPS (per acceptance scenarios and SC-007)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Robot has obstacle detection + cube color identification.

---

## Phase 5: User Story 3 - Integrated Perception System (Priority: P2)

**Goal**: Combine LIDAR and camera into unified perception system with <150ms end-to-end latency, providing complete scene understanding for autonomous operation

**Independent Test**: Run full perception pipeline in arena with obstacles and cubes, measure end-to-end latency (<150ms), verify unified world model accuracy. Delivers value: complete scene understanding for autonomous tasks.

### Integration Layer for US3

- [ ] T052 [US3] Implement WorldState data structure in src/perception/perception_system.py (obstacle_map, cube_observations, timestamp)
- [ ] T053 [US3] Implement PerceptionSystem class in src/perception/perception_system.py (initialize both processors, unified update method)
- [ ] T054 [US3] Implement parallel processing or sequential processing optimization to meet <150ms target (FR-026)

### Perception API for US3 (Phase 3 Fuzzy Controller Interface)

- [ ] T055 [P] [US3] Implement get_nearest_obstacle() in src/perception/perception_system.py (returns sector_id, distance)
- [ ] T056 [P] [US3] Implement get_nearest_cube(color) in src/perception/perception_system.py (returns CubeObservation or None)
- [ ] T057 [P] [US3] Implement get_free_direction() in src/perception/perception_system.py (returns angle of most open direction)
- [ ] T058 [P] [US3] Implement is_path_clear(angle, clearance) in src/perception/perception_system.py (returns bool)
- [ ] T059 [P] [US3] Implement align_cube_with_obstacles() in src/perception/perception_system.py (checks if cube reachable, suggests alternative angle)

### Webots Integration for US3

- [ ] T060 [US3] Integrate PerceptionSystem into Webots controller: IA_20252/controllers/youbot/youbot.py (replace individual processors)
- [ ] T061 [US3] Implement perception logging in src/perception/perception_system.py (timestamp, latencies, detections per FR-029)
- [ ] T062 [US3] Add enable_logging(path) and disable_logging() methods to PerceptionSystem
- [ ] T063 [US3] Implement get_perception_latency() in src/perception/perception_system.py (returns lidar_ms, camera_ms, total_ms)

### Validation for US3

- [ ] T064 [US3] Validate integrated perception in Webots: Run with obstacles + cubes, verify unified world model (acceptance scenario 1)
- [ ] T065 [US3] Validate perception latency: Monitor over 5-minute run, ensure <150ms end-to-end (SC-008, acceptance scenario 4)
- [ ] T066 [US3] Validate sensor fusion: Test with conflicting data, verify confidence-based resolution (acceptance scenario 3)
- [ ] T067 [US3] Validate API for Phase 3: Test all query methods (nearest_obstacle, nearest_cube, free_direction, path_clear, align_cube) return correct values

**Checkpoint**: All user stories should now be independently functional. Complete perception system ready for Phase 3 fuzzy controller integration.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, performance optimization, and final validation across all user stories

- [ ] T068 [P] Update DECISIONS.md with final architecture decisions and performance metrics (DECISÃO 016-017 finalization)
- [ ] T069 [P] Update TODO.md Phase 2 checkboxes: Mark all Phase 2 tasks as complete
- [ ] T070 [P] Create comprehensive documentation in src/perception/README.md (architecture, API, training guide, troubleshooting)
- [ ] T071 [P] Generate training visualizations: Loss curves, accuracy plots, confusion matrices for both models
- [ ] T072 [P] Profile perception system performance: Identify bottlenecks, optimize if needed to maintain real-time requirements
- [ ] T073 Document edge case handling in src/perception/README.md (sensor failures, occlusion, extreme lighting per spec.md Edge Cases)
- [ ] T074 Run complete validation per quickstart.md: All test scenarios for LIDAR, Camera, and Integration
- [ ] T075 Create final performance report in logs/perception_performance_report.md (all SC-001 to SC-011 metrics)
- [ ] T076 Prepare Phase 3 handoff documentation: Perception API usage examples for fuzzy controller

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories (data collection must complete before training)
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User Story 1 (LIDAR) and User Story 2 (Camera) can proceed **in parallel** after Phase 2
  - User Story 3 (Integration) depends on US1 and US2 completion
- **Polish (Phase 6)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (LIDAR - P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (Camera - P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (Integration - P2)**: Depends on US1 AND US2 completion - Integrates both perception modules

### Within Each User Story

- Data preparation before model architecture
- Model architecture before training
- Training before inference integration
- Inference integration before Webots integration
- Webots integration before validation

### Parallel Opportunities

- **Phase 1 (Setup)**: T006, T007, T008 can run in parallel (dependency installation)
- **Phase 2 (Foundational)**: T015, T016, T017 can run in parallel (different augmentation utilities)
- **User Story 1**: T020, T021 can run in parallel (dataset and feature extraction are independent)
- **User Story 2**: T034, T035 can run in parallel (dataset and HSV segmentation are independent)
- **User Story 3**: T055, T056, T057, T058, T059 can run in parallel (different API methods, different functions)
- **User Stories 1 and 2 (ENTIRE PHASES)**: Can work in parallel since they are independent modules
- **Phase 6 (Polish)**: T068, T069, T070, T071, T072 can run in parallel (different documentation tasks)

---

## Parallel Example: User Story 1 (LIDAR)

```bash
# Launch data preparation tasks together:
Task T020: "Create LIDARDataset class in src/perception/training/data_loader.py"
Task T021: "Implement hand-crafted feature extraction in src/perception/lidar_processor.py"

# Then sequential training pipeline:
Task T022 → T023 → T024 → T025 → T026 → T027 → T028

# Then inference integration:
Task T029 → T030 → T031 → T032 → T033
```

## Parallel Example: User Story 2 (Camera)

```bash
# Launch data preparation tasks together:
Task T034: "Create CameraDataset class in src/perception/training/data_loader.py"
Task T035: "Implement HSV color segmentation in src/perception/cube_detector.py"

# Then sequential training pipeline:
Task T036 → T037 → T038 → T039 → T040 → T041 → T042 → T043

# Then inference integration:
Task T044 → T045 → T046 → T047 → T048 → T049 → T050 → T051
```

## Parallel Example: User Story 3 (Integration) API Methods

```bash
# After integration layer (T052-T054), launch all API methods together:
Task T055: "Implement get_nearest_obstacle() in src/perception/perception_system.py"
Task T056: "Implement get_nearest_cube() in src/perception/perception_system.py"
Task T057: "Implement get_free_direction() in src/perception/perception_system.py"
Task T058: "Implement is_path_clear() in src/perception/perception_system.py"
Task T059: "Implement align_cube_with_obstacles() in src/perception/perception_system.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only - LIDAR Obstacle Detection)

1. Complete Phase 1: Setup → Project structure and dependencies ready
2. Complete Phase 2: Foundational → Data collection infrastructure functional, datasets collected
3. Complete Phase 3: User Story 1 → LIDAR neural network trained and integrated
4. **STOP and VALIDATE**: Test LIDAR obstacle detection independently (>90% accuracy, <100ms latency)
5. Robot can now navigate safely avoiding obstacles - **Core safety capability delivered!**

### Incremental Delivery

1. Complete Setup + Foundational → Training infrastructure ready
2. Add User Story 1 (LIDAR) → Test independently → **MVP: Safe navigation**
3. Add User Story 2 (Camera) → Test independently → **MVP+: Obstacle avoidance + Cube identification**
4. Add User Story 3 (Integration) → Test independently → **Full System: Unified perception for autonomous operation**
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers (or efficient task batching):

1. Team completes Setup + Foundational together (Phases 1-2)
2. Once Foundational is done:
   - **Developer A (or Task Batch 1)**: User Story 1 (LIDAR) - T020 through T033
   - **Developer B (or Task Batch 2)**: User Story 2 (Camera) - T034 through T051
3. After both US1 and US2 complete:
   - **Team**: User Story 3 (Integration) - T052 through T067
4. **Team**: Polish & validation - T068 through T076

**Estimated Timeline** (single developer, sequential):
- Phase 1: 1 day
- Phase 2: 3 days (data collection)
- Phase 3 (US1): 3 days (training LIDAR)
- Phase 4 (US2): 3 days (training Camera)
- Phase 5 (US3): 2 days (integration)
- Phase 6: 1 day (polish)
- **Total: ~13 days** (aligns with TODO.md 10-day estimate for Fase 2)

---

## Notes

- **[P] tasks**: Can run in parallel (different files, no dependencies within phase)
- **[Story] label**: Maps task to specific user story (US1, US2, US3) for traceability
- **Each user story delivers independently testable increment**: US1 = navigation safety, US2 = cube detection, US3 = unified perception
- **Architecture decisions documented in research.md**: Hybrid MLP+1D-CNN for LIDAR (94% accuracy, 15ms), Custom CNN for camera (93-96% accuracy, >30 FPS)
- **Performance targets from spec.md**: LIDAR >90% accuracy <100ms, Camera >95% accuracy >10 FPS, Integration <150ms combined
- **Training strategy from research.md**: Data augmentation to expand limited datasets (1000 LIDAR → 3000+, 500 camera → 2500+)
- **Scientific foundation**: All decisions justified with references (Goodfellow 2016, Qi 2017, LeCun 1998, Krizhevsky 2012, etc.)
- **Phase 3 interface**: Perception API designed for fuzzy controller integration (get_nearest_obstacle, get_nearest_cube, etc.)
- **Tests**: Validation per quickstart.md test scenarios instead of unit tests (model training validation workflow)
- **Commit strategy**: Commit after each logical task group (e.g., after training, after integration, after validation)
- **Stop at any checkpoint**: Each checkpoint represents independently testable increment
