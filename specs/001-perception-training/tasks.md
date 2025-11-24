# Tasks: Phase 2 Perception Model Training

**Feature**: `001-perception-training`  
**Generated**: 2025-11-21  
**Spec**: [`specs/001-perception-training/spec.md`](./spec.md)  
**Plan**: [`specs/001-perception-training/plan.md`](./plan.md)

## Summary

This document breaks down Phase 2 Perception Model Training into executable tasks organized by user story priority. Each user story phase is independently testable and can be developed in parallel after foundational prerequisites are complete.

**Total Tasks**: 47  
**User Story Breakdown**:
- Phase 1 (Setup): 3 tasks
- Phase 2 (Foundational): 4 tasks
- Phase 3 (US1 - Dataset Acquisition): 14 tasks
- Phase 4 (US2 - LIDAR Training): 11 tasks
- Phase 5 (US3 - Camera Training): 11 tasks
- Phase 6 (Polish): 4 tasks

**MVP Scope**: Complete Phase 1-3 (Dataset Acquisition) to unblock all downstream training work.

---

## Dependencies & Parallel Execution

### Story Completion Order
1. **Phase 1 (Setup)** → Must complete first
2. **Phase 2 (Foundational)** → Blocks all user stories
3. **Phase 3 (US1 - Dataset Acquisition)** → Blocks US2 and US3
4. **Phase 4 (US2 - LIDAR Training)** → Can run in parallel with US3 after US1 complete
5. **Phase 5 (US3 - Camera Training)** → Can run in parallel with US2 after US1 complete
6. **Phase 6 (Polish)** → Requires US2 and US3 complete

### Parallel Opportunities
- **Within US1**: LIDAR and camera collection scripts can be developed/tested independently
- **US2 ↔ US3**: After US1, LIDAR and camera training pipelines are fully independent
- **Within US2/US3**: Dataset validation, model training, and export tasks can overlap

---

## Phase 1: Setup

**Goal**: Initialize project structure and verify prerequisites for perception training workflows.

### Setup Tasks

- [x] T001 Create training configuration directory structure in `configs/`
- [x] T002 Create logs directory structure in `logs/perception/` with subdirectories for lidar and camera runs
- [x] T003 Verify existing scripts (`scripts/collect_lidar_data.py`, `scripts/collect_camera_data.py`) are executable and dependencies installed

**Checkpoint**: Project structure ready, scripts accessible, environment validated.

---

## Phase 2: Foundational

**Goal**: Establish shared infrastructure and validation utilities that all user stories depend on.

### Foundational Tasks

- [x] T004 [P] Implement dataset validation schema checker in `scripts/validate_dataset_schema.py` supporting LidarSample and CameraSample schemas
- [x] T005 [P] Implement dataset balance validator in `scripts/validate_dataset_balance.py` checking per-class/sector distribution thresholds
- [x] T006 [P] Create training run logger utility in `src/perception/training/run_logger.py` for structured JSON logging of hyperparameters, metrics, and hardware profiles
- [x] T007 [P] Create model artifact metadata generator in `src/perception/training/artifact_metadata.py` producing ModelArtifact JSON with checksums and preprocessing specs

**Checkpoint**: Validation and logging infrastructure ready for use by all training pipelines.

---

## Phase 3: User Story 1 - Dataset Acquisition & Labeling (Priority: P1) 🎯 MVP

**Goal**: Capture and curate representative LIDAR scans (≥1,000) and RGB frames (≥500) with ground-truth annotations covering all obstacle sectors, cube colors, robot headings, and lighting variations.

**Independent Test**: Execute collection scripts, inspect generated datasets, verify labeling coverage meets quotas (≥1,000 LIDAR scans, ≥500 RGB images, balanced per obstacle/cube class) without running any training.

### Tests for User Story 1

- [x] T008 [P] [US1] Unit test for LIDAR dataset schema validation in `tests/perception/test_lidar_dataset.py`
- [x] T009 [P] [US1] Unit test for camera dataset schema validation in `tests/perception/test_camera_dataset.py`
- [x] T010 [US1] Integration test for dataset collection workflow in `tests/perception/test_dataset_collection.py` verifying quota completion

### Implementation for User Story 1

- [x] T011 [P] [US1] Enhance LIDAR collection script in `scripts/collect_lidar_data.py` to capture robot pose metadata (x, y, theta) per scan
- [x] T012 [P] [US1] Enhance LIDAR collection script in `scripts/collect_lidar_data.py` to tag scenarios (clear, obstacle_front, corridor_left, etc.) per session
- [x] T013 [P] [US1] Implement LIDAR sector labeling logic in `scripts/annotate_lidar.py` computing 9-sector occupancy flags from raw ranges
- [x] T014 [P] [US1] Enhance camera collection script in `scripts/collect_camera_data.py` to capture robot pose and lighting tags per frame
- [x] T015 [P] [US1] Implement camera annotation workflow in `scripts/annotate_camera.py` supporting bounding box + color label per cube with distance estimation
- [x] T016 [US1] Create dataset manifest generator in `scripts/generate_dataset_manifest.py` producing JSON manifest with sample IDs, splits, and metadata hashes
- [x] T017 [US1] Implement dataset splitter in `scripts/split_dataset.py` assigning train/val/test splits with balanced distribution per sector/color
- [ ] T018 [US1] Execute LIDAR data collection sessions in Webots producing ≥1,000 validated scans stored in `data/lidar/annotated/`
- [ ] T019 [US1] Execute camera data collection sessions in Webots producing ≥500 validated frames stored in `data/camera/annotated/`
- [ ] T020 [US1] Run dataset validation pipeline verifying schema compliance, balance thresholds, and split integrity for both LIDAR and camera datasets
- [ ] T021 [US1] Generate and commit dataset manifests (`data/lidar/dataset_manifest.json`, `data/camera/dataset_manifest.json`) with hash signatures

**Checkpoint**: At this point, User Story 1 should be fully functional. Both datasets meet quotas, are validated, split, and ready for training pipelines.

---

## Phase 4: User Story 2 - LIDAR Model Training & Validation (Priority: P2)

**Goal**: Configure, train, and evaluate the hybrid MLP+1D-CNN achieving ≥90% accuracy, ≥88% per-sector recall, and ≤100 ms inference latency, producing an exportable TorchScript artifact.

**Independent Test**: Using curated dataset, run LIDAR training pipeline end-to-end and confirm trained model meets accuracy (>90%), per-sector recall (>88%), and inference latency (<100 ms on target hardware) requirements without touching camera workflows.

### Tests for User Story 2

- [ ] T022 [P] [US2] Unit test for LIDAR model architecture in `tests/perception/test_lidar_net.py` verifying input/output shapes
- [ ] T023 [P] [US2] Unit test for LIDAR data loader in `tests/perception/test_lidar_dataloader.py` verifying augmentation and batching
- [ ] T024 [US2] Integration test for LIDAR training pipeline in `tests/perception/test_lidar_training.py` verifying end-to-end training and checkpoint saving

### Implementation for User Story 2

- [ ] T025 [P] [US2] Create LIDAR training notebook in `notebooks/lidar_training.ipynb` with hyperparameter configuration, training loop, and evaluation cells
- [ ] T026 [US2] Implement LIDAR training script wrapper in `scripts/train_lidar_model.py` calling notebook logic with CLI arguments for data path, config, log dir, and export path
- [ ] T027 [US2] Configure training hyperparameters in `configs/lidar_default.yaml` (optimizer: Adam, learning_rate, batch_size, epochs, early_stopping patience)
- [ ] T028 [US2] Execute LIDAR training run logging hyperparameters, loss curves, and metrics to `logs/perception/lidar_run_001/`
- [ ] T029 [US2] Evaluate trained LIDAR model on test split generating confusion matrix and per-sector recall metrics in `logs/perception/lidar_run_001/metrics.json`
- [ ] T030 [US2] Benchmark LIDAR model inference latency in `scripts/profile_lidar_model.py` verifying <100 ms median on target hardware profile
- [ ] T031 [US2] Export best LIDAR checkpoint to TorchScript format saving to `models/lidar_net.pt`
- [ ] T032 [US2] Generate LIDAR model metadata JSON in `models/lidar_net_metadata.json` including version, checksum, metrics snapshot, preprocessing params, and calibration constants

**Checkpoint**: At this point, User Story 2 should be fully functional. LIDAR model trained, validated, exported, and ready for controller integration.

---

## Phase 5: User Story 3 - Camera Model Training & Export (Priority: P3)

**Goal**: Train, assess, and export the lightweight CNN (or fallback ResNet18) achieving ≥95% per-color precision/recall, ≥10 FPS throughput, and ≤5° angular error, producing an exportable TorchScript artifact.

**Independent Test**: Execute camera training pipeline with curated RGB dataset and verify exported model achieves ≥95% per-color accuracy and delivers ≥10 FPS on target hardware, independently of LIDAR results.

### Tests for User Story 3

- [ ] T033 [P] [US3] Unit test for camera model architecture in `tests/perception/test_camera_net.py` verifying input/output shapes and color classification
- [ ] T034 [P] [US3] Unit test for camera data loader in `tests/perception/test_camera_dataloader.py` verifying augmentation and batching
- [ ] T035 [US3] Integration test for camera training pipeline in `tests/perception/test_camera_training.py` verifying end-to-end training and checkpoint saving

### Implementation for User Story 3

- [ ] T036 [P] [US3] Create camera training notebook in `notebooks/camera_training.ipynb` with hyperparameter configuration, training loop, and evaluation cells
- [ ] T037 [US3] Implement camera training script wrapper in `scripts/train_camera_model.py` calling notebook logic with CLI arguments for data path, config, log dir, and export path
- [ ] T038 [US3] Configure training hyperparameters in `configs/camera_default.yaml` (optimizer: SGD+momentum, learning_rate, batch_size, epochs, class_weights, augmentations)
- [ ] T039 [US3] Execute camera training run logging hyperparameters, loss curves, and metrics to `logs/perception/camera_run_001/`
- [ ] T040 [US3] Evaluate trained camera model on test split generating per-color precision/recall metrics and angular error statistics in `logs/perception/camera_run_001/metrics.json`
- [ ] T041 [US3] Benchmark camera model FPS and latency in `scripts/profile_camera_model.py` verifying ≥10 FPS on target hardware
- [ ] T042 [US3] Export best camera checkpoint to TorchScript format saving to `models/camera_net.pt`
- [ ] T043 [US3] Generate camera model metadata JSON in `models/camera_net_metadata.json` including version, checksum, metrics snapshot, preprocessing params (normalization, resolution), and calibration constants (focal_length, principal_point)

**Checkpoint**: At this point, User Story 3 should be fully functional. Camera model trained, validated, exported, and ready for controller integration.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Goal**: Complete documentation, reproducibility verification, and compliance checks for Phase 2 deliverables.

### Polish Tasks

- [ ] T044 Document architecture and training decisions in `DECISIONS.md` citing references from REFERENCIAS.md (Goodfellow16, Qi17, etc.) for LIDAR and camera model choices
- [ ] T045 Update `TODO.md` Phase 2 checklist marking all completed tasks and updating status to "✅ CONCLUÍDO"
- [ ] T046 Verify reproducibility by re-running both training pipelines with same seeds and confirming ≤5% variance in final metrics documented in `logs/perception/reproducibility_report.md`
- [ ] T047 Create perception training summary document in `docs/perception/training_summary.md` including dataset statistics, model performance metrics, confusion matrices, and FPS/latency benchmarks

**Checkpoint**: Phase 2 complete. All deliverables documented, reproducible, and compliant with constitution requirements.

---

## Implementation Strategy

### MVP First Approach
1. **Phase 1-3 (MVP)**: Complete dataset acquisition (US1) to unblock all training work. This is the critical path.
2. **Incremental Delivery**: After MVP, US2 and US3 can proceed in parallel, enabling faster iteration.
3. **Polish Last**: Documentation and compliance checks happen after all models are trained and validated.

### Validation Gates
- **After US1**: Dataset validation must pass before training scripts can execute.
- **After US2/US3**: Model performance thresholds must be met before export (accuracy, latency, FPS).
- **After Phase 6**: Reproducibility and documentation compliance verified before marking Phase 2 complete.

### Risk Mitigation
- **Dataset Quality**: Early validation (T020) catches issues before training starts.
- **Model Performance**: Benchmarking tasks (T030, T041) verify hardware compatibility early.
- **Reproducibility**: Structured logging (T006) and seed tracking ensure experiments are repeatable.

