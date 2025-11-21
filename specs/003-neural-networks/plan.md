# Implementation Plan: Neural Network Perception System

**Branch**: `003-neural-networks` | **Date**: 2025-11-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-neural-networks/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement neural network perception system for YouBot autonomous robot using LIDAR obstacle detection and camera-based cube color classification. System processes LIDAR point clouds (667 points, 270° FOV) through neural network for obstacle mapping and camera images (512×512 RGB) through CNN for cube detection (green/blue/red). Technical approach: MLP or PointNet for LIDAR, custom CNN or pre-trained model for camera, PyTorch for training/inference, real-time integration in Webots controller. Target performance: >90% obstacle detection, >95% color classification, <100ms inference time, >10 FPS camera processing.

## Technical Context

**Language/Version**: Python 3.8+ (Webots R2023b controller requirement)
**Primary Dependencies**: PyTorch 2.0+, NumPy, SciPy, OpenCV, Matplotlib, scikit-learn
**Storage**: File-based (models/*.pth for trained models, data/ for datasets, logs/ for experiment tracking)
**Testing**: pytest (unit tests), custom validation scripts (accuracy metrics), Webots integration tests
**Target Platform**: Webots R2023b simulator on macOS/Linux, CPU inference (GPU optional for training)
**Project Type**: Single (robotics perception system integrated into existing youbot controller)
**Performance Goals**:
  - LIDAR: <100ms inference time, >90% obstacle detection accuracy
  - Camera: >10 FPS processing, >95% color classification accuracy
  - Combined: <150ms end-to-end latency, 5-min continuous operation without crashes
**Constraints**:
  - Real-time operation in 32ms Webots timestep
  - CPU-only inference in final demo (GPU for training acceptable)
  - Model size <50MB each for practical loading
  - No GPS allowed in final demonstration (project rule)
**Scale/Scope**:
  - 2 neural networks (LIDAR processor + camera detector)
  - Training datasets: >1000 LIDAR scans, >500 camera images
  - 7 key entities/classes (LIDARProcessor, CubeDetector, PerceptionSystem, etc.)
  - Integration with existing Phase 1 controllers (base, arm, gripper)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Core Principles Compliance

**I. Fundamentação Científica (NON-NEGOTIABLE)**
- ✅ **PASS**: Spec references REFERENCIAS.md papers (Goodfellow 2016, Qi 2017, Redmon 2016)
- ✅ **PASS**: Architecture choices documented for DECISIONS.md (FR-021, FR-022)
- ⚠️ **ACTION REQUIRED**: Must document final architecture choice in DECISIONS.md BEFORE implementation

**II. Rastreabilidade Total**
- ✅ **PASS**: SpecKit workflow being followed (specify → plan → tasks → implement)
- ✅ **PASS**: Git branch 003-neural-networks created
- ⚠️ **ACTION REQUIRED**: Update TODO.md when phase complete, log metrics in logs/

**III. Desenvolvimento Incremental por Fases**
- ✅ **PASS**: Phase 1 (sensor exploration) marked complete in TODO.md
- ✅ **PASS**: Phase 2 builds on Phase 1 analysis (notebooks/sensor_analysis.ipynb)
- ✅ **PASS**: Clear deliverables defined (SC-001 to SC-011)

**IV. Qualidade Senior**
- ✅ **PASS**: Modular architecture planned (src/perception/)
- ✅ **PASS**: Testing strategy defined (unit, integration, accuracy validation)
- ⚠️ **ACTION REQUIRED**: Implement tests achieving >80% coverage target

**V. Restrições Disciplinares (NON-NEGOTIABLE)**
- ✅ **PASS**: RNA requirement satisfied (MLP/CNN for LIDAR + CNN for camera)
- ✅ **PASS**: No supervisor.py modification planned
- ✅ **PASS**: No GPS in final demo (training only, per assumptions)
- ✅ **PASS**: Scientific justification required for all choices (FR-021, FR-022)
- ⚠️ **FUTURE**: Lógica Fuzzy will be Phase 3 (not this phase)

**VI. Workflow SpecKit**
- ✅ **PASS**: spec.md created with 29 FRs, 11 SCs, 3 user stories
- ✅ **PASS**: No [NEEDS CLARIFICATION] markers in spec
- 🔄 **IN PROGRESS**: plan.md (this file) being generated
- ⏳ **NEXT**: research.md → data-model.md → tasks.md → implement

### Constitution Compliance Summary

**Status**: ✅ **APPROVED TO PROCEED**

All mandatory gates pass. Action items for implementation phase:
1. Document architecture decision in DECISIONS.md (MLP vs PointNet, YOLO vs SSD vs custom)
2. Update TODO.md Phase 2 checkboxes as tasks complete
3. Maintain >80% test coverage target
4. Log training metrics and model performance
5. Ensure Fuzzy Logic integration planned for Phase 3 handoff

## Project Structure

### Documentation (this feature)

```text
specs/003-neural-networks/
├── plan.md              # This file (/speckit.plan command output)
├── spec.md              # Feature specification (created by /speckit.specify)
├── research.md          # Phase 0 output (architecture & training research)
├── data-model.md        # Phase 1 output (entities and data structures)
├── quickstart.md        # Phase 1 output (training & inference guide)
├── contracts/           # Phase 1 output (API contracts for perception modules)
│   ├── lidar_processor.py     # LIDARProcessor interface
│   ├── cube_detector.py       # CubeDetector interface
│   └── perception_system.py   # PerceptionSystem integration API
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/perception/                   # NEW: Neural network perception modules
├── lidar_processor.py            # LIDAR neural network wrapper
├── cube_detector.py              # Camera CNN wrapper
├── perception_system.py          # Integration layer (LIDAR + camera fusion)
├── training/                     # Training infrastructure
│   ├── train_lidar.py           # LIDAR model training script
│   ├── train_camera.py          # Camera model training script
│   ├── data_loader.py           # Dataset loading utilities
│   └── augmentation.py          # Data augmentation functions
└── models/                       # Neural network architectures
    ├── lidar_net.py             # LIDAR network definition (MLP/PointNet/1D-CNN)
    └── camera_net.py            # Camera network definition (Custom/YOLO/SSD/ResNet)

data/                             # NEW: Training/validation datasets
├── lidar/                        # LIDAR training data
│   ├── scans/                   # Raw LIDAR scans (.npy files)
│   └── labels/                  # Obstacle annotations (.json)
└── camera/                       # Camera training data
    ├── images/                  # RGB images (.png files)
    └── labels/                  # Cube bounding boxes + colors (.json)

models/                           # NEW: Trained model weights
├── lidar_net.pth                # Trained LIDAR model
├── camera_net.pth               # Trained camera model
└── metadata.json                # Model hyperparameters and metrics

notebooks/                        # EXISTING: Analysis notebooks
├── sensor_analysis.ipynb        # Phase 1 sensor exploration (already exists)
├── lidar_training.ipynb         # NEW: LIDAR model training experiments
└── camera_training.ipynb        # NEW: Camera model training experiments

IA_20252/controllers/youbot/      # EXISTING: Webots controller (to be modified)
├── youbot.py                    # Main controller - add perception integration
├── base.py, arm.py, gripper.py  # Existing control modules (unchanged)
└── test_controller.py           # Phase 1 tests (unchanged)

tests/perception/                 # NEW: Perception module tests
├── test_lidar_processor.py      # LIDAR unit tests
├── test_cube_detector.py        # Camera unit tests
├── test_perception_system.py    # Integration tests
└── test_data_loading.py         # Dataset utilities tests

logs/                             # NEW: Training and inference logs
├── lidar_training.log           # LIDAR training metrics
├── camera_training.log          # Camera training metrics
└── inference_performance.log    # Real-time performance benchmarks
```

**Structure Decision**: Single project structure chosen (Option 1). This is a robotics perception system that extends the existing youbot controller from Phase 1. All perception code lives in `src/perception/` to maintain modularity and separation from control logic (which will be Phase 3 fuzzy controller). Training infrastructure is separate from inference code to keep controller lightweight. Datasets and trained models are stored at repo root for easy access across notebooks and controllers.

**Integration Point**: Phase 1 delivered working `youbot.py` controller with sensor access. Phase 2 adds perception layer between sensors and controller. Phase 3 will add fuzzy logic between perception outputs and actuator commands.

## Complexity Tracking

> **No violations requiring justification**

Constitution Check passed all gates. No complexity violations detected. This phase follows standard incremental development:
- Builds on Phase 1 sensor exploration
- Uses mandatory technologies (RNA per discipline requirements)
- Maintains modular architecture
- Follows SpecKit workflow

No simpler alternatives rejected because requirements are minimal and architecture is straightforward perception pipeline.

---

## Post-Design Constitution Re-Check

*Re-evaluation after Phase 1 design artifacts complete*

### ✅ Design Artifacts Generated

**Phase 0 (Research):**
- ✅ research.md: Architecture decisions resolved (hybrid MLP+1D-CNN for LIDAR, custom CNN for camera)
- ✅ Scientific justification: All choices backed by REFERENCIAS.md papers

**Phase 1 (Design):**
- ✅ data-model.md: 7 core entities + 3 training entities defined
- ✅ contracts/: API interfaces for LIDARProcessor, CubeDetector, PerceptionSystem
- ✅ quickstart.md: Complete training and deployment guide

**Phase 1 (Agent Context):**
- ✅ CLAUDE.md updated: PyTorch 2.0+, NumPy, SciPy, OpenCV, Matplotlib added to Active Technologies

### ✅ Constitution Compliance Re-Check

**I. Fundamentação Científica:**
- ✅ research.md documents all architecture decisions with scientific references
- ✅ Hybrid LIDAR: Goodfellow 2016 (Ch 12), Lenz 2015, LeCun 1998
- ✅ Custom CNN: LeCun 1998, Krizhevsky 2012, Goodfellow 2016 (Ch 9, 11)
- ✅ Training strategy: Qi 2017, Redmon 2016, Kingma & Ba 2014
- ⚠️ **ACTION REQUIRED**: Document final decisions in DECISIONS.md before implementation (DECISÃO 016-017)

**II. Rastreabilidade:**
- ✅ All design artifacts in specs/003-neural-networks/
- ✅ Agent context updated (CLAUDE.md)
- ⚠️ **ACTION REQUIRED**: Update TODO.md when implementation starts

**III. Desenvolvimento Incremental:**
- ✅ Phase 2 plan complete and detailed
- ✅ Builds on Phase 1 (sensor_analysis.ipynb, arena_map.md)
- ✅ Clear path to Phase 3 (fuzzy controller integration points documented)

**IV. Qualidade Senior:**
- ✅ Modular architecture: src/perception/, data/, models/, tests/
- ✅ API contracts define clean interfaces
- ✅ Testing strategy in quickstart.md (pytest, accuracy validation, performance benchmarks)

**V. Restrições Disciplinares:**
- ✅ RNA requirement satisfied (2 neural networks: LIDAR + camera)
- ✅ No GPS in final demo (only for training data collection)
- ✅ Scientific justification for all architecture choices
- ✅ No supervisor.py modifications planned

**VI. Workflow SpecKit:**
- ✅ spec.md → plan.md → research.md → data-model.md → quickstart.md → contracts/
- ⏳ **NEXT**: /speckit.tasks to generate implementation tasks
- ⏳ **THEN**: /speckit.implement to execute Phase 2

### Final Status: ✅ **APPROVED FOR TASKS GENERATION**

All design gates passed. Ready to proceed to `/speckit.tasks` for granular task breakdown.

**Action Items Before Implementation:**
1. Run `/speckit.tasks` to generate tasks.md
2. Document architecture decisions in DECISIONS.md (DECISÃO 016: LIDAR architecture, DECISÃO 017: Camera architecture)
3. Update TODO.md Phase 2 section when implementation begins
4. Follow quickstart.md training guide during implementation
