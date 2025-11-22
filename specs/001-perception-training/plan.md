# Implementation Plan: Phase 2 Perception Model Training

**Branch**: `001-perception-training` | **Date**: 2025-11-21 | **Spec**: [`specs/001-perception-training/spec.md`](./spec.md)  
**Input**: Feature specification from `/specs/001-perception-training/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Complete Phase 2 by executing the data-collection and neural-network training workflows for the perception stack. Deliverables include curated LIDAR (≥1,000 scans) and camera (≥500 frames) datasets, trained hybrid MLP+1D-CNN for obstacle sectors (>90% accuracy, <100 ms latency), and lightweight CNN for cube detection (>95% per-color accuracy, ≥10 FPS). Each model must be exported with metadata for controller integration, backed by reproducible notebooks, logs, and documented scientific references per constitution.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.14 (Webots-compatible controller runtime)  
**Primary Dependencies**: PyTorch 2.x, Torchvision, NumPy, OpenCV, Matplotlib, scikit-learn, Pandas, custom scripts in `scripts/`  
**Storage**: Local filesystem datasets (`data/lidar/`, `data/camera/`) with metadata JSON/CSV; model artifacts in `models/`  
**Testing**: Pytest suites under `tests/perception/` plus notebook-driven validation cells (converted to automated scripts)  
**Target Platform**: macOS/Ubuntu development host for training; Webots controller hardware profile for latency benchmarks  
**Project Type**: Monorepo (simulation + perception + control)  
**Performance Goals**: LIDAR model ≥90% accuracy / ≥88% per-sector recall / ≤100 ms inference; Camera model ≥95% per-color precision/recall / ≥10 FPS / ≤5° angular error  
**Constraints**: Must document scientific references before implementation, obey dataset quotas, preserve storage under 20 GB, no GPS data usage, reproducible notebooks with ≤5% metric variance  
**Scale/Scope**: Phase covers 1k+ LIDAR samples, 500+ RGB images, two train/eval/export pipelines, four artifacts (two models + metadata)

## Constitution Check

*Gate must pass before Phase 0 research; re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Fundamentação científica required for model/training choices | ✅ | Research phase will cite REFERENCIAS.md entries (Goodfellow16, Qi17, Redmon16, etc.) before implementation. |
| Rastreabilidade (DECISIONS.md before coding) | ✅ | Plan mandates decision logging prior to executing scripts. |
| Sequential phases (Phase 2 only after Phase 1 complete) | ✅ | Phase 1 (environment/exploration) already closed per TODO.md; proceeding within Phase 2 scope. |
| Prohibited items (no GPS, no supervisor edits, no code in presentation) | ✅ | This phase only touches perception scripts/datasets; compliance tracked in quickstart + DECISIONS. |
| SpecKit workflow adherence | ✅ | Running specify → plan; clarify unneeded; will continue with tasks/implement after plan. |

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
├── perception/
│   ├── models/              # Neural architectures (lidar_net.py, camera_net.py)
│   ├── datasets/            # PyTorch dataset/dataloader helpers
│   ├── augmentation/        # Data augmentation utilities
│   └── perception_system.py # Integration entry point
├── control/
│   ├── fuzzy_controller.py
│   ├── fuzzy_rules.py
│   └── state_machine.py
├── navigation/
│   ├── local_map.py
│   └── odometry.py
├── manipulation/
│   ├── grasping.py
│   └── depositing.py
└── main_controller.py

scripts/
├── collect_lidar_data.py
├── collect_camera_data.py
├── annotate_lidar.py
├── annotate_camera.py
├── train_lidar_model.py
└── train_camera_model.py

data/
├── lidar/
└── camera/

models/
├── lidar_net.pt
├── lidar_net_metadata.json
├── camera_net.pt
└── camera_net_metadata.json

tests/
├── perception/
│   ├── test_lidar_dataset.py
│   ├── test_camera_dataset.py
│   └── test_perception_integration.py
└── control/
    └── ...
```

**Structure Decision**: Single monorepo layout focused on `src/perception`, `scripts/`, `data/`, `models/`, and `tests/perception`. This plan only adds documentation/contracts under `specs/001-perception-training/` and reuses existing source directories for implementation later.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
