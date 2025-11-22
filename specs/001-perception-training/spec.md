# Feature Specification: Phase 2 Perception Model Training

**Feature Branch**: `001-perception-training`  
**Created**: 2025-11-21  
**Status**: Draft  
**Input**: User description: "Completar Fase 2 (Treinamento de RNAs)"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Sensor Dataset Acquisition & Labeling (Priority: P1)

As the perception engineer, I can capture and curate representative LIDAR scans and RGB frames (with ground-truth annotations) so that subsequent training runs use unbiased, high-quality data that reflects arena conditions.

**Why this priority**: Data integrity is the critical path for every other perception deliverable—without trustworthy datasets, training and downstream control stages are blocked.

**Independent Test**: Execute the data collection scripts, inspect generated datasets, and verify that labeling coverage meets the defined quotas (≥1,000 LIDAR scans, ≥500 RGB images, balanced per obstacle/cube class) without running any training.

**Acceptance Scenarios**:

1. **Given** Webots simulator sessions covering varied cube placements and robot poses, **When** the operator runs the collection scripts, **Then** at least 1,000 LIDAR scans and 500 RGB frames are persisted with metadata describing scenario, pose, and timestamp.
2. **Given** the raw datasets, **When** annotation/notebook workflows are executed, **Then** every sample has verified labels (sector occupancy for LIDAR, bounding boxes + colors for RGB) and passes automated schema validation.

---

### User Story 2 - LIDAR Model Training & Validation (Priority: P2)

As the LIDAR modeling owner, I can configure, train, and evaluate the hybrid MLP+1D-CNN so that it achieves the specified accuracy and latency targets, producing an exportable artifact for integration.

**Why this priority**: Accurate obstacle classification enables fuzzy control and navigation; once data quality is guaranteed, LIDAR inference is the next dependency to unblock state machine work.

**Independent Test**: Using the curated dataset, run the LIDAR training pipeline end-to-end and confirm the trained model meets accuracy (>90%), per-sector recall (>88%), and inference latency (<100 ms on target hardware) requirements without touching camera workflows.

**Acceptance Scenarios**:

1. **Given** the prepared LIDAR dataset split into train/val/test, **When** the training notebook executes with documented hyperparameters, **Then** validation accuracy exceeds 90% and the confusion matrix shows no sector below 85% recall.
2. **Given** the best-performing checkpoint, **When** converted to the controller’s runtime format and benchmarked on the hardware profile, **Then** median inference latency remains under 100 ms with batch size 1 and memory footprint fits within 50 MB.

---

### User Story 3 - Camera Model Training & Export (Priority: P3)

As the vision modeling owner, I can train, assess, and export the lightweight CNN (or fallback backbone) that identifies cube colors and approximate poses so that manipulation and state machine logic receive reliable detections.

**Why this priority**: Camera inference is required after obstacle detection, and its success hinges on finishing LIDAR work; completing it solidifies perception outputs for grasping and navigation.

**Independent Test**: Execute the camera training pipeline with the curated RGB dataset and verify the exported model achieves ≥95% per-color accuracy and delivers ≥10 FPS on the target hardware, independently of LIDAR results.

**Acceptance Scenarios**:

1. **Given** the labeled RGB dataset with color-balanced splits, **When** the training notebook runs with documented augmentations, **Then** the resulting model reports ≥95% precision and recall for each color class on the hold-out set.
2. **Given** the serialized model artifact, **When** deployed in a benchmark harness on the controller hardware, **Then** it sustains at least 10 FPS end-to-end while emitting bounding boxes and color labels within ±5° angular error.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- Simulator crashes or sensor noise spikes mid-collection, leading to incomplete sequences—data pipeline must detect and discard corrupted samples automatically.
- Class imbalance (e.g., too few blue cubes) skews metrics—training scripts must surface imbalance reports and enforce minimum-per-class thresholds before training.
- Training run fails to converge due to incorrect hyperparameters—workflow needs documented defaults and early-stopping safeguards to avoid wasting compute.
- Exported model fails compatibility checks (runtime-format conversion errors, missing metadata)—pipeline must validate artifacts and re-run conversion if needed.
- Hardware benchmark reveals latency regressions relative to spec—process must include optimization steps (quantization/pruning) or fallback architectures.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: Provide scripted workflows to capture ≥1,000 annotated LIDAR scans spanning all nine obstacle sectors, four robot headings, and mixed cube layouts.
- **FR-002**: Provide scripted workflows to capture ≥500 annotated RGB frames covering all cube colors, distances (0.3 m–3 m), and lighting variations used in simulation.
- **FR-003**: Validate collected datasets automatically (schema, label completeness, per-class counts) and block training if quality gates fail.
- **FR-004**: Train the hybrid LIDAR model using the documented architecture, logging hyperparameters, loss curves, and checkpoints for reproducibility.
- **FR-005**: Evaluate LIDAR models on a reserved test split and generate confusion matrices plus latency benchmarks that confirm compliance with >90% accuracy and <100 ms inference goals.
- **FR-006**: Provide a deployable LIDAR model artifact plus accompanying metadata describing version, metrics, preprocessing requirements, and checksum for controller integration.
- **FR-007**: Train the lightweight camera CNN (or approved fallback) with documented augmentations and class weighting, capturing training artifacts similar to the LIDAR pipeline.
- **FR-008**: Evaluate camera models on a reserved test split, producing per-class precision/recall, FPS benchmarks (≥10), and angular error statistics (≤5° median).
- **FR-009**: Provide a deployable camera model artifact plus metadata describing input resolution, normalization, calibration constants, and integrity checks.
- **FR-010**: Publish comprehensive experiment logs (loss curves, metrics tables, hardware specs, random seeds) and store them in `logs/` plus the training notebooks in `notebooks/`.
- **FR-011**: Document the complete data and training process in DECISIONS.md (new entries) and cite supporting references before marking Phase 2 complete.

### Key Entities *(include if feature involves data)*

- **LidarSample**: Represents a single 360° scan with attributes for raw ranges, derived sector occupancy labels, robot pose metadata, and scenario tags.
- **CameraSample**: Represents one RGB frame with cube bounding boxes, color labels, distance estimates, capture pose, and lighting tags.
- **TrainingRun**: Captures hyperparameters, dataset splits, metrics, hardware profile, checkpoints produced, and audit trail for reproducibility.
- **ModelArtifact**: Serialized inference asset plus metadata (version, checksum, metrics, preprocessing requirements) ready for deployment in controllers.

## Assumptions & Dependencies *(optional but recommended)*

- Webots simulator remains the authoritative source for data collection; no physical robot capture is required in this phase.
- Existing scripts (`scripts/collect_lidar_data.py`, `scripts/collect_camera_data.py`, annotation utilities, and training notebooks) are functional but need parameterization and execution.
- Target hardware profile for latency benchmarks is the same used in earlier phases (YouBot controller CPU/GPU configuration documented in DECISIONS.md).
- Storage capacity is sufficient to hold ≥20 GB of raw and processed datasets without cleanup mid-cycle.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: Dataset completeness—>=1,000 LIDAR scans and >=500 RGB frames with 0 invalid or unlabeled samples after validation.
- **SC-002**: LIDAR model performance—>=90% overall accuracy, >=88% recall per sector, and <=100 ms median inference latency on target hardware.
- **SC-003**: Camera model performance—>=95% precision and recall for each cube color, angular error ≤5° median, and >=10 FPS processing throughput.
- **SC-004**: Reproducibility—training notebooks + logs enable re-running both models end-to-end with <=5% variance in final metrics, and all artifacts (datasets, models, metadata) are versioned in the repository.
