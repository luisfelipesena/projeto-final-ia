# Research Summary — Phase 2 Perception Model Training

> All findings cite core references from REFERENCIAS.md (Goodfellow16, Qi17, Redmon16, Liu16, Thrun05, etc.) to satisfy constitution Principle I.

## Decision 1: Dataset Coverage & Augmentation Strategy
- **Decision**: Capture ≥1,000 LIDAR scans and ≥500 RGB frames covering all obstacle sectors, cube colors, robot headings, and lighting variations. Apply rotation/noise/dropout augmentations for LIDAR and photometric/geometric augmentations for RGB during training.
- **Rationale**: Qi17 (PointNet) and Thrun05 emphasize viewpoint diversity for generalizable obstacle perception. Small robotics datasets benefit from augmentation to avoid overfitting (Goodfellow16 ch.7). Balancing per-class counts prevents bias when fuzzy controller consumes outputs.
- **Alternatives Considered**:
  1. **Smaller dataset (≤500 scans / ≤250 images)** — rejected: insufficient sector/color coverage, high variance metrics.
  2. **Synthetic data only** — rejected: Webots already simulates environment; further synthetic generation would not add diversity beyond what targeted capture sessions provide.

## Decision 2: LIDAR Model Architecture & Export
- **Decision**: Use existing hybrid MLP + 1D-CNN architecture (scripts + `src/perception/models/lidar_net.py`) with sector-wise sigmoid outputs; training via BCE loss, Adam optimizer, early stopping, and TorchScript export for deployment.
- **Rationale**: Hybrid model already engineered in Phase 2 infrastructure and aligns with Qi17’s feature fusion guidance. Sigmoid sectors allow simultaneous multi-sector occupancy. TorchScript provides deterministic runtime with low overhead on controller CPU.
- **Alternatives Considered**:
  1. **Pure PointNet** — rejected for now: heavier compute, needs GPU for inference, not guaranteed <100 ms on controller.
  2. **Classical rule-based sector mapping** — rejected: lacks learning capability, fails requirement for RNA-based perception.

## Decision 3: Camera Model & Performance Targets
- **Decision**: Default to lightweight CNN (Conv→BN→ReLU stacks + GAP + FC) with HSV-guided augmentations; fall back to ResNet18 (transfer learning) only if accuracy <93%. Export to TorchScript with metadata about normalization and calibration constants.
- **Rationale**: Redmon16 and Liu16 show that modest CNNs can achieve high accuracy for constrained classes if trained with augmentation. Lightweight model ensures ≥10 FPS on CPU-only controller; fallback ensures reserve capacity if base model underperforms.
- **Alternatives Considered**:
  1. **YOLOv5 nano** — rejected: heavier dependency chain, GPU-centric, overshoot scope.
  2. **Classical color-thresholding only** — rejected: fails to meet >95% accuracy and distance/angular estimation requirements.

## Decision 4: Experiment Tracking & Reproducibility
- **Decision**: Standardize notebook template + CLI wrappers to log hyperparameters, random seeds, dataset hashes, hardware profile, and metrics into `logs/perception/*.json`. Store best checkpoints + metadata JSON per model.
- **Rationale**: Goodfellow16 (ch.11) and industry best practices highlight reproducibility for ML experiments. Constitution requires traceability. Logging ensures ≤5% variance requirement is verifiable.
- **Alternatives Considered**:
  1. **Ad-hoc notebook notes only** — rejected: no automated verification, violates traceability principle.
  2. **Full-blown MLflow** — rejected for this phase to keep scope manageable; local JSON logs sufficient.

