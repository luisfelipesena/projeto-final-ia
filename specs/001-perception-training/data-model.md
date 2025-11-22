# Data Model — Phase 2 Perception Model Training

This document captures the logical entities and relationships required to execute the perception data and training workflows defined in the spec. It is implementation-agnostic and focuses on validation and traceability requirements.

## Entities

### LidarSample
| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `sample_id` | UUID | Unique identifier per scan | Required |
| `timestamp` | ISO8601 string | Capture time in simulator | Required |
| `robot_pose` | struct `{x, y, theta}` | Pose of robot when scan captured | Required, floats within arena bounds |
| `ranges` | float[360] | Raw LIDAR distances (meters) | Required, 0.05 m ≤ value ≤ 5.0 m |
| `sector_labels` | bool[9] | Occupancy flag per 40° sector | Required; at least one True if obstacle present |
| `scenario_tag` | enum | e.g., `clear`, `obstacle_front`, `corridor_left` | Must match predefined taxonomy |
| `split` | enum (`train`,`val`,`test`) | Dataset split assignment | Balanced per spec quotas |

### CameraSample
| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `sample_id` | UUID | Unique ID per frame | Required |
| `timestamp` | ISO8601 string | Capture time | Required |
| `robot_pose` | struct `{x, y, theta}` | Pose for distance estimation | Required |
| `image_path` | string | Relative path to frame file | Must exist |
| `bounding_boxes` | array of {`id`, `x`, `y`, `w`, `h`} | Pixel-space boxes | Width/height > 0; clamp within resolution |
| `colors` | array enum (`red`,`green`,`blue`) | Color per bbox ID | Alignment with bounding_boxes |
| `distance_estimates` | array float | Estimated distance per bbox | 0.2 m ≤ value ≤ 3.0 m |
| `lighting_tag` | enum | `default`, `bright`, `dim` | Required |
| `split` | enum (`train`,`val`,`test`) | Dataset split assignment | Balanced per color |

### TrainingRun
| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `run_id` | UUID | Unique experiment ID | Required |
| `model_type` | enum (`lidar`, `camera`) | Which pipeline executed | Required |
| `dataset_hash` | string | Hash of serialized dataset manifest | Required |
| `hyperparameters` | JSON object | Optimizer, LR, batch size, epochs, augmentations | No missing keys for defaults |
| `metrics` | JSON object | Accuracy, recall per class, loss curves, latency | Must include thresholds defined in spec |
| `hardware_profile` | JSON object | CPU/GPU model, RAM, OS | Required |
| `artifacts` | array of ModelArtifact references | IDs of produced checkpoints/exports | At least 1 (best checkpoint) |
| `notes` | markdown string | Citations, observations | Optional |

### ModelArtifact
| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `artifact_id` | UUID | Unique ID per export | Required |
| `model_type` | enum (`lidar`, `camera`) | Matches training run | Required |
| `format` | enum (`torchscript`, `onnx`, `metrics-json`) | Export format | TorchScript required for deployment |
| `file_path` | string | Relative path under `models/` | Must exist & checksum recorded |
| `checksum` | string | SHA256 of file | Required |
| `metrics_snapshot` | JSON object | Key performance metrics at export time | Must match TrainingRun metrics |
| `preprocessing` | JSON object | Normalization params, input resolution | Required |
| `calibration` | JSON object | For camera: focal length, principal point; for LIDAR: sector bounds | Required |
| `created_at` | ISO8601 string | Export timestamp | Required |

## Relationships
- `TrainingRun (1) → (n) ModelArtifact`: Each run may produce multiple artifacts (best checkpoint, quantized variant, metadata). Every artifact references its originating run.
- `LidarSample` and `CameraSample` link to `TrainingRun` via `dataset_hash`—training run must reference the manifest that lists included sample IDs for traceability.
- `ModelArtifact` metadata must embed references back to the spec version and DECISIONS.md entry ID that justified the chosen architecture/hyperparameters.

## Validation Rules
1. **Dataset Balance**: Automated validator ensures:
   - `sector_labels` distribution per sector deviates ≤10% from uniform.
   - `colors` distribution per cube color deviates ≤5% from uniform.
2. **Split Integrity**: No sample ID appears in more than one split.
3. **Metric Thresholds**:
   - LIDAR: `metrics.accuracy >= 0.90` and `metrics.latency_ms <= 100`.
   - Camera: `metrics.precision[color] >= 0.95`, `metrics.recall[color] >= 0.95`, `metrics.fps >= 10`.
4. **Reproducibility**: `TrainingRun` must record `random_seed` and `git_commit` inside `hyperparameters`.
5. **Metadata Completeness**: `ModelArtifact.preprocessing` must include `input_channels`, `resolution`, `mean`, `std`, and `normalization_space`.

