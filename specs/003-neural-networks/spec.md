# Feature Specification: Neural Network Perception System

**Feature Branch**: `003-neural-networks`
**Created**: 2025-11-21
**Status**: Draft
**Input**: User description: "Fase 2: Percepção com Redes Neurais - Implementar RNA para processamento LIDAR (detecção de obstáculos) e CNN para detecção de cubos coloridos, baseado nas análises de sensores da Fase 1"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - LIDAR Obstacle Detection (Priority: P1)

The YouBot neural network system processes LIDAR point cloud data in real-time to detect obstacles and create a local occupancy map, enabling safe navigation in the arena without GPS.

**Why this priority**: Core safety requirement - robot cannot navigate without obstacle detection. Must work before any other perception module. Directly addresses project requirement: "RNA para detecção de obstáculos e mapeamento do ambiente" (Goodfellow et al., 2016; Qi et al., 2017).

**Independent Test**: Can be tested by placing robot in arena with known obstacles, measuring detection accuracy and false positive/negative rates. Delivers immediate value: safe navigation capability.

**Acceptance Scenarios**:

1. **Given** robot is 0.5m from wooden box obstacle, **When** LIDAR scan is processed by neural network, **Then** obstacle is detected with >90% confidence and correct distance ±5cm
2. **Given** robot is in open space with no obstacles within 2m, **When** LIDAR scan is processed, **Then** no false obstacle detections occur
3. **Given** robot is moving at 0.2 m/s, **When** new LIDAR data arrives (10 Hz), **Then** neural network processes scan in <100ms maintaining real-time performance
4. **Given** robot detects obstacle at 1.0m ahead, **When** robot approaches to 0.3m, **Then** system maintains accurate distance tracking throughout approach
5. **Given** robot is near arena wall, **When** LIDAR scans boundary, **Then** wall is correctly classified as permanent obstacle vs movable object

---

### User Story 2 - Camera-Based Cube Detection (Priority: P1)

The YouBot CNN vision system identifies cube colors (green, blue, red) from camera images and localizes them in the scene, enabling targeted collection of cubes.

**Why this priority**: Equal P1 with obstacle detection - both are fundamental perception capabilities. Without color detection, robot cannot complete primary task (colored cube sorting). Addresses project requirement: "CNN para detecção de cubos coloridos" (Redmon et al., 2016; Goodfellow et al., 2016).

**Independent Test**: Can be tested by placing colored cubes at known positions, measuring detection accuracy, color classification precision, and localization error. Delivers immediate value: cube identification capability.

**Acceptance Scenarios**:

1. **Given** green cube is visible 1m ahead in camera frame, **When** CNN processes image, **Then** cube is detected with >95% confidence and classified as "green"
2. **Given** multiple cubes (green, blue, red) are visible simultaneously, **When** CNN processes frame, **Then** all cubes are detected and correctly classified by color
3. **Given** cube is partially occluded by obstacle, **When** CNN processes image, **Then** cube is still detected if >50% visible
4. **Given** arena lighting conditions vary (shadows, bright spots), **When** CNN processes images, **Then** detection accuracy remains >90% across lighting variations
5. **Given** camera is running at 30 FPS, **When** CNN processes frames, **Then** inference time is <33ms per frame (>10 FPS minimum)

---

### User Story 3 - Integrated Perception System (Priority: P2)

The YouBot combines LIDAR obstacle detection and camera cube detection into unified perception system that provides actionable scene understanding for navigation and manipulation tasks.

**Why this priority**: Integration step depends on both P1 modules working independently first. Provides higher-level capabilities but not strictly necessary for MVP testing of individual components.

**Independent Test**: Can be tested by running full perception pipeline in arena with obstacles and cubes, measuring end-to-end latency and decision quality. Delivers value: complete scene understanding for autonomous operation.

**Acceptance Scenarios**:

1. **Given** robot observes scene with obstacles and colored cubes, **When** perception system processes data, **Then** unified world model is created showing both obstacles and cube positions
2. **Given** perception system detects cube behind obstacle, **When** navigation planner queries scene, **Then** system reports cube as "visible but obstructed" with path-planning implications
3. **Given** both sensors provide conflicting information (rare), **When** fusion algorithm runs, **Then** system resolves conflict using sensor confidence weights
4. **Given** perception system runs continuously, **When** monitored over 5-minute task, **Then** combined LIDAR+camera processing maintains <150ms end-to-end latency

---

### Edge Cases

- **Sensor failure**: What happens when LIDAR returns invalid range data (0.0 or NaN)?
- **Camera occlusion**: How does system handle when gripper arm blocks camera view?
- **Multiple cubes overlapping**: How does CNN distinguish between two cubes of same color in close proximity?
- **Extreme lighting**: What happens under very dim or very bright arena lighting?
- **Moving during capture**: How does system handle motion blur when robot moves during camera frame capture?
- **Arena boundary ambiguity**: How does LIDAR distinguish between arena walls (permanent) and wooden box obstacles (to avoid)?
- **Model inference failure**: What happens if neural network crashes or returns malformed output?
- **Resource exhaustion**: How does system behave if GPU/CPU is overloaded and cannot meet real-time requirements?

## Requirements *(mandatory)*

### Functional Requirements

#### LIDAR Neural Network (FR-001 to FR-010)

- **FR-001**: System MUST implement neural network to process LIDAR point cloud data (667 points at 270° FOV)
- **FR-002**: Neural network MUST output obstacle detection with confidence scores (0.0 to 1.0 range)
- **FR-003**: System MUST achieve >90% obstacle detection accuracy on validation dataset
- **FR-004**: System MUST process LIDAR scans in <100ms on target hardware (real-time requirement)
- **FR-005**: System MUST generate local occupancy grid map (0.1m resolution) from LIDAR detections
- **FR-006**: System MUST distinguish between arena walls and movable obstacles
- **FR-007**: System MUST handle LIDAR range limits (0.01m to 3.5m) and invalid readings
- **FR-008**: System MUST maintain detection accuracy across different obstacle materials (wood, plastic)
- **FR-009**: System MUST provide distance and bearing to detected obstacles (polar coordinates)
- **FR-010**: System MUST collect and label LIDAR training dataset with >1000 scans from diverse arena positions

#### Camera CNN (FR-011 to FR-020)

- **FR-011**: System MUST implement CNN to detect and classify colored cubes in camera images (512×512 RGB)
- **FR-012**: CNN MUST classify cubes into exactly 3 classes: green, blue, red
- **FR-013**: System MUST achieve >95% color classification accuracy on validation dataset
- **FR-014**: System MUST provide bounding box coordinates for each detected cube (pixel coordinates)
- **FR-015**: System MUST process camera frames at minimum 10 FPS (inference time <100ms)
- **FR-016**: System MUST handle partial occlusion (detect cubes with >50% visibility)
- **FR-017**: System MUST be robust to lighting variations (shadows, reflections, brightness changes)
- **FR-018**: System MUST estimate cube distance using camera geometry and known cube size (0.05m)
- **FR-019**: System MUST reject false positives (non-cube objects) with <5% false positive rate
- **FR-020**: System MUST collect and label camera training dataset with >500 images across all cube colors

#### Model Architecture & Training (FR-021 to FR-024)

- **FR-021**: LIDAR network architecture MUST be chosen from: MLP (baseline), PointNet (advanced), or 1D CNN
- **FR-022**: Camera network architecture MUST be chosen from: Custom CNN, YOLOv5, SSD, or pre-trained ResNet backbone
- **FR-023**: All models MUST be trained with proper train/validation/test split (70/15/15 minimum)
- **FR-024**: Training MUST use data augmentation to improve robustness (rotation, noise, brightness for camera; noise, dropout for LIDAR)

#### Integration & Performance (FR-025 to FR-029)

- **FR-025**: System MUST fuse LIDAR and camera data into unified perception output
- **FR-026**: Perception system MUST run continuously in Webots simulation loop (32ms time step)
- **FR-027**: System MUST provide perception API for downstream controllers (fuzzy logic in Phase 3)
- **FR-028**: All neural network models MUST be serialized and loadable in Webots Python controller
- **FR-029**: System MUST log perception outputs for debugging (detections, confidence, timing)

### Key Entities

- **LIDARProcessor**: Neural network module that processes raw LIDAR scans and outputs obstacle detections with confidence scores
- **CubeDetector**: CNN module that processes camera images and outputs cube bounding boxes, colors, and confidence scores
- **PerceptionSystem**: Integration layer that combines LIDAR and camera outputs into unified world representation
- **TrainingDataset**: Labeled data collection for both LIDAR (obstacle annotations) and camera (cube bounding boxes + colors)
- **OccupancyGrid**: Local map representation (2D grid) showing free space, obstacles, and unknown areas derived from LIDAR
- **CubeObservation**: Data structure containing cube color, 2D position estimate, confidence, and timestamp
- **ObstacleObservation**: Data structure containing obstacle distance, bearing, type (wall/box), and confidence

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: LIDAR neural network achieves >90% obstacle detection accuracy on held-out test set (minimum 100 test scans)
- **SC-002**: Camera CNN achieves >95% cube color classification accuracy on held-out test set (minimum 50 test images per color)
- **SC-003**: LIDAR processing maintains <100ms inference time on target hardware (measured over 100 consecutive scans)
- **SC-004**: Camera processing maintains >10 FPS throughput (measured over 300 consecutive frames)
- **SC-005**: False positive rate for cube detection is <5% (measured in arena without cubes)
- **SC-006**: System detects all obstacles within 2m radius during 360° robot rotation (zero missed obstacles)
- **SC-007**: System correctly identifies all 3 cube colors when presented individually at 1m distance
- **SC-008**: Integrated perception system runs continuously for 5-minute simulation without crashes or memory leaks
- **SC-009**: Training datasets are collected with diversity: LIDAR from >20 arena positions, camera with >10 lighting conditions
- **SC-010**: Models are serialized to files (<50MB each) and successfully load in Webots controller
- **SC-011**: Perception outputs are logged with timestamps and can be replayed for analysis

## Assumptions

- Phase 1 sensor analysis (notebooks/sensor_analysis.ipynb) provides baseline understanding of LIDAR and camera data characteristics
- Webots R2023b Python controller supports PyTorch model inference (torch.jit or ONNX)
- Training will use GPU acceleration (CUDA) for efficiency, but inference must work on CPU for portability
- Arena map (docs/arena_map.md) provides known obstacle positions for training data labeling
- Supervisor spawns cubes randomly, so camera training data will be collected across multiple simulation runs
- GPS can be used during training data collection phase (not in final demonstration per project rules)

## Dependencies

- **Phase 1 Complete**: Sensor analysis notebooks and arena mapping must be finished (✅ COMPLETE)
- **PyTorch**: Deep learning framework for model development and inference (version 2.0+)
- **NumPy/SciPy**: Numerical processing for LIDAR point clouds and camera image preprocessing
- **OpenCV**: Camera image manipulation and augmentation
- **Matplotlib**: Visualization of training metrics and detection outputs
- **Webots API**: Camera.getImage(), Lidar.getPointCloud(), Lidar.getRangeImage() for sensor access
- **Hardware**: GPU for training (optional but recommended), CPU sufficient for inference

## Scope

### In Scope
- Neural network architecture selection and justification (DECISIONS.md)
- Training data collection using Webots simulation
- Model training, validation, and hyperparameter tuning
- Real-time inference integration in Webots controller
- Performance benchmarking and accuracy measurement
- Model serialization and loading
- Basic perception API for downstream modules

### Out of Scope
- Path planning algorithms (Phase 4)
- Fuzzy logic control system (Phase 3)
- Manipulation strategies for grasping (Phase 5)
- SLAM or global mapping (local perception only)
- Multi-robot coordination
- Sim-to-real transfer (Webots only)
- Advanced sensor fusion (Kalman filtering) - simple fusion sufficient for P2

## Notes

**Scientific Foundation**: All architecture choices must be justified with references from REFERENCIAS.md:
- **LIDAR**: Qi et al. (2017) PointNet for 3D point cloud processing, or simpler MLP baseline
- **Camera**: Redmon et al. (2016) YOLO, Liu et al. (2016) SSD, or transfer learning with ResNet (He et al., 2016)
- **Deep Learning Theory**: Goodfellow et al. (2016) Deep Learning textbook chapters on CNNs and training

**Decision Documentation**: All major architecture decisions (model choice, training strategy, hyperparameters) must be recorded in DECISIONS.md BEFORE implementation per project methodology.

**Performance Priority**: Real-time performance is critical - models must meet inference time requirements (<100ms) even if accuracy is slightly compromised. Better to have 88% accurate fast model than 92% accurate slow model.

**Testing Strategy**: Unit tests for data loading/preprocessing, integration tests for model inference in Webots loop, accuracy tests on validation sets.

**Phase 3 Interface**: Perception outputs will feed into fuzzy logic controller (Phase 3), so API design should consider fuzzy input requirements (distances, confidence scores, cube positions).
