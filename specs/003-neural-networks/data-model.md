# Data Model: Neural Network Perception System

**Phase 1 - Design**
**Date**: 2025-11-21
**Purpose**: Define entities, data structures, and relationships for Phase 2 perception system

---

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PerceptionSystem                              │
│  (Unified interface for obstacle + cube detection)               │
└────────────┬──────────────────────────────────┬──────────────────┘
             │                                   │
             ▼                                   ▼
    ┌────────────────┐                 ┌─────────────────┐
    │ LIDARProcessor │                 │  CubeDetector   │
    │  (Obstacles)   │                 │  (Color + BBox) │
    └────────┬───────┘                 └────────┬────────┘
             │                                   │
             │ processes                         │ processes
             ▼                                   ▼
    ┌────────────────┐                 ┌─────────────────┐
    │ LIDAR Scan     │                 │  Camera Image   │
    │ [667 ranges]   │                 │  512×512 RGB    │
    └────────────────┘                 └─────────────────┘
             │                                   │
             │ produces                          │ produces
             ▼                                   ▼
    ┌────────────────┐                 ┌─────────────────┐
    │ ObstacleMap    │                 │ CubeObservation │
    │ [9 sectors]    │                 │ (color + bbox)  │
    └────────────────┘                 └─────────────────┘
```

---

## Core Entities

### 1. LIDARProcessor

**Purpose:** Neural network wrapper for LIDAR obstacle detection

**Attributes:**
- `model`: torch.jit.ScriptModule - Trained hybrid MLP+CNN model
- `model_path`: str - Path to serialized model file
- `device`: str - "cpu" (inference device)
- `input_size`: int = 667 - Number of LIDAR points
- `output_size`: int = 9 - Number of sectors (30° each)
- `sector_angle`: float = 30.0 - Degrees per sector

**Methods:**
```python
def __init__(model_path: str)
    """Load trained model from disk"""

def process(ranges: np.ndarray) -> ObstacleMap
    """
    Process LIDAR scan and return obstacle map

    Args:
        ranges: [667] array of distances (meters)

    Returns:
        ObstacleMap with 9 sector probabilities
    """

def preprocess(ranges: np.ndarray) -> torch.Tensor
    """
    Preprocess raw LIDAR data
    - Handle invalid readings (0.0, NaN, >3.5m)
    - Normalize to [0,1]
    - Extract hand-crafted features

    Returns: [70] feature vector (64 CNN + 6 hand-crafted)
    """

def extract_features(ranges: np.ndarray) -> np.ndarray
    """
    Extract 6 hand-crafted features:
    - min_distance
    - mean_distance
    - std_distance
    - occupancy_ratio (<0.5m)
    - left_right_symmetry
    - range_variance
    """

def postprocess(logits: torch.Tensor) -> ObstacleMap
    """Convert [9] network logits to ObstacleMap"""
```

**Validation Rules:**
- Input ranges must be [667] shape
- Invalid readings (<0.01m, >3.5m) replaced with max range
- Output probabilities ∈ [0,1]

**State Transitions:** Stateless (pure inference)

---

### 2. CubeDetector

**Purpose:** CNN wrapper for cube color detection

**Attributes:**
- `model`: torch.jit.ScriptModule - Trained custom CNN
- `model_path`: str - Path to serialized model file
- `device`: str - "cpu"
- `input_size`: tuple = (512, 512) - Image resolution
- `num_classes`: int = 3 - [green, blue, red]
- `class_names`: list = ["green", "blue", "red"]

**Methods:**
```python
def __init__(model_path: str)
    """Load trained CNN model"""

def detect(image: np.ndarray) -> List[CubeObservation]
    """
    Detect cubes in camera image

    Args:
        image: [512, 512, 3] RGB array (uint8)

    Returns:
        List of CubeObservation (one per detected cube)
    """

def preprocess(image: np.ndarray) -> torch.Tensor
    """
    Preprocess image:
    - Normalize to [0,1]
    - Apply transforms (resize, normalize)

    Returns: [1, 3, 512, 512] tensor
    """

def segment_candidates(image: np.ndarray) -> List[BoundingBox]
    """
    Use color-based segmentation (HSV) to find candidate regions

    Returns: List of bounding boxes for candidate cubes
    """

def classify_region(image: np.ndarray, bbox: BoundingBox) -> CubeObservation
    """
    Classify cube color in bounding box region

    Returns: CubeObservation with color + confidence
    """

def estimate_distance(bbox: BoundingBox, cube_size: float = 0.05) -> float
    """
    Estimate cube distance from camera using known size (5cm)

    Returns: Distance in meters
    """
```

**Validation Rules:**
- Input image must be [512, 512, 3] shape
- Bounding boxes must be within image bounds
- Confidence threshold >0.5 for detection
- Max 15 detections per frame (cube count limit)

**State Transitions:** Stateless (pure inference)

---

### 3. PerceptionSystem

**Purpose:** Integration layer combining LIDAR + camera

**Attributes:**
- `lidar_processor`: LIDARProcessor
- `cube_detector`: CubeDetector
- `last_obstacle_map`: ObstacleMap - Cached LIDAR result
- `last_cube_observations`: List[CubeObservation] - Cached detections
- `update_rate_hz`: float = 10.0 - Perception loop frequency

**Methods:**
```python
def __init__(lidar_model_path: str, camera_model_path: str)
    """Initialize both processors"""

def update(lidar_ranges: np.ndarray, camera_image: np.ndarray) -> WorldState
    """
    Process both sensors and fuse results

    Args:
        lidar_ranges: [667] LIDAR distances
        camera_image: [512, 512, 3] RGB image

    Returns:
        WorldState with obstacles + cubes
    """

def get_obstacle_map() -> ObstacleMap
    """Return latest obstacle map"""

def get_cube_observations() -> List[CubeObservation]
    """Return latest cube detections"""

def get_nearest_obstacle() -> Tuple[int, float]
    """
    Returns: (sector_id, distance) for closest obstacle
    """

def get_nearest_cube(color: str) -> Optional[CubeObservation]
    """
    Find nearest cube of specified color

    Returns: CubeObservation or None if not detected
    """
```

**Validation Rules:**
- Update frequency ≤100 Hz (limit from 32ms timestep)
- Timestamps synchronized between sensors
- Thread-safe for concurrent access

**State Transitions:**
```
INITIALIZING → READY → RUNNING
                 ↓
            ERROR (model load fail)
```

---

## Data Structures

### ObstacleMap

**Purpose:** Representation of LIDAR obstacle detection results

```python
@dataclass
class ObstacleMap:
    """9-sector obstacle occupancy map"""

    probabilities: np.ndarray  # [9] P(obstacle) per sector
    timestamp: float           # Unix timestamp (seconds)
    min_distance: float        # Closest obstacle (meters)
    min_sector: int            # Sector with min distance [0-8]

    # Sector layout (30° each, 0° = forward):
    # [0]=[-45°, -15°], [1]=[-15°, 15°], [2]=[15°, 45°], ...
    # [4]=center-left, [5]=left, [6]=rear-left, [7]=rear, [8]=rear-right

    def is_obstacle(self, sector: int, threshold: float = 0.5) -> bool:
        """Check if sector has obstacle above threshold"""

    def get_free_sectors(self, threshold: float = 0.5) -> List[int]:
        """Return list of free (navigable) sectors"""

    def to_dict(self) -> dict:
        """Serialize for logging"""
```

**Validation:**
- probabilities.shape == (9,)
- probabilities ∈ [0,1]
- min_distance ∈ [0.01, 3.5]
- min_sector ∈ [0,8]

---

### CubeObservation

**Purpose:** Single cube detection result

```python
@dataclass
class CubeObservation:
    """Detected cube with color and location"""

    color: str                # "green" | "blue" | "red"
    bbox: BoundingBox         # Pixel coordinates in image
    confidence: float         # Classification confidence [0,1]
    distance: float           # Estimated distance (meters)
    angle: float              # Bearing from robot center (degrees)
    timestamp: float          # Unix timestamp

    def is_valid(self, min_confidence: float = 0.5) -> bool:
        """Check if detection confidence sufficient"""

    def to_dict(self) -> dict:
        """Serialize for logging"""
```

**Validation:**
- color ∈ ["green", "blue", "red"]
- confidence ∈ [0,1]
- distance ∈ [0.1, 3.0] (reasonable range)
- angle ∈ [-90, 90] (camera FOV)

---

### BoundingBox

**Purpose:** Image region coordinates

```python
@dataclass
class BoundingBox:
    """Rectangular region in image"""

    x_min: int     # Left edge (pixels)
    y_min: int     # Top edge
    x_max: int     # Right edge
    y_max: int     # Bottom edge

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x_min + self.x_max) // 2,
                (self.y_min + self.y_max) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """Intersection over Union with another box"""
```

**Validation:**
- 0 ≤ x_min < x_max ≤ 512
- 0 ≤ y_min < y_max ≤ 512
- area > 100 (minimum cube size, ~10×10 pixels)

---

### WorldState

**Purpose:** Unified perception output (LIDAR + camera)

```python
@dataclass
class WorldState:
    """Combined sensor perception"""

    obstacle_map: ObstacleMap
    cube_observations: List[CubeObservation]
    timestamp: float

    def get_navigation_command(self) -> Tuple[float, float, float]:
        """
        Placeholder for Phase 3 fuzzy controller interface

        Returns: (vx, vy, omega) velocity commands
        """

    def to_dict(self) -> dict:
        """Serialize full state for logging"""
```

---

## Training Data Entities

### LIDARDataset

**Purpose:** Training dataset for LIDAR neural network

```python
class LIDARDataset(torch.utils.data.Dataset):
    """LIDAR scans with obstacle labels"""

    def __init__(self, data_dir: str, split: str = "train",
                 augment: bool = True):
        """
        Load LIDAR dataset

        Args:
            data_dir: data/lidar/
            split: "train" | "val" | "test"
            augment: Apply data augmentation
        """
        self.scans = []      # List of [667] arrays
        self.labels = []     # List of [9] binary labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            scan: [667] ranges (augmented if training)
            label: [9] binary obstacle flags per sector
        """
```

**File Structure:**
```
data/lidar/
├── scans/
│   ├── scan_0000.npy  # [667] float32 array
│   ├── scan_0001.npy
│   └── ...
└── labels/
    ├── scan_0000.json  # {"sectors": [0,0,1,0,0,0,0,1,0], "timestamp": ...}
    └── ...
```

---

### CameraDataset

**Purpose:** Training dataset for camera CNN

```python
class CameraDataset(torch.utils.data.Dataset):
    """Camera images with cube color labels"""

    def __init__(self, data_dir: str, split: str = "train",
                 augment: bool = True):
        """
        Load camera dataset

        Args:
            data_dir: data/camera/
            split: "train" | "val" | "test"
            augment: Apply augmentation (brightness, hue, etc.)
        """
        self.images = []     # List of image paths
        self.labels = []     # List of class indices [0,1,2]
        self.augment = augment
        self.class_to_idx = {"green": 0, "blue": 1, "red": 2}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: [3, 512, 512] tensor (augmented if training)
            label: Class index [0,1,2]
        """
```

**File Structure:**
```
data/camera/
├── images/
│   ├── green_0000.png  # 512×512 RGB
│   ├── blue_0000.png
│   ├── red_0000.png
│   └── ...
└── labels/
    ├── green_0000.json  # {"color": "green", "bbox": [x1,y1,x2,y2]}
    └── ...
```

---

## Model Metadata

### ModelMetadata

**Purpose:** Store training hyperparameters and metrics

```python
@dataclass
class ModelMetadata:
    """Metadata for trained neural network"""

    model_name: str                # "lidar_hybrid_v1"
    architecture: str              # "MLP+1D-CNN" | "Custom CNN"
    parameters: int                # Total parameter count
    training_date: str             # ISO 8601 timestamp

    # Hyperparameters
    batch_size: int
    learning_rate: float
    optimizer: str
    epochs_trained: int

    # Performance metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    inference_time_ms: float       # Average on CPU

    # Dataset info
    train_size: int
    val_size: int
    test_size: int

    def save(self, path: str):
        """Save to JSON"""

    @classmethod
    def load(cls, path: str):
        """Load from JSON"""
```

**File:** `models/lidar_net_metadata.json`, `models/camera_net_metadata.json`

---

## Relationships

### LIDARProcessor → ObstacleMap
- **Cardinality:** 1 → 1 (each inference produces one map)
- **Timing:** Real-time (every Webots timestep, 32ms)

### CubeDetector → CubeObservation
- **Cardinality:** 1 → 0..15 (zero to 15 cubes per frame)
- **Timing:** Real-time (>10 FPS)

### PerceptionSystem → (LIDARProcessor, CubeDetector)
- **Composition:** PerceptionSystem owns both processors
- **Lifecycle:** Processors initialized once, reused

### WorldState → (ObstacleMap, List[CubeObservation])
- **Aggregation:** WorldState contains results from both sensors
- **Synchronization:** Timestamped for alignment

---

## State Machine

### PerceptionSystem Lifecycle

```
[INITIALIZING]
    ↓ load models
[READY]
    ↓ start simulation
[RUNNING] ←──┐
    ↓ update() │
[PROCESSING]──┘
    ↓ error or shutdown
[ERROR] or [STOPPED]
```

**States:**
- **INITIALIZING**: Loading models from disk
- **READY**: Models loaded, waiting for sensor data
- **RUNNING**: Actively processing sensor streams
- **PROCESSING**: Inference in progress (< 150ms)
- **ERROR**: Model load failure or inference crash
- **STOPPED**: Graceful shutdown

**Transitions:**
- INITIALIZING → READY: `torch.jit.load()` succeeds
- INITIALIZING → ERROR: Model file not found or corrupted
- READY → RUNNING: First sensor data received
- RUNNING → PROCESSING → RUNNING: Each `update()` call
- * → STOPPED: User termination

---

## Summary

**7 Core Entities:**
1. LIDARProcessor (LIDAR inference)
2. CubeDetector (camera inference)
3. PerceptionSystem (integration)
4. ObstacleMap (LIDAR output)
5. CubeObservation (camera output)
6. WorldState (unified output)
7. ModelMetadata (training tracking)

**3 Training Entities:**
1. LIDARDataset
2. CameraDataset
3. Data augmentation pipelines

**Key Relationships:**
- PerceptionSystem composes LIDARProcessor + CubeDetector
- WorldState aggregates ObstacleMap + List[CubeObservation]
- Datasets load from `data/` directory structure

**Phase 3 Interface:**
- WorldState provides inputs for fuzzy controller
- ObstacleMap.get_free_sectors() → navigation decisions
- CubeObservation.color + distance → approach/grasp decisions

**Status:** Data model design complete, ready for contract definition.
