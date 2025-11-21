# Sensor Data API Specification

**Feature**: 002-sensor-exploration
**Contract**: Data structures and processing functions for LIDAR and Camera sensors
**Date**: 2025-11-21

## Overview

This contract defines the data structures and analysis functions for FR-014 through FR-026 (sensor analysis requirements). Implementation location: `notebooks/01_sensor_exploration.ipynb`.

---

## LIDAR Data API

### Function: capture_lidar_specifications

**Requirement**: FR-015 - Document LIDAR data format

**Signature:**
```python
def capture_lidar_specifications(lidar: Device) -> dict:
    """
    Extract and document LIDAR sensor specifications.

    Args:
        lidar: Webots LIDAR device object

    Returns:
        dict: Sensor specifications
    """
```

**Return Structure:**
```python
{
    'horizontal_resolution': int,  # Number of points per scan
    'num_layers': int,             # Number of vertical layers
    'fov': float,                  # Horizontal field of view (radians)
    'vertical_fov': float,         # Vertical field of view (radians)
    'min_range': float,            # Minimum detection distance (m)
    'max_range': float,            # Maximum detection distance (m)
    'angular_resolution': float,   # Angle between points (radians)
}
```

**Success Criteria** (FR-015):
- All fields populated with numeric values
- Values match Webots LIDAR device configuration
- Documented in notebook markdown cells

---

### Function: capture_lidar_scan

**Requirement**: FR-014 - Read LIDAR range_image data

**Signature:**
```python
def capture_lidar_scan(lidar: Device) -> np.ndarray:
    """
    Capture single LIDAR scan.

    Args:
        lidar: Webots LIDAR device (enabled)

    Returns:
        np.ndarray: 1D array of distance measurements (meters)
    """
```

**Return Format:**
- Type: `numpy.ndarray`
- Shape: `(horizontal_resolution,)`
- Dtype: `float64`
- Values: Distance in meters or `np.inf` for no detection

**Success Criteria** (FR-014):
- Successfully reads range_image without errors
- Array length matches horizontal_resolution
- Contains valid distance measurements

---

### Function: analyze_lidar_ranges

**Requirement**: FR-016 - Document detection ranges

**Signature:**
```python
def analyze_lidar_ranges(scans: list[np.ndarray], specs: dict) -> dict:
    """
    Analyze LIDAR detection range capabilities.

    Args:
        scans: List of LIDAR scans captured at various distances
        specs: LIDAR specifications from capture_lidar_specifications

    Returns:
        dict: Range analysis results
    """
```

**Return Structure:**
```python
{
    'observed_min_range': float,   # Minimum distance observed (m)
    'observed_max_range': float,   # Maximum distance before infinity (m)
    'mean_range': float,           # Average detection distance (m)
    'std_range': float,            # Standard deviation (m)
    'valid_readings_pct': float,   # % of non-infinity readings
}
```

**Success Criteria** (FR-016):
- Documents min/max ranges with ≥10 test scans
- Ranges within spec limits: [min_range, max_range]
- Documented in notebook

---

### Function: visualize_lidar_polar

**Requirement**: FR-017 - Create polar visualizations

**Signature:**
```python
def visualize_lidar_polar(
    range_image: np.ndarray,
    fov: float,
    max_range: float,
    title: str = "LIDAR Scan"
) -> matplotlib.figure.Figure:
    """
    Generate polar plot of LIDAR scan.

    Args:
        range_image: 1D array of distances
        fov: Field of view (radians)
        max_range: Maximum range for plot scaling
        title: Plot title

    Returns:
        matplotlib.figure.Figure: Polar plot figure
    """
```

**Plot Requirements:**
- Projection: `polar`
- Theta range: [-fov/2, fov/2]
- Radius range: [0, max_range]
- Colormap: 'RdYlGn_r' (red=close, green=far)
- Grid: Enabled with radial labels

**Success Criteria** (FR-017):
- Plot clearly shows arena boundaries
- Obstacles visible as concentrated points
- Saved as PNG: `lidar_scan_example.png`

---

### Function: identify_obstacles_lidar

**Requirement**: FR-018 - Identify and mark obstacles

**Signature:**
```python
def identify_obstacles_lidar(
    range_image: np.ndarray,
    fov: float,
    threshold: float = 1.0
) -> list[dict]:
    """
    Detect obstacles from LIDAR scan using distance threshold.

    Args:
        range_image: 1D array of distances
        fov: Field of view (radians)
        threshold: Distance threshold for obstacle classification (m)

    Returns:
        list[dict]: Detected obstacles
    """
```

**Return Structure:**
```python
[
    {
        'distance': float,        # Distance to obstacle (m)
        'angle': float,           # Angle in robot frame (rad)
        'index': int,             # Index in range_image
        'confidence': float,      # 0-1 score based on neighbors
    },
    ...
]
```

**Success Criteria** (FR-018):
- Identifies wooden boxes and arena walls
- Marks obstacles in visualization with distinct color
- Documents detection method in notebook

---

## Camera Data API

### Function: capture_camera_specifications

**Requirement**: FR-021, FR-022 - Document camera resolution and FPS

**Signature:**
```python
def capture_camera_specifications(camera: Device) -> dict:
    """
    Extract and document camera sensor specifications.

    Args:
        camera: Webots Camera device object

    Returns:
        dict: Camera specifications
    """
```

**Return Structure:**
```python
{
    'width': int,              # Image width (pixels)
    'height': int,             # Image height (pixels)
    'fov': float,              # Horizontal field of view (radians)
    'fps': float,              # Frames per second (estimated)
    'channels': int,           # Number of color channels (4 for BGRA)
    'has_recognition': bool,   # Object recognition capability
}
```

**Success Criteria** (FR-021, FR-022):
- All fields documented
- FPS measured over 100 frames
- Documented in notebook

---

### Function: capture_camera_frame

**Requirement**: FR-020 - Capture RGB camera frames

**Signature:**
```python
def capture_camera_frame(camera: Device) -> np.ndarray:
    """
    Capture single camera frame and convert to RGB.

    Args:
        camera: Webots Camera device (enabled)

    Returns:
        np.ndarray: RGB image array (height, width, 3)
    """
```

**Return Format:**
- Type: `numpy.ndarray`
- Shape: `(height, width, 3)`
- Dtype: `uint8`
- Color order: RGB (converted from BGRA)

**Success Criteria** (FR-020):
- Successfully captures frame without errors
- RGB conversion correct
- Frame dimensions match camera specs

---

### Function: save_example_images

**Requirement**: FR-023 - Save example images for each cube color

**Signature:**
```python
def save_example_images(
    camera: Device,
    colors: list[str] = ['green', 'blue', 'red'],
    output_dir: str = 'media/cube_examples'
) -> dict[str, str]:
    """
    Capture and save example images of colored cubes.

    Args:
        camera: Webots Camera device
        colors: List of cube colors to capture
        output_dir: Directory to save images

    Returns:
        dict: Mapping of color -> filename
    """
```

**Return Structure:**
```python
{
    'green': 'media/cube_examples/green_cube_001.png',
    'blue': 'media/cube_examples/blue_cube_001.png',
    'red': 'media/cube_examples/red_cube_001.png',
}
```

**Success Criteria** (FR-023):
- At least 1 clear image per color
- Images show cube in focus
- File paths documented in notebook

---

### Function: detect_color_threshold

**Requirement**: FR-024 - Implement simple RGB threshold color detection

**Signature:**
```python
def detect_color_threshold(
    image_rgb: np.ndarray,
    method: str = 'hsv'
) -> dict:
    """
    Classify cube color using threshold method.

    Args:
        image_rgb: RGB image array (H, W, 3)
        method: 'hsv' or 'rgb' threshold method

    Returns:
        dict: Detection results
    """
```

**Return Structure:**
```python
{
    'detected_color': str,      # 'green', 'blue', 'red', or 'unknown'
    'confidence': float,        # 0-1 score (pixel count ratio)
    'pixel_counts': {           # Pixels matching each color
        'green': int,
        'blue': int,
        'red': int,
    },
    'method': str,              # 'hsv' or 'rgb'
}
```

**HSV Thresholds** (to be calibrated):
```python
HSV_RANGES = {
    'green': ((40, 50, 50), (80, 255, 255)),
    'blue': ((90, 50, 50), (130, 255, 255)),
    'red': ((0, 50, 50), (10, 255, 255)),  # Also check (170-180)
}
```

**Success Criteria** (FR-024):
- Implements HSV threshold classification
- Handles red hue wraparound (0/180)
- Returns confidence score

---

### Function: evaluate_color_detection_accuracy

**Requirement**: FR-025 - Document color detection accuracy

**Signature:**
```python
def evaluate_color_detection_accuracy(
    test_images: list[tuple[np.ndarray, str]]
) -> dict:
    """
    Evaluate threshold method on labeled test set.

    Args:
        test_images: List of (image, ground_truth_color) pairs

    Returns:
        dict: Accuracy metrics
    """
```

**Return Structure:**
```python
{
    'overall_accuracy': float,     # Correct / total (target: >0.80)
    'per_color_accuracy': {        # Accuracy per color
        'green': float,
        'blue': float,
        'red': float,
    },
    'confusion_matrix': np.ndarray,  # 3x3 confusion matrix
    'num_samples': int,            # Total test images
}
```

**Success Criteria** (FR-025):
- Overall accuracy >80% (SC-008)
- Tested on ≥30 images (10 per color)
- Results documented in notebook

---

## Data Processing Pipeline

### LIDAR Analysis Pipeline

```python
# 1. Capture specifications
specs = capture_lidar_specifications(lidar)

# 2. Collect multiple scans
scans = []
for _ in range(20):
    robot.step(time_step)
    scan = capture_lidar_scan(lidar)
    scans.append(scan)

# 3. Analyze ranges
range_analysis = analyze_lidar_ranges(scans, specs)

# 4. Visualize example scan
fig = visualize_lidar_polar(scans[0], specs['fov'], specs['max_range'])
plt.savefig('lidar_scan_example.png')

# 5. Detect obstacles
obstacles = identify_obstacles_lidar(scans[0], specs['fov'], threshold=1.0)

# 6. Document findings (FR-019)
print(f"LIDAR Specifications: {specs}")
print(f"Range Analysis: {range_analysis}")
print(f"Obstacles Detected: {len(obstacles)}")
```

### Camera Analysis Pipeline

```python
# 1. Capture specifications
specs = capture_camera_specifications(camera)

# 2. Measure FPS
fps = measure_camera_fps(camera, duration=10)
specs['fps'] = fps

# 3. Capture example images
example_files = save_example_images(camera, colors=['green', 'blue', 'red'])

# 4. Test color detection
test_images = []
for color, filepath in example_files.items():
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_images.append((img_rgb, color))

# 5. Evaluate accuracy
accuracy_results = evaluate_color_detection_accuracy(test_images)

# 6. Document findings (FR-026)
print(f"Camera Specifications: {specs}")
print(f"Color Detection Accuracy: {accuracy_results['overall_accuracy']:.1%}")
```

---

## Notebook Structure

**File**: `notebooks/01_sensor_exploration.ipynb`

**Required Sections:**

1. **Setup**
   - Import libraries
   - Initialize Webots robot
   - Enable LIDAR and camera sensors

2. **LIDAR Analysis** (FR-014 through FR-019)
   - Specifications documentation
   - Range analysis
   - Polar visualizations
   - Obstacle detection

3. **Camera Analysis** (FR-020 through FR-026)
   - Specifications documentation
   - Example image capture
   - Color detection implementation
   - Accuracy evaluation

4. **Results Summary**
   - Key findings table
   - Performance metrics
   - Recommendations for Phase 2

**Output Artifacts** (FR-019, FR-026):
- LIDAR visualizations (PNG)
- Camera example images (PNG)
- Sensor specifications (markdown tables)
- Accuracy metrics (tables/plots)

---

## Data Quality Validation

### LIDAR Data Quality

```python
def validate_lidar_data(scan: np.ndarray, specs: dict) -> bool:
    """Check if LIDAR scan is valid."""
    checks = {
        'correct_length': len(scan) == specs['horizontal_resolution'],
        'has_valid_readings': np.sum(np.isfinite(scan)) > 0,
        'within_range': np.all((scan >= specs['min_range']) | np.isinf(scan)),
        'not_all_inf': np.sum(np.isfinite(scan)) > len(scan) * 0.1,  # >10% valid
    }
    return all(checks.values())
```

### Camera Data Quality

```python
def validate_camera_frame(frame: np.ndarray, specs: dict) -> bool:
    """Check if camera frame is valid."""
    checks = {
        'correct_shape': frame.shape == (specs['height'], specs['width'], 3),
        'correct_dtype': frame.dtype == np.uint8,
        'has_content': np.mean(frame) > 10,  # Not all black
        'not_saturated': np.mean(frame) < 245,  # Not all white
    }
    return all(checks.values())
```

---

**Contract Status**: ✅ COMPLETE
**All sensor API specifications defined. Ready for implementation.**
