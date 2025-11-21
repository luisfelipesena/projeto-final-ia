# Data Model: Sensor Exploration and Control Validation

**Feature**: 002-sensor-exploration
**Date**: 2025-11-21
**Phase**: Phase 1 Design

## Overview

This document defines the data entities and structures for YouBot control validation and sensor data analysis. All entities are technology-agnostic (no Python classes, just conceptual models) to maintain separation between specification and implementation.

---

## Entity 1: Base Movement Command

**Description**: Represents a velocity command for the YouBot's omnidirectional base using mecanum wheels.

### Attributes

| Attribute | Type | Unit | Range | Description |
|-----------|------|------|-------|-------------|
| vx | Float | m/s | [-0.5, 0.5] | Linear velocity in X-axis (forward/backward) |
| vy | Float | m/s | [-0.5, 0.5] | Linear velocity in Y-axis (strafe left/right) |
| omega | Float | rad/s | [-1.0, 1.0] | Angular velocity (rotation clockwise/CCW) |
| timestamp | Float | seconds | [0, ∞) | Simulation time when command issued |

**Note**: Ranges are estimated limits to be validated during testing (FR-006).

### Validation Rules

- At least one velocity component must be non-zero for movement
- All three can be combined for diagonal motion with rotation
- Zero vector `(0, 0, 0)` commands full stop

### State Transitions

```
STATIONARY → (vx≠0 OR vy≠0 OR omega≠0) → MOVING
MOVING → (vx=0 AND vy=0 AND omega=0) → STATIONARY
```

### Relationships

- **Commands → Base State**: Each command updates the current motion state
- **Test Script → Commands**: Test script generates predefined command sequences

---

## Entity 2: Arm Position

**Description**: Represents the spatial configuration of the YouBot's 5-DOF manipulator arm using preset positions.

### Attributes

| Attribute | Type | Values | Description |
|-----------|------|--------|-------------|
| height | Enum | {FLOOR, FRONT, HIGH} | Vertical positioning preset |
| orientation | Enum | {FRONT, DOWN} | End-effector orientation preset |
| joint_angles | Tuple[Float] | 5 floats | Actual joint angles (radians) for each DOF |
| reach | Float (m) | [0, 0.655] | Distance from base center to gripper |
| timestamp | Float (s) | [0, ∞) | Simulation time of position |

**Note**: Preset names are illustrative - actual names from `arm.py` to be documented.

### Validation Rules

- Preset combinations must be mechanically feasible
- Joint angles must respect physical limits (to be measured)
- Reach cannot exceed 655mm (YouBot specification)

### State Transitions

```
REST → set_height(preset) → MOVING_ARM → POSITIONED
POSITIONED → set_orientation(preset) → MOVING_ARM → POSITIONED
```

### Relationships

- **Presets → Joint Angles**: Each preset maps to specific joint configuration
- **Arm Position → Gripper State**: Arm must be positioned before grasping

---

## Entity 3: Gripper State

**Description**: Represents the binary state of the YouBot's parallel jaw gripper.

### Attributes

| Attribute | Type | Values | Description |
|-----------|------|--------|-------------|
| is_closed | Boolean | {True, False} | Gripper jaw state (closed = gripping) |
| jaw_width | Float (mm) | [0, 50] | Distance between gripper fingers |
| grip_force | Float (N) | [0, max_force] | Force applied to grasped object |
| timestamp | Float (s) | [0, ∞) | Simulation time of state change |

**Note**: Max force to be determined from testing or Webots specs.

### Validation Rules

- Closed state (is_closed=True) implies jaw_width < threshold (~5mm)
- Open state (is_closed=False) implies jaw_width ≈ 50mm
- Grip force only meaningful when closed

### State Transitions

```
OPEN → grip() → CLOSING → CLOSED
CLOSED → release() → OPENING → OPEN
```

### Relationships

- **Gripper State → Object Grasped**: Closed state required to hold cube
- **Arm Position → Gripper State**: Arm positioning precedes gripper action

---

## Entity 4: LIDAR Scan

**Description**: Represents a single 2D LIDAR measurement consisting of distance readings at fixed angular intervals.

### Attributes

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| range_image | Array[Float] | meters | 1D array of distance measurements |
| num_points | Integer | count | Number of measurement points (= array length) |
| fov | Float | radians | Horizontal field of view |
| angular_resolution | Float | radians | Angle between consecutive measurements |
| max_range | Float | meters | Maximum detection distance |
| min_range | Float | meters | Minimum detection distance |
| timestamp | Float | seconds | Simulation time of scan |

### Derived Values

- **angular_resolution** = fov / (num_points - 1)
- **angles** = linspace(-fov/2, fov/2, num_points)

### Validation Rules

- All range values must be in [min_range, max_range] or infinity
- num_points must match array length
- Infinity values indicate no obstacle detected in that direction

### Relationships

- **LIDAR Scan → Obstacles**: Each scan may detect multiple obstacles
- **LIDAR Scan → Visualization**: Scan data generates polar plot

---

## Entity 5: Camera Frame

**Description**: Represents a single RGB image captured from the YouBot's camera at a specific moment.

### Attributes

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| image_data | Array[UInt8] | bytes | Raw BGRA image buffer |
| width | Integer | pixels | Image width |
| height | Integer | pixels | Image height |
| channels | Integer | count | Number of color channels (4 for BGRA) |
| fov | Float | radians | Horizontal field of view |
| timestamp | Float | seconds | Simulation time of capture |

### Image Format

- **Color Order**: BGRA (Blue, Green, Red, Alpha)
- **Data Layout**: Row-major, height × width × 4 bytes
- **Pixel Access**: `pixel(x, y) = image_data[y * width * 4 + x * 4 : y * width * 4 + x * 4 + 4]`

### Derived Values

- **RGB Conversion**: Drop alpha channel, reverse BGR → RGB
- **HSV Conversion**: Apply color space transformation for thresholding
- **Grayscale**: Luminance = 0.299*R + 0.587*G + 0.114*B

### Validation Rules

- width and height must be positive integers
- image_data length must equal width × height × 4
- All pixel values in [0, 255]

### Relationships

- **Camera Frame → Cube Detection**: Image processing identifies colored cubes
- **Camera Frame → Saved Images**: Example frames stored for analysis

---

## Entity 6: Obstacle

**Description**: Represents a detected object in the arena identified through LIDAR or camera analysis.

### Attributes

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| position_polar | Tuple[Float, Float] | (m, rad) | (distance, angle) in robot frame |
| position_cartesian | Tuple[Float, Float] | (m, m) | (x, y) in robot frame |
| type | Enum | {LIDAR_POINT, WOODEN_BOX, WALL} | Obstacle classification |
| distance | Float | meters | Distance from robot center |
| angle | Float | radians | Angle from robot forward direction |

### Derived Values

- **Cartesian from Polar**: x = distance * cos(angle), y = distance * sin(angle)
- **Polar from Cartesian**: distance = sqrt(x² + y²), angle = atan2(y, x)

### Validation Rules

- distance must be positive
- angle typically in [-π, π]
- position_polar and position_cartesian must be consistent

### Relationships

- **LIDAR Scan → Obstacles**: Each scan identifies multiple obstacles
- **Obstacles → Navigation**: Future phases use obstacle positions for avoidance

---

## Entity 7: Arena Map

**Description**: Represents the documented spatial layout of the IA_20252 arena including boundaries, deposit boxes, obstacles, and spawn zones.

### Attributes

| Attribute | Type | Unit | Description |
|-----------|------|------|-------------|
| length | Float | meters | Arena length (X-axis) |
| width | Float | meters | Arena width (Y-axis) |
| boundaries | Array[Tuple] | [(x,y), ...] | Corner coordinates of arena |
| deposit_boxes | Array[DepositBox] | - | Positions and colors of target boxes |
| obstacles | Array[ObstaclePosition] | - | Fixed wooden box positions |
| spawn_zone | BoundingBox | - | Cube spawn area boundaries |

### Sub-Entity: DepositBox

| Attribute | Type | Description |
|-----------|------|-------------|
| color | Enum | {GREEN, BLUE, RED} |
| position | Tuple[Float] | (x, y, z) in world frame |
| description | String | Human-readable location (e.g., "top center") |

### Sub-Entity: ObstaclePosition

| Attribute | Type | Description |
|-----------|------|-------------|
| type | String | "WoodenBox" |
| position | Tuple[Float] | (x, y, z) in world frame |
| dimensions | Tuple[Float] | (length, width, height) in meters |

### Sub-Entity: BoundingBox

| Attribute | Type | Description |
|-----------|------|-------------|
| x_min | Float | Minimum X coordinate |
| x_max | Float | Maximum X coordinate |
| y_min | Float | Minimum Y coordinate |
| y_max | Float | Maximum Y coordinate |

### Validation Rules

- All coordinates must be within arena boundaries
- Deposit boxes and obstacles must not overlap
- Spawn zone must be subset of arena dimensions

### Relationships

- **Arena Map → Path Planning**: Future phases use map for navigation
- **Arena Map → Test Coverage**: Ensures robot can reach all areas

---

## Data Flow Diagram

```
┌─────────────┐
│   Webots    │
│  Simulator  │
└──────┬──────┘
       │
       ├─► Base Commands ──► Base Movement
       ├─► Arm Commands ───► Arm Position
       ├─► Gripper Commands► Gripper State
       │
       ├─► LIDAR Device ───► LIDAR Scan ──► Obstacles
       ├─► Camera Device ──► Camera Frame ─► Cube Detection
       │
       └─► World File (.wbt) ─► Arena Map
                                    │
                                    └─► Navigation (Phase 4)
```

---

## Entity Relationships

```
Test Script
  │
  ├─► Generates Base Movement Commands ─► Updates Base State
  ├─► Generates Arm Position Commands ──► Updates Arm Position
  └─► Generates Gripper Commands ───────► Updates Gripper State

Sensor Analysis
  │
  ├─► Captures LIDAR Scans ─────► Extracts Obstacles
  │                                     │
  │                                     └─► Visualized in Polar Plots
  │
  └─► Captures Camera Frames ───► Detects Cube Colors
                                        │
                                        └─► Saved as Example Images

Arena Parsing
  │
  └─► Parses World File (.wbt) ─► Generates Arena Map
                                        │
                                        └─► Documented in arena_map.md
```

---

## Data Persistence

### Files Generated

| File | Entity/Data | Format | Size Estimate |
|------|-------------|--------|---------------|
| `lidar_sample.npy` | LIDAR Scan (range_image) | NumPy binary | ~10 KB |
| `camera_sample.npy` | Camera Frame (RGB) | NumPy binary | ~1 MB (640×480×3) |
| `green_cube.png` | Camera Frame (example) | PNG image | ~100 KB |
| `blue_cube.png` | Camera Frame (example) | PNG image | ~100 KB |
| `red_cube.png` | Camera Frame (example) | PNG image | ~100 KB |
| `lidar_scan.png` | LIDAR Visualization | PNG image | ~200 KB |
| `docs/arena_map.md` | Arena Map (markdown) | Text | ~5 KB |

### Test Logs

Test execution generates structured logs in `logs/`:
- `test_basic_controls.log` - Control validation results
- `sensor_analysis.log` - Sensor data collection results

---

## Data Quality Requirements

### LIDAR Scan Quality

- **Coverage**: >80% of FOV should return valid (non-infinity) readings in arena
- **Consistency**: Repeated scans from same position should have <5% variance
- **Accuracy**: Measured distances within ±5% of known obstacle distances

### Camera Frame Quality

- **Resolution**: Minimum 640×480 pixels (to be documented)
- **Frame Rate**: Minimum 10 FPS for real-time processing (Phase 2)
- **Color Fidelity**: RGB values should distinguish green/blue/red cubes (>80% accuracy)

### Control Command Quality

- **Repeatability**: Same command should produce similar motion (±10% variance)
- **Response Time**: Observable motion within 1 simulation step (~32ms)
- **Stability**: Commands should not cause oscillations or instability

---

**Data Model Status**: ✅ COMPLETE
**All entities defined. Ready for API contract generation.**
