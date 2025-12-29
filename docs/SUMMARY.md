# YouBot Cube Collection — Technical Summary

## Project Overview

**Objective**: Collect 15 colored cubes (red, green, blue) and deposit each in the corresponding color box.

**Platform**: Webots R2023b simulator with KUKA YouBot robot.

**Mandatory Requirements**:
- Neural Network (MLP or CNN) for classification task
- Fuzzy Logic for control
- NO GPS allowed (odometry-only localization)
- Fully autonomous (no teleoperation)

## AI Techniques Implemented

| Technique | Application | Implementation |
|-----------|-------------|----------------|
| **CNN** | Color classification | MobileNetV3-Small (transfer learning) |
| **Fuzzy Logic** | Reactive navigation | Trapezoidal membership functions |
| **A\*** | Global path planning | 12cm grid with obstacle inflation |

## Hardware Configuration

### KUKA YouBot
- **Base**: Mecanum wheels (omnidirectional)
- **Arm**: 5-DOF manipulator
- **Gripper**: Parallel jaw gripper
- **Dimensions**: ~58cm × 38cm

### Mecanum Kinematics
```
ω_wheel = (1/R) × [Vx ± Vy ± (LX+LY)×ω]

Parameters:
- R = 0.05m (wheel radius)
- LX = 0.228m (half wheelbase X)
- LY = 0.158m (half wheelbase Y)
```

## Sensors

### LIDAR (4 units)
- **Field of View**: 180° each (π radians)
- **Resolution**: 180 rays per LIDAR
- **Range**: 0.1m – 5.0m
- **Positions**: front (0°), rear (180°), left (90°), right (-90°)

### Camera
- **Resolution**: 128 × 128 pixels
- **Recognition API**: Enabled with maxRange = 3m
- **Position**: Front-mounted at (0.27, 0, -0.06)

### IR Distance Sensors (8 units)
- **Type**: Infra-red
- **Range**: 0 – 1m (via lookup table)
- **Positions**: front, front_left, front_right, rear, rear_left, rear_right, left, right

## Webots Recognition API

The Recognition API is a Webots simulator feature that returns information about objects visible to the camera.

**WbCameraRecognitionObject** fields:
- `position`: 3D position relative to camera
- `orientation`: Quaternion orientation
- `size`: Bounding box dimensions
- `colors`: Array of recognition colors (from DEF recognitionColors)
- `model`: Object model name

**Important**: Recognition API uses scene graph data (ground truth). For the mandatory neural network requirement, we use our own CNN to classify the cube color from the raw camera image.

```python
# Recognition API provides: WHERE is the cube
objects = camera.getRecognitionObjects()
for obj in objects:
    relative_pos = obj.getPosition()  # [x, y, z] from camera

# CNN provides: WHAT COLOR is the cube (mandatory requirement)
color = color_classifier.classify(camera_image)
```

## CNN: MobileNetV3-Small

### Architecture
MobileNetV3-Small is a lightweight CNN optimized for mobile devices (Howard et al., 2019).

**Key innovations**:
- **Depthwise Separable Convolutions**: Reduces parameters and computation
- **Squeeze-and-Excitation (SE) blocks**: Channel attention mechanism
- **Hard-Swish activation**: Efficient approximation of Swish

### Depthwise Separable Convolution
```
Traditional: O(k² × Ci × Co × H × W)
Depthwise:   O(k² × Ci × H × W)     (one filter per channel)
Pointwise:   O(Ci × Co × H × W)     (1×1 conv to combine)
Combined:    O(k² × Ci + Ci × Co)   (8-9× fewer operations)
```

### Transfer Learning Strategy
1. **Phase 1**: Freeze backbone, train classifier only (256 → 3 neurons)
2. **Phase 2**: Unfreeze backbone, fine-tune with low learning rate

### Performance
- **Accuracy**: 99.4% on test set
- **Input**: 64×64×3 RGB normalized
- **Output**: Softmax probabilities for {red, green, blue}
- **Fallback**: HSV heuristics when confidence < 0.5

## Fuzzy Logic Navigation

### Membership Functions
Distance to obstacle membership (trapezoidal):

```
μ(d)
1.0 |████████
    |        ████████
    |                ████████
0.0 |___|____|____|____|____
    0  0.25 0.45 0.65  2.0  d(m)
       very_close close  far
```

| Term | Peak Region | Description |
|------|-------------|-------------|
| `muito_perto` | 0 – 0.25m | Emergency zone |
| `perto` | ~0.45m | Caution zone |
| `longe` | > 0.65m | Safe zone |

### Fuzzy Rules

**Obstacle Avoidance Rules**:
```
IF distance = muito_perto THEN velocity = reverse, strafe = lateral
IF distance = perto THEN velocity = slow
IF lateral_left = blocked THEN strafe = right
IF lateral_right = blocked THEN strafe = left
```

**Target Alignment Rules**:
```
IF angle = large THEN velocity = stop, omega = maximum
IF angle = medium THEN velocity = slow, omega = proportional
IF angle = small AND distance = far THEN velocity = fast
```

### Defuzzification
- **Method**: Weighted centroid
- **Outputs**: Vx (linear), Vy (strafe), ω (angular)

### Car-Like Navigation
Navigation prioritizes forward motion with minimal lateral correction:
```python
# Fuzzy computes base velocities
vx, vy, omega = fuzzy_nav.compute(target_dist, target_angle, obs_front, obs_left, obs_right)

# Limit lateral to maintain car-like behavior
vy = max(-0.06, min(0.06, vy))  # Max 6cm/s lateral
```

**Key principle**: Robot always moves primarily forward, using small lateral corrections and rotation to avoid obstacles. No pure lateral strafe (holonomic mode).

## A* Path Planning

### Algorithm
A* (Hart, Nilsson, Raphael, 1968) finds optimal paths using:
```
f(n) = g(n) + h(n)

g(n) = actual cost from start to n
h(n) = heuristic estimate from n to goal (Manhattan distance)
```

### Occupancy Grid
- **Cell size**: 12cm × 12cm
- **Grid dimensions**: 58 × 33 cells (7m × 4m arena)
- **States**: FREE, OCCUPIED, INFLATED, UNKNOWN

### Obstacle Inflation
Robot is not a point - YouBot is 58cm × 38cm. Obstacles are inflated by 30cm (half diagonal + margin) to ensure safe paths.

```
┌─────────────┐
│  ░░░░░░░░░  │  ░ = inflated zone (30cm)
│  ░ ████ ░  │  █ = actual obstacle
│  ░░░░░░░░░  │
└─────────────┘
```

### LIDAR Raycasting Update
```python
def update_from_lidar(robot_pos, heading, ranges):
    for i, r in enumerate(ranges):
        angle = heading + ray_angles[i]
        # Bresenham line: mark cells as FREE
        for cell in bresenham(robot_pos, endpoint):
            grid[cell] = FREE
        # Endpoint: mark as OCCUPIED (if hit)
        if r < max_range:
            grid[endpoint] = OCCUPIED
```

## Strategic Waypoints

Routes use gradual waypoints (~20° angle change between consecutive points) for smooth car-like navigation:

**RED box route** (avoiding obstacle A at 0.6, 0.0):
```
(-0.25, 0.10) → (0.0, 0.25) → (0.25, 0.45) → (0.50, 0.60) →
(0.75, 0.70) → (1.00, 0.70) → (1.25, 0.60) → (1.45, 0.45) →
(1.60, 0.25) → (1.75, 0.10) → (1.85, 0.01) → RED_BOX
```

**Waypoint timeout**: If robot cannot reach a waypoint within 15 seconds, it skips to the next one.

## Finite State Machine

### States
```
┌─────────┐     ┌──────────┐     ┌─────────┐
│ SEARCH  │ ──► │ APPROACH │ ──► │  GRASP  │
└─────────┘     └──────────┘     └─────────┘
     ▲                                │
     │                                ▼
┌──────────────┐     ┌──────┐     ┌────────┐
│RETURN_TO_SPAWN│◄── │ DROP │ ◄── │ TO_BOX │
└──────────────┘     └──────┘     └────────┘
```

| State | Description |
|-------|-------------|
| `SEARCH` | Lawnmower exploration, detect cubes via Recognition API |
| `APPROACH` | Align and approach detected cube |
| `GRASP` | Lower arm, close gripper, verify grip |
| `TO_BOX` | Navigate to destination box using A* waypoints |
| `DROP` | Lower arm over box, release cube |
| `RETURN_TO_SPAWN` | Navigate back to spawn position |

## Arena Configuration

```
                    GREEN (0.48, 1.58)
                         ▲
    ┌─────────────────────────────────────┐
    │  D                     C            │
    │       E                             │
    │                  A          RED ──► │ (2.31, 0.01)
    │       F                             │
    │  G                     B            │
    └─────────────────────────────────────┘
                         ▼
                    BLUE (0.48, -1.62)

    SPAWN ● (-3.91, 0)
```

**Arena**: 7m × 4m, center at (-0.79, 0)

**Obstacles** (WoodenBox, 30cm × 30cm):
- A: (0.60, 0.00) - Central
- B: (1.96, -1.24) - Southeast
- C: (1.95, 1.25) - Northeast
- D: (-2.28, 1.50) - Northwest
- E: (-1.02, 0.75) - Center-north
- F: (-1.02, -0.74) - Center-south
- G: (-2.27, -1.51) - Southwest

## Critical Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EMERGENCY_STOP` | 0.20m | Immediate reverse |
| `WP_ARRIVAL` | 0.35-0.40m | Waypoint reached |
| `WP_TIMEOUT` | 15s | Skip unreachable waypoint |
| `ALIGN_TOLERANCE` | 0.20 rad (~11°) | Heading alignment |
| `GRASP_ATTEMPTS` | 6 max | Skip cube after failures |
| `GRASP_FALLBACK` | 3 attempts | Switch to horizontal arm pose |

## Code Architecture

```
controllers/youbot/
├── constants.py          # Arena configuration, box positions
├── routes.py             # Strategic waypoint generation
├── occupancy_grid.py     # A* pathfinding, grid management
├── fuzzy_navigator.py    # Fuzzy logic controller
├── color_classifier.py   # MobileNetV3 CNN
├── base.py               # Mecanum wheel control
├── arm.py                # 5-DOF arm control
├── gripper.py            # Gripper control
└── youbot_controller.py  # Main FSM controller (~1700 lines)
```

## Algorithm Limitations and Mitigations

### A* (Hart et al., 1968)
- **Limitation**: Paths may pass close to obstacles
- **Mitigation**: Obstacle inflation by 30cm

### Fuzzy Logic (Zadeh, 1965)
- **Limitation**: Can oscillate in narrow passages
- **Mitigation**: Car-like navigation with limited lateral velocity (max 6cm/s), timeout-based waypoint skip

### CNN/MobileNetV3 (Howard et al., 2019)
- **Limitation**: Sensitive to lighting variations
- **Mitigation**: HSV fallback when confidence < 0.5

## References

1. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

2. Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338-353.

3. Howard, A., Sandler, M., Chen, B., Wang, W., Chen, L. C., Tan, M., ... & Adam, H. (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 1314-1324.

4. Webots Documentation: https://cyberbotics.com/doc/reference/camera#wb_camera_recognition_get_objects
