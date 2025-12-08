# YouBot Cube Collection - Technical Summary

## Project Overview
Autonomous robot (KUKA YouBot) that collects 15 colored cubes (red, green, blue) and deposits them in corresponding colored boxes using LIDAR, RGB camera, neural networks, and fuzzy logic control.

**Course**: MATA64 - Artificial Intelligence (UFBA)

---

## Technical Decisions

### 1. Perception System

#### LIDAR (180° FOV, 5m range)
- **Purpose**: Obstacle detection and environment mapping
- **Implementation**: Processes 180 range readings to detect obstacles at front/left/right sectors
- **Grid Mapping**: Occupancy grid (12cm cells) updated via Bresenham raycasting algorithm

#### RGB Camera + Recognition API
- **Purpose**: Cube detection and color classification
- **Webots Recognition API**: Returns object position relative to camera in real-time
- **Color Classification**: Neural network (primary) + HSV heuristics (fallback when confidence < 0.5)

### 2. Neural Network Color Classifier

#### Architecture
Based on **MobileNetV3-Small** - a lightweight convolutional neural network optimized for mobile/embedded devices (Howard et al., 2019).

| Layer | Description |
|-------|-------------|
| Backbone | MobileNetV3-Small (pre-trained ImageNet) |
| Head | Custom classifier (256 → 3 classes) |
| Input | 64x64 RGB normalized (ImageNet mean/std) |
| Output | Softmax probabilities for red/green/blue |

#### Training Strategy
Two-phase transfer learning approach:
1. **Feature Extraction**: Freeze backbone, train classifier head only
2. **Fine-tuning**: Unfreeze entire network with reduced learning rate

| Metric | Value |
|--------|-------|
| Validation Accuracy | 99.4% |
| Dataset | Synthetic cubes with augmentation |
| Augmentations | Brightness, noise, blur variations |

#### Inference
- **Format**: ONNX for cross-platform compatibility
- **Fallback**: HSV heuristics (`color_from_rgb()`) when model confidence < 0.5

### 3. A* Path Planning Algorithm

The robot uses the **A* algorithm** (Hart, Nilsson & Raphael, 1968) for global path planning around obstacles.

#### Implementation (`OccupancyGrid.plan_path()`)

```
f(n) = g(n) + h(n)

where:
- g(n) = cost from start to node n
- h(n) = heuristic estimate (Manhattan distance to goal)
- f(n) = estimated total cost through node n
```

#### Grid Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Cell Size | 12cm | Balance between resolution and computation |
| Arena | 7m x 4m | World bounds from .wbt file |
| Grid Dimensions | 58 x 33 cells | Covers entire arena |

#### Obstacle Inflation
Critical for collision-free navigation with a non-point robot:

```python
wall_inflate = 0.30  # Robot half-width (~0.29m) + safety margin
```

This inflates all obstacles by the robot's half-width, ensuring A* generates paths with clearance for the entire robot body.

#### Waypoint Following
```python
waypoint = self._next_nav_point()  # Get next A* waypoint
if distance_to_waypoint < 0.25:    # Waypoint reached
    self._waypoints.pop(0)         # Pop and get next
```

### 4. Fuzzy Logic Navigation (FuzzyNavigator)

Based on Zadeh's **Fuzzy Set Theory** (1965), the navigation system uses linguistic variables and fuzzy rules for continuous, smooth control.

#### Membership Functions

| Variable | Function | Range | Description |
|----------|----------|-------|-------------|
| `_mu_close` | Triangular | 0-0.45m | Obstacle proximity danger |
| `_mu_very_close` | Triangular | 0-0.25m | Emergency stop trigger |
| `_mu_far` | Trapezoidal | 0.4-1.5m | Safe for full speed |
| `_mu_small_angle` | Triangular | 0-25° | Well-aligned with target |
| `_mu_medium_angle` | Triangular | 10-60° | Moderate alignment needed |
| `_mu_big_angle` | Triangular | 40-90° | Major rotation required |

#### Fuzzy Rules (simplified)
```
IF obstacle_very_close THEN reverse + strafe (NO rotation)
IF obstacle_close THEN reduce_speed
IF target_far AND angle_small THEN move_forward_fast
IF target_close AND aligned THEN approach_slow
IF angle_big THEN rotate_only
IF lateral_obstacle THEN strafe_away
```

#### Defuzzification
Weighted centroid method for continuous output:
```python
omega_fuzzy = (0.35 * mu_angle_big +
               0.18 * mu_angle_medium +
               0.05 * (1.0 - mu_angle_small))
omega = omega_fuzzy * angle_norm  # angle_norm in [-1, 1]
```

#### Output Limits
| Parameter | Range | Purpose |
|-----------|-------|---------|
| vx (forward) | -0.18 to 0.18 m/s | Base movement |
| vy (lateral) | -0.14 to 0.14 m/s | Strafe for obstacle avoidance |
| omega (angular) | -0.6 to 0.6 rad/s | Rotation control |

### 5. Localization (Odometry)

#### Mecanum Wheel Inverse Kinematics
The YouBot uses Mecanum wheels for omnidirectional movement:

```
vx = (w1 + w2 + w3 + w4) * R/4
vy = (-w1 + w2 + w3 - w4) * R/4
omega = (-w1 + w2 - w3 + w4) * R/(4*L)

where:
- R = wheel radius
- L = wheel base
- w1..w4 = wheel angular velocities
```

#### Pose Integration
```python
dx_world = cos(yaw) * vx - sin(yaw) * vy
dy_world = sin(yaw) * vx + cos(yaw) * vy
pose += [dx_world * dt, dy_world * dt, omega * dt]
```

#### Ground Truth Synchronization
To correct odometry drift, the system periodically syncs with Webots ground truth:
- **Search/Approach modes**: Every 2.0 seconds
- **TO_BOX mode**: Every 0.5 seconds (higher precision needed)

### 6. Finite State Machine Architecture

```
┌─────────┐    detect cube    ┌──────────┐
│  SEARCH │ ─────────────────▶│ APPROACH │
└────┬────┘                   └────┬─────┘
     │                              │
     │                         dist < 0.32m
     │                         AND aligned
     │                              │
     │  ┌─────────────────────┐    ▼
     │  │      DROP           │◀── ┌───────┐
     │  │ (release in box)    │    │ GRASP │
     │  └──────────┬──────────┘    └───┬───┘
     │             │                   │
     │         completed          has_object?
     │             │                   │
     │             ▼              yes  │  no
     └─────────────┘◀──────── TO_BOX ◀─┴───▶ SEARCH
```

#### State Descriptions

| State | Entry Condition | Exit Condition | Action |
|-------|-----------------|----------------|--------|
| SEARCH | Default / drop complete | Cube detected | Lawnmower pattern navigation |
| APPROACH | Cube detected | dist < 0.32m AND angle < 10° | Align then advance to cube |
| GRASP | Close to cube | Gripper verification | Arm down, close gripper, verify |
| TO_BOX | Grasp verified | dist < 0.35m to box | A* navigation to colored box |
| DROP | At box | Release complete | Release cube, reverse, rotate |

### 7. Grasp Verification System

#### Position Sensor Method
The gripper fingers have position sensors that indicate grip state:

| Condition | Finger Position | Interpretation |
|-----------|-----------------|----------------|
| Empty gripper | ~0.0 - 0.001m | Fingers fully closed |
| Holding 3cm cube | ~0.010 - 0.015m | Fingers stopped by object |
| Detection threshold | 0.003m | Conservative separator |

#### Verification Logic (`Gripper.has_object()`)
```python
# Object held if:
# 1) Average position > threshold (both fingers stopped), OR
# 2) Max position > threshold * 1.5 (at least one finger stopped)
return avg_pos > threshold or max_pos > threshold * 1.5
```

### 8. Reactive Obstacle Avoidance (TO_BOX Mode)

A* handles global path planning, but reactive avoidance handles unexpected obstacles:

| Threshold | Action | Purpose |
|-----------|--------|---------|
| < 0.30m (Emergency) | Reverse + strong rotation + strafe | Immediate collision avoidance |
| < 0.50m (Preventive) | Slow advance + rotation + strafe | Maintain clearance |
| > 0.50m (Normal) | Follow A* waypoints | Standard navigation |

#### Stuck Detection
If emergency mode persists for >25 cycles, aggressive escape maneuver:
```python
vx_cmd = -0.10      # Strong reverse
omega_cmd = ±0.6    # Very strong rotation
vy_cmd = ±0.18      # Maximum strafe
```

---

## Key Parameters Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grasp trigger distance | 0.32m | Safe approach distance |
| Forward buffer | cam_dist - 0.12 | Leave ~12cm for gripper reach |
| Gripper threshold | 0.003m | Distinguish empty vs holding |
| Obstacle inflation | 0.30m | Robot half-width + safety |
| A* cell size | 0.12m | Resolution vs computation |
| Emergency threshold | 0.30m | Last-resort collision avoidance |
| Preventive threshold | 0.50m | Comfortable clearance |
| GT sync (TO_BOX) | 0.5s | High precision for deposit |

---

## Hardware Configuration

```
YouBot Layout:
                    ┌─────────┐
                    │  ARM    │ (5-DOF)
    ┌───────────────┼─────────┼───────────────┐
    │               │ GRIPPER │               │
    │   ┌───────────┴─────────┴───────────┐   │
    │   │         CAMERA (+0.27m)         │   │
    │   │         LIDAR  (+0.28m)         │   │
    │   └─────────────────────────────────┘   │
    │  [W1]                           [W2]    │
    │                                         │
    │                 BASE                    │
    │            (~0.58m x 0.38m)             │
    │  [W3]                           [W4]    │
    └─────────────────────────────────────────┘
          Mecanum Wheels (omnidirectional)
```

---

## Arena Layout

### Box Positions (Pre-mapped)

| Color | Position (x, y) | Orientation |
|-------|-----------------|-------------|
| Green | (0.48, 1.58) | Standard |
| Blue | (0.48, -1.62) | Standard |
| Red | (2.31, 0.01) | Rotated 90° |

### Obstacles (WoodenBoxes)

| Name | Position (x, y) | Size |
|------|-----------------|------|
| A | (0.6, 0.0) | 0.3m x 0.3m |
| B | (1.96, -1.24) | 0.3m x 0.3m |
| C | (1.95, 1.25) | 0.3m x 0.3m |
| D | (-2.28, 1.5) | 0.3m x 0.3m |
| E | (-1.02, 0.75) | 0.3m x 0.3m |
| F | (-1.02, -0.74) | 0.3m x 0.3m |
| G | (-2.27, -1.51) | 0.3m x 0.3m |

---

## Dependencies

- **Webots R2023b+**: Simulation environment
- **Python 3.x**: Controller language
- **ONNX Runtime**: Neural network inference
- **NumPy**: Numerical operations

---

## Theoretical References

1. **A* Algorithm**: Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
2. **Fuzzy Logic**: Zadeh, L. A. (1965). "Fuzzy Sets" - Information and Control
3. **MobileNetV3**: Howard, A., et al. (2019). "Searching for MobileNetV3"
4. **Mecanum Kinematics**: Omnidirectional mobile robot inverse kinematics
5. **Occupancy Grid Mapping**: Elfes, A. (1989). "Using occupancy grids for mobile robot perception and navigation"
6. **Bresenham Line Algorithm**: Rasterization for raycasting

---

## Technical Documentation Links

- Webots Documentation: https://cyberbotics.com/doc/guide/index
- Webots LIDAR Reference: https://cyberbotics.com/doc/reference/lidar
- OpenCV (color processing): https://docs.opencv.org/4.x/
- DeepWiki Webots: https://deepwiki.com/cyberbotics/webots
