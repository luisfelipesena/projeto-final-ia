# YouBot Cube Collection - Technical Summary

## Project Overview
Autonomous robot (KUKA YouBot) that collects 15 colored cubes (red, green, blue) and deposits them in corresponding colored boxes using LIDAR, RGB camera, neural networks, and fuzzy logic control.

---

## Technical Decisions

### 1. Perception System

#### LIDAR (180° FOV, 5m range)
- **Purpose**: Obstacle detection and environment mapping
- **Implementation**: Processes point cloud to detect obstacles at front/left/right
- **Grid Mapping**: Occupancy grid (12cm cells) updated via raycasting

#### RGB Camera + Recognition API
- **Purpose**: Cube detection and color classification
- **Webots Recognition API**: Returns object position relative to camera
- **Color Classification**: Neural network (primary) + HSV heuristics (fallback)

### 2. Neural Network Color Classifier

#### Architecture
- **Backbone**: MobileNetV3-Small (pre-trained ImageNet)
- **Head**: Custom classifier (256 → 3 classes)
- **Input**: 64x64 RGB normalized (ImageNet mean/std)
- **Output**: Softmax probabilities for red/green/blue

#### Training
- **Dataset**: Synthetic cubes with augmentation (brightness, noise, blur)
- **Strategy**: Two-phase training
  1. Freeze backbone, train head only
  2. Fine-tune entire network with lower LR
- **Result**: 99.4% validation accuracy

#### Inference
- **Format**: ONNX for cross-platform compatibility
- **Fallback**: HSV heuristics when confidence < 0.5

### 3. Fuzzy Logic Navigation

#### Membership Functions
- **Distance**: near/medium/far (triangular functions)
- **Angle**: small/medium/big
- **Obstacle proximity**: close/safe

#### Rules (simplified)
```
IF target_far AND angle_small THEN move_forward_fast
IF target_near AND obstacle_close THEN strafe_away
IF angle_big THEN rotate_only
```

#### Output
- Forward velocity (vx): 0-0.18 m/s
- Lateral velocity (vy): 0-0.14 m/s
- Angular velocity (omega): 0-0.5 rad/s

### 4. Localization (Odometry)

#### Mecanum Wheel Kinematics
```
vx = (w1 + w2 + w3 + w4) * R/4
vy = (-w1 + w2 + w3 - w4) * R/4
omega = (-w1 + w2 - w3 + w4) * R/(4*L)
```
Where R = wheel radius, L = wheel base

#### Pose Integration
```python
dx = cos(yaw)*vx - sin(yaw)*vy
dy = sin(yaw)*vx + cos(yaw)*vy
pose += [dx*dt, dy*dt, omega*dt]
```

### 5. State Machine

```
┌─────────┐    detect cube    ┌──────────┐
│  SEARCH │ ─────────────────▶│ APPROACH │
└────┬────┘                   └────┬─────┘
     │                              │
     │                         dist < 0.32m
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

### 6. Grasp Verification

#### Position Sensor Method
- Gripper fingers have position sensors
- When empty: fingers close to 0.0
- When holding object: fingers stop at ~0.003-0.015
- Multiple samples averaged for reliability

---

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grasp trigger distance | 0.32m | Safe approach distance |
| Forward movement | cam_dist - 0.12 | Leave ~12cm for gripper reach |
| Gripper threshold | 0.003 | Distinguish empty vs holding |
| Approach speed | 0.08 m/s | Slow for precision |
| Navigation speed | 0.12 m/s | Balanced speed/safety |

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
    │                                         │
    │  [W3]                           [W4]    │
    └─────────────────────────────────────────┘
          Mecanum Wheels (omnidirectional)
```

---

## Box Positions (Pre-mapped)

| Color | Position (x, y) |
|-------|-----------------|
| Green | (0.48, 1.58)    |
| Blue  | (0.48, -1.62)   |
| Red   | (2.31, 0.01)    |

---

## Dependencies

- **Webots R2023b+**: Simulation environment
- **Python 3.x**: Controller language
- **ONNX Runtime**: Neural network inference
- **NumPy**: Numerical operations

---

## References

- Webots Documentation: https://cyberbotics.com/doc/guide/index
- Mecanum Kinematics: Odometry equations for omnidirectional wheels
- MobileNetV3: Howard et al., "Searching for MobileNetV3" (2019)
- Fuzzy Logic: Zadeh, "Fuzzy Sets" (1965)
