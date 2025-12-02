# YouBot Grasp Test - Complete Technical Documentation

## Overview
This document contains all mathematical and practical knowledge acquired through extensive testing of the YouBot grasp system in Webots.

## Robot Geometry

### YouBot Dimensions (Critical)
```
                    FRONT (positive X direction)
                           ↑
    +------------------+   |
    |   [Camera/LIDAR] |   |  Camera offset: +0.27m from center
    |        ↓         |   |
    |   [Robot Center] |   |  Reference point (0,0)
    |        ↓         |   |
    |    [Arm Base]    |   |  Arm base offset: ~-0.10m from center
    +------------------+   |
                           ↓
                    BACK (negative X direction)
```

### Key Offsets from Robot Center
| Component | X Offset | Y Offset | Notes |
|-----------|----------|----------|-------|
| Camera | +0.27m | 0 | Front-mounted |
| LIDAR | +0.28m | 0 | Front-mounted |
| Arm Base | ~-0.10m | 0 | Slightly behind center |
| Gripper (FRONT_FLOOR) | ~+0.25m | 0 | When arm extended |

### Wheel Geometry
- **Wheel Radius**: 0.05m
- **LX (longitudinal)**: 0.228m
- **LY (lateral)**: 0.158m
- **Max Speed**: 0.3 m/s

## Arm Configuration

### Joint Limits (CRITICAL - Never Exceed)
```python
ARM_LIMITS = {
    'arm1': (-2.949, 2.949),    # Base rotation (±169°)
    'arm2': (-1.13, 1.57),      # Shoulder - MOST RESTRICTIVE
    'arm3': (-2.635, 2.548),    # Elbow
    'arm4': (-1.78, 1.78),      # Wrist pitch
    'arm5': (-2.949, 2.949),    # Wrist roll
}
```

### Arm Segment Lengths
```python
ARM_LENGTHS = {
    'arm1': 0.253,  # Base height
    'arm2': 0.155,  # Upper arm
    'arm3': 0.135,  # Forearm
    'arm4': 0.081,  # Wrist
    'arm5': 0.105,  # Gripper mount
}
# Total reach: ~0.729m (theoretical max)
```

### Preset Positions (Tested & Validated)
```python
ARM_PRESETS = {
    'FRONT_FLOOR': {
        'arm1': 0.0,      # Facing forward
        'arm2': -0.97,    # Shoulder down
        'arm3': -1.55,    # Elbow bent
        'arm4': -0.61,    # Wrist angled
        'arm5': 0.0,      # No roll
        'reach': 0.25,    # ~25cm forward from robot center
        'height': 0.016,  # Ground level (cube height)
    },
    'FRONT_PLATE': {
        'arm1': 0.0,
        'arm2': -0.62,
        'arm3': -0.98,
        'arm4': -1.53,
        'arm5': 0.0,
        'reach': 0.20,    # ~20cm forward
        'height': 0.10,   # Plate height
    },
    'RESET': {
        'arm1': 0.0,
        'arm2': 1.57,
        'arm3': -2.635,
        'arm4': 1.78,
        'arm5': 0.0,
        'description': 'Arm folded back, safe transport position'
    },
}
```

## Gripper Configuration

### Physical Parameters
```python
GRIPPER = {
    'MIN_POS': 0.0,       # Fully closed
    'MAX_POS': 0.025,     # Fully open (25mm)
    'OFFSET_WHEN_LOCKED': 0.021,
    'VELOCITY': 0.03,     # rad/s for smooth operation
}
```

### Object Detection
```python
OBJECT_DETECTION = {
    'threshold': 0.002,   # Finger position > threshold = object held
    'cube_size': 0.03,    # 3cm cube
    'expected_finger_with_cube': 0.002 - 0.005,  # Range when holding cube
}
```

### Dual Finger Control (IMPORTANT)
The YouBot gripper has TWO independent finger motors:
- `finger::left`
- `finger::right`

**Both must be controlled simultaneously** for proper grasping:
```python
def grip(self):
    self.finger_left.setPosition(MIN_POS)
    self.finger_right.setPosition(MIN_POS)

def release(self):
    self.finger_left.setPosition(MAX_POS)
    self.finger_right.setPosition(MAX_POS)
```

## Grasp Sequence (Validated)

### Complete Sequence
```
1. OPEN GRIPPER     → release() + wait 1.0s
2. RESET ARM        → set_height(RESET) + wait 1.5s
3. OPEN GRIPPER     → release() again (ensure open)
4. LOWER ARM        → set_height(FRONT_FLOOR) + wait 2.5s
5. MOVE FORWARD     → move(0.05, 0, 0) for 2.0s = 10cm
6. CLOSE GRIPPER    → grip() + wait 1.5s
7. VERIFY           → check finger_position > threshold
8. LIFT             → set_height(FRONT_PLATE) + wait 2.0s
```

### Timing Summary
| Step | Duration | Cumulative |
|------|----------|------------|
| Open gripper | 1.0s | 1.0s |
| Reset arm | 1.5s | 2.5s |
| Lower arm | 2.5s | 5.0s |
| Move forward | 2.0s | 7.0s |
| Close gripper | 1.5s | 8.5s |
| Lift | 2.0s | 10.5s |
| **Total** | **~11s** | |

## Movement Calculations

### Forward Movement Formula
```python
# Mecanum wheel kinematics
def calculate_forward_time(distance_m, speed_m_s):
    return distance_m / speed_m_s

# Example: 10cm at 5cm/s
time = 0.10 / 0.05  # = 2.0 seconds
```

### Tested Configuration
```python
APPROACH = {
    'speed': 0.05,        # m/s (slow for precision)
    'distance': 0.10,     # 10cm
    'time': 2.0,          # seconds
    'lateral': 0,         # no sideways movement
    'rotation': 0,        # no turning
}
```

### Why Slow Speed?
- Fast movement (`forwards()` at SPEED=4.0 rad/s) causes:
  - Physics instability warnings
  - Robot drift/rotation
  - Imprecise positioning
- Slow movement (`move(0.05, 0, 0)`) provides:
  - Stable physics simulation
  - Straight-line movement
  - Precise cube alignment

## Initial Positioning

### Robot Starting Position
```python
ROBOT_START = {
    'x': -1.5,
    'y': 0.0,
    'z': 0.102,  # Height above ground
    'rotation': 0,  # Facing +X direction
}
```

### Test Cube Position (for grasp test)
```python
TEST_CUBE = {
    'x': -1.0,    # 50cm in front of robot
    'y': 0.0,     # Centered
    'z': 0.016,   # Half cube height (cube resting on ground)
    'size': 0.03, # 3cm cube
}
```

### Distance Calculation
```
Initial distance: robot_x - cube_x = -1.5 - (-1.0) = 0.5m = 50cm
Arm reach (FRONT_FLOOR): ~25cm from robot center
Required forward movement: 50 - 25 - 15 (safety) = 10cm
```

## Cube Specifications

### Physical Properties
```python
CUBE = {
    'size': (0.03, 0.03, 0.03),  # 3cm x 3cm x 3cm
    'mass': 0.03,                 # 30 grams
    'colors': ['red', 'green', 'blue'],
}
```

### HSV Color Ranges (for detection)
```python
HSV_RANGES = {
    'red': [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255)),
    ],
    'green': [((35, 100, 100), (85, 255, 255))],
    'blue': [((100, 100, 100), (130, 255, 255))],
}
```

## Box Deposit Locations

```python
BOX_POSITIONS = {
    'GREEN': {'x': 0.48, 'y': 1.58},
    'BLUE': {'x': 0.48, 'y': -1.62},
    'RED': {'x': 2.31, 'y': 0.01},
}
```

## Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Gripper closes empty | Cube not in position | Increase forward distance |
| Robot drifts sideways | High speed movement | Use slow `move(0.05, 0, 0)` |
| Arm doesn't reach cube | Wrong preset | Use FRONT_FLOOR preset |
| Physics warnings | Fast movements | Reduce speed, increase time |
| Finger reads 0.0 | Gripper missed cube | Adjust approach distance |
| Finger reads 0.025 | Gripper didn't close | Check both finger motors |

### Validation Checklist
- [ ] Both finger motors controlled (left AND right)
- [ ] Gripper opens BEFORE lowering arm
- [ ] Arm reaches FRONT_FLOOR before moving forward
- [ ] Forward movement is slow (≤0.05 m/s)
- [ ] Gripper closes AFTER cube is in position
- [ ] Object detection threshold is appropriate (0.002)

## Code References

### Controller Files
```
IA_20252/controllers/youbot_grasp_test/
├── youbot_grasp_test.py  # Main test controller
├── arm.py                # Arm control with presets
├── base.py               # Mecanum wheel kinematics
└── gripper.py            # Dual-finger gripper control
```

### Key Functions
- `Arm.set_height(preset)` - Set arm to validated preset
- `Gripper.grip()` / `Gripper.release()` - Control both fingers
- `Base.move(vx, vy, omega)` - Omnidirectional movement
- `has_object()` - Check if cube is held (finger_pos > threshold)

## References

- Zadeh (1965): Fuzzy Sets - for navigation decisions
- Mamdani & Assilian (1975): Fuzzy Logic Controller
- Habermann et al. (2013): MLP for LIDAR classification
- Saffiotti (1997): Fuzzy Navigation for mobile robots