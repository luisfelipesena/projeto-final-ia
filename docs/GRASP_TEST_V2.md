# YouBot Grasp Test V2 - Cube Detection + Grasp Integration

## Overview
This document describes the complete cube detection and grasping system validated through extensive testing.

**Test Result: SUCCESS**
- Cube detected: GREEN
- Final finger position: 0.0032
- Threshold: 0.002
- `has_object(): True`

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  CubeDetector   │───▶│  State Machine   │───▶│  Grasp Sequence │
│  (HSV + Size)   │    │  (SCAN/APPROACH) │    │  (Validated)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## State Machine

### States
```
SCANNING ──▶ APPROACHING ──▶ GRASPING ──▶ SUCCESS/FAILED
    ▲              │
    └──────────────┘ (if target lost)
```

### Transitions
| From | To | Condition |
|------|-----|-----------|
| SCANNING | APPROACHING | Cube detected (any color) |
| APPROACHING | GRASPING | Size >= 22px AND angle <= 8° |
| APPROACHING | SCANNING | Target lost for > 1s |
| GRASPING | SUCCESS | has_object() == True |
| GRASPING | FAILED | has_object() == False |

## Cube Detection

### CubeDetector Configuration
```python
# HSV Color Ranges (Webots calibrated)
HSV_RANGES = {
    'red': [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255)),
    ],
    'green': [((35, 100, 100), (85, 255, 255))],
    'blue': [((100, 100, 100), (130, 255, 255))],
}
```

### Size Filtering (Critical)
```python
MIN_CUBE_SIZE = 5    # Filter noise
MAX_CUBE_SIZE = 40   # Filter deposit boxes (they appear huge)
```

**Why this matters**: Deposit boxes are colored (green/blue/red) and appear as large colored rectangles. Without filtering, the robot would target them instead of cubes.

### Distance Estimation
Camera-based distance estimation is **unreliable**. Use pixel size instead:

| Cube Size (px) | Approximate Distance |
|----------------|---------------------|
| 7 px | ~50 cm |
| 14 px | ~25 cm |
| 20 px | ~18 cm |
| 25 px | ~14 cm (grasp ready) |

### Angle Calculation
```python
degrees_per_pixel = FOV_DEGREES / image_width  # 57° / 128px ≈ 0.445°/px
angle = (center_x - image_center) * degrees_per_pixel
# Positive angle = cube to the right
```

## Approach Logic

### Key Insight
**Always move forward** while rotating. Pure rotation causes the robot to spin without progress.

### Movement Rules
```python
if abs(angle) > 20:
    # Large angle: slow forward + strong rotation
    omega = -0.35 if angle > 0 else 0.35
    base.move(0.02, 0, omega)

elif abs(angle) > 10:
    # Medium angle: medium forward + medium rotation
    omega = -0.25 if angle > 0 else 0.25
    base.move(0.04, 0, omega)

else:
    # Small angle: fast forward + light rotation
    omega = -0.10 if angle > 0 else 0.10
    if abs(angle) < 3:
        omega = 0  # Don't oscillate
    base.move(0.06, 0, omega)
```

### Grasp Ready Criteria
Both conditions must be met:
```python
GRASP_READY_SIZE = 22   # Cube appears large enough
GRASP_READY_ANGLE = 8.0 # Cube is centered
```

## Grasp Sequence (Validated)

### Complete Sequence
```
Step 1: Open gripper           (1.0s)
Step 2: Reset arm              (1.5s)
Step 3: Lower to FRONT_FLOOR   (2.5s)
Step 4: Forward 10cm @ 5cm/s   (2.0s)
Step 5: Close gripper          (1.5s)
Step 6: Check has_object()
Step 7: Lift to FRONT_PLATE    (2.0s)
────────────────────────────────────
Total:                         ~11s
```

### Critical Parameters
```python
# Movement
FINAL_APPROACH_SPEED = 0.05  # m/s (slow for precision)
FINAL_APPROACH_TIME = 2.0    # seconds (= 10cm)

# Gripper
OBJECT_THRESHOLD = 0.002     # finger_pos > threshold = object held

# Arm Presets
FRONT_FLOOR = {
    'arm2': -0.97,
    'arm3': -1.55,
    'arm4': -0.61,
}
```

### Object Detection
```python
def has_object():
    finger_pos = finger_sensor.getValue()
    return finger_pos > 0.002

# When holding 3cm cube:
# - finger_pos ≈ 0.0025 - 0.0035
# When empty:
# - finger_pos ≈ 0.0000
```

## Hardware Configuration

### Gripper (Dual Finger)
```python
# BOTH fingers must be controlled
finger_left = robot.getDevice("finger::left")
finger_right = robot.getDevice("finger::right")

def grip():
    finger_left.setPosition(0.0)   # MIN_POS
    finger_right.setPosition(0.0)

def release():
    finger_left.setPosition(0.025) # MAX_POS
    finger_right.setPosition(0.025)
```

### Sensor Name
```python
# Correct: finger::leftsensor
# Wrong:   finger::left:sensor
finger_sensor = robot.getDevice("finger::leftsensor")
```

## Test Configuration

### World Setup
```
Robot position: (-1.5, 0.0, 0.102)
Test cube:      (-1.0, 0.0, 0.016)  # 50cm in front
Distance:       50cm initial
```

### Successful Test Log
```
[SCAN] *** CUBE DETECTED ***
       Color: GREEN
       Distance: 0.14m
       Angle: 0.0°
[APPROACH] Ready to grasp! Size: 25px, Angle: 0.0°
[GRASP 5] Closing gripper...
         Finger BEFORE: 0.0250
         Finger AFTER: 0.0025
[GRASP 6] Checking object...
         has_object(): True
GRASP TEST V2: *** SUCCESS ***
```

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Detects deposit box | No size filter | Add MAX_CUBE_SIZE = 40 |
| Robot spins in place | Pure rotation | Always include forward velocity |
| Angle stuck at ~27° | Lost tracking | Increase lost_frames timeout |
| Finger reads 0.0 | Cube missed | Check cube size/angle before grasp |
| Loop APPROACH→SCAN | Bad transition | Require both size AND angle criteria |

## Files Modified

### Controllers
- `IA_20252/controllers/youbot_grasp_test_v2/youbot_grasp_test_v2.py` - Main test
- `IA_20252/controllers/youbot_grasp_test/gripper.py` - Dual finger control

### Source
- `src/actuators/gripper_controller.py` - Dual finger + correct sensor
- `src/utils/config.py` - GRIP_THRESHOLD = 0.002

### Documentation
- `docs/GRASP_TEST.md` - Grasp mechanics
- `docs/GRASP_TEST_V2.md` - This file

## Next Steps

1. **Multi-cube test**: Place multiple cubes, verify robot collects nearest
2. **Deposit integration**: After grasp, navigate to correct color box
3. **Full autonomous**: 15 cubes collection with obstacle avoidance
4. **Fuzzy + RNA**: Integrate MATA64 requirements for navigation

## References

- GRASP_TEST.md - Detailed grasp mechanics
- CubeDetector - src/perception/cube_detector.py
- Arm presets - IA_20252/controllers/youbot_grasp_test/arm.py
