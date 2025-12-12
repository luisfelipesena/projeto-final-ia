
# Solving YouBot navigation and manipulation challenges in Webots

The core solutions to your KUKA YouBot problems involve **enforcing non-holonomic constraints on the mecanum platform** and implementing a **two-phase coarse-to-fine approach strategy** for manipulation. By setting lateral velocity Vy = 0 and constraining angular velocity ωz ≤ |Vx|/R_min, the robot will exhibit car-like behavior without spinning. For precise cube deposits, switch from LIDAR-based navigation to camera visual servoing when within 0.5m of the target box, and verify alignment before releasing the gripper.

---

## Mathematical foundation for car-like mecanum constraints

Your YouBot's spinning behavior stems from the unconstrained holonomic capabilities of mecanum wheels. The standard inverse kinematics for your parameters (WHEEL_RADIUS = 0.05m, LX = 0.228m, LY = 0.158m) maps body velocities to wheel speeds:

```
ω_fl = (1/r) × [Vx - Vy - (LX+LY)×ωz]  = 20×[Vx - Vy - 0.386×ωz]
ω_fr = (1/r) × [Vx + Vy + (LX+LY)×ωz]  = 20×[Vx + Vy + 0.386×ωz]
ω_rl = (1/r) × [Vx + Vy - (LX+LY)×ωz]  = 20×[Vx + Vy - 0.386×ωz]
ω_rr = (1/r) × [Vx - Vy + (LX+LY)×ωz]  = 20×[Vx - Vy + 0.386×ωz]
```

To enforce car-like behavior, apply two **Pfaffian velocity constraints**. First, eliminate lateral movement by setting **Vy = 0** always—this is the core non-holonomic "no sideslip" constraint that makes cars unable to move sideways. Second, constrain the angular velocity to enforce a minimum turning radius using **|ωz| ≤ |Vx|/R_min**. For your YouBot with effective wheelbase L = 0.456m and a practical steering angle limit of 30°, the minimum turning radius is R_min = L/tan(30°) ≈ **0.79 meters**.

The curvature κ = ωz/Vx must satisfy κ_max = 1.27 m⁻¹, meaning the robot cannot make turns tighter than 0.79m radius while moving forward. Implement this constraint layer between your path planner output and wheel velocity commands:

```python
def apply_car_like_constraints(Vx_cmd, Vy_cmd, omega_z_cmd, R_min=0.79):
    """Transform omnidirectional commands to car-like constrained commands."""
    Vy_constrained = 0.0  # Core non-holonomic constraint
    Vx_constrained = Vx_cmd
    
    kappa_max = 1.0 / R_min  # Maximum curvature = 1.27 m⁻¹
    
    if abs(Vx_constrained) > 0.01:
        kappa_cmd = omega_z_cmd / Vx_constrained
        kappa_constrained = max(-kappa_max, min(kappa_max, kappa_cmd))
        omega_z_constrained = kappa_constrained * Vx_constrained
    else:
        omega_z_constrained = 0.0  # No rotation when nearly stationary
    
    return Vx_constrained, Vy_constrained, omega_z_constrained
```

---

## Modifying fuzzy navigation rules to prevent 360° spins

Your FuzzyNavigator allows unrestricted omnidirectional commands, which causes the spinning behavior. Three specific modifications will constrain it to forward-oriented movement. First, **narrow the angular velocity output universe** from [-180°/s, 180°/s] to [-60°/s, 60°/s]—this prevents full rotation commands from ever being issued. Second, **add a forward motion coupling rule**: whenever heading error is non-zero, require positive forward velocity. Third, **suppress lateral velocity output** by adding a rule that zeros Vy regardless of other inputs.

The modified fuzzy rule base should include these critical rules:

```
# Prevent spin-in-place: If goal is behind, turn WHILE moving forward
IF heading_error IS NEGATIVE_BIG AND forward_velocity IS LOW 
THEN angular_vel IS TURN_LEFT_HARD AND linear_vel IS SLOW_FORWARD

IF heading_error IS POSITIVE_BIG AND forward_velocity IS LOW
THEN angular_vel IS TURN_RIGHT_HARD AND linear_vel IS SLOW_FORWARD

# Never allow pure rotation without forward motion
IF linear_velocity IS ZERO AND heading_error IS NOT ZERO
THEN linear_vel IS SLOW_FORWARD

# Reduce turn rate as speed increases (stability)
IF forward_velocity IS HIGH AND heading_error IS MEDIUM
THEN angular_vel IS TURN_SOFT  # Gentler turn at high speed
```

Modify your membership functions for angular velocity output to use narrower ranges: TURN_LEFT_HARD as [-60, -60, -30], TURN_LEFT_SOFT as [-40, -20, 0], STRAIGHT as [-10, 0, 10], TURN_RIGHT_SOFT as [0, 20, 40], and TURN_RIGHT_HARD as [30, 60, 60]. The key constraint wrapper should enforce **minimum forward velocity during any turn**:

```python
def constrained_fuzzy_output(heading_error, obstacles, current_vel):
    raw_angular = fuzzy_inference(heading_error, obstacles)
    raw_linear = compute_linear_vel(obstacles)
    
    # Prevent pure rotation (anti-spin constraint)
    MIN_FORWARD_DURING_TURN = 0.05  # m/s
    if abs(raw_linear) < 0.01 and abs(raw_angular) > 0.1:
        raw_linear = MIN_FORWARD_DURING_TURN * sign(cos(heading_error))
    
    # Limit angular velocity based on forward speed
    max_turn_rate = 1.0 * (1 - abs(raw_linear) / MAX_LINEAR_VEL)
    raw_angular = clip(raw_angular, -max_turn_rate, max_turn_rate)
    
    return (raw_linear, 0.0, raw_angular)  # Vy always zero
```

---

## Rear collision prevention without lateral trapping

The lateral trapping problem occurs because your current avoidance triggers lateral movement when rear sensors detect obstacles—but lateral movement into the obstacle's side is geometrically impossible to escape. The solution uses **graduated rear awareness zones** combined with **forward-plus-turn escape** rather than lateral escape.

Define three rear zones: CRITICAL (< 0.15m), WARNING (< 0.30m), and NOTICE (< 0.50m). When the rear is threatened but the front is clear, the response should always be **move forward while turning**, never lateral:

```python
def compute_rear_response(rear_distances, front_clear, heading_error):
    min_rear = min(rear_distances.values())
    
    # Critical: rear blocked, front clear → escape forward
    if min_rear < 0.15 and front_clear:
        return {'linear_vel': 0.2, 'angular_vel': 0.0, 'lateral_vel': 0.0}
    
    # Warning: asymmetric detection → turn toward clearer side
    if min_rear < 0.30:
        if rear_distances['left'] > rear_distances['right']:
            return {'linear_vel': 0.15, 'angular_vel': 0.3, 'lateral_vel': 0.0}
        else:
            return {'linear_vel': 0.15, 'angular_vel': -0.3, 'lateral_vel': 0.0}
    
    return {'action': 'NORMAL'}
```

For stuck detection and escape, monitor position history over 20 samples. If movement is less than 10cm over this window, the robot is stuck. The escape sequence should **identify the most open direction from LIDAR** and execute forward-biased escape—preferring forward, then forward-plus-turn, with reverse only as last resort:

```python
def execute_escape(lidar_scan, ir_sensors):
    front_clear = scan_sector(lidar_scan, -30, 30)
    left_clear = scan_sector(lidar_scan, 30, 90)
    right_clear = scan_sector(lidar_scan, -90, -30)
    rear_clear = min(ir_sensors['rear'])
    
    # Sort by clearance with forward bias (1.2x multiplier)
    clearances = [
        ('forward', front_clear * 1.2, (0.2, 0, 0)),
        ('left', left_clear, (0.1, 0, 0.5)),
        ('right', right_clear, (0.1, 0, -0.5)),
        ('reverse', rear_clear, (-0.1, 0, 0))
    ]
    best = max(clearances, key=lambda x: x[1])
    return best[2]  # Return (vx, vy, omega)
```

---

## Vector Field Histogram for reactive obstacle avoidance

VFH provides superior reactive navigation compared to pure fuzzy control for your cluttered arena. The algorithm reduces 2D Cartesian obstacle data to a 1D polar histogram, then selects steering directions through valleys in that histogram. For your 180° LIDAR with 0.1-5m range, use **α = 5° angular resolution** (36 sectors covering the forward hemisphere).

Build the polar histogram by iterating through LIDAR readings, computing obstacle magnitudes inversely proportional to distance, and accumulating into angular sectors:

```python
def build_polar_histogram(lidar_scan, alpha=5):
    n = 180 // alpha  # 36 sectors for 180° LIDAR
    polar_hist = [0] * n
    
    for i, distance in enumerate(lidar_scan):
        if distance < 3.0:  # Within relevant range
            angle = i * lidar_angular_resolution
            sector = int(angle / alpha)
            # Magnitude inversely proportional to distance
            magnitude = (5.0 - distance) ** 2 / 25.0
            polar_hist[sector] += magnitude
    
    # Smooth the histogram
    smoothed = moving_average(polar_hist, window=3)
    return smoothed
```

For direction selection, find **valleys** (contiguous sectors below threshold) and choose the valley closest to the target direction. If the valley is wide (> 90°), steer toward the edge nearest the target rather than the center:

```python
def select_steering_direction(polar_hist, target_direction, threshold=0.5):
    valleys = find_valleys_below_threshold(polar_hist, threshold)
    if not valleys:
        return None  # Trapped
    
    best_valley = min(valleys, key=lambda v: angular_distance(v.center, target_direction))
    
    if best_valley.width > 90:
        # Wide valley: choose edge closest to target
        if angular_distance(best_valley.start, target_direction) < \
           angular_distance(best_valley.end, target_direction):
            return best_valley.start + 22.5  # Offset from edge
        else:
            return best_valley.end - 22.5
    else:
        return best_valley.center
```

---

## Two-phase approach strategy for precise cube deposit

Your deposit misalignment stems from using coarse navigation all the way to the drop position. The solution is a **coarse-to-fine paradigm**: use LIDAR/odometry navigation to approach within 0.5m, then switch to camera-based visual servoing for the final **50cm**.

**Phase 1 (Coarse)**: Navigate using your existing A* pathfinding and fuzzy control until LIDAR detects the box edge and distance falls below 0.5m. This gets the robot "in the neighborhood" without precision requirements.

**Phase 2 (Fine)**: Switch to a closed-loop visual servoing controller. Use the RGB camera to detect the target box by color segmentation, compute the centroid, and servoing to align the gripper projection with the box center:

```python
def fine_alignment_loop(box_color, ALIGNMENT_THRESHOLD=0.005):
    alignment_attempts = 0
    MAX_ATTEMPTS = 5
    
    while alignment_attempts < MAX_ATTEMPTS:
        image = capture_image()
        box_detection = detect_box_by_color(image, box_color)
        
        if not box_detection.valid:
            alignment_attempts += 1
            adjust_camera_view()
            continue
        
        # Compute error between gripper projection and box center
        gripper_xy = project_gripper_to_image()
        error = box_detection.center - gripper_xy
        error_meters = pixels_to_meters(error, estimated_depth)
        
        if magnitude(error_meters) < ALIGNMENT_THRESHOLD:
            # Verify stability over 5 cycles
            if verify_stable_alignment(5, ALIGNMENT_THRESHOLD):
                return SUCCESS
        
        # Apply proportional correction
        K_p = 0.3  # Lower gain for precision
        correction = K_p * error_meters
        move_arm_relative(correction)
        alignment_attempts += 1
    
    return ALIGNMENT_FAILED
```

---

## Verifying cube position before and after release

The repeated pick-drop loop occurs because the FSM transitions to "search" after release without verifying success. Implement **three verification checkpoints**: pre-release alignment, height verification, and post-release detection.

**Pre-release verification** ensures the gripper is properly over the box opening:

```python
def verify_gripper_over_box():
    image = capture_image()
    box_rect = detect_box_opening(image, target_color)
    cube_footprint = estimate_cube_footprint_in_image()
    
    alignment_ok = box_rect.contains(cube_footprint.center)
    fits_ok = box_rect.contains_all_corners(cube_footprint)
    
    return alignment_ok and fits_ok
```

**Height verification** uses your IR sensors to confirm the gripper has descended into the box volume—the front IR sensors should detect the box walls when the cube is properly positioned inside:

```python
def verify_depth_position(expected_box_depth=0.08):
    gripper_height = get_gripper_height()
    ir_front = read_ir_sensors()['front']
    
    height_ok = gripper_height < expected_box_depth + 0.02
    walls_detected = any(d < 0.15 for d in ir_front)
    
    return height_ok and walls_detected
```

**Post-release verification** captures an image after releasing and raising the gripper, then checks if the cube color appears within the box region:

```python
def verify_cube_in_box(box_color, cube_color):
    image = capture_image()
    box_mask = color_segment(image, box_color)
    box_roi = bounding_rect(box_mask)
    
    box_interior = crop(image, box_roi)
    cube_mask = color_segment(box_interior, cube_color)
    
    cube_detected = pixel_count(cube_mask) > MIN_CUBE_AREA
    cube_centered = distance(centroid(cube_mask), box_roi.center) < MAX_OFFSET
    
    return cube_detected and cube_centered
```

---

## Enhanced FSM with recovery states

Modify your FSM to include explicit verification and recovery states. The critical additions are **FINE_ALIGN** (visual servoing phase), **VERIFY_POSITION** (pre-release check), **VERIFY_DROP** (post-release check), and **RECOVERY** (failure handling):

```
TRANSITIONS:
search → approach:        cube_detected AND color_matched
approach → grasp:         distance < GRASP_RANGE
grasp → to_box:           gripper_closed AND cube_secure
to_box → fine_align:      distance_to_box < 0.5m
fine_align → verify_pos:  visual_error < 5mm for 3 cycles
verify_pos → drop:        all_checks_pass
verify_pos → fine_align:  checks_fail (retry up to 3×)
verify_pos → recovery:    retry_count > 3
drop → verify_drop:       gripper_open AND 200ms timeout
verify_drop → search:     cube_confirmed_in_box
verify_drop → recovery:   cube_not_in_box

RECOVERY STATE:
  IF cube_still_in_gripper:
    retry_count++
    IF retry_count < MAX_RETRIES: GOTO fine_align
    ELSE: GOTO search (abandon cube)
  ELIF cube_dropped_outside:
    move_away_from_box()
    GOTO search (re-detect and re-pick)
```

To break the infinite pick-drop loop, track **per-cube attempt history** using a dictionary keyed by cube ID (from the Recognition API). If a specific cube exceeds 3 failed deposit attempts, add it to a cooldown list and skip it for 30 seconds:

```python
cube_attempt_history = {}
MAX_ATTEMPTS_PER_CUBE = 3
COOLDOWN_TIME = 30

def should_attempt_cube(cube_id):
    if cube_id in cube_attempt_history:
        attempts, last_time = cube_attempt_history[cube_id]
        if attempts >= MAX_ATTEMPTS_PER_CUBE:
            if current_time - last_time < COOLDOWN_TIME:
                return False  # Skip this cube
    return True
```

---

## Reference implementations from GitHub

Several repositories provide working code relevant to your task. The **official Webots YouBot implementation** at `github.com/cyberbotics/webots/tree/master/projects/robots/kuka/youbot` includes `base.c` with wheel velocity control functions demonstrating proper mecanum kinematics. The **Kuka-youBot-Mobile-Manipulation** repository at `github.com/241abhishek/Kuka-youBot-Mobile-Manipulation` implements complete pick-and-place with feedforward + PI feedback control that can serve as a manipulation reference.

For navigation algorithms, **Youbot-RVO** at `github.com/ipab-rad/Youbot-RVO` implements Reciprocal Velocity Obstacles for dynamic obstacle avoidance with human tracking via Kinect—useful for multi-agent scenarios. The **ros-a-star-vfh** repository at `github.com/WhizK1D/ros-a-star-vfh` combines global A* planning with local VFH planning, which matches your current architecture.

For VFH specifically, **vfh-python** at `github.com/vanderbiltrobotics/vfh-python` provides a clean Python implementation, while **vmc_project** at `github.com/oselin/vmc_project` offers ROS-integrated VFH with real-time histogram visualization. The **AGV_Mecanum** repository at `github.com/ffrige/AGV_Mecanum` contains kinematic model implementations that demonstrate the constraint equations derived above.

---

## Recommended tuning parameters for your 7m × 4m arena

Based on your arena dimensions and the mathematical analysis, use these starting parameters:

- **Minimum turning radius**: R_min = 0.5m (allows maneuvering in tight spaces while maintaining car-like behavior)
- **Maximum angular velocity**: 1.0 rad/s (prevents spinning, was likely higher before)
- **Minimum forward velocity during turns**: 0.05 m/s (ensures forward progress)
- **VFH threshold**: 400-600 (tune based on sensor noise; lower = more conservative)
- **Fine alignment threshold**: 5mm (adequate for cube width)
- **Approach phase switch distance**: 0.5m (transition from coarse to fine navigation)
- **Rear critical distance**: 0.15m (emergency zone)
- **Rear warning distance**: 0.30m (asymmetric turn zone)
- **Maximum cube attempts before cooldown**: 3

The constraint factor can be tuned between 0.0 (fully omnidirectional) and 1.0 (fully car-like) using smooth interpolation during the transition period as you test the new behavior.

---

## Conclusion

The fundamental fix for both problems involves treating your holonomic mecanum platform as if it were non-holonomic. For navigation, the mathematical constraint **Vy = 0 AND |ωz| ≤ |Vx|/R_min** eliminates 360° spins and forces car-like arcs that won't trap the robot laterally against obstacles. For manipulation, the two-phase coarse-to-fine approach with explicit visual verification before and after release ensures cubes land inside boxes—and the enhanced FSM with per-cube tracking prevents infinite retry loops.

The VFH algorithm provides more robust reactive avoidance than pure fuzzy control, while the rear-awareness system using forward-plus-turn escape (never lateral) prevents the lateral trapping scenario. These solutions are mathematically grounded in kinematic constraints and follow established patterns from academic literature and working implementations like those in the Webots and ROS YouBot ecosystems.