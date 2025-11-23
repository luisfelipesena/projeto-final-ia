# Feature Specification: Fuzzy Control System for Autonomous Navigation

**Feature ID**: 003-fuzzy-controller  
**Created**: 2025-11-23  
**Status**: Draft  
**Priority**: High  
**Target Phase**: Phase 3 - Control with Fuzzy Logic

---

## Overview

### Problem Statement

The YouBot autonomous system requires an intelligent control mechanism to navigate the arena, avoid obstacles, locate colored cubes, and execute manipulation tasks (grasp and deposit). Traditional control approaches (PID, pure reactive) struggle with the uncertainty and imprecision inherent in sensor data and the complexity of coordinating multiple behaviors (obstacle avoidance, target seeking, manipulation).

### Proposed Solution

Implement a fuzzy logic control system that translates imprecise sensor inputs (distances, angles) into smooth control outputs (velocities, actions) using linguistic rules. The system will coordinate multiple behaviors through a state machine, enabling the robot to autonomously complete the cube collection task.

### Business Value

- **Robustness**: Handles sensor noise and uncertainty gracefully
- **Interpretability**: Rule-based logic is transparent and debuggable
- **Adaptability**: Easy to tune and extend behaviors without rewriting code
- **Academic Compliance**: Satisfies project requirement for fuzzy logic implementation

---

## User Stories

### Primary Users

**User Type**: Autonomous Robot System  
**Context**: Operating in simulated arena environment with obstacles and colored cubes

### Core User Stories

**US-001: Obstacle Avoidance**  
As the robot system, I need to detect obstacles and adjust my path to avoid collisions, so that I can navigate safely through the arena.

**Acceptance Criteria**:
- Robot maintains minimum 0.3m clearance from obstacles
- Robot smoothly adjusts trajectory when obstacles detected
- Robot stops if obstacle is too close (<0.2m)
- Zero collisions during normal operation

---

**US-002: Cube Search and Approach**  
As the robot system, I need to locate colored cubes and navigate toward them, so that I can position myself for grasping.

**Acceptance Criteria**:
- Robot systematically explores arena when no cube is visible
- Robot turns toward cube when detected
- Robot approaches cube while maintaining safe distance from obstacles
- Robot stops at optimal grasping distance (0.15-0.25m from cube)

---

**US-003: Cube Manipulation**  
As the robot system, I need to grasp detected cubes and transport them to the correct deposit box, so that I can complete the collection task.

**Acceptance Criteria**:
- Robot aligns with cube before attempting grasp
- Robot successfully grasps cube >90% of attempts
- Robot navigates to correct color-coded deposit box
- Robot releases cube accurately into deposit box

---

**US-004: Behavior Coordination**  
As the robot system, I need to coordinate multiple behaviors (search, approach, grasp, deliver) in a logical sequence, so that I can complete the full task workflow.

**Acceptance Criteria**:
- Robot transitions between states based on sensor inputs and task progress
- Robot prioritizes obstacle avoidance over all other behaviors
- Robot completes full cycle: search → approach → grasp → deliver → release → search
- Robot handles failure cases (failed grasp, lost cube) by returning to search

---

## Functional Requirements

### FR-001: Fuzzy Input Variables

**Description**: Define linguistic variables for sensor inputs

**Input Variables**:
- `distance_to_obstacle`: Distance to nearest obstacle in meters
  - Linguistic terms: {very_close, close, medium, far}
  - Range: [0.0, 5.0] meters
  
- `angle_to_obstacle`: Angular position of nearest obstacle
  - Linguistic terms: {left, center, right}
  - Range: [-135°, +135°] (LIDAR FOV)
  
- `distance_to_cube`: Distance to detected cube in meters
  - Linguistic terms: {very_close, close, medium, far}
  - Range: [0.0, 3.0] meters
  
- `angle_to_cube`: Angular position of detected cube
  - Linguistic terms: {left_strong, left, center, right, right_strong}
  - Range: [-90°, +90°] (camera FOV)
  
- `cube_detected`: Boolean flag for cube visibility
  - Values: {true, false}
  
- `holding_cube`: Boolean flag for grasp status
  - Values: {true, false}

**Acceptance Criteria**:
- All fuzzy variables have defined membership functions
- Membership functions cover full input range
- Overlapping membership functions ensure smooth transitions

---

### FR-002: Fuzzy Output Variables

**Description**: Define linguistic variables for control outputs

**Output Variables**:
- `linear_velocity`: Forward/backward speed in m/s
  - Linguistic terms: {stop, slow, medium, fast}
  - Range: [-0.3, 0.5] m/s
  
- `angular_velocity`: Rotational speed in rad/s
  - Linguistic terms: {left_strong, left, straight, right, right_strong}
  - Range: [-1.0, 1.0] rad/s
  
- `action`: High-level behavior command
  - Linguistic terms: {search, approach, grasp, navigate_to_box, release}
  - Discrete values

**Acceptance Criteria**:
- Output membership functions produce smooth control signals
- Defuzzification method (centroid) yields continuous outputs
- Action outputs trigger appropriate state transitions

---

### FR-003: Fuzzy Rule Base

**Description**: Define fuzzy rules for behavior control

**Rule Categories**:

1. **Obstacle Avoidance Rules** (Highest Priority):
   - IF distance_to_obstacle IS very_close THEN linear_velocity IS stop AND angular_velocity IS left_strong
   - IF distance_to_obstacle IS close AND angle_to_obstacle IS center THEN linear_velocity IS slow AND angular_velocity IS right
   - IF distance_to_obstacle IS close AND angle_to_obstacle IS left THEN angular_velocity IS right
   - IF distance_to_obstacle IS close AND angle_to_obstacle IS right THEN angular_velocity IS left

2. **Cube Search Rules**:
   - IF cube_detected IS false AND distance_to_obstacle IS far THEN action IS search AND linear_velocity IS medium
   - IF cube_detected IS false AND distance_to_obstacle IS medium THEN action IS search AND linear_velocity IS slow

3. **Cube Approach Rules**:
   - IF cube_detected IS true AND distance_to_cube IS far THEN action IS approach AND linear_velocity IS medium
   - IF cube_detected IS true AND distance_to_cube IS medium AND angle_to_cube IS center THEN linear_velocity IS slow
   - IF cube_detected IS true AND angle_to_cube IS left THEN angular_velocity IS left
   - IF cube_detected IS true AND angle_to_cube IS right THEN angular_velocity IS right
   - IF cube_detected IS true AND distance_to_cube IS very_close THEN linear_velocity IS stop AND action IS grasp

4. **Manipulation Rules**:
   - IF holding_cube IS true THEN action IS navigate_to_box
   - IF holding_cube IS true AND distance_to_obstacle IS close THEN linear_velocity IS slow
   - IF holding_cube IS true AND [at deposit box] THEN action IS release

**Acceptance Criteria**:
- Minimum 15 fuzzy rules defined
- Rules cover all primary scenarios (obstacle avoidance, search, approach, manipulation)
- Rule priorities ensure obstacle avoidance overrides other behaviors
- Rules produce smooth, non-oscillatory control outputs

---

### FR-004: State Machine Integration

**Description**: Coordinate fuzzy controller with high-level state machine

**States**:
1. **SEARCH**: Explore arena to find cubes
2. **APPROACH**: Navigate toward detected cube
3. **ALIGN**: Fine-tune position for grasping
4. **GRASP**: Execute grasp maneuver
5. **NAVIGATE_TO_BOX**: Transport cube to deposit location
6. **RELEASE**: Deposit cube in correct box
7. **RECOVERY**: Handle failure cases

**State Transitions**:
- SEARCH → APPROACH: cube_detected becomes true
- APPROACH → ALIGN: distance_to_cube < 0.3m
- ALIGN → GRASP: robot aligned with cube (angle < 5°)
- GRASP → NAVIGATE_TO_BOX: holding_cube becomes true
- NAVIGATE_TO_BOX → RELEASE: arrived at deposit box
- RELEASE → SEARCH: cube released successfully
- Any state → RECOVERY: failure detected (timeout, lost cube, failed grasp)
- RECOVERY → SEARCH: recovery action completed

**Acceptance Criteria**:
- State machine executes correct transitions based on sensor inputs
- Each state invokes appropriate fuzzy rules
- State transitions are logged for debugging
- Recovery state handles all failure modes

---

### FR-005: Perception Integration

**Description**: Interface with perception system to obtain sensor inputs

**Inputs from Perception**:
- LIDAR obstacle map (9-sector occupancy)
- Nearest obstacle distance and angle
- Detected cube information (color, distance, angle)
- Camera cube detection confidence

**Acceptance Criteria**:
- Fuzzy controller receives updated sensor data at 10Hz minimum
- Controller handles missing/invalid sensor data gracefully
- Obstacle data from LIDAR is prioritized for safety
- Cube detection confidence threshold (>0.7) filters false positives

---

## Non-Functional Requirements

### NFR-001: Real-Time Performance

**Requirement**: Fuzzy inference must execute within control loop timing constraints

**Acceptance Criteria**:
- Fuzzy inference completes in <10ms per cycle
- Control loop maintains 10Hz update rate
- No dropped sensor readings due to processing delays

---

### NFR-002: Robustness

**Requirement**: System handles sensor noise and uncertainty

**Acceptance Criteria**:
- Controller operates correctly with ±10% sensor noise
- Smooth control outputs despite noisy inputs
- No oscillations or chattering in control signals

---

### NFR-003: Tunability

**Requirement**: Fuzzy parameters are easily adjustable

**Acceptance Criteria**:
- Membership functions defined in configuration files (YAML/JSON)
- Rules defined in human-readable format
- Parameter changes take effect without code recompilation
- Tuning interface allows testing different parameter sets

---

## Success Criteria

### Quantitative Metrics

1. **Task Completion**: Robot successfully collects and deposits 15/15 cubes in <10 minutes
2. **Safety**: Zero collisions with obstacles during normal operation
3. **Efficiency**: Average time per cube <40 seconds
4. **Grasp Success**: >90% grasp success rate
5. **Navigation Smoothness**: Angular velocity changes <0.5 rad/s² (no jerky movements)

### Qualitative Metrics

1. **Behavior Interpretability**: Observers can understand robot decisions from rule activations
2. **Robustness**: System recovers from failures (failed grasp, lost cube) without manual intervention
3. **Adaptability**: New behaviors can be added by defining new rules without modifying core logic

---

## User Scenarios

### Scenario 1: Successful Cube Collection

**Context**: Robot starts in SEARCH state, arena has visible cube and obstacles

**Flow**:
1. Robot rotates slowly while scanning for cubes (SEARCH state)
2. Camera detects green cube at 2m distance, 30° to the right
3. Fuzzy controller outputs: angular_velocity = right, linear_velocity = medium
4. Robot turns toward cube while avoiding obstacles (APPROACH state)
5. Distance to cube reduces to 0.25m, robot stops (ALIGN state)
6. Robot fine-tunes position, executes grasp (GRASP state)
7. Grasp successful, holding_cube = true
8. Robot navigates to green deposit box (NAVIGATE_TO_BOX state)
9. Robot arrives at box, releases cube (RELEASE state)
10. Robot returns to SEARCH state

**Expected Outcome**: Cube successfully deposited in correct box, robot ready for next cube

---

### Scenario 2: Obstacle Avoidance During Approach

**Context**: Robot approaching cube, obstacle appears in path

**Flow**:
1. Robot in APPROACH state, moving toward cube at 1.5m distance
2. LIDAR detects obstacle at 0.4m distance, directly ahead
3. Fuzzy rules for obstacle avoidance activate (higher priority)
4. Controller outputs: linear_velocity = slow, angular_velocity = right
5. Robot adjusts path around obstacle
6. Obstacle cleared, robot resumes approach to cube
7. Robot reaches cube and executes grasp

**Expected Outcome**: Robot avoids obstacle while maintaining progress toward cube

---

### Scenario 3: Failed Grasp Recovery

**Context**: Robot attempts grasp but fails to secure cube

**Flow**:
1. Robot in GRASP state, executes grasp maneuver
2. Grasp fails (holding_cube remains false after timeout)
3. State machine transitions to RECOVERY state
4. Robot backs up 0.3m and re-aligns
5. Robot transitions to ALIGN state
6. Robot attempts grasp again
7. Grasp successful on second attempt

**Expected Outcome**: Robot recovers from failure and completes task

---

## Edge Cases

### EC-001: Multiple Cubes Visible

**Scenario**: Camera detects multiple cubes simultaneously

**Handling**: Select nearest cube as target, ignore others until current cube is collected

---

### EC-002: Cube Lost During Approach

**Scenario**: Cube moves out of camera FOV while robot is approaching

**Handling**: Transition to RECOVERY state, execute search pattern in last known direction

---

### EC-003: Stuck in Corner

**Scenario**: Robot navigates into corner with obstacles on three sides

**Handling**: Execute escape maneuver (back up and rotate 180°), return to SEARCH state

---

### EC-004: All Cubes Collected

**Scenario**: No cubes remain in arena

**Handling**: Continue search pattern for 30 seconds, then enter IDLE state

---

## Assumptions

1. **Sensor Availability**: LIDAR and camera provide reliable data at 10Hz minimum
2. **Perception Accuracy**: Obstacle detection >95% accurate, cube detection >90% accurate
3. **Arena Constraints**: Arena dimensions and obstacle positions are static (no moving obstacles)
4. **Cube Properties**: Cubes are stationary and do not move when robot approaches
5. **Deposit Boxes**: Deposit box locations are known and fixed
6. **Grasp Mechanism**: Arm and gripper hardware can execute grasp commands reliably
7. **Computational Resources**: Sufficient processing power for 10Hz control loop

---

## Dependencies

### Internal Dependencies

- **Perception System** (Phase 2): LIDAR obstacle detection and camera cube detection must be functional
- **Robot Controllers** (Phase 1): Base, arm, and gripper control interfaces must be available
- **Odometry** (if implemented): Position tracking for navigation to deposit boxes

### External Dependencies

- **Webots Simulator**: Simulation environment for testing
- **Python Fuzzy Library**: scikit-fuzzy or similar for fuzzy inference
- **NumPy**: Numerical operations for sensor data processing

---

## Out of Scope

The following are explicitly **not** included in this feature:

1. **Path Planning**: No global path planning or A* algorithms (reactive navigation only)
2. **SLAM**: No simultaneous localization and mapping
3. **Multi-Robot Coordination**: Single robot operation only
4. **Dynamic Obstacles**: No handling of moving obstacles
5. **Cube Manipulation Optimization**: No advanced grasp planning or force control
6. **Learning/Adaptation**: No online learning or parameter adaptation (fixed rules)
7. **GPS Usage**: No GPS sensor (prohibited in final demonstration)

---

## Open Questions

None - all critical decisions have reasonable defaults based on project requirements and fuzzy control best practices.

---

## References

### Academic References

- Zadeh, L. A. (1965). "Fuzzy Sets". Information and Control.
- Mamdani, E. H., & Assilian, S. (1975). "An Experiment in Linguistic Synthesis with a Fuzzy Logic Controller"
- Saffiotti, A. (1997). "The uses of fuzzy logic in autonomous robot navigation"

### Project References

- CLAUDE.md: Project context and requirements
- TODO.md: Phase 3 implementation plan
- REFERENCIAS.md: Complete bibliography
