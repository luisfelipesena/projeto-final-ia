# Feature Specification: Fuzzy Logic Control System

**Feature Branch**: `004-fuzzy-control`
**Created**: 2025-11-21
**Status**: Draft
**Input**: User description: "Implement fuzzy logic controller for YouBot autonomous navigation and manipulation control with state machine for task execution"

## User Scenarios & Testing

### User Story 1 - Obstacle Avoidance Navigation (Priority: P1) 🎯 MVP

The robot navigates through the arena avoiding obstacles in real-time using LIDAR-based fuzzy logic decisions, maintaining safe distances and smooth trajectories.

**Why this priority**: Core safety requirement - robot must never collide with obstacles. This is the foundation for all other behaviors and can be tested independently without cube detection or manipulation.

**Independent Test**: Place robot in arena with obstacles at various distances and angles. Robot should navigate for 5 minutes without collisions, maintaining minimum 0.3m clearance from obstacles. Delivers immediate value: safe autonomous navigation.

**Acceptance Scenarios**:

1. **Given** robot is moving forward at medium speed, **When** obstacle detected 0.5m ahead in center sector, **Then** robot reduces speed to slow and steers away smoothly
2. **Given** robot is stationary, **When** obstacle detected <0.3m in any sector, **Then** robot stops immediately and turns away from obstacle
3. **Given** robot is navigating, **When** multiple obstacles surround robot, **Then** robot identifies clearest path and navigates toward it
4. **Given** robot is turning, **When** new obstacle appears in turning direction, **Then** robot adjusts angular velocity to avoid collision

---

### User Story 2 - Cube Approach and Acquisition (Priority: P1) 🎯 MVP

When a cube is detected, robot approaches it smoothly using fuzzy logic to adjust velocities based on distance and angle, positioning itself for successful grasping.

**Why this priority**: Primary task requirement - robot must collect cubes. This can be tested independently by placing single cube in clear area. Depends only on cube detection (mock-able).

**Independent Test**: Place single cube in open area. Robot should detect, approach, and position itself within grasping range (0.15m, ±5°) within 30 seconds. Success rate >80%. Delivers value: cube collection capability.

**Acceptance Scenarios**:

1. **Given** cube detected 2m away at -30° angle, **When** robot begins approach, **Then** robot turns toward cube and moves at medium speed
2. **Given** robot 0.5m from cube at center, **When** approaching, **Then** robot reduces to slow speed and fine-tunes alignment
3. **Given** robot 0.15m from cube aligned, **When** distance threshold reached, **Then** robot stops and triggers grasp action
4. **Given** robot approaching cube, **When** obstacle appears between robot and cube, **Then** robot prioritizes obstacle avoidance then resumes approach

---

### User Story 3 - Navigation to Target Box (Priority: P2)

After grasping a cube, robot navigates to the corresponding color-coded deposit box using fuzzy logic for path smoothness and obstacle avoidance en route.

**Why this priority**: Completes the collect-deposit cycle. Can be tested independently by manually placing cube in gripper and commanding navigation to specific box.

**Independent Test**: Place cube in gripper, set target box coordinates. Robot should navigate to box location within 1.5 minutes while avoiding obstacles. Success criteria: arrives within 0.3m of box, <3 collisions per 10 runs.

**Acceptance Scenarios**:

1. **Given** robot holding green cube at position A, **When** target is green box at position B, **Then** robot navigates smoothly to box maintaining safe clearances
2. **Given** robot en route to box, **When** obstacles encountered, **Then** robot adjusts trajectory while maintaining general direction toward target
3. **Given** robot 0.5m from target box, **When** final approach begins, **Then** robot reduces speed and aligns for deposit
4. **Given** robot at box location, **When** positioned correctly, **Then** robot transitions to deposit state

---

### User Story 4 - State Machine Coordination (Priority: P2)

Robot executes complete cube collection cycle by transitioning through states (Searching → Approaching → Grasping → Navigating → Depositing) with fuzzy logic driving actions within each state.

**Why this priority**: Integrates all behaviors into cohesive system. Represents complete autonomous operation.

**Independent Test**: Run full cycle: start in empty arena with 3 cubes. Robot should autonomously collect and deposit all 3 cubes, demonstrating proper state transitions. Success if completes cycle without manual intervention.

**Acceptance Scenarios**:

1. **Given** robot in SEARCHING state, **When** no cubes visible, **Then** fuzzy controller commands exploration pattern (slow rotation + forward movement)
2. **Given** robot in SEARCHING state, **When** cube detected, **Then** state transitions to APPROACHING and fuzzy adjusts velocities toward cube
3. **Given** robot in APPROACHING state, **When** cube within grasp range, **Then** state transitions to GRASPING and fuzzy stops motion
4. **Given** robot in GRASPING state, **When** grasp sequence completes successfully, **Then** state transitions to NAVIGATING_TO_BOX
5. **Given** robot in NAVIGATING_TO_BOX state, **When** box reached, **Then** state transitions to DEPOSITING
6. **Given** robot in DEPOSITING state, **When** deposit complete, **Then** state transitions to SEARCHING for next cube
7. **Given** robot in any state, **When** critical obstacle detected <0.3m, **Then** state overrides to AVOIDING until clear

---

### Edge Cases

- What happens when robot gets stuck in a corner with obstacles on three sides?
  - Fuzzy logic should identify clearest sector (even if tight) and execute slow, careful escape maneuver
- How does system handle cube detection lost during approach?
  - State machine should return to SEARCHING, fuzzy logic resumes exploration pattern
- What if robot approaches wrong color box while holding cube?
  - Navigation fuzzy logic should reject incorrect target and reroute to correct box based on cube color
- What happens when multiple cubes are visible simultaneously?
  - Fuzzy priority rules select nearest cube with clearest path (lowest obstacle density)
- How does robot behave when LIDAR detects cube as obstacle during approach?
  - Cube detection has higher priority in APPROACHING state; obstacle avoidance relaxes threshold for target objects
- What if robot fails to grasp cube after 3 attempts?
  - State machine should timeout, return to SEARCHING, and mark cube location as problematic

## Requirements

### Functional Requirements

#### Fuzzy Controller Core

- **FR-001**: System MUST implement Mamdani fuzzy inference system using scikit-fuzzy library
- **FR-002**: System MUST define linguistic variables for inputs: distance_to_obstacle {very_near, near, medium, far}, angle_to_obstacle {left, center, right}, distance_to_cube {very_near, near, medium, far}, angle_to_cube {strong_left, left, center, right, strong_right}, cube_detected {true, false}, holding_cube {true, false}
- **FR-003**: System MUST define linguistic variables for outputs: linear_velocity {stop, slow, medium, fast}, angular_velocity {strong_left, left, straight, right, strong_right}, action {search, approach, grasp, navigate_to_box, deposit}
- **FR-004**: System MUST use membership functions (triangular, trapezoidal, or Gaussian) with ranges validated for arena scale and robot capabilities
- **FR-005**: System MUST implement minimum 20 fuzzy rules covering: obstacle avoidance (highest priority), cube search behavior, cube approach control, navigation to target box
- **FR-006**: System MUST use centroid defuzzification method to compute crisp output values
- **FR-007**: System MUST prioritize obstacle avoidance rules over all other behaviors (safety first)
- **FR-008**: Fuzzy controller MUST execute within 50ms per decision cycle to maintain real-time responsiveness

#### State Machine

- **FR-009**: System MUST implement state machine with states: SEARCHING, APPROACHING, GRASPING, NAVIGATING_TO_BOX, DEPOSITING, AVOIDING
- **FR-010**: State machine MUST define clear transition conditions between all states based on sensor inputs and task progress
- **FR-011**: System MUST allow AVOIDING state to override any other state when critical obstacle detected (<0.3m)
- **FR-012**: State machine MUST track current cube color when holding cube to navigate to correct deposit box
- **FR-013**: System MUST return to SEARCHING state after successful deposit or failed grasp attempts (max 3 retries)

#### Integration & Control

- **FR-014**: System MUST interface with perception module to receive obstacle maps (9-sector LIDAR) and cube detections (position, color, distance)
- **FR-015**: System MUST translate fuzzy outputs (velocities, action) into robot commands for base (vx, vy, omega), arm positions, and gripper actions
- **FR-016**: System MUST implement smooth velocity transitions to avoid jerky motion (acceleration limits: 0.5 m/s²)
- **FR-017**: System MUST log all state transitions and fuzzy decisions for debugging and analysis
- **FR-018**: System MUST provide visualization of current state, active fuzzy rules, and membership values during operation

#### Safety & Constraints

- **FR-019**: System MUST maintain minimum clearance of 0.3m from obstacles at all times
- **FR-020**: System MUST limit maximum linear velocity to 0.3 m/s and angular velocity to 0.5 rad/s for safety
- **FR-021**: System MUST stop all motion when critical sensor failure detected (no LIDAR data, no camera data)
- **FR-022**: System MUST implement timeout mechanisms for each state to prevent infinite loops (max 2 minutes per state)

### Key Entities

- **FuzzyController**: Implements Mamdani inference with linguistic variables, membership functions, rules database, and defuzzification. Consumes sensor data, produces velocity commands and action decisions.
- **StateMachine**: Manages robot operational states, transition logic, state-specific behaviors. Tracks task progress (cubes collected, current target).
- **FuzzyRule**: Represents single IF-THEN rule with antecedent (input conditions), consequent (output actions), and priority weight.
- **RobotState**: Current operational context including: position estimate, held cube color, current state enum, active fuzzy rules, sensor readings snapshot.
- **ActionCommand**: Output from fuzzy controller containing: linear velocity, angular velocity, action type, confidence level.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Robot completes obstacle avoidance test (5 minutes navigation in obstacle-filled arena) with zero collisions in >90% of runs
- **SC-002**: Robot successfully approaches and positions for cube grasping with <5cm position error and <10° angular error in >80% of attempts
- **SC-003**: Robot navigates from random start position to target box within 90 seconds in >75% of runs
- **SC-004**: Fuzzy controller decision cycle executes in <50ms, enabling real-time 20Hz control loop
- **SC-005**: State machine transitions occur correctly with <2% false transitions (wrong state entered) across 100 cycle tests
- **SC-006**: Robot completes full collect-deposit cycle for single cube within 3 minutes in >70% of attempts
- **SC-007**: Smooth motion quality: velocity changes do not exceed 0.5 m/s² jerk, resulting in observable smooth trajectories
- **SC-008**: System demonstrates correct priority handling: obstacle avoidance overrides cube approach in 100% of test cases where obstacle appears during approach

## Dependencies

### Technical Dependencies

- **Phase 2 Output (Mock-able)**: Perception system providing obstacle maps and cube detections
  - Can be developed with simulated sensor data initially
  - Real perception integration in Phase 6
- **YouBot Controller API**: Base movement (move(vx, vy, omega)), arm control (set_position), gripper control (grip/release)
- **Python Libraries**: scikit-fuzzy (fuzzy logic), numpy (numerical operations), matplotlib (membership function visualization)

### External Dependencies

- Webots simulation environment for testing and validation
- Arena configuration with obstacles and cubes
- Logging infrastructure for debugging

## Assumptions

- Perception system (Phase 2) provides 9-sector obstacle map at 10Hz update rate
- Cube detection provides position (x, y), color, and distance at 10Hz when visible
- Robot base accepts velocity commands (vx, vy, omega) at 20Hz control rate
- Arena dimensions known: 7.0m × 4.0m with fixed deposit box positions
- Deposit box positions are pre-configured (hardcoded initially): green(-2, 1.5), blue(0, 1.5), red(2, 1.5)
- Robot physical constraints: max speed 0.5 m/s, max rotation 1.0 rad/s (fuzzy limits to 60% for safety)
- LIDAR provides reliable obstacle detection at ranges 0.1m to 5.0m
- Membership function ranges based on arena scale: near obstacles <0.5m, far obstacles >2.0m, near cubes <0.3m, far cubes >1.5m
- Fuzzy rules assume deterministic sensor readings (noise handled by perception layer)
- State machine assumes arm and gripper operations are synchronous (blocking calls)

## Constraints & Limitations

- Fuzzy logic inherently reactive - no long-term path planning or global optimization
- State machine deterministic - cannot handle concurrent states or parallel behaviors
- Obstacle avoidance purely local - may not escape complex trap scenarios (U-shapes, dead ends)
- No learning or adaptation - fuzzy rules and membership functions are static after tuning
- Performance depends on perception quality - fuzzy decisions only as good as sensor data
- Computational overhead: fuzzy inference for all rules on every cycle, limits to ~30-40 rules maximum for real-time performance

## Out of Scope

- **Path Planning Algorithms**: No A*, RRT, or global trajectory optimization (reactive navigation only)
- **Localization/Mapping**: No SLAM, no odometry integration, no position tracking (perception provides all spatial awareness)
- **Learning/Adaptation**: No online rule tuning, no reinforcement learning, no parameter optimization during runtime
- **Multi-Robot Coordination**: Single robot operation only
- **Advanced Manipulation**: Arm motion planning limited to predefined positions (no inverse kinematics solver, no collision checking for arm)
- **Perception Development**: Assumes perception module exists (Phase 2 responsibility)
- **Grasp Planning**: Gripper control is simple open/close (no force sensing, no adaptive grasping)
- **Error Recovery**: Basic timeout mechanisms only (no sophisticated recovery strategies for hardware failures)

## References

Based on TODO.md Phase 3 requirements and scientific foundations:

- **Zadeh (1965)**: Fuzzy Sets - theoretical foundation for fuzzy logic
- **Mamdani & Assilian (1975)**: Fuzzy controller design principles
- **Saffiotti (1997)**: Fuzzy logic for mobile robot navigation
- **Antonelli et al. (2007)**: Path tracking with fuzzy control
- **Bischoff et al. (2011)**: YouBot robot specifications and constraints
- **Taheri et al. (2015)**: Mecanum wheel kinematics and control
