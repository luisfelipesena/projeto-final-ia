# Data Model: Fuzzy Logic Control System

**Feature**: 004-fuzzy-control
**Created**: 2025-11-21
**Purpose**: Define data structures for fuzzy controller, state machine, and integration interfaces

## Overview

This document defines the core data structures for the fuzzy logic control system. All structures are designed for real-time robot control with strict performance constraints (<50ms inference cycle).

## Core Entities

### 1. FuzzyController

**Purpose**: Implements Mamdani fuzzy inference system for converting sensor inputs to control outputs.

**Structure**:
```python
class FuzzyController:
    """
    Mamdani fuzzy inference system for robot control

    Attributes:
        linguistic_vars: Dict of input/output variables with membership functions
        rules: List of fuzzy rules (IF-THEN with weights)
        defuzz_method: Defuzzification strategy ('centroid' default)
        cache: Optional MF evaluation cache for performance
    """

    # Input Variables (6)
    distance_to_obstacle: LinguisticVariable  # {very_near, near, medium, far}
    angle_to_obstacle: LinguisticVariable     # {left, center, right}
    distance_to_cube: LinguisticVariable      # {very_near, near, medium, far}
    angle_to_cube: LinguisticVariable         # {strong_left, left, center, right, strong_right}
    cube_detected: bool                        # Crisp input
    holding_cube: bool                         # Crisp input

    # Output Variables (3)
    linear_velocity: LinguisticVariable       # {stop, slow, medium, fast}
    angular_velocity: LinguisticVariable      # {strong_left, left, straight, right, strong_right}
    action: LinguisticVariable                # {search, approach, grasp, navigate, deposit}
```

**Relationships**:
- Consumes `PerceptionData` from perception module
- Produces `ActionCommand` for robot controller
- Contains 20-30 `FuzzyRule` instances

**Validation Rules** (from FR-001 to FR-008):
- MUST use Mamdani inference (not Sugeno)
- MUST execute in <50ms
- MUST prioritize obstacle avoidance rules (highest weight)
- ALL membership functions MUST overlap 50% with neighbors

---

### 2. LinguisticVariable

**Purpose**: Represents a fuzzy variable with multiple membership functions.

**Structure**:
```python
class LinguisticVariable:
    """
    Fuzzy linguistic variable (input or output)

    Attributes:
        name: Variable identifier (e.g., 'distance_to_obstacle')
        universe: Range of crisp values (e.g., [0.0, 5.0] meters)
        membership_functions: Dict of MF name → function
        mf_type: 'triangular' | 'trapezoidal' | 'gaussian'
    """

    name: str
    universe: Tuple[float, float]              # (min, max) range
    membership_functions: Dict[str, MembershipFunction]
    mf_type: str                               # Per research.md recommendations
```

**Membership Functions per Variable** (from research.md):
- **distance_to_obstacle**: 5 MFs {very_near: 0-0.5m, near: 0.3-1.0m, medium: 0.8-2.0m, far: 1.5-5.0m}
- **angle_to_obstacle**: 7 MFs covering -135° to +135° (270° LIDAR FOV)
- **distance_to_cube**: 5 MFs {very_near: 0-0.2m, near: 0.15-0.5m, medium: 0.4-1.5m, far: 1.0-3.0m}
- **angle_to_cube**: 7 MFs {strong_left: -135° to -60°, left: -90° to -20°, ...}
- **linear_velocity**: 4 MFs {stop: 0, slow: 0.05-0.15 m/s, medium: 0.1-0.25 m/s, fast: 0.2-0.3 m/s}
- **angular_velocity**: 5 MFs {strong_left: 0.3-0.5 rad/s, left: 0.1-0.3, straight: -0.05-0.05, ...}

**Validation Rules** (from research.md):
- MFs MUST overlap 50% (±20% tuning range)
- Universe ranges MUST cover all possible sensor values
- Triangular MFs preferred (performance), Gaussian fallback (accuracy)

---

### 3. MembershipFunction

**Purpose**: Defines shape and parameters of a single membership function.

**Structure**:
```python
class MembershipFunction:
    """
    Single membership function (e.g., 'near' in distance_to_obstacle)

    Attributes:
        label: Human-readable name (e.g., 'very_near')
        shape: 'trimf' | 'trapmf' | 'gaussmf'
        params: Shape-specific parameters
    """

    label: str
    shape: str                                 # scikit-fuzzy function type
    params: Tuple[float, ...]                  # (a, b, c) for trimf, (a, b, c, d) for trapmf
```

**Example - Triangular MF**:
```python
# distance_to_obstacle['very_near']
MembershipFunction(
    label='very_near',
    shape='trimf',
    params=(0.0, 0.0, 0.5)  # Start at 0, peak at 0, end at 0.5m
)
```

---

### 4. FuzzyRule

**Purpose**: Represents a single IF-THEN fuzzy rule with priority weight.

**Structure**:
```python
class FuzzyRule:
    """
    IF-THEN fuzzy rule with priority

    Attributes:
        rule_id: Unique identifier (e.g., 'R001_obstacle_avoidance')
        antecedents: List of input conditions (AND/OR logic)
        consequents: List of output assignments
        weight: Priority weight (1.0-10.0)
        category: 'safety' | 'task' | 'exploration'
    """

    rule_id: str
    antecedents: List[Condition]               # IF conditions
    consequents: List[Assignment]              # THEN assignments
    weight: float                              # 1.0-10.0 (10.0 = highest priority)
    category: str                              # For filtering and priority
```

**Example Rule**:
```python
# Rule: IF distance_to_obstacle IS very_near THEN linear_velocity IS stop
FuzzyRule(
    rule_id='R001_emergency_stop',
    antecedents=[
        Condition(var='distance_to_obstacle', mf='very_near')
    ],
    consequents=[
        Assignment(var='linear_velocity', mf='stop'),
        Assignment(var='angular_velocity', mf='strong_left')
    ],
    weight=10.0,  # Maximum priority (safety)
    category='safety'
)
```

**Validation Rules** (from FR-005, FR-007):
- Minimum 20 rules MUST be defined
- Safety rules (category='safety') MUST have weight ≥8.0
- Safety rules MUST cover ALL obstacle scenarios (distance × angle combinations)

---

### 5. StateMachine

**Purpose**: Manages robot operational states and transitions.

**Structure**:
```python
class StateMachine:
    """
    Finite state machine for task coordination

    Attributes:
        current_state: Active state enum
        previous_state: For state history tracking
        state_start_time: Timestamp when current state entered
        transitions: Dict of state → allowed next states
        timeout_limits: Max duration per state (prevent infinite loops)
    """

    current_state: RobotState                  # Enum value
    previous_state: RobotState                 # For recovery
    state_start_time: float                    # Unix timestamp
    transitions: Dict[RobotState, List[RobotState]]
    timeout_limits: Dict[RobotState, float]    # Seconds (FR-022: max 120s)

    # State tracking
    cubes_collected: int                       # Progress counter (0-15)
    current_cube_color: Optional[str]          # 'green' | 'blue' | 'red'
    grasp_attempts: int                        # Failed grasp counter (max 3)
```

**State Enumeration**:
```python
class RobotState(Enum):
    SEARCHING = 1          # Looking for cubes
    APPROACHING = 2        # Moving toward detected cube
    GRASPING = 3           # Executing grasp sequence
    NAVIGATING_TO_BOX = 4  # Moving toward deposit box
    DEPOSITING = 5         # Executing deposit sequence
    AVOIDING = 6           # Override state for obstacle collision risk
```

**State Transitions** (from FR-009 to FR-013):
```
SEARCHING → APPROACHING        # Cube detected
SEARCHING → AVOIDING           # Obstacle <0.3m

APPROACHING → GRASPING         # Cube within grasp range (0.15m, ±5°)
APPROACHING → SEARCHING        # Cube detection lost
APPROACHING → AVOIDING         # Obstacle <0.3m

GRASPING → NAVIGATING_TO_BOX   # Grasp successful
GRASPING → SEARCHING           # Grasp failed (3 attempts)

NAVIGATING_TO_BOX → DEPOSITING # Reached target box
NAVIGATING_TO_BOX → AVOIDING   # Obstacle <0.3m

DEPOSITING → SEARCHING         # Deposit complete

AVOIDING → [previous_state]    # Obstacle cleared (distance >0.5m)
```

**Validation Rules**:
- AVOIDING state MUST override all others (FR-011)
- Timeout MUST trigger after 2 minutes per state (FR-022)
- Grasp retry count MUST reset after successful deposit

---

### 6. PerceptionData

**Purpose**: Input data from perception module (Phase 2) consumed by fuzzy controller.

**Structure**:
```python
class PerceptionData:
    """
    Sensor data aggregated by perception module

    Attributes:
        obstacle_map: 9-sector LIDAR occupancy
        detected_cubes: List of visible cubes with positions
        timestamp: Unix timestamp of sensor readings
    """

    obstacle_map: ObstacleMap                  # From src/perception/lidar_processor.py
    detected_cubes: List[CubeObservation]      # From src/perception/cube_detector.py
    timestamp: float                           # Sensor data age
```

**ObstacleMap** (defined in Phase 2):
```python
class ObstacleMap:
    sectors: np.ndarray                        # [9] binary occupancy (0/1)
    probabilities: np.ndarray                  # [9] confidence [0, 1]
    min_distances: np.ndarray                  # [9] closest obstacle per sector (meters)
```

**CubeObservation** (defined in Phase 2):
```python
class CubeObservation:
    color: str                                 # 'green' | 'blue' | 'red'
    distance: float                            # Distance from robot (meters)
    angle: float                               # Bearing from robot center (degrees)
    bbox: BoundingBox                          # Camera space coordinates
    confidence: float                          # Detection confidence [0, 1]
```

**Integration Note**: During Phase 3 development, use `MockPerceptionData` (see contracts/perception_mock.py) to simulate sensor inputs.

---

### 7. ActionCommand

**Purpose**: Output from fuzzy controller to robot actuators.

**Structure**:
```python
class ActionCommand:
    """
    Control command produced by fuzzy inference

    Attributes:
        linear_velocity: Forward/backward speed (m/s)
        angular_velocity: Rotation speed (rad/s)
        action: High-level behavior intent
        confidence: Inference confidence (0-1)
        active_rules: Rules that fired this cycle
    """

    linear_velocity: float                     # -0.3 to 0.3 m/s (FR-020)
    angular_velocity: float                    # -0.5 to 0.5 rad/s (FR-020)
    action: str                                # 'search' | 'approach' | 'grasp' | 'navigate' | 'deposit'
    confidence: float                          # Aggregated rule strength
    active_rules: List[str]                    # Rule IDs that fired (for logging)
    timestamp: float                           # Decision timestamp
```

**Validation Rules** (from FR-015, FR-016, FR-020):
- Velocities MUST be limited: |vx| ≤ 0.3 m/s, |ω| ≤ 0.5 rad/s
- Acceleration MUST not exceed 0.5 m/s² between cycles (smooth motion)
- Action string MUST match one of 5 valid values

---

### 8. RobotControllerState

**Purpose**: Unified state tracking for robot controller integration layer.

**Structure**:
```python
class RobotControllerState:
    """
    Complete robot state combining fuzzy + state machine + perception

    Attributes:
        fuzzy: FuzzyController instance
        state_machine: StateMachine instance
        last_perception: Most recent PerceptionData
        last_command: Most recent ActionCommand
        control_loop_time: Timestamp of last cycle
    """

    fuzzy: FuzzyController
    state_machine: StateMachine
    last_perception: PerceptionData
    last_command: ActionCommand
    control_loop_time: float

    # Performance metrics
    avg_inference_time: float                  # Rolling average (last 100 cycles)
    total_cycles: int                          # Cycle counter
```

---

## Data Flow Diagram

```
┌─────────────────┐
│ Perception      │
│ Module          │
│ (Phase 2)       │
└────────┬────────┘
         │ PerceptionData
         │ (ObstacleMap + CubeObservation)
         ▼
┌─────────────────┐
│ FuzzyController │ ◄── LinguisticVariables (6 inputs, 3 outputs)
│                 │ ◄── FuzzyRules (20-30 rules)
│ - Fuzzification │
│ - Inference     │
│ - Defuzzification│
└────────┬────────┘
         │ ActionCommand
         │ (velocities + action intent)
         ▼
┌─────────────────┐
│ StateMachine    │
│                 │
│ - Current state │
│ - Transitions   │
│ - Timeouts      │
└────────┬────────┘
         │ RobotState + ActionCommand
         ▼
┌─────────────────┐
│ Robot Actuators │
│ - Base (vx, ω)  │
│ - Arm (preset)  │
│ - Gripper       │
└─────────────────┘
```

---

## State Transitions Flow

```
     START
       │
       ▼
  ┌─────────┐
  │SEARCHING│◄────────────────┐
  └────┬────┘                 │
       │ cube_detected        │
       ▼                      │
  ┌──────────┐                │
  │APPROACHING│               │
  └────┬─────┘                │
       │ grasp_range_reached  │
       ▼                      │
  ┌────────┐                  │
  │GRASPING│                  │
  └────┬───┘                  │
       │ grasp_success        │
       ▼                      │
  ┌────────────────┐          │
  │NAVIGATING_TO_BOX│         │
  └────────┬───────┘          │
           │ box_reached      │
           ▼                  │
      ┌──────────┐            │
      │DEPOSITING│            │
      └─────┬────┘            │
            │ deposit_complete│
            └─────────────────┘

    ANY STATE
        │ obstacle_critical
        ▼
    ┌────────┐
    │AVOIDING│
    └────┬───┘
         │ obstacle_cleared
         └──► return to previous_state
```

---

## Validation Summary

All data structures satisfy functional requirements from spec.md:

- **FR-001 to FR-008**: FuzzyController with Mamdani inference, linguistic variables, membership functions, rules, defuzzification
- **FR-009 to FR-013**: StateMachine with 6 states, transitions, override logic, cube tracking
- **FR-014**: PerceptionData interface matches Phase 2 contracts
- **FR-015**: ActionCommand translates to robot commands
- **FR-016**: Smooth velocity transitions tracked in RobotControllerState
- **FR-017**: Logging via active_rules tracking
- **FR-018**: Visualization data available in all structures
- **FR-019 to FR-022**: Safety constraints enforced in validation rules

**Next Phase**: Generate contracts/ interfaces for implementation (quickstart.md + contract files)
