# Quickstart: Fuzzy Logic Control System

**Feature**: 004-fuzzy-control
**Purpose**: Get started with fuzzy controller development and testing
**Prerequisites**: Python 3.8+, scikit-fuzzy, numpy, matplotlib

## Overview

This guide shows how to:
1. Initialize and use the fuzzy controller
2. Define custom fuzzy rules
3. Test with mock perception data
4. Integrate with state machine
5. Visualize membership functions and decisions

## Installation

```bash
# Install dependencies
pip install scikit-fuzzy>=0.4.2 numpy>=1.24.0 matplotlib>=3.7.0

# Verify installation
python -c "import skfuzzy as fuzz; print('scikit-fuzzy version:', fuzz.__version__)"
```

## Basic Usage

### 1. Initialize Fuzzy Controller

```python
from src.control.fuzzy_controller import FuzzyController
from specs.004-fuzzy-control.contracts.fuzzy_controller import FuzzyInputs

# Create controller with default configuration
controller = FuzzyController()
controller.initialize()  # Loads default rules and membership functions

# Verify setup
print(f"Loaded {len(controller.get_rules())} rules")
print(f"Linguistic variables: {list(controller.get_linguistic_variables().keys())}")
```

**Expected Output**:
```
Loaded 25 rules
Linguistic variables: ['distance_to_obstacle', 'angle_to_obstacle', 'distance_to_cube',
                       'angle_to_cube', 'linear_velocity', 'angular_velocity', 'action']
```

### 2. Run Fuzzy Inference

```python
# Create crisp inputs (from sensors)
inputs = FuzzyInputs(
    distance_to_obstacle=0.8,  # 0.8m to nearest obstacle
    angle_to_obstacle=15.0,    # 15° right
    distance_to_cube=1.2,      # 1.2m to detected cube
    angle_to_cube=-30.0,       # 30° left
    cube_detected=True,
    holding_cube=False
)

# Perform inference
outputs = controller.infer(inputs)

print(f"Linear velocity: {outputs.linear_velocity:.3f} m/s")
print(f"Angular velocity: {outputs.angular_velocity:.3f} rad/s")
print(f"Action: {outputs.action}")
print(f"Confidence: {outputs.confidence:.2f}")
print(f"Active rules: {outputs.active_rules}")
```

**Expected Output**:
```
Linear velocity: 0.180 m/s
Angular velocity: -0.120 rad/s
Action: approach
Confidence: 0.78
Active rules: ['R005_approach_cube', 'R012_turn_toward_cube', 'R003_slow_near_obstacle']
```

### 3. Integrate with State Machine

```python
from src.control.state_machine import StateMachine, RobotState
from specs.004-fuzzy-control.contracts.state_machine import StateTransitionConditions

# Initialize state machine
sm = StateMachine(initial_state=RobotState.SEARCHING)

# Control loop example
while True:
    # Get sensor data (from Webots or mock)
    perception_data = get_perception_data()  # Your sensor interface

    # Convert to fuzzy inputs
    fuzzy_inputs = FuzzyInputs(
        distance_to_obstacle=perception_data.obstacle_map.min_distances.min(),
        angle_to_obstacle=compute_obstacle_angle(perception_data.obstacle_map),
        distance_to_cube=perception_data.detected_cubes[0].distance if perception_data.detected_cubes else 999.0,
        angle_to_cube=perception_data.detected_cubes[0].angle if perception_data.detected_cubes else 0.0,
        cube_detected=len(perception_data.detected_cubes) > 0,
        holding_cube=sm.get_target_cube_color() is not None
    )

    # Fuzzy inference
    fuzzy_outputs = controller.infer(fuzzy_inputs)

    # State machine update
    conditions = StateTransitionConditions(
        cube_detected=fuzzy_inputs.cube_detected,
        cube_distance=fuzzy_inputs.distance_to_cube,
        cube_angle=fuzzy_inputs.angle_to_cube,
        obstacle_distance=fuzzy_inputs.distance_to_obstacle,
        holding_cube=fuzzy_inputs.holding_cube,
        at_target_box=check_at_box(),  # Your logic
        grasp_success=check_grasp(),   # Your logic
        deposit_complete=False
    )
    new_state = sm.update(conditions)

    # Send commands to robot
    send_velocity_command(fuzzy_outputs.linear_velocity, fuzzy_outputs.angular_velocity)

    # Log state transitions
    if new_state != sm.current_state:
        print(f"State transition: {sm.current_state} → {new_state}")

    time.sleep(0.05)  # 20Hz control loop
```

## Testing with Mock Data

### 4. Unit Test Fuzzy Rules

```python
from specs.004-fuzzy-control.contracts.perception_mock import MockPerceptionSystem

# Create mock perception
mock = MockPerceptionSystem(seed=42)

# Test scenario: Obstacle ahead, cube to the left
scenario_data = mock.get_scenario('obstacle_front')

# Extract fuzzy inputs
inputs = FuzzyInputs(
    distance_to_obstacle=scenario_data.obstacle_map.min_distances[3],  # Center sector
    angle_to_obstacle=0.0,
    distance_to_cube=999.0,  # No cube in this scenario
    angle_to_cube=0.0,
    cube_detected=False,
    holding_cube=False
)

# Test inference
outputs = controller.infer(inputs)

# Assertions for obstacle avoidance
assert outputs.linear_velocity < 0.1, "Should slow down near obstacle"
assert abs(outputs.angular_velocity) > 0.2, "Should turn away from obstacle"
assert 'R001_emergency_stop' in outputs.active_rules or 'R002_obstacle_avoidance' in outputs.active_rules
print("✅ Obstacle avoidance test PASSED")
```

### 5. Test Predefined Scenarios

```python
# Test all safety scenarios
test_scenarios = [
    'obstacle_critical',     # 0.2m ahead - emergency stop
    'obstacle_front',        # 0.5m ahead - slow + turn
    'corner_trap',           # Surrounded - escape behavior
    'narrow_passage'         # Left+right obstacles - careful forward
]

for scenario_name in test_scenarios:
    data = mock.get_scenario(scenario_name)
    inputs = convert_perception_to_fuzzy_inputs(data)
    outputs = controller.infer(inputs)

    print(f"\n{scenario_name}:")
    print(f"  Linear: {outputs.linear_velocity:.3f}, Angular: {outputs.angular_velocity:.3f}")
    print(f"  Active rules: {outputs.active_rules[:3]}")  # First 3
```

## Adding Custom Rules

### 6. Define New Fuzzy Rule

```python
from specs.004-fuzzy-control.contracts.fuzzy_controller import (
    FuzzyRule, FuzzyRuleCondition, FuzzyRuleAssignment
)

# Create custom rule: IF cube is very_near AND aligned THEN stop and grasp
custom_rule = FuzzyRule(
    rule_id='R099_custom_precise_grasp',
    antecedents=[
        FuzzyRuleCondition(variable='distance_to_cube', membership_function='very_near'),
        FuzzyRuleCondition(variable='angle_to_cube', membership_function='center')
    ],
    consequents=[
        FuzzyRuleAssignment(variable='linear_velocity', membership_function='stop'),
        FuzzyRuleAssignment(variable='angular_velocity', membership_function='straight'),
        FuzzyRuleAssignment(variable='action', membership_function='grasp')
    ],
    weight=9.0,  # High priority (just below safety)
    category='task'
)

# Add to controller
controller.add_rule(custom_rule)
print(f"✅ Added rule {custom_rule.rule_id}")
```

## Visualization

### 7. Plot Membership Functions

```python
# Visualize input variables
controller.visualize_membership_functions('distance_to_obstacle', save_path='docs/mf_distance_obstacle.png')
controller.visualize_membership_functions('angle_to_cube', save_path='docs/mf_angle_cube.png')

# Visualize output variables
controller.visualize_membership_functions('linear_velocity', save_path='docs/mf_linear_vel.png')
controller.visualize_membership_functions('angular_velocity', save_path='docs/mf_angular_vel.png')

print("✅ Membership function plots saved to docs/")
```

### 8. Debug Active Rules

```python
# Run inference with logging enabled
controller_debug = FuzzyController(config={'logging': True})
controller_debug.initialize()

inputs = FuzzyInputs(
    distance_to_obstacle=0.4,
    angle_to_obstacle=0.0,
    distance_to_cube=999.0,
    angle_to_cube=0.0,
    cube_detected=False,
    holding_cube=False
)

outputs = controller_debug.infer(inputs)

# Check logs/fuzzy_decisions.log for detailed rule activation levels
```

**Example Log Output**:
```
[2025-11-21 14:23:45] Fuzzification:
  distance_to_obstacle=0.4 → {very_near: 0.4, near: 0.6, medium: 0.0, far: 0.0}
  angle_to_obstacle=0.0 → {left: 0.0, center: 1.0, right: 0.0}

[2025-11-21 14:23:45] Rule Activation:
  R001_emergency_stop: 0.4 (weight=10.0)
  R002_obstacle_avoidance: 0.6 (weight=9.0)
  R015_explore: 0.2 (weight=3.0)

[2025-11-21 14:23:45] Defuzzification:
  linear_velocity: 0.08 m/s (weighted centroid)
  angular_velocity: 0.25 rad/s (weighted centroid)
```

## Performance Testing

### 9. Benchmark Inference Time

```python
import time

# Warm-up
for _ in range(10):
    controller.infer(inputs)

# Measure 1000 cycles
start = time.time()
for _ in range(1000):
    outputs = controller.infer(inputs)
elapsed = time.time() - start

avg_time_ms = (elapsed / 1000) * 1000
print(f"Average inference time: {avg_time_ms:.2f} ms")
print(f"Target: <50ms ({'✅ PASS' if avg_time_ms < 50 else '❌ FAIL'})")
```

**Expected Result**: 10-30ms (well under 50ms requirement)

## Integration with Webots

### 10. Connect to Webots Controller

```python
# In controllers/youbot/youbot.py

from src.control.fuzzy_controller import FuzzyController
from src.control.state_machine import StateMachine, RobotState
from src.perception.perception_system import PerceptionSystem

class YouBotController:
    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())

        # Initialize modules
        self.perception = PerceptionSystem(robot)
        self.fuzzy = FuzzyController()
        self.fuzzy.initialize()
        self.state_machine = StateMachine()

    def run(self):
        while self.robot.step(self.timestep) != -1:
            # Perception
            perception_data = self.perception.update()

            # Fuzzy inference
            fuzzy_inputs = self.convert_perception_to_fuzzy(perception_data)
            fuzzy_outputs = self.fuzzy.infer(fuzzy_inputs)

            # State machine
            conditions = self.create_transition_conditions(perception_data)
            new_state = self.state_machine.update(conditions)

            # Actuate
            self.send_base_velocity(fuzzy_outputs.linear_velocity, fuzzy_outputs.angular_velocity)
```

## Troubleshooting

**Q: Inference time >50ms**
A: Enable caching in config: `FuzzyController(config={'enable_cache': True})`

**Q: Robot oscillates/unstable**
A: Check membership function overlap (should be 50% ±20%). Visualize MFs to verify smooth transitions.

**Q: Rules not firing**
A: Enable logging and check rule activation levels. Verify input ranges match universe definitions.

**Q: State machine stuck**
A: Check timeout settings (max 120s per state). Verify transition conditions match sensor data.

## Next Steps

1. **Run validation**: `pytest tests/control/ -v`
2. **Tune membership functions**: Adjust ranges in `src/control/fuzzy_rules.py`
3. **Add domain-specific rules**: Use template in section 6
4. **Integrate with Webots**: Follow section 10
5. **Performance profiling**: Use section 9 benchmark

## References

- **spec.md**: Complete feature requirements
- **data-model.md**: Data structure definitions
- **research.md**: Fuzzy logic best practices
- **contracts/**: Interface contracts for all modules

---

**For implementation guidance, see**: `specs/004-fuzzy-control/tasks.md` (generated by `/speckit.tasks`)
