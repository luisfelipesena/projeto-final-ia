# Implementation Plan: Fuzzy Control System for Autonomous Navigation

**Branch**: `005-fuzzy-controller` | **Date**: 2025-11-23 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/005-fuzzy-controller/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a fuzzy logic control system that translates imprecise sensor inputs (LIDAR distances/angles, camera cube detections) into smooth control outputs (linear/angular velocities, high-level actions). The system coordinates multiple behaviors (obstacle avoidance, cube search, approach, manipulation) through a state machine, enabling the YouBot to autonomously collect 15 colored cubes and deposit them in correct boxes within 10 minutes.

**Primary Technical Approach**:
- Fuzzy inference engine using scikit-fuzzy (Mamdani method)
- 6 input variables (distance/angle to obstacles and cubes, detection flags)
- 3 output variables (linear velocity, angular velocity, action command)
- Minimum 15 fuzzy rules with priority-based activation
- 7-state machine (SEARCH → APPROACH → ALIGN → GRASP → NAVIGATE_TO_BOX → RELEASE → RECOVERY)
- Integration with existing perception system (Phase 2) and robot controllers (Phase 1)

## Technical Context

**Language/Version**: Python 3.14 (current project standard)  
**Primary Dependencies**: 
- scikit-fuzzy 0.4.2+ (fuzzy inference engine)
- NumPy 2.2+ (numerical operations)
- Existing: PyTorch (perception models), Webots controller API

**Storage**: 
- Configuration files: YAML for membership functions and fuzzy rules
- Logs: JSON-formatted state transitions and rule activations

**Testing**: pytest (existing test infrastructure)  
**Target Platform**: Webots R2023b simulator on macOS/Linux  
**Project Type**: Single project (robotic control system)  

**Performance Goals**: 
- Fuzzy inference: <10ms per cycle
- Control loop: 10Hz update rate minimum
- Task completion: 15 cubes in <10 minutes
- Grasp success: >90%

**Constraints**: 
- Real-time: No dropped sensor readings
- Safety: Zero collisions with obstacles
- Smoothness: Angular velocity changes <0.5 rad/s²
- No GPS usage (prohibited in final demonstration)

**Scale/Scope**: 
- 6 fuzzy input variables, 3 output variables
- Minimum 15 fuzzy rules (likely 20-25 for robustness)
- 7 states in state machine
- Integration with 2 perception modules (LIDAR, camera)
- 4 manipulation primitives (search, approach, grasp, release)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Principle I: Fundamentação Científica

**Status**: PASS

- Fuzzy logic foundation: Zadeh (1965), Mamdani & Assilian (1975)
- Robot navigation: Saffiotti (1997) - fuzzy logic in autonomous navigation
- State machine coordination: Standard robotics practice (Thrun et al., 2005)
- All decisions will be documented in DECISIONS.md with scientific references

**Action**: Document fuzzy membership function design and rule base in DECISIONS.md before implementation

### ✅ Principle II: Rastreabilidade Total

**Status**: PASS

- Feature tracked in specs/005-fuzzy-controller/
- Implementation will update DECISIONS.md for each technical choice
- Git commits will reference decision IDs
- State transitions and rule activations logged for debugging

**Action**: Create DECISION entries for:
- Fuzzy library choice (scikit-fuzzy vs alternatives)
- Membership function shapes (triangular vs trapezoidal vs gaussian)
- Defuzzification method (centroid vs bisector vs MOM)
- State machine implementation approach

### ✅ Principle III: Desenvolvimento Incremental

**Status**: PASS

- Phase 3 (Fuzzy Control) follows completed Phase 1 (Robot Controllers) and Phase 2 (Perception Infrastructure)
- Deliverables: Fuzzy controller module, state machine, integration tests
- TODO.md will be updated with Phase 3 completion status

**Action**: Mark Phase 3 tasks in TODO.md as in-progress/complete

### ✅ Principle IV: Qualidade Senior

**Status**: PASS

- Modular architecture: `src/control/fuzzy_controller.py`, `src/control/state_machine.py`
- Unit tests for fuzzy inference, state transitions
- PEP8 compliance, type hints, docstrings
- Target: >80% test coverage for control module

**Action**: Create test suite in `tests/control/`

### ✅ Principle V: Restrições Disciplinares

**Status**: PASS

- ✅ No modification to supervisor.py
- ✅ No GPS usage (reactive navigation only)
- ✅ Fuzzy logic requirement satisfied (core feature)
- ✅ Scientific foundation (Zadeh, Mamdani, Saffiotti)

**Action**: Ensure presentation explains fuzzy logic with citations (no code shown)

### ✅ Principle VI: Workflow SpecKit

**Status**: PASS

- ✅ `/speckit.specify` completed → spec.md created
- 🔄 `/speckit.plan` in progress → plan.md (this file)
- ⏳ Next: `/speckit.tasks` → tasks.md
- ⏳ Then: `/speckit.implement` → execute tasks

**Action**: Follow SpecKit workflow to completion

## Project Structure

### Documentation (this feature)

```text
specs/005-fuzzy-controller/
├── spec.md              # Feature specification (completed)
├── checklists/
│   └── requirements.md  # Spec quality checklist (completed)
├── plan.md              # This file (/speckit.plan output)
├── research.md          # Phase 0 output (to be generated)
├── data-model.md        # Phase 1 output (to be generated)
├── quickstart.md        # Phase 1 output (to be generated)
├── contracts/           # Phase 1 output (API contracts if needed)
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/control/                    # NEW - Fuzzy control module
├── __init__.py
├── fuzzy_controller.py         # Main fuzzy inference engine
├── state_machine.py            # State machine coordinator
├── membership_functions.py     # Fuzzy variable definitions
├── rule_base.py                # Fuzzy rules
└── config/                     # Configuration files
    ├── membership_functions.yaml
    └── fuzzy_rules.yaml

src/perception/                 # EXISTING - Integration point
├── lidar_processor.py          # Provides obstacle data
├── cube_detector.py            # Provides cube detection data
└── perception_system.py        # Unified perception interface

src/robot/                      # EXISTING - Integration point
├── base.py                     # Base movement commands
├── arm.py                      # Arm positioning
└── gripper.py                  # Grasp/release

tests/control/                  # NEW - Test suite
├── test_fuzzy_controller.py
├── test_state_machine.py
├── test_membership_functions.py
├── test_rule_base.py
└── test_integration.py

notebooks/                      # NEW - Development/tuning
└── fuzzy_tuning.ipynb          # Interactive fuzzy parameter tuning

logs/                           # EXISTING - Logging
└── fuzzy_control/              # NEW - Control logs
    ├── state_transitions.json
    └── rule_activations.json
```

**Structure Decision**: Single project structure (Option 1) is appropriate for this robotic control system. The fuzzy control module (`src/control/`) integrates with existing perception (`src/perception/`) and robot control (`src/robot/`) modules. Configuration files in YAML format enable easy tuning without code changes, satisfying NFR-003 (Tunability).

## Complexity Tracking

> **No violations** - All constitution principles satisfied.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| N/A       | N/A        | N/A                                  |

**Justification**: This feature aligns with the project's incremental development plan (Phase 3) and satisfies the mandatory fuzzy logic requirement. The chosen architecture (modular, testable, configurable) follows senior-level quality standards without introducing unnecessary complexity.

---

## Phase 0: Research & Design Decisions

**Objective**: Resolve technical unknowns and establish design patterns

### Research Tasks

1. **Fuzzy Library Selection**
   - **Question**: scikit-fuzzy vs fuzzylite vs custom implementation?
   - **Research**: Compare features, performance, Python 3.14 compatibility
   - **Output**: Decision in research.md with justification

2. **Membership Function Design**
   - **Question**: Triangular vs trapezoidal vs Gaussian shapes?
   - **Research**: Literature on fuzzy robot control (Saffiotti 1997, others)
   - **Output**: Recommended shapes per variable type

3. **Defuzzification Method**
   - **Question**: Centroid vs bisector vs mean-of-maximum?
   - **Research**: Impact on control smoothness and computational cost
   - **Output**: Method selection with trade-offs

4. **State Machine Implementation**
   - **Question**: Enum-based vs class-based vs library (transitions)?
   - **Research**: Python state machine patterns, testability
   - **Output**: Implementation approach

5. **Rule Priority Mechanism**
   - **Question**: How to ensure obstacle avoidance overrides other behaviors?
   - **Research**: Fuzzy rule weighting, hierarchical fuzzy systems
   - **Output**: Priority implementation strategy

6. **Configuration Format**
   - **Question**: YAML vs JSON vs Python dict for fuzzy parameters?
   - **Research**: Readability, validation, hot-reload capability
   - **Output**: Format choice with schema

### Deliverable: research.md

Document all research findings with:
- Decision made
- Rationale (scientific references)
- Alternatives considered
- Trade-offs analyzed

---

## Phase 1: Design & Contracts

**Prerequisites**: research.md complete

### 1. Data Model (`data-model.md`)

**Entities**:

1. **FuzzyVariable**
   - name: str
   - range: tuple[float, float]
   - terms: dict[str, MembershipFunction]
   - type: "input" | "output"

2. **MembershipFunction**
   - name: str
   - shape: "triangular" | "trapezoidal" | "gaussian"
   - parameters: list[float]

3. **FuzzyRule**
   - id: str
   - priority: int
   - antecedents: list[tuple[str, str]]  # [(variable, term), ...]
   - consequents: list[tuple[str, str]]
   - description: str

4. **RobotState** (Enum)
   - SEARCH, APPROACH, ALIGN, GRASP, NAVIGATE_TO_BOX, RELEASE, RECOVERY

5. **PerceptionInput**
   - obstacle_distance: float
   - obstacle_angle: float
   - cube_detected: bool
   - cube_distance: float | None
   - cube_angle: float | None
   - cube_color: str | None
   - holding_cube: bool

6. **ControlOutput**
   - linear_velocity: float
   - angular_velocity: float
   - action: str
   - active_rules: list[str]  # For debugging

### 2. API Contracts (`contracts/`)

**Internal API** (Python interfaces):

```python
# contracts/fuzzy_controller_interface.py
class IFuzzyController(Protocol):
    def compute_control(self, perception: PerceptionInput) -> ControlOutput:
        """
        Compute control outputs from perception inputs using fuzzy inference.
        
        Args:
            perception: Current sensor data and robot state
            
        Returns:
            Control outputs (velocities, action command)
        """
        ...

# contracts/state_machine_interface.py
class IStateMachine(Protocol):
    def update(self, perception: PerceptionInput, control: ControlOutput) -> RobotState:
        """
        Update state machine based on perception and control outputs.
        
        Args:
            perception: Current sensor data
            control: Fuzzy controller outputs
            
        Returns:
            New robot state
        """
        ...
    
    def get_current_state(self) -> RobotState:
        """Get current state"""
        ...
    
    def reset(self) -> None:
        """Reset to initial state (SEARCH)"""
        ...
```

### 3. Quickstart Guide (`quickstart.md`)

**Content**:
- How to configure fuzzy parameters (YAML editing)
- How to run fuzzy controller in simulation
- How to tune membership functions interactively
- How to add new fuzzy rules
- How to debug rule activations (log inspection)

### 4. Agent Context Update

Run `.specify/scripts/bash/update-agent-context.sh claude` to add:
- scikit-fuzzy to technology stack
- Fuzzy control patterns to context
- State machine implementation approach

---

## Phase 2: Task Breakdown

**Output**: `tasks.md` (generated by `/speckit.tasks` command)

**Expected task categories** (40-50 tasks total):

1. **Setup & Configuration** (5 tasks)
   - Create module structure
   - Setup YAML schemas
   - Configure logging

2. **Fuzzy Variables** (10 tasks)
   - Define input variables (6)
   - Define output variables (3)
   - Implement membership functions
   - Validate ranges and overlaps

3. **Fuzzy Rules** (8 tasks)
   - Implement obstacle avoidance rules (4)
   - Implement search rules (2)
   - Implement approach rules (5)
   - Implement manipulation rules (3)
   - Implement rule priority mechanism

4. **Fuzzy Inference Engine** (6 tasks)
   - Implement fuzzification
   - Implement rule evaluation
   - Implement defuzzification
   - Integrate with scikit-fuzzy
   - Performance optimization (<10ms)

5. **State Machine** (7 tasks)
   - Implement state enum
   - Implement state transitions
   - Implement state-specific behaviors
   - Implement recovery logic
   - Add state logging

6. **Integration** (8 tasks)
   - Interface with perception system
   - Interface with robot controllers
   - Implement main control loop
   - Add telemetry/logging
   - Handle edge cases

7. **Testing** (10 tasks)
   - Unit tests: fuzzy variables
   - Unit tests: rule evaluation
   - Unit tests: state machine
   - Integration tests: perception → control
   - Integration tests: control → robot
   - Simulation tests: full task execution
   - Performance tests: timing constraints
   - Robustness tests: sensor noise

8. **Tuning & Validation** (6 tasks)
   - Create tuning notebook
   - Tune membership functions
   - Tune rule weights/priorities
   - Validate success criteria (15/15 cubes, <10 min)
   - Validate safety (zero collisions)
   - Validate smoothness (no oscillations)

---

## Implementation Order

1. **Phase 0**: Research (1 day)
2. **Phase 1**: Design & Contracts (1 day)
3. **Phase 2**: Task Generation (`/speckit.tasks`) (automated)
4. **Phase 3**: Implementation (`/speckit.implement`) (4-5 days)
   - Day 1: Fuzzy variables + membership functions
   - Day 2: Fuzzy rules + inference engine
   - Day 3: State machine + integration
   - Day 4: Testing + debugging
   - Day 5: Tuning + validation
5. **Phase 4**: Validation & PR (1 day)

**Total Estimated Time**: 7 days (aligns with TODO.md Phase 3 allocation)

---

## Success Criteria Validation

**From spec.md Success Criteria**:

1. ✅ **Task Completion**: 15/15 cubes in <10 min → Test in simulation with 3+ runs
2. ✅ **Safety**: Zero collisions → Monitor LIDAR data during runs
3. ✅ **Efficiency**: <40s per cube → Log timestamps per cube
4. ✅ **Grasp Success**: >90% → Track grasp attempts vs successes
5. ✅ **Smoothness**: Angular velocity changes <0.5 rad/s² → Log velocity derivatives

**Validation Method**: Automated test suite + manual simulation runs with metrics logging

---

## Dependencies

**Internal** (must be functional):
- ✅ `src/perception/lidar_processor.py` - LIDAR obstacle detection (Phase 2)
- ✅ `src/perception/cube_detector.py` - Camera cube detection (Phase 2)
- ✅ `src/robot/base.py` - Base movement control (Phase 1)
- ✅ `src/robot/arm.py` - Arm positioning (Phase 1)
- ✅ `src/robot/gripper.py` - Grasp/release (Phase 1)

**External**:
- scikit-fuzzy 0.4.2+ (to be installed)
- NumPy 2.2+ (already installed)
- PyYAML (for configuration files)

**Action**: Install scikit-fuzzy and PyYAML in venv before implementation

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Fuzzy rules produce oscillatory control | Medium | High | Extensive tuning with simulation, overlap membership functions |
| State transitions too frequent (chattering) | Medium | Medium | Add hysteresis to transition conditions, minimum dwell time |
| Inference too slow (>10ms) | Low | High | Profile code, optimize rule evaluation, consider rule reduction |
| Integration issues with perception | Low | Medium | Well-defined interfaces (contracts/), integration tests |
| Grasp success <90% | Medium | High | Tune ALIGN state precision, add retry logic in RECOVERY |

---

## Next Steps

1. ✅ Complete this plan.md
2. ⏳ Generate research.md (Phase 0)
3. ⏳ Generate data-model.md, contracts/, quickstart.md (Phase 1)
4. ⏳ Run `/speckit.tasks` to generate tasks.md
5. ⏳ Run `/speckit.implement` to execute implementation
6. ⏳ Validate and create PR

**Command to continue**: `/speckit.tasks` (after research and design phases complete)

