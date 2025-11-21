# Implementation Plan: Fuzzy Logic Control System

**Branch**: `004-fuzzy-control` | **Date**: 2025-11-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-fuzzy-control/spec.md`

## Summary

Implement Mamdani fuzzy inference system for YouBot autonomous navigation and manipulation control. System uses fuzzy logic to translate sensor inputs (obstacle distances, cube positions) into smooth velocity commands and behavioral decisions. Integrated with 6-state machine (SEARCHING, APPROACHING, GRASPING, NAVIGATING_TO_BOX, DEPOSITING, AVOIDING) to coordinate complete cube collection cycles. Primary goal: reactive collision-free navigation with smooth trajectories and intelligent cube approach behavior.

**Technical Approach**: scikit-fuzzy library for Mamdani inference, ~20-30 rules with triangular/trapezoidal membership functions, 20Hz control loop with <50ms decision cycle, mock perception interface enabling independent development before Phase 2 integration.

## Technical Context

**Language/Version**: Python 3.8+ (compatible with Webots R2023b controller API)
**Primary Dependencies**:
- scikit-fuzzy 0.4.2+ (Mamdani fuzzy inference)
- numpy >=1.24.0 (numerical operations, array processing)
- matplotlib >=3.7.0 (membership function visualization, debugging plots)

**Storage**: File-based logging (logs/fuzzy_decisions.log, logs/state_transitions.log)
**Testing**: pytest >=7.4.0 (unit tests for fuzzy rules, state machine transitions)
**Target Platform**: Webots R2023b simulation environment (Linux/macOS/Windows)
**Project Type**: Single project - robot control module integrated into existing codebase
**Performance Goals**:
- Decision cycle: <50ms (20Hz control loop)
- State transition latency: <10ms
- Zero computational overhead causing missed sensor updates

**Constraints**:
- Real-time operation: Cannot block robot control loop
- Safety-first: Obstacle avoidance rules must always fire before approach/navigation
- Deterministic: Same inputs always produce same outputs (no randomness in fuzzy inference)
- Memory: <50MB for rule evaluation and state tracking

**Scale/Scope**:
- 6 states in state machine
- 20-30 fuzzy rules
- 6 input linguistic variables (distance_to_obstacle, angle_to_obstacle, distance_to_cube, angle_to_cube, cube_detected, holding_cube)
- 3 output linguistic variables (linear_velocity, angular_velocity, action)
- Each linguistic variable: 3-5 membership functions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Fundamentação Científica (Principle I)

**Status**: PASS - All design decisions have scientific foundation

- **Fuzzy Logic Choice**: Zadeh (1965) fuzzy sets theory, Mamdani & Assilian (1975) fuzzy controller design
- **Reactive Navigation**: Saffiotti (1997) fuzzy logic for mobile robot navigation
- **State Machine Pattern**: Standard robotics paradigm, Thrun et al. (2005) probabilistic robotics
- **Membership Functions**: Industry-standard triangular/trapezoidal shapes (Mamdani 1975)

**References to cite in presentation**:
1. Zadeh (1965) - Fuzzy Sets
2. Mamdani & Assilian (1975) - Fuzzy Controller Application
3. Saffiotti (1997) - Fuzzy Navigation
4. Antonelli et al. (2007) - Path Tracking with Fuzzy

### ✅ Rastreabilidade (Principle II)

**Status**: PASS - Documentation strategy defined

- **DECISÃO 018**: Fuzzy controller architecture (Mamdani vs Sugeno, rule count, membership function types)
- **DECISÃO 019**: State machine design (states, transitions, override logic)
- **DECISÃO 020**: Integration with perception (mock interface design)
- All decisions will be documented in DECISIONS.md BEFORE implementation
- Git commits reference decisions: `feat(fuzzy): implement obstacle avoidance rules (DECISÃO 018)`

### ✅ Desenvolvimento Incremental (Principle III)

**Status**: PASS - Follows Phase 3 of TODO.md

- **Current Phase**: Fase 3 - Controle Fuzzy (7 days planned)
- **Previous Phase**: Fase 2 - Percepção RNA (infrastructure complete, training pending)
- **Dependencies**: Can proceed with mock perception data
- **Deliverables**: Fuzzy controller module, state machine, unit tests, DECISIONS.md updates

**Phase 2 Status**: 🟡 Infrastructure complete, real perception training deferred to after Phase 3
- Allows parallel development: Fuzzy control with mocks now, real integration in Phase 6

### ✅ Qualidade Senior (Principle IV)

**Status**: PASS - Modular architecture defined

**Module Structure**:
```python
src/control/
├── __init__.py
├── fuzzy_controller.py      # FuzzyController class, Mamdani inference
├── fuzzy_rules.py           # Rules database, membership functions
├── state_machine.py          # StateMachine class, RobotState enum
└── robot_controller.py       # Integration: fuzzy + state machine

tests/control/
├── __init__.py
├── test_fuzzy_controller.py  # Unit tests for rules, inference
├── test_state_machine.py     # Unit tests for transitions
└── test_integration.py       # Integration tests with mock perception
```

**Testing Strategy**:
- Unit tests: Each fuzzy rule individually testable
- Integration tests: Full cycle with mock sensors
- Target: >80% coverage
- Test-driven: Write tests before implementation

### ✅ Restrições Disciplinares (Principle V)

**Status**: PASS - No violations

- ✅ No modifications to supervisor.py
- ✅ No code will appear in video (only diagrams, plots, behavior visualizations)
- ✅ Fuzzy logic is **mandatory requirement** for discipline
- ✅ All design justified by scientific papers

### ✅ Workflow SpecKit (Principle VI)

**Status**: PASS - Following workflow

1. ✅ `/speckit.specify` - spec.md created with 4 user stories, 22 FRs, 8 success criteria
2. ⏭️ `/speckit.plan` - THIS DOCUMENT (in progress)
3. 📋 `/speckit.tasks` - Next: Generate tasks.md from plan
4. 🔨 `/speckit.implement` - Then: Execute tasks with DECISIONS.md updates
5. ✅ `/speckit.analyze` - Final: Validate consistency

**Learning from Previous Phases**:
- Phase 1 (Webots Setup): Learned controller API patterns, testing infrastructure
- Phase 2 (Perception): Learned architecture separation (models/ vs inference/), data flow design
- Will read DECISIONS.md 001-017 before starting implementation

### ✅ Post-Design Re-Check Complete

After Phase 1 (design artifacts generated), verified:
- [x] Fuzzy rules coverage complete (all safety scenarios) - Rule structure supports weighted priorities (safety=8.0-10.0), category='safety' filter, 15-25 rules planned
- [x] State machine handles all edge cases from spec - AVOIDING override, timeouts, grasp retry, cube detection lost, corner trap transitions all defined
- [x] Mock perception interface matches Phase 2 contract - ObstacleMap, CubeObservation, PerceptionData match src/perception/ interfaces
- [x] Performance constraints achievable (<50ms decision cycle) - Research shows 10-30ms typical with caching, well under 50ms target

## Project Structure

### Documentation (this feature)

```text
specs/004-fuzzy-control/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file (in progress)
├── research.md          # Phase 0: Fuzzy logic best practices, rule tuning strategies
├── data-model.md        # Phase 1: FuzzyController, StateMachine, RobotState data structures
├── quickstart.md        # Phase 1: How to use fuzzy controller, add rules, test
├── contracts/           # Phase 1: Perception mock interface, control command outputs
│   ├── fuzzy_controller.py   # FuzzyController interface
│   ├── state_machine.py      # StateMachine interface
│   └── perception_mock.py    # Mock perception for testing
└── tasks.md             # Phase 2: NOT created by /speckit.plan, created by /speckit.tasks
```

### Source Code (repository root)

```text
src/control/                          # NEW: Fuzzy control module
├── __init__.py                       # Exports: FuzzyController, StateMachine
├── fuzzy_controller.py               # Mamdani inference engine
├── fuzzy_rules.py                    # Rules database, membership functions
├── state_machine.py                  # State transitions, RobotState
└── robot_controller.py               # Integration layer

src/perception/                       # EXISTING: Phase 2 infrastructure
├── __init__.py
├── lidar_processor.py                # Mock-able: ObstacleMap interface
└── cube_detector.py                  # Mock-able: CubeObservation interface

tests/control/                        # NEW: Fuzzy control tests
├── __init__.py
├── test_fuzzy_controller.py
├── test_state_machine.py
├── test_integration.py               # Uses mock perception
└── fixtures/
    └── perception_mock.py            # Mock implementations

docs/                                 # NEW: Fuzzy documentation
├── fuzzy_membership_functions.md    # Plots and ranges for each variable
├── fuzzy_rules.md                    # Complete rules table
└── state_machine_diagram.png        # State transition diagram

logs/                                 # EXISTING: Logging directory
├── fuzzy_decisions.log               # NEW: Fuzzy inference logs
└── state_transitions.log             # NEW: State machine logs
```

**Structure Decision**: Single project architecture (Option 1) integrated into existing `src/` structure. Fuzzy control is a new top-level module `src/control/` that depends on perception interfaces from `src/perception/`. Uses mock implementations during development (Phase 3) and real perception integration deferred to Phase 6 (Integration phase per TODO.md).

**Rationale**: Modular separation allows independent development and testing. Mock-based testing enables Phase 3 completion without waiting for Phase 2 neural network training. Aligns with constitution Principle III (incremental development) and TODO.md Phase 3 timeline.

## Complexity Tracking

> **Not Required**: No constitution violations detected.

All gates pass:
- ✅ Scientific foundation for fuzzy logic (Zadeh, Mamdani, Saffiotti)
- ✅ Documentation strategy (DECISÃO 018-020 to be created)
- ✅ Incremental development (Phase 3 follows Phase 2 infrastructure)
- ✅ Modular architecture (src/control/ separation)
- ✅ No disciplinary violations
- ✅ SpecKit workflow followed

No complexity justification needed.

---

**Phase 0 (Research) begins below**: Resolve unknowns, research best practices
