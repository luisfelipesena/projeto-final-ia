# Implementation Plan: Webots Environment Setup and Validation

**Branch**: `001-webots-setup` | **Date**: 2025-11-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-webots-setup/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Establish functional Webots R2023b simulation environment with Python 3.8+ integration, enabling development of autonomous YouBot system. Setup includes environment installation, dependency configuration, sensor validation (LIDAR + RGB camera), and automated testing infrastructure. This foundational phase blocks all subsequent development phases.

## Technical Context

**Language/Version**: Python 3.8+ (requirement for Webots R2023b controller compatibility)
**Primary Dependencies**: Webots R2023b simulator, pytest (testing), numpy/scipy (sensor data processing)
**Storage**: File-based (world files .wbt, controller scripts, test logs)
**Testing**: pytest for automated validation tests
**Target Platform**: macOS (Intel/Apple Silicon primary), Linux Ubuntu 22.04+ (secondary)
**Project Type**: Single project (robotics simulation environment setup)
**Performance Goals**: Simulation load time <30s, sensor data availability <1s from start, setup completion <30min
**Constraints**: Webots R2023b API fixed version (no modifications), supervisor.py immutable, LIDAR 512 points, camera ≥640x480
**Scale/Scope**: Phase 1.1 of 8-phase project, 15 cubes spawn validation, 2-device reproducibility target

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Fundamentação Científica (Principle I)
- **Status**: PASS
- **Evidence**: References Michel (2004) for Webots, Bischoff et al. (2011) for YouBot specs
- **Action**: Document Webots R2023b version selection in DECISIONS.md (DECISÃO 005)

### ✅ Rastreabilidade Total (Principle II)
- **Status**: PASS
- **Evidence**: Setup decisions will be tracked in DECISIONS.md, docs/environment.md captures configuration
- **Action**: Update DECISIONS.md before implementation

### ✅ Desenvolvimento Incremental (Principle III)
- **Status**: PASS
- **Evidence**: This is Phase 1.1 - first phase execution, blocks all subsequent phases
- **Action**: Mark TODO.md Phase 1.1 items upon completion

### ✅ Qualidade Senior (Principle IV)
- **Status**: PASS
- **Evidence**: Automated pytest tests (FR-012), documentation requirements (FR-013, FR-014), >80% test coverage target
- **Action**: Implement test suite in tests/test_webots_setup.py

### ✅ Restrições Disciplinares (Principle V)
- **Status**: PASS
- **Evidence**: FR-011 explicitly prevents supervisor.py modification, no GPS usage required
- **Constraints**:
  - ❌ NO modification to supervisor.py
  - ❌ NO code in presentation video
  - ❌ NO GPS sensor usage
  - ✅ Document all decisions with citations

### Constitution Compliance: ✅ ALL GATES PASSED

## Project Structure

### Documentation (this feature)

```text
specs/001-webots-setup/
├── spec.md              # Feature specification (already created)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (next step)
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (validation test specifications)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
# Existing structure (DO NOT MODIFY)
IA_20252/
├── controllers/
│   ├── youbot/          # Robot controller scripts (Python)
│   └── supervisor/      # Supervisor script (IMMUTABLE per FR-011)
├── worlds/
│   └── IA_20252.wbt     # World file (to be validated)
└── libraries/           # Webots libraries

# New structure for this feature
tests/
├── test_webots_setup.py # Automated validation tests (FR-012)
└── fixtures/            # Test data and helper functions

docs/
├── environment.md       # Environment configuration details (FR-014)
└── setup/               # Setup guides and troubleshooting

# Project root
├── venv/                # Python virtual environment (FR-003)
├── requirements.txt     # Already exists with dependencies
├── README.md            # Setup instructions (FR-013)
├── DECISIONS.md         # Decision log (existing, to be updated)
└── logs/                # Test and validation logs
```

**Structure Decision**: Using single-project structure with existing IA_20252/ layout preserved. Setup phase creates documentation (docs/), testing infrastructure (tests/), and Python environment (venv/) without modifying Webots-provided files. This maintains compliance with Principle V (no supervisor.py modification) while enabling Phase 2+ development.

## Complexity Tracking

> **No constitution violations - this section is not applicable**

All constitution checks passed. No complexity violations to justify.

---

## Implementation Plan Summary

### Phase 0: Research ✅ COMPLETE

**Output**: `research.md` with 4 major research areas:
1. Webots R2023b installation best practices (official installer approach)
2. Python-Webots integration strategy (system Python + venv hybrid)
3. Automated testing framework selection (pytest multi-layer)
4. Sensor validation patterns (LIDAR 512-point, Camera 128x128 BGRA)

**Key Decisions Identified**:
- DECISÃO 005: Webots R2023b installation method (official DMG/DEB)
- DECISÃO 006: Python integration strategy (system + venv hybrid)
- DECISÃO 007: Testing framework (pytest with markers)
- DECISÃO 008: Sensor validation approach (multi-stage validation)

### Phase 1: Design & Contracts ✅ COMPLETE

**Outputs**:
1. `data-model.md` - 6 entity definitions:
   - WebotsInstallation (version, paths, validation)
   - PythonEnvironment (venv, dependencies, PYTHONPATH)
   - WorldFileConfiguration (world file, supervisor immutability)
   - SensorConfiguration (LIDAR + Camera specs)
   - ValidationTest (individual test results)
   - SetupValidationReport (aggregated results)

2. `quickstart.md` - 5-step setup guide:
   - Step 1: Install Webots R2023b (10 min)
   - Step 2: Configure Python 3.8+ (5 min)
   - Step 3: Configure PYTHONPATH (2 min)
   - Step 4: Validate world file (2 min)
   - Step 5: Run validation tests (5 min)
   - Includes troubleshooting section

3. `contracts/test_specifications.md` - Test contracts:
   - 4 Phase 1.1 tests (installation + environment validation)
   - 3 Phase 2 tests (sensor validation, deferred)
   - Test fixtures and execution strategy
   - CI/CD integration patterns

4. Agent context updated: CLAUDE.md now includes Python 3.8+, Webots R2023b, pytest

### Phase 2: Tasks ⏳ NEXT STEP

**Command**: `/speckit.tasks`

**Expected Output**: `tasks.md` with implementation checklist
- Task 1: Install Webots R2023b
- Task 2: Configure Python environment
- Task 3: Implement validation tests
- Task 4: Document setup process
- Task 5: Validate on second machine (reproducibility)

---

## Next Actions

1. **Execute `/speckit.tasks`**:
   - Generate granular task breakdown in `tasks.md`
   - Define dependencies between tasks
   - Assign priority levels

2. **Execute `/speckit.implement`**:
   - Follow tasks.md checklist
   - Implement test suite (tests/test_webots_setup.py)
   - Create documentation (README.md, docs/environment.md)
   - Update DECISIONS.md with 4 decisions

3. **Validation**:
   - Run pytest suite (target: 4/4 tests pass)
   - Verify world loads <30s
   - Test on second machine (SC-008)
   - Mark TODO.md Phase 1.1 complete

---

## Artifacts Generated

**Planning Phase**:
- ✅ specs/001-webots-setup/spec.md (specification)
- ✅ specs/001-webots-setup/plan.md (this file)
- ✅ specs/001-webots-setup/research.md (research findings)
- ✅ specs/001-webots-setup/data-model.md (entity definitions)
- ✅ specs/001-webots-setup/quickstart.md (setup guide)
- ✅ specs/001-webots-setup/contracts/test_specifications.md (test contracts)

**Implementation Phase** (Next):
- ⏳ specs/001-webots-setup/tasks.md (implementation checklist)
- ⏳ tests/test_webots_setup.py (validation suite)
- ⏳ docs/environment.md (configuration capture)
- ⏳ DECISIONS.md (4 new decisions: 005-008)
- ⏳ README.md updates (setup instructions)

---

## Constitution Re-Check Post-Design

### ✅ Fundamentação Científica (Principle I)
- **Status**: PASS
- **Evidence**: Research.md documents scientific basis for all decisions (Michel 2004, Webots docs, pytest patterns)
- **Action**: DECISIONS.md will cite research findings

### ✅ Rastreabilidade Total (Principle II)
- **Status**: PASS
- **Evidence**: All design artifacts create audit trail (research → data-model → contracts → tasks)
- **Action**: Implementation will update DECISIONS.md before execution

### ✅ Desenvolvimento Incremental (Principle III)
- **Status**: PASS
- **Evidence**: Phase 1.1 scope maintained, no Phase 2 dependencies
- **Action**: Mark TODO.md Phase 1.1 complete after validation

### ✅ Qualidade Senior (Principle IV)
- **Status**: PASS
- **Evidence**: Comprehensive test suite designed (4 tests + 3 deferred), multi-layer validation
- **Action**: Achieve >80% coverage (currently: validation code only)

### ✅ Restrições Disciplinares (Principle V)
- **Status**: PASS
- **Evidence**: Constitution compliance test added (test_supervisor_file_not_modified)
- **Constraints Enforced**:
  - ❌ NO supervisor.py modification (test will fail if violated)
  - ❌ NO code in video (setup phase only, no presentation yet)
  - ❌ NO GPS usage (not applicable to setup phase)

### Constitution Compliance Post-Design: ✅ ALL GATES PASSED

---

**Plan Status**: Phase 0 & 1 complete, ready for `/speckit.tasks`

**Branch**: 001-webots-setup
**Last Updated**: 2025-11-18
