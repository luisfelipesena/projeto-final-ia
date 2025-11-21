# Feature Specification: Webots Environment Setup and Validation

**Feature Branch**: `001-webots-setup`
**Created**: 2025-11-18
**Status**: Draft
**Input**: Setup do Webots conforme Fase 1.1 - Validar Webots R2023b funcional com Python 3.8+, simulação IA_20252.wbt e 15 cubos

## User Scenarios & Testing

### User Story 1 - Initial Environment Setup (Priority: P0 - Blocker)

A developer needs to set up the Webots simulation environment on their machine to begin development of the autonomous YouBot system. This is the foundational setup that enables all subsequent development work.

**Why this priority**: This is a blocking requirement - no other development can proceed without a functional Webots environment. It establishes the baseline for the entire project.

**Independent Test**: Can be fully tested by installing Webots, running the provided world file (IA_20252.wbt), and verifying that the simulation loads without errors and displays 15 colored cubes.

**Acceptance Scenarios**:

1. **Given** a fresh development machine with no Webots installation, **When** the developer follows setup instructions, **Then** Webots R2023b is installed and launches successfully
2. **Given** Webots is installed, **When** the developer opens the IA_20252.wbt world file, **Then** the simulation loads within 30 seconds without errors
3. **Given** the simulation is running, **When** the developer observes the arena, **Then** exactly 15 cubes are visible with colors (green, blue, or red) distributed randomly
4. **Given** the simulation is running, **When** the developer checks the Webots console, **Then** no error messages appear and the supervisor reports successful spawn completion

---

### User Story 2 - Python Environment Configuration (Priority: P0 - Blocker)

A developer needs to configure Python environment with correct version and dependencies to enable controller development for the YouBot robot.

**Why this priority**: Python controllers are mandatory for the project. Without proper Python setup, no robot control logic can be developed or tested.

**Independent Test**: Can be tested independently by verifying Python version, creating a virtual environment, installing dependencies, and running a simple test script that imports the Webots controller module.

**Acceptance Scenarios**:

1. **Given** Python 3.8+ is installed on the system, **When** the developer checks Python version, **Then** the version number is 3.8 or higher
2. **Given** the project directory exists, **When** the developer creates a virtual environment, **Then** a `venv/` directory is created and can be activated
3. **Given** the virtual environment is activated, **When** the developer installs requirements.txt, **Then** all dependencies install without errors
4. **Given** dependencies are installed, **When** a Webots controller script runs, **Then** the `controller` module is accessible and functional

---

### User Story 3 - Sensor Functionality Validation (Priority: P1 - Critical)

A developer needs to verify that the YouBot's sensors (LIDAR and RGB camera) are functional and returning valid data to enable perception system development.

**Why this priority**: Sensor data is critical for obstacle detection and cube identification. Validating sensors early prevents wasted effort on algorithms that can't access data.

**Independent Test**: Can be tested by running a simple controller script that reads from LIDAR and camera, prints sample data, and verifies data formats match expected specifications.

**Acceptance Scenarios**:

1. **Given** the simulation is running with YouBot active, **When** LIDAR is queried, **Then** it returns an array of range measurements (e.g., 512 points with values in meters)
2. **Given** the simulation is running, **When** the camera is queried, **Then** it returns image data with expected resolution (128x128 pixels in BGRA format)
3. **Given** sensor data is being collected, **When** the developer inspects LIDAR ranges, **Then** values are numeric and represent plausible distances (0-5 meters, with 'inf' for no obstacle)
4. **Given** sensor data is being collected, **When** the developer captures a camera frame, **Then** colored cubes are visible in the image data

---

### User Story 4 - Automated Setup Validation (Priority: P2 - Important)

A developer needs automated tests to verify that the environment setup is complete and correct, enabling quick validation on new machines or after updates.

**Why this priority**: Automated validation reduces manual verification time and catches environment issues early. Important for reproducibility across different developers and machines.

**Independent Test**: Can be tested by running a pytest suite that checks Webots installation, Python version, world file existence, and basic simulation functionality.

**Acceptance Scenarios**:

1. **Given** the setup is complete, **When** the developer runs `pytest tests/test_webots_setup.py`, **Then** all tests pass (4/4)
2. **Given** Webots is not installed, **When** the validation tests run, **Then** the Webots installation test fails with a clear error message
3. **Given** Python version is below 3.8, **When** the validation tests run, **Then** the Python version test fails and reports the installed version
4. **Given** the world file is missing, **When** the validation tests run, **Then** the file existence test fails and reports the expected path

---

### Edge Cases

- **What happens when Webots is a different version (not R2023b)?** System should detect version mismatch and warn user, documenting potential compatibility issues
- **How does system handle missing or corrupted world file?** Webots will fail to load with an error; setup validation should catch this before manual testing
- **What if Python controller module is not accessible?** Simulation will fail to start robot controller; environment path configuration needed
- **How to handle systems with insufficient hardware (low RAM, old GPU)?** Simulation may run slowly or crash; document minimum hardware requirements and provide low-quality graphics fallback
- **What if supervisor spawns fewer than 15 cubes (collision issues)?** Supervisor logs warning about failed spawns; this is acceptable if total spawned ≥ 10, but should be documented

## Requirements

### Functional Requirements

- **FR-001**: System MUST install Webots R2023b on the development machine (macOS or Linux Ubuntu 22.04+)
- **FR-002**: System MUST verify Python version is 3.8 or higher
- **FR-003**: System MUST create and configure a Python virtual environment in the project directory
- **FR-004**: System MUST install all dependencies listed in requirements.txt without errors
- **FR-005**: System MUST open and load the IA_20252.wbt world file within 30 seconds
- **FR-006**: System MUST spawn exactly 15 colored cubes in the arena (random distribution of green, blue, red)
- **FR-007**: System MUST position the YouBot robot at the initial starting location
- **FR-008**: System MUST display the three colored deposit boxes (green, blue, red) in their designated positions
- **FR-009**: System MUST enable and provide data from the YouBot's LIDAR sensor
- **FR-010**: System MUST enable and provide image data from the YouBot's RGB camera
- **FR-011**: System MUST execute the supervisor script to spawn cubes without modifications to supervisor.py
- **FR-012**: System MUST provide automated validation tests that verify installation completeness
- **FR-013**: System MUST document the setup process in README.md with step-by-step instructions
- **FR-014**: System MUST record environment configuration details in docs/environment.md
- **FR-015**: System MUST document setup decisions in DECISIONS.md (DECISÃO 005: Webots R2023b selection)

### Key Entities

- **Webots Environment**: The simulation software (version R2023b), provides 3D physics engine, sensor simulation, and robot control interface
- **Python Virtual Environment**: Isolated Python environment (venv/), contains project dependencies and prevents system-wide conflicts
- **World File**: IA_20252.wbt file, defines arena layout, obstacles (wooden boxes), deposit boxes, and YouBot starting position
- **Colored Cubes**: 15 objects with recognition colors (green/blue/red), spawned randomly by supervisor, represent items to be collected
- **YouBot Robot**: Mobile manipulator with LIDAR sensor, RGB camera, omnidirectional base, 5-DOF arm, and parallel gripper
- **Validation Tests**: Automated pytest suite (tests/test_webots_setup.py), verifies environment correctness

## Success Criteria

### Measurable Outcomes

- **SC-001**: Developer can complete entire setup process in under 30 minutes (excluding download time)
- **SC-002**: Webots simulation loads the world file in under 30 seconds on standard development hardware
- **SC-003**: Automated validation tests achieve 100% pass rate (4/4 tests) when environment is correctly configured
- **SC-004**: Supervisor successfully spawns at least 14 out of 15 cubes in 95% of simulation starts (allowing for rare collision edge cases)
- **SC-005**: LIDAR sensor returns valid range data (array of 512 points) within 1 second of simulation start
- **SC-006**: RGB camera returns valid image frames at expected resolution (128x128 BGRA format) within 1 second of simulation start
- **SC-007**: Setup documentation enables a new developer to configure the environment independently without additional support
- **SC-008**: Environment setup is reproducible across different machines (verified on at least 2 different development machines)

## Assumptions

- Development will primarily occur on macOS with Intel/Apple Silicon; Linux (Ubuntu 22.04+) compatibility will be documented but not required for all developers
- Developers have basic familiarity with terminal/command line operations
- Internet connectivity is available for downloading Webots and pip packages
- Standard development machine has minimum 8GB RAM and dedicated graphics (or integrated GPU with sufficient VRAM)
- Webots R2023b API is stable and will not change during the project timeline (until 2026-01-06)
- The provided IA_20252/ directory structure and code base are correct and will not be modified (especially supervisor.py)
- Git is already installed and configured on the development machine
- Developers will use the provided Python code structure (controllers in IA_20252/controllers/)

## Dependencies

- **External**: Webots R2023b installer (download from cyberbotics.com)
- **External**: Python 3.8+ interpreter installed on system
- **External**: pip package manager for Python dependency installation
- **Internal**: requirements.txt file with pinned dependency versions
- **Internal**: IA_20252/ directory structure with worlds/, controllers/, and libraries/
- **Internal**: .specify/memory/constitution.md for project principles and constraints
- **Internal**: TODO.md Phase 1.1 checklist items

## Constraints

- **MUST NOT modify supervisor.py** (constitution.md Principle V) - any changes to cube spawning logic will result in point deduction
- **MUST use Webots R2023b specifically** - other versions may have API incompatibilities or different behavior
- **MUST NOT use GPS sensor** - navigation must be based solely on LIDAR and camera (constitution requirement)
- **MUST document all setup decisions in DECISIONS.md** (constitution.md Principle II - Rastreabilidade Total)
- **MUST follow 8-phase incremental development** (constitution.md Principle III) - Phase 1 must complete before Phase 2
- **Setup must be testable** - all validation criteria must be verifiable through automated tests or clear manual steps

## Out of Scope

This specification does NOT cover:

- Development of robot control logic or autonomous behaviors (covered in later phases)
- Implementation of neural networks for LIDAR processing (Phase 2)
- Fuzzy logic controller design (Phase 3)
- Path planning or navigation algorithms (Phase 4)
- Grasping and manipulation sequences (Phase 5)
- Integration of perception and control systems (Phase 6)
- Performance optimization and tuning (Phase 7)
- Video presentation creation (Phase 8)
- Installation of development tools beyond Webots and Python (e.g., IDEs, git clients)
- Configuration of hardware-accelerated graphics or CUDA (optional, not required)
- Troubleshooting of Webots internal bugs or rendering issues (defer to Webots documentation/support)

## Clarifications

### Session 2025-11-18

- **Camera Resolution Correction**: Initial spec mentioned 640x480 pixels, but actual world file configuration (IA_20252.wbt) uses 128x128 resolution in BGRA format. Updated User Story 3 and SC-006 to reflect accurate resolution based on research findings from world file analysis. Lower resolution is intentional for performance optimization during proof-of-concept development.

## References

This specification is based on:

- **TODO.md Phase 1.1**: "Setup do Webots" checklist items
- **constitution.md**: Core principles (especially Principles I-V)
- **CLAUDE.md**: Project context and technical requirements
- **Michel (2004)**: "Webots: Professional Mobile Robot Simulation" - foundational paper on Webots simulator
- **Bischoff et al. (2011)**: "KUKA youBot - a mobile manipulator for research and education" - YouBot specifications

Scientific basis for decisions will be documented in DECISIONS.md following project methodology.
