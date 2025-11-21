# Feature Specification: Sensor Exploration and Control Validation

**Feature Branch**: `002-sensor-exploration`
**Created**: 2025-11-21
**Status**: Draft
**Input**: User description: "Fase 1.2-1.3: Sensor Exploration - Test YouBot controls and analyze LIDAR/camera sensor data"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Base Movement Control Validation (Priority: P1)

As a robotics engineer, I need to validate that the YouBot's base responds correctly to movement commands so that I can build reliable navigation behaviors in future phases.

**Why this priority**: This is foundational - without validated base control, no navigation or autonomous behavior is possible. All future phases depend on this working correctly.

**Independent Test**: Can be fully tested by executing directional movement commands (forward, backward, strafe left/right, rotate clockwise/counterclockwise) and observing the robot's physical response in Webots. Delivers confidence that the base control API works as expected.

**Acceptance Scenarios**:

1. **Given** the YouBot is stationary in the arena, **When** a forward velocity command is sent, **Then** the robot moves forward in a straight line at the commanded speed
2. **Given** the YouBot is stationary, **When** a backward velocity command is sent, **Then** the robot moves backward in a straight line
3. **Given** the YouBot is stationary, **When** a strafe left command is sent, **Then** the robot moves laterally to the left without rotating
4. **Given** the YouBot is stationary, **When** a strafe right command is sent, **Then** the robot moves laterally to the right without rotating
5. **Given** the YouBot is stationary, **When** a rotate clockwise command is sent, **Then** the robot rotates in place clockwise
6. **Given** the YouBot is stationary, **When** a rotate counterclockwise command is sent, **Then** the robot rotates in place counterclockwise
7. **Given** the YouBot is moving, **When** a stop command is sent, **Then** the robot comes to a complete stop
8. **Given** movement commands are sent, **When** movement limits are exceeded, **Then** the system records and documents these limits

---

### User Story 2 - Arm and Gripper Control Validation (Priority: P1)

As a robotics engineer, I need to validate that the YouBot's arm and gripper respond correctly to positioning and grasping commands so that I can implement reliable object manipulation in future phases.

**Why this priority**: Manipulation is core to the project's mission (picking and placing cubes). Without validated arm/gripper control, the autonomous collection task cannot be completed.

**Independent Test**: Can be fully tested by commanding preset arm positions (height adjustments, orientation changes) and gripper actions (open, close) while observing the physical response in Webots. Delivers confidence that manipulation commands work reliably.

**Acceptance Scenarios**:

1. **Given** the arm is in rest position, **When** a set_height command is sent, **Then** the arm moves to the specified height
2. **Given** the arm is at a specific height, **When** a set_orientation command is sent, **Then** the arm rotates to the specified orientation
3. **Given** the gripper is open, **When** a grip command is sent, **Then** the gripper closes
4. **Given** the gripper is closed, **When** a release command is sent, **Then** the gripper opens
5. **Given** the arm is moving, **When** movement limits are reached, **Then** the system documents these joint limits
6. **Given** arm commands are sent, **When** the arm reaches each preset position, **Then** the final position matches the commanded position within acceptable tolerance

---

### User Story 3 - LIDAR Data Analysis (Priority: P2)

As a robotics engineer, I need to understand the LIDAR sensor's raw data format, range capabilities, and field of view so that I can design effective obstacle detection algorithms for Phase 2.

**Why this priority**: LIDAR is the primary sensor for obstacle avoidance, which is mandatory for autonomous navigation. Understanding the data structure is prerequisite for implementing neural network processing in Phase 2.

**Independent Test**: Can be fully tested by capturing LIDAR data in various scenarios (open space, near obstacles, different distances) and analyzing the output format, number of measurement points, angular resolution, and maximum range. Delivers documented sensor characteristics needed for algorithm design.

**Acceptance Scenarios**:

1. **Given** the YouBot is in the arena, **When** LIDAR data is captured, **Then** the raw range_image data is successfully read and contains numeric distance values
2. **Given** LIDAR data is captured, **When** the data format is analyzed, **Then** the number of measurement points, angular resolution, and field of view are documented
3. **Given** the YouBot is at varying distances from obstacles, **When** LIDAR measurements are taken, **Then** the maximum and minimum detection ranges are documented
4. **Given** LIDAR data is collected, **When** the data is visualized using matplotlib, **Then** a polar plot clearly shows the arena boundaries and obstacle positions
5. **Given** obstacles are placed at known positions, **When** LIDAR data is analyzed, **Then** the measured distances match the actual distances within sensor accuracy
6. **Given** multiple LIDAR scans are captured, **When** visualizations are generated, **Then** obstacles are consistently identifiable in the visualization

---

### User Story 4 - Camera RGB Analysis (Priority: P2)

As a robotics engineer, I need to understand the RGB camera's image format, resolution, frame rate, and color detection capabilities so that I can design effective cube identification algorithms for Phase 2.

**Why this priority**: The RGB camera is mandatory for identifying cube colors (green, blue, red), which is essential for the sorting task. Understanding the camera characteristics is prerequisite for implementing CNN-based detection in Phase 2.

**Independent Test**: Can be fully tested by capturing camera frames under various conditions (different lighting, cube positions, distances) and analyzing image resolution, FPS, color accuracy, and simple color threshold performance. Delivers documented camera characteristics and baseline color detection feasibility.

**Acceptance Scenarios**:

1. **Given** the YouBot is in the arena, **When** camera frames are captured, **Then** the frames are successfully read with consistent resolution
2. **Given** camera frames are being captured, **When** the frame rate is measured, **Then** the FPS is documented for future processing pipeline design
3. **Given** green, blue, and red cubes are visible, **When** frames are captured, **Then** example images are saved for each cube color
4. **Given** a simple RGB threshold is applied, **When** cubes are detected in test images, **Then** the threshold successfully distinguishes green, blue, and red cubes with documented accuracy
5. **Given** cubes at varying distances, **When** images are captured, **Then** the effective detection range is documented
6. **Given** various lighting conditions in the arena, **When** frames are analyzed, **Then** color detection robustness is documented for future algorithm design

---

### User Story 5 - Arena Mapping (Priority: P3)

As a robotics engineer, I need to document the arena's physical dimensions, deposit box locations, and typical obstacle distribution so that I can design navigation strategies and test coverage in future phases.

**Why this priority**: Understanding the operational environment helps optimize path planning and ensures the robot can navigate the full arena. This is supporting information for Phase 4 (Navigation) but not immediately blocking other work.

**Independent Test**: Can be fully tested by manually measuring arena dimensions in Webots, recording deposit box coordinates, and documenting obstacle positions. Delivers a schematic map that can be referenced when designing navigation algorithms.

**Acceptance Scenarios**:

1. **Given** the arena is loaded in Webots, **When** measurements are taken, **Then** the total arena dimensions (length, width) are documented in meters
2. **Given** the arena contains deposit boxes, **When** their positions are recorded, **Then** the approximate coordinates of green, blue, and red deposit boxes are documented
3. **Given** wooden box obstacles are present, **When** their distribution is analyzed, **Then** typical obstacle positions and spacing are documented
4. **Given** spawn coordinates are defined in supervisor.py, **When** the spawn ranges are analyzed, **Then** the cube spawn area boundaries are documented
5. **Given** all measurements are complete, **When** a schematic map is created, **Then** the map shows arena boundaries, deposit boxes, typical obstacles, and spawn zones in a clear visual format

---

### Edge Cases

- What happens when movement commands exceed the YouBot's mechanical limits (e.g., arm joint limits, maximum base velocity)?
- How does the LIDAR sensor behave when no obstacles are in range (e.g., reading infinity or max range)?
- What happens when the camera captures frames with no cubes visible?
- How does color detection perform under edge lighting conditions (very bright or very dark areas)?
- What happens when LIDAR or camera data capture fails or returns corrupted data?
- How does the system handle rapid command changes (e.g., switching from forward to backward immediately)?

## Requirements *(mandatory)*

### Functional Requirements

#### Base Movement Control
- **FR-001**: System MUST successfully execute forward movement commands at specified velocities
- **FR-002**: System MUST successfully execute backward movement commands at specified velocities
- **FR-003**: System MUST successfully execute strafe left and strafe right commands (omnidirectional movement)
- **FR-004**: System MUST successfully execute clockwise and counterclockwise rotation commands
- **FR-005**: System MUST successfully execute stop commands to halt all movement
- **FR-006**: System MUST document maximum and minimum velocity limits for base movement
- **FR-007**: System MUST create automated test script `tests/test_basic_controls.py` that validates all base movement commands

#### Arm and Gripper Control
- **FR-008**: System MUST successfully execute set_height commands to position the arm at specified heights
- **FR-009**: System MUST successfully execute set_orientation commands to rotate the arm to specified orientations
- **FR-010**: System MUST successfully execute grip commands to close the gripper
- **FR-011**: System MUST successfully execute release commands to open the gripper
- **FR-012**: System MUST document joint limits and range of motion for the arm
- **FR-013**: Test script MUST validate all arm positioning and gripper commands

#### LIDAR Sensor Analysis
- **FR-014**: System MUST successfully read LIDAR range_image data from the sensor
- **FR-015**: System MUST document the LIDAR data format including number of measurement points, angular resolution, and field of view
- **FR-016**: System MUST document the maximum and minimum detection ranges of the LIDAR sensor
- **FR-017**: System MUST create polar visualizations of LIDAR scans using matplotlib that clearly show obstacles
- **FR-018**: System MUST identify and mark obstacles in LIDAR visualizations
- **FR-019**: System MUST create Jupyter notebook `notebooks/01_sensor_exploration.ipynb` containing LIDAR analysis and visualizations

#### Camera RGB Analysis
- **FR-020**: System MUST successfully capture RGB camera frames
- **FR-021**: System MUST document camera resolution (width x height in pixels)
- **FR-022**: System MUST document camera frame rate (FPS)
- **FR-023**: System MUST save example images for green, blue, and red cubes
- **FR-024**: System MUST implement and test simple RGB threshold-based color detection for green, blue, and red
- **FR-025**: System MUST document color detection accuracy for the threshold approach
- **FR-026**: Jupyter notebook MUST contain camera analysis, example images, and color detection results

#### Arena Mapping
- **FR-027**: System MUST document arena dimensions (length and width in meters)
- **FR-028**: System MUST document approximate coordinates of green, blue, and red deposit boxes
- **FR-029**: System MUST document typical distribution and positions of wooden box obstacles
- **FR-030**: System MUST create schematic map in `docs/arena_map.md` showing boundaries, deposit boxes, obstacles, and spawn zones

### Key Entities

- **Base Movement Command**: Represents a velocity command for the omnidirectional base (vx: forward/backward, vy: strafe left/right, omega: rotation)
- **Arm Position**: Represents the configuration of the 5-DOF arm (height, orientation, joint angles)
- **Gripper State**: Represents the state of the parallel gripper (open/closed, grip force)
- **LIDAR Scan**: Represents a single 2D LIDAR measurement containing distance values for each angular position within the field of view
- **Camera Frame**: Represents a single RGB image captured from the YouBot's camera (resolution, timestamp, pixel data)
- **Obstacle**: Represents a detected object in the arena identified through LIDAR or camera analysis (position, distance, type)
- **Arena Map**: Represents the documented spatial layout including dimensions, deposit box locations, obstacle positions, and spawn zones

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All base movement commands (forward, backward, strafe left/right, rotate CW/CCW, stop) execute successfully in Webots simulation with observable motion matching the command intent
- **SC-002**: All arm positioning commands (set_height, set_orientation) execute successfully with final positions matching commanded positions within 5% tolerance
- **SC-003**: All gripper commands (grip, release) execute successfully with observable gripper state changes in simulation
- **SC-004**: Automated test script `tests/test_basic_controls.py` passes 100% of test cases validating base, arm, and gripper controls
- **SC-005**: LIDAR sensor data is successfully captured and documented, including data format (number of points, angular resolution, FOV) and range capabilities (max/min distance)
- **SC-006**: LIDAR visualizations clearly show arena boundaries and obstacles with identifiable geometric shapes
- **SC-007**: Camera successfully captures frames with documented resolution and FPS
- **SC-008**: Simple RGB threshold achieves at least 80% accuracy in distinguishing green, blue, and red cubes in test images under standard arena lighting
- **SC-009**: Jupyter notebook `notebooks/01_sensor_exploration.ipynb` contains complete analysis with visualizations for both LIDAR and camera sensors
- **SC-010**: Arena map document `docs/arena_map.md` contains schematic diagram with all required elements (dimensions, deposit boxes, obstacles, spawn zones)
- **SC-011**: All documented movement limits, sensor characteristics, and arena measurements are recorded with sufficient detail to support algorithm design in Phase 2

## Assumptions

- Webots R2023b simulator is installed and configured correctly (validated in Phase 1.1)
- World file `IA_20252.wbt` loads successfully with 15 cubes spawned (validated in Phase 1.1)
- Python venv is active with all required dependencies installed (numpy, matplotlib, scipy, opencv-python)
- YouBot controllers (`youbot.py`, `base.py`, `arm.py`, `gripper.py`) are functional from base code
- LIDAR and camera sensors are enabled in the robot configuration
- Arena lighting is consistent with default Webots environment
- Supervisor script spawns cubes in documented ranges: X: [-3, 1.75], Y: [-1, 1], Z: size/2
- Standard arena setup includes 9 wooden box obstacles (WoodenBox) and 3 colored deposit boxes (PlasticFruitBox)
- Testing can be performed manually through Webots GUI (automated simulation control not required for this phase)
- Jupyter notebook environment is available for analysis and visualization

## Dependencies

- **Phase 1.1**: Webots setup must be complete and validated (completed)
- **Base Code**: YouBot controller files must be functional
- **Webots Simulator**: R2023b running on macOS 15.1 with Apple M4 Pro
- **Python Environment**: Python 3.14.0 with virtual environment
- **Required Libraries**: numpy (1.26.4), matplotlib (3.10.7), scipy (1.16.3), opencv-python (4.11.0.86)
- **Documentation**: Constitution principles require DECISIONS.md updates and scientific citations

## Scope

### In Scope
- Testing and validating all YouBot control interfaces (base, arm, gripper)
- Analyzing LIDAR sensor data format, range, and capabilities
- Analyzing camera RGB data format, resolution, and color detection feasibility
- Creating visualizations for LIDAR scans and camera frames
- Documenting arena spatial layout and dimensions
- Creating automated test script for control validation
- Creating Jupyter notebook for sensor analysis
- Documenting all findings for use in Phase 2 algorithm design

### Out of Scope
- Implementing neural networks for LIDAR or camera processing (Phase 2)
- Implementing autonomous navigation behaviors (Phase 4)
- Implementing grasping sequences (Phase 5)
- Training machine learning models (Phase 2)
- Creating occupancy grid maps (Phase 4)
- Path planning algorithms (Phase 4)
- Fuzzy logic controllers (Phase 3)
- Full system integration (Phase 6)
- Performance optimization (Phase 7)

## Notes

This specification covers Phase 1.2-1.3 of the YouBot autonomous system project for MATA64. The primary goal is exploratory validation and analysis to establish a solid foundation for implementing AI-based perception (Phase 2) and control (Phase 3) systems.

All technical decisions during implementation must be documented in DECISIONS.md following the constitution principles. Scientific citations must reference papers from REFERENCIAS.md, particularly:
- Bischoff et al. (2011): YouBot specifications
- Michel (2004): Webots documentation
- Additional references as appropriate for sensor analysis methodologies
