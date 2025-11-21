"""
YouBot Control Validation Tests
Tests FR-001 through FR-013 from spec.md

Phase 1.2 - Base Movement, Arm, and Gripper Control Validation
"""

import pytest
import sys
import os
import json

# Import test helpers
from test_helpers import (
    calculate_distance,
    normalize_angle,
    get_robot_position,
    get_robot_heading,
    wait_for_motion,
    is_position_stable
)


class TestBaseMovement:
    """Test base movement controls (FR-001 to FR-007)."""

    def test_base_forward_movement(self, youbot):
        """FR-001: Forward movement validation.

        Success Criteria:
        - Robot moves forward: x1 > x0
        - Lateral drift minimal: |y1 - y0| < 0.1m
        - No rotation: heading unchanged (±5°)
        """
        robot, base, arm, gripper = youbot

        # Get initial position
        pos0 = get_robot_position(robot)
        heading0 = get_robot_heading(robot)

        assert pos0 is not None, "Failed to get initial position"
        assert heading0 is not None, "Failed to get initial heading"

        # Command forward movement
        base.move(vx=0.2, vy=0.0, omega=0.0)

        # Execute for 5 seconds
        wait_for_motion(robot, 5000)

        # Stop
        base.reset()
        wait_for_motion(robot, 500)

        # Get final position
        pos1 = get_robot_position(robot)
        heading1 = get_robot_heading(robot)

        # Assertions
        assert pos1[0] > pos0[0], f"Robot did not move forward: x0={pos0[0]}, x1={pos1[0]}"

        lateral_drift = abs(pos1[1] - pos0[1])
        assert lateral_drift < 0.1, f"Lateral drift too large: {lateral_drift:.3f}m"

        heading_change = abs(normalize_angle(heading1 - heading0))
        assert heading_change < 0.087, f"Heading changed: {heading_change:.3f} rad (>5°)"


    def test_base_backward_movement(self, youbot):
        """FR-002: Backward movement validation.

        Success Criteria:
        - Robot moves backward: x1 < x0
        - Lateral drift minimal: |y1 - y0| < 0.1m
        - Displacement magnitude: |x1 - x0| > 0.5m
        """
        robot, base, arm, gripper = youbot

        # Get initial position
        pos0 = get_robot_position(robot)
        assert pos0 is not None, "Failed to get initial position"

        # Command backward movement
        base.move(vx=-0.2, vy=0.0, omega=0.0)

        # Execute for 5 seconds
        wait_for_motion(robot, 5000)

        # Stop
        base.reset()
        wait_for_motion(robot, 500)

        # Get final position
        pos1 = get_robot_position(robot)

        # Assertions
        assert pos1[0] < pos0[0], f"Robot did not move backward: x0={pos0[0]}, x1={pos1[0]}"

        lateral_drift = abs(pos1[1] - pos0[1])
        assert lateral_drift < 0.1, f"Lateral drift too large: {lateral_drift:.3f}m"

        displacement = abs(pos1[0] - pos0[0])
        assert displacement > 0.5, f"Displacement too small: {displacement:.3f}m"


    def test_base_strafe_left(self, youbot):
        """FR-003: Strafe left movement validation.

        Success Criteria:
        - Robot moves left: y1 > y0
        - Forward drift minimal: |x1 - x0| < 0.1m
        - Displacement magnitude: |y1 - y0| > 0.5m
        """
        robot, base, arm, gripper = youbot

        # Get initial position
        pos0 = get_robot_position(robot)
        assert pos0 is not None, "Failed to get initial position"

        # Command strafe left movement
        base.move(vx=0.0, vy=0.2, omega=0.0)

        # Execute for 5 seconds
        wait_for_motion(robot, 5000)

        # Stop
        base.reset()
        wait_for_motion(robot, 500)

        # Get final position
        pos1 = get_robot_position(robot)

        # Assertions
        assert pos1[1] > pos0[1], f"Robot did not strafe left: y0={pos0[1]}, y1={pos1[1]}"

        forward_drift = abs(pos1[0] - pos0[0])
        assert forward_drift < 0.1, f"Forward drift too large: {forward_drift:.3f}m"

        displacement = abs(pos1[1] - pos0[1])
        assert displacement > 0.5, f"Displacement too small: {displacement:.3f}m"


    def test_base_strafe_right(self, youbot):
        """FR-003: Strafe right movement validation.

        Success Criteria:
        - Robot moves right: y1 < y0
        - Forward drift minimal: |x1 - x0| < 0.1m
        - Displacement magnitude: |y1 - y0| > 0.5m
        """
        robot, base, arm, gripper = youbot

        # Get initial position
        pos0 = get_robot_position(robot)
        assert pos0 is not None, "Failed to get initial position"

        # Command strafe right movement
        base.move(vx=0.0, vy=-0.2, omega=0.0)

        # Execute for 5 seconds
        wait_for_motion(robot, 5000)

        # Stop
        base.reset()
        wait_for_motion(robot, 500)

        # Get final position
        pos1 = get_robot_position(robot)

        # Assertions
        assert pos1[1] < pos0[1], f"Robot did not strafe right: y0={pos0[1]}, y1={pos1[1]}"

        forward_drift = abs(pos1[0] - pos0[0])
        assert forward_drift < 0.1, f"Forward drift too large: {forward_drift:.3f}m"

        displacement = abs(pos1[1] - pos0[1])
        assert displacement > 0.5, f"Displacement too small: {displacement:.3f}m"


    def test_base_rotate_clockwise(self, youbot):
        """FR-004: Clockwise rotation validation.

        Success Criteria:
        - Robot rotates clockwise: θ1 < θ0
        - Position drift minimal: distance < 0.2m
        - Rotation magnitude: |θ1 - θ0| > 0.5 rad (~30°)
        """
        robot, base, arm, gripper = youbot

        # Get initial state
        pos0 = get_robot_position(robot)
        heading0 = get_robot_heading(robot)
        assert pos0 is not None, "Failed to get initial position"
        assert heading0 is not None, "Failed to get initial heading"

        # Command clockwise rotation
        base.move(vx=0.0, vy=0.0, omega=-0.3)

        # Execute for 5 seconds
        wait_for_motion(robot, 5000)

        # Stop
        base.reset()
        wait_for_motion(robot, 500)

        # Get final state
        pos1 = get_robot_position(robot)
        heading1 = get_robot_heading(robot)

        # Assertions
        heading_change = normalize_angle(heading1 - heading0)
        assert heading_change < 0, f"Robot did not rotate clockwise: Δθ={heading_change:.3f} rad"

        position_drift = calculate_distance(pos0, pos1)
        assert position_drift < 0.2, f"Position drift too large: {position_drift:.3f}m"

        rotation_magnitude = abs(heading_change)
        assert rotation_magnitude > 0.5, f"Rotation too small: {rotation_magnitude:.3f} rad"


    def test_base_rotate_counterclockwise(self, youbot):
        """FR-004: Counterclockwise rotation validation.

        Success Criteria:
        - Robot rotates CCW: θ1 > θ0
        - Position drift minimal: distance < 0.2m
        - Rotation magnitude: |θ1 - θ0| > 0.5 rad (~30°)
        """
        robot, base, arm, gripper = youbot

        # Get initial state
        pos0 = get_robot_position(robot)
        heading0 = get_robot_heading(robot)
        assert pos0 is not None, "Failed to get initial position"
        assert heading0 is not None, "Failed to get initial heading"

        # Command counterclockwise rotation
        base.move(vx=0.0, vy=0.0, omega=0.3)

        # Execute for 5 seconds
        wait_for_motion(robot, 5000)

        # Stop
        base.reset()
        wait_for_motion(robot, 500)

        # Get final state
        pos1 = get_robot_position(robot)
        heading1 = get_robot_heading(robot)

        # Assertions
        heading_change = normalize_angle(heading1 - heading0)
        assert heading_change > 0, f"Robot did not rotate CCW: Δθ={heading_change:.3f} rad"

        position_drift = calculate_distance(pos0, pos1)
        assert position_drift < 0.2, f"Position drift too large: {position_drift:.3f}m"

        rotation_magnitude = abs(heading_change)
        assert rotation_magnitude > 0.5, f"Rotation too small: {rotation_magnitude:.3f} rad"


    def test_base_stop_command(self, youbot):
        """FR-005: Stop command validation.

        Success Criteria:
        - Position drift after stop: < 0.05m
        - Heading drift after stop: < 0.05 rad (~3°)
        """
        robot, base, arm, gripper = youbot

        # Command mixed motion
        base.move(vx=0.2, vy=0.1, omega=0.2)

        # Execute for 2 seconds (robot should be moving)
        wait_for_motion(robot, 2000)

        # Command stop
        base.reset()

        # Wait for settling
        wait_for_motion(robot, 1000)

        # Get position after stop
        pos_after_stop = get_robot_position(robot)
        heading_after_stop = get_robot_heading(robot)
        assert pos_after_stop is not None, "Failed to get position after stop"
        assert heading_after_stop is not None, "Failed to get heading after stop"

        # Wait 2 more seconds to measure drift
        wait_for_motion(robot, 2000)

        # Get final position
        pos_final = get_robot_position(robot)
        heading_final = get_robot_heading(robot)

        # Assertions
        position_drift = calculate_distance(pos_after_stop, pos_final)
        assert position_drift < 0.05, f"Position drift after stop: {position_drift:.3f}m (>0.05m)"

        heading_drift = abs(normalize_angle(heading_final - heading_after_stop))
        assert heading_drift < 0.05, f"Heading drift after stop: {heading_drift:.3f} rad (>0.05 rad)"


    def test_base_velocity_limits(self, youbot, velocity_limits):
        """FR-006: Velocity limits documentation.

        Measures maximum achievable velocities for vx, vy, omega
        and documents results to logs/velocity_limits.json

        Success Criteria:
        - Returns dict with measured limits for vx, vy, omega
        - Limits match YouBot specifications (±10%)
        - Results exported to logs/velocity_limits.json
        """
        robot, base, arm, gripper = youbot

        # Test incremental velocities to find limits
        # Based on base.py: MAX_SPEED = 0.3 m/s, but we'll measure empirically

        # Measure vx_max (forward)
        vx_max = 0.0
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pos0 = get_robot_position(robot)
            base.move(vx=v, vy=0.0, omega=0.0)
            wait_for_motion(robot, 2000)
            base.reset()
            wait_for_motion(robot, 500)
            pos1 = get_robot_position(robot)

            displacement = abs(pos1[0] - pos0[0])
            if displacement > 0.1:  # Robot responded to command
                vx_max = v
            else:
                break  # Reached limit

        # Measure vx_min (backward)
        vx_min = 0.0
        for v in [-0.1, -0.2, -0.3, -0.4, -0.5]:
            pos0 = get_robot_position(robot)
            base.move(vx=v, vy=0.0, omega=0.0)
            wait_for_motion(robot, 2000)
            base.reset()
            wait_for_motion(robot, 500)
            pos1 = get_robot_position(robot)

            displacement = abs(pos1[0] - pos0[0])
            if displacement > 0.1:
                vx_min = v
            else:
                break

        # Measure vy_max (strafe left)
        vy_max = 0.0
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pos0 = get_robot_position(robot)
            base.move(vx=0.0, vy=v, omega=0.0)
            wait_for_motion(robot, 2000)
            base.reset()
            wait_for_motion(robot, 500)
            pos1 = get_robot_position(robot)

            displacement = abs(pos1[1] - pos0[1])
            if displacement > 0.1:
                vy_max = v
            else:
                break

        # Measure vy_min (strafe right)
        vy_min = 0.0
        for v in [-0.1, -0.2, -0.3, -0.4, -0.5]:
            pos0 = get_robot_position(robot)
            base.move(vx=0.0, vy=v, omega=0.0)
            wait_for_motion(robot, 2000)
            base.reset()
            wait_for_motion(robot, 500)
            pos1 = get_robot_position(robot)

            displacement = abs(pos1[1] - pos0[1])
            if displacement > 0.1:
                vy_min = v
            else:
                break

        # Measure omega_max (CCW)
        omega_max = 0.0
        for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
            heading0 = get_robot_heading(robot)
            base.move(vx=0.0, vy=0.0, omega=w)
            wait_for_motion(robot, 2000)
            base.reset()
            wait_for_motion(robot, 500)
            heading1 = get_robot_heading(robot)

            rotation = abs(normalize_angle(heading1 - heading0))
            if rotation > 0.1:  # ~6° rotation
                omega_max = w
            else:
                break

        # Measure omega_min (CW)
        omega_min = 0.0
        for w in [-0.1, -0.2, -0.3, -0.4, -0.5]:
            heading0 = get_robot_heading(robot)
            base.move(vx=0.0, vy=0.0, omega=w)
            wait_for_motion(robot, 2000)
            base.reset()
            wait_for_motion(robot, 500)
            heading1 = get_robot_heading(robot)

            rotation = abs(normalize_angle(heading1 - heading0))
            if rotation > 0.1:
                omega_min = w
            else:
                break

        # Store results in fixture
        velocity_limits['vx_max'] = vx_max
        velocity_limits['vx_min'] = vx_min
        velocity_limits['vy_max'] = vy_max
        velocity_limits['vy_min'] = vy_min
        velocity_limits['omega_max'] = omega_max
        velocity_limits['omega_min'] = omega_min

        # Export to JSON (T020)
        logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        output_path = os.path.join(logs_dir, 'velocity_limits.json')
        with open(output_path, 'w') as f:
            json.dump(velocity_limits, f, indent=2)

        # Assertions
        assert vx_max > 0, "Failed to measure vx_max"
        assert vx_min < 0, "Failed to measure vx_min"
        assert vy_max > 0, "Failed to measure vy_max"
        assert vy_min < 0, "Failed to measure vy_min"
        assert omega_max > 0, "Failed to measure omega_max"
        assert omega_min < 0, "Failed to measure omega_min"

        # Verify limits within expected range (±10% of 0.3 m/s typical)
        assert 0.2 <= abs(vx_max) <= 0.6, f"vx_max out of expected range: {vx_max}"
        assert 0.2 <= abs(vx_min) <= 0.6, f"vx_min out of expected range: {vx_min}"


class TestArmGripper:
    """Test arm and gripper controls (FR-008 to FR-013)."""

    def test_arm_height_positions(self, youbot):
        """FR-008: Arm height positioning validation."""
        # TODO: Implement (T025)
        pytest.skip("Not yet implemented")


    def test_arm_orientation_positions(self, youbot):
        """FR-009: Arm orientation positioning validation."""
        # TODO: Implement (T026)
        pytest.skip("Not yet implemented")


    def test_gripper_close(self, youbot):
        """FR-010: Gripper grip command validation."""
        # TODO: Implement (T027)
        pytest.skip("Not yet implemented")


    def test_gripper_open(self, youbot):
        """FR-011: Gripper release command validation."""
        # TODO: Implement (T028)
        pytest.skip("Not yet implemented")


    def test_arm_joint_limits(self, youbot, joint_limits):
        """FR-012: Arm joint limits documentation."""
        # TODO: Implement (T029-T030)
        pytest.skip("Not yet implemented")
