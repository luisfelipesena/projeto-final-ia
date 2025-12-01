#!/usr/bin/env python3
"""
YouBot MCP Server - Model Context Protocol server for Webots YouBot control

MATA64 Final Project: Autonomous Cube Collection
University: UFBA - Federal University of Bahia

This MCP server enables LLMs to control a YouBot robot in Webots simulation
for an autonomous cube collection task. It provides tools for:
- Base movement (omnidirectional mecanum wheels)
- Arm control (5-DOF arm with inverse kinematics)
- Gripper manipulation
- Camera and LIDAR perception
- High-level task execution (grasp, deposit)

Architecture:
    MCP Server <-> commands.json/status.json <-> Webots Controller

Usage:
    python youbot_mcp_server.py
"""

import json
import time
import base64
import logging
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict
from mcp.server.fastmcp import FastMCP

# Configure logging (stderr only - stdout reserved for MCP protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'youbot_mcp.log'),
    ]
)
logger = logging.getLogger('YouBotMCP')

# ==================== CONSTANTS ====================

MCP_DIR = Path(__file__).parent
DATA_DIR = MCP_DIR / "data" / "youbot"
COMMANDS_FILE = DATA_DIR / "commands.json"
STATUS_FILE = DATA_DIR / "status.json"
CAMERA_IMAGE_FILE = DATA_DIR / "camera_image.jpg"
LIDAR_DATA_FILE = DATA_DIR / "lidar_data.json"

CHARACTER_LIMIT = 25000  # Max response size
COMMAND_TIMEOUT = 10.0  # Seconds to wait for status update

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==================== ENUMS ====================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class ArmHeight(str, Enum):
    """Arm height presets."""
    FRONT_FLOOR = "FRONT_FLOOR"
    FRONT_PLATE = "FRONT_PLATE"
    FRONT_CARDBOARD_BOX = "FRONT_CARDBOARD_BOX"
    RESET = "RESET"
    BACK_PLATE_HIGH = "BACK_PLATE_HIGH"
    BACK_PLATE_LOW = "BACK_PLATE_LOW"
    HANOI_PREPARE = "HANOI_PREPARE"


class ArmOrientation(str, Enum):
    """Arm orientation presets."""
    BACK_LEFT = "BACK_LEFT"
    LEFT = "LEFT"
    FRONT_LEFT = "FRONT_LEFT"
    FRONT = "FRONT"
    FRONT_RIGHT = "FRONT_RIGHT"
    RIGHT = "RIGHT"
    BACK_RIGHT = "BACK_RIGHT"


class RobotState(str, Enum):
    """Controller state machine states."""
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    GRASPING = "GRASPING"
    DEPOSITING = "DEPOSITING"
    AVOIDING = "AVOIDING"


class CubeColor(str, Enum):
    """Cube and deposit box colors."""
    GREEN = "green"
    BLUE = "blue"
    RED = "red"


# ==================== PYDANTIC MODELS ====================

class MoveBaseInput(BaseModel):
    """Input for omnidirectional base movement."""
    model_config = ConfigDict(str_strip_whitespace=True)

    vx: float = Field(
        default=0.0,
        description="Forward velocity in m/s. Positive=forward, negative=backward. (e.g., 0.1, -0.2)",
        ge=-0.3, le=0.3
    )
    vy: float = Field(
        default=0.0,
        description="Lateral velocity in m/s. Positive=left, negative=right. (e.g., 0.1 for strafe left)",
        ge=-0.3, le=0.3
    )
    omega: float = Field(
        default=0.0,
        description="Angular velocity in rad/s. Positive=counter-clockwise. (e.g., 0.5 for slow left turn)",
        ge=-1.0, le=1.0
    )


class MoveForwardInput(BaseModel):
    """Input for forward movement by distance."""
    model_config = ConfigDict(str_strip_whitespace=True)

    distance_m: float = Field(
        ...,
        description="Distance to move in meters (e.g., 0.5 for 50cm)",
        gt=0, le=3.0
    )
    speed: float = Field(
        default=0.1,
        description="Movement speed in m/s (0.05=slow, 0.15=medium, 0.3=fast)",
        ge=0.05, le=0.3
    )


class RotateInput(BaseModel):
    """Input for in-place rotation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    angle_deg: float = Field(
        ...,
        description="Angle to rotate in degrees. Positive=counter-clockwise. (e.g., 90, -45)",
        ge=-360, le=360
    )
    speed: float = Field(
        default=0.5,
        description="Angular speed in rad/s",
        ge=0.1, le=1.5
    )


class ArmHeightInput(BaseModel):
    """Input for arm height control."""
    height: ArmHeight = Field(
        default=ArmHeight.RESET,
        description="Arm height preset. FRONT_FLOOR=pick from ground, RESET=folded position"
    )


class ArmOrientationInput(BaseModel):
    """Input for arm orientation control."""
    orientation: ArmOrientation = Field(
        default=ArmOrientation.FRONT,
        description="Arm base rotation. FRONT=camera direction, BACK=behind robot"
    )


class ArmIKInput(BaseModel):
    """Input for inverse kinematics arm positioning."""
    model_config = ConfigDict(str_strip_whitespace=True)

    x: float = Field(..., description="X position in meters (left/right from arm base)", ge=-0.5, le=0.5)
    y: float = Field(..., description="Y position in meters (forward from arm base)", ge=0.1, le=0.6)
    z: float = Field(..., description="Z position in meters (height from ground)", ge=0.0, le=0.5)


class GripperGapInput(BaseModel):
    """Input for gripper gap control."""
    gap_m: float = Field(
        default=0.025,
        description="Gap between gripper fingers in meters (0.0=closed, 0.05=fully open)",
        ge=0.0, le=0.05
    )


class DepositInput(BaseModel):
    """Input for cube deposit operation."""
    color: CubeColor = Field(
        ...,
        description="Color of cube to deposit (determines target box)"
    )


class StateInput(BaseModel):
    """Input for state machine control."""
    state: RobotState = Field(
        ...,
        description="Target state for the controller"
    )


class StatusInput(BaseModel):
    """Input for status query with format option."""
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable, 'json' for parsing"
    )


# ==================== SHARED UTILITIES ====================

def write_command(command: dict) -> bool:
    """Write command to commands.json for Webots controller."""
    try:
        command["timestamp"] = time.time()
        command["id"] = int(time.time() * 1000)
        with open(COMMANDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(command, f, indent=2)
        logger.info(f"Command: {command['action']}")
        return True
    except Exception as e:
        logger.error(f"Write command failed: {e}")
        return False


def read_status() -> Optional[dict]:
    """Read current status from status.json."""
    try:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Read status failed: {e}")
    return None


def wait_for_status_update(timeout: float = COMMAND_TIMEOUT) -> Optional[dict]:
    """Wait for controller to update status after command."""
    start_time = time.time()
    status = read_status()
    last_ts = status.get("last_update", 0) if status else 0

    while time.time() - start_time < timeout:
        status = read_status()
        if status and status.get("last_update", 0) > last_ts:
            return status
        time.sleep(0.1)

    return None


def format_status_markdown(status: dict) -> str:
    """Format robot status as markdown."""
    lines = ["# YouBot Status", ""]

    state = status.get("current_state", "UNKNOWN")
    cubes = status.get("cubes_collected", 0)
    lines.append(f"**State**: {state} | **Cubes Collected**: {cubes}/15")
    lines.append("")

    # Base velocity
    vel = status.get("base_velocity", {})
    lines.append("## Movement")
    lines.append(f"- Forward: {vel.get('vx', 0):.2f} m/s")
    lines.append(f"- Lateral: {vel.get('vy', 0):.2f} m/s")
    lines.append(f"- Rotation: {vel.get('omega', 0):.2f} rad/s")
    lines.append("")

    # Arm & Gripper
    lines.append("## Manipulator")
    lines.append(f"- Arm Height: {status.get('arm_height', 'UNKNOWN')}")
    lines.append(f"- Arm Orientation: {status.get('arm_orientation', 'UNKNOWN')}")
    lines.append(f"- Gripper: {status.get('gripper_state', 'unknown')}")
    if status.get('gripper_has_object'):
        lines.append("- **Object detected in gripper**")
    lines.append("")

    # Perception
    lines.append("## Perception")
    obstacle_dist = status.get("min_obstacle_distance", float('inf'))
    lines.append(f"- Nearest obstacle: {obstacle_dist:.2f}m" if obstacle_dist < 10 else "- No obstacles detected")

    detections = status.get("cube_detections", [])
    if detections:
        lines.append(f"- Cubes visible: {len(detections)}")
        for det in detections[:3]:
            lines.append(f"  - {det['color']}: {det['distance']:.2f}m at {det['angle']:.1f}°")
    else:
        lines.append("- No cubes in view")

    return "\n".join(lines)


# ==================== MCP SERVER ====================

mcp = FastMCP("youbot_mcp")

# ==================== SYSTEM TOOLS ====================

@mcp.tool(
    name="youbot_check_connection",
    annotations={
        "title": "Check Webots Connection",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_check_connection() -> str:
    """Check if Webots controller is connected and responding.

    Use this tool first to verify the simulation is running before
    sending commands. The controller must be active in Webots for
    any robot commands to work.

    Returns:
        str: Connection status with details about last update time

    Examples:
        - Use when: Starting a new session with the robot
        - Use when: Commands seem to not be executing
        - Don't use: During active task execution (adds latency)
    """
    status = read_status()
    if status:
        last_update = status.get("last_update", 0)
        age = time.time() - last_update
        state = status.get("current_state", "UNKNOWN")
        cubes = status.get("cubes_collected", 0)

        if age < 5.0:
            return f"Connected. State: {state}, Cubes: {cubes}/15, Last update: {age:.1f}s ago"
        else:
            return f"Controller stale ({age:.0f}s since last update). Start Webots simulation."

    return "Not connected. Start Webots simulation with YouBot MCP controller."


@mcp.tool(
    name="youbot_get_status",
    annotations={
        "title": "Get Robot Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_get_status(params: StatusInput) -> str:
    """Get complete robot status including sensors and actuator states.

    Returns comprehensive information about the robot's current state,
    including movement, arm position, gripper state, and sensor readings.

    Args:
        params: StatusInput with response_format (markdown or json)

    Returns:
        str: Robot status in requested format

    Examples:
        - Use when: Planning next action based on current state
        - Use when: Checking if grasp was successful
        - Use when: Monitoring task progress
    """
    status = read_status()
    if not status:
        return "Error: No status available. Check controller connection."

    if params.response_format == ResponseFormat.MARKDOWN:
        return format_status_markdown(status)
    else:
        return json.dumps(status, indent=2, default=str)


# ==================== BASE MOVEMENT TOOLS ====================

@mcp.tool(
    name="youbot_move_base",
    annotations={
        "title": "Move Base (Continuous)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_move_base(params: MoveBaseInput) -> str:
    """Set YouBot base velocities for omnidirectional movement.

    The YouBot uses mecanum wheels allowing independent control of:
    - Forward/backward (vx)
    - Left/right strafe (vy)
    - Rotation (omega)

    Velocities persist until changed or stopped. Use youbot_stop_base
    to halt movement.

    Args:
        params: MoveBaseInput with vx, vy, omega velocities

    Returns:
        str: Confirmation of velocity settings

    Examples:
        - Move forward slowly: vx=0.1, vy=0, omega=0
        - Strafe left: vx=0, vy=0.1, omega=0
        - Rotate right: vx=0, vy=0, omega=-0.5
        - Diagonal: vx=0.1, vy=0.1, omega=0
    """
    cmd = {
        "action": "move_base",
        "params": {"vx": params.vx, "vy": params.vy, "omega": params.omega}
    }

    if write_command(cmd):
        return f"Base velocities set: vx={params.vx:.2f} vy={params.vy:.2f} omega={params.omega:.2f}"
    return "Error: Failed to send command. Check controller connection."


@mcp.tool(
    name="youbot_stop_base",
    annotations={
        "title": "Stop Base Movement",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_stop_base() -> str:
    """Stop all base movement immediately.

    Halts all wheel motors. Safe to call multiple times.
    Always call before starting manipulation tasks.

    Returns:
        str: Confirmation that base stopped

    Examples:
        - Use before: Grasping a cube
        - Use when: Obstacle detected
        - Use after: Reaching target position
    """
    if write_command({"action": "stop_base", "params": {}}):
        return "Base stopped"
    return "Error: Failed to stop. Check connection."


@mcp.tool(
    name="youbot_move_forward",
    annotations={
        "title": "Move Forward Distance",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_move_forward(params: MoveForwardInput) -> str:
    """Move forward a specific distance then stop.

    Blocking operation - waits for movement to complete.
    Robot will stop automatically after reaching distance.

    Args:
        params: MoveForwardInput with distance_m and speed

    Returns:
        str: Movement result

    Examples:
        - Approach cube: distance_m=0.3, speed=0.1
        - Quick repositioning: distance_m=0.5, speed=0.2
    """
    cmd = {
        "action": "move_forward",
        "params": {"distance_m": params.distance_m, "speed": params.speed}
    }

    if write_command(cmd):
        wait_for_status_update(timeout=params.distance_m/params.speed + 3)
        return f"Moved forward {params.distance_m:.2f}m"
    return "Error: Failed to send command."


@mcp.tool(
    name="youbot_rotate",
    annotations={
        "title": "Rotate In Place",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_rotate(params: RotateInput) -> str:
    """Rotate in place by specified angle.

    Blocking operation - waits for rotation to complete.
    Positive angle = counter-clockwise, negative = clockwise.

    Args:
        params: RotateInput with angle_deg and speed

    Returns:
        str: Rotation result

    Examples:
        - Turn left 90°: angle_deg=90
        - Turn right 45°: angle_deg=-45
        - Turn around: angle_deg=180
    """
    cmd = {
        "action": "rotate",
        "params": {"angle_deg": params.angle_deg, "speed": params.speed}
    }

    if write_command(cmd):
        wait_for_status_update(timeout=abs(params.angle_deg)/30 + 3)
        return f"Rotated {params.angle_deg:.1f}°"
    return "Error: Failed to send command."


# ==================== ARM CONTROL TOOLS ====================

@mcp.tool(
    name="youbot_set_arm_height",
    annotations={
        "title": "Set Arm Height Preset",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_set_arm_height(params: ArmHeightInput) -> str:
    """Set arm to predefined height configuration.

    Available presets:
    - FRONT_FLOOR: Ready to pick objects from ground
    - FRONT_PLATE: Mid-height for tables
    - RESET: Folded against robot body (safe for movement)
    - BACK_PLATE_HIGH/LOW: For placing objects behind

    Args:
        params: ArmHeightInput with height preset

    Returns:
        str: Arm position result

    Examples:
        - Before grasping: height=FRONT_FLOOR
        - Safe movement: height=RESET
    """
    cmd = {"action": "set_arm_height", "params": {"height": params.height.value}}

    if write_command(cmd):
        wait_for_status_update(timeout=5)
        return f"Arm height set to {params.height.value}"
    return "Error: Failed to send command."


@mcp.tool(
    name="youbot_set_arm_orientation",
    annotations={
        "title": "Set Arm Orientation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_set_arm_orientation(params: ArmOrientationInput) -> str:
    """Set arm base rotation orientation.

    Rotates the arm base to face different directions.
    FRONT faces same direction as camera.

    Args:
        params: ArmOrientationInput with orientation preset

    Returns:
        str: Arm orientation result
    """
    cmd = {"action": "set_arm_orientation", "params": {"orientation": params.orientation.value}}

    if write_command(cmd):
        wait_for_status_update(timeout=3)
        return f"Arm orientation set to {params.orientation.value}"
    return "Error: Failed to send command."


@mcp.tool(
    name="youbot_set_arm_position",
    annotations={
        "title": "Set Arm Position (IK)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_set_arm_position(params: ArmIKInput) -> str:
    """Set arm end-effector position using inverse kinematics.

    Moves gripper to specified 3D position relative to arm base.
    Not all positions are reachable - will move to closest valid position.

    Args:
        params: ArmIKInput with x, y, z coordinates in meters

    Returns:
        str: Arm position result

    Examples:
        - Reach forward: x=0, y=0.4, z=0.1
        - Reach down: x=0, y=0.3, z=0.05
    """
    cmd = {"action": "set_arm_ik", "params": {"x": params.x, "y": params.y, "z": params.z}}

    if write_command(cmd):
        wait_for_status_update(timeout=5)
        return f"Arm moved to ({params.x:.3f}, {params.y:.3f}, {params.z:.3f})"
    return "Error: Failed to send command."


@mcp.tool(
    name="youbot_reset_arm",
    annotations={
        "title": "Reset Arm to Rest",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_reset_arm() -> str:
    """Reset arm to default resting position.

    Folds arm against robot body. Safe for base movement.
    Call after grasping operations complete.

    Returns:
        str: Reset confirmation
    """
    if write_command({"action": "reset_arm", "params": {}}):
        wait_for_status_update(timeout=5)
        return "Arm reset to rest position"
    return "Error: Failed to send command."


# ==================== GRIPPER TOOLS ====================

@mcp.tool(
    name="youbot_grip",
    annotations={
        "title": "Close Gripper",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_grip() -> str:
    """Close gripper to grasp an object.

    Closes gripper fingers. If object is present, fingers will
    stop when contact is made. Check status to verify grasp.

    Returns:
        str: Gripper result with object detection status

    Examples:
        - Use when: Cube is positioned between gripper fingers
        - After: youbot_set_arm_height(FRONT_FLOOR)
    """
    if write_command({"action": "grip", "params": {}}):
        status = wait_for_status_update(timeout=3)
        if status:
            has_object = status.get("gripper_has_object", False)
            return f"Gripper closed. Object detected: {has_object}"
        return "Gripper closed"
    return "Error: Failed to send command."


@mcp.tool(
    name="youbot_release",
    annotations={
        "title": "Open Gripper",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_release() -> str:
    """Open gripper to release object.

    Opens gripper fingers fully. Use when depositing cubes
    or to release a failed grasp.

    Returns:
        str: Release confirmation
    """
    if write_command({"action": "release", "params": {}}):
        wait_for_status_update(timeout=2)
        return "Gripper opened"
    return "Error: Failed to send command."


@mcp.tool(
    name="youbot_set_gripper_gap",
    annotations={
        "title": "Set Gripper Gap",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_set_gripper_gap(params: GripperGapInput) -> str:
    """Set gripper opening gap.

    Fine control of gripper opening for precise manipulation.
    Cubes are 5cm, so gap of 0.06m (60mm) gives clearance.

    Args:
        params: GripperGapInput with gap_m

    Returns:
        str: Gap setting confirmation
    """
    if write_command({"action": "set_gripper_gap", "params": {"gap": params.gap_m}}):
        return f"Gripper gap set to {params.gap_m*1000:.0f}mm"
    return "Error: Failed to send command."


# ==================== PERCEPTION TOOLS ====================

@mcp.tool(
    name="youbot_get_camera_image",
    annotations={
        "title": "Capture Camera Image",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_get_camera_image() -> str:
    """Capture current camera image from robot.

    Returns base64-encoded JPEG of robot's front camera view.
    Use for visual inspection of environment.

    Returns:
        str: Base64 image data or error message

    Examples:
        - Use when: Verifying cube position before grasp
        - Use when: Debugging detection issues
    """
    if not write_command({"action": "capture_camera", "params": {}}):
        return "Error: Failed to request image"

    start_time = time.time()
    while time.time() - start_time < 5.0:
        if CAMERA_IMAGE_FILE.exists():
            try:
                with open(CAMERA_IMAGE_FILE, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{img_data}"
            except Exception as e:
                return f"Error reading image: {e}"
        time.sleep(0.1)

    return "Timeout waiting for camera image"


@mcp.tool(
    name="youbot_detect_cubes",
    annotations={
        "title": "Detect Cubes in View",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_detect_cubes() -> str:
    """Detect colored cubes in current camera view.

    Uses HSV color segmentation to find green, blue, and red cubes.
    Returns distance and angle for each detection.

    Returns:
        str: JSON with detected cubes (color, distance, angle, confidence)

    Examples:
        - Use when: Searching for cubes
        - Use when: Deciding which cube to approach
    """
    if write_command({"action": "detect_cubes", "params": {}}):
        status = wait_for_status_update(timeout=3)
        if status:
            detections = status.get("cube_detections", [])
            target = status.get("current_target")

            result = {
                "count": len(detections),
                "detections": detections[:5],
                "target": target
            }
            return json.dumps(result, indent=2)
    return json.dumps({"count": 0, "detections": [], "error": "Detection failed"})


@mcp.tool(
    name="youbot_get_lidar_data",
    annotations={
        "title": "Get LIDAR Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_get_lidar_data() -> str:
    """Get LIDAR range data for obstacle detection.

    Returns sector-based obstacle analysis from 360° LIDAR scan.
    Useful for navigation and collision avoidance.

    Returns:
        str: JSON with LIDAR ranges and obstacle analysis
    """
    if write_command({"action": "get_lidar", "params": {}}):
        time.sleep(0.2)

        if LIDAR_DATA_FILE.exists():
            try:
                with open(LIDAR_DATA_FILE, 'r') as f:
                    return f.read()
            except Exception:
                pass

        status = wait_for_status_update(timeout=2)
        if status:
            return json.dumps({
                "min_distance": status.get("min_obstacle_distance", float('inf')),
                "sectors": status.get("obstacle_sectors", {}),
                "enabled": status.get("lidar_enabled", False)
            }, indent=2)

    return json.dumps({"error": "Failed to get LIDAR data"})


# ==================== HIGH-LEVEL TASK TOOLS ====================

@mcp.tool(
    name="youbot_grasp_cube",
    annotations={
        "title": "Execute Grasp Sequence",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_grasp_cube() -> str:
    """Execute complete grasp sequence for nearest detected cube.

    High-level operation that:
    1. Prepares arm (lowers to floor position)
    2. Opens gripper
    3. Closes gripper to grasp
    4. Lifts arm
    5. Verifies grasp success

    Call after youbot_stop_base() when aligned with cube.

    Returns:
        str: Grasp result with success/failure details

    Examples:
        - Use after: Approaching and aligning with cube
        - Prerequisite: Robot must be stopped and cube in gripper range
    """
    if write_command({"action": "grasp_sequence", "params": {}}):
        status = wait_for_status_update(timeout=15)
        if status:
            success = status.get("grasp_success", False)
            color = status.get("grasped_cube_color", "unknown")
            cubes = status.get("cubes_collected", 0)

            if success:
                return f"Grasp SUCCESS! Color: {color}, Total cubes: {cubes}/15"
            return f"Grasp FAILED. Cube may have moved. Try repositioning."
        return "Grasp sequence timeout - check robot state"
    return "Error: Failed to start grasp sequence"


@mcp.tool(
    name="youbot_deposit_cube",
    annotations={
        "title": "Deposit Cube in Box",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_deposit_cube(params: DepositInput) -> str:
    """Navigate to deposit box and release cube.

    High-level operation that:
    1. Rotates toward target box
    2. Drives to box location
    3. Positions arm over box
    4. Releases cube
    5. Returns arm to rest

    Box locations:
    - Green: Back-left (0.48, 1.58)
    - Blue: Back-right (0.48, -1.62)
    - Red: Right side (2.31, 0.01)

    Args:
        params: DepositInput with target color

    Returns:
        str: Deposit result
    """
    if write_command({"action": "deposit_cube", "params": {"color": params.color.value}}):
        status = wait_for_status_update(timeout=30)
        if status:
            success = status.get("deposit_success", False)
            cubes = status.get("cubes_collected", 0)
            return f"Deposit {'SUCCESS' if success else 'FAILED'}. Progress: {cubes}/15 cubes"
        return "Deposit sequence timeout"
    return "Error: Failed to start deposit"


@mcp.tool(
    name="youbot_set_state",
    annotations={
        "title": "Set Controller State",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_set_state(params: StateInput) -> str:
    """Set robot controller state machine state.

    Manually override controller state. Use with caution.

    States:
    - IDLE: No active task
    - SEARCHING: Rotating to find cubes
    - APPROACHING: Moving toward detected cube
    - GRASPING: Executing grasp sequence
    - DEPOSITING: Moving to box and depositing
    - AVOIDING: Backing away from obstacle

    Args:
        params: StateInput with target state

    Returns:
        str: State change confirmation
    """
    if write_command({"action": "set_state", "params": {"state": params.state.value}}):
        return f"State set to {params.state.value}"
    return "Error: Failed to change state"


@mcp.tool(
    name="youbot_start_autonomous",
    annotations={
        "title": "Start Autonomous Mode",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_start_autonomous() -> str:
    """Start autonomous cube collection task.

    Robot will autonomously:
    1. Search for cubes
    2. Approach and grasp
    3. Identify color
    4. Deposit in correct box
    5. Repeat until 15 cubes collected

    Use youbot_get_status() to monitor progress.

    Returns:
        str: Task start confirmation
    """
    if write_command({"action": "start_autonomous", "params": {}}):
        return "Autonomous task started. Use youbot_get_status() to monitor."
    return "Error: Failed to start autonomous mode"


@mcp.tool(
    name="youbot_stop_autonomous",
    annotations={
        "title": "Stop Autonomous Mode",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def youbot_stop_autonomous() -> str:
    """Stop autonomous task and return to IDLE.

    Immediately halts autonomous operation.
    Robot will stop and wait for manual commands.

    Returns:
        str: Stop confirmation
    """
    if write_command({"action": "stop_autonomous", "params": {}}):
        return "Autonomous task stopped. Robot now in IDLE state."
    return "Error: Failed to stop autonomous mode"


# ==================== INFO TOOLS ====================

@mcp.tool(
    name="youbot_get_deposit_boxes",
    annotations={
        "title": "Get Box Locations",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def youbot_get_deposit_boxes() -> str:
    """Get coordinates of all deposit boxes.

    Returns fixed positions of the three colored deposit boxes
    in the arena. Useful for navigation planning.

    Returns:
        str: Markdown formatted box locations
    """
    return """# Deposit Box Locations

| Color | X (m) | Y (m) | Description |
|-------|-------|-------|-------------|
| Green | 0.48  | 1.58  | Back-left corner |
| Blue  | 0.48  | -1.62 | Back-right corner |
| Red   | 2.31  | 0.01  | Right side center |

Coordinates are in world frame (from initial robot position).
"""


@mcp.tool(
    name="youbot_reload_world",
    annotations={
        "title": "Reload Webots World",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def youbot_reload_world() -> str:
    """Reload Webots world simulation (macOS only via AppleScript).

    Sends Cmd+Shift+R to Webots to reload the world.
    This restarts all controllers and respawns cubes.

    Prerequisites:
    - macOS with Webots running
    - Terminal/osascript must have Accessibility permissions

    Returns:
        str: Reload status
    """
    if platform.system() != "Darwin":
        return "Error: World reload only supported on macOS"

    try:
        # Activate Webots and send Cmd+Shift+R
        subprocess.run(
            ["osascript", "-e", 'tell application "Webots" to activate'],
            check=True, capture_output=True, timeout=5
        )
        time.sleep(0.5)
        subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to keystroke "r" using {shift down, command down}'],
            check=True, capture_output=True, timeout=5
        )
        logger.info("World reload triggered via AppleScript")
        return "World reload triggered. Wait ~5s for controllers to restart."
    except subprocess.CalledProcessError as e:
        return f"Error: AppleScript failed - {e.stderr.decode() if e.stderr else str(e)}"
    except subprocess.TimeoutExpired:
        return "Error: AppleScript timeout"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool(
    name="youbot_list_capabilities",
    annotations={
        "title": "List All Capabilities",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def youbot_list_capabilities() -> str:
    """List all available robot capabilities and tools.

    Returns organized list of all MCP tools available for
    controlling the YouBot robot.

    Returns:
        str: Markdown formatted capability list
    """
    return """# YouBot MCP Capabilities

## System
- `youbot_check_connection` - Verify Webots controller is running
- `youbot_get_status` - Get complete robot state

## Base Movement (Mecanum Wheels)
- `youbot_move_base` - Set continuous velocities (vx, vy, omega)
- `youbot_stop_base` - Stop all movement
- `youbot_move_forward` - Move specific distance
- `youbot_rotate` - Rotate specific angle

## Arm Control (5-DOF)
- `youbot_set_arm_height` - Set height preset (FLOOR, PLATE, RESET)
- `youbot_set_arm_orientation` - Set rotation (FRONT, LEFT, BACK)
- `youbot_set_arm_position` - Inverse kinematics (x, y, z)
- `youbot_reset_arm` - Return to rest position

## Gripper
- `youbot_grip` - Close gripper
- `youbot_release` - Open gripper
- `youbot_set_gripper_gap` - Set specific opening

## Perception
- `youbot_get_camera_image` - Capture RGB image
- `youbot_detect_cubes` - Find colored cubes
- `youbot_get_lidar_data` - Get obstacle ranges

## High-Level Tasks
- `youbot_grasp_cube` - Complete grasp sequence
- `youbot_deposit_cube` - Navigate and deposit
- `youbot_start_autonomous` - Start full task
- `youbot_stop_autonomous` - Stop task

## Info
- `youbot_get_deposit_boxes` - Box coordinates
- `youbot_list_capabilities` - This list

## Simulation Control (macOS)
- `youbot_reload_world` - Reload Webots world (Cmd+Shift+R)
"""


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    logger.info("Starting YouBot MCP Server")
    logger.info(f"Data directory: {DATA_DIR}")
    mcp.run()
