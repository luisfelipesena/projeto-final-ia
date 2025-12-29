"""
Strategic route planning for YouBot navigation.
Provides forward-oriented waypoints for car-like navigation.
"""

import math
from constants import BOX_POSITIONS, SPAWN_POSITION, KNOWN_OBSTACLES


def wrap_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def color_from_rgb(r, g, b):
    """Determine cube color from RGB values (0-1 range)."""
    if r > 0.5 and g < 0.5 and b < 0.5:
        return "red"
    elif g > 0.5 and r < 0.5 and b < 0.5:
        return "green"
    elif b > 0.5 and r < 0.5 and g < 0.5:
        return "blue"
    return None


def get_route_to_box(current_pos, destination_color):
    """
    Returns list of waypoints from current_pos to the destination box.
    
    Strategy: CAR-LIKE navigation with COMPLETE obstacle clearance.
    - Go STRAIGHT through center corridor until past obstacles E/F
    - INCLINE to avoid obstacles
    - MAINTAIN inclination until REAR WHEEL passes obstacle (~30cm extra)
    - Only then ALIGN with box
    
    Robot dimensions: ~58cm length, ~38cm width, rear wheel ~29cm from center.
    
    Obstacles:
    - E: (-1.02, 0.75) - upper left
    - F: (-1.02, -0.74) - lower left  
    - A: (0.6, 0.0) - central, radius ~0.25m
    """
    x, y = current_pos[0], current_pos[1]
    waypoints = []
    
    box_pos = BOX_POSITIONS.get(destination_color)
    if not box_pos:
        return [current_pos]
    
    if destination_color == "blue":
        # BLUE at (0.48, -1.62)
        # Position-aware routing

        if y < -1.2:
            # Already close, go direct
            waypoints.append((0.48, -1.35))
            waypoints.append(box_pos)
        elif y < -0.7:
            # In southern region
            if x < 0.35:
                waypoints.append((0.35, -1.25))
            waypoints.append((0.48, -1.35))
            waypoints.append(box_pos)
        elif y < -0.3:
            # Mid-south
            waypoints.append((0.25, -1.00))
            waypoints.append((0.35, -1.25))
            waypoints.append((0.48, -1.35))
            waypoints.append(box_pos)
        else:
            # Full route from north/center
            if x < -0.5:
                waypoints.append((-0.45, 0.0))
            if x < -0.15:
                waypoints.append((-0.15, 0.0))
            if x < 0.0:
                waypoints.append((0.0, -0.35))
            waypoints.append((0.15, -0.70))
            waypoints.append((0.25, -1.00))
            waypoints.append((0.35, -1.25))
            waypoints.append((0.48, -1.35))
            waypoints.append(box_pos)

    elif destination_color == "green":
        # GREEN at (0.48, 1.58)
        # Position-aware routing

        if y > 1.2:
            # Already close, go direct
            waypoints.append((0.48, 1.35))
            waypoints.append(box_pos)
        elif y > 0.7:
            # In northern region
            if x < 0.35:
                waypoints.append((0.35, 1.25))
            waypoints.append((0.48, 1.35))
            waypoints.append(box_pos)
        elif y > 0.3:
            # Mid-north
            waypoints.append((0.25, 1.00))
            waypoints.append((0.35, 1.25))
            waypoints.append((0.48, 1.35))
            waypoints.append(box_pos)
        else:
            # Full route from south/center
            if x < -0.5:
                waypoints.append((-0.45, 0.0))
            if x < -0.15:
                waypoints.append((-0.15, 0.0))
            if x < 0.0:
                waypoints.append((0.0, 0.35))
            waypoints.append((0.15, 0.70))
            waypoints.append((0.25, 1.00))
            waypoints.append((0.35, 1.25))
            waypoints.append((0.48, 1.35))
            waypoints.append(box_pos)
        
    elif destination_color == "red":
        # RED at (2.31, 0.01)
        # Obstacle A at (0.6, 0.0), radius ~0.25m
        # Only add waypoints that are AHEAD of current position

        # If already close to box, go direct
        if x > 1.8:
            # Already past obstacles, go direct to box
            if abs(y) > 0.3:
                waypoints.append((x + 0.1, y * 0.5))  # Move toward center
            waypoints.append((2.05, 0.01))
            waypoints.append(box_pos)
        elif x > 1.2:
            # Past obstacle A, converge to center path
            waypoints.append((1.55, 0.15 if y >= 0 else -0.15))
            waypoints.append((1.85, 0.01))
            waypoints.append((2.05, 0.01))
            waypoints.append(box_pos)
        elif x > 0.9:
            # Just past A, still need clearance before curving back
            if abs(y) > 0.50:
                # Already at high deviation, can start gradual return
                waypoints.append((1.30, 0.50 if y >= 0 else -0.50))
            waypoints.append((1.50, 0.30 if y >= 0 else -0.30))
            waypoints.append((1.70, 0.15 if y >= 0 else -0.15))
            waypoints.append((1.90, 0.01))
            waypoints.append((2.05, 0.01))
            waypoints.append(box_pos)
        else:
            # Full route needed - avoid obstacle A
            # CRITICAL: Start deviation EARLY and GRADUALLY (max 25° angle between waypoints)
            # Obstacle A at (0.6, 0.0), radius ~0.25m. Robot needs 55cm clearance.

            if x < -0.5:
                waypoints.append((-0.45, 0.0))

            # GRADUAL CURVE - start deviating BEFORE x=0 to avoid sharp turns
            if y >= 0:
                # North path - gradual curve with ~20° angle steps
                if x < -0.25:
                    waypoints.append((-0.25, 0.10))   # Start slight deviation early
                if x < 0.0:
                    waypoints.append((0.0, 0.25))     # Continue gradual curve
                if x < 0.25:
                    waypoints.append((0.25, 0.45))    # Increase deviation gradually
                if x < 0.50:
                    waypoints.append((0.50, 0.60))    # Continue north
                if x < 0.75:
                    waypoints.append((0.75, 0.70))    # Peak deviation past A center
                if x < 1.00:
                    waypoints.append((1.00, 0.70))    # Maintain - past A edge
                if x < 1.25:
                    waypoints.append((1.25, 0.60))    # Start gradual return
                if x < 1.45:
                    waypoints.append((1.45, 0.45))    # Continue curve back
                waypoints.append((1.60, 0.25))        # Almost aligned
            else:
                # South path - gradual curve with ~20° angle steps
                if x < -0.25:
                    waypoints.append((-0.25, -0.10))  # Start slight deviation early
                if x < 0.0:
                    waypoints.append((0.0, -0.25))    # Continue gradual curve
                if x < 0.25:
                    waypoints.append((0.25, -0.45))   # Increase deviation gradually
                if x < 0.50:
                    waypoints.append((0.50, -0.60))   # Continue south
                if x < 0.75:
                    waypoints.append((0.75, -0.70))   # Peak deviation past A center
                if x < 1.00:
                    waypoints.append((1.00, -0.70))   # Maintain - past A edge
                if x < 1.25:
                    waypoints.append((1.25, -0.60))   # Start gradual return
                if x < 1.45:
                    waypoints.append((1.45, -0.45))   # Continue curve back
                waypoints.append((1.60, -0.25))       # Almost aligned

            waypoints.append((1.75, 0.10 if y >= 0 else -0.10))
            waypoints.append((1.85, 0.01))
            waypoints.append((2.05, 0.01))
            waypoints.append(box_pos)
    
    return waypoints


def get_return_route(current_pos, from_color):
    """
    Returns strategic waypoints for returning to spawn after depositing a cube.

    IMPROVED: Higher density waypoints (0.40m spacing), better obstacle A avoidance.
    Obstacle A at (0.6, 0.0) with radius ~0.25m - must keep y > 0.30 or y < -0.30 when x > 0.
    """
    x, y = current_pos[0], current_pos[1]
    waypoints = []

    if from_color == "red":
        # RED box at (2.31, 0.01) - robot faces EAST, needs to go WEST
        # Strategy: Go SOUTH first (90° turn easier than 180°), then WEST
        # FIXED: Only add waypoints if robot is east of them
        if x > 1.5:
            waypoints.append((1.30, -0.40))   # South first
        if x > 1.0:
            waypoints.append((0.90, -0.50))   # Continue south-west
        if x > 0.5:
            waypoints.append((0.40, -0.40))   # Curve toward corridor
        if x > -0.05:
            waypoints.append((0.00, -0.20))   # Clear of A's influence
        if x > -0.40:
            waypoints.append((-0.35, 0.00))   # Corridor entry

    elif from_color == "green":
        # GREEN box at (0.48, 1.58) - robot faces NORTH, needs to go SOUTH-WEST
        # CRITICAL: Must stay NORTH of obstacle A (y > 0.30) until x < 0
        # FIXED: Only add eastern waypoints if robot is actually near the box (x > 0)
        if y > 0.8 and x > 0.0:
            waypoints.append((0.35, 0.50))    # Start south from box
        if y > 0.4 and x > 0.0:
            waypoints.append((0.15, 0.35))    # SAFE: north of A
        if x > -0.2 and x < 0.5:
            waypoints.append((-0.25, 0.30))   # Go WEST, still north of A
        if x > -0.55:
            waypoints.append((-0.50, 0.15))   # Continue west
        if x > -0.75:
            waypoints.append((-0.70, 0.00))   # Corridor entry

    elif from_color == "blue":
        # BLUE box at (0.48, -1.62) - robot faces SOUTH, needs to go NORTH-WEST
        # FIXED: Only add waypoints if robot is east of them
        if y < -0.8 and x > 0.3:
            waypoints.append((0.40, -0.50))   # Arc north
        if y < -0.3 and x > 0.2:
            waypoints.append((0.25, -0.25))   # Continue north
        if y > -0.3 and x > 0.15:
            # Near center with A nearby - go NORTH of A
            waypoints.append((0.20, 0.30))    # North of A
            waypoints.append((-0.10, 0.25))   # Continue west-north
        if x > -0.45:
            waypoints.append((-0.40, 0.10))   # Approach corridor
        if x > -0.70:
            waypoints.append((-0.65, 0.00))   # Corridor entry

    # Common corridor to spawn - HIGHER DENSITY (0.40m spacing)
    corridor_waypoints = [
        (-0.90, 0.00),    # Before E/F
        (-1.30, 0.00),    # Between E(-1.02, 0.75) and F(-1.02, -0.74)
        (-1.70, 0.00),    # Past E/F
        (-2.10, 0.00),    # Continue west
        (-2.50, 0.00),    # Clear area
        (-2.90, 0.00),    # Continue west
        (-3.30, 0.00),    # Approaching spawn
        SPAWN_POSITION    # (-3.91, 0.0)
    ]

    for wp in corridor_waypoints:
        # Only add if WEST of current position (with margin)
        if wp[0] < x - 0.25:
            waypoints.append(wp)

    # Always ensure spawn is final destination
    if not waypoints or waypoints[-1] != SPAWN_POSITION:
        waypoints.append(SPAWN_POSITION)

    return waypoints


def distance_to_obstacle(pos, obstacle):
    """Calculate distance from position to obstacle edge."""
    ox, oy, radius = obstacle
    dist = math.hypot(pos[0] - ox, pos[1] - oy)
    return dist - radius


def check_path_clearance(start, end, obstacles, min_clearance=0.30):
    """Check if straight path between two points has sufficient clearance."""
    # Sample points along the path
    num_samples = 10
    for i in range(num_samples + 1):
        t = i / num_samples
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        
        for obs in obstacles:
            if distance_to_obstacle((x, y), obs) < min_clearance:
                return False
    return True
