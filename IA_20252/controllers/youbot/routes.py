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
            # Just past A, need gradual return
            waypoints.append((1.25, 0.35 if y >= 0 else -0.35))
            waypoints.append((1.55, 0.15 if y >= 0 else -0.15))
            waypoints.append((1.85, 0.01))
            waypoints.append((2.05, 0.01))
            waypoints.append(box_pos)
        else:
            # Full route needed - avoid obstacle A
            if x < -0.5:
                waypoints.append((-0.45, 0.0))
            if x < 0.0:
                waypoints.append((0.0, 0.0))

            # Detour around A based on current y
            if y >= 0:
                if x < 0.25:
                    waypoints.append((0.25, 0.45))
                if x < 0.50:
                    waypoints.append((0.50, 0.55))
                if x < 0.75:
                    waypoints.append((0.75, 0.55))
                waypoints.append((1.00, 0.50))
                waypoints.append((1.25, 0.35))
            else:
                if x < 0.25:
                    waypoints.append((0.25, -0.45))
                if x < 0.50:
                    waypoints.append((0.50, -0.55))
                if x < 0.75:
                    waypoints.append((0.75, -0.55))
                waypoints.append((1.00, -0.50))
                waypoints.append((1.25, -0.35))

            waypoints.append((1.55, 0.15 if y >= 0 else -0.15))
            waypoints.append((1.85, 0.01))
            waypoints.append((2.05, 0.01))
            waypoints.append(box_pos)
    
    return waypoints


def get_return_route(current_pos, from_color):
    """
    Returns strategic waypoints for returning to spawn after depositing a cube.
    
    Strategy: Forward-oriented waypoints for car-like navigation.
    Now takes current_pos into account to generate adaptive waypoints.
    
    Key insight: After retreat phase, robot may be in different positions.
    We generate waypoints based on WHERE the robot actually is, not where
    we expected it to be.
    """
    x, y = current_pos[0], current_pos[1]
    waypoints = []
    
    if from_color == "red":
        # RED box at (2.31, 0.01)
        # Robot was facing EAST, now needs to go WEST
        # KEY INSIGHT: Going SOUTH first (90° right turn) is easier than 180° turn
        
        # Strategy: Go SOUTH to clear obstacles, then WEST through southern corridor
        if x > 1.0:
            # First go SOUTH - only requires ~90° right turn from EAST
            waypoints.append((1.20, -0.50))   # South first (easy turn)
            waypoints.append((0.80, -0.60))   # Continue south-west
        if x > 0.50:
            waypoints.append((0.40, -0.50))   # West through southern area
        if x > 0.0:
            waypoints.append((0.00, -0.30))   # Curve toward center
        waypoints.append((-0.30, -0.10))      # Into center corridor
        
    elif from_color == "green":
        # GREEN box at (0.48, 1.58)
        # Robot was facing NORTH, needs to go SOUTH-WEST
        # CRITICAL: Obstacle A at (0.6, 0.0) blocks direct path to center!
        # Must go WEST first, staying NORTH of A, then curve to corridor

        if y > 0.80:
            waypoints.append((0.40, 0.60))   # Start south from box
        if y > 0.40:
            waypoints.append((0.25, 0.40))   # Continue south

        # KEY FIX: Stay NORTH of obstacle A (0.6, 0.0) by keeping y > 0.25
        # Then go WEST before dropping to y=0 corridor
        if x > 0.0:
            waypoints.append((-0.10, 0.35))  # Go WEST, stay NORTH of A
        if x > -0.35:
            waypoints.append((-0.40, 0.25))  # Continue WEST, clear of A
        waypoints.append((-0.60, 0.10))      # Approach corridor from north
        waypoints.append((-0.80, 0.0))       # Into center corridor
        
    elif from_color == "blue":
        # BLUE box at (0.48, -1.62)
        # Robot was facing SOUTH, needs to go NORTH-WEST
        # After retreat, robot may end up at various Y positions
        
        if y < -0.80:
            waypoints.append((0.50, -0.60))  # Arc north
        if y < -0.40:
            waypoints.append((0.35, -0.30))  # Continue north
        
        # If robot is already in the center area (y > -0.40), 
        # we need waypoints that guide it WEST without requiring 180° turn
        if y > -0.40:
            # Robot is close to center, avoid obstacle A at (0.6, 0.0)
            if x > 0.40:
                # Need to go around obstacle A - go NORTH of it
                waypoints.append((0.35, 0.35))   # Go north of A
                waypoints.append((0.0, 0.30))    # Continue west-north
            elif x > 0.0:
                waypoints.append((-0.10, 0.15))  # Direct to corridor
            waypoints.append((-0.40, 0.05))      # Into corridor
        else:
            waypoints.append((0.10, -0.10))      # Toward center
            waypoints.append((-0.25, 0.0))       # Center corridor
    
    # Common path through center corridor to spawn
    # Only add waypoints that are WEST of current x position
    corridor_waypoints = [
        (-0.60, 0.0),    # Before E/F
        (-1.30, 0.0),    # Between E and F (corridor)
        (-1.80, 0.0),    # Past E/F
        (-2.50, 0.0),    # Clear area
        (-3.20, 0.0),    # Approaching spawn
        SPAWN_POSITION   # (-3.91, 0.0)
    ]
    
    for wp in corridor_waypoints:
        # Only add if west of current position (with small margin)
        if wp[0] < x - 0.30:
            waypoints.append(wp)
    
    # Always ensure spawn is the final destination
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
