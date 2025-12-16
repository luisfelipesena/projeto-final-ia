"""
Occupancy Grid for navigation and path planning.
Implements A* pathfinding with obstacle avoidance.
"""

import heapq
import math


class OccupancyGrid:
    """Grid-based map for navigation with A* pathfinding."""
    
    # Cell states
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    BOX = 3
    CUBE = 4

    def __init__(self, arena_center, arena_size, cell_size=0.12):
        """Initialize occupancy grid.
        
        Args:
            arena_center: (x, y) center of arena
            arena_size: (width, height) of arena
            cell_size: size of each grid cell in meters
        """
        self.cell_size = cell_size
        self.min_x = arena_center[0] - arena_size[0] / 2.0
        self.min_y = arena_center[1] - arena_size[1] / 2.0
        self.max_x = arena_center[0] + arena_size[0] / 2.0
        self.max_y = arena_center[1] + arena_size[1] / 2.0
        self.width = int(math.ceil(arena_size[0] / cell_size))
        self.height = int(math.ceil(arena_size[1] / cell_size))
        self.grid = [
            [self.UNKNOWN for _ in range(self.width)] for _ in range(self.height)
        ]
        self.static_mask = [
            [False for _ in range(self.width)] for _ in range(self.height)
        ]

    def in_bounds(self, gx, gy):
        """Check if grid coordinates are within bounds."""
        return 0 <= gx < self.width and 0 <= gy < self.height

    def world_to_cell(self, x, y):
        """Convert world coordinates to grid cell coordinates."""
        if math.isnan(x) or math.isnan(y):
            return None
        gx = int((x - self.min_x) / self.cell_size)
        gy = int((y - self.min_y) / self.cell_size)
        if not self.in_bounds(gx, gy):
            return None
        return gx, gy

    def cell_to_world(self, gx, gy):
        """Convert grid cell coordinates to world coordinates (cell center)."""
        wx = self.min_x + (gx + 0.5) * self.cell_size
        wy = self.min_y + (gy + 0.5) * self.cell_size
        return wx, wy

    def get(self, gx, gy):
        """Get cell state at grid coordinates."""
        if not self.in_bounds(gx, gy):
            return self.UNKNOWN
        return self.grid[gy][gx]

    def set(self, gx, gy, value, static=False, overwrite_static=False):
        """Set cell state at grid coordinates.
        
        Args:
            gx, gy: grid coordinates
            value: new cell state
            static: mark cell as static (unchangeable)
            overwrite_static: allow overwriting static cells
            
        Returns:
            bool: True if cell was modified
        """
        if not self.in_bounds(gx, gy):
            return False
        if self.static_mask[gy][gx] and not overwrite_static:
            return False
        if self.grid[gy][gx] == value:
            if static and not self.static_mask[gy][gx]:
                self.static_mask[gy][gx] = True
            return False
        self.grid[gy][gx] = value
        if static:
            self.static_mask[gy][gx] = True
        return True

    def fill_disk(self, x, y, radius, value, static=False):
        """Fill a circular area with a value.
        
        Args:
            x, y: world coordinates of center
            radius: radius in meters
            value: cell state to fill
            static: mark cells as static
        """
        min_x = x - radius
        max_x = x + radius
        min_y = y - radius
        max_y = y + radius
        cell_min = self.world_to_cell(min_x, min_y)
        cell_max = self.world_to_cell(max_x, max_y)
        if cell_min is None or cell_max is None:
            return
        gx_min, gy_min = cell_min
        gx_max, gy_max = cell_max
        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                wx, wy = self.cell_to_world(gx, gy)
                if math.hypot(wx - x, wy - y) <= radius:
                    self.set(gx, gy, value, static=static)

    def fill_border(self, value, static=True):
        """Fill arena border with obstacle cells."""
        for gx in range(self.width):
            self.set(gx, 0, value, static=static)
            self.set(gx, self.height - 1, value, static=static)
        for gy in range(self.height):
            self.set(0, gy, value, static=static)
            self.set(self.width - 1, gy, value, static=static)

    def _bresenham(self, start, end):
        """Bresenham's line algorithm for ray casting."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        points = []
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def raycast(self, start_world, end_world, hit_state, free_state=None):
        """Cast ray from start to end, marking cells.
        
        Args:
            start_world: (x, y) start position in world coords
            end_world: (x, y) end position in world coords
            hit_state: state to mark the hit cell
            free_state: state to mark traversed cells (default: FREE)
            
        Returns:
            bool: True if the hit cell was modified
        """
        if free_state is None:
            free_state = self.FREE
        start_cell = self.world_to_cell(*start_world)
        end_cell = self.world_to_cell(*end_world)
        if start_cell is None or end_cell is None:
            return False
        line = self._bresenham(start_cell, end_cell)
        # Mark free cells until penultimate
        for gx, gy in line[:-1]:
            self.set(gx, gy, free_state)
        gx_hit, gy_hit = line[-1]
        return self.set(gx_hit, gy_hit, hit_state)

    def plan_path(self, start_world, goal_world):
        """A* pathfinding with fallback for blocked cells.
        
        Args:
            start_world: (x, y) start position in world coords
            goal_world: (x, y) goal position in world coords
            
        Returns:
            list: List of (x, y) waypoints in world coordinates
        """
        start_cell = self.world_to_cell(*start_world)
        goal_cell = self.world_to_cell(*goal_world)
        if start_cell is None or goal_cell is None:
            return []
        if start_cell == goal_cell:
            return []

        # If start cell is blocked, find nearest free cell
        if self.get(*start_cell) == self.OBSTACLE:
            start_cell = self._find_nearest_free_cell(start_cell)
            if start_cell is None:
                return []
        
        # If goal cell is blocked (except BOX which is destination), find adjacent
        goal_val = self.get(*goal_cell)
        if goal_val == self.OBSTACLE:
            goal_cell = self._find_nearest_free_cell(goal_cell)
            if goal_cell is None:
                return []

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        came_from = {}
        g_score = {start_cell: 0}
        goal = goal_cell

        visited = set()
        max_iterations = self.width * self.height

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == goal:
                break
            cx, cy = current
            # 8-directional movement for smoother paths
            neighbors = [
                (cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1),
                (cx + 1, cy + 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx - 1, cy - 1)
            ]
            for nx, ny in neighbors:
                if not self.in_bounds(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                cell_val = self.get(nx, ny)
                # Avoid OBSTACLE and BOX (except if it's the goal)
                if cell_val == self.OBSTACLE:
                    continue
                if cell_val == self.BOX and (nx, ny) != goal:
                    continue
                # Cost: 1.0 for orthogonal, 1.414 for diagonal
                is_diag = abs(nx - cx) + abs(ny - cy) == 2
                step_cost = 1.414 if is_diag else 1.0
                tentative_g = g_score[current] + step_cost
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, 1e9):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        if goal not in came_from and goal != start_cell:
            return []

        path_cells = []
        node = goal
        while node != start_cell:
            path_cells.append(node)
            node = came_from.get(node, start_cell)
            if node == start_cell:
                break
        path_cells.reverse()
        return [self.cell_to_world(gx, gy) for gx, gy in path_cells]

    def _find_nearest_free_cell(self, blocked_cell, max_radius=5):
        """Find the nearest FREE cell from a blocked cell."""
        bx, by = blocked_cell
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # Only border of square
                    nx, ny = bx + dx, by + dy
                    if self.in_bounds(nx, ny):
                        val = self.get(nx, ny)
                        if val != self.OBSTACLE and val != self.BOX:
                            return (nx, ny)
        return None
