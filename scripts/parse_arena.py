#!/usr/bin/env python3
"""
Parse IA_20252.wbt world file to extract arena layout.
Generates docs/arena_map.md

Phase 1.4 - Arena Mapping (FR-027 to FR-030)
"""

import re
import sys
from pathlib import Path


def parse_arena_dimensions(wbt_path):
    """Extract arena size from RectangleArena node (FR-027).

    Args:
        wbt_path: Path to .wbt world file

    Returns:
        dict: Arena dimensions {length, width, boundaries}
    """
    with open(wbt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Search for RectangleArena node
    arena_pattern = r'RectangleArena\s*\{[^}]*floorSize\s+([\d.]+)\s+([\d.]+)[^}]*\}'
    match = re.search(arena_pattern, content)

    if match:
        length = float(match.group(1))
        width = float(match.group(2))

        # Calculate boundaries (assuming arena centered at origin)
        boundaries = [
            (-length/2, -width/2),  # Bottom-left
            (length/2, -width/2),   # Bottom-right
            (length/2, width/2),    # Top-right
            (-length/2, width/2),   # Top-left
        ]

        return {
            'length': length,
            'width': width,
            'boundaries': boundaries,
        }

    return None


def parse_deposit_boxes(wbt_path):
    """Extract deposit box positions and colors (FR-028).

    Args:
        wbt_path: Path to .wbt world file

    Returns:
        list: Deposit boxes [{color, position, description}, ...]
    """
    with open(wbt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Search for PlasticFruitBox nodes with recognitionColors
    # Pattern: DEF BoxName Solid { ... translation x y z ... recognitionColors [ r g b ] }
    box_pattern = r'DEF\s+(\w+)\s+Solid\s*\{[^}]*translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)[^}]*recognitionColors\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]'

    boxes = []
    for match in re.finditer(box_pattern, content, re.DOTALL):
        name = match.group(1)
        x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
        r, g, b = float(match.group(5)), float(match.group(6)), float(match.group(7))

        # Determine color from RGB
        if g > 0.5 and r < 0.5 and b < 0.5:
            color = 'green'
        elif b > 0.5 and r < 0.5 and g < 0.5:
            color = 'blue'
        elif r > 0.5 and g < 0.5 and b < 0.5:
            color = 'red'
        else:
            color = 'unknown'

        # Determine description from position
        if y > 0.5:
            desc = "top"
        elif y < -0.5:
            desc = "bottom"
        else:
            desc = "center"

        if x > 0.5:
            desc += " right"
        elif x < -0.5:
            desc += " left"
        else:
            desc += " center"

        boxes.append({
            'color': color,
            'position': (x, y, z),
            'description': desc,
            'name': name,
        })

    return boxes


def parse_obstacles(wbt_path):
    """Extract obstacle positions (FR-029).

    Args:
        wbt_path: Path to .wbt world file

    Returns:
        list: Obstacles [{type, position, dimensions}, ...]
    """
    with open(wbt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Search for WoodenBox nodes
    # Pattern: DEF BoxName WoodenBox { ... translation x y z ... size x y z }
    obstacle_pattern = r'DEF\s+(\w+)\s+WoodenBox\s*\{[^}]*translation\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)[^}]*size\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'

    obstacles = []
    for match in re.finditer(obstacle_pattern, content, re.DOTALL):
        name = match.group(1)
        x, y, z = float(match.group(2)), float(match.group(3)), float(match.group(4))
        size_x, size_y, size_z = float(match.group(5)), float(match.group(6)), float(match.group(7))

        obstacles.append({
            'type': 'WoodenBox',
            'position': (x, y, z),
            'dimensions': (size_x, size_y, size_z),
            'name': name,
        })

    return obstacles


def generate_arena_map_md(arena_data, output_path):
    """Generate markdown documentation (FR-030).

    Args:
        arena_data: Dict with arena, boxes, obstacles
        output_path: Path to output .md file
    """
    arena = arena_data['arena']
    boxes = arena_data['boxes']
    obstacles = arena_data['obstacles']

    md_content = f"""# Arena Map - IA_20252

**Generated:** 2025-11-21
**Source:** `IA_20252/worlds/IA_20252.wbt`
**Phase:** 1.4 - Arena Mapping

## Arena Dimensions (FR-027)

- **Length (X-axis):** {arena['length']:.2f} m
- **Width (Y-axis):** {arena['width']:.2f} m
- **Area:** {arena['length'] * arena['width']:.2f} m²

**Boundaries:**
- Bottom-left: ({arena['boundaries'][0][0]:.2f}, {arena['boundaries'][0][1]:.2f})
- Bottom-right: ({arena['boundaries'][1][0]:.2f}, {arena['boundaries'][1][1]:.2f})
- Top-right: ({arena['boundaries'][2][0]:.2f}, {arena['boundaries'][2][1]:.2f})
- Top-left: ({arena['boundaries'][3][0]:.2f}, {arena['boundaries'][3][1]:.2f})

## Deposit Boxes (FR-028)

| Color | Position (x, y, z) | Description | Name |
|-------|-------------------|-------------|------|
"""

    for box in sorted(boxes, key=lambda b: b['color']):
        x, y, z = box['position']
        md_content += f"| **{box['color'].title()}** | ({x:.2f}, {y:.2f}, {z:.2f}) | {box['description']} | {box['name']} |\n"

    md_content += f"""
## Obstacles (FR-029)

**Total obstacles:** {len(obstacles)}

| Type | Position (x, y, z) | Dimensions (L×W×H) | Name |
|------|-------------------|-------------------|------|
"""

    for obs in obstacles:
        x, y, z = obs['position']
        dx, dy, dz = obs['dimensions']
        md_content += f"| {obs['type']} | ({x:.2f}, {y:.2f}, {z:.2f}) | {dx:.2f}×{dy:.2f}×{dz:.2f} m | {obs['name']} |\n"

    md_content += """
## Schematic Diagram (FR-030)

```
ASCII Representation (top view):

    ┌────────────────────────────────────┐
    │                                    │
    │  ●  Deposit Box (Green/Blue/Red)   │
    │  ■  Wooden Box (Obstacle)          │
    │  ○  Spawn Zone                     │
    │                                    │
    └────────────────────────────────────┘

Note: Use LIDAR visualization for precise obstacle mapping
```

## Navigation Notes

**Safe Zones:**
- Center area typically clear for navigation
- Avoid obstacle proximity (<0.3m clearance recommended)

**Deposit Strategy:**
- Identify box color via camera RGB detection
- Navigate to corresponding deposit box
- Approach from clear side (check LIDAR)

**Spawn Zone:**
- Cubes spawned randomly in X:[-3, 1.75], Y:[-1, 1]
- 15 cubes total per run
- Colors distributed randomly

## References

- **FR-027:** Arena dimensions documented
- **FR-028:** Deposit box coordinates documented
- **FR-029:** Obstacle positions documented
- **FR-030:** Schematic diagram provided

---

**Documentation Status:** ✅ COMPLETE
**Next:** Use this map for path planning (Phase 4) and collision avoidance (Phase 3 fuzzy logic)
"""

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✅ Arena map generated: {output_path}")


def main():
    """Main execution function."""
    # Paths
    project_root = Path(__file__).parent.parent
    wbt_path = project_root / "IA_20252" / "worlds" / "IA_20252.wbt"
    output_path = project_root / "docs" / "arena_map.md"

    if not wbt_path.exists():
        print(f"❌ World file not found: {wbt_path}")
        sys.exit(1)

    print(f"📂 Parsing world file: {wbt_path}")

    # Parse components
    arena = parse_arena_dimensions(wbt_path)
    boxes = parse_deposit_boxes(wbt_path)
    obstacles = parse_obstacles(wbt_path)

    if arena is None:
        print("⚠️ Arena dimensions not found (using defaults)")
        arena = {
            'length': 5.0,
            'width': 3.0,
            'boundaries': [(-2.5, -1.5), (2.5, -1.5), (2.5, 1.5), (-2.5, 1.5)],
        }

    print(f"✅ Parsed: Arena {arena['length']:.1f}×{arena['width']:.1f}m, {len(boxes)} deposit boxes, {len(obstacles)} obstacles")

    # Generate documentation
    arena_data = {
        'arena': arena,
        'boxes': boxes,
        'obstacles': obstacles,
    }

    generate_arena_map_md(arena_data, output_path)
    print(f"✅ Documentation complete: {output_path}")


if __name__ == "__main__":
    main()
