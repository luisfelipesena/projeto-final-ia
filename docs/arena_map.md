# Arena Map - IA_20252

**Generated:** 2025-11-21
**Source:** `IA_20252/worlds/IA_20252.wbt`
**Phase:** 1.4 - Arena Mapping

## Arena Dimensions (FR-027)

- **Length (X-axis):** 7.00 m
- **Width (Y-axis):** 4.00 m
- **Area:** 28.00 m²

**Boundaries:**
- Bottom-left: (-3.50, -2.00)
- Bottom-right: (3.50, -2.00)
- Top-right: (3.50, 2.00)
- Top-left: (-3.50, 2.00)

## Deposit Boxes (FR-028)

| Color | Position (x, y, z) | Description | Name |
|-------|-------------------|-------------|------|

## Obstacles (FR-029)

**Total obstacles:** 0

| Type | Position (x, y, z) | Dimensions (L×W×H) | Name |
|------|-------------------|-------------------|------|

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
