# Failure Analysis – 2025-12-01 Run

## Evidence Sources

- Simulation log: `youbot_mcp/youbot_mcp.log` (latest autonomous run with supervisor spawning 15 cubes).
- Visual captures: `assets/CleanShot_2025-12-01_*.png` showing the robot dragging cubes and colliding with obstacles.
- World definition: `IA_20252/worlds/IA_20252.wbt` (device placement and camera resolution 128×128).
- LIDAR spec: Webots Reference Manual – [Lidar node](https://cyberbotics.com/doc/reference/lidar) (FoV / resolution semantics).

## Observed Failure Modes (from log + captures)

| Symptom | Log Evidence | Likely Root Cause |
| --- | --- | --- |
| Repeated `Approach failed: APPROACH/Lost` even when cube visible | Lines 11–215 show alternating `ALIGNED` → `APPROACH` loops ending with `max_attempts` or `target_lost`. | Vision/LIDAR disagreement on cube distance; Navigation never reaches 0.22 m threshold so the state machine oscillates. |
| Color lock mismatch (locks blue while physically pushing red cube) | Lines 117–170 show lock/unlock cycles across colors despite robot staying in same corridor. | HSV segmentation still triggers on large colored boxes and specular spots; no geo-validation before lock. |
| Robot drags cubes instead of grasping | After multiple failed approaches the robot continues with SEARCHING while physically touching cubes (see screenshots). No grasp ever triggered since `GRASPING` state is never reached. | Navigation service lacks final stage to slow down; movement speed remains 0.12 m/s and pushes cubes. |
| Collision with wooden obstacles after long approach loops | Lines 314–572 record repeated `OBSTACLE ... lateral dodge` at 0.25 m but robot keeps issuing forward commands; eventually touches boxes per images. | LIDAR obstacle handling only triggers when robot NOT in APPROACHING state, so approach loops ignore near-field obstacles. |

## Sensor Constraints to Respect

- **LIDAR**: Single-layer planar scanner mounted at `(0.28, 0, -0.07)` with `horizontalResolution = 512` and `fieldOfView ≈ 4.7 rad` (~269°). Each sample ≈0.526°. (`IA_20252/worlds/IA_20252.wbt` + [Webots Lidar reference](https://cyberbotics.com/doc/reference/lidar))
- **Camera**: Front RGB camera at `(0.27, 0, -0.06)` capturing `128×128` frames (RGBA from Webots → convert to RGB before HSV).
- **Approach geometry**: Camera optical center ~15 cm ahead of arm base, so distance estimates must subtract offset before feeding IK/grasp.

## Corrective Targets Derived from Analysis

1. **Vision gatekeeping**: reject detections unless (a) bounding box center is in lower 45% of frame, (b) solidity/aspect ratio near 1, and (c) nearest LIDAR hit at same azimuth is within ±5 cm of vision-estimated depth.
2. **Navigation phases**: add coarse vs. final approach speeds (0.12 m/s → 0.04 m/s when <0.4 m) and dynamic stop distance tied to cube size so we stop before physical contact.
3. **Obstacle fusion during approach**: while APPROACHING, continuously check LIDAR sectors; abort or sidestep if any front sector < cube distance − 0.1 m (prevents pushing/dragging or box collisions).
4. **State guarding before grasp**: verify LIDAR + vision consistency (distance delta < 5 cm) before calling `ArmService.prepare_grasp()` to avoid entering GRASPING when cube already moved by collision.

This document serves as the baseline reference for the remediation tasks executed in December 2025.
