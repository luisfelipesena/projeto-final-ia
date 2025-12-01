# MCP Validation Scenarios (Post-Fix)

These scenarios exercise the new perception + controller changes without modifying the Webots world. Follow the launch steps from `docs/MCP_PUBLISH_GUIDE.md` (activate `venv`, run Webots with `--mcp`, and keep `youbot_mcp` tail open).

## Scenario A – Box-Close Pickups

Purpose: Ensure HSV/LIDAR gating ignores deposit boxes while creeping into the final approach phase.

1. In Webots, drag 3 cubes within 0.3 m of the green and blue bins before starting the controller.
2. Start the YouBot MCP mode. Observe in `youbot_mcp.log`:
   - `[VisionService] Locked to ...` lines should now alternate less frequently.
   - `[Startup] Front clearance ... backing up` should appear only if the run begins too close to the wall.
   - `[Navigation] FINAL phase engaged` and `[Navigation] FINAL dwell satisfied` should appear before `[Grasping]` for close cubes.
3. Let the robot attempt at least 5 pickups. Capture screenshots showing the robot slowing down near the bins and add them to `assets/` for reference.

## Scenario B – Obstacle Corridor Stress Test

Purpose: Validate obstacle-map integration and lateral dodge rules.

1. Using the supervisor, position 4 wooden boxes in a corridor approximately 0.9 m wide.
2. Place cubes at the far side of the corridor.
3. Start MCP mode and monitor:
   - `[Navigation] OBSTACLE ... lateral dodge` events should be followed by `[Navigation] FINAL phase engaged` instead of repeated max-attempt failures.
   - `[Grasping] Clearance check failed` should **not** appear.
   - `[Navigation] FINAL dwell satisfied` should precede `[Grasping]` once the cube is within 0.28 m.
4. When a grasp succeeds, check that `grasp_log.txt` records the finger sensor delta around 0.003 and front LIDAR clearance >0.2 m.

## Evidence to Capture

- Append relevant excerpts to `youbot_mcp/youbot_mcp.log` (or copy to a dated file) for each scenario.
- Add two new PNG captures per scenario to `assets/` with filenames `validation-bin-*.png` and `validation-corridor-*.png`.
- Reference these artifacts when updating `DECISIONS.md` and the final presentation.
- Note in your log excerpts whether `[Startup]`, `[Navigation] FINAL dwell satisfied`, and `[Grasping]` appear in order; this confirms the new handoff logic is working.

> **Tip:** If Webots cannot be run on this machine, execute the scenarios on the lab workstation, then copy the updated `assets/` and `youbot_mcp.log` back into this repository before committing.
