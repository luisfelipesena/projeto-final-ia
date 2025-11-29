#!/usr/bin/env python3
"""
Webots Simulation Monitor

Real-time monitoring of YouBot simulation via log files.
Run this while Webots simulation is running to see status.

Usage:
    python scripts/webots_monitor.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import re

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
MAIN_LOG = LOGS_DIR / "main_controller.log"
STATE_LOG = LOGS_DIR / "state_transitions.log"


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def tail_file(filepath: Path, n_lines: int = 20) -> list:
    """Get last N lines from file"""
    if not filepath.exists():
        return []

    with open(filepath, 'r') as f:
        lines = f.readlines()
    return lines[-n_lines:]


def parse_state_line(line: str) -> dict:
    """Parse state transition line"""
    # Pattern: [timestamp] INFO: Transition: STATE1 → STATE2 (context: {...})
    match = re.search(r'Transition: (\w+) → (\w+)', line)
    if match:
        return {
            'from': match.group(1),
            'to': match.group(2),
            'line': line.strip()
        }
    return None


def parse_search_line(line: str) -> dict:
    """Parse SEARCH debug line"""
    # Pattern: [SEARCH] cube_detected=X, cube_dist=X.XXm, obstacle_dist=X.XXm
    if '[SEARCH]' in line:
        cube_det = 'True' in line and 'cube_detected=True' in line
        dist_match = re.search(r'cube_dist=(\d+\.\d+)', line)
        obs_match = re.search(r'obstacle_dist=(\d+\.\d+)', line)
        return {
            'cube_detected': cube_det,
            'cube_dist': float(dist_match.group(1)) if dist_match else None,
            'obstacle_dist': float(obs_match.group(1)) if obs_match else None
        }
    return None


def parse_nav_line(line: str) -> dict:
    """Parse NAV debug line"""
    # Pattern: NAV: pose=(X.XX, Y.XX, θ=XX.X°) → target=(X.XX, Y.XX) dist=X.XXm
    if 'NAV:' in line:
        pose_match = re.search(r'pose=\(([-\d.]+), ([-\d.]+)', line)
        target_match = re.search(r'target=\(([-\d.]+), ([-\d.]+)\)', line)
        dist_match = re.search(r'dist=([\d.]+)m', line)
        return {
            'robot_x': float(pose_match.group(1)) if pose_match else None,
            'robot_y': float(pose_match.group(2)) if pose_match else None,
            'target_x': float(target_match.group(1)) if target_match else None,
            'target_y': float(target_match.group(2)) if target_match else None,
            'distance': float(dist_match.group(1)) if dist_match else None
        }
    return None


def get_current_state(lines: list) -> str:
    """Get most recent state from log lines"""
    for line in reversed(lines):
        parsed = parse_state_line(line)
        if parsed:
            return parsed['to']
    return 'UNKNOWN'


def get_cubes_collected(lines: list) -> int:
    """Get cubes collected count"""
    for line in reversed(lines):
        match = re.search(r'Total: (\d+)/15', line)
        if match:
            return int(match.group(1))
    return 0


def monitor():
    """Main monitoring loop"""
    print("=" * 60)
    print("  YouBot Webots Monitor")
    print("  Press Ctrl+C to exit")
    print("=" * 60)
    print()

    if not LOGS_DIR.exists():
        print(f"[ERROR] Logs directory not found: {LOGS_DIR}")
        print("Make sure Webots simulation is running!")
        return

    last_state = None
    transition_count = 0

    try:
        while True:
            clear_screen()

            print("=" * 60)
            print(f"  YouBot Monitor | {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)

            # Read logs
            main_lines = tail_file(MAIN_LOG, 50)
            state_lines = tail_file(STATE_LOG, 20)

            # Current state
            current_state = get_current_state(state_lines)
            if current_state != last_state:
                transition_count += 1
                last_state = current_state

            cubes = get_cubes_collected(main_lines)

            print(f"\n  STATE: {current_state}")
            print(f"  CUBES: {cubes}/15")
            print(f"  TRANSITIONS: {transition_count}")

            # Parse recent search info
            print("\n  --- PERCEPTION ---")
            for line in reversed(main_lines):
                search_info = parse_search_line(line)
                if search_info:
                    print(f"  Cube detected: {search_info['cube_detected']}")
                    if search_info['cube_dist']:
                        print(f"  Cube distance: {search_info['cube_dist']:.2f}m")
                    if search_info['obstacle_dist']:
                        print(f"  Obstacle dist: {search_info['obstacle_dist']:.2f}m")
                    break

            # Parse navigation info
            for line in reversed(main_lines):
                nav_info = parse_nav_line(line)
                if nav_info and nav_info['robot_x'] is not None:
                    print(f"\n  --- NAVIGATION ---")
                    print(f"  Robot: ({nav_info['robot_x']:.2f}, {nav_info['robot_y']:.2f})")
                    if nav_info['target_x']:
                        print(f"  Target: ({nav_info['target_x']:.2f}, {nav_info['target_y']:.2f})")
                    if nav_info['distance']:
                        print(f"  Distance: {nav_info['distance']:.2f}m")
                    break

            # Recent transitions
            print("\n  --- RECENT TRANSITIONS ---")
            recent_transitions = []
            for line in state_lines[-10:]:
                parsed = parse_state_line(line)
                if parsed:
                    recent_transitions.append(f"  {parsed['from']} → {parsed['to']}")

            for t in recent_transitions[-5:]:
                print(t)

            # Color debug
            print("\n  --- COLOR DEBUG ---")
            for line in reversed(main_lines):
                if '[ColorDebug]' in line:
                    match = re.search(r'\[ColorDebug\] (.+)', line)
                    if match:
                        print(f"  {match.group(1)}")
                    break

            # Gripper debug
            for line in reversed(main_lines):
                if '[Gripper]' in line:
                    match = re.search(r'\[Gripper\] (.+)', line)
                    if match:
                        print(f"\n  --- GRIPPER ---")
                        print(f"  {match.group(1)}")
                    break

            print("\n" + "=" * 60)
            print("  [Ctrl+C to exit]")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    monitor()
