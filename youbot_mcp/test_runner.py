#!/usr/bin/env python3
"""
Autonomous test runner for YouBot MCP
Monitors status.json and grasp_log.txt, reports outcomes.

Usage:
    python test_runner.py [timeout_seconds]

The test runner will:
1. Send start_autonomous command to the controller
2. Monitor status.json for state changes
3. Parse grasp_log.txt for finger_pos values
4. Report success/failure metrics
"""
import json
import time
import re
import sys
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data" / "youbot"
STATUS_FILE = DATA_DIR / "status.json"
GRASP_LOG = DATA_DIR / "grasp_log.txt"
COMMANDS_FILE = DATA_DIR / "commands.json"
NAV_LOG = DATA_DIR / "nav_debug.log"


class TestRunner:
    def __init__(self):
        self.grasp_attempts = 0
        self.grasp_successes = 0
        self.finger_positions = []
        self.state_history = []
        self.start_time = None

    def send_command(self, action: str, params: dict = None):
        """Send command to controller via commands.json"""
        cmd = {
            "id": int(time.time() * 1000),
            "action": action,
            "params": params or {}
        }
        COMMANDS_FILE.write_text(json.dumps(cmd, indent=2))
        print(f"[CMD] Sent: {action}")

    def read_status(self) -> dict:
        """Read current status from status.json"""
        try:
            text = STATUS_FILE.read_text()
            # Handle Infinity values in JSON
            text = re.sub(r': Infinity', ': 999', text)
            text = re.sub(r': -Infinity', ': -999', text)
            return json.loads(text)
        except Exception as e:
            return {}

    def parse_grasp_log(self) -> list:
        """Parse grasp_log.txt for finger_pos values"""
        results = []
        if GRASP_LOG.exists():
            text = GRASP_LOG.read_text()
            for line in text.split('\n'):
                if 'finger_pos_after_close' in line:
                    match = re.search(r'finger_pos_after_close=([\d.]+)', line)
                    if match:
                        results.append(float(match.group(1)))
        return results

    def clear_logs(self):
        """Clear old log files for fresh test"""
        if GRASP_LOG.exists():
            GRASP_LOG.write_text("")
        if NAV_LOG.exists():
            NAV_LOG.write_text("")
        print("[INIT] Cleared old logs")

    def wait_for_controller(self, timeout=30) -> bool:
        """Wait for controller to be ready (status.json updating)"""
        print("[INIT] Waiting for controller...")
        start = time.time()
        last_update = 0

        while time.time() - start < timeout:
            status = self.read_status()
            if status:
                update_time = status.get('last_update', 0)
                if update_time > last_update:
                    print(f"[INIT] Controller ready (version: {status.get('version', 'unknown')})")
                    return True
                last_update = update_time
            time.sleep(0.5)

        print("[ERROR] Controller not responding")
        return False

    def run_test(self, timeout_seconds=300) -> dict:
        """Run autonomous test and return metrics"""
        self.start_time = time.time()

        # Initialize
        self.clear_logs()

        # Wait for controller
        if not self.wait_for_controller():
            return self._report("CONTROLLER_ERROR", 0)

        # Start autonomous mode
        self.send_command("start_autonomous")
        time.sleep(1)

        last_state = None
        last_cubes = 0
        stuck_counter = 0
        grasping_counter = 0

        print(f"\n[TEST] Starting autonomous test (timeout: {timeout_seconds}s)")
        print("=" * 60)

        while time.time() - self.start_time < timeout_seconds:
            status = self.read_status()
            if not status:
                time.sleep(0.5)
                continue

            state = status.get('current_state', 'UNKNOWN')
            cubes = status.get('cubes_collected', 0)
            target = status.get('current_target')
            autonomous = status.get('autonomous_mode', False)

            # Check if autonomous mode is still on
            if not autonomous and last_state is not None:
                print(f"[WARN] Autonomous mode disabled!")

            # Log state changes
            if state != last_state:
                elapsed = time.time() - self.start_time
                target_info = ""
                if target:
                    target_info = f" -> {target.get('color', '?')} at {target.get('distance', 0):.2f}m, {target.get('angle', 0):.1f}deg"
                print(f"[{elapsed:6.1f}s] {state:12}{target_info} (cubes: {cubes}/15)")
                self.state_history.append((elapsed, state, cubes))
                last_state = state
                stuck_counter = 0

                # Reset grasping counter on state change
                if state != 'GRASPING':
                    grasping_counter = 0

            # Track grasping state
            if state == 'GRASPING':
                grasping_counter += 1
                # Check grasp log for new entries every second
                if grasping_counter % 2 == 0:
                    finger_pos = self.parse_grasp_log()
                    new_attempts = len(finger_pos) - len(self.finger_positions)
                    if new_attempts > 0:
                        for i in range(new_attempts):
                            pos = finger_pos[len(self.finger_positions) + i]
                            success = "SUCCESS" if pos > 0.003 else "FAILED"
                            print(f"[GRASP] Attempt: finger_pos={pos:.5f} -> {success}")
                        self.finger_positions = finger_pos

            # Detect progress
            if cubes > last_cubes:
                print(f"[SUCCESS] Cube #{cubes} collected!")
                last_cubes = cubes

            # Detect stuck
            stuck_counter += 1
            if stuck_counter > 120:  # 60 seconds stuck
                print(f"[WARN] Stuck in {state} for 60s")
                stuck_counter = 0

            # Check completion
            if cubes >= 15:
                return self._report("SUCCESS", cubes)

            time.sleep(0.5)

        return self._report("TIMEOUT", status.get('cubes_collected', 0))

    def _report(self, result: str, cubes: int) -> dict:
        """Generate final test report"""
        finger_pos = self.parse_grasp_log()
        elapsed = time.time() - self.start_time if self.start_time else 0

        grasp_successes = sum(1 for p in finger_pos if p > 0.003)
        grasp_failures = len(finger_pos) - grasp_successes

        report = {
            "result": result,
            "cubes_collected": cubes,
            "duration_seconds": elapsed,
            "grasp_attempts": len(finger_pos),
            "grasp_successes": grasp_successes,
            "grasp_failures": grasp_failures,
            "finger_positions": finger_pos,
            "state_history": self.state_history
        }

        print("\n" + "=" * 60)
        print(f"TEST RESULT: {result}")
        print("=" * 60)
        print(f"  Cubes collected:  {cubes}/15")
        print(f"  Duration:         {elapsed:.1f}s")
        print(f"  Grasp attempts:   {len(finger_pos)}")
        print(f"  Grasp successes:  {grasp_successes}")
        print(f"  Grasp failures:   {grasp_failures}")

        if finger_pos:
            print(f"\n  Finger positions:")
            for i, pos in enumerate(finger_pos):
                status = "OK (object)" if pos > 0.003 else "FAIL (empty)"
                print(f"    #{i+1}: {pos:.5f} -> {status}")

        print("\n  State history:")
        for elapsed_t, state, cube_count in self.state_history[:20]:  # First 20 transitions
            print(f"    [{elapsed_t:6.1f}s] {state} (cubes: {cube_count})")
        if len(self.state_history) > 20:
            print(f"    ... and {len(self.state_history) - 20} more transitions")

        print("=" * 60)

        # Save report to file
        report_file = DATA_DIR / f"test_report_{int(time.time())}.json"
        report_file.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nReport saved to: {report_file}")

        return report

    def stop_test(self):
        """Stop autonomous mode"""
        self.send_command("stop_autonomous")
        print("[CMD] Stopped autonomous mode")


def main():
    timeout = 300  # 5 minutes default
    if len(sys.argv) > 1:
        try:
            timeout = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [timeout_seconds]")
            sys.exit(1)

    print(f"""
================================================================================
YouBot MCP Test Runner
================================================================================
Timeout:     {timeout}s
Status file: {STATUS_FILE}
Grasp log:   {GRASP_LOG}
Commands:    {COMMANDS_FILE}
================================================================================
""")

    runner = TestRunner()
    try:
        result = runner.run_test(timeout_seconds=timeout)
        sys.exit(0 if result['result'] == 'SUCCESS' else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping test...")
        runner.stop_test()
        sys.exit(2)


if __name__ == "__main__":
    main()
