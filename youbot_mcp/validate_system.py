
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration
MCP_DIR = Path(__file__).parent
DATA_DIR = MCP_DIR / "data" / "youbot"
COMMANDS_FILE = DATA_DIR / "commands.json"
STATUS_FILE = DATA_DIR / "status.json"

class YouBotValidator:
    def __init__(self):
        self.command_id = 1
        self.start_time = time.time()
        self.cubes_collected = 0
        self.last_state = "UNKNOWN"
        self.state_start_time = time.time()
        
        # Ensure data dir exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    def send_command(self, action: str, params: Dict[str, Any] = None):
        """Send command to robot."""
        cmd = {
            "id": self.command_id,
            "action": action,
            "params": params or {},
            "timestamp": time.time()
        }
        
        with open(COMMANDS_FILE, 'w') as f:
            json.dump(cmd, f, indent=2)
            
        print(f"[Validator] Sent command {self.command_id}: {action}")
        self.command_id += 1
        
    def read_status(self) -> Optional[Dict[str, Any]]:
        """Read robot status."""
        try:
            if not STATUS_FILE.exists():
                return None
                
            with open(STATUS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Validator] Error reading status: {e}")
            return None

    def run(self):
        print("Starting YouBot End-to-End Validation...")
        print("Please ensure Webots simulation is running with youbot_mcp_controller.")
        
        # Wait for status file (robot ready)
        print("Waiting for robot connection...")
        while True:
            status = self.read_status()
            if status:
                print(f"Robot connected! Version: {status.get('version')}")
                break
            time.sleep(1)
            
        # Start autonomous mode
        print("Starting autonomous mode...")
        self.send_command("start_autonomous")
        
        # Monitor loop
        try:
            while True:
                status = self.read_status()
                if not status:
                    time.sleep(0.5)
                    continue
                    
                current_state = status.get("current_state", "UNKNOWN")
                collected = status.get("cubes_collected", 0)
                
                # State change detection
                if current_state != self.last_state:
                    duration = time.time() - self.state_start_time
                    print(f"[State] {self.last_state} -> {current_state} (took {duration:.1f}s)")
                    self.last_state = current_state
                    self.state_start_time = time.time()
                    
                # Cube collection detection
                if collected > self.cubes_collected:
                    print(f"[SUCCESS] Cube collected! Total: {collected}/15")
                    self.cubes_collected = collected
                    
                # Timeout checks
                state_duration = time.time() - self.state_start_time
                if current_state == "SEARCHING" and state_duration > 60:
                    print("[WARNING] Stuck in SEARCHING for 60s - check vision or navigation")
                elif current_state == "APPROACHING" and state_duration > 30:
                    print("[WARNING] Stuck in APPROACHING for 30s - check approach logic")
                elif current_state == "GRASPING" and state_duration > 15:
                    print("[WARNING] Stuck in GRASPING for 15s - check arm/gripper")
                    
                # Completion check
                if collected >= 15:
                    print("\n[VICTORY] All 15 cubes collected!")
                    break
                    
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nValidation stopped by user.")
            self.send_command("stop_autonomous")

if __name__ == "__main__":
    validator = YouBotValidator()
    validator.run()
