"""
Copyright 1996-2024 Cyberbotics Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Description: Python wrapper for YouBot gripper control
"""

# Gripper positions
MIN_POS = 0.0
MAX_POS = 0.025
OFFSET_WHEN_LOCKED = 0.021

def bound(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))

class Gripper:
    """Controls the YouBot parallel gripper"""
    
    def __init__(self, robot):
        """Initialize gripper motors

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Get gripper finger motor (single motor controls both fingers)
        self.finger = robot.getDevice("finger::left")

        # Get position sensor from motor
        self.finger_sensor = None
        if self.finger:
            self.finger_sensor = self.finger.getPositionSensor()
            if self.finger_sensor:
                self.finger_sensor.enable(self.time_step)
            self.finger.setVelocity(0.03)
        else:
            print("Warning: Could not find gripper motor 'finger::left'")

        # Current state
        self.is_gripping = False
    
    def grip(self):
        """Close gripper to grip an object"""
        if self.finger:
            self.finger.setPosition(MIN_POS)
        self.is_gripping = True
    
    def release(self):
        """Open gripper to release an object"""
        if self.finger:
            self.finger.setPosition(MAX_POS)
        self.is_gripping = False
    
    def set_gap(self, gap):
        """Set gripper to a specific gap width between fingers
        
        Args:
            gap: desired gap between fingers in meters
        """
        # Calculate motor position with offset compensation
        v = bound(0.5 * (gap - OFFSET_WHEN_LOCKED), MIN_POS, MAX_POS)
        
        if self.finger:
            self.finger.setPosition(v)
        
        self.is_gripping = (v < MAX_POS / 2)
    
    def is_closed(self):
        """Check if gripper is in closed/gripping state

        Returns:
            bool: True if gripper is gripping
        """
        return self.is_gripping

    def has_object(self):
        """Check if gripper has grasped an object.

        Object detected if fingers didn't fully close (stopped by object).

        Returns:
            bool: True if object detected between fingers
        """
        if not self.finger_sensor:
            # CRITICAL: Without sensor, cannot detect object reliably
            # Return False to prevent false positives (grasping thin air)
            print("[GRIPPER] WARNING: No sensor - cannot detect object")
            return False

        pos = self.finger_sensor.getValue()
        # If gripping but position > MIN_POS + threshold, object is present
        # 3cm cube would stop fingers at ~0.008-0.015
        # Increased threshold to 0.006 to reduce false positives
        threshold = 0.006
        has_obj = self.is_gripping and pos > (MIN_POS + threshold)
        print(f"[GRIPPER] has_object: pos={pos:.4f}, threshold={MIN_POS + threshold:.4f}, result={has_obj}")
        return has_obj