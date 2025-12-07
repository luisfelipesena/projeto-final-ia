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
        
        # Get gripper finger motors (both sides)
        self.finger_left = robot.getDevice("finger::left")
        self.finger_right = robot.getDevice("finger::right")

        self.sensor_left = None
        self.sensor_right = None

        for motor_attr in ("finger_left", "finger_right"):
            motor = getattr(self, motor_attr)
            if motor:
                motor.setVelocity(0.03)
                sensor = motor.getPositionSensor()
                if sensor:
                    sensor.enable(self.time_step)
                if motor_attr == "finger_left":
                    self.sensor_left = sensor
                else:
                    self.sensor_right = sensor
            else:
                print(f"Warning: Could not find gripper motor '{motor_attr.replace('_', '::')}'")

        # Alias for backward compatibility
        self.finger = self.finger_left

        # Current state
        self.is_gripping = False
    
    def grip(self):
        """Close gripper to grip an object"""
        if self.finger_left:
            self.finger_left.setPosition(MIN_POS)
        if self.finger_right:
            self.finger_right.setPosition(MIN_POS)
        self.is_gripping = True
    
    def release(self):
        """Open gripper to release an object"""
        if self.finger_left:
            self.finger_left.setPosition(MAX_POS)
        if self.finger_right:
            self.finger_right.setPosition(MAX_POS)
        self.is_gripping = False
    
    def set_gap(self, gap):
        """Set gripper to a specific gap width between fingers
        
        Args:
            gap: desired gap between fingers in meters
        """
        # Calculate motor position with offset compensation
        v = bound(0.5 * (gap - OFFSET_WHEN_LOCKED), MIN_POS, MAX_POS)
        
        if self.finger_left:
            self.finger_left.setPosition(v)
        if self.finger_right:
            self.finger_right.setPosition(v)
        
        self.is_gripping = v < MAX_POS / 2
    
    def is_closed(self):
        """Check if gripper is in closed/gripping state
        
        Returns:
            bool: True if gripper is gripping
        """
        return self.is_gripping

    def finger_positions(self):
        """Return current finger sensor readings if available"""
        left = self.sensor_left.getValue() if self.sensor_left else None
        right = self.sensor_right.getValue() if self.sensor_right else None
        return left, right

    def has_object(self, threshold=0.002):
        """Detect if an object is held based on finger positions"""
        left, right = self.finger_positions()
        samples = [v for v in (left, right) if v is not None]
        if not samples:
            return self.is_gripping
        return all(v <= threshold for v in samples)