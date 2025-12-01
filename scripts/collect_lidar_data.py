"""
LIDAR Data Collection Script for Neural Network Training.

Run this in Webots to collect labeled LIDAR data for training the
obstacle detection MLP.

Usage:
1. Copy to Webots controller folder
2. Set as controller for YouBot
3. Run simulation and drive robot manually or let it wander
4. Data is automatically labeled based on distance thresholds
5. Stop simulation to save data to file

The script saves:
- Raw LIDAR readings (512 points)
- Labels for each sector (9 sectors: obstacle/clear)
"""

from controller import Robot, Keyboard
import numpy as np
import json
import os
from datetime import datetime

# Configuration
NUM_POINTS = 512
NUM_SECTORS = 9
OBSTACLE_THRESHOLD = 0.5  # meters
MIN_RANGE = 0.01
MAX_RANGE = 5.0
COLLECTION_INTERVAL = 5  # Collect every N steps


class LidarDataCollector:
    """Collects and labels LIDAR data for NN training."""

    def __init__(self, robot: Robot):
        """Initialize collector.

        Args:
            robot: Webots Robot instance
        """
        self.robot = robot
        self.time_step = int(robot.getBasicTimeStep())

        # Initialize LIDAR
        self.lidar = robot.getDevice("lidar")
        if self.lidar:
            self.lidar.enable(self.time_step)
            print(f"[COLLECTOR] LIDAR enabled: {self.lidar.getNumberOfPoints()} points")
        else:
            raise RuntimeError("LIDAR not found")

        # Initialize keyboard for manual control
        self.keyboard = robot.getKeyboard()
        self.keyboard.enable(self.time_step)

        # Initialize base motors for manual driving
        self.wheels = []
        for name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = robot.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)

        # Data storage
        self.data = {
            'lidar_readings': [],
            'sector_labels': [],
            'timestamps': [],
            'metadata': {
                'num_points': NUM_POINTS,
                'num_sectors': NUM_SECTORS,
                'obstacle_threshold': OBSTACLE_THRESHOLD,
                'collection_date': datetime.now().isoformat(),
            }
        }

        self.step_count = 0
        self.samples_collected = 0

    def _get_lidar_data(self) -> np.ndarray:
        """Get current LIDAR range data."""
        ranges = self.lidar.getRangeImage()
        if ranges is None:
            return None

        data = np.array(ranges, dtype=np.float32)

        # Filter invalid readings
        data = np.where(
            (data > MIN_RANGE) & (data < MAX_RANGE),
            data,
            MAX_RANGE
        )

        return data

    def _compute_labels(self, ranges: np.ndarray) -> np.ndarray:
        """Compute obstacle labels for each sector.

        Args:
            ranges: LIDAR range data (512 points)

        Returns:
            Binary labels for 9 sectors (1 = obstacle, 0 = clear)
        """
        points_per_sector = NUM_POINTS // NUM_SECTORS
        labels = np.zeros(NUM_SECTORS, dtype=np.int32)

        for i in range(NUM_SECTORS):
            start = i * points_per_sector
            end = start + points_per_sector
            sector_min = np.min(ranges[start:end])

            # Label as obstacle if any point below threshold
            labels[i] = 1 if sector_min < OBSTACLE_THRESHOLD else 0

        return labels

    def _handle_keyboard(self) -> None:
        """Handle keyboard input for manual driving."""
        key = self.keyboard.getKey()

        speed = 4.0
        speeds = [0.0, 0.0, 0.0, 0.0]

        if key == ord('W') or key == Keyboard.UP:
            speeds = [speed, speed, speed, speed]
        elif key == ord('S') or key == Keyboard.DOWN:
            speeds = [-speed, -speed, -speed, -speed]
        elif key == ord('A') or key == Keyboard.LEFT:
            speeds = [-speed, speed, -speed, speed]
        elif key == ord('D') or key == Keyboard.RIGHT:
            speeds = [speed, -speed, speed, -speed]
        elif key == ord('Q'):  # Strafe left
            speeds = [speed, -speed, -speed, speed]
        elif key == ord('E'):  # Strafe right
            speeds = [-speed, speed, speed, -speed]

        for wheel, spd in zip(self.wheels, speeds):
            wheel.setVelocity(spd)

    def collect_sample(self) -> bool:
        """Collect one labeled sample.

        Returns:
            True if sample was collected
        """
        ranges = self._get_lidar_data()
        if ranges is None:
            return False

        labels = self._compute_labels(ranges)

        # Store sample
        self.data['lidar_readings'].append(ranges.tolist())
        self.data['sector_labels'].append(labels.tolist())
        self.data['timestamps'].append(self.robot.getTime())

        self.samples_collected += 1

        if self.samples_collected % 100 == 0:
            print(f"[COLLECTOR] Samples collected: {self.samples_collected}")

        return True

    def save_data(self, filepath: str = None) -> str:
        """Save collected data to file.

        Args:
            filepath: Output file path (default: auto-generated)

        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/lidar/lidar_data_{timestamp}.json"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Update metadata
        self.data['metadata']['total_samples'] = self.samples_collected
        self.data['metadata']['duration_seconds'] = self.robot.getTime()

        with open(filepath, 'w') as f:
            json.dump(self.data, f)

        print(f"[COLLECTOR] Data saved to {filepath}")
        print(f"[COLLECTOR] Total samples: {self.samples_collected}")

        return filepath

    def run(self) -> None:
        """Main collection loop."""
        print("[COLLECTOR] Starting data collection")
        print("[COLLECTOR] Controls: WASD/Arrows to drive, Q/E to strafe")
        print("[COLLECTOR] Data is collected automatically while driving")
        print("[COLLECTOR] Stop simulation to save data")

        try:
            while self.robot.step(self.time_step) != -1:
                self.step_count += 1
                self._handle_keyboard()

                # Collect at intervals
                if self.step_count % COLLECTION_INTERVAL == 0:
                    self.collect_sample()

        except KeyboardInterrupt:
            pass
        finally:
            self.save_data()


def main():
    """Entry point."""
    robot = Robot()
    collector = LidarDataCollector(robot)
    collector.run()


if __name__ == "__main__":
    main()
