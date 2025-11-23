#!/usr/bin/env python3
"""
LIDAR Data Collection Script

Collects LIDAR scans from Webots simulation for training obstacle detection neural network.
Saves scans as .npz files with 667 range values and 9-sector occupancy labels.

Usage:
    From Webots: Run this as a controller
    From Terminal (Mock): python scripts/collect_lidar_data.py --mock --pose-log poses.json
    Target: 1000+ scans with varied obstacle configurations

Data format:
    - ranges: [667] float32 (LIDAR measurements in meters)
    - labels: [9] float32 (binary occupancy per sector: 0=free, 1=occupied)
    - metadata: dict with timestamp, position, orientation, scenario_tag
"""

import sys
import os
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add Webots controller path only if not in mock mode
if '--mock' not in sys.argv:
    sys.path.append(os.path.join(os.environ.get('WEBOTS_HOME', ''), 'lib', 'controller', 'python'))
    from controller import Robot
else:
    Robot = None


class LIDARDataCollector:
    """Collects and labels LIDAR scans for neural network training"""

    SECTORS = 9  # 9 sectors × 30° each = 270° FOV
    SECTOR_ANGLE = 30.0  # degrees
    OBSTACLE_THRESHOLD = 2.0  # meters - distance to consider obstacle
    LIDAR_POINTS = 667

    def __init__(self, output_dir: str = "data/lidar", mock: bool = False, pose_log: str = None):
        self.mock = mock
        self.pose_log_path = pose_log
        self.pose_data = []
        self.current_pose_idx = 0

        if not self.mock:
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())

            # Initialize LIDAR
            self.lidar = self.robot.getDevice("lidar")
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

            # Initialize GPS for ground truth positioning
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.timestep)

            # Initialize compass for orientation
            self.compass = self.robot.getDevice("compass")
            self.compass.enable(self.timestep)
        else:
            print("⚠ Running in MOCK mode (no Webots connection)")
            if self.pose_log_path:
                with open(self.pose_log_path, 'r') as f:
                    self.pose_data = json.load(f)
                print(f"  Loaded {len(self.pose_data)} poses from {self.pose_log_path}")

        # Setup output directories
        self.output_dir = Path(output_dir)
        self.scans_dir = self.output_dir / "scans"
        self.labels_dir = self.output_dir / "labels"
        self.scans_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.scan_count = 0
        self.start_time = datetime.now()

        print(f"✓ LIDAR Data Collector initialized")
        print(f"  Output: {self.output_dir}")
        print(f"  Sectors: {self.SECTORS}, Threshold: {self.OBSTACLE_THRESHOLD}m")
        print(f"  Target: 1000+ scans\n")

    def get_lidar_ranges(self) -> np.ndarray:
        """Get LIDAR range measurements"""
        if self.mock:
            # Generate synthetic LIDAR data
            # Random ranges between 0 and 5 meters
            return np.random.uniform(0, 5.0, self.LIDAR_POINTS).astype(np.float32)
        else:
            ranges = self.lidar.getRangeImage()
            return np.array(ranges, dtype=np.float32)

    def compute_sector_labels(self, ranges: np.ndarray) -> np.ndarray:
        """
        Compute 9-sector occupancy labels from LIDAR ranges

        Args:
            ranges: [667] LIDAR measurements

        Returns:
            labels: [9] binary occupancy (1 = obstacle detected)
        """
        num_points = len(ranges)
        points_per_sector = num_points // self.SECTORS
        labels = np.zeros(self.SECTORS, dtype=np.float32)

        for sector_idx in range(self.SECTORS):
            start_idx = sector_idx * points_per_sector
            end_idx = start_idx + points_per_sector
            sector_ranges = ranges[start_idx:end_idx]

            # Sector is occupied if ANY point is below threshold
            # Filter out invalid readings (inf, nan)
            valid_ranges = sector_ranges[np.isfinite(sector_ranges)]
            if len(valid_ranges) > 0:
                min_distance = np.min(valid_ranges)
                labels[sector_idx] = 1.0 if min_distance < self.OBSTACLE_THRESHOLD else 0.0

        return labels

    def get_metadata(self) -> dict:
        """Get robot pose metadata from GPS and compass or pose log"""
        if self.mock:
            if self.pose_data and self.current_pose_idx < len(self.pose_data):
                pose = self.pose_data[self.current_pose_idx]
                self.current_pose_idx = (self.current_pose_idx + 1) % len(self.pose_data)
                return {
                    'timestamp': datetime.now().isoformat(),
                    'position': pose.get('position', [0, 0, 0]),
                    'orientation': pose.get('orientation', 0.0),
                    'scan_id': self.scan_count,
                    'scenario_tag': pose.get('scenario_tag', 'mock_scenario')
                }
            else:
                return {
                    'timestamp': datetime.now().isoformat(),
                    'position': [0.0, 0.0, 0.0],
                    'orientation': 0.0,
                    'scan_id': self.scan_count,
                    'scenario_tag': 'mock_random'
                }
        else:
            position = self.gps.getValues()
            compass_values = self.compass.getValues()
            orientation = np.arctan2(compass_values[0], compass_values[1])

            return {
                'timestamp': datetime.now().isoformat(),
                'position': list(position),
                'orientation': float(orientation),
                'scan_id': self.scan_count,
                'scenario_tag': 'simulation'
            }

    def save_scan(self, ranges: np.ndarray, labels: np.ndarray, metadata: dict):
        """Save LIDAR scan with labels and metadata"""
        scan_id = f"scan_{self.scan_count:05d}"

        # Save scan data
        scan_path = self.scans_dir / f"{scan_id}.npz"
        np.savez_compressed(
            scan_path,
            ranges=ranges,
            labels=labels,
            metadata=metadata
        )

        self.scan_count += 1

        # Progress update every 50 scans
        if self.scan_count % 50 == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.scan_count / elapsed if elapsed > 0 else 0
            occupied = int(np.sum(labels))
            print(f"  [{self.scan_count:4d}] {rate:.1f} scans/s | Occupied sectors: {occupied}/9")

    def collect_static_scan(self):
        """Collect a single scan at current robot position"""
        # Get LIDAR data
        ranges = self.get_lidar_ranges()

        # Compute labels
        labels = self.compute_sector_labels(ranges)

        # Get metadata
        metadata = self.get_metadata()

        # Save
        self.save_scan(ranges, labels, metadata)

    def run_collection(self, num_scans: int = 1000, move_robot: bool = False):
        """
        Main collection loop

        Args:
            num_scans: Target number of scans to collect
            move_robot: If True, randomly move robot between scans for diversity
        """
        print(f"Starting collection: {num_scans} scans\n")

        if not self.mock:
            # Wait for sensors to initialize
            for _ in range(10):
                self.robot.step(self.timestep)

        while self.scan_count < num_scans:
            if not self.mock:
                if self.robot.step(self.timestep) == -1:
                    break
            
            # Collect scan
            self.collect_static_scan()

            if not self.mock:
                # Optional: Add small delay between scans
                for _ in range(5):
                    self.robot.step(self.timestep)

                # Optional: Move robot for data diversity (implement if needed)
                if move_robot:
                    # TODO: Implement random movement strategy
                    pass
            
            # In mock mode, just loop fast
            if self.mock and self.scan_count >= num_scans:
                break

        # Final statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n✓ Collection complete!")
        print(f"  Total scans: {self.scan_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average rate: {self.scan_count/elapsed:.1f} scans/s")
        print(f"  Saved to: {self.scans_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LIDAR Data Collection")
    parser.add_argument('--output-dir', type=str, default='data/lidar', help='Output directory')
    parser.add_argument('--num-scans', type=int, default=1000, help='Number of scans to collect')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no Webots)')
    parser.add_argument('--pose-log', type=str, help='Path to JSON file with robot poses (mock mode)')
    
    args = parser.parse_args()

    collector = LIDARDataCollector(
        output_dir=args.output_dir,
        mock=args.mock,
        pose_log=args.pose_log
    )
    collector.run_collection(num_scans=args.num_scans, move_robot=False)


if __name__ == "__main__":
    main()
