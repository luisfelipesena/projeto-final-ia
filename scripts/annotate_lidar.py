#!/usr/bin/env python3
"""
LIDAR Data Annotation Tool

Interactive tool to review and correct 9-sector occupancy labels for LIDAR scans.
Displays polar plot of LIDAR scan with sector boundaries and current labels.

Usage:
    python scripts/annotate_lidar.py [--data-dir data/lidar]

Controls:
    - Left/Right arrow: Navigate scans
    - 1-9: Toggle sector label (0→1 or 1→0)
    - s: Save current labels
    - q: Quit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Wedge
import json


class LIDARAnnotator:
    """Interactive annotation tool for LIDAR scans"""

    SECTORS = 9
    SECTOR_ANGLE = 30.0  # degrees
    FOV = 270.0  # LIDAR field of view

    def __init__(self, data_dir: str = "data/lidar"):
        self.data_dir = Path(data_dir)
        self.scans_dir = self.data_dir / "scans"

        # Load all scan files
        self.scan_files = sorted(list(self.scans_dir.glob("*.npz")))
        if len(self.scan_files) == 0:
            raise FileNotFoundError(f"No scan files found in {self.scans_dir}")

        self.current_idx = 0
        self.current_scan = None
        self.modified = False

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print(f"✓ LIDAR Annotator loaded")
        print(f"  Found {len(self.scan_files)} scans")
        print(f"\nControls:")
        print("  ←/→ : Navigate scans")
        print("  1-9 : Toggle sector label")
        print("  s   : Save labels")
        print("  q   : Quit\n")

    def load_scan(self, idx: int):
        """Load scan data from file"""
        scan_file = self.scan_files[idx]
        data = np.load(scan_file, allow_pickle=True)

        self.current_scan = {
            'file': scan_file,
            'ranges': data['ranges'],
            'labels': data['labels'].copy(),  # Copy to allow modification
            'metadata': data['metadata'].item() if 'metadata' in data else {}
        }

        self.modified = False

    def plot_scan(self):
        """Plot current scan with sector labels"""
        self.ax.clear()

        ranges = self.current_scan['ranges']
        labels = self.current_scan['labels']
        num_points = len(ranges)

        # Convert LIDAR points to polar coordinates
        # LIDAR FOV: -135° to +135° (270° total)
        angles = np.linspace(-self.FOV/2, self.FOV/2, num_points)
        angles_rad = np.deg2rad(angles)

        # Plot LIDAR points
        self.ax.scatter(angles_rad, ranges, c='gray', s=1, alpha=0.5, label='LIDAR points')

        # Plot sector boundaries and labels
        points_per_sector = num_points // self.SECTORS
        for sector_idx in range(self.SECTORS):
            start_angle = -self.FOV/2 + sector_idx * self.SECTOR_ANGLE
            end_angle = start_angle + self.SECTOR_ANGLE

            # Get sector color based on label
            occupied = labels[sector_idx] > 0.5
            color = 'red' if occupied else 'green'
            alpha = 0.2

            # Draw sector wedge
            wedge = Wedge(
                (0, 0),
                r=5.0,  # Max range for visualization
                theta1=start_angle,
                theta2=end_angle,
                facecolor=color,
                alpha=alpha,
                edgecolor='black',
                linewidth=1
            )
            self.ax.add_patch(wedge)

            # Add sector number
            mid_angle = (start_angle + end_angle) / 2
            mid_angle_rad = np.deg2rad(mid_angle)
            label_text = f"{sector_idx+1}\n({'O' if occupied else 'F'})"
            self.ax.text(mid_angle_rad, 5.5, label_text,
                        ha='center', va='center', fontsize=10, fontweight='bold')

        # Configure plot
        self.ax.set_ylim(0, 6)
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_title(
            f"Scan {self.current_idx + 1}/{len(self.scan_files)} - "
            f"{self.current_scan['file'].name}\n"
            f"{'[MODIFIED]' if self.modified else ''}",
            fontsize=12,
            pad=20
        )

        self.ax.grid(True)
        self.fig.canvas.draw()

    def toggle_sector_label(self, sector_idx: int):
        """Toggle label for specified sector"""
        if 0 <= sector_idx < self.SECTORS:
            current = self.current_scan['labels'][sector_idx]
            self.current_scan['labels'][sector_idx] = 1.0 if current < 0.5 else 0.0
            self.modified = True
            print(f"  Sector {sector_idx+1}: {current:.0f} → {self.current_scan['labels'][sector_idx]:.0f}")
            self.plot_scan()

    def save_labels(self):
        """Save modified labels to file"""
        if not self.modified:
            print("  No changes to save")
            return

        scan_file = self.current_scan['file']

        # Load original data
        data = np.load(scan_file, allow_pickle=True)

        # Update labels
        np.savez_compressed(
            scan_file,
            ranges=data['ranges'],
            labels=self.current_scan['labels'],
            metadata=data['metadata'].item() if 'metadata' in data else {}
        )

        print(f"  ✓ Saved labels to {scan_file.name}")
        self.modified = False
        self.plot_scan()

    def next_scan(self):
        """Move to next scan"""
        if self.modified:
            print("  Warning: Unsaved changes!")

        self.current_idx = (self.current_idx + 1) % len(self.scan_files)
        self.load_scan(self.current_idx)
        self.plot_scan()

    def prev_scan(self):
        """Move to previous scan"""
        if self.modified:
            print("  Warning: Unsaved changes!")

        self.current_idx = (self.current_idx - 1) % len(self.scan_files)
        self.load_scan(self.current_idx)
        self.plot_scan()

    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.next_scan()
        elif event.key == 'left':
            self.prev_scan()
        elif event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            sector_idx = int(event.key) - 1
            self.toggle_sector_label(sector_idx)
        elif event.key == 's':
            self.save_labels()
        elif event.key == 'q':
            plt.close()

    def run(self):
        """Start annotation interface"""
        self.load_scan(0)
        self.plot_scan()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="LIDAR Data Annotation Tool")
    parser.add_argument('--data-dir', type=str, default='data/lidar',
                       help='Path to LIDAR data directory')
    args = parser.parse_args()

    annotator = LIDARAnnotator(data_dir=args.data_dir)
    annotator.run()


if __name__ == "__main__":
    main()
