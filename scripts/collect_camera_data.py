#!/usr/bin/env python3
"""
Camera Data Collection Script

Collects RGB images from Webots camera for training cube color classification CNN.
Saves images with color labels (green/blue/red) for supervised learning.

Usage:
    From Webots: Run this as a controller
    CLI: python scripts/collect_camera_data.py --lighting default --sessions 15 --frames-per-session 40
    Target: 500+ images with balanced color distribution

Data format (per data-model.md CameraSample):
    - sample_id: UUID
    - timestamp: ISO8601 string
    - robot_pose: {x, y, theta}
    - image_path: relative path to PNG file
    - bounding_boxes: array of {id, x, y, w, h}
    - colors: array of color enums (red/green/blue)
    - distance_estimates: array of floats (meters)
    - lighting_tag: enum (default/bright/dim) - T014
    - split: enum (train/val/test)
"""

import sys
import os
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import json
from uuid import uuid4

# Add Webots controller path
sys.path.append(os.path.join(os.environ.get('WEBOTS_HOME', ''), 'lib', 'controller', 'python'))

from controller import Robot


class CameraDataCollector:
    """Collects and labels camera images for cube detection CNN"""

    COLOR_CLASSES = {
        'green': 0,
        'blue': 1,
        'red': 2
    }

    def __init__(self, output_dir: str = "data/camera", lighting_tag: str = "default", session_id: str = None):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        # Lighting tag for this session (T014)
        self.lighting_tag = lighting_tag
        self.session_id = session_id or str(uuid4())[:8]

        # Initialize camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # Camera specs (per spec.md FR-018)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        self.fov = self.camera.getFov()

        # Initialize GPS for ground truth (T014)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(self.timestep)

        # Initialize compass for orientation (T014)
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(self.timestep)

        # Setup output directories
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "raw" / f"session_{self.session_id}" / "images"
        self.labels_dir = self.output_dir / "raw" / f"session_{self.session_id}" / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_count = 0
        self.class_counts = {color: 0 for color in self.COLOR_CLASSES.keys()}
        self.start_time = datetime.now()

        print(f"✓ Camera Data Collector initialized")
        print(f"  Session: {self.session_id}")
        print(f"  Lighting: {self.lighting_tag}")
        print(f"  Output: {self.output_dir}")
        print(f"  Resolution: {self.width}×{self.height}")
        print(f"  FOV: {self.fov:.2f} rad")
        print(f"  Target: 500+ images (balanced across colors)\n")

    def get_camera_image(self) -> np.ndarray:
        """Get RGB image from camera"""
        image = self.camera.getImageArray()

        # Convert Webots image format (width, height, BGRA) to (height, width, RGB)
        height, width = self.height, self.width
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(width):
            for j in range(height):
                pixel = image[i][j]
                rgb_image[j, i, 0] = self.camera.imageGetRed(pixel, i, j)
                rgb_image[j, i, 1] = self.camera.imageGetGreen(pixel, i, j)
                rgb_image[j, i, 2] = self.camera.imageGetBlue(pixel, i, j)

        return rgb_image

    def get_recognized_objects(self) -> list:
        """Get recognized cubes from camera recognition system"""
        objects = self.camera.getRecognitionObjects()
        cubes = []

        for obj in objects:
            # Filter for cubes based on recognition colors
            colors = obj.getColors()

            # Map recognition colors to our classes
            # Green: [0, 1, 0], Blue: [0, 0, 1], Red: [1, 0, 0]
            color_class = None
            if colors[1] > 0.8 and colors[0] < 0.2 and colors[2] < 0.2:
                color_class = 'green'
            elif colors[2] > 0.8 and colors[0] < 0.2 and colors[1] < 0.2:
                color_class = 'blue'
            elif colors[0] > 0.8 and colors[1] < 0.2 and colors[2] < 0.2:
                color_class = 'red'

            if color_class:
                # Get object position on image
                pos_on_image = obj.getPositionOnImage()
                size_on_image = obj.getSizeOnImage()

                cubes.append({
                    'color': color_class,
                    'bbox': {
                        'x': int(pos_on_image[0] - size_on_image[0] / 2),
                        'y': int(pos_on_image[1] - size_on_image[1] / 2),
                        'width': int(size_on_image[0]),
                        'height': int(size_on_image[1])
                    },
                    'position_3d': obj.getPosition()
                })

        return cubes

    def get_metadata(self) -> dict:
        """Get camera and robot metadata (T014)"""
        position = self.gps.getValues()
        compass_values = self.compass.getValues()
        theta = np.arctan2(compass_values[0], compass_values[1])

        return {
            'sample_id': str(uuid4()),  # Unique ID per frame
            'timestamp': datetime.now().isoformat() + "Z",
            'robot_pose': {  # T014: x, y, theta structure
                'x': float(position[0]),
                'y': float(position[1]),
                'theta': float(theta)
            },
            'lighting_tag': self.lighting_tag,  # T014: lighting tagging
            'session_id': self.session_id,
            'frame_index': self.image_count,
            'camera': {
                'width': self.width,
                'height': self.height,
                'fov': self.fov
            }
        }

    def save_image(self, image: np.ndarray, cubes: list, metadata: dict):
        """Save image with labels and metadata"""
        image_id = f"img_{self.image_count:05d}"

        # Save image as PNG
        from PIL import Image
        img_pil = Image.fromarray(image)
        img_path = self.images_dir / f"{image_id}.png"
        img_pil.save(img_path)

        # Prepare label data
        label_data = {
            'image_id': image_id,
            'cubes': cubes,
            'metadata': metadata
        }

        # Save label as JSON
        label_path = self.labels_dir / f"{image_id}.json"
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)

        # Update statistics
        for cube in cubes:
            self.class_counts[cube['color']] += 1

        self.image_count += 1

        # Progress update
        if self.image_count % 20 == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.image_count / elapsed if elapsed > 0 else 0
            print(f"  [{self.image_count:4d}] {rate:.1f} imgs/s | "
                  f"Green: {self.class_counts['green']:3d}, "
                  f"Blue: {self.class_counts['blue']:3d}, "
                  f"Red: {self.class_counts['red']:3d}")

    def collect_image(self):
        """Collect a single image with current visible cubes"""
        # Get camera image
        image = self.get_camera_image()

        # Get recognized cubes
        cubes = self.get_recognized_objects()

        # Skip if no cubes visible
        if len(cubes) == 0:
            return False

        # Get metadata
        metadata = self.get_metadata()

        # Save
        self.save_image(image, cubes, metadata)
        return True

    def run_collection(self, num_images: int = 500, move_robot: bool = False):
        """
        Main collection loop

        Args:
            num_images: Target number of images to collect
            move_robot: If True, move robot to capture different viewpoints
        """
        print(f"Starting collection: {num_images} images\n")

        # Wait for sensors to initialize
        for _ in range(10):
            self.robot.step(self.timestep)

        attempts = 0
        max_attempts = num_images * 3  # Allow retries if no cubes visible

        while self.robot.step(self.timestep) != -1 and self.image_count < num_images and attempts < max_attempts:
            attempts += 1

            # Try to collect image
            success = self.collect_image()

            if success:
                # Small delay between captures
                for _ in range(3):
                    self.robot.step(self.timestep)
            else:
                # No cubes visible, wait longer or move
                for _ in range(10):
                    self.robot.step(self.timestep)

            # Optional: Move robot for viewpoint diversity
            if move_robot and attempts % 50 == 0:
                # TODO: Implement movement strategy
                pass

        # Final statistics
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\n✓ Collection complete!")
        print(f"  Total images: {self.image_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average rate: {self.image_count/elapsed:.1f} imgs/s")
        print(f"  Class distribution:")
        for color, count in self.class_counts.items():
            pct = (count / max(1, sum(self.class_counts.values()))) * 100
            print(f"    {color}: {count} ({pct:.1f}%)")
        print(f"  Saved to: {self.images_dir}")


def main():
    """Main entry point for Webots controller"""
    parser = argparse.ArgumentParser(description="Collect camera images with lighting tags")
    parser.add_argument("--output", default="data/camera", help="Output directory")
    parser.add_argument("--lighting", default="default",
                        choices=["default", "bright", "dim"],
                        help="Lighting condition tag for this session (T014)")
    parser.add_argument("--frames-per-session", type=int, default=40, help="Number of frames per session")
    parser.add_argument("--session-id", default=None, help="Session ID (auto-generated if not provided)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")

    args = parser.parse_args()
    np.random.seed(args.seed)

    collector = CameraDataCollector(
        output_dir=args.output,
        lighting_tag=args.lighting,
        session_id=args.session_id
    )
    collector.run_collection(num_images=args.frames_per_session, move_robot=False)


if __name__ == "__main__":
    main()
