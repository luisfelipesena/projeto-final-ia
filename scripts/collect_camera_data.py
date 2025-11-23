#!/usr/bin/env python3
"""
Camera Data Collection Script

Collects RGB images from Webots camera for training cube color classification CNN.
Saves images with color labels (green/blue/red) for supervised learning.

Usage:
    From Webots: Run this as a controller
    From Terminal (Mock): python scripts/collect_camera_data.py --mock --metadata lighting.json
    Target: 500+ images with balanced color distribution

Data format:
    - images: 512×512 RGB PNG files
    - labels: JSON files with color class and bbox coordinates
    - metadata: Camera parameters, timestamp, robot pose, lighting_tag
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


class CameraDataCollector:
    """Collects and labels camera images for cube detection CNN"""

    COLOR_CLASSES = {
        'green': 0,
        'blue': 1,
        'red': 2
    }

    def __init__(self, output_dir: str = "data/camera", mock: bool = False, metadata_file: str = None):
        self.mock = mock
        self.metadata_file = metadata_file
        self.external_metadata = {}
        
        if self.metadata_file:
            with open(self.metadata_file, 'r') as f:
                self.external_metadata = json.load(f)
            print(f"  Loaded metadata from {self.metadata_file}")

        if not self.mock:
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())

            # Initialize camera
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(self.timestep)
            self.camera.recognitionEnable(self.timestep)

            # Camera specs (per spec.md FR-018)
            self.width = self.camera.getWidth()
            self.height = self.camera.getHeight()
            self.fov = self.camera.getFov()

            # Initialize GPS for ground truth
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.timestep)

            # Initialize compass for orientation
            self.compass = self.robot.getDevice("compass")
            self.compass.enable(self.timestep)
        else:
            print("⚠ Running in MOCK mode (no Webots connection)")
            self.width = 512
            self.height = 512
            self.fov = 1.57

        # Setup output directories
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_count = 0
        self.class_counts = {color: 0 for color in self.COLOR_CLASSES.keys()}
        self.start_time = datetime.now()

        print(f"✓ Camera Data Collector initialized")
        print(f"  Output: {self.output_dir}")
        print(f"  Resolution: {self.width}×{self.height}")
        print(f"  FOV: {self.fov:.2f} rad")
        print(f"  Target: 500+ images (balanced across colors)\n")

    def get_camera_image(self) -> np.ndarray:
        """Get RGB image from camera"""
        if self.mock:
            # Generate synthetic noise image
            return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        else:
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
        if self.mock:
            # Generate random mock cubes
            cubes = []
            if np.random.random() > 0.3:  # 70% chance of having cubes
                color = np.random.choice(list(self.COLOR_CLASSES.keys()))
                cubes.append({
                    'color': color,
                    'bbox': {
                        'x': np.random.randint(0, 400),
                        'y': np.random.randint(0, 400),
                        'width': 50,
                        'height': 50
                    },
                    'position_3d': [1.0, 0.0, 0.0]
                })
            return cubes
        
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
        """Get camera and robot metadata"""
        if self.mock:
            position = [0.0, 0.0, 0.0]
            orientation = 0.0
        else:
            position = self.gps.getValues()
            compass_values = self.compass.getValues()
            orientation = np.arctan2(compass_values[0], compass_values[1])

        metadata = {
            'timestamp': datetime.now().isoformat(),
            'image_id': self.image_count,
            'camera': {
                'width': self.width,
                'height': self.height,
                'fov': self.fov
            },
            'robot_pose': {
                'position': list(position),
                'orientation': float(orientation)
            }
        }
        
        # Merge external metadata (e.g. lighting tags)
        if self.external_metadata:
            metadata.update(self.external_metadata)
            
        return metadata

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

        if not self.mock:
            # Wait for sensors to initialize
            for _ in range(10):
                self.robot.step(self.timestep)

        attempts = 0
        max_attempts = num_images * 3  # Allow retries if no cubes visible

        while self.image_count < num_images and attempts < max_attempts:
            if not self.mock:
                if self.robot.step(self.timestep) == -1:
                    break
            
            attempts += 1

            # Try to collect image
            success = self.collect_image()

            if success:
                if not self.mock:
                    # Small delay between captures
                    for _ in range(3):
                        self.robot.step(self.timestep)
            else:
                if not self.mock:
                    # No cubes visible, wait longer or move
                    for _ in range(10):
                        self.robot.step(self.timestep)

            if not self.mock:
                # Optional: Move robot for viewpoint diversity
                if move_robot and attempts % 50 == 0:
                    # TODO: Implement movement strategy
                    pass
            
            # In mock mode, just loop fast
            if self.mock and self.image_count >= num_images:
                break

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
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Camera Data Collection")
    parser.add_argument('--output-dir', type=str, default='data/camera', help='Output directory')
    parser.add_argument('--num-images', type=int, default=500, help='Number of images to collect')
    parser.add_argument('--mock', action='store_true', help='Run in mock mode (no Webots)')
    parser.add_argument('--metadata', type=str, help='Path to JSON file with external metadata (e.g. lighting)')
    
    args = parser.parse_args()

    collector = CameraDataCollector(
        output_dir=args.output_dir,
        mock=args.mock,
        metadata_file=args.metadata
    )
    collector.run_collection(num_images=args.num_images, move_robot=False)


if __name__ == "__main__":
    main()
