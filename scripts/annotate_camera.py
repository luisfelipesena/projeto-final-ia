#!/usr/bin/env python3
"""
Camera Data Annotation Tool

Interactive tool to review and correct color labels and bounding boxes for cube images.
Auto-labeling mode (T015) uses HSV color segmentation + distance estimation.

Usage:
    Interactive: python scripts/annotate_camera.py --data-dir data/camera
    Auto-label:  python scripts/annotate_camera.py --auto-label --input data/camera/raw --output data/camera/annotated

Controls (interactive):
    - Left/Right arrow: Navigate images
    - Click bbox: Select for editing
    - g/b/r: Change selected bbox color to green/blue/red
    - Delete: Remove selected bbox
    - n: Add new bbox (click-drag to draw)
    - a: Auto-label current image (T015)
    - s: Save current labels
    - q: Quit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from tqdm import tqdm
from uuid import uuid4


# HSV color ranges for cube detection (T015)
HSV_COLOR_RANGES = {
    'red': [(0, 100, 100), (10, 255, 255)],    # Lower red hue
    'red2': [(170, 100, 100), (180, 255, 255)],  # Upper red hue (wrap-around)
    'green': [(40, 50, 50), (80, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)]
}

# Camera calibration (T015 - for distance estimation)
FOCAL_LENGTH = 462.0  # pixels (estimated from Webots camera FOV)
CUBE_SIZE = 0.05  # meters (5cm cubes)


class CameraAnnotator:
    """Interactive annotation tool for camera images"""

    COLOR_CLASSES = ['green', 'blue', 'red']
    COLOR_MAP = {
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'red': (1, 0, 0)
    }

    def __init__(self, data_dir: str = "data/camera"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"

        # Load all image files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        if len(self.image_files) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.current_idx = 0
        self.current_data = None
        self.modified = False
        self.selected_cube_idx = None
        self.drawing_bbox = False
        self.bbox_start = None

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        print(f"✓ Camera Annotator loaded")
        print(f"  Found {len(self.image_files)} images")
        print(f"\nControls:")
        print("  ←/→     : Navigate images")
        print("  Click   : Select bbox")
        print("  g/b/r   : Change color (green/blue/red)")
        print("  Delete  : Remove selected bbox")
        print("  n       : Add new bbox mode (click-drag)")
        print("  s       : Save labels")
        print("  q       : Quit\n")

    def load_image(self, idx: int):
        """Load image and corresponding labels"""
        img_file = self.image_files[idx]
        label_file = self.labels_dir / f"{img_file.stem}.json"

        # Load image
        image = np.array(Image.open(img_file))

        # Load or create labels
        if label_file.exists():
            with open(label_file, 'r') as f:
                label_data = json.load(f)
        else:
            # Create empty label
            label_data = {
                'image_id': img_file.stem,
                'cubes': [],
                'metadata': {}
            }

        self.current_data = {
            'file': img_file,
            'image': image,
            'labels': label_data
        }

        self.modified = False
        self.selected_cube_idx = None

    def plot_image(self):
        """Plot current image with bounding boxes"""
        self.ax.clear()

        image = self.current_data['image']
        cubes = self.current_data['labels']['cubes']

        # Display image
        self.ax.imshow(image)

        # Draw bounding boxes
        for idx, cube in enumerate(cubes):
            bbox = cube['bbox']
            color = cube['color']

            # Determine style (selected vs normal)
            if idx == self.selected_cube_idx:
                edgecolor = 'yellow'
                linewidth = 3
            else:
                edgecolor = self.COLOR_MAP[color]
                linewidth = 2

            # Draw bbox
            rect = Rectangle(
                (bbox['x'], bbox['y']),
                bbox['width'],
                bbox['height'],
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor='none'
            )
            self.ax.add_patch(rect)

            # Add label text
            label_text = f"{color} ({idx+1})"
            self.ax.text(
                bbox['x'], bbox['y'] - 5,
                label_text,
                color=edgecolor,
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
            )

        # Configure plot
        self.ax.set_xlim(0, image.shape[1])
        self.ax.set_ylim(image.shape[0], 0)
        self.ax.axis('off')

        title = (f"Image {self.current_idx + 1}/{len(self.image_files)} - "
                f"{self.current_data['file'].name}\n"
                f"Cubes: {len(cubes)} {'[MODIFIED]' if self.modified else ''}")
        if self.selected_cube_idx is not None:
            title += f" | Selected: {self.selected_cube_idx + 1}"

        self.ax.set_title(title, fontsize=12, pad=10)
        self.fig.canvas.draw()

    def find_bbox_at_point(self, x: int, y: int) -> int:
        """Find bbox containing point (x, y)"""
        cubes = self.current_data['labels']['cubes']

        for idx, cube in enumerate(cubes):
            bbox = cube['bbox']
            if (bbox['x'] <= x <= bbox['x'] + bbox['width'] and
                bbox['y'] <= y <= bbox['y'] + bbox['height']):
                return idx

        return None

    def change_color(self, new_color: str):
        """Change color of selected bbox"""
        if self.selected_cube_idx is None:
            print("  No bbox selected")
            return

        cubes = self.current_data['labels']['cubes']
        old_color = cubes[self.selected_cube_idx]['color']
        cubes[self.selected_cube_idx]['color'] = new_color

        print(f"  Cube {self.selected_cube_idx+1}: {old_color} → {new_color}")
        self.modified = True
        self.plot_image()

    def delete_bbox(self):
        """Delete selected bbox"""
        if self.selected_cube_idx is None:
            print("  No bbox selected")
            return

        cubes = self.current_data['labels']['cubes']
        deleted = cubes.pop(self.selected_cube_idx)
        print(f"  Deleted cube {self.selected_cube_idx+1} ({deleted['color']})")

        self.selected_cube_idx = None
        self.modified = True
        self.plot_image()

    def add_bbox(self, x1: int, y1: int, x2: int, y2: int, color: str = 'green'):
        """Add new bounding box"""
        bbox = {
            'x': min(x1, x2),
            'y': min(y1, y2),
            'width': abs(x2 - x1),
            'height': abs(y2 - y1)
        }

        cube = {
            'color': color,
            'bbox': bbox,
            'position_3d': None  # Not available in manual annotation
        }

        self.current_data['labels']['cubes'].append(cube)
        self.modified = True
        print(f"  Added new {color} cube")
        self.plot_image()

    def save_labels(self):
        """Save modified labels to file"""
        if not self.modified:
            print("  No changes to save")
            return

        label_file = self.labels_dir / f"{self.current_data['file'].stem}.json"

        with open(label_file, 'w') as f:
            json.dump(self.current_data['labels'], f, indent=2)

        print(f"  ✓ Saved labels to {label_file.name}")
        self.modified = False
        self.plot_image()

    def next_image(self):
        """Move to next image"""
        if self.modified:
            print("  Warning: Unsaved changes!")

        self.current_idx = (self.current_idx + 1) % len(self.image_files)
        self.load_image(self.current_idx)
        self.plot_image()

    def prev_image(self):
        """Move to previous image"""
        if self.modified:
            print("  Warning: Unsaved changes!")

        self.current_idx = (self.current_idx - 1) % len(self.image_files)
        self.load_image(self.current_idx)
        self.plot_image()

    def on_click(self, event):
        """Handle mouse click"""
        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.drawing_bbox:
            self.bbox_start = (x, y)
        else:
            # Select bbox at click location
            idx = self.find_bbox_at_point(x, y)
            self.selected_cube_idx = idx
            if idx is not None:
                print(f"  Selected cube {idx+1} ({self.current_data['labels']['cubes'][idx]['color']})")
            self.plot_image()

    def on_release(self, event):
        """Handle mouse release"""
        if not self.drawing_bbox or self.bbox_start is None:
            return

        if event.inaxes != self.ax:
            return

        x, y = int(event.xdata), int(event.ydata)
        x1, y1 = self.bbox_start

        # Add bbox if drag distance is significant
        if abs(x - x1) > 10 and abs(y - y1) > 10:
            self.add_bbox(x1, y1, x, y)

        self.drawing_bbox = False
        self.bbox_start = None

    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.next_image()
        elif event.key == 'left':
            self.prev_image()
        elif event.key == 'g':
            self.change_color('green')
        elif event.key == 'b':
            self.change_color('blue')
        elif event.key == 'r':
            self.change_color('red')
        elif event.key == 'delete':
            self.delete_bbox()
        elif event.key == 'n':
            self.drawing_bbox = True
            print("  Draw new bbox mode: click and drag")
        elif event.key == 's':
            self.save_labels()
        elif event.key == 'q':
            plt.close()

    def run(self):
        """Start annotation interface"""
        self.load_image(0)
        self.plot_image()
        plt.show()


def detect_cubes_hsv(image: np.ndarray) -> list:
    """
    Auto-detect cubes using HSV color segmentation (T015).

    Returns list of {color, bbox, distance_estimate}
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    detections = []

    for color_name, (lower, upper) in HSV_COLOR_RANGES.items():
        if color_name == 'red2':
            continue  # Handled with main red range

        # Create mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Handle red wrap-around
        if color_name == 'red':
            lower2, upper2 = HSV_COLOR_RANGES['red2']
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Estimate distance (T015)
            # distance = (CUBE_SIZE * FOCAL_LENGTH) / bbox_height_pixels
            distance = (CUBE_SIZE * FOCAL_LENGTH) / max(w, h) if max(w, h) > 0 else 0.0

            detections.append({
                'id': str(uuid4())[:8],
                'color': color_name,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'distance': float(distance)
            })

    return detections


def auto_label_batch(input_dir: str, output_dir: str):
    """
    Batch auto-labeling of camera images using HSV (T015).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(list(input_path.rglob("*.png")))
    if len(image_files) == 0:
        print(f"No images found in {input_path}")
        return

    print(f"✓ Auto-labeling {len(image_files)} images...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Method: HSV color segmentation + distance estimation\n")

    for img_file in tqdm(image_files, desc="Auto-labeling"):
        # Load image
        image = np.array(Image.open(img_file))

        # Load metadata if exists
        label_file = img_file.parent.parent / "labels" / f"{img_file.stem}.json"
        metadata = {}
        if label_file.exists():
            with open(label_file, 'r') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})

        # Detect cubes (T015)
        detections = detect_cubes_hsv(image)

        # Prepare annotated data
        annotated = {
            'sample_id': metadata.get('sample_id', str(uuid4())),
            'timestamp': metadata.get('timestamp', ''),
            'image_path': str(img_file.relative_to(input_path)),
            'bounding_boxes': [d['bbox'] for d in detections],
            'colors': [d['color'] for d in detections],
            'distance_estimates': [d['distance'] for d in detections],
            'robot_pose': metadata.get('robot_pose', {}),
            'lighting_tag': metadata.get('lighting_tag', 'default')
        }

        # Save
        relative_path = img_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(annotated, f, indent=2)

    print(f"\n✓ Auto-labeling complete: {len(image_files)} images processed")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Camera Data Annotation Tool (T015)")
    parser.add_argument('--data-dir', type=str, default='data/camera',
                       help='Path to camera data directory (interactive mode)')
    parser.add_argument('--auto-label', action='store_true',
                       help='Batch auto-labeling mode (non-interactive)')
    parser.add_argument('--input', type=str, default='data/camera/raw',
                       help='Input directory for auto-labeling')
    parser.add_argument('--output', type=str, default='data/camera/annotated',
                       help='Output directory for annotated data')

    args = parser.parse_args()

    if args.auto_label:
        # Batch auto-labeling mode (T015)
        auto_label_batch(args.input, args.output)
    else:
        # Interactive annotation mode
        annotator = CameraAnnotator(data_dir=args.data_dir)
        annotator.run()


if __name__ == "__main__":
    main()
