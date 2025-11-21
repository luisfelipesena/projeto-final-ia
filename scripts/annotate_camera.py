#!/usr/bin/env python3
"""
Camera Data Annotation Tool

Interactive tool to review and correct color labels and bounding boxes for cube images.
Displays image with overlaid bboxes and allows manual correction.

Usage:
    python scripts/annotate_camera.py [--data-dir data/camera]

Controls:
    - Left/Right arrow: Navigate images
    - Click bbox: Select for editing
    - g/b/r: Change selected bbox color to green/blue/red
    - Delete: Remove selected bbox
    - n: Add new bbox (click-drag to draw)
    - s: Save current labels
    - q: Quit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


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


def main():
    parser = argparse.ArgumentParser(description="Camera Data Annotation Tool")
    parser.add_argument('--data-dir', type=str, default='data/camera',
                       help='Path to camera data directory')
    args = parser.parse_args()

    annotator = CameraAnnotator(data_dir=args.data_dir)
    annotator.run()


if __name__ == "__main__":
    main()
