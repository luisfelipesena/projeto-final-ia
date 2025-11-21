#!/usr/bin/env python3
"""
Train/Val/Test Data Split Utility

Splits collected LIDAR and camera data into train/val/test sets with stratification.
Creates split metadata files for reproducible experiments.

Usage:
    python scripts/split_data.py --data-type lidar [--train 0.7 --val 0.15 --test 0.15]
    python scripts/split_data.py --data-type camera [--train 0.7 --val 0.15 --test 0.15]

Output:
    - data/{lidar|camera}/splits.json: Contains file lists for each split
    - Stratified by label distribution (LIDAR: occupancy, Camera: color classes)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


class DataSplitter:
    """Split dataset into train/val/test with stratification"""

    def __init__(
        self,
        data_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Args:
            data_dir: Root data directory (e.g., 'data/lidar' or 'data/camera')
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        np.random.seed(random_seed)

    def split_lidar_data(self) -> Dict[str, List[str]]:
        """
        Split LIDAR data with stratification by occupancy patterns

        Stratification: Group scans by number of occupied sectors (0-9)
        to ensure balanced distribution across splits
        """
        scans_dir = self.data_dir / "scans"
        scan_files = sorted(list(scans_dir.glob("*.npz")))

        if len(scan_files) == 0:
            raise FileNotFoundError(f"No scan files found in {scans_dir}")

        print(f"Found {len(scan_files)} LIDAR scans")

        # Group by occupancy count for stratification
        occupancy_groups = {}
        for scan_file in scan_files:
            data = np.load(scan_file)
            labels = data['labels']
            occupancy_count = int(np.sum(labels))

            if occupancy_count not in occupancy_groups:
                occupancy_groups[occupancy_count] = []
            occupancy_groups[occupancy_count].append(scan_file.name)

        print(f"Occupancy distribution:")
        for count, files in sorted(occupancy_groups.items()):
            print(f"  {count} occupied sectors: {len(files)} scans")

        # Split each group proportionally
        train_files, val_files, test_files = [], [], []

        for occupancy_count, files in occupancy_groups.items():
            # Shuffle files in group
            shuffled = np.random.permutation(files)

            # Compute split indices
            n = len(shuffled)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            # Split
            train_files.extend(shuffled[:train_end])
            val_files.extend(shuffled[train_end:val_end])
            test_files.extend(shuffled[val_end:])

        return {
            'train': sorted(train_files),
            'val': sorted(val_files),
            'test': sorted(test_files)
        }

    def split_camera_data(self) -> Dict[str, List[str]]:
        """
        Split camera data with stratification by color class

        Stratification: Ensure each split has balanced representation
        of green, blue, and red cubes
        """
        images_dir = self.data_dir / "images"
        labels_dir = self.data_dir / "labels"

        image_files = sorted(list(images_dir.glob("*.png")))

        if len(image_files) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

        print(f"Found {len(image_files)} camera images")

        # Group by dominant color class
        color_groups = {'green': [], 'blue': [], 'red': [], 'mixed': []}

        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.json"

            if not label_file.exists():
                continue

            with open(label_file, 'r') as f:
                label_data = json.load(f)

            # Count color occurrences
            colors = [cube['color'] for cube in label_data['cubes']]
            color_counts = Counter(colors)

            # Categorize image
            if len(color_counts) == 0:
                continue  # Skip empty images
            elif len(color_counts) == 1:
                # Single color
                dominant_color = list(color_counts.keys())[0]
                color_groups[dominant_color].append(img_file.name)
            else:
                # Multiple colors
                color_groups['mixed'].append(img_file.name)

        print(f"Color distribution:")
        for color, files in color_groups.items():
            print(f"  {color}: {len(files)} images")

        # Split each group proportionally
        train_files, val_files, test_files = [], [], []

        for color, files in color_groups.items():
            if len(files) == 0:
                continue

            # Shuffle files in group
            shuffled = np.random.permutation(files)

            # Compute split indices
            n = len(shuffled)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            # Split
            train_files.extend(shuffled[:train_end])
            val_files.extend(shuffled[train_end:val_end])
            test_files.extend(shuffled[val_end:])

        return {
            'train': sorted(train_files),
            'val': sorted(val_files),
            'test': sorted(test_files)
        }

    def save_splits(self, splits: Dict[str, List[str]], data_type: str):
        """Save split metadata to JSON file"""
        output_file = self.data_dir / "splits.json"

        split_metadata = {
            'data_type': data_type,
            'ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'random_seed': self.random_seed,
            'counts': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            },
            'splits': splits
        }

        with open(output_file, 'w') as f:
            json.dump(split_metadata, f, indent=2)

        print(f"\n✓ Splits saved to {output_file}")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val:   {len(splits['val'])} samples")
        print(f"  Test:  {len(splits['test'])} samples")


def main():
    parser = argparse.ArgumentParser(description="Train/Val/Test Data Split")
    parser.add_argument('--data-type', type=str, required=True, choices=['lidar', 'camera'],
                       help='Type of data to split')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Data directory (defaults to data/{lidar|camera})')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir is None:
        args.data_dir = f"data/{args.data_type}"

    # Create splitter
    splitter = DataSplitter(
        data_dir=args.data_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed
    )

    # Split data based on type
    if args.data_type == 'lidar':
        splits = splitter.split_lidar_data()
    else:  # camera
        splits = splitter.split_camera_data()

    # Save splits
    splitter.save_splits(splits, args.data_type)


if __name__ == "__main__":
    main()
