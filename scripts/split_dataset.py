#!/usr/bin/env python3
"""
Train/Val/Test Data Split Utility

Splits collected LIDAR and camera data into train/val/test sets with stratification.
Uses the dataset manifest to ensure consistent and reproducible splits.

Usage:
    python scripts/split_dataset.py --manifest data/lidar/manifest.json [--train 0.7 --val 0.15 --test 0.15]
    python scripts/split_dataset.py --manifest data/camera/manifest.json [--train 0.7 --val 0.15 --test 0.15]

Output:
    - data/{lidar|camera}/splits.json: Contains file lists for each split
    - Stratified by label distribution (LIDAR: occupancy, Camera: color classes)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter


class DataSplitter:
    """Split dataset into train/val/test with stratification using manifest"""

    def __init__(
        self,
        manifest_path: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Args:
            manifest_path: Path to dataset manifest JSON
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
            
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        self.data_type = self.manifest.get('dataset_type', 'unknown')
        self.data_dir = Path(self.manifest.get('root_dir', self.manifest_path.parent))
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        np.random.seed(random_seed)

    def split_data(self) -> Dict[str, List[str]]:
        """
        Split data with stratification based on dataset type
        """
        samples = self.manifest.get('samples', [])
        if not samples:
            raise ValueError("No samples found in manifest")
            
        print(f"Found {len(samples)} samples in manifest ({self.data_type})")
        
        # Group samples for stratification
        groups = {}
        
        if self.data_type == 'lidar':
            # Stratify by occupancy count
            for sample in samples:
                key = sample.get('occupied_sectors', 0)
                if key not in groups:
                    groups[key] = []
                groups[key].append(sample['id'])
                
        elif self.data_type == 'camera':
            # Stratify by dominant color or 'mixed'
            for sample in samples:
                classes = sample.get('classes', [])
                if not classes:
                    key = 'empty'
                elif len(set(classes)) == 1:
                    key = classes[0]
                else:
                    key = 'mixed'
                    
                if key not in groups:
                    groups[key] = []
                groups[key].append(sample['id'])
        else:
            print(f"Warning: Unknown dataset type '{self.data_type}', using random split")
            groups['all'] = [s['id'] for s in samples]

        # Print distribution
        print(f"Stratification groups:")
        for key, items in sorted(groups.items()):
            print(f"  {key}: {len(items)} samples")

        # Split each group proportionally
        train_ids, val_ids, test_ids = [], [], []

        for key, ids in groups.items():
            # Shuffle ids in group
            shuffled = np.random.permutation(ids)

            # Compute split indices
            n = len(shuffled)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            # Split
            train_ids.extend(shuffled[:train_end])
            val_ids.extend(shuffled[train_end:val_end])
            test_ids.extend(shuffled[val_end:])

        return {
            'train': sorted(train_ids),
            'val': sorted(val_ids),
            'test': sorted(test_ids)
        }

    def save_splits(self, splits: Dict[str, List[str]]):
        """Save split metadata to JSON file"""
        output_file = self.data_dir / "splits.json"

        split_metadata = {
            'dataset_type': self.data_type,
            'manifest_path': str(self.manifest_path),
            'generated_at': str(np.datetime64('now')),
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
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to dataset manifest JSON')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Create splitter
    splitter = DataSplitter(
        manifest_path=args.manifest,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed
    )

    # Split data
    splits = splitter.split_data()

    # Save splits
    splitter.save_splits(splits)


if __name__ == "__main__":
    main()
