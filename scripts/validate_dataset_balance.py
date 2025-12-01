#!/usr/bin/env python3
"""
Dataset Balance Validation Script

Validates dataset balance per class/sector distribution according to data-model.md
thresholds:
- LIDAR: sector_labels distribution per sector deviates ≤10% from uniform
- Camera: colors distribution per cube color deviates ≤5% from uniform

Usage:
    python scripts/validate_dataset_balance.py --input data/lidar/annotated/ --type lidar
    python scripts/validate_dataset_balance.py --input data/camera/annotated/ --type camera
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np


def validate_lidar_balance(samples: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate LIDAR dataset balance (sector distribution)"""
    errors = []
    
    # Count sector occupancies
    sector_counts = [0] * 9
    total_samples = len(samples)
    
    if total_samples == 0:
        errors.append("No samples found")
        return False, errors
    
    for sample in samples:
        if 'sector_labels' not in sample:
            continue
        sector_labels = sample['sector_labels']
        if len(sector_labels) != 9:
            continue
        for i, occupied in enumerate(sector_labels):
            if occupied:
                sector_counts[i] += 1
    
    # Calculate expected uniform distribution
    total_occupancies = sum(sector_counts)
    if total_occupancies == 0:
        errors.append("No sector occupancies found (all sectors empty)")
        return False, errors
    
    expected_per_sector = total_occupancies / 9.0
    max_deviation_threshold = 0.10  # 10% deviation allowed
    
    # Check each sector
    for i, count in enumerate(sector_counts):
        deviation = abs(count - expected_per_sector) / expected_per_sector if expected_per_sector > 0 else 1.0
        if deviation > max_deviation_threshold:
            errors.append(
                f"Sector {i}: {count} occupancies (expected ~{expected_per_sector:.1f}), "
                f"deviation {deviation*100:.1f}% exceeds 10% threshold"
            )
    
    return len(errors) == 0, errors


def validate_camera_balance(samples: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate camera dataset balance (color distribution)"""
    errors = []
    
    # Count colors
    color_counts = Counter()
    total_cubes = 0
    
    for sample in samples:
        if 'colors' not in sample:
            continue
        colors = sample['colors']
        if not isinstance(colors, list):
            continue
        for color in colors:
            if color in ['red', 'green', 'blue']:
                color_counts[color] += 1
                total_cubes += 1
    
    if total_cubes == 0:
        errors.append("No cube colors found")
        return False, errors
    
    # Calculate expected uniform distribution
    expected_per_color = total_cubes / 3.0
    max_deviation_threshold = 0.05  # 5% deviation allowed
    
    # Check each color
    for color in ['red', 'green', 'blue']:
        count = color_counts[color]
        deviation = abs(count - expected_per_color) / expected_per_color if expected_per_color > 0 else 1.0
        if deviation > max_deviation_threshold:
            errors.append(
                f"Color {color}: {count} cubes (expected ~{expected_per_color:.1f}), "
                f"deviation {deviation*100:.1f}% exceeds 5% threshold"
            )
    
    return len(errors) == 0, errors


def validate_split_integrity(samples: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate that no sample ID appears in more than one split"""
    errors = []
    
    split_samples = {'train': set(), 'val': set(), 'test': set()}
    
    for sample in samples:
        sample_id = sample.get('sample_id')
        split = sample.get('split')
        
        if not sample_id:
            continue
        
        if split not in ['train', 'val', 'test']:
            continue
        
        # Check if sample_id already in another split
        for other_split, other_ids in split_samples.items():
            if other_split != split and sample_id in other_ids:
                errors.append(f"Sample {sample_id} appears in both {split} and {other_split}")
        
        split_samples[split].add(sample_id)
    
    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description='Validate dataset balance')
    parser.add_argument('--input', type=str, required=True, help='Input directory or manifest file')
    parser.add_argument('--type', type=str, required=True, choices=['lidar', 'camera'], help='Dataset type')
    parser.add_argument('--manifest', action='store_true', help='Input is a manifest JSON file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    samples = []
    if args.manifest:
        # Load from manifest
        with open(input_path, 'r') as f:
            manifest = json.load(f)
            samples = manifest.get('samples', [])
    else:
        # Load from directory (assume JSON files)
        for json_file in input_path.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
                else:
                    samples.append(data)
    
    if not samples:
        print(f"WARNING: No samples found in {args.input}", file=sys.stderr)
        sys.exit(0)
    
    all_errors = []
    
    # Validate balance
    if args.type == 'lidar':
        balanced, errors = validate_lidar_balance(samples)
        if not balanced:
            all_errors.extend(errors)
    else:
        balanced, errors = validate_camera_balance(samples)
        if not balanced:
            all_errors.extend(errors)
    
    # Validate split integrity
    split_ok, split_errors = validate_split_integrity(samples)
    if not split_ok:
        all_errors.extend(split_errors)
    
    # Print summary
    print(f"Dataset balance validation for {args.type} dataset:")
    print(f"  Total samples: {len(samples)}")
    
    if args.type == 'lidar':
        sector_counts = [0] * 9
        for sample in samples:
            if 'sector_labels' in sample and len(sample['sector_labels']) == 9:
                for i, occupied in enumerate(sample['sector_labels']):
                    if occupied:
                        sector_counts[i] += 1
        print(f"  Sector occupancies: {sector_counts}")
    else:
        color_counts = Counter()
        for sample in samples:
            if 'colors' in sample:
                for color in sample['colors']:
                    if color in ['red', 'green', 'blue']:
                        color_counts[color] += 1
        print(f"  Color distribution: {dict(color_counts)}")
    
    # Print split distribution
    split_counts = Counter(s.get('split', 'unknown') for s in samples)
    print(f"  Split distribution: {dict(split_counts)}")
    
    if all_errors:
        print("\n✗ Balance validation failed:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n✓ Balance validation passed")
        sys.exit(0)


if __name__ == '__main__':
    main()


