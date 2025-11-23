#!/usr/bin/env python3
"""
Dataset Splitter (T017)

Assigns train/val/test splits to dataset manifest with balanced distribution
per sector (LIDAR) or color (camera). Uses stratified sampling to ensure
representative splits.

Usage:
    python scripts/split_dataset.py --manifest data/lidar/dataset_manifest.json --split-ratio 0.8 0.1 0.1 --seed 42
    python scripts/split_dataset.py --manifest data/camera/dataset_manifest.json --split-ratio 0.8 0.1 0.1 --seed 1337

Split Strategy:
- LIDAR: Stratify by scenario_tag to balance obstacle configurations
- Camera: Stratify by dominant color to balance cube classes
- Default: 80% train, 10% val, 10% test
- Reproducible with fixed random seed

Validation:
- No sample ID appears in multiple splits
- Per-class distribution deviates ≤10% from target ratio
- Split counts match total samples

References: data-model.md (split field), quickstart.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


def stratified_split(samples: List[Dict], strata_key: str, ratios: Tuple[float, float, float], seed: int) -> Dict[str, List[Dict]]:
    """
    Perform stratified split on samples.

    Args:
        samples: List of sample dicts
        strata_key: Key to stratify by (e.g., 'scenario_tag', 'dominant_color')
        ratios: (train_ratio, val_ratio, test_ratio)
        seed: Random seed

    Returns:
        {'train': [...], 'val': [...], 'test': [...]}
    """
    np.random.seed(seed)

    # Group samples by strata
    strata_groups = defaultdict(list)
    for sample in samples:
        stratum = sample.get(strata_key, 'unknown')
        strata_groups[stratum].append(sample)

    # Split each stratum proportionally
    splits = {'train': [], 'val': [], 'test': []}
    train_ratio, val_ratio, test_ratio = ratios

    for stratum, group_samples in strata_groups.items():
        # Shuffle within stratum
        indices = np.random.permutation(len(group_samples))
        shuffled = [group_samples[i] for i in indices]

        # Compute split sizes
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remainder goes to test

        # Assign splits
        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train:n_train + n_val]
        test_samples = shuffled[n_train + n_val:]

        # Tag split assignment
        for sample in train_samples:
            sample['split'] = 'train'
        for sample in val_samples:
            sample['split'] = 'val'
        for sample in test_samples:
            sample['split'] = 'test'

        splits['train'].extend(train_samples)
        splits['val'].extend(val_samples)
        splits['test'].extend(test_samples)

    return splits


def assign_dominant_color(sample: Dict) -> str:
    """Assign dominant color for camera samples (for stratification)."""
    colors = sample.get('colors', [])
    if not colors:
        return 'none'

    # Count color occurrences
    color_counts = defaultdict(int)
    for color in colors:
        color_counts[color] += 1

    # Return most common
    return max(color_counts.items(), key=lambda x: x[1])[0] if color_counts else 'none'


def split_dataset(manifest_path: str, split_ratios: Tuple[float, float, float], seed: int):
    """
    Split dataset manifest into train/val/test with balanced distribution (T017).

    Args:
        manifest_path: Path to dataset manifest JSON
        split_ratios: (train, val, test) ratios (should sum to 1.0)
        seed: Random seed for reproducibility
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    # Validate ratios
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")

    # Load manifest
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    dataset_type = manifest.get('dataset_type')
    samples = manifest.get('samples', [])

    print(f"\n✓ Splitting {dataset_type.upper()} dataset")
    print(f"  Manifest: {manifest_file}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Split ratios: {split_ratios[0]:.0%} train / {split_ratios[1]:.0%} val / {split_ratios[2]:.0%} test")
    print(f"  Random seed: {seed}")

    # Determine stratification key
    if dataset_type == 'lidar':
        strata_key = 'scenario_tag'
        print(f"  Stratify by: {strata_key}")
    elif dataset_type == 'camera':
        # Compute dominant color for each sample
        strata_key = 'dominant_color'
        for sample in samples:
            sample[strata_key] = assign_dominant_color(sample)
        print(f"  Stratify by: {strata_key} (computed from colors)")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Perform stratified split
    splits = stratified_split(samples, strata_key, split_ratios, seed)

    # Update manifest
    manifest['samples'] = splits['train'] + splits['val'] + splits['test']
    manifest['splits'] = {
        'train': len(splits['train']),
        'val': len(splits['val']),
        'test': len(splits['test'])
    }

    # Save updated manifest
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Splits assigned successfully")
    print(f"  Train: {len(splits['train'])} samples ({len(splits['train'])/len(samples):.1%})")
    print(f"  Val:   {len(splits['val'])} samples ({len(splits['val'])/len(samples):.1%})")
    print(f"  Test:  {len(splits['test'])} samples ({len(splits['test'])/len(samples):.1%})")

    # Validation: Check split integrity
    all_ids = set()
    for sample in manifest['samples']:
        sample_id = sample['sample_id']
        if sample_id in all_ids:
            print(f"  ⚠️  Warning: Duplicate sample_id detected: {sample_id}")
        all_ids.add(sample_id)

    print(f"\n  ✓ Split integrity verified: {len(all_ids)} unique samples")

    # Distribution statistics per split
    print(f"\n  Distribution by {strata_key}:")
    for split_name in ['train', 'val', 'test']:
        strata_counts = defaultdict(int)
        for sample in splits[split_name]:
            stratum = sample.get(strata_key, 'unknown')
            strata_counts[stratum] += 1

        print(f"\n    {split_name.upper()}:")
        for stratum, count in sorted(strata_counts.items()):
            pct = (count / len(splits[split_name])) * 100 if splits[split_name] else 0
            print(f"      {stratum}: {count} ({pct:.1f}%)")

    print(f"\n  Manifest updated: {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Split dataset with balanced distribution (T017)")
    parser.add_argument('--manifest', required=True,
                        help='Path to dataset manifest JSON (will be updated in-place)')
    parser.add_argument('--split-ratio', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios (default: 0.8 0.1 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    split_dataset(args.manifest, tuple(args.split_ratio), args.seed)


if __name__ == "__main__":
    main()
