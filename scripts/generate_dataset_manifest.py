#!/usr/bin/env python3
"""
Dataset Manifest Generator (T016)

Generates JSON manifest with sample IDs, splits, and metadata hashes for
LIDAR and camera datasets. Manifests enable reproducibility and traceability.

Usage:
    python scripts/generate_dataset_manifest.py --dataset lidar --input data/lidar/annotated --output data/lidar/dataset_manifest.json
    python scripts/generate_dataset_manifest.py --dataset camera --input data/camera/annotated --output data/camera/dataset_manifest.json

Output Format:
{
  "dataset_hash": "sha256_of_manifest",
  "dataset_type": "lidar" | "camera",
  "created_at": "ISO8601_timestamp",
  "total_samples": 1200,
  "splits": {"train": 960, "val": 120, "test": 120},
  "samples": [
    {
      "sample_id": "uuid",
      "file_path": "relative/path/to/file",
      "split": "train",
      "metadata_hash": "sha256_of_sample"
    },
    ...
  ]
}

References: data-model.md (TrainingRun.dataset_hash), plan.md
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import numpy as np


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_lidar_samples(input_dir: Path) -> List[Dict]:
    """Load LIDAR samples from annotated directory."""
    samples = []
    scan_files = sorted(list(input_dir.rglob("*.npz")))

    print(f"  Found {len(scan_files)} LIDAR scans")

    for scan_file in scan_files:
        # Load metadata
        data = np.load(scan_file, allow_pickle=True)
        metadata = data['metadata'].item() if 'metadata' in data else {}

        sample = {
            'sample_id': metadata.get('sample_id', scan_file.stem),
            'file_path': str(scan_file.relative_to(input_dir)),
            'split': None,  # Assigned by split_dataset.py
            'metadata_hash': compute_file_hash(scan_file),
            'scenario_tag': metadata.get('scenario_tag', 'unknown'),
            'session_id': metadata.get('session_id', 'unknown')
        }
        samples.append(sample)

    return samples


def load_camera_samples(input_dir: Path) -> List[Dict]:
    """Load camera samples from annotated directory."""
    samples = []
    label_files = sorted(list(input_dir.rglob("*.json")))

    print(f"  Found {len(label_files)} camera frames")

    for label_file in label_files:
        # Load label data
        with open(label_file, 'r') as f:
            data = json.load(f)

        sample = {
            'sample_id': data.get('sample_id', label_file.stem),
            'file_path': str(label_file.relative_to(input_dir)),
            'split': None,  # Assigned by split_dataset.py
            'metadata_hash': compute_file_hash(label_file),
            'colors': data.get('colors', []),
            'lighting_tag': data.get('lighting_tag', 'default'),
            'session_id': data.get('session_id', 'unknown') if isinstance(data, dict) else 'unknown'
        }
        samples.append(sample)

    return samples


def generate_manifest(dataset_type: str, input_dir: str, output_path: str):
    """
    Generate dataset manifest with sample metadata and hashes (T016).

    Args:
        dataset_type: 'lidar' or 'camera'
        input_dir: Directory containing annotated samples
        output_path: Path to save manifest JSON
    """
    input_path = Path(input_dir)
    output_file = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    print(f"\n✓ Generating {dataset_type.upper()} dataset manifest")
    print(f"  Input: {input_path}")

    # Load samples
    if dataset_type == 'lidar':
        samples = load_lidar_samples(input_path)
    elif dataset_type == 'camera':
        samples = load_camera_samples(input_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Create manifest
    manifest = {
        'dataset_type': dataset_type,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'total_samples': len(samples),
        'splits': {
            'train': 0,  # Populated by split_dataset.py
            'val': 0,
            'test': 0
        },
        'samples': samples
    }

    # Compute manifest hash (excluding the hash field itself)
    manifest_str = json.dumps(manifest, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_str.encode()).hexdigest()
    manifest['dataset_hash'] = manifest_hash

    # Save manifest
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifest generated successfully")
    print(f"  Total samples: {len(samples)}")
    print(f"  Dataset hash: {manifest_hash[:16]}...")
    print(f"  Saved to: {output_file}")

    # Distribution statistics
    if dataset_type == 'lidar':
        scenario_counts = {}
        for sample in samples:
            tag = sample.get('scenario_tag', 'unknown')
            scenario_counts[tag] = scenario_counts.get(tag, 0) + 1
        print(f"\n  Scenario distribution:")
        for scenario, count in sorted(scenario_counts.items()):
            print(f"    {scenario}: {count}")

    elif dataset_type == 'camera':
        color_counts = {'red': 0, 'green': 0, 'blue': 0}
        lighting_counts = {}
        for sample in samples:
            for color in sample.get('colors', []):
                if color in color_counts:
                    color_counts[color] += 1
            tag = sample.get('lighting_tag', 'default')
            lighting_counts[tag] = lighting_counts.get(tag, 0) + 1

        print(f"\n  Color distribution:")
        for color, count in sorted(color_counts.items()):
            print(f"    {color}: {count}")
        print(f"\n  Lighting distribution:")
        for lighting, count in sorted(lighting_counts.items()):
            print(f"    {lighting}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset manifest (T016)")
    parser.add_argument('--dataset', required=True, choices=['lidar', 'camera'],
                        help='Dataset type')
    parser.add_argument('--input', required=True,
                        help='Input directory with annotated samples')
    parser.add_argument('--output', required=True,
                        help='Output manifest file path (JSON)')

    args = parser.parse_args()
    generate_manifest(args.dataset, args.input, args.output)


if __name__ == "__main__":
    main()
