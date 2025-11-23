#!/usr/bin/env python3
"""
Generate Dataset Manifest

Creates a JSON manifest for the dataset, including file hashes, metadata, and class distribution.
This ensures dataset integrity and provides a single source of truth for training.

Usage:
    python scripts/generate_dataset_manifest.py --data-dir data/lidar --type lidar --output data/lidar/manifest.json
    python scripts/generate_dataset_manifest.py --data-dir data/camera --type camera --output data/camera/manifest.json
"""

import argparse
import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def process_lidar_dataset(data_dir: Path) -> Dict[str, Any]:
    """Process LIDAR dataset and return manifest data"""
    scans_dir = data_dir / "scans"
    scan_files = sorted(list(scans_dir.glob("*.npz")))
    
    samples = []
    occupancy_counts = {}
    
    print(f"Processing {len(scan_files)} LIDAR scans...")
    
    for scan_file in scan_files:
        # Load data to get metadata and labels
        try:
            data = np.load(scan_file, allow_pickle=True)
            labels = data['labels']
            metadata = data['metadata'].item() if 'metadata' in data else {}
            
            # Calculate occupancy count
            occupied_count = int(np.sum(labels))
            occupancy_counts[occupied_count] = occupancy_counts.get(occupied_count, 0) + 1
            
            # Create sample entry
            sample = {
                'id': scan_file.stem,
                'file_path': str(scan_file.relative_to(data_dir)),
                'hash': calculate_file_hash(scan_file),
                'labels': labels.tolist(),
                'occupied_sectors': occupied_count,
                'metadata': metadata
            }
            samples.append(sample)
            
        except Exception as e:
            print(f"Error processing {scan_file}: {e}")
            
    return {
        'type': 'lidar',
        'count': len(samples),
        'stats': {
            'occupancy_distribution': occupancy_counts
        },
        'samples': samples
    }


def process_camera_dataset(data_dir: Path) -> Dict[str, Any]:
    """Process Camera dataset and return manifest data"""
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    image_files = sorted(list(images_dir.glob("*.png")))
    
    samples = []
    class_counts = {'green': 0, 'blue': 0, 'red': 0}
    
    print(f"Processing {len(image_files)} camera images...")
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.json"
        
        if not label_file.exists():
            print(f"Warning: No label found for {img_file}")
            continue
            
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            
            cubes = label_data.get('cubes', [])
            metadata = label_data.get('metadata', {})
            
            # Update stats
            sample_classes = []
            for cube in cubes:
                color = cube['color']
                class_counts[color] = class_counts.get(color, 0) + 1
                sample_classes.append(color)
            
            # Create sample entry
            sample = {
                'id': img_file.stem,
                'image_path': str(img_file.relative_to(data_dir)),
                'label_path': str(label_file.relative_to(data_dir)),
                'image_hash': calculate_file_hash(img_file),
                'label_hash': calculate_file_hash(label_file),
                'cubes': cubes,
                'classes': sample_classes,
                'metadata': metadata
            }
            samples.append(sample)
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            
    return {
        'type': 'camera',
        'count': len(samples),
        'stats': {
            'class_counts': class_counts
        },
        'samples': samples
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Dataset Manifest")
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--type', type=str, required=True, choices=['lidar', 'camera'], help='Dataset type')
    parser.add_argument('--output', type=str, required=True, help='Output manifest JSON file')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} not found")
        return
        
    if args.type == 'lidar':
        manifest_data = process_lidar_dataset(data_dir)
    else:
        manifest_data = process_camera_dataset(data_dir)
        
    # Add global metadata
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'dataset_type': args.type,
        'root_dir': str(data_dir.resolve()),
        **manifest_data
    }
    
    # Save manifest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"✓ Manifest generated at {output_path}")
    print(f"  Total samples: {manifest['count']}")


if __name__ == "__main__":
    main()
