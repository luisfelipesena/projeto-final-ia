#!/usr/bin/env python3
"""
Dataset Schema Validation Script

Validates LidarSample and CameraSample schemas according to data-model.md
specifications.

Usage:
    python scripts/validate_dataset_schema.py --input data/lidar/annotated/ --type lidar
    python scripts/validate_dataset_schema.py --input data/camera/annotated/ --type camera
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from uuid import UUID
import re


def validate_uuid(value: str) -> bool:
    """Validate UUID format"""
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def validate_iso8601(value: str) -> bool:
    """Basic ISO8601 timestamp validation"""
    pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$'
    return bool(re.match(pattern, value))


def validate_lidar_sample(sample: Dict[str, Any]) -> List[str]:
    """Validate a single LidarSample against schema"""
    errors = []
    
    # Required fields
    required_fields = ['sample_id', 'timestamp', 'robot_pose', 'ranges', 'sector_labels', 'scenario_tag', 'split']
    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors  # Can't validate further without required fields
    
    # Validate sample_id (UUID)
    if not validate_uuid(sample['sample_id']):
        errors.append(f"Invalid UUID format for sample_id: {sample['sample_id']}")
    
    # Validate timestamp (ISO8601)
    if not validate_iso8601(sample['timestamp']):
        errors.append(f"Invalid ISO8601 timestamp: {sample['timestamp']}")
    
    # Validate robot_pose (struct {x, y, theta})
    if not isinstance(sample['robot_pose'], dict):
        errors.append("robot_pose must be a dict")
    else:
        pose_fields = ['x', 'y', 'theta']
        for field in pose_fields:
            if field not in sample['robot_pose']:
                errors.append(f"robot_pose missing field: {field}")
            elif not isinstance(sample['robot_pose'][field], (int, float)):
                errors.append(f"robot_pose.{field} must be numeric")
    
    # Validate ranges (float[360])
    if not isinstance(sample['ranges'], list):
        errors.append("ranges must be a list")
    elif len(sample['ranges']) != 360:
        errors.append(f"ranges must have exactly 360 elements, got {len(sample['ranges'])}")
    else:
        for i, r in enumerate(sample['ranges']):
            if not isinstance(r, (int, float)):
                errors.append(f"ranges[{i}] must be numeric")
            elif not (0.05 <= r <= 5.0):
                errors.append(f"ranges[{i}] = {r} outside valid range [0.05, 5.0]")
    
    # Validate sector_labels (bool[9])
    if not isinstance(sample['sector_labels'], list):
        errors.append("sector_labels must be a list")
    elif len(sample['sector_labels']) != 9:
        errors.append(f"sector_labels must have exactly 9 elements, got {len(sample['sector_labels'])}")
    else:
        for i, label in enumerate(sample['sector_labels']):
            if not isinstance(label, bool):
                errors.append(f"sector_labels[{i}] must be boolean")
    
    # Validate scenario_tag (enum)
    valid_scenarios = ['clear', 'obstacle_front', 'obstacle_critical', 'corridor_left', 'corridor_right', 
                       'cube_center_near', 'cube_center_far', 'cube_left', 'cube_right', 'mixed']
    if sample['scenario_tag'] not in valid_scenarios:
        errors.append(f"Invalid scenario_tag: {sample['scenario_tag']}. Must be one of {valid_scenarios}")
    
    # Validate split (enum)
    valid_splits = ['train', 'val', 'test']
    if sample['split'] not in valid_splits:
        errors.append(f"Invalid split: {sample['split']}. Must be one of {valid_splits}")
    
    return errors


def validate_camera_sample(sample: Dict[str, Any]) -> List[str]:
    """Validate a single CameraSample against schema"""
    errors = []
    
    # Required fields
    required_fields = ['sample_id', 'timestamp', 'robot_pose', 'image_path', 'bounding_boxes', 'colors', 
                       'distance_estimates', 'lighting_tag', 'split']
    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Validate sample_id (UUID)
    if not validate_uuid(sample['sample_id']):
        errors.append(f"Invalid UUID format for sample_id: {sample['sample_id']}")
    
    # Validate timestamp (ISO8601)
    if not validate_iso8601(sample['timestamp']):
        errors.append(f"Invalid ISO8601 timestamp: {sample['timestamp']}")
    
    # Validate robot_pose (struct {x, y, theta})
    if not isinstance(sample['robot_pose'], dict):
        errors.append("robot_pose must be a dict")
    else:
        pose_fields = ['x', 'y', 'theta']
        for field in pose_fields:
            if field not in sample['robot_pose']:
                errors.append(f"robot_pose missing field: {field}")
            elif not isinstance(sample['robot_pose'][field], (int, float)):
                errors.append(f"robot_pose.{field} must be numeric")
    
    # Validate image_path (string, must exist)
    image_path = Path(sample['image_path'])
    if not image_path.exists():
        errors.append(f"image_path does not exist: {sample['image_path']}")
    
    # Validate bounding_boxes (array of {id, x, y, w, h})
    if not isinstance(sample['bounding_boxes'], list):
        errors.append("bounding_boxes must be a list")
    else:
        for i, bbox in enumerate(sample['bounding_boxes']):
            if not isinstance(bbox, dict):
                errors.append(f"bounding_boxes[{i}] must be a dict")
            else:
                bbox_fields = ['id', 'x', 'y', 'w', 'h']
                for field in bbox_fields:
                    if field not in bbox:
                        errors.append(f"bounding_boxes[{i}] missing field: {field}")
                    elif not isinstance(bbox[field], (int, float)):
                        errors.append(f"bounding_boxes[{i}].{field} must be numeric")
                if 'w' in bbox and bbox['w'] <= 0:
                    errors.append(f"bounding_boxes[{i}].w must be > 0")
                if 'h' in bbox and bbox['h'] <= 0:
                    errors.append(f"bounding_boxes[{i}].h must be > 0")
    
    # Validate colors (array enum)
    valid_colors = ['red', 'green', 'blue']
    if not isinstance(sample['colors'], list):
        errors.append("colors must be a list")
    else:
        for i, color in enumerate(sample['colors']):
            if color not in valid_colors:
                errors.append(f"colors[{i}] = {color} not in {valid_colors}")
    
    # Validate distance_estimates alignment with bounding_boxes
    if isinstance(sample['bounding_boxes'], list) and isinstance(sample['distance_estimates'], list):
        if len(sample['distance_estimates']) != len(sample['bounding_boxes']):
            errors.append(f"distance_estimates length ({len(sample['distance_estimates'])}) must match bounding_boxes length ({len(sample['bounding_boxes'])})")
        else:
            for i, dist in enumerate(sample['distance_estimates']):
                if not isinstance(dist, (int, float)):
                    errors.append(f"distance_estimates[{i}] must be numeric")
                elif not (0.2 <= dist <= 3.0):
                    errors.append(f"distance_estimates[{i}] = {dist} outside valid range [0.2, 3.0]")
    
    # Validate lighting_tag (enum)
    valid_lighting = ['default', 'bright', 'dim']
    if sample['lighting_tag'] not in valid_lighting:
        errors.append(f"Invalid lighting_tag: {sample['lighting_tag']}. Must be one of {valid_lighting}")
    
    # Validate split (enum)
    valid_splits = ['train', 'val', 'test']
    if sample['split'] not in valid_splits:
        errors.append(f"Invalid split: {sample['split']}. Must be one of {valid_splits}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description='Validate dataset schema')
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
    
    # Validate each sample
    total_errors = 0
    for i, sample in enumerate(samples):
        if args.type == 'lidar':
            errors = validate_lidar_sample(sample)
        else:
            errors = validate_camera_sample(sample)
        
        if errors:
            total_errors += len(errors)
            sample_id = sample.get('sample_id', f'sample_{i}')
            print(f"Sample {sample_id} has {len(errors)} error(s):", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
    
    if total_errors == 0:
        print(f"✓ Validation passed: {len(samples)} samples validated successfully")
        sys.exit(0)
    else:
        print(f"✗ Validation failed: {total_errors} error(s) found in {len(samples)} samples", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


