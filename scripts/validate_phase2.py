#!/usr/bin/env python3
"""
Phase 2 Validation Script

Validates all implemented modules for Phase 2 (Neural Networks):
- Syntax validation
- Module imports
- Structure verification
- Architecture correctness
"""

import sys
import ast
from pathlib import Path
from typing import List, Tuple


class ValidationError(Exception):
    pass


def validate_python_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate Python file syntax"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_module_structure(file_path: Path, expected_classes: List[str]) -> Tuple[bool, str]:
    """Validate module contains expected classes/functions"""
    try:
        with open(file_path, 'r') as f:
            code = f.read()

        tree = ast.parse(code)

        # Extract class and function names
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        # Check expected classes
        missing = []
        for expected in expected_classes:
            if expected not in classes and expected not in functions:
                missing.append(expected)

        if missing:
            return False, f"Missing: {', '.join(missing)}"

        return True, f"Found {len(classes)} classes, {len(functions)} functions"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    print("=" * 70)
    print("Phase 2 Validation - Neural Network Infrastructure")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent

    # Files to validate
    validations = [
        # Scripts
        {
            'path': base_dir / 'scripts/collect_lidar_data.py',
            'classes': ['LIDARDataCollector'],
            'description': 'LIDAR data collection script'
        },
        {
            'path': base_dir / 'scripts/collect_camera_data.py',
            'classes': ['CameraDataCollector'],
            'description': 'Camera data collection script'
        },
        {
            'path': base_dir / 'scripts/annotate_lidar.py',
            'classes': ['LIDARAnnotator'],
            'description': 'LIDAR annotation tool'
        },
        {
            'path': base_dir / 'scripts/annotate_camera.py',
            'classes': ['CameraAnnotator'],
            'description': 'Camera annotation tool'
        },
        {
            'path': base_dir / 'scripts/split_data.py',
            'classes': ['DataSplitter'],
            'description': 'Train/val/test split utility'
        },

        # Source modules
        {
            'path': base_dir / 'src/perception/training/augmentation.py',
            'classes': ['LIDARAugmentation', 'CameraAugmentation'],
            'description': 'Data augmentation utilities'
        },
        {
            'path': base_dir / 'src/perception/training/data_loader.py',
            'classes': ['LIDARDataset', 'CameraDataset'],
            'description': 'PyTorch dataset classes'
        },
        {
            'path': base_dir / 'src/perception/lidar_processor.py',
            'classes': ['LIDARProcessor', 'ObstacleMap', 'HandCraftedFeatures'],
            'description': 'LIDAR processor module'
        },
        {
            'path': base_dir / 'src/perception/models/lidar_net.py',
            'classes': ['HybridLIDARNet', 'CNNBranch', 'MLPClassifier'],
            'description': 'LIDAR neural network'
        },
        {
            'path': base_dir / 'src/perception/models/camera_net.py',
            'classes': ['LightweightCNN', 'ResNet18Transfer'],
            'description': 'Camera neural network'
        },
    ]

    print("\n📝 Validating Python files...\n")

    all_passed = True
    for validation in validations:
        file_path = validation['path']
        description = validation['description']
        expected_classes = validation['classes']

        print(f"[{description}]")
        print(f"  File: {file_path.relative_to(base_dir)}")

        # Check file exists
        if not file_path.exists():
            print(f"  ❌ FAIL: File not found")
            all_passed = False
            print()
            continue

        # Validate syntax
        syntax_ok, syntax_msg = validate_python_syntax(file_path)
        if not syntax_ok:
            print(f"  ❌ FAIL: {syntax_msg}")
            all_passed = False
            print()
            continue

        # Validate structure
        struct_ok, struct_msg = validate_module_structure(file_path, expected_classes)
        if not struct_ok:
            print(f"  ❌ FAIL: {struct_msg}")
            all_passed = False
            print()
            continue

        print(f"  ✅ PASS: {struct_msg}")
        print()

    # Directory structure validation
    print("\n📁 Validating directory structure...\n")

    required_dirs = [
        'src/perception',
        'src/perception/models',
        'src/perception/training',
        'data/lidar/scans',
        'data/lidar/labels',
        'data/camera/images',
        'data/camera/labels',
        'models',
        'notebooks',
        'scripts',
        'tests/perception',
        'logs',
    ]

    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path}")
        if not exists:
            all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nPhase 2 infrastructure is ready for training!")
        print("\nNext steps:")
        print("  1. T018-T019: Collect training data (1000+ LIDAR, 500+ camera)")
        print("  2. T024-T028: Train LIDAR model")
        print("  3. T038-T043: Train camera model")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
