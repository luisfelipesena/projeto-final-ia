"""
Unit tests for data augmentation utilities
"""

import pytest
import numpy as np


def test_lidar_augmentation_import():
    """Test LIDAR augmentation can be imported"""
    from src.perception.training.augmentation import LIDARAugmentation
    assert LIDARAugmentation is not None


def test_camera_augmentation_import():
    """Test camera augmentation can be imported"""
    from src.perception.training.augmentation import CameraAugmentation
    assert CameraAugmentation is not None


def test_lidar_augmentation_initialization():
    """Test LIDAR augmentation initializes with correct parameters"""
    from src.perception.training.augmentation import LIDARAugmentation

    aug = LIDARAugmentation(
        noise_std=0.05,
        dropout_prob=0.1,
        rotation_range=10.0,
        apply_prob=0.5
    )

    assert aug.noise_std == 0.05
    assert aug.dropout_prob == 0.1
    assert aug.rotation_range == 10.0
    assert aug.apply_prob == 0.5


def test_camera_augmentation_initialization():
    """Test camera augmentation initializes with correct parameters"""
    from src.perception.training.augmentation import CameraAugmentation

    aug = CameraAugmentation(
        brightness_range=(0.8, 1.2),
        hue_range=(-0.05, 0.05),
        flip_prob=0.5,
        rotation_range=15.0
    )

    assert aug.brightness_range == (0.8, 1.2)
    assert aug.hue_range == (-0.05, 0.05)
    assert aug.flip_prob == 0.5
    assert aug.rotation_range == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
