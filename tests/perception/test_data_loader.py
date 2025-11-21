"""
Unit tests for data loaders
"""

import pytest


def test_dataset_imports():
    """Test dataset classes can be imported"""
    from src.perception.training.data_loader import LIDARDataset, CameraDataset
    assert LIDARDataset is not None
    assert CameraDataset is not None


def test_loader_functions_import():
    """Test loader factory functions can be imported"""
    from src.perception.training.data_loader import get_lidar_loaders, get_camera_loaders
    assert get_lidar_loaders is not None
    assert get_camera_loaders is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
