"""
Unit tests for neural network models
"""

import pytest


def test_lidar_net_import():
    """Test LIDAR network can be imported"""
    from src.perception.models.lidar_net import HybridLIDARNet
    assert HybridLIDARNet is not None


def test_camera_net_import():
    """Test camera network can be imported"""
    from src.perception.models.camera_net import LightweightCNN
    assert LightweightCNN is not None


def test_lidar_processor_import():
    """Test LIDAR processor can be imported"""
    from src.perception.lidar_processor import LIDARProcessor, ObstacleMap, HandCraftedFeatures
    assert LIDARProcessor is not None
    assert ObstacleMap is not None
    assert HandCraftedFeatures is not None


def test_models_init():
    """Test models package initialization"""
    from src.perception import models
    assert hasattr(models, 'HybridLIDARNet')
    assert hasattr(models, 'LightweightCNN')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
