"""
Neural Network Architectures

This module defines the neural network architectures:
- simple_lidar_mlp: SimpleLIDARMLP for fast obstacle detection (DECISAO 028)
- lidar_net: Hybrid MLP + 1D-CNN for LIDAR processing (complex)
- camera_net: Custom Lightweight CNN for cube detection
"""

from .simple_lidar_mlp import SimpleLIDARMLP
from .lidar_net import HybridLIDARNet, CNNBranch, MLPClassifier
from .camera_net import LightweightCNN, ResNet18Transfer, create_camera_model

__all__ = [
    'SimpleLIDARMLP',
    'HybridLIDARNet',
    'CNNBranch',
    'MLPClassifier',
    'LightweightCNN',
    'ResNet18Transfer',
    'create_camera_model',
]
