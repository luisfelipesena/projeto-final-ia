"""
SimpleLIDARMLP - Neural Network for LIDAR-based Obstacle Detection
MATA64 Requirement: RNA for environment mapping

References:
- Goodfellow et al. (2016) - Deep Learning fundamentals
- Thrun et al. (2005) - Probabilistic Robotics (sensor processing)

Architecture:
    Input: N LIDAR points (normalized distances)
    Hidden1: 128 neurons + ReLU + Dropout(0.2)
    Hidden2: 64 neurons + ReLU + Dropout(0.2)
    Output: 9 sectors (obstacle probability 0-1)

Each sector covers ~40 degrees of FOV (360/9 = 40 deg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class SimpleLIDARMLP(nn.Module):
    """
    MLP for classifying LIDAR sectors as obstacle/free.

    Simple but effective architecture for real-time inference.
    Designed for quick implementation and easy debugging.
    """

    def __init__(self, input_size: int = 512, num_sectors: int = 9):
        """
        Args:
            input_size: Number of LIDAR points (default 512)
            num_sectors: Number of output sectors (default 9)
        """
        super().__init__()
        self.input_size = input_size
        self.num_sectors = num_sectors

        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_sectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LIDAR scan tensor (batch, input_size) with normalized distances

        Returns:
            Obstacle probabilities per sector (batch, num_sectors)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def preprocess_lidar(raw_scan: List[float], max_range: float = 5.0,
                         target_size: int = 512) -> torch.Tensor:
        """
        Preprocess raw LIDAR scan for network input.

        Args:
            raw_scan: List of distances from LIDAR
            max_range: Maximum range for normalization (meters)
            target_size: Expected input size (resample if different)

        Returns:
            Normalized tensor (1, target_size) ready for inference
        """
        scan = np.array(raw_scan, dtype=np.float32)

        # Handle inf/nan values
        scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=0.0)

        # Clip to max range
        scan = np.clip(scan, 0, max_range)

        # Resample if needed
        if len(scan) != target_size:
            indices = np.linspace(0, len(scan) - 1, target_size).astype(int)
            scan = scan[indices]

        # Normalize to [0, 1]
        scan = scan / max_range

        return torch.tensor(scan, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def create_labels_from_scan(scan: np.ndarray, threshold: float = 0.5,
                                num_sectors: int = 9) -> np.ndarray:
        """
        Auto-label: sector with min distance < threshold = obstacle (1)

        Args:
            scan: Normalized LIDAR scan (0-1 range)
            threshold: Normalized threshold (0.5 = 2.5m at max_range=5.0)
            num_sectors: Number of output sectors

        Returns:
            Labels array (num_sectors,) with 0/1 values
        """
        points_per_sector = len(scan) // num_sectors
        labels = np.zeros(num_sectors, dtype=np.float32)

        for i in range(num_sectors):
            start = i * points_per_sector
            end = start + points_per_sector
            sector_min = np.min(scan[start:end])
            labels[i] = 1.0 if sector_min < threshold else 0.0

        return labels


def test_simple_lidar_mlp():
    """Test SimpleLIDARMLP architecture"""
    print("Testing SimpleLIDARMLP...")

    model = SimpleLIDARMLP(input_size=512, num_sectors=9)
    print(f"  Parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 512)

    with torch.no_grad():
        output = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == (batch_size, 9)
    assert torch.all((output >= 0) & (output <= 1))

    # Test preprocessing
    raw_scan = [1.5] * 667  # Simulated raw scan with different size
    processed = SimpleLIDARMLP.preprocess_lidar(raw_scan, max_range=5.0, target_size=512)
    print(f"  Preprocess: {len(raw_scan)} points -> {processed.shape}")

    # Test auto-labeling
    labels = SimpleLIDARMLP.create_labels_from_scan(
        np.array([0.2, 0.8] * 256),  # Alternating near/far
        threshold=0.5
    )
    print(f"  Auto-labels shape: {labels.shape}")

    print("  SimpleLIDARMLP test passed")


if __name__ == "__main__":
    test_simple_lidar_mlp()
