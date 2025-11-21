"""
Hybrid LIDAR Neural Network

Architecture (per DECISÃO 016):
- 1D-CNN branch: Processes raw 667 LIDAR points → 64 features
- Hand-crafted branch: 6 statistical features
- Fusion: Concatenate 70 features → MLP classifier
- Output: 9-sector occupancy probabilities

Performance targets:
- Accuracy: >90% (SC-001)
- Latency: <100ms (SC-003)
- Parameters: ~250K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNBranch(nn.Module):
    """
    1D-CNN branch for processing raw LIDAR points

    Architecture:
    - Conv1D(1 → 32, k=7) + ReLU + MaxPool(2)
    - Conv1D(32 → 64, k=5) + ReLU + MaxPool(2)
    - Conv1D(64 → 64, k=3) + ReLU + AdaptiveAvgPool
    - Output: [64] features
    """

    def __init__(self, input_size: int = 667):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 667] LIDAR ranges

        Returns:
            features: [batch, 64] CNN features
        """
        # Add channel dimension: [batch, 667] → [batch, 1, 667]
        x = x.unsqueeze(1)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Global pooling: [batch, 64, seq_len] → [batch, 64, 1]
        x = self.global_pool(x)

        # Remove last dimension: [batch, 64, 1] → [batch, 64]
        x = x.squeeze(-1)

        return x


class MLPClassifier(nn.Module):
    """
    MLP classifier for fused features

    Architecture:
    - Linear(70 → 128) + ReLU + Dropout(0.3)
    - Linear(128 → 64) + ReLU + Dropout(0.3)
    - Linear(64 → 9) + Sigmoid
    - Output: [9] sector probabilities
    """

    def __init__(self, input_dim: int = 70, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.3):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dims[1], 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 70] fused features

        Returns:
            probabilities: [batch, 9] sector occupancy probabilities
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)  # Binary classification per sector

        return x


class HybridLIDARNet(nn.Module):
    """
    Hybrid LIDAR obstacle detection network

    Combines 1D-CNN features with hand-crafted statistical features
    for robust obstacle detection.

    Architecture:
    1. CNN branch: [667] → Conv1D layers → [64]
    2. Hand-crafted branch: [6] features (extracted externally)
    3. Fusion: Concatenate → [70]
    4. MLP classifier: [70] → [128] → [64] → [9]

    Usage:
        model = HybridLIDARNet()
        ranges = torch.randn(32, 667)  # Batch of 32 scans
        hand_crafted = torch.randn(32, 6)  # Hand-crafted features
        output = model(ranges, hand_crafted)  # [32, 9] probabilities
    """

    def __init__(self):
        super().__init__()

        self.cnn_branch = CNNBranch(input_size=667)
        self.mlp_classifier = MLPClassifier(input_dim=70, hidden_dims=(128, 64), dropout=0.3)

    def forward(self, ranges: torch.Tensor, hand_crafted: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            ranges: [batch, 667] normalized LIDAR ranges
            hand_crafted: [batch, 6] hand-crafted features

        Returns:
            probabilities: [batch, 9] sector occupancy probabilities
        """
        # CNN branch
        cnn_features = self.cnn_branch(ranges)  # [batch, 64]

        # Fuse with hand-crafted features
        fused = torch.cat([cnn_features, hand_crafted], dim=1)  # [batch, 70]

        # Classify
        probabilities = self.mlp_classifier(fused)  # [batch, 9]

        return probabilities

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_lidar_net():
    """Test LIDAR network architecture"""
    print("Testing HybridLIDARNet...")

    model = HybridLIDARNet()
    print(f"  Parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    ranges = torch.randn(batch_size, 667)
    hand_crafted = torch.randn(batch_size, 6)

    with torch.no_grad():
        output = model(ranges, hand_crafted)

    print(f"  Input shapes: ranges={ranges.shape}, hand_crafted={hand_crafted.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    assert output.shape == (batch_size, 9)
    assert torch.all((output >= 0) & (output <= 1))  # Probabilities

    print("  ✓ HybridLIDARNet test passed")


def export_to_torchscript(model: HybridLIDARNet, output_path: str = "models/lidar_net.pt"):
    """
    Export trained model to TorchScript format for deployment

    Args:
        model: Trained HybridLIDARNet
        output_path: Output file path
    """
    model.eval()

    # Create example inputs
    example_ranges = torch.randn(1, 667)
    example_hand_crafted = torch.randn(1, 6)

    # Trace model
    traced = torch.jit.trace(model, (example_ranges, example_hand_crafted))

    # Save
    traced.save(output_path)
    print(f"✓ Model exported to {output_path}")

    # Verify
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        output_original = model(example_ranges, example_hand_crafted)
        output_loaded = loaded(example_ranges, example_hand_crafted)

    assert torch.allclose(output_original, output_loaded, atol=1e-5)
    print(f"  ✓ Verification passed")


if __name__ == "__main__":
    test_lidar_net()
