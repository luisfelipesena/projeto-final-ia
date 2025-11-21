"""
Custom Lightweight CNN for Cube Color Classification

Architecture (per DECISÃO 017):
- 3 Conv2D layers with BatchNorm + ReLU + MaxPool
- GlobalAveragePooling
- 2 FC layers with Dropout
- Output: 3-class softmax (green, blue, red)

Performance targets:
- Accuracy: >95% per color (SC-002)
- FPS: >10 (SC-004)
- Parameters: ~250K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchvision.models as models


class LightweightCNN(nn.Module):
    """
    Custom lightweight CNN for cube color classification

    Architecture:
    - Conv2D(3 → 32, k=7, s=2) + BN + ReLU + MaxPool(2)
    - Conv2D(32 → 64, k=5, s=2) + BN + ReLU + MaxPool(2)
    - Conv2D(64 → 128, k=3, s=1) + BN + ReLU + MaxPool(2)
    - GlobalAvgPool
    - FC(128 → 64) + ReLU + Dropout(0.5)
    - FC(64 → 3) + Softmax

    Input: [batch, 3, 512, 512] RGB images
    Output: [batch, 3] class probabilities (green, blue, red)
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch, 3, 512, 512] normalized RGB images

        Returns:
            logits: [batch, 3] class logits (apply softmax for probabilities)
        """
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
        x = self.pool3(x)

        # Global pooling: [batch, 128, H, W] → [batch, 128, 1, 1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: [batch, 128]

        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNet18Transfer(nn.Module):
    """
    ResNet18 transfer learning fallback (if accuracy <93%)

    Uses pre-trained ResNet18 with frozen early layers.
    Only fine-tunes final layers for 3-class classification.

    Performance: 95-98% accuracy but slower inference (~5-10 FPS)
    """

    def __init__(self, num_classes: int = 3, freeze_layers: int = 6):
        super().__init__()

        # Load pre-trained ResNet18
        self.backbone = models.resnet18(pretrained=True)

        # Freeze early layers
        layers = list(self.backbone.children())
        for layer in layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch, 3, 512, 512] normalized RGB images

        Returns:
            logits: [batch, 3] class logits
        """
        return self.backbone(x)

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_camera_model(use_transfer_learning: bool = False) -> nn.Module:
    """
    Factory function to create camera model

    Args:
        use_transfer_learning: If True, use ResNet18 (fallback)

    Returns:
        Camera model (LightweightCNN or ResNet18Transfer)
    """
    if use_transfer_learning:
        model = ResNet18Transfer()
        print("✓ Using ResNet18 transfer learning")
    else:
        model = LightweightCNN()
        print("✓ Using custom Lightweight CNN")

    print(f"  Parameters: {model.count_parameters():,}")
    return model


def test_camera_net():
    """Test camera network architectures"""
    print("Testing LightweightCNN...")

    model = LightweightCNN()
    print(f"  Parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 512, 512)

    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)

    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"  Probability sum: {probs.sum(dim=1)[0]:.3f} (should be 1.0)")

    assert logits.shape == (batch_size, 3)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    print("  ✓ LightweightCNN test passed")

    print("\nTesting ResNet18Transfer...")
    model_transfer = ResNet18Transfer()
    print(f"  Parameters: {model_transfer.count_parameters():,}")

    with torch.no_grad():
        logits = model_transfer(images)
        probs = F.softmax(logits, dim=1)

    assert logits.shape == (batch_size, 3)
    print("  ✓ ResNet18Transfer test passed")


def export_to_torchscript(model: nn.Module, output_path: str = "models/camera_net.pt"):
    """
    Export trained model to TorchScript format for deployment

    Args:
        model: Trained camera model
        output_path: Output file path
    """
    model.eval()

    # Create example input
    example_input = torch.randn(1, 3, 512, 512)

    # Trace model
    traced = torch.jit.trace(model, example_input)

    # Save
    traced.save(output_path)
    print(f"✓ Model exported to {output_path}")

    # Verify
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        output_original = model(example_input)
        output_loaded = loaded(example_input)

    assert torch.allclose(output_original, output_loaded, atol=1e-5)
    print(f"  ✓ Verification passed")


if __name__ == "__main__":
    test_camera_net()
