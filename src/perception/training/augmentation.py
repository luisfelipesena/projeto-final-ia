"""
Data Augmentation Utilities

Implements augmentation strategies for LIDAR and camera data to improve model generalization:
- LIDAR: Gaussian noise, dropout, rotation (per research.md)
- Camera: Brightness, hue, flip, rotation (per research.md)
"""

import numpy as np
import torch
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF


class LIDARAugmentation:
    """
    LIDAR data augmentation for obstacle detection training

    Techniques (per DECISÃO 016):
    - Gaussian noise: σ=0.05m (simulates sensor noise)
    - Dropout: 10% points (simulates occlusions)
    - Rotation: ±10° (simulates orientation uncertainty)
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        dropout_prob: float = 0.1,
        rotation_range: float = 10.0,
        apply_prob: float = 0.5
    ):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise (meters)
            dropout_prob: Probability of dropping each point
            rotation_range: Max rotation angle (degrees)
            apply_prob: Probability of applying each augmentation
        """
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rotation_range = rotation_range
        self.apply_prob = apply_prob

    def add_noise(self, ranges: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to LIDAR ranges"""
        noise = np.random.normal(0, self.noise_std, size=ranges.shape)
        noisy_ranges = ranges + noise
        # Clip to valid range [0, inf)
        return np.maximum(noisy_ranges, 0.0)

    def apply_dropout(self, ranges: np.ndarray) -> np.ndarray:
        """Randomly drop LIDAR points (set to max range)"""
        mask = np.random.random(size=ranges.shape) > self.dropout_prob
        dropped_ranges = ranges.copy()
        dropped_ranges[~mask] = np.inf  # Mark dropped points as no return
        return dropped_ranges

    def rotate(self, ranges: np.ndarray) -> np.ndarray:
        """Rotate LIDAR scan by random angle"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        num_points = len(ranges)
        shift = int((angle / 270.0) * num_points)  # 270° FOV

        # Circular shift
        rotated = np.roll(ranges, shift)
        return rotated

    def __call__(self, ranges: np.ndarray) -> np.ndarray:
        """
        Apply augmentation pipeline to LIDAR scan

        Args:
            ranges: [667] LIDAR range measurements

        Returns:
            augmented_ranges: [667] augmented LIDAR ranges
        """
        augmented = ranges.copy()

        # Apply each augmentation with probability
        if np.random.random() < self.apply_prob:
            augmented = self.add_noise(augmented)

        if np.random.random() < self.apply_prob:
            augmented = self.apply_dropout(augmented)

        if np.random.random() < self.apply_prob:
            augmented = self.rotate(augmented)

        return augmented


class CameraAugmentation:
    """
    Camera image augmentation for cube detection training

    Techniques (per DECISÃO 017):
    - Brightness: ±20%
    - Hue: ±10°
    - Horizontal flip: 50%
    - Rotation: ±15°
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.05, 0.05),
        flip_prob: float = 0.5,
        rotation_range: float = 15.0
    ):
        """
        Args:
            brightness_range: Min and max brightness factors
            hue_range: Min and max hue shift (fraction of 360°)
            flip_prob: Probability of horizontal flip
            rotation_range: Max rotation angle (degrees)
        """
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range

    def adjust_brightness(self, image: Image.Image) -> Image.Image:
        """Adjust image brightness"""
        factor = np.random.uniform(*self.brightness_range)
        return TF.adjust_brightness(image, factor)

    def adjust_hue(self, image: Image.Image) -> Image.Image:
        """Adjust image hue"""
        factor = np.random.uniform(*self.hue_range)
        return TF.adjust_hue(image, factor)

    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        """Flip image horizontally"""
        if np.random.random() < self.flip_prob:
            return TF.hflip(image)
        return image

    def rotate(self, image: Image.Image) -> Image.Image:
        """Rotate image by random angle"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        return TF.rotate(image, angle)

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation pipeline to camera image

        Args:
            image: PIL Image (RGB)

        Returns:
            augmented_image: Augmented PIL Image
        """
        # Apply augmentations
        image = self.adjust_brightness(image)
        image = self.adjust_hue(image)
        image = self.horizontal_flip(image)
        image = self.rotate(image)

        return image


class TorchLIDARAugmentation(torch.nn.Module):
    """PyTorch-compatible LIDAR augmentation for DataLoader"""

    def __init__(self, **kwargs):
        super().__init__()
        self.aug = LIDARAugmentation(**kwargs)

    def forward(self, ranges: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to tensor"""
        # Convert to numpy, augment, convert back
        ranges_np = ranges.numpy()
        augmented_np = self.aug(ranges_np)
        return torch.from_numpy(augmented_np)


class TorchCameraAugmentation(torch.nn.Module):
    """PyTorch-compatible camera augmentation for DataLoader"""

    def __init__(self, **kwargs):
        super().__init__()
        self.aug = CameraAugmentation(**kwargs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to tensor"""
        # Convert to PIL, augment, convert back
        image_pil = TF.to_pil_image(image)
        augmented_pil = self.aug(image_pil)
        return TF.to_tensor(augmented_pil)


def test_lidar_augmentation():
    """Test LIDAR augmentation"""
    print("Testing LIDAR augmentation...")

    # Create sample scan
    ranges = np.random.uniform(0.5, 5.0, size=667)

    # Apply augmentation
    aug = LIDARAugmentation()
    augmented = aug(ranges)

    assert augmented.shape == ranges.shape
    assert not np.array_equal(augmented, ranges)
    print(f"  ✓ Original range: [{ranges.min():.2f}, {ranges.max():.2f}]")
    print(f"  ✓ Augmented range: [{augmented.min():.2f}, {augmented.max():.2f}]")


def test_camera_augmentation():
    """Test camera augmentation"""
    print("\nTesting camera augmentation...")

    # Create sample image
    image = Image.new('RGB', (512, 512), color=(100, 150, 200))

    # Apply augmentation
    aug = CameraAugmentation()
    augmented = aug(image)

    assert augmented.size == image.size
    print(f"  ✓ Original size: {image.size}")
    print(f"  ✓ Augmented size: {augmented.size}")


if __name__ == "__main__":
    test_lidar_augmentation()
    test_camera_augmentation()
    print("\n✓ All augmentation tests passed")
