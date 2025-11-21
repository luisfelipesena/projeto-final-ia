"""
Dataset Classes for Neural Network Training

Implements PyTorch Dataset classes for LIDAR and camera data:
- LIDARDataset: Loads LIDAR scans with augmentation
- CameraDataset: Loads camera images with augmentation
- Supports train/val/test splits from splits.json
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import Optional, Tuple, List
from PIL import Image
import torchvision.transforms as T

from .augmentation import LIDARAugmentation, CameraAugmentation


class LIDARDataset(Dataset):
    """
    PyTorch Dataset for LIDAR obstacle detection

    Loads LIDAR scans with 667 range measurements and 9-sector occupancy labels.
    Applies augmentation during training.

    Args:
        data_dir: Root data directory (e.g., 'data/lidar')
        split: 'train', 'val', or 'test'
        augmentation: LIDARAugmentation instance (applied only if split='train')
        normalize: If True, normalize ranges to [0, 1]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        augmentation: Optional[LIDARAugmentation] = None,
        normalize: bool = True
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.scans_dir = self.data_dir / "scans"
        self.split = split
        self.normalize = normalize

        # Load split metadata
        splits_file = self.data_dir / "splits.json"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_file}\n"
                f"Run: python scripts/split_data.py --data-type lidar"
            )

        with open(splits_file, 'r') as f:
            splits_data = json.load(f)

        self.file_list = splits_data['splits'][split]

        # Setup augmentation (only for training)
        if split == 'train' and augmentation is not None:
            self.augmentation = augmentation
        else:
            self.augmentation = None

        print(f"✓ LIDARDataset ({split}): {len(self.file_list)} samples")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get LIDAR scan and labels

        Returns:
            ranges: [667] float32 tensor (normalized if enabled)
            labels: [9] float32 tensor (binary occupancy)
        """
        # Load scan
        scan_file = self.scans_dir / self.file_list[idx]
        data = np.load(scan_file)

        ranges = data['ranges'].astype(np.float32)
        labels = data['labels'].astype(np.float32)

        # Apply augmentation if training
        if self.augmentation is not None:
            ranges = self.augmentation(ranges)

        # Normalize ranges
        if self.normalize:
            # Clip to reasonable range [0, 10m] and normalize
            ranges = np.clip(ranges, 0.0, 10.0) / 10.0

        # Convert to tensors
        ranges = torch.from_numpy(ranges)
        labels = torch.from_numpy(labels)

        return ranges, labels

    def get_class_distribution(self) -> np.ndarray:
        """Compute label distribution for class weighting"""
        all_labels = []

        for filename in self.file_list:
            scan_file = self.scans_dir / filename
            data = np.load(scan_file)
            all_labels.append(data['labels'])

        all_labels = np.array(all_labels)
        return np.mean(all_labels, axis=0)


class CameraDataset(Dataset):
    """
    PyTorch Dataset for camera-based cube detection

    Loads RGB images with cube color labels.
    Applies augmentation during training.

    Args:
        data_dir: Root data directory (e.g., 'data/camera')
        split: 'train', 'val', or 'test'
        augmentation: CameraAugmentation instance (applied only if split='train')
        image_size: Target image size (width, height)
    """

    COLOR_TO_IDX = {'green': 0, 'blue': 1, 'red': 2}
    IDX_TO_COLOR = {0: 'green', 1: 'blue', 2: 'red'}

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        augmentation: Optional[CameraAugmentation] = None,
        image_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.split = split
        self.image_size = image_size

        # Load split metadata
        splits_file = self.data_dir / "splits.json"
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_file}\n"
                f"Run: python scripts/split_data.py --data-type camera"
            )

        with open(splits_file, 'r') as f:
            splits_data = json.load(f)

        self.file_list = splits_data['splits'][split]

        # Setup augmentation (only for training)
        if split == 'train' and augmentation is not None:
            self.augmentation = augmentation
        else:
            self.augmentation = None

        # Basic transforms (always applied)
        self.base_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"✓ CameraDataset ({split}): {len(self.file_list)} samples")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get camera image and color label

        Returns:
            image: [3, H, W] float32 tensor (normalized)
            label: [3] float32 tensor (one-hot encoded color class)
        """
        # Load image
        img_file = self.images_dir / self.file_list[idx]
        image = Image.open(img_file).convert('RGB')

        # Load label
        label_file = self.labels_dir / f"{img_file.stem}.json"
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        # Get dominant color (most frequent in image)
        cubes = label_data['cubes']
        if len(cubes) == 0:
            # Fallback: green (shouldn't happen with proper filtering)
            color_idx = 0
        else:
            colors = [cube['color'] for cube in cubes]
            dominant_color = max(set(colors), key=colors.count)
            color_idx = self.COLOR_TO_IDX[dominant_color]

        # Apply augmentation if training
        if self.augmentation is not None:
            image = self.augmentation(image)

        # Apply base transform
        image = self.base_transform(image)

        # Convert label to one-hot
        label = torch.zeros(3, dtype=torch.float32)
        label[color_idx] = 1.0

        return image, label

    def get_class_distribution(self) -> np.ndarray:
        """Compute color class distribution for class weighting"""
        class_counts = np.zeros(3, dtype=np.int32)

        for filename in self.file_list:
            label_file = self.labels_dir / f"{Path(filename).stem}.json"
            with open(label_file, 'r') as f:
                label_data = json.load(f)

            cubes = label_data['cubes']
            if len(cubes) > 0:
                colors = [cube['color'] for cube in cubes]
                dominant_color = max(set(colors), key=colors.count)
                color_idx = self.COLOR_TO_IDX[dominant_color]
                class_counts[color_idx] += 1

        return class_counts / max(1, class_counts.sum())


def get_lidar_loaders(
    data_dir: str = "data/lidar",
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation: bool = True
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test DataLoaders for LIDAR dataset

    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        augmentation: If True, apply augmentation to training set

    Returns:
        train_loader, val_loader, test_loader
    """
    # Setup augmentation
    aug = LIDARAugmentation() if augmentation else None

    # Create datasets
    train_dataset = LIDARDataset(data_dir, split='train', augmentation=aug)
    val_dataset = LIDARDataset(data_dir, split='val', augmentation=None)
    test_dataset = LIDARDataset(data_dir, split='test', augmentation=None)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_camera_loaders(
    data_dir: str = "data/camera",
    batch_size: int = 16,
    num_workers: int = 4,
    augmentation: bool = True
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create train/val/test DataLoaders for camera dataset

    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        augmentation: If True, apply augmentation to training set

    Returns:
        train_loader, val_loader, test_loader
    """
    # Setup augmentation
    aug = CameraAugmentation() if augmentation else None

    # Create datasets
    train_dataset = CameraDataset(data_dir, split='train', augmentation=aug)
    val_dataset = CameraDataset(data_dir, split='val', augmentation=None)
    test_dataset = CameraDataset(data_dir, split='test', augmentation=None)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing LIDARDataset...")
    try:
        dataset = LIDARDataset("data/lidar", split='train')
        print(f"  Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            ranges, labels = dataset[0]
            print(f"  Sample shape: ranges={ranges.shape}, labels={labels.shape}")
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")

    print("\nTesting CameraDataset...")
    try:
        dataset = CameraDataset("data/camera", split='train')
        print(f"  Dataset size: {len(dataset)}")
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"  Sample shape: image={image.shape}, label={label.shape}")
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
