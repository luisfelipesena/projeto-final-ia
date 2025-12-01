"""
SimpleLIDARMLP - Bridge to src/perception/lidar_mlp.
"""
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


class SimpleLIDARMLP(nn.Module if TORCH_AVAILABLE else object):
    """Simple LIDAR MLP for obstacle detection per sector."""

    def __init__(self, input_size: int = 512, num_sectors: int = 9, hidden_size: int = 128):
        if TORCH_AVAILABLE:
            super().__init__()

        self.input_size = input_size
        self.num_sectors = num_sectors

        if TORCH_AVAILABLE:
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 2, num_sectors),
                nn.Sigmoid(),
            )

    def forward(self, x):
        if TORCH_AVAILABLE:
            return self.network(x)
        return x

    @staticmethod
    def preprocess_lidar(ranges: List[float], max_range: float = 5.0, target_size: int = 512) -> 'torch.Tensor':
        """Preprocess LIDAR ranges for model input."""
        data = np.array(ranges, dtype=np.float32)

        # Resize if needed
        if len(data) != target_size:
            indices = np.linspace(0, len(data) - 1, target_size).astype(int)
            data = data[indices]

        # Normalize
        data = np.clip(data, 0.01, max_range)
        data = (data - 0.01) / (max_range - 0.01)

        if TORCH_AVAILABLE:
            return torch.from_numpy(data).unsqueeze(0)
        return data

    def predict(self, ranges: List[float]) -> List[float]:
        """Predict obstacle probabilities per sector."""
        if not TORCH_AVAILABLE:
            return self._fallback_predict(ranges)

        input_tensor = self.preprocess_lidar(ranges)
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
        return output.squeeze().tolist()

    def _fallback_predict(self, ranges: List[float]) -> List[float]:
        """Fallback when PyTorch not available."""
        points_per_sector = len(ranges) // self.num_sectors
        obstacles = []

        for i in range(self.num_sectors):
            start = i * points_per_sector
            end = start + points_per_sector
            sector = ranges[start:end]
            valid = [r for r in sector if 0.01 < r < 5.0]
            if valid:
                min_dist = min(valid)
                obstacles.append(1.0 if min_dist < 0.5 else 0.0)
            else:
                obstacles.append(0.0)

        return obstacles
