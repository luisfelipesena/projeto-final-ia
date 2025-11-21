"""
LIDAR Processor Module

Neural network-based obstacle detection using LIDAR data:
- LIDARProcessor: Main inference class
- ObstacleMap: 9-sector occupancy representation
- Hand-crafted feature extraction (min, mean, std, occupancy, symmetry, variance)

Architecture: Hybrid MLP + 1D-CNN (per DECISÃO 016)
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class ObstacleMap:
    """
    9-sector occupancy map from LIDAR data

    Sectors are numbered 0-8 from left to right (270° FOV):
    - Sector 0: -135° to -105° (left)
    - Sector 4: -15° to +15° (front)
    - Sector 8: +105° to +135° (right)
    """

    sectors: np.ndarray  # [9] binary occupancy (0=free, 1=occupied)
    probabilities: np.ndarray  # [9] occupancy probabilities [0, 1]
    min_distances: np.ndarray  # [9] minimum distance per sector (meters)

    def is_sector_occupied(self, sector_idx: int, threshold: float = 0.5) -> bool:
        """Check if sector is occupied above threshold"""
        return self.probabilities[sector_idx] > threshold

    def get_clearance(self, sectors: list) -> float:
        """Get minimum clearance distance for specified sectors"""
        distances = self.min_distances[sectors]
        valid = distances[np.isfinite(distances)]
        return np.min(valid) if len(valid) > 0 else np.inf

    def is_path_clear(self, sectors: list, min_clearance: float = 1.0) -> bool:
        """Check if path through sectors is clear"""
        clearance = self.get_clearance(sectors)
        return clearance >= min_clearance


class HandCraftedFeatures:
    """
    Extract hand-crafted features from LIDAR scan

    Features (per DECISÃO 016):
    1. min_range: Minimum range measurement
    2. mean_range: Mean range
    3. std_range: Standard deviation of ranges
    4. occupancy_ratio: Fraction of points below threshold
    5. left_right_symmetry: Symmetry score between left/right
    6. variance: Range variance
    """

    OBSTACLE_THRESHOLD = 2.0  # meters

    @staticmethod
    def extract(ranges: np.ndarray) -> np.ndarray:
        """
        Extract 6 hand-crafted features from LIDAR scan

        Args:
            ranges: [667] LIDAR range measurements

        Returns:
            features: [6] hand-crafted features
        """
        # Filter invalid readings
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) == 0:
            # No valid data - return zeros
            return np.zeros(6, dtype=np.float32)

        # Feature 1: Minimum range
        min_range = np.min(valid_ranges)

        # Feature 2: Mean range
        mean_range = np.mean(valid_ranges)

        # Feature 3: Standard deviation
        std_range = np.std(valid_ranges)

        # Feature 4: Occupancy ratio (fraction below threshold)
        occupancy_ratio = np.mean(valid_ranges < HandCraftedFeatures.OBSTACLE_THRESHOLD)

        # Feature 5: Left-right symmetry
        # Split scan in half and compare
        mid = len(ranges) // 2
        left_ranges = ranges[:mid]
        right_ranges = ranges[mid:]

        # Reverse right to align with left
        right_reversed = right_ranges[::-1]

        # Compute correlation (higher = more symmetric)
        left_valid = left_ranges[np.isfinite(left_ranges)]
        right_valid = right_reversed[np.isfinite(right_reversed)]

        if len(left_valid) > 0 and len(right_valid) > 0:
            # Normalize and compute correlation
            min_len = min(len(left_valid), len(right_valid))
            left_norm = left_valid[:min_len] / (np.max(left_valid) + 1e-6)
            right_norm = right_valid[:min_len] / (np.max(right_valid) + 1e-6)
            symmetry = np.corrcoef(left_norm, right_norm)[0, 1]
            symmetry = np.clip(symmetry, -1.0, 1.0)  # Ensure valid range
        else:
            symmetry = 0.0

        # Feature 6: Variance
        variance = np.var(valid_ranges)

        features = np.array([
            min_range,
            mean_range,
            std_range,
            occupancy_ratio,
            symmetry,
            variance
        ], dtype=np.float32)

        return features


class LIDARProcessor:
    """
    Neural network-based LIDAR obstacle detection

    Loads trained model and performs inference on LIDAR scans.
    Returns 9-sector ObstacleMap with occupancy probabilities.

    Usage:
        processor = LIDARProcessor(model_path="models/lidar_net.pt")
        obstacle_map = processor.process(lidar_ranges)
        if obstacle_map.is_sector_occupied(4):  # Front sector
            print("Obstacle detected ahead!")
    """

    SECTORS = 9
    INPUT_SIZE = 667

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize LIDAR processor

        Args:
            model_path: Path to trained TorchScript model
            device: Computation device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)

        # Load model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = torch.jit.load(str(model_file), map_location=self.device)
        self.model.eval()

        print(f"✓ LIDARProcessor loaded from {model_path}")

    def preprocess(self, ranges: np.ndarray) -> torch.Tensor:
        """
        Preprocess LIDAR ranges for inference

        Args:
            ranges: [667] LIDAR measurements

        Returns:
            input_tensor: [1, 667] preprocessed tensor
        """
        # Handle invalid readings
        ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.0)

        # Normalize to [0, 1]
        ranges_normalized = np.clip(ranges, 0.0, 10.0) / 10.0

        # Convert to tensor
        tensor = torch.from_numpy(ranges_normalized).float()
        tensor = tensor.unsqueeze(0)  # Add batch dimension

        return tensor.to(self.device)

    def postprocess(self, output: torch.Tensor, ranges: np.ndarray) -> ObstacleMap:
        """
        Convert model output to ObstacleMap

        Args:
            output: [1, 9] model predictions (probabilities)
            ranges: [667] original LIDAR measurements

        Returns:
            ObstacleMap with sector occupancy information
        """
        # Get probabilities
        probabilities = output.squeeze(0).cpu().numpy()

        # Binary occupancy (threshold at 0.5)
        sectors = (probabilities > 0.5).astype(np.float32)

        # Compute min distances per sector
        num_points = len(ranges)
        points_per_sector = num_points // self.SECTORS
        min_distances = np.zeros(self.SECTORS, dtype=np.float32)

        for sector_idx in range(self.SECTORS):
            start_idx = sector_idx * points_per_sector
            end_idx = start_idx + points_per_sector
            sector_ranges = ranges[start_idx:end_idx]

            valid_ranges = sector_ranges[np.isfinite(sector_ranges)]
            if len(valid_ranges) > 0:
                min_distances[sector_idx] = np.min(valid_ranges)
            else:
                min_distances[sector_idx] = np.inf

        return ObstacleMap(
            sectors=sectors,
            probabilities=probabilities,
            min_distances=min_distances
        )

    @torch.no_grad()
    def process(self, ranges: np.ndarray) -> ObstacleMap:
        """
        Process LIDAR scan and detect obstacles

        Args:
            ranges: [667] LIDAR range measurements

        Returns:
            ObstacleMap with 9-sector occupancy information
        """
        # Preprocess
        input_tensor = self.preprocess(ranges)

        # Inference
        output = self.model(input_tensor)

        # Postprocess
        obstacle_map = self.postprocess(output, ranges)

        return obstacle_map


def test_hand_crafted_features():
    """Test hand-crafted feature extraction"""
    print("Testing hand-crafted features...")

    # Create sample scan
    ranges = np.random.uniform(0.5, 5.0, size=667)

    # Extract features
    features = HandCraftedFeatures.extract(ranges)

    print(f"  Features shape: {features.shape}")
    print(f"  Min range: {features[0]:.2f}m")
    print(f"  Mean range: {features[1]:.2f}m")
    print(f"  Std range: {features[2]:.2f}m")
    print(f"  Occupancy ratio: {features[3]:.2f}")
    print(f"  Symmetry: {features[4]:.2f}")
    print(f"  Variance: {features[5]:.2f}")

    assert features.shape == (6,)
    print("  ✓ Hand-crafted features test passed")


if __name__ == "__main__":
    test_hand_crafted_features()
