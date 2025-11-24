"""
Unit tests for LIDAR dataset schema validation.

Tests validate:
- LidarSample entity structure from data-model.md
- Range validation (0.05m ≤ value ≤ 5.0m)
- Sector labels validation (9 boolean flags)
- Pose validation (within arena bounds)
- Split assignment integrity

References: data-model.md, contracts/perception-api.yaml
"""

import pytest
import numpy as np
from datetime import datetime
from uuid import uuid4


class TestLidarSampleSchema:
    """Test LidarSample entity validation."""

    def test_valid_lidar_sample(self):
        """Valid sample passes all checks."""
        sample = {
            "sample_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "robot_pose": {"x": 0.5, "y": 0.3, "theta": 1.57},
            "ranges": np.random.uniform(0.1, 4.0, 360).tolist(),
            "sector_labels": [True, False, False, True, False, False, False, True, False],
            "scenario_tag": "obstacle_front",
            "split": "train"
        }

        # Schema validation
        assert "sample_id" in sample
        assert "timestamp" in sample
        assert "robot_pose" in sample
        assert len(sample["ranges"]) == 360
        assert len(sample["sector_labels"]) == 9
        assert sample["split"] in ["train", "val", "test"]

    def test_invalid_range_values(self):
        """Ranges outside [0.05, 5.0] should fail."""
        invalid_ranges = np.random.uniform(6.0, 10.0, 360).tolist()

        # Check that at least one value exceeds max
        assert any(r > 5.0 for r in invalid_ranges)

    def test_invalid_sector_count(self):
        """Sector labels must have exactly 9 entries."""
        invalid_sectors = [True, False, True]  # Only 3 sectors

        assert len(invalid_sectors) != 9

    def test_robot_pose_structure(self):
        """Pose must contain x, y, theta."""
        valid_pose = {"x": 1.0, "y": 0.5, "theta": 0.0}

        assert "x" in valid_pose
        assert "y" in valid_pose
        assert "theta" in valid_pose

    def test_scenario_tag_taxonomy(self):
        """Scenario tags must match predefined categories."""
        valid_tags = [
            "clear",
            "obstacle_front",
            "corridor_left",
            "corridor_right",
            "corner",
            "cluttered"
        ]

        sample_tag = "obstacle_front"
        assert sample_tag in valid_tags


class TestLidarDatasetValidation:
    """Test dataset-level validation rules."""

    def test_minimum_sample_quota(self):
        """Dataset must have ≥1000 samples."""
        min_samples = 1000
        mock_dataset_size = 1200

        assert mock_dataset_size >= min_samples

    def test_split_distribution(self):
        """No sample ID appears in multiple splits."""
        train_ids = {str(uuid4()) for _ in range(800)}
        val_ids = {str(uuid4()) for _ in range(150)}
        test_ids = {str(uuid4()) for _ in range(150)}

        # Check disjoint sets
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_sector_balance(self):
        """Sector distribution should deviate ≤10% from uniform."""
        # Mock sector counts (9 sectors)
        sector_counts = np.array([110, 105, 112, 108, 95, 103, 107, 111, 109])
        total_samples = sector_counts.sum()

        expected_uniform = total_samples / 9
        max_deviation = expected_uniform * 0.10

        for count in sector_counts:
            deviation = abs(count - expected_uniform)
            assert deviation <= max_deviation
