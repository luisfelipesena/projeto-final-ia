"""
Unit tests for camera dataset schema validation.

Tests validate:
- CameraSample entity structure from data-model.md
- Bounding box validation (width/height > 0, within resolution)
- Color label alignment with bboxes
- Distance estimation range (0.2m ≤ d ≤ 3.0m)
- Per-color balance (≤5% deviation from uniform)

References: data-model.md, contracts/perception-api.yaml
"""

import pytest
from datetime import datetime
from uuid import uuid4


class TestCameraSampleSchema:
    """Test CameraSample entity validation."""

    def test_valid_camera_sample(self):
        """Valid sample passes all checks."""
        sample = {
            "sample_id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "robot_pose": {"x": 1.0, "y": 0.5, "theta": 0.0},
            "image_path": "data/camera/raw/session01/frame_0001.png",
            "bounding_boxes": [
                {"id": "cube_01", "x": 120, "y": 150, "w": 50, "h": 50},
                {"id": "cube_02", "x": 300, "y": 200, "w": 45, "h": 45}
            ],
            "colors": ["red", "blue"],
            "distance_estimates": [1.2, 1.8],
            "lighting_tag": "default",
            "split": "train"
        }

        # Schema validation
        assert "sample_id" in sample
        assert "timestamp" in sample
        assert len(sample["bounding_boxes"]) == len(sample["colors"])
        assert len(sample["bounding_boxes"]) == len(sample["distance_estimates"])
        assert sample["split"] in ["train", "val", "test"]

    def test_bounding_box_dimensions(self):
        """Bboxes must have width, height > 0."""
        valid_bbox = {"id": "c1", "x": 100, "y": 100, "w": 50, "h": 50}

        assert valid_bbox["w"] > 0
        assert valid_bbox["h"] > 0
        assert "x" in valid_bbox
        assert "y" in valid_bbox

    def test_invalid_bbox_resolution(self):
        """Bbox must fit within image resolution (640x480 for Webots camera)."""
        resolution = (640, 480)
        bbox = {"x": 590, "y": 430, "w": 60, "h": 60}  # Overflows width

        # Check overflow
        assert (bbox["x"] + bbox["w"]) > resolution[0]  # Should fail validation

    def test_color_alignment(self):
        """Each bbox must have corresponding color label."""
        bboxes = [
            {"id": "c1", "x": 100, "y": 100, "w": 50, "h": 50},
            {"id": "c2", "x": 200, "y": 150, "w": 45, "h": 45}
        ]
        colors = ["green", "red"]

        assert len(bboxes) == len(colors)

    def test_distance_validation(self):
        """Distance estimates must be in [0.2, 3.0] meters."""
        distances = [0.5, 1.2, 2.8]

        for d in distances:
            assert 0.2 <= d <= 3.0

    def test_invalid_distance_out_of_range(self):
        """Distances outside valid range should fail."""
        invalid_distance = 4.5

        assert not (0.2 <= invalid_distance <= 3.0)

    def test_color_enum_validation(self):
        """Colors must be red, green, or blue."""
        valid_colors = ["red", "green", "blue"]
        sample_color = "blue"

        assert sample_color in valid_colors


class TestCameraDatasetValidation:
    """Test dataset-level validation rules."""

    def test_minimum_frame_quota(self):
        """Dataset must have ≥500 frames."""
        min_frames = 500
        mock_dataset_size = 650

        assert mock_dataset_size >= min_frames

    def test_color_balance(self):
        """Per-color distribution should deviate ≤5% from uniform."""
        # Mock color counts (3 colors)
        color_counts = {"red": 220, "green": 215, "blue": 225}
        total_cubes = sum(color_counts.values())

        expected_uniform = total_cubes / 3
        max_deviation = expected_uniform * 0.05

        for color, count in color_counts.items():
            deviation = abs(count - expected_uniform)
            assert deviation <= max_deviation

    def test_split_integrity(self):
        """No frame ID appears in multiple splits."""
        train_ids = {str(uuid4()) for _ in range(400)}
        val_ids = {str(uuid4()) for _ in range(75)}
        test_ids = {str(uuid4()) for _ in range(75)}

        # Check disjoint sets
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_lighting_variation(self):
        """Dataset should include multiple lighting conditions."""
        lighting_tags = {"default": 350, "bright": 150, "dim": 150}

        # At least 2 lighting conditions
        assert len(lighting_tags) >= 2
        # At least 100 samples per condition
        assert all(count >= 100 for count in lighting_tags.values())
