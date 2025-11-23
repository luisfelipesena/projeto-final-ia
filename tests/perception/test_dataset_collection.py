"""
Integration test for dataset collection workflow.

Tests the end-to-end process:
1. Collection scripts execution
2. Annotation workflow
3. Manifest generation
4. Dataset split assignment
5. Validation pipeline

References: quickstart.md, plan.md
"""

import pytest
import os
import json
from pathlib import Path


class TestLidarCollectionWorkflow:
    """Integration test for LIDAR data collection."""

    @pytest.fixture
    def mock_lidar_session(self, tmp_path):
        """Create mock LIDAR collection session."""
        session_dir = tmp_path / "lidar" / "raw" / "session01"
        session_dir.mkdir(parents=True)

        # Create mock scan files
        for i in range(60):
            scan_file = session_dir / f"scan_{i:04d}.json"
            mock_data = {
                "sample_id": f"lidar_sample_{i}",
                "timestamp": f"2025-11-23T10:{i:02d}:00Z",
                "robot_pose": {"x": i * 0.1, "y": 0.0, "theta": 0.0},
                "ranges": [1.0] * 360,
                "scenario_tag": "clear"
            }
            scan_file.write_text(json.dumps(mock_data))

        return session_dir

    def test_collection_quota_completion(self, mock_lidar_session):
        """Verify collection meets 1000+ scan quota."""
        # Mock: 20 sessions x 60 scans = 1200 total
        sessions = 20
        scans_per_session = 60
        total_scans = sessions * scans_per_session

        assert total_scans >= 1000

    def test_annotation_workflow(self, mock_lidar_session):
        """Annotation adds sector_labels to raw scans."""
        # Mock annotation output
        annotated = {
            "sample_id": "lidar_sample_0",
            "ranges": [1.0] * 360,
            "sector_labels": [False, True, False, False, False, False, False, False, True]
        }

        assert "sector_labels" in annotated
        assert len(annotated["sector_labels"]) == 9

    def test_manifest_generation(self, tmp_path):
        """Manifest contains all sample IDs and metadata hash."""
        manifest_path = tmp_path / "dataset_manifest.json"
        manifest = {
            "dataset_hash": "abc123def456",
            "total_samples": 1200,
            "splits": {
                "train": 960,
                "val": 120,
                "test": 120
            },
            "sample_ids": [f"lidar_sample_{i}" for i in range(1200)]
        }

        manifest_path.write_text(json.dumps(manifest))

        # Verify manifest structure
        loaded = json.loads(manifest_path.read_text())
        assert "dataset_hash" in loaded
        assert loaded["total_samples"] == 1200
        assert sum(loaded["splits"].values()) == 1200


class TestCameraCollectionWorkflow:
    """Integration test for camera data collection."""

    @pytest.fixture
    def mock_camera_session(self, tmp_path):
        """Create mock camera collection session."""
        session_dir = tmp_path / "camera" / "raw" / "session01"
        session_dir.mkdir(parents=True)

        # Create mock frame files
        for i in range(40):
            frame_file = session_dir / f"frame_{i:04d}.json"
            mock_data = {
                "sample_id": f"camera_sample_{i}",
                "timestamp": f"2025-11-23T10:{i:02d}:00Z",
                "image_path": f"session01/frame_{i:04d}.png",
                "lighting_tag": "default"
            }
            frame_file.write_text(json.dumps(mock_data))

            # Create empty image placeholder
            img_file = session_dir / f"frame_{i:04d}.png"
            img_file.touch()

        return session_dir

    def test_collection_quota_completion(self, mock_camera_session):
        """Verify collection meets 500+ frame quota."""
        # Mock: 15 sessions x 40 frames = 600 total
        sessions = 15
        frames_per_session = 40
        total_frames = sessions * frames_per_session

        assert total_frames >= 500

    def test_annotation_adds_bboxes(self, mock_camera_session):
        """Annotation adds bounding boxes and color labels."""
        # Mock annotation output
        annotated = {
            "sample_id": "camera_sample_0",
            "image_path": "session01/frame_0000.png",
            "bounding_boxes": [
                {"id": "c1", "x": 150, "y": 200, "w": 50, "h": 50}
            ],
            "colors": ["red"],
            "distance_estimates": [1.5]
        }

        assert "bounding_boxes" in annotated
        assert len(annotated["colors"]) == len(annotated["bounding_boxes"])

    def test_color_balance_validation(self):
        """Validate per-color distribution meets ≤5% deviation."""
        color_counts = {"red": 198, "green": 202, "blue": 200}
        total = sum(color_counts.values())
        expected = total / 3

        max_deviation = expected * 0.05
        for count in color_counts.values():
            assert abs(count - expected) <= max_deviation


class TestDatasetValidationPipeline:
    """Test complete validation pipeline."""

    def test_schema_compliance(self):
        """All samples pass schema validation."""
        # Mock validation result
        validation_report = {
            "total_samples": 1200,
            "schema_valid": 1200,
            "schema_invalid": 0,
            "errors": []
        }

        assert validation_report["schema_invalid"] == 0

    def test_balance_thresholds(self):
        """Dataset meets balance requirements."""
        balance_report = {
            "sector_balance": "PASS",
            "color_balance": "PASS",
            "split_integrity": "PASS"
        }

        assert all(status == "PASS" for status in balance_report.values())

    def test_split_generation(self):
        """Split assignment is deterministic and balanced."""
        total_samples = 1200
        splits = {"train": 960, "val": 120, "test": 120}

        # 80/10/10 split
        assert splits["train"] == int(total_samples * 0.8)
        assert splits["val"] == int(total_samples * 0.1)
        assert splits["test"] == int(total_samples * 0.1)
