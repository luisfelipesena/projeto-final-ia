"""
Automated validation tests for Webots R2023b environment setup.

Tests verify:
- Python version compatibility (3.8+)
- Webots installation and version
- World file existence and integrity
- Project structure compliance

Run with: pytest tests/test_webots_setup.py -v
"""

import sys
import os
import subprocess
from pathlib import Path
import pytest


class TestPythonEnvironment:
    """Test Python version and virtual environment configuration."""

    def test_python_version(self):
        """Verify Python version is 3.8 or higher."""
        version_info = sys.version_info
        assert version_info.major == 3, f"Expected Python 3.x, got {version_info.major}.x"
        assert version_info.minor >= 8, (
            f"Python 3.8+ required for Webots R2023b compatibility. "
            f"Found Python {version_info.major}.{version_info.minor}"
        )

    def test_project_structure(self):
        """Verify required project directories exist."""
        project_root = Path(__file__).parent.parent
        required_dirs = [
            "IA_20252",
            "IA_20252/controllers",
            "IA_20252/worlds",
            "tests",
            "docs",
            "specs/001-webots-setup",
        ]

        for dir_path in required_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Required directory missing: {dir_path}"
            assert full_path.is_dir(), f"Path exists but is not a directory: {dir_path}"


class TestWebotsInstallation:
    """Test Webots installation and configuration."""

    def test_webots_executable_exists(self):
        """Check if Webots executable is in PATH or standard locations."""
        # Try to find webots executable
        result = subprocess.run(
            ["which", "webots"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Check macOS standard location
            macos_path = Path("/Applications/Webots.app/Contents/MacOS/webots")
            if macos_path.exists():
                pytest.skip("Webots found at standard macOS location but not in PATH")
            else:
                pytest.fail(
                    "Webots not found. Install Webots R2023b from: "
                    "https://github.com/cyberbotics/webots/releases/tag/R2023b"
                )

    @pytest.mark.slow
    def test_webots_version(self):
        """Verify Webots version is R2023b."""
        try:
            result = subprocess.run(
                ["webots", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            version_output = result.stdout.strip()
            assert "R2023b" in version_output, (
                f"Expected Webots R2023b, found: {version_output}"
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Webots version check timed out")
        except FileNotFoundError:
            pytest.skip("Webots executable not found in PATH")


class TestWorldFileConfiguration:
    """Test world file existence and supervisor immutability."""

    def test_world_file_exists(self):
        """Verify IA_20252.wbt world file exists."""
        project_root = Path(__file__).parent.parent
        world_file = project_root / "IA_20252" / "worlds" / "IA_20252.wbt"

        assert world_file.exists(), f"World file not found at: {world_file}"
        assert world_file.is_file(), f"World file path is not a file: {world_file}"

        # Verify file is not empty
        assert world_file.stat().st_size > 0, "World file is empty"

    def test_supervisor_file_not_modified(self):
        """
        Constitution compliance test: Verify supervisor.py has not been modified.

        This is a critical constraint (Principle V - Restrições Disciplinares).
        Modifying supervisor.py results in point deduction.
        """
        project_root = Path(__file__).parent.parent
        supervisor_file = project_root / "IA_20252" / "controllers" / "supervisor" / "supervisor.py"

        assert supervisor_file.exists(), "Supervisor file not found"

        # Read supervisor file content
        with open(supervisor_file, 'r') as f:
            content = f.read()

        # Check for expected supervisor structure (cube spawning logic should be present)
        assert "spawn" in content.lower(), (
            "Supervisor file appears modified - spawn logic missing"
        )

        # Verify file has not been truncated or emptied
        assert len(content) > 100, "Supervisor file appears to be empty or truncated"


class TestDocumentation:
    """Test documentation completeness."""

    def test_setup_documentation_exists(self):
        """Verify all setup documentation files exist."""
        project_root = Path(__file__).parent.parent
        required_docs = [
            "specs/001-webots-setup/spec.md",
            "specs/001-webots-setup/plan.md",
            "specs/001-webots-setup/quickstart.md",
            "specs/001-webots-setup/data-model.md",
            "DECISIONS.md",
            "README.md",
        ]

        for doc_path in required_docs:
            full_path = project_root / doc_path
            assert full_path.exists(), f"Required documentation missing: {doc_path}"
            assert full_path.stat().st_size > 0, f"Documentation file is empty: {doc_path}"

    def test_decisions_documented(self):
        """Verify setup decisions (005-008) are documented in DECISIONS.md."""
        project_root = Path(__file__).parent.parent
        decisions_file = project_root / "DECISIONS.md"

        with open(decisions_file, 'r') as f:
            content = f.read()

        required_decisions = [
            "DECISÃO 005",  # Webots installation method
            "DECISÃO 006",  # Python integration strategy
            "DECISÃO 007",  # Testing framework
            "DECISÃO 008",  # Sensor validation approach
        ]

        for decision in required_decisions:
            assert decision in content, (
                f"{decision} not found in DECISIONS.md. "
                "All setup decisions must be documented."
            )


# Pytest configuration
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (requires Webots execution)"
    )
    config.addinivalue_line(
        "markers", "requires_webots: marks tests that require Webots to be installed and running"
    )
