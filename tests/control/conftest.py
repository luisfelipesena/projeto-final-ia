"""
Pytest configuration for control module tests (fuzzy logic, state machine)

These tests do NOT require Webots simulation.
"""

import pytest
import sys
import os

# Add src to path
SRC_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, SRC_PATH)


@pytest.fixture(autouse=True)
def reset_robot():
    """Override root conftest fixture - control tests don't need Webots."""
    yield


@pytest.fixture
def youbot():
    """Override root conftest fixture - control tests don't need Webots."""
    pytest.skip("Control tests don't use the youbot fixture")


@pytest.fixture
def robot():
    """Override root conftest fixture - control tests don't need Webots."""
    pytest.skip("Control tests don't use the robot fixture")
