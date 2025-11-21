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


