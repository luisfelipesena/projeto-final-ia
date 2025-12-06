"""Webots controller entrypoint."""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # controllers/ directory

# Add current dir for local imports (app, config, etc.)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Add parent dir for sibling imports (youbot module)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app import main  # noqa: E402


if __name__ == "__main__":
    main()
