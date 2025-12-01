"""
CubeDetector bridge - wraps src/perception/cube_detector.
"""
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Re-export from src
try:
    from perception.cube_detector import CubeDetector, CubeDetection
except ImportError:
    # Fallback simple detector if src not available
    from dataclasses import dataclass
    from typing import List, Optional
    import numpy as np

    @dataclass
    class CubeDetection:
        color: str
        center_x: int
        center_y: int
        width: int
        height: int
        area: int
        distance: float
        angle: float
        confidence: float

    class CubeDetector:
        def __init__(self):
            pass

        def detect(self, image: np.ndarray) -> List[CubeDetection]:
            return []

        def detect_nearest(self, image: np.ndarray) -> Optional[CubeDetection]:
            return None
