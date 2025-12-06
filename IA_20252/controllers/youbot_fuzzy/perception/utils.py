"""Utility helpers for perception stack."""

from __future__ import annotations

from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy might be missing in some setups
    np = None  # type: ignore

from sensors.camera_stream import CameraFrame


def frame_to_bgr(frame: Optional[CameraFrame]):
    """Convert a Webots CameraFrame into a BGR numpy array."""
    if frame is None or frame.image is None or np is None:
        return None
    try:
        raw = np.frombuffer(frame.image, dtype=np.uint8)
        image = raw.reshape((frame.height, frame.width, 4))
        bgr = np.ascontiguousarray(image[:, :, :3])
        return bgr
    except ValueError:
        return None
