"""Utility to generate synthetic cube dataset inside Webots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from controller import Supervisor
except ImportError:  # pragma: no cover
    Supervisor = None  # type: ignore


def _ensure_camera(supervisor: Supervisor, timestep: int):
    camera = supervisor.getDevice("camera")
    camera.enable(timestep)
    return camera


def _randomize_cube(supervisor: Supervisor, cube_def: str):
    if np is None:
        return
    node = supervisor.getFromDef(cube_def)
    if node is None:
        return
    translation = node.getField("translation")
    rotation = node.getField("rotation")
    tx = np.random.uniform(-3.0, 1.75)
    ty = np.random.uniform(-1.0, 1.0)
    tz = 0.015
    translation.setSFVec3f([tx, ty, tz])
    rotation.setSFRotation([0, 0, 1, np.random.uniform(0, 3.14)])


def capture_samples(output_dir: Path, class_name: str, camera, supervisor: Supervisor, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        supervisor.step(supervisor.getBasicTimeStep())
        filename = output_dir / f"{class_name}_{idx:04d}.png"
        camera.saveImage(str(filename), 100)


def generate_dataset(classes: Iterable[str], samples_per_class: int, output_root: Path):
    if Supervisor is None:  # pragma: no cover
        raise RuntimeError("Supervisor API not available outside Webots.")
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    camera = _ensure_camera(supervisor, timestep)
    for color in classes:
        cube_def = f"cube_template_{color}".upper()
        target_dir = output_root / color.lower()
        for idx in range(samples_per_class):
            _randomize_cube(supervisor, cube_def)
            for _ in range(5):
                supervisor.step(timestep)
            filename = target_dir / f"{color.lower()}_{idx:04d}.png"
            target_dir.mkdir(parents=True, exist_ok=True)
            camera.saveImage(str(filename), 100)


if __name__ == "__main__":  # pragma: no cover
    dataset_root = Path(os.environ.get("YOUBOT_DATASET_DIR", "datasets/cubes/train"))
    generate_dataset(["RED", "GREEN", "BLUE"], samples_per_class=200, output_root=dataset_root)
