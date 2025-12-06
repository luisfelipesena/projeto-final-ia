"""Optional ICP correction to compensate odometry drift."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - dependency may not exist
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None  # type: ignore

from data_types import LidarPoint


class ICPCorrection:
    def __init__(self, distance_threshold: float = 0.15, icp_weight: float = 0.6):
        self._enabled = o3d is not None and np is not None
        self._distance_threshold = distance_threshold
        self._icp_weight = icp_weight
        self._reference: Optional[o3d.geometry.PointCloud] = None
        self._last_pose = None

    def available(self) -> bool:
        return self._enabled

    @property
    def distance_threshold(self) -> float:
        return self._distance_threshold

    def correct(self, points: Sequence[LidarPoint], odom_pose: Sequence[float]) -> Sequence[float]:
        if not self.available() or not points or np is None or o3d is None:
            return odom_pose
        current_cloud = self._build_cloud(points, odom_pose)
        if self._reference is None:
            self._reference = current_cloud
            self._last_pose = tuple(odom_pose)
            return odom_pose
        try:  # pragma: no cover - heavy computation
            result = o3d.pipelines.registration.registration_icp(
                current_cloud,
                self._reference,
                self._distance_threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
        except Exception:
            return odom_pose
        if result.fitness < 0.4:
            return odom_pose
        correction = result.transformation
        dx = float(correction[0, 3])
        dy = float(correction[1, 3])
        dtheta = float(np.arctan2(correction[1, 0], correction[0, 0]))
        blended = (
            (1 - self._icp_weight) * np.array(odom_pose)
            + self._icp_weight * np.array([
                odom_pose[0] + dx,
                odom_pose[1] + dy,
                odom_pose[2] + dtheta,
            ])
        )
        self._reference = current_cloud
        self._last_pose = tuple(blended.tolist())
        return blended.tolist()

    @staticmethod
    def _build_cloud(points: Sequence[LidarPoint], pose: Sequence[float]):
        if np is None or o3d is None:
            raise RuntimeError("Open3D not available")
        px, py, theta = pose
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        coords = []
        for point in points:
            gx = px + point.x * cos_t - point.y * sin_t
            gy = py + point.x * sin_t + point.y * cos_t
            coords.append([gx, gy, 0.0])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(coords, dtype=np.float32))
        return cloud
