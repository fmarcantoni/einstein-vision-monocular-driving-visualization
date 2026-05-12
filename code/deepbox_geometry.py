"""
deepbox_geometry.py
===================

Deep3DBox-style 3D box fitting utilities derived from the paper
"3D Bounding Box Estimation Using Deep Learning and Geometry"
(Mousavian et al., CVPR 2017 / arXiv:1612.00496).

The external ``3DVehicleDetection`` repository was useful as a notebook
reference for the project, but the runtime helper in this file is implemented
as a self-contained paper-based geometry solver that can be called directly
from ``vehicle_3d_detection.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_BOX_INDICES: List[List[int]] = []
for i in (1, 3, 5, 7):
    for j in (1, 3, 5, 7):
        for m in (0, 1, 2, 3):
            for n in (0, 1, 2, 3):
                _BOX_INDICES.append([i, j, m, n])


def build_projection_matrix(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    return np.asarray(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def init_points3d(dims_hwl: Sequence[float]) -> np.ndarray:
    dims = np.asarray(dims_hwl, dtype=np.float64).reshape(3)
    points3d = np.zeros((8, 3), dtype=np.float64)
    cnt = 0
    for i in (1, -1):
        for j in (1, -1):
            for k in (1, -1):
                points3d[cnt] = dims[[1, 0, 2]].T / 2.0 * np.asarray([i, k, j * i], dtype=np.float64)
                cnt += 1
    return points3d


def rotation_y(yaw_cam_rad: float) -> np.ndarray:
    return np.asarray(
        [
            [math.cos(yaw_cam_rad), 0.0, math.sin(yaw_cam_rad)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw_cam_rad), 0.0, math.cos(yaw_cam_rad)],
        ],
        dtype=np.float64,
    )


def solve_least_square(W: np.ndarray, y: np.ndarray) -> np.ndarray:
    U, sigma, VT = np.linalg.svd(W)
    sigma_mat = np.eye(4, 3, dtype=np.float64) * sigma
    return VT.T @ np.linalg.pinv(sigma_mat) @ U.T @ y


def points3d_to_2d(
    points3d: np.ndarray,
    center_cam: np.ndarray,
    rot_mat: np.ndarray,
    proj_mat: np.ndarray,
) -> np.ndarray:
    points2d: List[np.ndarray] = []
    for point3d in points3d:
        point = center_cam + rot_mat @ point3d.reshape((-1, 1))
        point_h = np.append(point.reshape(3), 1.0)
        projected = proj_mat @ point_h
        projected_xy = projected[:2] / max(projected[2], 1e-6)
        points2d.append(projected_xy)
    return np.asarray(points2d, dtype=np.float64)


def compute_bbox_error(
    points2d: np.ndarray,
    bbox_xyxy: Sequence[float],
) -> float:
    xmin, ymin, xmax, ymax = [float(v) for v in bbox_xyxy[:4]]
    new_box = np.asarray(
        [
            np.min(points2d[:, 0]),
            np.min(points2d[:, 1]),
            np.max(points2d[:, 0]),
            np.max(points2d[:, 1]),
        ],
        dtype=np.float64,
    )
    ref_box = np.asarray([xmin, ymin, xmax, ymax], dtype=np.float64)
    return float(np.sum(np.abs(new_box - ref_box)))


def compute_center(
    points3d: np.ndarray,
    rot_mat: np.ndarray,
    proj_mat: np.ndarray,
    bbox_xyxy: Sequence[float],
    indices: Iterable[Sequence[int]] = _BOX_INDICES,
) -> Optional[np.ndarray]:
    xmin, ymin, xmax, ymax = [float(v) for v in bbox_xyxy[:4]]
    box = np.asarray([xmin, xmax, ymin, ymax], dtype=np.float64).reshape((-1, 1))

    fx = float(proj_mat[0][0])
    fy = float(proj_mat[1][1])
    u0 = float(proj_mat[0][2])
    v0 = float(proj_mat[1][2])

    W = np.asarray(
        [
            [fx, 0.0, u0 - box[0, 0]],
            [fx, 0.0, u0 - box[1, 0]],
            [0.0, fy, v0 - box[2, 0]],
            [0.0, fy, v0 - box[3, 0]],
        ],
        dtype=np.float64,
    )

    best_center = None
    best_error = float("inf")

    for ind in indices:
        y = np.zeros((4, 1), dtype=np.float64)
        for row_idx, point_idx in enumerate(ind):
            rotated_point = rot_mat @ points3d[int(point_idx)].reshape((-1, 1))
            y[row_idx] = box[row_idx] * proj_mat[2, 3] - np.dot(W[row_idx], rotated_point) - proj_mat[row_idx // 2, 3]

        center = solve_least_square(W, y)
        if not np.isfinite(center).all():
            continue
        if float(center[2, 0]) <= 0.1:
            continue

        points2d = points3d_to_2d(points3d, center, rot_mat, proj_mat)
        error = compute_bbox_error(points2d, bbox_xyxy)
        if error < best_error:
            best_center = center
            best_error = error

    return best_center


@dataclass
class BoxFitResult:
    yaw_cam_rad: float
    yaw_blender_rad: float
    theta_ray_rad: float
    alpha_local_rad: float
    center_cam_m: List[float]
    bottom_center_cam_m: List[float]
    bottom_center_blender_m: List[float]
    dims_hwl_m: List[float]
    corners_2d: List[List[float]]
    orientation_face_2d: Optional[List[List[float]]]
    bbox_projected: List[int]
    orientation_anchor_2d: List[float]
    orientation_tip_2d: List[float]
    reprojection_error: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "yaw_cam_rad": round(self.yaw_cam_rad, 5),
            "yaw_blender_rad": round(self.yaw_blender_rad, 5),
            "theta_ray_rad": round(self.theta_ray_rad, 5),
            "alpha_local_rad": round(self.alpha_local_rad, 5),
            "center_cam_m": [round(v, 4) for v in self.center_cam_m],
            "bottom_center_cam_m": [round(v, 4) for v in self.bottom_center_cam_m],
            "bottom_center_blender_m": [round(v, 4) for v in self.bottom_center_blender_m],
            "dims_hwl_m": [round(v, 4) for v in self.dims_hwl_m],
            "corners_2d": [[round(v, 1) for v in pt] for pt in self.corners_2d],
            "orientation_face_2d": (
                [[round(v, 1) for v in pt] for pt in self.orientation_face_2d]
                if self.orientation_face_2d is not None
                else None
            ),
            "bbox_projected": [int(v) for v in self.bbox_projected],
            "orientation_anchor_2d": [round(v, 1) for v in self.orientation_anchor_2d],
            "orientation_tip_2d": [round(v, 1) for v in self.orientation_tip_2d],
            "reprojection_error": round(self.reprojection_error, 4),
        }


def camera_to_blender_bottom_center(
    bottom_center_cam_m: Sequence[float],
    cam_height_m: float,
) -> List[float]:
    x_cam, y_cam, z_cam = [float(v) for v in bottom_center_cam_m[:3]]
    return [
        round(z_cam, 4),
        round(-x_cam, 4),
        round(max(0.0, float(cam_height_m) - y_cam), 4),
    ]


def yaw_cam_to_blender(yaw_cam_rad: float) -> float:
    return float(-yaw_cam_rad)


def project_camera_point(
    point_cam_xyz: Sequence[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> List[float]:
    x_cam, y_cam, z_cam = [float(v) for v in point_cam_xyz[:3]]
    z_safe = max(z_cam, 1e-6)
    return [
        float(fx * x_cam / z_safe + cx),
        float(fy * y_cam / z_safe + cy),
    ]


def compute_theta_ray(
    bbox_xyxy: Sequence[float],
    fx: float,
    cx: float,
) -> float:
    x1, _, x2, _ = [float(v) for v in bbox_xyxy[:4]]
    u = 0.5 * (x1 + x2)
    return float(math.atan2(u - float(cx), float(fx)))


def _refine_yaw_candidates(yaw_center: float) -> np.ndarray:
    local = np.linspace(yaw_center - math.radians(12.0), yaw_center + math.radians(12.0), 25)
    return np.asarray([wrap_angle(v) for v in local], dtype=np.float64)


def _refine_alpha_candidates(alpha_center: float) -> np.ndarray:
    local = np.linspace(alpha_center - math.radians(12.0), alpha_center + math.radians(12.0), 25)
    return np.asarray([wrap_angle(v) for v in local], dtype=np.float64)


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def fit_box(
    bbox_xyxy: Sequence[int],
    dims_hwl_m: Sequence[float],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_height_m: float,
    depth_prior_m: Optional[float] = None,
    yaw_hint_cam_rad: Optional[float] = None,
    coarse_yaw_samples: int = 181,
) -> Optional[BoxFitResult]:
    proj_mat = build_projection_matrix(fx, fy, cx, cy)
    points3d = init_points3d(dims_hwl_m)
    bbox = [float(v) for v in bbox_xyxy[:4]]
    bbox_w = max(1.0, bbox[2] - bbox[0])
    bbox_h = max(1.0, bbox[3] - bbox[1])
    theta_ray = compute_theta_ray(bbox, fx, cx)

    coarse_alphas = np.linspace(-math.pi, math.pi, int(max(24, coarse_yaw_samples)), endpoint=False, dtype=np.float64)
    alpha_hint_local_rad = None
    if yaw_hint_cam_rad is not None:
        alpha_hint_local_rad = wrap_angle(float(yaw_hint_cam_rad) - theta_ray)
        coarse_alphas = np.unique(
            np.concatenate([coarse_alphas, _refine_alpha_candidates(float(alpha_hint_local_rad))])
        )

    best: Optional[Tuple[float, float, np.ndarray, np.ndarray, float]] = None
    best_score = float("inf")

    for alpha_local in coarse_alphas:
        yaw_cam = wrap_angle(float(alpha_local) + theta_ray)
        rot_mat = rotation_y(float(yaw_cam))
        center = compute_center(points3d, rot_mat, proj_mat, bbox)
        if center is None:
            continue

        center_xyz = center.reshape(3)
        if not np.isfinite(center_xyz).all():
            continue
        if center_xyz[2] <= 0.4 or center_xyz[2] > 180.0:
            continue

        points2d = points3d_to_2d(points3d, center, rot_mat, proj_mat)
        bbox_error = compute_bbox_error(points2d, bbox)

        depth_penalty = 0.0
        if depth_prior_m is not None and math.isfinite(depth_prior_m) and depth_prior_m > 0.1:
            depth_penalty = 0.35 * (abs(center_xyz[2] - float(depth_prior_m)) / max(float(depth_prior_m), 1.0)) * (bbox_w + bbox_h)

        shape_penalty = 0.04 * abs(center_xyz[0]) + 0.02 * abs(center_xyz[1])
        alpha_penalty = 0.0
        if alpha_hint_local_rad is not None:
            alpha_penalty = 0.06 * abs(wrap_angle(float(alpha_local) - float(alpha_hint_local_rad))) * (bbox_w + 0.5 * bbox_h)
        score = bbox_error + depth_penalty + shape_penalty + alpha_penalty

        if score < best_score:
            best_score = score
            best = (float(alpha_local), float(yaw_cam), center, points2d, float(bbox_error))

    if best is None:
        return None

    coarse_alpha, _, _, _, _ = best
    fine_alphas = _refine_alpha_candidates(coarse_alpha)
    for alpha_local in fine_alphas:
        yaw_cam = wrap_angle(float(alpha_local) + theta_ray)
        rot_mat = rotation_y(float(yaw_cam))
        center = compute_center(points3d, rot_mat, proj_mat, bbox)
        if center is None:
            continue
        center_xyz = center.reshape(3)
        if not np.isfinite(center_xyz).all():
            continue
        if center_xyz[2] <= 0.4 or center_xyz[2] > 180.0:
            continue

        points2d = points3d_to_2d(points3d, center, rot_mat, proj_mat)
        bbox_error = compute_bbox_error(points2d, bbox)
        depth_penalty = 0.0
        if depth_prior_m is not None and math.isfinite(depth_prior_m) and depth_prior_m > 0.1:
            depth_penalty = 0.35 * (abs(center_xyz[2] - float(depth_prior_m)) / max(float(depth_prior_m), 1.0)) * (bbox_w + bbox_h)
        shape_penalty = 0.04 * abs(center_xyz[0]) + 0.02 * abs(center_xyz[1])
        alpha_penalty = 0.0
        if alpha_hint_local_rad is not None:
            alpha_penalty = 0.06 * abs(wrap_angle(float(alpha_local) - float(alpha_hint_local_rad))) * (bbox_w + 0.5 * bbox_h)
        score = bbox_error + depth_penalty + shape_penalty + alpha_penalty
        if score < best_score:
            best_score = score
            best = (float(alpha_local), float(yaw_cam), center, points2d, float(bbox_error))

    if best is None:
        return None

    alpha_local, yaw_cam, center, points2d, bbox_error = best
    center_xyz = center.reshape(3)
    dims = [float(v) for v in dims_hwl_m[:3]]
    bottom_center_cam = [
        float(center_xyz[0]),
        float(center_xyz[1] + 0.5 * dims[0]),
        float(center_xyz[2]),
    ]
    bottom_center_blender = camera_to_blender_bottom_center(bottom_center_cam, cam_height_m)
    yaw_blender = yaw_cam_to_blender(yaw_cam)

    proj_bbox = [
        int(np.min(points2d[:, 0])),
        int(np.min(points2d[:, 1])),
        int(np.max(points2d[:, 0])),
        int(np.max(points2d[:, 1])),
    ]
    rot_mat = rotation_y(float(yaw_cam))
    center_2d = project_camera_point(center_xyz, fx, fy, cx, cy)
    front_offset = rot_mat @ np.asarray([[0.0], [0.0], [0.55 * dims[2]]], dtype=np.float64)
    front_center_cam = center.reshape(3, 1) + front_offset
    front_center_2d = project_camera_point(front_center_cam.reshape(3), fx, fy, cx, cy)
    orientation_face_2d = [points2d[idx].tolist() for idx in (0, 1, 7, 6)]

    return BoxFitResult(
        yaw_cam_rad=float(wrap_angle(yaw_cam)),
        yaw_blender_rad=float(wrap_angle(yaw_blender)),
        theta_ray_rad=float(theta_ray),
        alpha_local_rad=float(wrap_angle(alpha_local)),
        center_cam_m=[float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])],
        bottom_center_cam_m=bottom_center_cam,
        bottom_center_blender_m=bottom_center_blender,
        dims_hwl_m=dims,
        corners_2d=[[float(pt[0]), float(pt[1])] for pt in points2d.tolist()],
        orientation_face_2d=[[float(pt[0]), float(pt[1])] for pt in orientation_face_2d],
        bbox_projected=proj_bbox,
        orientation_anchor_2d=center_2d,
        orientation_tip_2d=front_center_2d,
        reprojection_error=float(bbox_error),
    )
