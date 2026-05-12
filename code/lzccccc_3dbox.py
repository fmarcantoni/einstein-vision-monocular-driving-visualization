from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from deepbox_geometry import camera_to_blender_bottom_center, project_camera_point, yaw_cam_to_blender


LZ_REPO_DIR = (Path(__file__).parent / "external" / "lzccccc-3d-bounding-box").resolve()
_AUTO_WEIGHT_CANDIDATES = ("3dbox_weights_mob.hdf5", "3dbox_weights_vgg.hdf5")
_CLASS_MAP = {
    "car": "Car",
    "sedan": "Car",
    "hatchback": "Car",
    "suv": "Car",
    "pickup": "Car",
    "truck": "Car",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}
_CLASS_AVERAGES = {
    "Car": np.asarray([1.526083, 1.62859, 3.883954], dtype=np.float32),
    "Cyclist": np.asarray([1.737203, 0.596773, 1.763546], dtype=np.float32),
    "Pedestrian": np.asarray([1.760706, 0.660189, 0.842284], dtype=np.float32),
}


def _resolve_weights(repo_dir: Path, requested: Optional[str]) -> Optional[Path]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser().resolve()
        return path if path.exists() else None
    for candidate in _AUTO_WEIGHT_CANDIDATES:
        path = repo_dir / candidate
        if path.exists():
            return path
    for path in repo_dir.rglob("*.hdf5"):
        return path.resolve()
    for path in repo_dir.rglob("*.h5"):
        return path.resolve()
    return None


class Lzccccc3DBoxAdapter:
    """
    TensorFlow/Keras adapter around lzccccc/3d-bounding-box-estimation-for-autonomous-driving.

    This wraps the repo's prediction ingredients directly:
    - network() from model/{mobilenet_v2,vgg16}.py
    - recover_angle()
    - compute_orientaion()
    - translation_constraints()
    """

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        weights_path: Optional[str] = "auto",
        network: str = "mobilenet_v2",
    ) -> None:
        self.repo_dir = Path(repo_dir).expanduser().resolve() if repo_dir else LZ_REPO_DIR
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"lzccccc repo not found: {self.repo_dir}")

        self.weights_path = _resolve_weights(self.repo_dir, weights_path)
        if self.weights_path is None:
            raise FileNotFoundError(
                "No lzccccc weights found. Expected 3dbox_weights_mob.hdf5 / 3dbox_weights_vgg.hdf5 "
                "or pass --lzccccc-weights."
            )

        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        try:
            import tensorflow as tf  # noqa: F401
        except Exception as exc:
            raise ImportError("TensorFlow is required for the lzccccc backend.") from exc

        from utils.correspondece_constraint import (
            compute_orientaion,
            detectionInfo,
            recover_angle,
            translation_constraints,
        )

        self._compute_orientaion = compute_orientaion
        self._detectionInfo = detectionInfo
        self._recover_angle = recover_angle
        self._translation_constraints = translation_constraints
        self.network_name = str(network or "mobilenet_v2").lower()
        if self.network_name == "vgg16":
            from model import vgg16 as nn
        else:
            from model import mobilenet_v2 as nn
            self.network_name = "mobilenet_v2"

        self.model = nn.network()
        self.model.load_weights(str(self.weights_path))
        self.bin_num = 2
        self.norm_h = 224
        self.norm_w = 224

    def supports_class(self, cls_name: str) -> bool:
        return str(cls_name or "").lower() in _CLASS_MAP

    def predict(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: Sequence[int],
        cls_name: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        cam_height_m: float,
        dims_override_hwl_m: Optional[Sequence[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        mapped_class = _CLASS_MAP.get(str(cls_name or "").lower())
        if mapped_class is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
        if x2 <= x1 or y2 <= y1:
            return None

        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (self.norm_h, self.norm_w))
        patch = np.asarray(patch, dtype=np.float32)
        patch -= np.asarray([[[103.939, 116.779, 123.68]]], dtype=np.float32)
        patch = np.expand_dims(patch, 0)

        pred = self.model.predict(patch, verbose=0)
        dim_delta = np.asarray(pred[0][0], dtype=np.float32)
        bin_anchor = np.asarray(pred[1][0], dtype=np.float32)
        bin_confidence = np.asarray(pred[2][0], dtype=np.float32)

        if dims_override_hwl_m is not None:
            dims = np.asarray(dims_override_hwl_m, dtype=np.float32)
        else:
            dims = _CLASS_AVERAGES[mapped_class] + dim_delta
        dims = np.asarray(dims, dtype=np.float32)

        line = [
            mapped_class,
            "0", "0", "0",
            str(float(x1)), str(float(y1)), str(float(x2)), str(float(y2)),
            str(float(dims[0])), str(float(dims[1])), str(float(dims[2])),
            "0", "0", "1", "0",
        ]
        obj = self._detectionInfo(line)
        obj.h, obj.w, obj.l = [float(v) for v in dims.tolist()]
        obj.alpha = float(self._recover_angle(bin_anchor, bin_confidence, self.bin_num))

        P2 = np.asarray(
            [
                [float(fx), 0.0, float(cx), 0.0],
                [0.0, float(fy), float(cy), 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        yaw_cam, rot_local = self._compute_orientaion(P2, obj)
        obj.rot_global = float(yaw_cam)
        obj.tx, obj.ty, obj.tz = self._translation_constraints(P2, obj, rot_local)

        R = np.array(
            [
                [np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                [0, 1, 0],
                [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)],
            ],
            dtype=np.float32,
        )
        x_corners = np.asarray([0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0], dtype=np.float32) - obj.l / 2.0
        y_corners = np.asarray([0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h], dtype=np.float32) - obj.h
        z_corners = np.asarray([0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0], dtype=np.float32) - obj.w / 2.0
        corners_cam = R.dot(np.vstack([x_corners, y_corners, z_corners]))
        corners_cam += np.asarray([obj.tx, obj.ty, obj.tz], dtype=np.float32).reshape((3, 1))

        corners_h = np.vstack([corners_cam, np.ones((corners_cam.shape[-1]), dtype=np.float32)])
        corners_2d = P2.dot(corners_h)
        corners_2d = corners_2d / np.maximum(corners_2d[2:3], 1e-6)
        corners_2d = corners_2d[:2].T

        bbox_projected = [
            int(round(float(np.min(corners_2d[:, 0])))),
            int(round(float(np.min(corners_2d[:, 1])))),
            int(round(float(np.max(corners_2d[:, 0])))),
            int(round(float(np.max(corners_2d[:, 1])))),
        ]
        reproj_error = float(np.sum(np.abs(np.asarray(bbox_projected, dtype=np.float32) - np.asarray([x1, y1, x2, y2], dtype=np.float32))))

        bottom_center_cam = [float(obj.tx), float(obj.ty), float(obj.tz)]
        center_cam = [float(obj.tx), float(obj.ty - obj.h / 2.0), float(obj.tz)]
        bottom_center_blender = camera_to_blender_bottom_center(bottom_center_cam, cam_height_m)
        front_bottom_cam = [
            float(obj.tx + math.cos(obj.rot_global) * obj.l / 2.0),
            float(obj.ty),
            float(obj.tz - math.sin(obj.rot_global) * obj.l / 2.0),
        ]
        orientation_anchor_2d = project_camera_point(bottom_center_cam, float(fx), float(fy), float(cx), float(cy))
        orientation_tip_2d = project_camera_point(front_bottom_cam, float(fx), float(fy), float(cx), float(cy))

        return {
            "yaw_cam_rad": float(obj.rot_global),
            "yaw_blender_rad": float(yaw_cam_to_blender(float(obj.rot_global))),
            "theta_ray_rad": float(obj.rot_global - obj.alpha),
            "alpha_local_rad": float(obj.alpha),
            "center_cam_m": center_cam,
            "bottom_center_cam_m": bottom_center_cam,
            "bottom_center_blender_m": bottom_center_blender,
            "dims_hwl_m": [float(obj.h), float(obj.w), float(obj.l)],
            "corners_2d": [[float(pt[0]), float(pt[1])] for pt in corners_2d.tolist()],
            "orientation_face_2d": [[float(corners_2d[idx][0]), float(corners_2d[idx][1])] for idx in (0, 1, 2, 3)],
            "bbox_projected": bbox_projected,
            "orientation_anchor_2d": [float(v) for v in orientation_anchor_2d],
            "orientation_tip_2d": [float(v) for v in orientation_tip_2d],
            "reprojection_error": reproj_error,
            "backend": "lzccccc_3d_bounding_box",
            "repo_class": mapped_class,
            "network": self.network_name,
            "dims_source": "override" if dims_override_hwl_m is not None else "predicted",
        }


import cv2  # placed last to keep import errors localized when TF is missing
