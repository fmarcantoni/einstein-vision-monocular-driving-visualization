from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchvision.models import vgg

from deepbox_geometry import (
    camera_to_blender_bottom_center,
    project_camera_point,
    yaw_cam_to_blender,
)


SKHADEM_REPO_DIR = (Path(__file__).parent / "external" / "3D-BoundingBox").resolve()
_AUTO_WEIGHT_CANDIDATES = ("epoch_50.pk", "epoch_50.pkl", "epoch_10.pk", "epoch_10.pkl")
_CLASS_MAP = {
    "car": "car",
    "sedan": "car",
    "hatchback": "car",
    "suv": "car",
    "pickup": "car",
    "truck": "truck",
    "bicycle": "cyclist",
}


def _resolve_device(device: str) -> torch.device:
    requested = str(device or "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _resolve_weights(repo_dir: Path, requested: Optional[str]) -> Optional[Path]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser().resolve()
        return path if path.exists() else None
    candidate_dirs = [
        Path("weights").resolve(),
        (repo_dir / "weights").resolve(),
        repo_dir.resolve(),
    ]
    for weights_dir in candidate_dirs:
        for candidate in _AUTO_WEIGHT_CANDIDATES:
            path = weights_dir / candidate
            if path.exists():
                return path
    for weights_dir in candidate_dirs:
        matches = list(weights_dir.glob("epoch_*.pk")) + list(weights_dir.glob("epoch_*.pkl"))
        for path in sorted(matches, reverse=True):
            if path.exists():
                return path.resolve()
    return None


def _generate_angle_bins(bins: int) -> np.ndarray:
    angle_bins = np.zeros(bins, dtype=np.float32)
    interval = 2.0 * np.pi / float(bins)
    for idx in range(1, bins):
        angle_bins[idx] = idx * interval
    angle_bins += interval / 2.0
    return angle_bins


class SkhademDeepBoxAdapter:
    def __init__(
        self,
        repo_dir: Optional[str] = None,
        weights_path: Optional[str] = "auto",
        device: str = "auto",
        bins: int = 2,
    ) -> None:
        self.repo_dir = Path(repo_dir).expanduser().resolve() if repo_dir else SKHADEM_REPO_DIR
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"3D-BoundingBox repo not found: {self.repo_dir}")

        self.weights_path = _resolve_weights(self.repo_dir, weights_path)
        if self.weights_path is None:
            raise FileNotFoundError(
                "No skhadem Deep3DBox weights found. "
                "Expected weights/epoch_50.pk(.l), external/3D-BoundingBox/weights/epoch_50.pk(.l), or --skhadem-weights."
            )

        repo_parent = str(self.repo_dir)
        if repo_parent not in sys.path:
            sys.path.insert(0, repo_parent)

        from library.Math import calc_location, create_corners, rotation_matrix
        from torch_lib.ClassAverages import ClassAverages
        from torch_lib.Dataset import DetectedObject
        from torch_lib.Model import Model

        self._calc_location = calc_location
        self._create_corners = create_corners
        self._rotation_matrix = rotation_matrix
        self._detected_object_cls = DetectedObject
        self._class_averages = ClassAverages()
        self._bins = int(max(1, bins))
        self._angle_bins = _generate_angle_bins(self._bins)
        self.device = _resolve_device(device)

        try:
            backbone = vgg.vgg19_bn(weights=None)
        except TypeError:
            backbone = vgg.vgg19_bn(pretrained=False)
        self.model = Model(features=backbone.features, bins=self._bins).to(self.device)

        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def supports_class(self, cls_name: str) -> bool:
        mapped = _CLASS_MAP.get(str(cls_name or "").lower())
        return bool(mapped and self._class_averages.recognized_class(mapped))

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
    ) -> Optional[Dict[str, Any]]:
        mapped_class = _CLASS_MAP.get(str(cls_name or "").lower())
        if not mapped_class or not self._class_averages.recognized_class(mapped_class):
            return None

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy[:4]]
        if x2 <= x1 or y2 <= y1:
            return None

        proj_matrix = np.asarray(
            [
                [float(fx), 0.0, float(cx), 0.0],
                [0.0, float(fy), float(cy), 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        box_2d = [(x1, y1), (x2, y2)]

        detected = self._detected_object_cls(frame_bgr, mapped_class, box_2d, proj_matrix)
        input_tensor = detected.img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            orient, conf, dim = self.model(input_tensor)

        orient_np = orient.detach().cpu().numpy()[0, :, :]
        conf_np = conf.detach().cpu().numpy()[0, :]
        dim_np = dim.detach().cpu().numpy()[0, :]
        dim_np = dim_np + self._class_averages.get_item(mapped_class)

        argmax = int(np.argmax(conf_np))
        orient_vec = orient_np[argmax, :]
        alpha = float(np.arctan2(orient_vec[1], orient_vec[0]))
        alpha += float(self._angle_bins[argmax])
        alpha -= float(np.pi)
        theta_ray = float(detected.theta_ray)

        location, _ = self._calc_location(dim_np, proj_matrix, box_2d, alpha, theta_ray)
        location = np.asarray(location, dtype=np.float64).reshape(3)
        if not np.isfinite(location).all() or float(location[2]) <= 0.1:
            return None

        yaw_cam = float(alpha + theta_ray)
        rot_mat = self._rotation_matrix(yaw_cam)
        corners_cam = np.asarray(
            self._create_corners(dim_np, location=location, R=rot_mat),
            dtype=np.float64,
        )
        corners_2d = [
            project_camera_point(point, fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
            for point in corners_cam
        ]
        bbox_projected = [
            int(round(min(pt[0] for pt in corners_2d))),
            int(round(min(pt[1] for pt in corners_2d))),
            int(round(max(pt[0] for pt in corners_2d))),
            int(round(max(pt[1] for pt in corners_2d))),
        ]

        h, w, l = [float(v) for v in dim_np.tolist()]
        bottom_center_cam = [float(location[0]), float(location[1] + h / 2.0), float(location[2])]
        bottom_center_blender = camera_to_blender_bottom_center(bottom_center_cam, cam_height_m)

        front_offset = rot_mat @ np.asarray([l / 2.0, 0.0, 0.0], dtype=np.float64)
        front_bottom_cam = [
            float(location[0] + front_offset[0]),
            float(location[1] + h / 2.0),
            float(location[2] + front_offset[2]),
        ]
        orientation_anchor_2d = project_camera_point(bottom_center_cam, float(fx), float(fy), float(cx), float(cy))
        orientation_tip_2d = project_camera_point(front_bottom_cam, float(fx), float(fy), float(cx), float(cy))

        reproj_error = float(np.sum(np.abs(np.asarray(bbox_projected, dtype=np.float64) - np.asarray([x1, y1, x2, y2], dtype=np.float64))))

        return {
            "yaw_cam_rad": float(yaw_cam),
            "yaw_blender_rad": float(yaw_cam_to_blender(yaw_cam)),
            "theta_ray_rad": float(theta_ray),
            "alpha_local_rad": float(alpha),
            "center_cam_m": [float(location[0]), float(location[1]), float(location[2])],
            "bottom_center_cam_m": bottom_center_cam,
            "bottom_center_blender_m": bottom_center_blender,
            "dims_hwl_m": [h, w, l],
            "corners_2d": [[float(pt[0]), float(pt[1])] for pt in corners_2d],
            "orientation_face_2d": [[float(corners_2d[idx][0]), float(corners_2d[idx][1])] for idx in (0, 1, 2, 3)],
            "bbox_projected": bbox_projected,
            "orientation_anchor_2d": [float(v) for v in orientation_anchor_2d],
            "orientation_tip_2d": [float(v) for v in orientation_tip_2d],
            "reprojection_error": reproj_error,
            "backend": "skhadem_3d_boundingbox",
            "repo_class": mapped_class,
        }
