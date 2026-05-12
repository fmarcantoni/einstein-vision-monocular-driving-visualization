"""
vehicle_3d_detection.py
=======================

Monocular 3D vehicle detection stage for the autonomous-driving pipeline.

Responsibilities
----------------
* estimate 3D bounding boxes, orientation, and dimensions for each vehicle
* produce projected 3D bbox overlays for visual debugging
* output JSON consumable by scene_assembler.py and blender.py

Approach
--------
This module implements a Deep3DBox-style geometry fitting pipeline:

1.  Run Ultralytics YOLO to get 2D vehicle detections.
2.  Load or estimate monocular depth for each detection.
3.  Use the geometry constraints from "3D Bounding Box Estimation Using
    Deep Learning and Geometry" (Mousavian et al., CVPR 2017 /
    arXiv:1612.00496) to fit a 3D box and solve for vehicle
    center/orientation from the 2D box, working in local orientation
    alpha plus theta_ray before converting back to global yaw.
4.  Fall back to monocular depth lifting only if the geometric fit fails.

For the 2D proposals feeding the box fit, `auto` always targets YOLO26. If no
local YOLO26 checkpoint is present, the official Ultralytics model name is
used directly so the detector does not silently fall back to YOLO11.

Output schema
-------------
::

    {
      "source": "...",
      "frames_written": N,
      "frames": [
        {
          "frame_idx": 0,
          "timestamp_s": 0.0,
          "vehicles_3d": [
            {
              "id": 0,
              "class": "car",
              "subclass": "car",
              "bbox_2d": [x1, y1, x2, y2],
              "confidence": 0.92,
              "depth_m": 15.3,
              "position_3d": [x, y, z],
              "dimensions_3d": [h, w, l],
              "orientation_rad": 0.12,
              "bbox_3d_corners": [[x,y], ...],  # 8 projected corners
            }
          ]
        }
      ]
    }
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import math
import time
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np

from cersar_3d_detection import Cersar3DDetectionAdapter
from deepbox_geometry import fit_box
from lzccccc_3dbox import Lzccccc3DBoxAdapter
from project_setup import (
    infer_scene_name,
    mirror_stage_output,
    resolve_existing_artifact,
    scene_output_layout,
)
from skhadem_deepbox import SkhademDeepBoxAdapter
from vehicle_subclassification import VehicleSubclassifier


# ─────────────────────────────────────────────────────────────────────────────
# Constants — prior vehicle dimensions (h, w, l) in metres
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_DIMS: Dict[str, Tuple[float, float, float]] = {
    "car":        (1.50, 1.85, 4.50),
    "sedan":      (1.47, 1.82, 4.72),
    "hatchback":  (1.50, 1.78, 4.18),
    "suv":        (1.72, 1.94, 4.88),
    "pickup":     (1.82, 2.02, 5.35),
    "truck":      (3.40, 2.50, 8.00),
    "motorcycle": (1.20, 0.82, 2.20),
    "bicycle":    (1.10, 0.50, 1.80),
}

_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "truck"}
_MODEL_LABEL_ALIASES = {
    "bicycle": "bicycle",
    "bike": "bicycle",
    "car": "car",
    "sedan": "car",
    "suv": "car",
    "hatchback": "car",
    "pickup": "car",
    "pickup_truck": "car",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "lorry": "truck",
    "bus": "truck",
    "van": "truck",
}

# Default camera intrinsics fallback (Tesla Model S front, 1280x720)
_DEFAULT_FX = 910.0
_DEFAULT_FY = 910.0
_DEFAULT_CX = 640.0
_DEFAULT_CY = 360.0
_DEFAULT_CAM_HEIGHT = 1.5  # metres
_DEFAULT_YOLO_IMGSZ = 960
_DEFAULT_COARSE_YAW_SAMPLES = 61
_MIN_GEOM_BBOX_W = 20
_MIN_GEOM_BBOX_H = 20
_DEFAULT_DEEPBOX_BACKEND = "skhadem"
_DEFAULT_VEHICLE_NMS_IOU = 0.55
_DEFAULT_TRACK_MATCH_IOU = 0.25
_DEFAULT_TEMPORAL_ALPHA = 0.72
_DEFAULT_MAX_VEHICLES = 24
_MIN_CANDIDATE_BBOX_W = 18
_MIN_CANDIDATE_BBOX_H = 18
_DISABLED_MODEL_STRINGS = {"none", "off", "disable", "disabled"}
_BACKEND_SELECTION_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "auto": (
        "skhadem_3d_boundingbox",
        "lzccccc_3d_bounding_box",
        "cersar_3d_detection_repo",
        "deep3dbox_geometry_alpha_search",
        "fallback_depth_smoothed",
        "fallback_depth",
    ),
    "skhadem": (
        "skhadem_3d_boundingbox",
        "deep3dbox_geometry_alpha_search",
        "lzccccc_3d_bounding_box",
        "cersar_3d_detection_repo",
        "fallback_depth_smoothed",
        "fallback_depth",
    ),
    "lzccccc": (
        "lzccccc_3d_bounding_box",
        "deep3dbox_geometry_alpha_search",
        "skhadem_3d_boundingbox",
        "cersar_3d_detection_repo",
        "fallback_depth_smoothed",
        "fallback_depth",
    ),
    "cersar": (
        "cersar_3d_detection_repo",
        "deep3dbox_geometry_alpha_search",
        "skhadem_3d_boundingbox",
        "lzccccc_3d_bounding_box",
        "fallback_depth_smoothed",
        "fallback_depth",
    ),
    "geometry": (
        "deep3dbox_geometry_alpha_search",
        "skhadem_3d_boundingbox",
        "lzccccc_3d_bounding_box",
        "cersar_3d_detection_repo",
        "fallback_depth_smoothed",
        "fallback_depth",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Device resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(requested: str = "auto") -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if requested in ("mps", "auto"):
        mps_ok = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
        if mps_ok:
            return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle subclassification (reuse from object_detection)
# ─────────────────────────────────────────────────────────────────────────────

def _subclassify_vehicle(cls_name: str, bbox: List[int]) -> str:
    return cls_name


def _resolve_model_path(requested: str) -> str:
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        return str(path.resolve()) if path.exists() else requested
    resolved = resolve_existing_artifact(("yolo26x.pt", "yolo26l.pt", "yolo26m.pt", "yolo26s.pt", "yolo26n.pt"))
    if resolved:
        return resolved
    return "yolo26x.pt"


def _default_intrinsics_native_size(cx: float, cy: float) -> Optional[Tuple[float, float]]:
    if abs(float(cx) - _DEFAULT_CX) < 1e-3 and abs(float(cy) - _DEFAULT_CY) < 1e-3:
        return 2.0 * float(_DEFAULT_CX), 2.0 * float(_DEFAULT_CY)
    return None


def _canonicalize_model_label(raw_label: str) -> Optional[str]:
    norm = str(raw_label or "").strip().lower().replace(" ", "_")
    return _MODEL_LABEL_ALIASES.get(norm)


def _infer_camera_view(requested_view: Optional[str], video_path: Optional[str]) -> str:
    requested = str(requested_view or "auto").strip().lower()
    if requested in {"front", "back", "left", "right"}:
        return requested

    name = str(video_path or "").lower()
    if "front" in name:
        return "front"
    if "back" in name or "rear" in name:
        return "back"
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    return "front"


def _vehicle_from_detection_entry(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cls_name = str(raw.get("class") or raw.get("cls") or "").strip()
    if cls_name not in _VEHICLE_CLASSES:
        return None
    bbox = raw.get("bbox") or raw.get("bbox_2d")
    if not bbox or len(bbox) < 4:
        return None
    try:
        bbox_xyxy = [int(round(float(v))) for v in bbox[:4]]
        confidence = float(raw.get("confidence", 1.0))
    except (TypeError, ValueError):
        return None
    subclass = str(raw.get("subclass") or cls_name)
    return {
        "class": cls_name,
        "subclass": subclass,
        "bbox": bbox_xyxy,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3D geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_center_depth(
    bbox: List[int],
    depth_map: Optional[np.ndarray],
) -> Optional[float]:
    """Sample median depth inside the bbox from a depth map."""
    if depth_map is None:
        return None
    x1, y1, x2, y2 = bbox
    H, W = depth_map.shape[:2]
    # Shrink bbox to centre 60% for more reliable depth
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hw, hh = max(1, (x2 - x1) * 3 // 10), max(1, (y2 - y1) * 3 // 10)
    rx1 = max(0, cx - hw)
    ry1 = max(0, cy - hh)
    rx2 = min(W, cx + hw)
    ry2 = min(H, cy + hh)
    patch = depth_map[ry1:ry2, rx1:rx2]
    valid = patch[(patch > 0.1) & np.isfinite(patch)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _bbox_area(bbox: List[int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_iou(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
    if inter <= 0.0:
        return 0.0
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return inter / max(union, 1e-6)


def _bbox_overlap_ratio(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = float(max(0, ix2 - ix1) * max(0, iy2 - iy1))
    if inter <= 0.0:
        return 0.0
    return inter / max(min(_bbox_area(box_a), _bbox_area(box_b)), 1e-6)


def _bbox_is_reasonable(
    bbox: List[int],
    *,
    min_w: int = _MIN_CANDIDATE_BBOX_W,
    min_h: int = _MIN_CANDIDATE_BBOX_H,
) -> bool:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= min_w and (y2 - y1) >= min_h


def _normalize_bbox_xyxy(bbox: List[Any], width: int, height: int) -> Optional[List[int]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _nms_vehicle_candidates(
    candidates: List[Dict[str, Any]],
    *,
    iou_threshold: float,
    max_keep: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(
        (
            item for item in candidates
            if _bbox_is_reasonable([int(v) for v in item.get("bbox", [])[:4]])
        ),
        key=lambda item: (float(item.get("confidence", 0.0)), _bbox_area([int(v) for v in item["bbox"][:4]])),
        reverse=True,
    )
    kept: List[Dict[str, Any]] = []
    for item in ordered:
        bbox = [int(v) for v in item["bbox"][:4]]
        duplicate = False
        for prev in kept:
            prev_bbox = [int(v) for v in prev["bbox"][:4]]
            if _bbox_iou(bbox, prev_bbox) >= iou_threshold:
                duplicate = True
                break
            if _bbox_overlap_ratio(bbox, prev_bbox) >= 0.88:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(item)
        if len(kept) >= max_keep:
            break
    return kept


def _estimate_depth_from_bbox(
    bbox: List[int],
    cls_name: str,
    subclass: str,
    fy: float,
) -> float:
    """Fallback: estimate depth from known object height and bbox height."""
    x1, y1, x2, y2 = bbox
    bbox_h = max(y2 - y1, 1)
    known_h = VEHICLE_DIMS.get(subclass, VEHICLE_DIMS.get(cls_name, (1.5, 1.8, 4.5)))[0]
    return (known_h * fy) / bbox_h


def _back_project_bbox_center(
    bbox: List[int],
    depth_m: float,
    fx: float, fy: float, cx: float, cy: float,
    cam_height: float,
) -> List[float]:
    """Back-project the bottom-centre of the bbox to 3D camera coordinates."""
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0
    v = y2  # bottom of bbox ~ ground contact point

    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m

    # Convert to vehicle-centric: forward=+X_blender, left=+Y_blender, up=+Z_blender
    bx = Z               # depth → forward
    by = -X              # lateral
    bz = cam_height - Y  # height (ground = 0)

    return [round(bx, 3), round(by, 3), round(max(0.0, bz), 3)]


def _estimate_yaw(
    bbox: List[int],
    position_3d: List[float],
    frame_w: int,
) -> float:
    """
    Estimate vehicle yaw angle from bbox position in the image.

    Vehicles in the centre of the frame face roughly forward (yaw ~ 0).
    Vehicles to the left/right and with asymmetric bboxes are angled.
    """
    x1, y1, x2, y2 = bbox
    cx_bbox = (x1 + x2) / 2.0
    cx_img = frame_w / 2.0

    # Lateral offset normalised to [-1, 1]
    lateral_norm = (cx_bbox - cx_img) / cx_img

    # Bbox aspect asymmetry: wider bboxes for angled cars
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    aspect = w / h

    # Base yaw from lateral position
    yaw = -lateral_norm * 0.35

    # Adjust for aspect ratio — side views are wider relative to height
    if aspect > 1.8:
        yaw += math.copysign(0.4, lateral_norm)
    elif aspect > 1.3:
        yaw += math.copysign(0.2, lateral_norm)

    return float(np.clip(yaw, -math.pi, math.pi))


def _reprojection_error_is_reasonable(
    reprojection_error: Optional[float],
    bbox_xyxy: List[int],
) -> bool:
    if reprojection_error is None:
        return False
    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)
    max_err = max(42.0, 0.20 * float(bbox_w + bbox_h))
    return float(reprojection_error) <= max_err


def _projected_bbox_is_reasonable(
    projected_bbox: Optional[List[float]],
    bbox_xyxy: List[int],
) -> bool:
    if not projected_bbox or len(projected_bbox) < 4:
        return False
    try:
        proj = [int(round(float(v))) for v in projected_bbox[:4]]
    except (TypeError, ValueError):
        return False
    if not _bbox_is_reasonable(proj, min_w=8, min_h=8):
        return False

    iou = _bbox_iou(proj, bbox_xyxy)
    if iou >= 0.18:
        return True

    px1, py1, px2, py2 = proj
    x1, y1, x2, y2 = bbox_xyxy
    p_center = np.asarray([(px1 + px2) * 0.5, (py1 + py2) * 0.5], dtype=np.float32)
    b_center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    center_dist = float(np.linalg.norm(p_center - b_center))
    diag = math.hypot(max(x2 - x1, 1), max(y2 - y1, 1))
    area_ratio = _bbox_area(proj) / max(_bbox_area(bbox_xyxy), 1e-6)
    return center_dist <= 0.35 * diag and 0.35 <= area_ratio <= 2.8


def _projected_bbox_alignment_metrics(
    projected_bbox: Optional[List[float]],
    bbox_xyxy: List[int],
) -> Optional[Dict[str, float]]:
    if not projected_bbox or len(projected_bbox) < 4:
        return None
    try:
        proj = [int(round(float(v))) for v in projected_bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not _bbox_is_reasonable(proj, min_w=8, min_h=8):
        return None
    px1, py1, px2, py2 = proj
    x1, y1, x2, y2 = bbox_xyxy
    p_center = np.asarray([(px1 + px2) * 0.5, (py1 + py2) * 0.5], dtype=np.float32)
    b_center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    center_dist = float(np.linalg.norm(p_center - b_center))
    diag = math.hypot(max(x2 - x1, 1), max(y2 - y1, 1))
    area_ratio = _bbox_area(proj) / max(_bbox_area(bbox_xyxy), 1e-6)
    return {
        "iou": float(_bbox_iou(proj, bbox_xyxy)),
        "center_dist_norm": center_dist / max(diag, 1.0),
        "area_ratio": float(area_ratio),
    }


def _candidate_fit_is_extreme(
    candidate: Dict[str, Any],
    bbox_xyxy: List[int],
) -> bool:
    projected_bbox = candidate.get("bbox_projected") or candidate.get("bbox_2d_projected")
    metrics = _projected_bbox_alignment_metrics(projected_bbox, bbox_xyxy)
    if metrics is None:
        return True
    area_ratio = float(metrics["area_ratio"])
    center_norm = float(metrics["center_dist_norm"])
    reproj = float(candidate.get("reprojection_error") or 0.0)
    bbox_w = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
    bbox_h = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
    reproj_cap = max(140.0, 0.48 * float(bbox_w + bbox_h))
    if area_ratio < 0.12 or area_ratio > 6.5:
        return True
    if center_norm > 1.35:
        return True
    if reproj > reproj_cap and len(candidate.get("corners_2d") or []) == 8:
        return True
    return False


def _candidate_fit_score(
    candidate: Dict[str, Any],
    bbox_xyxy: List[int],
    depth_prior_m: Optional[float],
) -> float:
    bbox_w = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
    bbox_h = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
    diag = max(math.hypot(bbox_w, bbox_h), 1.0)
    projected_bbox = candidate.get("bbox_projected") or candidate.get("bbox_2d_projected")
    metrics = _projected_bbox_alignment_metrics(projected_bbox, bbox_xyxy) or {
        "iou": 0.0,
        "center_dist_norm": 1.5,
        "area_ratio": 0.0,
    }
    reproj = float(candidate.get("reprojection_error") or 0.0)
    reproj_norm = min(5.0, reproj / diag)
    iou = float(metrics["iou"])
    center_norm = float(metrics["center_dist_norm"])
    area_ratio = max(float(metrics["area_ratio"]), 1e-6)
    score = 2.10 * reproj_norm + 1.35 * (1.0 - iou) + 1.25 * center_norm + 0.55 * abs(math.log(area_ratio))

    if depth_prior_m is not None and math.isfinite(float(depth_prior_m)) and float(depth_prior_m) > 0.1:
        depth_val = None
        bottom_center_cam = candidate.get("bottom_center_cam_m")
        if isinstance(bottom_center_cam, list) and len(bottom_center_cam) >= 3:
            depth_val = float(bottom_center_cam[2])
        elif isinstance(candidate.get("center_cam_m"), list) and len(candidate["center_cam_m"]) >= 3:
            depth_val = float(candidate["center_cam_m"][2])
        if depth_val is not None and math.isfinite(depth_val):
            score += 0.85 * abs(depth_val - float(depth_prior_m)) / max(float(depth_prior_m), 1.0)

    corners = candidate.get("corners_2d") or []
    if len(corners) != 8:
        score += 1.40

    backend = str(candidate.get("backend") or "").lower()
    if backend.endswith("_depth_hint"):
        score += 1.10
    elif backend in {"fallback_depth", "fallback_depth_smoothed"}:
        score += 2.00
    elif backend in {"cersar_3d_detection_repo", "skhadem_3d_boundingbox", "lzccccc_3d_bounding_box"}:
        score -= 0.12
    return float(score)


def _select_backend_candidate(
    candidates: List[Dict[str, Any]],
    requested_backend: str,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    order = _BACKEND_SELECTION_PRIORITY.get(
        str(requested_backend or "auto").lower(),
        _BACKEND_SELECTION_PRIORITY["auto"],
    )
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        backend = str(item.get("backend") or "")
        grouped.setdefault(backend, []).append(item)
    for backend in order:
        matches = grouped.get(backend) or []
        if matches:
            return min(matches, key=lambda item: float(item.get("fit_quality", 1e9)))
    return min(candidates, key=lambda item: float(item.get("fit_quality", 1e9)))


def _build_3d_bbox_corners(
    position_3d: List[float],
    dims: Tuple[float, float, float],
    yaw: float,
) -> List[List[float]]:
    """
    Build 8 corners of an oriented 3D bbox in vehicle-centric coordinates.

    dims: (h, w, l)
    Returns list of 8 [bx, by, bz] points.
    """
    h, w, l = dims
    # Half-extents
    dx = l / 2.0
    dy = w / 2.0
    dz = h

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    cx, cy, cz = position_3d
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (0, 1):
                lx = sx * dx
                ly = sy * dy
                lz = sz * dz
                # Rotate around Z axis by yaw
                rx = lx * cos_y - ly * sin_y + cx
                ry = lx * sin_y + ly * cos_y + cy
                rz = lz + cz
                corners.append([round(rx, 3), round(ry, 3), round(rz, 3)])
    return corners


def _project_3d_to_image(
    corners_3d: List[List[float]],
    fx: float, fy: float, cx: float, cy: float,
    cam_height: float,
) -> List[List[float]]:
    """Project 3D bbox corners (in vehicle-centric coords) back to image pixels."""
    projected = []
    for bx, by, bz in corners_3d:
        Z = bx  # forward → depth
        if Z < 0.5:
            Z = 0.5
        X = -by
        Y = cam_height - bz

        u = fx * X / Z + cx
        v = fy * Y / Z + cy
        projected.append([round(u, 1), round(v, 1)])
    return projected


def _as_int_point(pt: List[float]) -> Tuple[int, int]:
    return (int(round(float(pt[0]))), int(round(float(pt[1]))))


def _quad_center(quad: List[List[float]]) -> np.ndarray:
    arr = np.asarray(quad, dtype=np.float32)
    return np.mean(arr, axis=0)


def _order_quad_points(face: List[List[float]]) -> np.ndarray:
    pts = np.asarray(face, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        return pts.astype(np.int32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.round(ordered).astype(np.int32)


def _vehicle_box_faces(corners: List[List[float]]) -> Dict[str, List[List[float]]]:
    """
    Return named faces using the corner ordering already assumed by draw():
    bottom = [0, 1, 3, 2], top = [4, 5, 7, 6].
    """
    bottom = [corners[i] for i in [0, 1, 3, 2]]
    top = [corners[i] for i in [4, 5, 7, 6]]
    return {
        "bottom": bottom,
        "top": top,
        "side_0": [bottom[0], bottom[1], top[1], top[0]],
        "side_1": [bottom[1], bottom[2], top[2], top[1]],
        "side_2": [bottom[2], bottom[3], top[3], top[2]],
        "side_3": [bottom[3], bottom[0], top[0], top[3]],
    }


def _select_orientation_face(
    corners: List[List[float]],
    anchor_xy: Optional[List[float]],
    tip_xy: Optional[List[float]],
) -> Optional[List[List[float]]]:
    if len(corners) != 8:
        return None
    if not (
        isinstance(anchor_xy, list)
        and len(anchor_xy) >= 2
        and isinstance(tip_xy, list)
        and len(tip_xy) >= 2
    ):
        return None

    anchor = np.asarray(anchor_xy[:2], dtype=np.float32)
    tip = np.asarray(tip_xy[:2], dtype=np.float32)
    direction = tip - anchor
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return None
    direction /= norm

    faces = _vehicle_box_faces(corners)
    best_face: Optional[List[List[float]]] = None
    best_score = -1e9
    for face_name in ("side_0", "side_1", "side_2", "side_3"):
        face = faces[face_name]
        center_vec = _quad_center(face) - anchor
        score = float(np.dot(center_vec, direction))
        if score > best_score:
            best_score = score
            best_face = face
    return best_face


def _draw_highlighted_face(
    image: np.ndarray,
    face: List[List[float]],
    *,
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int],
    fill_alpha: float = 0.20,
    edge_thickness: int = 3,
) -> None:
    if len(face) != 4:
        return
    pts = _order_quad_points(face)
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, pts, fill_color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(fill_alpha), image, 1.0 - float(fill_alpha), 0.0, dst=image)
    cv2.polylines(image, [pts], isClosed=True, color=edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA)


def _blend_scalar(prev_value: Optional[float], curr_value: Optional[float], alpha: float) -> Optional[float]:
    if curr_value is None:
        return prev_value
    if prev_value is None:
        return curr_value
    return float((1.0 - alpha) * float(prev_value) + alpha * float(curr_value))


def _wrap_angle_rad(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _blend_angle(prev_angle: Optional[float], curr_angle: Optional[float], alpha: float) -> Optional[float]:
    if curr_angle is None:
        return prev_angle
    if prev_angle is None:
        return curr_angle
    prev = float(prev_angle)
    curr = float(curr_angle)
    delta = _wrap_angle_rad(curr - prev)
    return _wrap_angle_rad(prev + alpha * delta)


def _align_yaw_to_reference(yaw_rad: Optional[float], ref_yaw_rad: Optional[float]) -> Optional[float]:
    if yaw_rad is None:
        return None
    yaw = float(yaw_rad)
    if ref_yaw_rad is None:
        return yaw
    ref = float(ref_yaw_rad)
    candidates = [yaw, _wrap_angle_rad(yaw + math.pi)]
    return min(candidates, key=lambda cand: abs(_wrap_angle_rad(cand - ref)))


def _blend_point_list(
    prev_points: Optional[List[List[float]]],
    curr_points: Optional[List[List[float]]],
    alpha: float,
) -> Optional[List[List[float]]]:
    if not curr_points:
        return prev_points
    if not prev_points:
        return curr_points
    if len(prev_points) != len(curr_points):
        return curr_points
    blended: List[List[float]] = []
    for prev_pt, curr_pt in zip(prev_points, curr_points):
        if len(prev_pt) < 2 or len(curr_pt) < 2:
            return curr_points
        blended.append([
            round((1.0 - alpha) * float(prev_pt[0]) + alpha * float(curr_pt[0]), 2),
            round((1.0 - alpha) * float(prev_pt[1]) + alpha * float(curr_pt[1]), 2),
        ])
    return blended


def _blend_bbox(prev_bbox: Optional[List[int]], curr_bbox: Optional[List[int]], alpha: float) -> Optional[List[int]]:
    if curr_bbox is None:
        return prev_bbox
    if prev_bbox is None:
        return curr_bbox
    return [
        int(round((1.0 - alpha) * float(prev_bbox[0]) + alpha * float(curr_bbox[0]))),
        int(round((1.0 - alpha) * float(prev_bbox[1]) + alpha * float(curr_bbox[1]))),
        int(round((1.0 - alpha) * float(prev_bbox[2]) + alpha * float(curr_bbox[2]))),
        int(round((1.0 - alpha) * float(prev_bbox[3]) + alpha * float(curr_bbox[3]))),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle3DDetector
# ─────────────────────────────────────────────────────────────────────────────

class Vehicle3DDetector:
    """
    Monocular 3D vehicle detection: YOLO 2D + depth + geometry lifting.

    Parameters
    ----------
    device : "auto" | "cuda" | "mps" | "cpu"
    yolo_model : YOLO model for 2D detection
    fx, fy, cx, cy : camera intrinsics (overridden by calibration if available)
    cam_height : camera height above ground in metres
    """

    def __init__(
        self,
        device: str = "auto",
        yolo_model: str = "auto",
        car_subclass_model: Optional[str] = "auto",
        deepbox_backend: str = _DEFAULT_DEEPBOX_BACKEND,
        cersar_weights: Optional[str] = "auto",
        cersar_repo: Optional[str] = None,
        lzccccc_weights: Optional[str] = "auto",
        lzccccc_repo: Optional[str] = None,
        lzccccc_network: str = "mobilenet_v2",
        skhadem_weights: Optional[str] = "auto",
        skhadem_repo: Optional[str] = None,
        fx: float = _DEFAULT_FX,
        fy: float = _DEFAULT_FY,
        cx: float = _DEFAULT_CX,
        cy: float = _DEFAULT_CY,
        cam_height: float = _DEFAULT_CAM_HEIGHT,
        imgsz: int = _DEFAULT_YOLO_IMGSZ,
        coarse_yaw_samples: int = _DEFAULT_COARSE_YAW_SAMPLES,
        inline_depth: bool = False,
        vehicle_nms_iou: float = _DEFAULT_VEHICLE_NMS_IOU,
        track_match_iou: float = _DEFAULT_TRACK_MATCH_IOU,
        temporal_alpha: float = _DEFAULT_TEMPORAL_ALPHA,
        max_vehicles: int = _DEFAULT_MAX_VEHICLES,
    ) -> None:
        self.device = _resolve_device(device)
        self._base_fx = float(fx)
        self._base_fy = float(fy)
        self._base_cx = float(cx)
        self._base_cy = float(cy)
        self._base_cam_height = float(cam_height)
        self._native_intrinsics_size = _default_intrinsics_native_size(cx, cy)
        self.camera_view = "front"
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.cam_height = float(cam_height)
        self.yolo_model_path = _resolve_model_path(yolo_model)
        self.imgsz = int(max(640, imgsz))
        self.coarse_yaw_samples = int(max(25, coarse_yaw_samples))
        self.inline_depth = bool(inline_depth)
        self.deepbox_backend = str(deepbox_backend or "auto").lower()
        self.vehicle_nms_iou = float(max(0.05, min(0.95, vehicle_nms_iou)))
        self.track_match_iou = float(max(0.05, min(0.95, track_match_iou)))
        self.temporal_alpha = float(max(0.05, min(1.0, temporal_alpha)))
        self.max_vehicles = int(max(1, max_vehicles))
        self._prev_tracks: List[Dict[str, Any]] = []
        self._next_track_id = 0

        try:
            from ultralytics import YOLO
            self._yolo = YOLO(self.yolo_model_path)
            print(f"[Vehicle3DDetector] YOLO loaded ({self.yolo_model_path})  device='{self.device}'")
        except ImportError as exc:
            raise ImportError("ultralytics is required.  pip install ultralytics") from exc

        # Try to load depth estimator
        self._depth_estimator = None
        if self.inline_depth:
            try:
                from depth_estimation import DepthEstimator
                self._depth_estimator = DepthEstimator(device=self.device)
                print("[Vehicle3DDetector] Depth estimator loaded for inline depth.")
            except Exception:
                print("[Vehicle3DDetector] Depth estimator not available; will use bbox-based depth fallback.")
        else:
            print("[Vehicle3DDetector] Inline depth disabled; expecting cached depth_maps.npz or bbox-depth fallback.")

        subclass_device = "cuda" if self.device == "cuda" else "cpu"
        self._vehicle_subclassifier = None
        if (
            car_subclass_model is not None
            and str(car_subclass_model).strip().lower() not in _DISABLED_MODEL_STRINGS
        ):
            self._vehicle_subclassifier = VehicleSubclassifier(
                model_name=str(car_subclass_model or "auto"),
                device=subclass_device,
            )

        self._skhadem_adapter: Optional[SkhademDeepBoxAdapter] = None
        if self.deepbox_backend in {"auto", "skhadem"}:
            try:
                self._skhadem_adapter = SkhademDeepBoxAdapter(
                    repo_dir=skhadem_repo,
                    weights_path=skhadem_weights,
                    device=self.device,
                )
                print("[Vehicle3DDetector] skhadem 3D-BoundingBox backend loaded.")
            except Exception as exc:
                if self.deepbox_backend == "skhadem":
                    raise RuntimeError(
                        "Requested --deepbox-backend skhadem, but the repo/weights could not be loaded."
                    ) from exc
                print(f"[Vehicle3DDetector] skhadem backend unavailable; using geometry fallback. ({exc})")

        self._cersar_adapter: Optional[Cersar3DDetectionAdapter] = None
        if self.deepbox_backend in {"auto", "cersar"}:
            try:
                self._cersar_adapter = Cersar3DDetectionAdapter(
                    repo_dir=cersar_repo,
                    weights_path=cersar_weights,
                )
                print("[Vehicle3DDetector] cersar 3D_detection backend loaded.")
            except Exception as exc:
                if self.deepbox_backend == "cersar":
                    raise RuntimeError(
                        "Requested --deepbox-backend cersar, but the repo/weights/runtime could not be loaded."
                    ) from exc
                print(f"[Vehicle3DDetector] cersar backend unavailable; trying other backends. ({exc})")

        self._lzccccc_adapter: Optional[Lzccccc3DBoxAdapter] = None
        if self.deepbox_backend in {"auto", "lzccccc"}:
            try:
                self._lzccccc_adapter = Lzccccc3DBoxAdapter(
                    repo_dir=lzccccc_repo,
                    weights_path=lzccccc_weights,
                    network=lzccccc_network,
                )
                print("[Vehicle3DDetector] lzccccc 3D box backend loaded.")
            except Exception as exc:
                if self.deepbox_backend == "lzccccc":
                    raise RuntimeError(
                        "Requested --deepbox-backend lzccccc, but the repo/weights could not be loaded."
                    ) from exc
                print(f"[Vehicle3DDetector] lzccccc backend unavailable; trying other backends. ({exc})")

    def configure_camera(
        self,
        *,
        view: Optional[str] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        cam_height: Optional[float] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        native_frame_width: Optional[int] = None,
        native_frame_height: Optional[int] = None,
    ) -> None:
        if view:
            self.camera_view = str(view)
        if fx is not None:
            self._base_fx = float(fx)
        if fy is not None:
            self._base_fy = float(fy)
        if cx is not None:
            self._base_cx = float(cx)
        if cy is not None:
            self._base_cy = float(cy)
        if cam_height is not None:
            self._base_cam_height = float(cam_height)

        if native_frame_width is not None and native_frame_height is not None:
            self._native_intrinsics_size = (
                max(1.0, float(native_frame_width)),
                max(1.0, float(native_frame_height)),
            )
        elif any(value is not None for value in (fx, fy, cx, cy)) and self._native_intrinsics_size is None:
            self._native_intrinsics_size = _default_intrinsics_native_size(self._base_cx, self._base_cy)

        sx = sy = 1.0
        if frame_width and frame_height and self._native_intrinsics_size is not None:
            native_w = max(self._native_intrinsics_size[0], 1.0)
            native_h = max(self._native_intrinsics_size[1], 1.0)
            sx = float(frame_width) / native_w
            sy = float(frame_height) / native_h

        self.fx = float(self._base_fx * sx)
        self.fy = float(self._base_fy * sy)
        self.cx = float(self._base_cx * sx)
        self.cy = float(self._base_cy * sy)
        self.cam_height = float(self._base_cam_height)

    def _stabilize_vehicles(self, vehicles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not vehicles:
            self._prev_tracks = []
            return []

        alpha = self.temporal_alpha
        prev_tracks = list(self._prev_tracks)
        matched_prev: set[int] = set()
        stabilized: List[Dict[str, Any]] = []

        for current in sorted(vehicles, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            bbox = [int(v) for v in current["bbox_2d"][:4]]
            cls_name = str(current.get("class") or "")
            best_idx = -1
            best_score = -1e9

            for idx, prev in enumerate(prev_tracks):
                if idx in matched_prev:
                    continue
                if str(prev.get("class") or "") != cls_name:
                    continue
                prev_bbox = [int(v) for v in prev["bbox_2d"][:4]]
                iou = _bbox_iou(bbox, prev_bbox)
                if iou < self.track_match_iou:
                    continue
                depth_now = float(current.get("depth_m") or 0.0)
                depth_prev = float(prev.get("depth_m") or 0.0)
                depth_penalty = abs(depth_now - depth_prev) / max(depth_now, depth_prev, 1.0)
                score = iou - 0.12 * depth_penalty
                if score > best_score:
                    best_score = score
                    best_idx = idx

            smoothed = dict(current)
            if best_idx >= 0:
                prev = prev_tracks[best_idx]
                matched_prev.add(best_idx)
                smoothed["id"] = int(prev["id"])
                if str(smoothed.get("subclass") or "") == "car" and str(prev.get("subclass") or "") != "car":
                    smoothed["subclass"] = str(prev["subclass"])
                smoothed["bbox_2d"] = _blend_bbox(prev.get("bbox_2d"), current.get("bbox_2d"), alpha)
                smoothed["depth_m"] = round(_blend_scalar(prev.get("depth_m"), current.get("depth_m"), alpha) or 0.0, 3)
                prev_pos = list(prev.get("position_3d") or [0.0, 0.0, 0.0])
                curr_pos = list(current.get("position_3d") or [0.0, 0.0, 0.0])
                motion_heading = None
                if len(prev_pos) >= 2 and len(curr_pos) >= 2:
                    dx = float(curr_pos[0]) - float(prev_pos[0])
                    dy = float(curr_pos[1]) - float(prev_pos[1])
                    if math.hypot(dx, dy) >= 0.18:
                        motion_heading = math.atan2(dy, dx)
                prev_yaw = _align_yaw_to_reference(prev.get("orientation_rad"), motion_heading)
                curr_yaw = _align_yaw_to_reference(current.get("orientation_rad"), motion_heading if motion_heading is not None else prev_yaw)
                yaw_delta = abs(_wrap_angle_rad(float(curr_yaw or 0.0) - float(prev_yaw or 0.0)))
                yaw_alpha = min(alpha, 0.50) if yaw_delta >= 0.30 else min(alpha, 0.62)
                smoothed["orientation_rad"] = round(_blend_angle(prev_yaw, curr_yaw, yaw_alpha) or 0.0, 4)
                smoothed["position_3d"] = [
                    round(_blend_scalar(prev_pt, curr_pt, alpha) or 0.0, 3)
                    for prev_pt, curr_pt in zip(
                        prev_pos,
                        curr_pos,
                    )
                ]
                smoothed["dimensions_3d"] = [
                    round(_blend_scalar(prev_dim, curr_dim, alpha) or 0.0, 3)
                    for prev_dim, curr_dim in zip(
                        list(prev.get("dimensions_3d") or [0.0, 0.0, 0.0]),
                        list(current.get("dimensions_3d") or [0.0, 0.0, 0.0]),
                    )
                ]
                smoothed["bbox_3d_corners"] = _blend_point_list(
                    prev.get("bbox_3d_corners"),
                    current.get("bbox_3d_corners"),
                    alpha,
                ) or list(current.get("bbox_3d_corners") or [])
                smoothed["bbox_2d_projected"] = _blend_bbox(
                    prev.get("bbox_2d_projected"),
                    current.get("bbox_2d_projected"),
                    alpha,
                ) or list(current.get("bbox_2d_projected") or [])
                smoothed["orientation_anchor_2d"] = (
                    _blend_point_list(
                        [prev.get("orientation_anchor_2d")] if prev.get("orientation_anchor_2d") else None,
                        [current.get("orientation_anchor_2d")] if current.get("orientation_anchor_2d") else None,
                        alpha,
                    ) or ([current.get("orientation_anchor_2d")] if current.get("orientation_anchor_2d") else [])
                )
                smoothed["orientation_tip_2d"] = (
                    _blend_point_list(
                        [prev.get("orientation_tip_2d")] if prev.get("orientation_tip_2d") else None,
                        [current.get("orientation_tip_2d")] if current.get("orientation_tip_2d") else None,
                        alpha,
                    ) or ([current.get("orientation_tip_2d")] if current.get("orientation_tip_2d") else [])
                )
                smoothed["orientation_face_2d"] = _blend_point_list(
                    prev.get("orientation_face_2d"),
                    current.get("orientation_face_2d"),
                    alpha,
                ) or list(current.get("orientation_face_2d") or prev.get("orientation_face_2d") or [])
                if isinstance(smoothed["orientation_anchor_2d"], list) and len(smoothed["orientation_anchor_2d"]) == 1:
                    smoothed["orientation_anchor_2d"] = smoothed["orientation_anchor_2d"][0]
                if isinstance(smoothed["orientation_tip_2d"], list) and len(smoothed["orientation_tip_2d"]) == 1:
                    smoothed["orientation_tip_2d"] = smoothed["orientation_tip_2d"][0]
            else:
                smoothed["id"] = self._next_track_id
                self._next_track_id += 1

            stabilized.append(smoothed)

        self._prev_tracks = [dict(item) for item in stabilized]
        return stabilized

    def _apply_vehicle_subclassification(
        self,
        frame_bgr: np.ndarray,
        candidates: List[Dict[str, Any]],
    ) -> None:
        if self._vehicle_subclassifier is None:
            return
        car_indices = []
        for idx, item in enumerate(candidates):
            if str(item.get("class")) != "car":
                continue
            existing = str(item.get("subclass") or "car").lower()
            if existing in {"sedan", "suv", "hatchback", "pickup"}:
                continue
            car_indices.append(idx)
        if not car_indices:
            return
        predictions = self._vehicle_subclassifier.classify_car_boxes(
            frame_bgr,
            [candidates[idx]["bbox"] for idx in car_indices],
        )
        for idx, pred in zip(car_indices, predictions):
            if pred.subtype in {"sedan", "suv", "hatchback", "pickup"}:
                candidates[idx]["subclass"] = pred.subtype

    def detect(
        self,
        frame_bgr: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        precomputed_vehicles: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect vehicles and estimate 3D bounding boxes.

        Parameters
        ----------
        frame_bgr : H x W x 3 uint8 BGR image
        depth_map : optional pre-computed depth map (float32, metres)

        Returns list of vehicle dicts with 3D information.
        """
        H, W = frame_bgr.shape[:2]

        # Get depth map if not provided
        if depth_map is None and self._depth_estimator is not None:
            depth_map = self._depth_estimator.estimate(frame_bgr)

        candidates: List[Dict[str, Any]] = []
        if precomputed_vehicles is not None:
            for item in precomputed_vehicles:
                parsed = _vehicle_from_detection_entry(item)
                if parsed is not None:
                    parsed_bbox = _normalize_bbox_xyxy(parsed.get("bbox"), W, H)
                    if parsed_bbox is None:
                        continue
                    parsed["bbox"] = parsed_bbox
                    candidates.append(parsed)
        else:
            results = self._yolo(
                frame_bgr,
                verbose=False,
                device=self.device,
                imgsz=self.imgsz,
            )
            result = results[0]
            names = result.names if hasattr(result, "names") else {}
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf < 0.40:
                    continue

                cls_id = int(box.cls.item())
                raw_label = names.get(cls_id, str(cls_id))
                cls_name = _canonicalize_model_label(raw_label)
                if cls_name not in _VEHICLE_CLASSES:
                    continue
                bbox = _normalize_bbox_xyxy(box.xyxy[0].tolist(), W, H)
                if bbox is None:
                    continue
                candidates.append({
                    "class": cls_name,
                    "subclass": _subclassify_vehicle(cls_name, bbox),
                    "bbox": bbox,
                    "confidence": conf,
                })

        candidates = _nms_vehicle_candidates(
            candidates,
            iou_threshold=self.vehicle_nms_iou,
            max_keep=self.max_vehicles,
        )
        self._apply_vehicle_subclassification(frame_bgr, candidates)

        vehicles: List[Dict[str, Any]] = []
        for idx, item in enumerate(candidates):
            conf = float(item["confidence"])
            cls_name = str(item["class"])
            bbox = [int(v) for v in item["bbox"][:4]]
            subclass = str(item.get("subclass") or _subclassify_vehicle(cls_name, bbox))
            x1, y1, x2, y2 = bbox

            # Depth
            depth_m = _bbox_center_depth(bbox, depth_map)
            if depth_m is None or depth_m <= 0.5 or depth_m > 120.0:
                depth_m = _estimate_depth_from_bbox(bbox, cls_name, subclass, self.fy)

            dims_prior = VEHICLE_DIMS.get(subclass, VEHICLE_DIMS.get(cls_name, (1.5, 1.85, 4.5)))
            bbox_w = max(x2 - x1, 1)
            bbox_h = max(y2 - y1, 1)
            yaw_hint_blender = _estimate_yaw(bbox, [0.0, 0.0, float(depth_m)], W)
            fit = None
            cersar_fit = None
            lzccccc_fit = None
            skhadem_fit = None
            candidate_pool: List[Dict[str, Any]] = []

            cersar_override_dims = dims_prior if subclass in {"motorcycle", "bicycle"} else None
            if self._cersar_adapter is not None:
                try:
                    cersar_fit = self._cersar_adapter.predict(
                        frame_bgr=frame_bgr,
                        bbox_xyxy=bbox,
                        cls_name=subclass if subclass in {"sedan", "suv", "hatchback", "pickup"} else cls_name,
                        fx=self.fx,
                        fy=self.fy,
                        cx=self.cx,
                        cy=self.cy,
                        cam_height_m=self.cam_height,
                        depth_prior_m=depth_m,
                        dims_override_hwl_m=cersar_override_dims,
                    )
                except Exception:
                    cersar_fit = None
            if cersar_fit is not None:
                cersar_fit = dict(cersar_fit)
                cersar_fit["fit_quality"] = round(_candidate_fit_score(cersar_fit, bbox, depth_m), 4)
                if not _candidate_fit_is_extreme(cersar_fit, bbox):
                    candidate_pool.append(cersar_fit)

            if self._skhadem_adapter is not None:
                try:
                    skhadem_fit = self._skhadem_adapter.predict(
                        frame_bgr=frame_bgr,
                        bbox_xyxy=bbox,
                        cls_name=subclass if subclass in {"sedan", "suv", "hatchback", "pickup"} else cls_name,
                        fx=self.fx,
                        fy=self.fy,
                        cx=self.cx,
                        cy=self.cy,
                        cam_height_m=self.cam_height,
                    )
                except Exception:
                    skhadem_fit = None
            if skhadem_fit is not None:
                skhadem_fit = dict(skhadem_fit)
                skhadem_fit["fit_quality"] = round(_candidate_fit_score(skhadem_fit, bbox, depth_m), 4)
                if not _candidate_fit_is_extreme(skhadem_fit, bbox):
                    candidate_pool.append(skhadem_fit)

            if self._lzccccc_adapter is not None:
                try:
                    lz_class = subclass if subclass in {"sedan", "suv", "hatchback", "pickup"} else cls_name
                    use_dims_override = lz_class in {"truck", "motorcycle", "bicycle"}
                    lzccccc_fit = self._lzccccc_adapter.predict(
                        frame_bgr=frame_bgr,
                        bbox_xyxy=bbox,
                        cls_name=lz_class,
                        fx=self.fx,
                        fy=self.fy,
                        cx=self.cx,
                        cy=self.cy,
                        cam_height_m=self.cam_height,
                        dims_override_hwl_m=dims_prior if use_dims_override else None,
                    )
                except Exception:
                    lzccccc_fit = None
            if lzccccc_fit is not None:
                lzccccc_fit = dict(lzccccc_fit)
                lzccccc_fit["fit_quality"] = round(_candidate_fit_score(lzccccc_fit, bbox, depth_m), 4)
                if not _candidate_fit_is_extreme(lzccccc_fit, bbox):
                    candidate_pool.append(lzccccc_fit)

            dims = dims_prior
            if bbox_w >= _MIN_GEOM_BBOX_W and bbox_h >= _MIN_GEOM_BBOX_H:
                fit = fit_box(
                    bbox_xyxy=bbox,
                    dims_hwl_m=dims,
                    fx=self.fx,
                    fy=self.fy,
                    cx=self.cx,
                    cy=self.cy,
                    cam_height_m=self.cam_height,
                    depth_prior_m=depth_m,
                    yaw_hint_cam_rad=-float(yaw_hint_blender),
                    coarse_yaw_samples=self.coarse_yaw_samples,
                )
            if fit is not None:
                fit_candidate = {
                    "dims_hwl_m": [float(v) for v in fit.dims_hwl_m],
                    "bottom_center_blender_m": [float(v) for v in fit.bottom_center_blender_m],
                    "yaw_blender_rad": float(fit.yaw_blender_rad),
                    "corners_2d": list(fit.corners_2d),
                    "bottom_center_cam_m": [float(v) for v in fit.bottom_center_cam_m],
                    "bbox_projected": [int(v) for v in fit.bbox_projected],
                    "reprojection_error": float(fit.reprojection_error),
                    "center_cam_m": [float(v) for v in fit.center_cam_m],
                    "orientation_anchor_2d": [float(v) for v in fit.orientation_anchor_2d],
                    "orientation_tip_2d": [float(v) for v in fit.orientation_tip_2d],
                    "orientation_face_2d": [[float(pt[0]), float(pt[1])] for pt in (fit.orientation_face_2d or [])],
                    "theta_ray_rad": float(fit.theta_ray_rad),
                    "alpha_local_rad": float(fit.alpha_local_rad),
                    "backend": "deep3dbox_geometry_alpha_search",
                }
                fit_candidate["fit_quality"] = round(_candidate_fit_score(fit_candidate, bbox, depth_m), 4)
                if not _candidate_fit_is_extreme(fit_candidate, bbox):
                    candidate_pool.append(fit_candidate)

            best_candidate = _select_backend_candidate(candidate_pool, self.deepbox_backend)

            if best_candidate is not None:
                dims = tuple(float(v) for v in best_candidate["dims_hwl_m"])
                position_3d = [round(float(v), 3) for v in best_candidate["bottom_center_blender_m"]]
                yaw = float(best_candidate["yaw_blender_rad"])
                corners_2d = list(best_candidate.get("corners_2d") or [])
                depth_out = float(best_candidate["bottom_center_cam_m"][2])
                projected_bbox = list(best_candidate["bbox_projected"])
                reproj_error = float(best_candidate["reprojection_error"])
                center_cam_m = [round(float(v), 4) for v in best_candidate["center_cam_m"]]
                orientation_anchor_2d = list(best_candidate["orientation_anchor_2d"])
                orientation_tip_2d = list(best_candidate["orientation_tip_2d"])
                theta_ray_rad = float(best_candidate["theta_ray_rad"])
                alpha_local_rad = float(best_candidate["alpha_local_rad"])
                backend = str(best_candidate["backend"])
                fit_quality = float(best_candidate.get("fit_quality", 0.0))
                orientation_face_2d = [
                    [round(float(pt[0]), 2), round(float(pt[1]), 2)]
                    for pt in list(best_candidate.get("orientation_face_2d") or [])
                ]
            else:
                position_3d = _back_project_bbox_center(
                    bbox, depth_m, self.fx, self.fy, self.cx, self.cy, self.cam_height
                )
                yaw = yaw_hint_blender
                corners_2d = []
                depth_out = depth_m
                projected_bbox = bbox
                reproj_error = None
                center_cam_m = None
                anchor_x = 0.5 * (x1 + x2)
                anchor_y = 0.5 * (y1 + y2)
                orientation_anchor_2d = [anchor_x, anchor_y]
                orientation_tip_2d = [
                    anchor_x + 28.0 * math.sin(yaw),
                    anchor_y - 28.0 * math.cos(yaw),
                ]
                theta_ray_rad = None
                alpha_local_rad = None
                backend = "fallback_depth"
                fit_quality = None
                orientation_face_2d = []

            vehicles.append({
                "id": idx,
                "class": cls_name,
                "subclass": subclass,
                "bbox_2d": bbox,
                "confidence": round(conf, 4),
                "depth_m": round(float(depth_out), 3),
                "position_3d": position_3d,
                "dimensions_3d": [round(d, 3) for d in dims],
                "orientation_rad": round(yaw, 4),
                "bbox_3d_corners": corners_2d,
                "bbox_2d_projected": projected_bbox,
                "center_cam_m": center_cam_m,
                "orientation_anchor_2d": [round(float(v), 2) for v in orientation_anchor_2d],
                "orientation_tip_2d": [round(float(v), 2) for v in orientation_tip_2d],
                "orientation_face_2d": orientation_face_2d,
                "theta_ray_rad": round(float(theta_ray_rad), 5) if theta_ray_rad is not None else None,
                "alpha_local_rad": round(float(alpha_local_rad), 5) if alpha_local_rad is not None else None,
                "backend": backend,
                "reprojection_error": round(float(reproj_error), 4) if reproj_error is not None else None,
                "fit_quality": round(float(fit_quality), 4) if fit_quality is not None else None,
            })

        vehicles = sorted(
            vehicles,
            key=lambda item: (float(item.get("confidence", 0.0)), -float(item.get("depth_m", 0.0))),
            reverse=True,
        )
        return self._stabilize_vehicles(vehicles)

    def draw(self, frame_bgr: np.ndarray, vehicles: List[Dict[str, Any]]) -> np.ndarray:
        """Draw projected 3D bounding boxes on the frame."""
        vis = frame_bgr.copy()
        base_color_car = (96, 245, 110)
        base_color_other = (92, 236, 156)
        highlight_color = (255, 90, 35)
        highlight_fill = (255, 138, 72)

        for veh in sorted(vehicles, key=lambda item: float(item.get("depth_m", 0.0)), reverse=True):
            corners = veh["bbox_3d_corners"]
            backend = str(veh.get("backend", ""))
            geometry_backed = (
                len(corners) == 8
                and backend not in {"fallback_depth", "fallback_depth_smoothed", "cersar_3d_detection_depth_hint"}
            )
            color = base_color_car if veh["class"] == "car" else base_color_other
            if not geometry_backed:
                color = (120, 185, 255) if veh["class"] == "car" else (140, 210, 255)
            bx1, by1, bx2, by2 = veh["bbox_2d"]
            anchor_xy = veh.get("orientation_anchor_2d")
            tip_xy = veh.get("orientation_tip_2d")
            box_scale = max(bx2 - bx1, by2 - by1)
            major_thickness = 2 if geometry_backed and box_scale >= 90 else 1
            minor_thickness = 1

            if len(corners) == 8:
                faces = _vehicle_box_faces(corners)
                bottom = faces["bottom"]
                top = faces["top"]

                for i in range(4):
                    j = (i + 1) % 4
                    p1 = _as_int_point(bottom[i])
                    p2 = _as_int_point(bottom[j])
                    cv2.line(vis, p1, p2, color, major_thickness, cv2.LINE_AA)

                for i in range(4):
                    j = (i + 1) % 4
                    p1 = _as_int_point(top[i])
                    p2 = _as_int_point(top[j])
                    cv2.line(vis, p1, p2, color, major_thickness, cv2.LINE_AA)

                for i in range(4):
                    p1 = _as_int_point(bottom[i])
                    p2 = _as_int_point(top[i])
                    cv2.line(vis, p1, p2, color, minor_thickness, cv2.LINE_AA)

                orientation_face = veh.get("orientation_face_2d")
                if not isinstance(orientation_face, list) or len(orientation_face) != 4:
                    orientation_face = _select_orientation_face(corners, anchor_xy, tip_xy)
                if geometry_backed and orientation_face is not None and backend != "cersar_3d_detection_repo":
                    _draw_highlighted_face(
                        vis,
                        orientation_face,
                        fill_color=highlight_fill,
                        edge_color=highlight_color,
                        fill_alpha=0.16,
                        edge_thickness=max(2, major_thickness),
                    )
            else:
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, major_thickness)

            # Label
            label = (
                f"#{veh['id']} {veh['subclass']} {veh['confidence']:.2f} "
                f"d={veh['depth_m']:.1f}m"
            )
            if geometry_backed:
                label += f" yaw={math.degrees(veh['orientation_rad']):.0f}deg"
            else:
                label += " approx"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_bg = vis.copy()
            box_y = max(min(by1 - th - 10, vis.shape[0] - th - 10), 0)
            cv2.rectangle(label_bg, (bx1 - 2, box_y), (bx1 + tw + 6, box_y + th + 8), (18, 18, 18), cv2.FILLED)
            cv2.addWeighted(label_bg, 0.45, vis, 0.55, 0, vis)
            cv2.putText(vis, label, (bx1, box_y + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        return vis


# ─────────────────────────────────────────────────────────────────────────────
# Safer video output helpers for macOS / VS Code
# ─────────────────────────────────────────────────────────────────────────────

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def reencode_for_vscode(src_video: Path, dst_video: Path) -> bool:
    if not ffmpeg_available():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_video),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst_video),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


class SafeVideoWriter:
    def __init__(self, requested_output: Path, fps: float, width: int, height: int, vscode_compatible: bool = True):
        self.requested_output = requested_output
        self.vscode_compatible = vscode_compatible
        requested_output.parent.mkdir(parents=True, exist_ok=True)

        self.temp_video = requested_output.with_suffix(".tmp.avi")
        if self.temp_video.exists():
            try:
                self.temp_video.unlink()
            except OSError:
                pass

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = cv2.VideoWriter(str(self.temp_video), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError("Could not open MJPG AVI writer.")

        print(f"[SafeVideoWriter] Writing temp AVI -> {self.temp_video}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> Path:
        self.writer.release()

        if self.vscode_compatible and self.requested_output.suffix.lower() == ".mp4":
            ok = reencode_for_vscode(self.temp_video, self.requested_output)
            if ok:
                try:
                    self.temp_video.unlink(missing_ok=True)
                except Exception:
                    pass
                print(f"[SafeVideoWriter] Re-encoded -> {self.requested_output}")
                return self.requested_output

        fallback = self.requested_output.with_suffix(".avi")
        if fallback.exists():
            try:
                fallback.unlink()
            except OSError:
                pass
        self.temp_video.rename(fallback)
        print(f"[SafeVideoWriter] Using AVI fallback -> {fallback}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle3DPipeline — video processor
# ─────────────────────────────────────────────────────────────────────────────

class Vehicle3DPipeline:
    """End-to-end video processor for 3D vehicle detection."""

    def __init__(
        self,
        detector: Optional[Vehicle3DDetector] = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.detector = detector or Vehicle3DDetector(device=device, **kwargs)

    @staticmethod
    def _open_writer(path: Path, fps: float, W: int, H: int) -> "SafeVideoWriter":
        return SafeVideoWriter(path, fps, W, H, vscode_compatible=True)

    def run(
        self,
        video_path: str,
        out_video: str,
        out_json: str,
        depth_npz: Optional[str] = None,
        detections_json: Optional[str] = None,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
    ) -> List[Dict[str, Any]]:
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {src}")

        out_video_path = Path(out_video)
        out_json_path = Path(out_json)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Load depth archive if available
        depth_store = None
        if depth_npz and Path(depth_npz).exists():
            depth_store = np.load(depth_npz, allow_pickle=False)
            print(f"[Vehicle3DPipeline] Loaded depth archive: {depth_npz}")

        detection_store: Optional[Dict[int, List[Dict[str, Any]]]] = None
        if detections_json and Path(detections_json).exists():
            with open(detections_json, "r") as f:
                raw = json.load(f)
            detection_store = {}
            for frame_record in raw.get("frames", []):
                try:
                    frame_idx = int(frame_record.get("frame_idx"))
                except (TypeError, ValueError):
                    continue
                vehicles = []
                for det in frame_record.get("detections", []):
                    parsed = _vehicle_from_detection_entry(det)
                    if parsed is not None:
                        vehicles.append(parsed)
                detection_store[frame_idx] = vehicles
            print(f"[Vehicle3DPipeline] Loaded 2D vehicle detections: {detections_json}")

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = src_fps / max(frame_skip, 1)
        self.detector.configure_camera(frame_width=src_W, frame_height=src_H)
        print(
            "[Vehicle3DPipeline] Active camera "
            f"view='{self.detector.camera_view}' "
            f"fx={self.detector.fx:.1f} fy={self.detector.fy:.1f} "
            f"cx={self.detector.cx:.1f} cy={self.detector.cy:.1f} "
            f"h={self.detector.cam_height:.2f}"
        )

        print(f"\n{'='*64}")
        print(f"  Vehicle3DPipeline.run()")
        print(f"  Input  : {src}  ({src_W}x{src_H} @ {src_fps:.1f} fps)")
        print(f"  Output : {out_video_path}")
        print(f"  JSON   : {out_json_path}")
        print(f"{'='*64}\n")

        writer = self._open_writer(out_video_path, out_fps, src_W, src_H)
        final_video = out_video_path

        all_records: List[Dict[str, Any]] = []
        total_dets = 0
        written = 0
        src_idx = 0
        t_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                ts = src_idx / src_fps

                # Get depth map from archive if available
                depth_map = None
                if depth_store is not None:
                    key = str(written)
                    if key in depth_store.files:
                        depth_map = depth_store[key]

                precomputed = None
                if detection_store is not None:
                    precomputed = detection_store.get(written)
                    if precomputed is None:
                        precomputed = detection_store.get(src_idx)

                vehicles = self.detector.detect(
                    frame,
                    depth_map=depth_map,
                    precomputed_vehicles=precomputed,
                )
                all_records.append({
                    "frame_idx": written,
                    "timestamp_s": round(ts, 4),
                    "vehicles_3d": vehicles,
                })
                total_dets += len(vehicles)

                vis = self.detector.draw(frame, vehicles)

                # HUD
                hud = f"Frame {written}  Vehicles3D: {len(vehicles)}  Total: {total_dets}"
                cv2.putText(vis, hud, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 230, 255), 1, cv2.LINE_AA)

                writer.write(vis)
                written += 1
                src_idx += 1

                if written % 30 == 0:
                    elapsed = time.time() - t_start
                    fps_est = written / max(elapsed, 1e-6)
                    pct = src_idx / src_total * 100 if src_total > 0 else 0
                    print(f"  frame {src_idx:5d}/{src_total}  ({pct:5.1f}%)  "
                          f"{fps_est:5.1f} fps  vehs={len(vehicles)}",
                          end="\r", flush=True)

                if max_frames is not None and written >= max_frames:
                    print(f"\n  Reached max_frames={max_frames}.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            cap.release()
            final_video = writer.close()
            if depth_store is not None:
                depth_store.close()

        with open(out_json_path, "w") as f:
            json.dump({
                "source": str(src),
                "camera": {
                    "view": self.detector.camera_view,
                    "fx": round(float(self.detector.fx), 4),
                    "fy": round(float(self.detector.fy), 4),
                    "cx": round(float(self.detector.cx), 4),
                    "cy": round(float(self.detector.cy), 4),
                    "camera_height_m": round(float(self.detector.cam_height), 4),
                },
                "frames_written": written,
                "total_detections": total_dets,
                "final_video": str(final_video),
                "frames": all_records,
            }, f, indent=2)

        elapsed_total = time.time() - t_start
        print(f"\n\n{'='*64}")
        print(f"  Done.  {written} frames, {total_dets} vehicles")
        print(f"  Wall time: {elapsed_total:.1f}s ({written/max(elapsed_total,1e-6):.1f} fps)")
        print(f"  Output: {final_video}")
        print(f"  JSON:   {out_json_path}")
        print(f"{'='*64}")

        return all_records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monocular 3D vehicle detection pipeline")
    parser.add_argument("--video", type=str, required=True, help="Input video path (.mp4)")
    parser.add_argument("--scene", default=None,
                        help="Scene id (e.g. scene10); inferred from path if omitted")
    parser.add_argument("--view", default="auto", choices=["auto", "front", "back", "left", "right"],
                        help="Camera view for calibration. 'auto' infers from the input filename.")
    parser.add_argument("--out-video", default=None, help="Annotated output video path")
    parser.add_argument("--out-json", default=None, help="3D detection JSON path")
    parser.add_argument("--depth-npz", default=None,
                        help="Pre-computed depth npz (from depth_estimation.py)")
    parser.add_argument("--detections-json", default=None,
                        help="Optional 2D detections JSON from object_detection.py; when provided the 3D stage reuses those vehicle boxes instead of rerunning YOLO.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--yolo-model", default="auto",
                        help="YOLO model for 2D detection. 'auto' resolves to YOLO26 weights/models only.")
    parser.add_argument("--car-subclass-model", default="auto",
                        help="Optional car-subclassification model. 'auto' uses the local HF cache when available; 'none' disables it.")
    parser.add_argument("--imgsz", type=int, default=_DEFAULT_YOLO_IMGSZ,
                        help="YOLO inference size when 2D detections are not being reused from detections.json.")
    parser.add_argument("--coarse-yaw-samples", type=int, default=_DEFAULT_COARSE_YAW_SAMPLES,
                        help="Number of coarse yaw samples for the Deep3DBox search. Lower is faster.")
    parser.add_argument("--vehicle-nms-iou", type=float, default=_DEFAULT_VEHICLE_NMS_IOU,
                        help="Class-agnostic NMS IoU applied to vehicle proposals before 3D lifting.")
    parser.add_argument("--track-match-iou", type=float, default=_DEFAULT_TRACK_MATCH_IOU,
                        help="Minimum IoU for matching a vehicle to the previous frame for temporal smoothing.")
    parser.add_argument("--temporal-alpha", type=float, default=_DEFAULT_TEMPORAL_ALPHA,
                        help="Current-frame weight for temporal smoothing (higher = less lag, lower = smoother).")
    parser.add_argument("--max-vehicles", type=int, default=_DEFAULT_MAX_VEHICLES,
                        help="Maximum number of vehicle proposals to keep per frame after NMS.")
    parser.add_argument("--inline-depth", action="store_true",
                        help="Run inline monocular depth if a cached depth npz is unavailable. Slower than using output/<scene>/depth/depth_maps.npz.")
    parser.add_argument("--deepbox-backend", default=_DEFAULT_DEEPBOX_BACKEND,
                        choices=["auto", "cersar", "lzccccc", "skhadem", "geometry"],
                        help="3D box backend. Default prefers 'skhadem' with epoch_50.pk(.l); 'cersar' uses external/3D_detection + weights/weights.h5 with depth/calibration validation; 'lzccccc' uses external/lzccccc-3d-bounding-box; 'geometry' forces the local paper-style solver.")
    parser.add_argument("--cersar-weights", default="auto",
                        help="Path to cersar 3D_detection .h5 weights. 'auto' prefers weights/weights.h5, then external/3D_detection/model_saved/weights.h5.")
    parser.add_argument("--cersar-repo", default=None,
                        help="Optional path to a cloned cersar 3D_detection repo.")
    parser.add_argument("--lzccccc-weights", default="auto",
                        help="Path to lzccccc .hdf5 weights. 'auto' looks for 3dbox_weights_mob.hdf5 / 3dbox_weights_vgg.hdf5 under external/lzccccc-3d-bounding-box.")
    parser.add_argument("--lzccccc-repo", default=None,
                        help="Optional path to a cloned lzccccc 3D bounding box repo.")
    parser.add_argument("--lzccccc-network", default="mobilenet_v2", choices=["mobilenet_v2", "vgg16"],
                        help="lzccccc network family to instantiate for inference.")
    parser.add_argument("--skhadem-weights", default="auto",
                        help="Path to skhadem 3D-BoundingBox .pk/.pkl weights. 'auto' prefers weights/epoch_50.pk(.l), then external/3D-BoundingBox/weights/epoch_50.pk(.l) / epoch_10.pk(.l).")
    parser.add_argument("--skhadem-repo", default=None,
                        help="Optional path to a cloned skhadem 3D-BoundingBox repo.")

    # Camera intrinsics (overridable)
    parser.add_argument("--fx", type=float, default=_DEFAULT_FX)
    parser.add_argument("--fy", type=float, default=_DEFAULT_FY)
    parser.add_argument("--cx", type=float, default=_DEFAULT_CX)
    parser.add_argument("--cy", type=float, default=_DEFAULT_CY)
    parser.add_argument("--cam-height", type=float, default=_DEFAULT_CAM_HEIGHT)

    args = parser.parse_args()

    scene_name = infer_scene_name(args.scene, args.video, args.out_video, args.out_json)
    output_layout = scene_output_layout(scene_name, create=True)

    out_video = (
        str(Path(args.out_video).resolve()) if args.out_video
        else str((output_layout.detections / "vehicle_3d_output.mp4").resolve())
    )
    out_json = (
        str(Path(args.out_json).resolve()) if args.out_json
        else str((output_layout.detections / "vehicle_3d_detections.json").resolve())
    )

    # Try to load calibration
    fx, fy, cx, cy, cam_h = args.fx, args.fy, args.cx, args.cy, args.cam_height
    camera_view = _infer_camera_view(args.view, args.video)
    try:
        from calibration import load_calibration
        calib = load_calibration(camera_view)
        fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
        cam_h = calib.camera_height_m
        print(
            f"[main] Using calibrated intrinsics for view='{camera_view}': "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        )
    except Exception:
        print(
            f"[main] Using default intrinsics for view='{camera_view}': "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        )

    # Depth npz auto-discovery
    depth_npz = args.depth_npz
    if depth_npz is None:
        candidates = [
            output_layout.depth / "depth_maps.npz",
            output_layout.depth / "depth_frames.npz",
            Path(f"P3Data/Sequences/{scene_name}/Depth/depth_maps.npz"),
            Path(f"P3Data/Sequences/{scene_name}/Depth/depth_frames.npz"),
        ]
        for cand in candidates:
            if cand.exists():
                depth_npz = str(cand)
                print(f"[main] Auto-discovered depth: {depth_npz}")
                break

    detections_json = args.detections_json
    if detections_json is None:
        candidates = [
            output_layout.detections / "detections.json",
        ]
        for cand in candidates:
            if cand.exists():
                detections_json = str(cand)
                print(f"[main] Auto-discovered 2D detections: {detections_json}")
                break

    detector = Vehicle3DDetector(
        device=args.device,
        yolo_model=args.yolo_model,
        car_subclass_model=args.car_subclass_model,
        fx=fx, fy=fy, cx=cx, cy=cy,
        cam_height=cam_h,
        imgsz=args.imgsz,
        coarse_yaw_samples=args.coarse_yaw_samples,
        inline_depth=args.inline_depth,
        vehicle_nms_iou=args.vehicle_nms_iou,
        track_match_iou=args.track_match_iou,
        temporal_alpha=args.temporal_alpha,
        max_vehicles=args.max_vehicles,
        deepbox_backend=args.deepbox_backend,
        cersar_weights=args.cersar_weights,
        cersar_repo=args.cersar_repo,
        lzccccc_weights=args.lzccccc_weights,
        lzccccc_repo=args.lzccccc_repo,
        lzccccc_network=args.lzccccc_network,
        skhadem_weights=args.skhadem_weights,
        skhadem_repo=args.skhadem_repo,
    )
    detector.configure_camera(
        view=camera_view,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        cam_height=cam_h,
    )
    pipe = Vehicle3DPipeline(detector=detector)
    pipe.run(
        args.video,
        out_video=out_video,
        out_json=out_json,
        depth_npz=depth_npz,
        detections_json=detections_json,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )

    if not args.out_video:
        mirror_stage_output(out_video, scene_name, "detections", Path(out_video).name)
    if not args.out_json:
        mirror_stage_output(out_json, scene_name, "detections", Path(out_json).name)
