"""
scene_assembler.py
==================

Scene assembly stage for the autonomous-driving reconstruction pipeline.

Purpose
-------
This script is the bridge between the perception outputs and Blender.  It takes
heterogeneous per-frame detections, traffic-light estimates, lane/road
geometry, depth maps, and calibration metadata, and consolidates them into one
compact scene description that Blender can render deterministically.

Inputs
------
The assembler can fuse information from the following project outputs:

* ``output/<scene>/detections/`` for per-frame or aggregated detection outputs
* ``output/<scene>/traffic_lights/`` for traffic-light states
* ``output/<scene>/lanes/`` for lane and reconstructed road context
* ``output/<scene>/depth/`` or ``P3Data/Sequences/<scene>/Depth/`` for depth
  archives
* legacy mirrors such as ``renders/<scene>/...`` and
  ``output/detections/<scene>/...`` are still accepted automatically
* ``P3Data/Sequences/<scene>/Depth/depth_frames.npz`` for per-frame depth

Outputs
-------
The generated JSON keeps only derived geometry and Blender-facing metadata:

* calibrated camera settings
* per-frame road and lane geometry
* tracked, scaled, and oriented semantic objects
* rendering defaults for the depth-shell stage

Raw depth maps are intentionally left on disk and referenced lazily by the
Blender renderer to keep the scene file compact.  By default the assembled
scene is written to ``output/<scene>/scene_data/scene_assembled.json`` and
mirrored to the legacy ``output/scene_data/<scene>/`` location.

Robustness strategy
-------------------
The assembler performs more than a simple format conversion.  It also:

* validates depth against bounding-box-implied size
* stabilizes tracks across time
* biases detections from visible surfaces toward plausible object centers
* suppresses implausible moving detections
* removes object placements that fall outside the reconstructed road corridor

Entry point
-----------
Run directly from the project root, for example::

    python scene_assembler.py --scene scene1 --view front --out output/scene.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


_THIS = Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS))

from calibration import (  # noqa: E402
    CalibData,
    export_blender_camera,
    load_calibration,
    pixel_to_ground,
    pixel_to_world,
)
from project_setup import (  # noqa: E402
    infer_scene_name,
    mirror_stage_output,
    scene_output_layout,
)


# ============================================================================
# Geometry / asset priors
# ============================================================================

DEFAULT_FPS = 15.0
MAX_VALID_DEPTH_M = 120.0
MAX_GROUND_DEPTH_M = 90.0
MAX_GROUND_LATERAL_M = 24.0
DEFAULT_DEPTH_SHELL_MIN_M = 2.5
DEFAULT_DEPTH_SHELL_TOP_CROP = 0.18
DEFAULT_DEPTH_SHELL_BOTTOM_CROP = 0.22
DEFAULT_DEPTH_SHELL_FOREGROUND_DEPTH_M = 3.8
DEFAULT_DEPTH_SHELL_FOREGROUND_ROW_FRAC = 0.72
DEFAULT_DEPTH_SHELL_FOREGROUND_BOOST_M = 4.0

# Real-world dimensions: (height, width, length) in metres
CLASS_DIMS: Dict[str, Tuple[float, float, float]] = {
    "car":            (1.50, 1.85, 4.50),
    "sedan":          (1.47, 1.82, 4.72),
    "hatchback":      (1.50, 1.78, 4.18),
    "suv":            (1.72, 1.94, 4.88),
    "pickup":         (1.82, 2.02, 5.35),
    "truck":          (3.40, 2.50, 8.00),
    "motorcycle":     (1.20, 0.82, 2.20),
    "bicycle":        (1.10, 0.50, 1.80),
    "pedestrian":     (1.75, 0.55, 0.45),
    "traffic_light":  (4.20, 0.45, 0.45),
    "traffic_sign":   (2.10, 0.65, 0.15),
    "stop_sign":      (2.20, 0.75, 0.15),
    "speed_limit":    (2.20, 0.65, 0.15),
    "dustbin":        (1.10, 0.75, 0.75),
    "traffic_pole":   (3.50, 0.20, 0.20),
    "traffic_cone":   (0.72, 0.38, 0.38),
    "traffic_cylinder": (1.00, 0.40, 0.40),
}

# Visible extents represented by the image-space bbox.  For elevated street
# furniture the detector usually sees the lamp head / sign face, not the full
# pole height, so the reference dimensions differ from the full asset size.
CLASS_BBOX_DIMS: Dict[str, Tuple[float, float, float]] = {
    **CLASS_DIMS,
    "traffic_light": (0.85, 0.38, 0.30),
    "traffic_sign": (0.75, 0.65, 0.12),
    "stop_sign": (0.78, 0.78, 0.12),
    "speed_limit": (0.75, 0.65, 0.12),
}

# Approximate world height of the visible bbox centre for elevated upright
# semantics.  This lets us lift the lamp/sign head first and then drop the
# asset base to the ground instead of incorrectly assuming the bbox bottom is
# the contact point on the road.
CLASS_VISIBLE_CENTER_HEIGHT_M: Dict[str, float] = {
    "traffic_light": 3.95,
    "traffic_sign": 2.00,
    "stop_sign": 2.10,
    "speed_limit": 2.00,
}

ASSET_LIBRARY: Dict[str, List[str]] = {
    "car": [
        "Vehicles/SedanAndHatchback.blend",
        "Vehicles/SUV.blend",
        "Vehicles/PickupTruck.blend",
    ],
    "sedan": ["Vehicles/SedanAndHatchback.blend"],
    "hatchback": ["Vehicles/SedanAndHatchback.blend"],
    "suv": ["Vehicles/SUV.blend"],
    "pickup": ["Vehicles/PickupTruck.blend"],
    "truck": ["Vehicles/Truck.blend"],
    "bus": ["Vehicles/Truck.blend"],
    "motorcycle": ["Vehicles/Motorcycle.blend"],
    "bicycle": ["Vehicles/Bicycle.blend"],
    "pedestrian": ["Pedestrain.blend"],
    "traffic_light": ["TrafficSignal.blend"],
    "traffic_sign": ["SpeedLimitSign.blend"],
    "stop_sign": ["StopSign.blend"],
    "speed_limit": ["SpeedLimitSign.blend"],
    "dustbin": ["Dustbin.blend"],
    "traffic_pole": ["TrafficAssets.blend"],
    "traffic_cone": ["TrafficConeAndCylinder.blend"],
    "traffic_cylinder": ["TrafficConeAndCylinder.blend"],
}

GROUND_ANCHORED_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_light",
    "traffic_sign",
    "stop_sign",
    "speed_limit",
    "dustbin",
    "traffic_pole",
    "traffic_cone",
    "traffic_cylinder",
}

STATIC_UPRIGHT_CLASSES = {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}
MOVING_CLASSES = {"car", "truck", "motorcycle", "bicycle", "pedestrian"}
ROAD_CONTEXT_CLASSES = {"car", "truck", "motorcycle", "bicycle"}
FLOW_YAW_CLASSES = {"car", "truck", "motorcycle", "bicycle"}
SUPPORTED_RENDERABLE_SIGNS = {"stop_sign", "speed_limit"}


@dataclass
class StandardDetection:
    det_id: Any
    raw_class: str
    canonical_class: str
    subtype: str
    bbox: List[int]
    confidence: float
    source: str
    depth_m: Optional[float] = None
    source_track_id: Optional[Any] = None
    source_yaw_rad: Optional[float] = None
    tl_color: Optional[str] = None
    tl_color_conf: Optional[float] = None
    tl_state: Optional[str] = None
    tl_shape: Optional[str] = None
    tl_decision_source: Optional[str] = None
    tl_detic_color_check: Optional[str] = None
    tl_detic_color_conf: Optional[float] = None
    tl_detic_color_agrees: Optional[bool] = None
    provided_position_blender: Optional[List[float]] = None
    speed_limit_value: Optional[Any] = None
    sign_label: Optional[str] = None


@dataclass
class TrackState:
    track_id: int
    frame_idx: int
    canonical_class: str
    subtype: str
    position: Tuple[float, float, float]
    bbox: List[int]
    depth_m: float
    source_track_id: Optional[Any] = None
    flow_track_id: Optional[Any] = None
    vehicle_3d_track_id: Optional[Any] = None


class LazyDepthStore:
    """Lazy `npz` wrapper so we do not expand thousands of depth maps in RAM."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path.resolve() if path else None
        self._npz: Optional[np.lib.npyio.NpzFile] = None
        self._keys: set[int] = set()

        if self.path and self.path.exists():
            self._npz = np.load(str(self.path), allow_pickle=False)
            self._keys = {
                int(k) for k in self._npz.files if str(k).isdigit()
            }

    def __bool__(self) -> bool:
        return self._npz is not None

    def keys(self) -> set[int]:
        return set(self._keys)

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._npz is None:
            return None
        key = str(int(frame_idx))
        if key not in self._npz.files:
            return None
        return self._npz[key]

    def sample_shape(self) -> Optional[Tuple[int, int]]:
        if self._npz is None or not self._npz.files:
            return None
        arr = self._npz[self._npz.files[0]]
        return int(arr.shape[0]), int(arr.shape[1])

    def close(self) -> None:
        if self._npz is not None:
            self._npz.close()
            self._npz = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def build_ego_vehicle_meta(view: str, calib: CalibData) -> Dict[str, Any]:
    """
    Describe the ego vehicle so Blender can render the host car and place the
    rest of the scene in a car-centric frame instead of a camera-centric one.

    The assembled detections remain in the historical camera-centric frame for
    compatibility.  Blender uses the exported mount offset to shift the camera
    and all reconstructed geometry into the ego-vehicle frame at render time.
    """

    view_key = str(view or "front").strip().lower()
    default_mounts: Dict[str, Tuple[float, float]] = {
        "front": (1.20, 0.00),
        "back": (-1.15, 0.00),
        "left": (0.15, 0.92),
        "right": (0.15, -0.92),
    }
    proxy_styles: Dict[str, str] = {
        "front": "hood",
        "back": "rear_deck",
        "left": "left_side",
        "right": "right_side",
    }

    mount_x, mount_y = default_mounts.get(view_key, default_mounts["front"])
    asset = choose_asset("car", "sedan", f"ego:{view_key}")

    return {
        "class": "car",
        "subtype": "sedan",
        "asset": asset,
        "proxy_style": proxy_styles.get(view_key, "hood"),
        "render_proxy_only": False,
        "color": "white",
        "dims_m": {
            "length": 4.65,
            "width": 1.85,
            "height": 1.52,
        },
        "origin_blender": [0.0, 0.0, 0.0],
        "camera_mount_blender": [round(float(mount_x), 4), round(float(mount_y), 4), round(float(calib.camera_height_m), 4)],
        "scene_world_offset_blender": [round(float(mount_x), 4), round(float(mount_y), 4), 0.0],
        "coordinate_frame": "ego_vehicle_base_center",
    }


class LazyFrameJsonStore:
    """Lazy loader for `frame_XXXXX.json` directories."""

    def __init__(self, frame_dir: Optional[Path]) -> None:
        self.frame_dir = frame_dir.resolve() if frame_dir else None
        self.path_map: Dict[int, Path] = {}
        self._cache: Dict[int, Dict[str, Any]] = {}

        if self.frame_dir and self.frame_dir.exists():
            for path in sorted(self.frame_dir.glob("frame_*.json")):
                try:
                    idx = int(path.stem.split("_")[-1])
                except ValueError:
                    continue
                self.path_map[idx] = path.resolve()

    def __bool__(self) -> bool:
        return bool(self.path_map)

    def keys(self) -> set[int]:
        return set(self.path_map.keys())

    def get(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        path = self.path_map.get(frame_idx)
        if path is None:
            return None
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"[assembler] WARNING — failed to parse {path.name}: {exc}")
            return None
        self._cache[frame_idx] = data
        return data


# ============================================================================
# Generic helpers
# ============================================================================

def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def clip_bbox(bbox: Sequence[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return [x1, y1, x2, y2]


def bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter) / float(union)


def resolve_path(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve())
    except FileNotFoundError:
        return str(path)


def first_existing_dir(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()
    return None


def frame_json_file_count(path: Optional[Path]) -> int:
    if path is None or not path.exists() or not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("frame_*.json"))


def best_frame_json_dir(candidates: Sequence[Path]) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_count = 0
    fallback = first_existing_dir(candidates)

    for path in candidates:
        count = frame_json_file_count(path)
        if count > best_count:
            best_path = path.resolve()
            best_count = count

    if best_path is not None:
        return best_path
    return fallback


def first_match(patterns: Sequence[str], bases: Sequence[Path]) -> Optional[Path]:
    for base in bases:
        if not base.exists():
            continue
        for pattern in patterns:
            matches = sorted(base.glob(pattern))
            if matches:
                return matches[0].resolve()
    return None


def canonicalize_class(raw_label: str) -> Tuple[str, str]:
    label = str(raw_label or "unknown").strip().lower().replace(" ", "_")
    aliases = {
        "person": ("pedestrian", "pedestrian"),
        "pedestrian": ("pedestrian", "pedestrian"),
        "car": ("car", "car"),
        "sedan": ("car", "car"),
        "hatchback": ("car", "car"),
        "suv": ("car", "car"),
        "pickup": ("car", "car"),
        "pickup_truck": ("car", "car"),
        "truck": ("truck", "truck"),
        "bus": ("truck", "truck"),
        "van": ("truck", "truck"),
        "motorcycle": ("motorcycle", "motorcycle"),
        "bike": ("bicycle", "bicycle"),
        "bicycle": ("bicycle", "bicycle"),
        "traffic_light": ("traffic_light", "traffic_light"),
        "trafficlight": ("traffic_light", "traffic_light"),
        "traffic_sign": ("traffic_sign", "traffic_sign"),
        "stop_sign": ("stop_sign", "stop_sign"),
        "speed_limit": ("speed_limit", "speed_limit"),
        "speed_limit_sign": ("speed_limit", "speed_limit"),
        "dustbin": ("dustbin", "dustbin"),
        "traffic_pole": ("traffic_pole", "traffic_pole"),
        "traffic_cone": ("traffic_cone", "traffic_cone"),
        "traffic_cylinder": ("traffic_cylinder", "traffic_cylinder"),
    }
    return aliases.get(label, (label, label))


def normalise_sign_label(raw_label: Any) -> str:
    label = str(raw_label or "").strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label


def canonicalize_sign_detection(
    canonical_class: str,
    subtype: str,
    raw_label: Any,
    sign_label: Any,
    speed_limit_value: Any = None,
) -> Tuple[str, str, Optional[str]]:
    if canonical_class not in {"traffic_sign", "stop_sign", "speed_limit"}:
        return canonical_class, subtype, None

    hint = normalise_sign_label(sign_label or raw_label or subtype or canonical_class)
    if canonical_class == "stop_sign" or "stop" in hint:
        return "stop_sign", "stop_sign", "stop_sign"
    if canonical_class == "speed_limit" or speed_limit_value is not None:
        return "speed_limit", "speed_limit", "speed_limit"
    if any(token in hint for token in ("speed_limit", "speedlimit", "mph", "kmh", "limit_")):
        return "speed_limit", "speed_limit", hint or "speed_limit"
    return "traffic_sign", (hint or "traffic_sign"), (hint or None)


def is_supported_renderable_sign(canonical_class: str, subtype: str, sign_label: Optional[str]) -> bool:
    if canonical_class in SUPPORTED_RENDERABLE_SIGNS:
        return True
    norm = normalise_sign_label(sign_label or subtype or canonical_class)
    return norm in SUPPORTED_RENDERABLE_SIGNS


def get_class_dims(canonical_class: str, subtype: str) -> Tuple[float, float, float]:
    if subtype in CLASS_DIMS:
        return CLASS_DIMS[subtype]
    if canonical_class in CLASS_DIMS:
        return CLASS_DIMS[canonical_class]
    return CLASS_DIMS["car"]


def get_bbox_reference_dims(canonical_class: str, subtype: str) -> Tuple[float, float, float]:
    """Return the physical extent that the detector bbox most closely represents."""
    if subtype in CLASS_BBOX_DIMS:
        return CLASS_BBOX_DIMS[subtype]
    if canonical_class in CLASS_BBOX_DIMS:
        return CLASS_BBOX_DIMS[canonical_class]
    return get_class_dims(canonical_class, subtype)


def get_visible_center_height(canonical_class: str, subtype: str, scale: float = 1.0) -> float:
    """Approximate the world height of the visible semantic centre above the ground."""
    if subtype in CLASS_VISIBLE_CENTER_HEIGHT_M:
        base_height = float(CLASS_VISIBLE_CENTER_HEIGHT_M[subtype])
        return base_height if subtype in STATIC_UPRIGHT_CLASSES else base_height * float(scale)
    if canonical_class in CLASS_VISIBLE_CENTER_HEIGHT_M:
        base_height = float(CLASS_VISIBLE_CENTER_HEIGHT_M[canonical_class])
        return base_height if canonical_class in STATIC_UPRIGHT_CLASSES else base_height * float(scale)

    asset_h, _, _ = get_class_dims(canonical_class, subtype)
    return 0.5 * float(asset_h) * float(scale)


def choose_asset(canonical_class: str, subtype: str, track_seed: str) -> str:
    candidates = (
        ASSET_LIBRARY.get(subtype)
        or ASSET_LIBRARY.get(canonical_class)
        or [""]
    )
    if len(candidates) == 1:
        return candidates[0]
    return candidates[stable_hash(track_seed) % len(candidates)]


def choose_stable_vehicle_subtype(
    canonical_class: str,
    detections: Sequence[Dict[str, Any]],
) -> Optional[str]:
    if canonical_class != "car":
        return None

    scores: Dict[str, float] = {"sedan": 0.0, "suv": 0.0, "hatchback": 0.0, "pickup": 0.0}
    for det in detections:
        subtype = str(det.get("subclass", "")).strip().lower()
        if subtype not in scores:
            continue
        confidence = float(det.get("confidence", 1.0) or 1.0)
        scores[subtype] += max(0.05, confidence)

    best_subtype, best_score = max(scores.items(), key=lambda item: item[1])
    return best_subtype if best_score > 0.0 else None


def blender_from_fused_position(values: Optional[Sequence[float]]) -> Optional[List[float]]:
    if not values:
        return None
    if len(values) < 2:
        return None
    lateral = float(values[0])
    depth = float(values[1])
    height = float(values[2]) if len(values) > 2 else 0.0
    return [round(depth, 4), round(-lateral, 4), round(max(0.0, height), 4)]


def keep_ground_point(
    bx: float,
    by: float,
    bz: float = 0.0,
    *,
    max_depth_m: float = MAX_GROUND_DEPTH_M,
    max_lateral_m: float = MAX_GROUND_LATERAL_M,
) -> Optional[List[float]]:
    if not (math.isfinite(bx) and math.isfinite(by) and math.isfinite(bz)):
        return None
    if bx < 0.35 or bx > max_depth_m:
        return None
    if abs(by) > max_lateral_m:
        return None
    if bz < -1.0 or bz > 4.0:
        return None
    return [round(float(bx), 4), round(float(by), 4), round(float(bz), 4)]


def fused_ground_points_to_blender(points: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for pt in points:
        if len(pt) < 2:
            continue
        lateral = float(pt[0])
        depth = float(pt[1])
        keep = keep_ground_point(depth, -lateral, 0.0)
        if keep is not None:
            out.append(keep)
    return out


def build_frame_index(frames: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        if "frame_idx" not in frame:
            continue
        try:
            idx = int(frame["frame_idx"])
        except (TypeError, ValueError):
            continue
        index[idx] = frame
    return index


def load_json(path: Optional[Path], label: str) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        print(f"[assembler] WARNING — {label} not found; skipping.")
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[assembler] ERROR loading {label} from {path}: {exc}")
        return None
    frame_count = len(data.get("frames", [])) if isinstance(data, dict) else 0
    print(f"[assembler] {label:<18} loaded ({frame_count:4d} frames) ← {path}")
    return data


def build_flow_index(flow_data: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index optical-flow records by the current frame index."""
    if not flow_data:
        return {}

    index: Dict[int, Dict[str, Any]] = {}
    for frame in flow_data.get("frames", []):
        frame_idx = frame.get("frame_idx")
        if frame_idx is None and frame.get("pair_idx") is not None:
            frame_idx = int(frame.get("pair_idx", 0)) + 1
        try:
            key = int(frame_idx)
        except (TypeError, ValueError):
            continue
        index[key] = frame
    return index


def match_flow_to_detection(
    std_det: StandardDetection,
    flow_frame: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Match one detection to a per-object flow record using id first, then IoU."""
    if not flow_frame:
        return None

    object_flows = flow_frame.get("object_flows", [])
    if not object_flows:
        return None

    if std_det.det_id is not None:
        for flow_obj in object_flows:
            if flow_obj.get("det_id") == std_det.det_id:
                return flow_obj

    best_match: Optional[Dict[str, Any]] = None
    best_iou = 0.0
    for flow_obj in object_flows:
        flow_bbox = flow_obj.get("bbox")
        if not isinstance(flow_bbox, list) or len(flow_bbox) < 4:
            continue
        iou = bbox_iou(std_det.bbox, flow_bbox[:4])
        if iou > best_iou:
            best_iou = iou
            best_match = flow_obj
    return best_match if best_iou >= 0.55 else None


def infer_video_fps(video_path: Optional[Path]) -> Optional[float]:
    if video_path is None or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps > 1.0:
        return fps
    return None


def infer_fps(
    video_path: Optional[Path],
    det_data: Optional[Dict[str, Any]],
    tl_data: Optional[Dict[str, Any]],
    lane_data: Optional[Dict[str, Any]],
) -> float:
    for data in (det_data, tl_data, lane_data):
        frames = (data or {}).get("frames", [])
        timestamps = [
            float(frame.get("timestamp_s"))
            for frame in frames
            if frame.get("timestamp_s") is not None
        ]
        if len(timestamps) < 3:
            continue
        diffs = [
            b - a
            for a, b in zip(timestamps[:-1], timestamps[1:])
            if b > a
        ]
        if diffs:
            median_dt = statistics.median(diffs)
            if median_dt > 0.0:
                return max(1.0, 1.0 / median_dt)

    fps = infer_video_fps(video_path)
    if fps is not None:
        return fps

    return DEFAULT_FPS


def median_bbox_depth(
    bbox: Sequence[int],
    depth_map: np.ndarray,
    inner_frac: float = 0.45,
) -> Optional[float]:
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return None

    margin_x = int((x2 - x1) * (1.0 - inner_frac) * 0.5)
    margin_y = int((y2 - y1) * (1.0 - inner_frac) * 0.5)
    ix1, iy1 = x1 + margin_x, y1 + margin_y
    ix2, iy2 = x2 - margin_x, y2 - margin_y
    if ix2 <= ix1:
        ix1, ix2 = x1, x2
    if iy2 <= iy1:
        iy1, iy2 = y1, y2

    patch = depth_map[iy1:iy2, ix1:ix2].astype(np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0.1) & (patch < MAX_VALID_DEPTH_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_to_world_at_height(
    u: float,
    v: float,
    world_height_m: float,
    calib: CalibData,
) -> Optional[Tuple[float, float, float]]:
    """
    Intersect the pixel ray with a horizontal plane at ``world_height_m``.

    ``world_height_m`` is expressed in the same world frame used by
    ``pixel_to_ground`` where the road plane is ``Y_world = 0`` and the camera
    sits at ``calib.camera_height_m``.
    """
    rx = (float(u) - calib.cx) / calib.fx
    ry = (float(v) - calib.cy) / calib.fy
    rz = 1.0

    cp = math.cos(calib.pitch_rad)
    sp = math.sin(calib.pitch_rad)
    ray_yw = -sp * rz - cp * ry

    if abs(ray_yw) < 1e-9:
        return None

    t = (float(world_height_m) - float(calib.camera_height_m)) / ray_yw
    if t <= 0.0:
        return None

    return rx * t, ry * t, rz * t


def camera_point_to_blender_world(
    X: float,
    Y: float,
    Z: float,
    calib: CalibData,
) -> Tuple[float, float, float]:
    """Camera-space point to Blender/world coordinates with the ground at z=0."""
    return (
        float(Z),
        -float(X),
        max(0.0, float(calib.camera_height_m) - float(Y)),
    )


def blender_world_to_camera_point(
    bx: float,
    by: float,
    bz: float,
    calib: CalibData,
) -> Tuple[float, float, float]:
    """Inverse of ``camera_point_to_blender_world``."""
    return (
        -float(by),
        float(calib.camera_height_m) - float(bz),
        float(bx),
    )


def project_blender_point_to_pixel(
    position_blender: Sequence[float],
    calib: CalibData,
) -> Optional[Tuple[float, float, float]]:
    """Project a Blender-space point back into the calibrated image."""
    if len(position_blender) < 3:
        return None
    cam_x, cam_y, cam_z = blender_world_to_camera_point(
        float(position_blender[0]),
        float(position_blender[1]),
        float(position_blender[2]),
        calib,
    )
    if cam_z <= 1e-6:
        return None
    u = calib.fx * cam_x / cam_z + calib.cx
    v = calib.fy * cam_y / cam_z + calib.cy
    return float(u), float(v), float(cam_z)


def min_plausible_depth_for_bbox(
    bbox_w: int,
    bbox_h: int,
    canonical_class: str,
    subtype: str,
    calib: CalibData,
    factor: float = 0.65,
) -> Optional[float]:
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    known_h, known_w, _ = get_bbox_reference_dims(canonical_class, subtype)
    candidates: List[float] = []

    if bbox_h > 0:
        candidates.append(float(factor) * known_h * calib.fy / max(float(bbox_h), 1.0))
    if bbox_w > 0:
        candidates.append(float(factor) * known_w * calib.fx / max(float(bbox_w), 1.0))

    if not candidates:
        return None
    return float(min(candidates))


def moving_detection_is_implausible(
    canonical_class: str,
    bbox_w: int,
    bbox_h: int,
    confidence: float,
    real_h_est: float,
    real_w_est: float,
    known_h: float,
    known_w: float,
) -> bool:
    if canonical_class not in MOVING_CLASSES:
        return False

    aspect = float(bbox_w) / max(float(bbox_h), 1.0)
    area = int(bbox_w * bbox_h)
    size_ratio_h = real_h_est / max(known_h, 1e-6)
    size_ratio_w = real_w_est / max(known_w, 1e-6)

    if canonical_class in {"car", "truck", "bus"} and aspect > 5.5:
        return True
    if confidence < 0.62 and aspect > 5.0:
        return True
    if confidence < 0.58 and area < 3500:
        return True
    if bbox_h < 20 and aspect > 4.2:
        return True
    if max(size_ratio_h, size_ratio_w) < 0.22 and area < 9000:
        return True
    return False


def lift_bbox_to_3d(
    bbox: Sequence[int],
    depth_m: Optional[float],
    canonical_class: str,
    subtype: str,
    calib: CalibData,
    frame_w: int,
    frame_h: int,
    provided_position_blender: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    bbox = clip_bbox(bbox, frame_w, frame_h)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None

    asset_h, asset_w, asset_l = get_class_dims(canonical_class, subtype)
    bbox_ref_h, bbox_ref_w, bbox_ref_l = get_bbox_reference_dims(canonical_class, subtype)
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    cx = 0.5 * (x1 + x2)
    bottom_v = float(y2)
    center_v = 0.5 * (y1 + y2)

    bx = by = bz = None
    visible_center_blender: Optional[List[float]] = None
    anchor_source = "none"
    anchor_depth = float(depth_m) if depth_m is not None else None
    bbox_depth_floor = min_plausible_depth_for_bbox(
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        canonical_class=canonical_class,
        subtype=subtype,
        calib=calib,
    )
    bbox_nominal_depth = min_plausible_depth_for_bbox(
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        canonical_class=canonical_class,
        subtype=subtype,
        calib=calib,
        factor=1.0,
    )

    if (
        anchor_depth is not None
        and canonical_class in MOVING_CLASSES
        and bbox_depth_floor is not None
        and anchor_depth < bbox_depth_floor
    ):
        anchor_depth = bbox_depth_floor

    if canonical_class in {"traffic_sign", "stop_sign", "speed_limit"}:
        sign_depth = anchor_depth
        if bbox_nominal_depth is not None:
            sign_depth = max(float(anchor_depth or 0.0), 0.9 * float(bbox_nominal_depth))
        if sign_depth is not None and 0.1 < float(sign_depth) < MAX_VALID_DEPTH_M:
            Xc, Yc, Zc = pixel_to_world(cx, center_v, float(sign_depth), calib)
            cx_b, cy_b, cz_b = camera_point_to_blender_world(Xc, Yc, Zc, calib)
            visible_center_blender = [round(cx_b, 4), round(cy_b, 4), round(cz_b, 4)]
            bx, by, _ = cx_b, cy_b, cz_b
            bz = 0.0
            anchor_depth = float(sign_depth)
            anchor_source = "sign_center_depth"

    if bx is None and canonical_class == "traffic_light":
        tl_depth = anchor_depth
        if bbox_nominal_depth is not None:
            nominal = float(bbox_nominal_depth)
            if tl_depth is None or not (0.1 < float(tl_depth) < MAX_VALID_DEPTH_M):
                tl_depth = nominal
            else:
                raw_depth = float(tl_depth)
                rel_gap = abs(raw_depth - nominal) / max(nominal, 1.0)
                if rel_gap <= 0.40:
                    tl_depth = 0.55 * raw_depth + 0.45 * nominal
                else:
                    tl_depth = max(raw_depth, 0.90 * nominal)
        if tl_depth is not None and 0.8 < float(tl_depth) < MAX_VALID_DEPTH_M:
            Xc, Yc, Zc = pixel_to_world(cx, center_v, float(tl_depth), calib)
            cx_b, cy_b, cz_b = camera_point_to_blender_world(Xc, Yc, Zc, calib)
            visible_center_blender = [round(cx_b, 4), round(cy_b, 4), round(cz_b, 4)]
            bx, by, _ = cx_b, cy_b, cz_b
            bz = 0.0
            anchor_depth = float(tl_depth)
            anchor_source = "traffic_light_depth"

    if bx is None and canonical_class in STATIC_UPRIGHT_CLASSES:
        head_center_height = get_visible_center_height(canonical_class, subtype)
        head_hit = pixel_to_world_at_height(cx, center_v, head_center_height, calib)
        if head_hit is not None and 0.8 < float(head_hit[2]) < MAX_VALID_DEPTH_M:
            cx_b, cy_b, cz_b = camera_point_to_blender_world(*head_hit, calib)
            visible_center_blender = [round(cx_b, 4), round(cy_b, 4), round(cz_b, 4)]
            bx, by, _ = cx_b, cy_b, cz_b
            bz = 0.0
            anchor_depth = float(head_hit[2])
            anchor_source = "height_plane"

    if bx is None and anchor_depth is not None and 0.1 < anchor_depth < MAX_VALID_DEPTH_M:
        Xc, Yc, Zc = pixel_to_world(cx, bottom_v, anchor_depth, calib)
        bx, by, bz = camera_point_to_blender_world(Xc, Yc, Zc, calib)
        if canonical_class in GROUND_ANCHORED_CLASSES:
            bz = 0.0
        anchor_source = "bbox_depth_corrected" if depth_m is not None and anchor_depth != float(depth_m) else "bbox_depth"

    if bx is None and canonical_class in GROUND_ANCHORED_CLASSES:
        ground_hit = pixel_to_ground(cx, bottom_v, calib)
        if ground_hit is not None:
            bx, by, _ = camera_point_to_blender_world(*ground_hit, calib)
            bz = 0.0
            anchor_depth = float(ground_hit[2])
            anchor_source = "ground_plane"

    if bx is None and provided_position_blender is not None and len(provided_position_blender) >= 3:
        bx = float(provided_position_blender[0])
        by = float(provided_position_blender[1])
        bz = 0.0 if canonical_class in GROUND_ANCHORED_CLASSES else float(provided_position_blender[2])
        anchor_depth = float(provided_position_blender[0])
        anchor_source = "provided_position"

    if bx is None or by is None or bz is None:
        return None

    depth_for_scale = float(anchor_depth or depth_m or 0.0)
    if depth_for_scale <= 0.1:
        depth_for_scale = max(asset_l, 1.0)

    real_h_est = bbox_h * depth_for_scale / max(1e-6, calib.fy)
    real_w_est = bbox_w * depth_for_scale / max(1e-6, calib.fx)
    scale_from_h = real_h_est / max(bbox_ref_h, 1e-6)
    scale_from_w = real_w_est / max(bbox_ref_w, 1e-6)

    plausible_scales: List[float] = []
    if 0.45 * bbox_ref_h <= real_h_est <= 2.25 * bbox_ref_h:
        plausible_scales.append(scale_from_h)
    if 0.45 * bbox_ref_w <= real_w_est <= 2.25 * bbox_ref_w:
        plausible_scales.append(scale_from_w)

    if canonical_class in {"car", "truck", "bus"}:
        lo, hi = 0.65, 1.25
    elif canonical_class in {"motorcycle", "bicycle"}:
        lo, hi = 0.60, 1.20
    elif canonical_class == "pedestrian":
        lo, hi = 0.85, 1.20
    elif canonical_class == "traffic_light":
        lo, hi = 0.85, 1.15
    elif canonical_class in {"traffic_sign", "stop_sign", "speed_limit"}:
        lo, hi = 0.85, 1.25
    elif canonical_class in STATIC_UPRIGHT_CLASSES:
        lo, hi = 0.75, 1.35
    else:
        lo, hi = 0.75, 1.50

    scale = statistics.mean(plausible_scales) if plausible_scales else 1.0
    scale = float(np.clip(scale, lo, hi))

    return {
        "position_blender": [round(bx, 4), round(by, 4), round(bz, 4)],
        "visible_center_blender": visible_center_blender,
        "depth_m": round(depth_for_scale, 4),
        "scale": round(scale, 4),
        "real_h_est": round(real_h_est, 4),
        "real_w_est": round(real_w_est, 4),
        "dims_m": [round(asset_h, 3), round(asset_w, 3), round(asset_l, 3)],
        "bbox_dims_m": [round(bbox_ref_h, 3), round(bbox_ref_w, 3), round(bbox_ref_l, 3)],
        "anchor_source": anchor_source,
        "bbox_depth_floor_m": round(float(bbox_depth_floor), 4) if bbox_depth_floor is not None else None,
    }


def projection_closure_metrics(
    det: Dict[str, Any],
    calib: CalibData,
) -> Optional[Dict[str, float]]:
    """Project the lifted detection back into image space and compare with its bbox."""
    bbox = det.get("bbox", [])
    if len(bbox) < 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bbox_w = max(1.0, x2 - x1)
    bbox_h = max(1.0, y2 - y1)
    cls = str(det.get("class", "unknown"))
    subtype = str(det.get("subclass", cls))
    scale = float(det.get("scale", 1.0))
    visible_h, visible_w, _ = get_bbox_reference_dims(cls, subtype)
    position = det.get("position_blender", [0.0, 0.0, 0.0])
    if len(position) < 3:
        return None

    visible_center = det.get("visible_center_blender")
    if isinstance(visible_center, list) and len(visible_center) >= 3:
        projection_point = [float(visible_center[0]), float(visible_center[1]), float(visible_center[2])]
    else:
        visible_center_z = float(position[2]) + get_visible_center_height(cls, subtype, scale=scale)
        projection_point = [float(position[0]), float(position[1]), visible_center_z]
    projection = project_blender_point_to_pixel(projection_point, calib)
    if projection is None:
        return None

    u_proj, v_proj, depth_proj = projection
    if depth_proj <= 1e-6:
        return None

    expected_h_px = float(visible_h) * scale * calib.fy / depth_proj
    expected_w_px = float(visible_w) * scale * calib.fx / depth_proj

    return {
        "u_proj": float(u_proj),
        "v_proj": float(v_proj),
        "depth_proj": float(depth_proj),
        "expected_h_px": float(expected_h_px),
        "expected_w_px": float(expected_w_px),
        "h_ratio": bbox_h / max(expected_h_px, 1e-6),
        "w_ratio": bbox_w / max(expected_w_px, 1e-6),
        "center_dx_norm": abs(float(0.5 * (x1 + x2)) - float(u_proj)) / bbox_w,
        "center_dy_norm": abs(float(0.5 * (y1 + y2)) - float(v_proj)) / bbox_h,
    }


def projection_closure_error(closure: Optional[Dict[str, float]]) -> float:
    if closure is None:
        return float("inf")
    h_ratio = max(float(closure.get("h_ratio", 1.0)), 1e-6)
    w_ratio = max(float(closure.get("w_ratio", 1.0)), 1e-6)
    return (
        1.55 * abs(math.log(h_ratio))
        + 1.30 * abs(math.log(w_ratio))
        + 1.40 * float(closure.get("center_dx_norm", 0.0))
        + 1.65 * float(closure.get("center_dy_norm", 0.0))
    )


def evaluate_position_candidate(
    det: Dict[str, Any],
    candidate_position: Sequence[float],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
    candidate_depth: Optional[float] = None,
    candidate_dims: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    candidate_det = dict(det)
    candidate_det["position_blender"] = [round(float(v), 4) for v in candidate_position[:3]]
    candidate_det.pop("visible_center_blender", None)
    if candidate_depth is not None:
        candidate_det["depth_m"] = round(float(candidate_depth), 4)
    if candidate_dims is not None and len(candidate_dims) >= 3:
        candidate_det["dims_m"] = [round(float(v), 3) for v in candidate_dims[:3]]

    closure = projection_closure_metrics(candidate_det, calib)
    score = projection_closure_error(closure)
    implausible = detection_is_implausible_after_lift(
        candidate_det,
        calib=calib,
        frame_w=frame_w,
        frame_h=frame_h,
        track_len=1,
    )
    return {
        "det": candidate_det,
        "closure": closure,
        "score": float(score),
        "implausible": bool(implausible),
    }


def reconcile_vehicle_3d_position(
    det: Dict[str, Any],
    v3d_position: Sequence[float],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
    v3d_depth: Optional[float] = None,
    v3d_dims: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    if len(v3d_position) < 3:
        return None
    lifted_position = det.get("lifted_position_blender") or det.get("position_blender")
    if not isinstance(lifted_position, list) or len(lifted_position) < 3:
        return None

    lifted_depth = det.get("lifted_depth_m", det.get("depth_m"))
    lifted_eval = evaluate_position_candidate(
        det,
        lifted_position,
        calib=calib,
        frame_w=frame_w,
        frame_h=frame_h,
        candidate_depth=float(lifted_depth) if lifted_depth is not None else None,
        candidate_dims=det.get("dims_m"),
    )
    v3d_eval = evaluate_position_candidate(
        det,
        v3d_position,
        calib=calib,
        frame_w=frame_w,
        frame_h=frame_h,
        candidate_depth=float(v3d_depth) if v3d_depth is not None else float(v3d_position[0]),
        candidate_dims=v3d_dims or det.get("dims_m"),
    )

    lifted_pos = [float(v) for v in lifted_position[:3]]
    v3d_pos = [float(v) for v in v3d_position[:3]]
    depth_ref = max(0.5, float(lifted_depth) if lifted_depth is not None else float(lifted_pos[0]))
    depth_gap = abs(v3d_pos[0] - lifted_pos[0])
    lateral_gap = abs(v3d_pos[1] - lifted_pos[1])
    rel_depth_gap = depth_gap / max(depth_ref, 1.0)
    lateral_limit = max(0.9, 0.10 * max(v3d_pos[0], lifted_pos[0]) + 0.25)
    depth_limit = max(2.5, 0.18 * depth_ref + 0.75)

    choose_v3d = False
    blend = False
    if not v3d_eval["implausible"]:
        if lifted_eval["implausible"]:
            choose_v3d = True
        elif rel_depth_gap <= 0.34 and lateral_gap <= 1.15 * lateral_limit:
            if v3d_eval["score"] + 0.08 < lifted_eval["score"]:
                choose_v3d = True
            elif (
                depth_gap <= depth_limit
                and lateral_gap <= lateral_limit
                and abs(v3d_eval["score"] - lifted_eval["score"]) <= 0.45
            ):
                blend = True

    if choose_v3d:
        chosen_pos = [round(float(v), 4) for v in v3d_pos]
        chosen_depth = round(float(v3d_depth if v3d_depth is not None else v3d_pos[0]), 4)
        return {
            "position_blender": chosen_pos,
            "depth_m": chosen_depth,
            "position_source": "vehicle_3d_validated",
            "projection_closure": v3d_eval["closure"],
            "position_consistency": {
                "lifted_score": round(float(lifted_eval["score"]), 4),
                "vehicle_3d_score": round(float(v3d_eval["score"]), 4),
                "depth_gap_m": round(float(depth_gap), 4),
                "lateral_gap_m": round(float(lateral_gap), 4),
                "mode": "vehicle_3d",
            },
        }

    if blend:
        lifted_score = max(float(lifted_eval["score"]), 1e-4)
        v3d_score = max(float(v3d_eval["score"]), 1e-4)
        weight_v3d = lifted_score / (lifted_score + v3d_score)
        weight_v3d = float(np.clip(weight_v3d, 0.35, 0.72))
        blended = [
            round((1.0 - weight_v3d) * lifted_pos[i] + weight_v3d * v3d_pos[i], 4)
            for i in range(3)
        ]
        blended_depth = round((1.0 - weight_v3d) * float(lifted_pos[0]) + weight_v3d * float(v3d_pos[0]), 4)
        blended_eval = evaluate_position_candidate(
            det,
            blended,
            calib=calib,
            frame_w=frame_w,
            frame_h=frame_h,
            candidate_depth=blended_depth,
            candidate_dims=v3d_dims or det.get("dims_m"),
        )
        if not blended_eval["implausible"]:
            return {
                "position_blender": blended,
                "depth_m": blended_depth,
                "position_source": "vehicle_3d_blend",
                "projection_closure": blended_eval["closure"],
                "position_consistency": {
                    "lifted_score": round(float(lifted_eval["score"]), 4),
                    "vehicle_3d_score": round(float(v3d_eval["score"]), 4),
                    "blended_score": round(float(blended_eval["score"]), 4),
                    "depth_gap_m": round(float(depth_gap), 4),
                    "lateral_gap_m": round(float(lateral_gap), 4),
                    "mode": "blend",
                },
            }

    return {
        "position_blender": [round(float(v), 4) for v in lifted_pos],
        "depth_m": round(float(lifted_depth) if lifted_depth is not None else float(lifted_pos[0]), 4),
        "position_source": "bbox_depth_validated",
        "projection_closure": lifted_eval["closure"],
        "position_consistency": {
            "lifted_score": round(float(lifted_eval["score"]), 4),
            "vehicle_3d_score": round(float(v3d_eval["score"]), 4),
            "depth_gap_m": round(float(depth_gap), 4),
            "lateral_gap_m": round(float(lateral_gap), 4),
            "mode": "bbox_depth",
        },
    }


def detection_is_implausible_after_lift(
    det: Dict[str, Any],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
    track_len: int = 1,
) -> bool:
    """Class-aware plausibility gate applied after a 3-D placement has been estimated."""
    bbox = det.get("bbox", [])
    if len(bbox) < 4:
        return True

    cls = str(det.get("class", "unknown"))
    subtype = str(det.get("subclass", cls))
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    position = det.get("position_blender", [0.0, 0.0, 0.0])
    if len(position) < 3:
        return True

    x_m = float(position[0])
    y_m = float(position[1])
    scale = float(det.get("scale", 1.0))
    closure = projection_closure_metrics(det, calib)
    if closure is None:
        return True

    touches_top = y1 <= max(2, int(round(0.01 * frame_h)))
    touches_side = x1 <= 2 or x2 >= (frame_w - 2)
    bbox_cx = 0.5 * (x1 + x2)
    edge_frac = min(float(bbox_cx) / max(float(frame_w), 1.0), float(frame_w - bbox_cx) / max(float(frame_w), 1.0))
    h_ratio = float(closure["h_ratio"])
    w_ratio = float(closure["w_ratio"])
    center_dx = float(closure["center_dx_norm"])
    center_dy = float(closure["center_dy_norm"])
    visible_h, visible_w, _ = get_bbox_reference_dims(cls, subtype)

    if cls in MOVING_CLASSES:
        if h_ratio < 0.18 and w_ratio < 0.18:
            return True
        if center_dx > 1.4 and center_dy > 1.6:
            return True
        return False

    if cls in STATIC_UPRIGHT_CLASSES:
        if not (0.20 <= h_ratio <= 4.0):
            return True
        if not (0.12 <= w_ratio <= 5.0):
            return True
        max_center_dx = 1.10
        max_center_dy = 1.35
        if cls == "traffic_light":
            if track_len >= 4:
                max_center_dx = 1.45
                max_center_dy = 2.85
            else:
                max_center_dx = 1.20
                max_center_dy = 1.90
        elif cls in {"traffic_sign", "stop_sign", "speed_limit"}:
            if track_len >= 4:
                max_center_dx = 1.35
                max_center_dy = 2.10
            else:
                max_center_dx = 1.20
                max_center_dy = 1.75
        if center_dx > max_center_dx or center_dy > max_center_dy:
            return True
        if touches_top and x_m < 10.0:
            return True
        if cls == "traffic_light":
            if bbox_h < 12 and x_m < 35.0:
                return True
            if bbox_h < 22 and x_m < 18.0:
                return True
            if touches_top and x_m < 18.0 and center_dy > 0.55:
                return True
            if touches_side and x_m < 8.0:
                return True
            if bbox_h < 28 and edge_frac < 0.10 and abs(y_m) > max(10.0, 0.18 * max(x_m, 1.0)):
                return True
            if abs(y_m) > max(18.0, 0.32 * max(x_m, 1.0)):
                return True
            if track_len < 3 and bbox_h < 18 and edge_frac < 0.08:
                return True
        if cls in {"traffic_sign", "stop_sign", "speed_limit"}:
            if bbox_h < 10 and x_m < 25.0:
                return True
        if track_len < 2 and (touches_top or h_ratio < 0.35 or w_ratio < 0.20):
            return True

        visible_center_height = get_visible_center_height(cls, subtype, scale=scale)
        if visible_center_height < 0.8 * visible_h:
            return True
        if abs(y_m) > max(28.0, 2.2 * x_m):
            return True
        return False

    return False


def static_semantic_is_grossly_implausible(
    det: Dict[str, Any],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
) -> bool:
    """
    Lightweight prefilter for upright semantics before track lengths are known.

    The detailed upright-semantic gate is applied later by
    ``gate_stationary_semantics`` after track association.  At assembly time we
    only reject obviously impossible placements so that stable long-lived signs
    and traffic lights are not dropped too early.
    """
    bbox = det.get("bbox", [])
    position = det.get("position_blender", [])
    if len(bbox) < 4 or len(position) < 3:
        return True

    cls = str(det.get("class", "unknown"))
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    bbox_h = max(1, y2 - y1)
    x_m = float(position[0])
    y_m = float(position[1])
    closure = projection_closure_metrics(det, calib)
    if closure is None:
        return True

    h_ratio = float(closure["h_ratio"])
    w_ratio = float(closure["w_ratio"])
    center_dx = float(closure["center_dx_norm"])
    center_dy = float(closure["center_dy_norm"])
    touches_top = y1 <= max(2, int(round(0.01 * frame_h)))
    touches_side = x1 <= 2 or x2 >= (frame_w - 2)
    bbox_cx = 0.5 * (x1 + x2)
    edge_frac = min(float(bbox_cx) / max(float(frame_w), 1.0), float(frame_w - bbox_cx) / max(float(frame_w), 1.0))

    if not (0.12 <= h_ratio <= 6.0):
        return True
    if not (0.08 <= w_ratio <= 6.5):
        return True
    if center_dx > 2.5 or center_dy > 3.1:
        return True

    if cls == "traffic_light":
        if bbox_h < 10 and x_m < 25.0:
            return True
        if touches_top and x_m < 8.0:
            return True
        if touches_side and x_m < 5.0:
            return True
        if bbox_h < 24 and edge_frac < 0.10 and abs(y_m) > max(8.0, 0.16 * max(x_m, 1.0)):
            return True
        if abs(y_m) > max(16.0, 0.28 * max(x_m, 1.0)):
            return True
    elif cls in {"traffic_sign", "stop_sign", "speed_limit"}:
        if bbox_h < 8 and x_m < 18.0:
            return True

    if abs(y_m) > max(34.0, 2.7 * max(x_m, 1.0)):
        return True
    return False


def lift_lane_points_to_3d(
    points_2d: Sequence[Sequence[float]],
    calib: CalibData,
    depth_map: Optional[np.ndarray],
) -> List[List[float]]:
    out: List[List[float]] = []
    width = int(depth_map.shape[1]) if depth_map is not None else 1280
    height = int(depth_map.shape[0]) if depth_map is not None else 720

    for pt in points_2d:
        if len(pt) < 2:
            continue
        u = float(pt[0])
        v = float(pt[1])

        ground_hit = pixel_to_ground(u, v, calib)
        if ground_hit is not None:
            bx, by, _ = camera_point_to_blender_world(*ground_hit, calib)
            keep = keep_ground_point(bx, by, 0.0)
            if keep is not None:
                out.append(keep)
            continue

        if depth_map is not None:
            ui = int(np.clip(round(u), 0, width - 1))
            vi = int(np.clip(round(v), 0, height - 1))
            depth_m = float(depth_map[vi, ui])
            if depth_m > 0.1:
                Xc, Yc, Zc = pixel_to_world(u, v, depth_m, calib)
                bx, by, _ = camera_point_to_blender_world(Xc, Yc, Zc, calib)
                keep = keep_ground_point(bx, by, 0.0)
                if keep is not None:
                    out.append(keep)

    return out


# ============================================================================
# Input discovery / loading
# ============================================================================

def discover_paths(
    scene: str,
    view: str,
    data_root: Path,
    renders_dir: Path,
) -> Dict[str, Optional[Path]]:
    project_root = _THIS
    seq_dir = data_root / "Sequences" / scene
    layout = scene_output_layout(scene)

    render_bases = [
        layout.legacy_repo_renders,
        layout.legacy_output_renders,
        renders_dir,
        project_root / "renders" / scene,
        project_root / "renders",
        project_root / "output" / "renders" / scene,
        project_root / "output" / "renders",
    ]

    paths: Dict[str, Optional[Path]] = {}
    paths["frame_json_dir"] = best_frame_json_dir(
        [
            layout.detections,
            layout.legacy_detections,
            project_root / "output" / "detections" / scene,
            project_root / "detections" / scene,
        ]
    )
    paths["det_json"] = first_match(
        ["detections.json", "*detections*.json", "detection*.json"],
        [layout.detections, *render_bases],
    )
    paths["tl_json"] = first_match(
        ["traffic_lights.json", "*traffic_light*.json", "*traffic*.json"],
        [layout.traffic_lights, *render_bases],
    )
    paths["lane_json"] = first_match(
        ["combined.json", "*combined*.json", "lane*.json", "lanes*.json"],
        [layout.lanes, *render_bases],
    )
    paths["depth_npz"] = first_match(
        ["depth_frames.npz", "depth_maps.npz", "*depth*.npz", "*.npz"],
        [layout.depth, seq_dir / "Depth", layout.legacy_depth, project_root / "output" / "depth" / scene],
    )
    paths["flow_json"] = first_match(
        ["flow_results.json", "*flow*.json"],
        [layout.flow, layout.legacy_repo_renders, layout.legacy_output_renders, renders_dir],
    )
    paths["pose_json"] = first_match(
        ["pose_keypoints.json", "*pose*.json", "poses/poses.json", "poses/*pose*.json"],
        [layout.detections, layout.legacy_repo_renders, layout.legacy_output_renders, *render_bases],
    )
    paths["vehicle_3d_json"] = first_match(
        ["vehicle_3d_detections.json", "*vehicle_3d*.json"],
        [layout.detections, *render_bases],
    )
    paths["video"] = first_match(
        [f"*{view}*undistort*.mp4", f"*{view}*.mp4", "*.mp4"],
        [seq_dir / "Undist", seq_dir / "Raw"],
    )
    return paths


def load_depth_store(path: Optional[Path]) -> LazyDepthStore:
    store = LazyDepthStore(path)
    if store:
        print(
            f"[assembler] {'depth npz':<18} ready  ({len(store.keys()):4d} frames) ← {store.path}"
        )
    else:
        print("[assembler] WARNING — depth npz not found; falling back to JSON depth only.")
    return store


def load_frame_json_store(path: Optional[Path]) -> LazyFrameJsonStore:
    store = LazyFrameJsonStore(path)
    if store:
        print(
            f"[assembler] {'frame json dir':<18} ready  ({len(store.keys()):4d} frames) ← {store.frame_dir}"
        )
    else:
        print("[assembler] WARNING — per-frame fused JSON directory not found.")
    return store


class SequentialVideoFrameReader:
    """Sequential video reader for light-state sampling during assembly."""

    def __init__(self, path: Optional[Path], fallback_frame_dirs: Optional[Sequence[Path]] = None) -> None:
        self.path = path.resolve() if path else None
        self.cap: Optional[cv2.VideoCapture] = None
        self.next_frame_idx = 0
        self.fallback_frame_dirs = [Path(p).resolve() for p in (fallback_frame_dirs or [])]
        if self.path and self.path.exists():
            cap = cv2.VideoCapture(str(self.path))
            if cap.isOpened():
                self.cap = cap

    def __bool__(self) -> bool:
        return self.cap is not None or any(path.exists() for path in self.fallback_frame_dirs)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _reopen(self) -> None:
        self.close()
        if self.path and self.path.exists():
            cap = cv2.VideoCapture(str(self.path))
            if cap.isOpened():
                self.cap = cap
                self.next_frame_idx = 0

    def get(self, frame_idx: int) -> Optional[np.ndarray]:
        frame = self._get_from_video(frame_idx)
        if frame is not None:
            return frame
        return self._get_from_fallback_frames(frame_idx)

    def _get_from_video(self, frame_idx: int) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        target = int(frame_idx)
        if target < self.next_frame_idx:
            self._reopen()
            if self.cap is None:
                return None

        while self.next_frame_idx < target:
            ok = self.cap.grab()
            if not ok:
                return None
            self.next_frame_idx += 1

        ok, frame = self.cap.read()
        if not ok:
            return None
        self.next_frame_idx += 1
        return frame

    def _get_from_fallback_frames(self, frame_idx: int) -> Optional[np.ndarray]:
        frame_no = int(frame_idx) + 1
        names = (
            f"frame_{frame_no:06d}_real.png",
            f"frame_{frame_no:06d}.png",
        )
        for base_dir in self.fallback_frame_dirs:
            for name in names:
                path = base_dir / name
                if path.exists():
                    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
        return None


def _sample_vehicle_light_patch(
    frame_bgr: np.ndarray,
    bbox: Sequence[int],
    *,
    x_frac_lo: float,
    x_frac_hi: float,
    y_frac_lo: float,
    y_frac_hi: float,
) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px1 = int(round(x1 + x_frac_lo * bw))
    px2 = int(round(x1 + x_frac_hi * bw))
    py1 = int(round(y1 + y_frac_lo * bh))
    py2 = int(round(y1 + y_frac_hi * bh))
    px1 = int(np.clip(px1, x1, x2 - 1))
    px2 = int(np.clip(px2, px1 + 1, x2))
    py1 = int(np.clip(py1, y1, y2 - 1))
    py2 = int(np.clip(py2, py1 + 1, y2))
    patch = frame_bgr[py1:py2, px1:px2]
    return patch if patch.size else None


def _mask_energy(mask: np.ndarray, value_channel: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    frac = float(np.mean(mask.astype(np.float32)))
    if frac <= 1e-6:
        return 0.0
    mean_v = float(np.mean(value_channel[mask])) / 255.0 if np.any(mask) else 0.0
    return frac * (0.45 + 0.55 * mean_v)


def _normalise_track_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _track_token_relation(a: Any, b: Any) -> int:
    token_a = _normalise_track_token(a)
    token_b = _normalise_track_token(b)
    if token_a is None or token_b is None:
        return 0
    return 1 if token_a == token_b else -1


def infer_vehicle_signal_state(
    frame_bgr: Optional[np.ndarray],
    det: Dict[str, Any],
) -> Dict[str, Any]:
    if frame_bgr is None:
        return {}
    bbox = det.get("bbox", [])
    if len(bbox) < 4:
        return {}
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    if bw < 36 or bh < 24:
        return {}

    left_patch = _sample_vehicle_light_patch(frame_bgr, bbox, x_frac_lo=0.03, x_frac_hi=0.24, y_frac_lo=0.24, y_frac_hi=0.58)
    right_patch = _sample_vehicle_light_patch(frame_bgr, bbox, x_frac_lo=0.76, x_frac_hi=0.97, y_frac_lo=0.24, y_frac_hi=0.58)
    if left_patch is None or right_patch is None:
        return {}

    def patch_scores(patch: np.ndarray) -> Tuple[float, float]:
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[..., 0]
        cr = ycrcb[..., 1]
        cb = ycrcb[..., 2]
        brake_mask = (cr >= 150) & ((cr.astype(np.int16) - cb.astype(np.int16)) >= 24) & (y >= 38)
        brake_score = _mask_energy(brake_mask, y)

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        amber_mask = ((h >= 8) & (h <= 35) & (s >= 70) & (v >= 95))
        amber_score = _mask_energy(amber_mask, np.maximum(v, y))
        return brake_score, amber_score

    brake_left, amber_left = patch_scores(left_patch)
    brake_right, amber_right = patch_scores(right_patch)
    brake_total = 0.5 * (brake_left + brake_right)

    brake_symmetry = min(brake_left, brake_right) / max(max(brake_left, brake_right), 1e-6)
    brake_on = (
        brake_total >= 0.050
        and max(brake_left, brake_right) >= 0.036
        and brake_symmetry >= 0.28
    )
    left_on = (
        amber_left >= 0.090
        and amber_left >= max(amber_right * 1.45, amber_right + 0.040)
        and amber_left >= brake_left + 0.020
    )
    right_on = (
        amber_right >= 0.090
        and amber_right >= max(amber_left * 1.45, amber_left + 0.040)
        and amber_right >= brake_right + 0.020
    )

    return {
        "visual_brake_lights_on": bool(brake_on),
        "visual_indicator_left_on": bool(left_on),
        "visual_indicator_right_on": bool(right_on),
        "visual_light_scores": {
            "brake_left": round(float(brake_left), 4),
            "brake_right": round(float(brake_right), 4),
            "brake_symmetry": round(float(brake_symmetry), 4),
            "amber_left": round(float(amber_left), 4),
            "amber_right": round(float(amber_right), 4),
        },
    }


# ============================================================================
# Standardisation
# ============================================================================

def standardise_aggregated_detection(raw: Dict[str, Any], source: str) -> Optional[StandardDetection]:
    bbox = raw.get("bbox")
    if bbox is None:
        return None
    raw_class = raw.get("class", "unknown")
    canonical_class, subtype = canonicalize_class(raw_class)
    sign_label = raw.get("sign_label", raw.get("subclass", raw.get("raw_label")))
    canonical_class, subtype, norm_sign_label = canonicalize_sign_detection(
        canonical_class=canonical_class,
        subtype=subtype,
        raw_label=raw.get("raw_label", raw_class),
        sign_label=sign_label,
        speed_limit_value=raw.get("speed_limit_value"),
    )
    return StandardDetection(
        det_id=raw.get("id"),
        raw_class=str(raw_class),
        canonical_class=canonical_class,
        subtype=subtype,
        bbox=[int(v) for v in bbox[:4]],
        confidence=float(raw.get("confidence", 0.0)),
        depth_m=raw.get("depth_m"),
        source_track_id=raw.get("track_id"),
        source=source,
        tl_color=raw.get("signal_color", raw.get("color")),
        tl_color_conf=raw.get("state_confidence", raw.get("color_confidence")),
        tl_state=raw.get("signal_state"),
        tl_shape=raw.get("signal_shape"),
        tl_decision_source=raw.get("decision_source"),
        tl_detic_color_check=raw.get("detic_color_check"),
        tl_detic_color_conf=raw.get("detic_color_confidence"),
        tl_detic_color_agrees=raw.get("detic_color_agrees"),
        speed_limit_value=raw.get("speed_limit_value"),
        sign_label=norm_sign_label,
        provided_position_blender=(
            [float(v) for v in raw.get("position_3d", [])[:3]]
            if raw.get("position_3d") is not None
            else None
        ),
    )


def standardise_fused_frame(
    fused_frame: Dict[str, Any],
) -> Tuple[List[StandardDetection], List[StandardDetection]]:
    objects: List[StandardDetection] = []
    traffic_lights: List[StandardDetection] = []

    for raw in fused_frame.get("vehicles", []):
        raw_class = raw.get("class_name", "car")
        canonical_class, subtype = canonicalize_class(raw_class)
        objects.append(
            StandardDetection(
                det_id=raw.get("id"),
                raw_class=str(raw_class),
                canonical_class=canonical_class,
                subtype=subtype,
                bbox=[int(v) for v in raw.get("bbox_px", raw.get("bbox", []))[:4]],
                confidence=float(raw.get("confidence", 0.0)),
                depth_m=raw.get("depth_m"),
                source="fused_vehicle",
                source_yaw_rad=raw.get("rotation_z"),
                provided_position_blender=blender_from_fused_position(raw.get("position_3d")),
            )
        )

    for raw in fused_frame.get("pedestrians", []):
        objects.append(
            StandardDetection(
                det_id=raw.get("id"),
                raw_class="pedestrian",
                canonical_class="pedestrian",
                subtype="pedestrian",
                bbox=[int(v) for v in raw.get("bbox_px", raw.get("bbox", []))[:4]],
                confidence=float(raw.get("confidence", 0.0)),
                depth_m=raw.get("depth_m"),
                source="fused_pedestrian",
                source_yaw_rad=raw.get("rotation_z"),
                provided_position_blender=blender_from_fused_position(raw.get("position_3d")),
            )
        )

    for raw in fused_frame.get("stop_signs", []):
        objects.append(
            StandardDetection(
                det_id=raw.get("id"),
                raw_class="stop_sign",
                canonical_class="stop_sign",
                subtype="stop_sign",
                bbox=[int(v) for v in raw.get("bbox_px", raw.get("bbox", []))[:4]],
                confidence=float(raw.get("confidence", 0.0)),
                depth_m=raw.get("depth_m"),
                source="fused_stop_sign",
                provided_position_blender=blender_from_fused_position(raw.get("position_3d")),
                sign_label="stop_sign",
            )
        )

    for raw in fused_frame.get("traffic_signs", []):
        raw_label = raw.get("class_name", raw.get("class", "traffic_sign"))
        canonical_class, subtype = canonicalize_class(raw_label)
        canonical_class, subtype, norm_sign_label = canonicalize_sign_detection(
            canonical_class=canonical_class,
            subtype=subtype,
            raw_label=raw_label,
            sign_label=raw.get("sign_label", raw_label),
            speed_limit_value=raw.get("speed_limit_value"),
        )
        objects.append(
            StandardDetection(
                det_id=raw.get("id"),
                raw_class=str(raw_label),
                canonical_class=canonical_class,
                subtype=subtype,
                bbox=[int(v) for v in raw.get("bbox_px", raw.get("bbox", []))[:4]],
                confidence=float(raw.get("confidence", 0.0)),
                depth_m=raw.get("depth_m"),
                source="fused_traffic_sign",
                provided_position_blender=blender_from_fused_position(raw.get("position_3d")),
                speed_limit_value=raw.get("speed_limit_value"),
                sign_label=norm_sign_label,
            )
        )

    for raw in fused_frame.get("speed_limits", []):
        objects.append(
            StandardDetection(
                det_id=raw.get("id"),
                raw_class="speed_limit",
                canonical_class="speed_limit",
                subtype="speed_limit",
                bbox=[int(v) for v in raw.get("bbox_px", raw.get("bbox", []))[:4]],
                confidence=float(raw.get("confidence", 0.0)),
                depth_m=raw.get("depth_m"),
                source="fused_speed_limit",
                provided_position_blender=blender_from_fused_position(raw.get("position_3d")),
                speed_limit_value=raw.get("speed_limit_value"),
                sign_label="speed_limit",
            )
        )

    for raw in fused_frame.get("traffic_lights", []):
        traffic_lights.append(
            StandardDetection(
                det_id=raw.get("id"),
                raw_class="traffic_light",
                canonical_class="traffic_light",
                subtype="traffic_light",
                bbox=[int(v) for v in raw.get("bbox_px", raw.get("bbox", []))[:4]],
                confidence=float(raw.get("confidence", 0.0)),
                depth_m=raw.get("depth_m"),
                source="fused_traffic_light",
                tl_color=raw.get("signal_color", raw.get("color", "unknown")),
                tl_color_conf=raw.get("state_confidence", raw.get("color_confidence")),
                tl_state=raw.get("signal_state"),
                tl_shape=raw.get("signal_shape"),
                tl_decision_source=raw.get("decision_source"),
                tl_detic_color_check=raw.get("detic_color_check"),
                tl_detic_color_conf=raw.get("detic_color_confidence"),
                tl_detic_color_agrees=raw.get("detic_color_agrees"),
                provided_position_blender=blender_from_fused_position(raw.get("position_3d")),
            )
        )

    return objects, traffic_lights


def standard_detection_match(
    candidate: StandardDetection,
    existing: StandardDetection,
) -> bool:
    same_class = candidate.canonical_class == existing.canonical_class
    if not same_class:
        return False

    if candidate.det_id is not None and existing.det_id is not None:
        if candidate.det_id == existing.det_id:
            return True

    iou_thresh = 0.25 if candidate.canonical_class in STATIC_UPRIGHT_CLASSES else 0.55
    return bbox_iou(candidate.bbox, existing.bbox) >= iou_thresh


def merge_standard_detection_metadata(
    primary: StandardDetection,
    secondary: StandardDetection,
) -> None:
    primary.confidence = max(float(primary.confidence), float(secondary.confidence))

    if primary.depth_m is None and secondary.depth_m is not None:
        primary.depth_m = secondary.depth_m
    if primary.source_yaw_rad is None and secondary.source_yaw_rad is not None:
        primary.source_yaw_rad = secondary.source_yaw_rad
    if primary.provided_position_blender is None and secondary.provided_position_blender is not None:
        primary.provided_position_blender = list(secondary.provided_position_blender)
    if primary.source_track_id is None and secondary.source_track_id is not None:
        primary.source_track_id = secondary.source_track_id

    if primary.subtype in {"car", "traffic_sign"} and secondary.subtype not in {"", "unknown"}:
        primary.subtype = secondary.subtype
    if primary.canonical_class == "traffic_sign" and secondary.canonical_class in SUPPORTED_RENDERABLE_SIGNS:
        primary.canonical_class = secondary.canonical_class
        primary.subtype = secondary.subtype
    if primary.raw_class in {"unknown", primary.canonical_class} and secondary.raw_class:
        primary.raw_class = secondary.raw_class
    if primary.sign_label in {None, "", "traffic_sign"} and secondary.sign_label not in {None, "", "traffic_sign"}:
        primary.sign_label = secondary.sign_label

    # Prefer any concrete traffic-light state over "unknown".
    if secondary.tl_color not in {None, "", "unknown"}:
        if primary.tl_color in {None, "", "unknown"} or float(secondary.tl_color_conf or 0.0) >= float(primary.tl_color_conf or 0.0):
            primary.tl_color = secondary.tl_color
            primary.tl_color_conf = secondary.tl_color_conf
            primary.tl_state = secondary.tl_state or primary.tl_state
            primary.tl_shape = secondary.tl_shape or primary.tl_shape
            primary.tl_decision_source = secondary.tl_decision_source or primary.tl_decision_source
            primary.tl_detic_color_check = secondary.tl_detic_color_check or primary.tl_detic_color_check
            primary.tl_detic_color_conf = secondary.tl_detic_color_conf or primary.tl_detic_color_conf
            if secondary.tl_detic_color_agrees is not None:
                primary.tl_detic_color_agrees = secondary.tl_detic_color_agrees
    elif primary.tl_color is None and secondary.tl_color is not None:
        primary.tl_color = secondary.tl_color
        primary.tl_color_conf = secondary.tl_color_conf

    if primary.speed_limit_value is None and secondary.speed_limit_value is not None:
        primary.speed_limit_value = secondary.speed_limit_value


def append_or_merge_standard_detection(
    items: List[StandardDetection],
    candidate: StandardDetection,
) -> None:
    for existing in items:
        if standard_detection_match(candidate, existing):
            merge_standard_detection_metadata(existing, candidate)
            return
    items.append(candidate)


def assemble_standard_detection(
    det: StandardDetection,
    depth_map: Optional[np.ndarray],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
    flow_match: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if len(det.bbox) < 4:
        return None
    if det.canonical_class == "traffic_sign" and not is_supported_renderable_sign(
        det.canonical_class,
        det.subtype,
        det.sign_label,
    ):
        return None
    if det.canonical_class == "traffic_light":
        bbox_w = max(1, int(det.bbox[2]) - int(det.bbox[0]))
        bbox_h = max(1, int(det.bbox[3]) - int(det.bbox[1]))
        bbox_area = bbox_w * bbox_h
        decision_source = str(det.tl_decision_source or "").strip().lower()
        if decision_source == "classical" and float(det.confidence or 0.0) < 0.60:
            if bbox_h < 14 or bbox_area < 220:
                return None

    depth_m = float(det.depth_m) if det.depth_m is not None else None
    if (depth_m is None or depth_m <= 0.1 or depth_m > MAX_VALID_DEPTH_M) and depth_map is not None:
        depth_m = median_bbox_depth(det.bbox, depth_map)

    placement = lift_bbox_to_3d(
        bbox=det.bbox,
        depth_m=depth_m,
        canonical_class=det.canonical_class,
        subtype=det.subtype,
        calib=calib,
        frame_w=frame_w,
        frame_h=frame_h,
        provided_position_blender=det.provided_position_blender,
    )
    if placement is None:
        return None

    record: Dict[str, Any] = {
        "frame_det_id": det.det_id,
        "class": det.canonical_class,
        "subclass": det.subtype,
        "raw_class": det.raw_class,
        "bbox": [int(v) for v in det.bbox],
        "confidence": round(float(det.confidence), 4),
        "depth_m": placement["depth_m"],
        "position_blender": placement["position_blender"],
        "lifted_position_blender": placement["position_blender"],
        "lifted_depth_m": placement["depth_m"],
        "position_source": "bbox_depth",
        "scale": placement["scale"],
        "real_h_est": placement["real_h_est"],
        "real_w_est": placement["real_w_est"],
        "dims_m": placement["dims_m"],
        "bbox_dims_m": placement["bbox_dims_m"],
        "anchor_source": placement["anchor_source"],
        "bbox_depth_floor_m": placement.get("bbox_depth_floor_m"),
        "asset": "",
        "track_id": None,
        "source_track_id": det.source_track_id,
        "yaw_rad": round(float(det.source_yaw_rad), 4) if det.source_yaw_rad is not None else None,
        "source_yaw_rad": round(float(det.source_yaw_rad), 4) if det.source_yaw_rad is not None else None,
        "yaw_source": "source_detection" if det.source_yaw_rad is not None else None,
        "source": det.source,
    }
    if placement.get("visible_center_blender") is not None:
        record["visible_center_blender"] = placement["visible_center_blender"]
    if det.tl_color is not None:
        record["tl_color"] = det.tl_color
        record["tl_color_conf"] = round(float(det.tl_color_conf or 0.0), 4)
    if det.tl_state is not None:
        record["tl_state"] = str(det.tl_state)
    if det.tl_shape is not None:
        record["tl_shape"] = str(det.tl_shape)
    if det.tl_decision_source is not None:
        record["tl_decision_source"] = str(det.tl_decision_source)
    if det.tl_detic_color_check is not None:
        record["tl_detic_color_check"] = str(det.tl_detic_color_check)
        record["tl_detic_color_conf"] = round(float(det.tl_detic_color_conf or 0.0), 4)
    if det.tl_detic_color_agrees is not None:
        record["tl_detic_color_agrees"] = bool(det.tl_detic_color_agrees)
    if det.speed_limit_value is not None:
        try:
            record["speed_limit_value"] = int(det.speed_limit_value)
        except (TypeError, ValueError):
            pass
    if det.sign_label is not None:
        record["sign_label"] = str(det.sign_label)
    if flow_match is not None and det.canonical_class in FLOW_YAW_CLASSES:
        motion_vec = flow_match.get("motion_vec")
        if isinstance(motion_vec, list) and len(motion_vec) >= 2:
            record["flow_motion_vec"] = [round(float(motion_vec[0]), 4), round(float(motion_vec[1]), 4)]
        if flow_match.get("magnitude_px") is not None:
            record["flow_magnitude_px"] = round(float(flow_match["magnitude_px"]), 4)
        if flow_match.get("angle_deg") is not None:
            record["flow_angle_deg"] = round(float(flow_match["angle_deg"]), 3)
    if flow_match is not None and det.canonical_class in ROAD_CONTEXT_CLASSES:
        if flow_match.get("speed_px") is not None:
            record["speed_px"] = round(float(flow_match["speed_px"]), 4)
        if flow_match.get("direction") is not None:
            record["motion_direction"] = str(flow_match["direction"])
        if flow_match.get("moving") is not None:
            record["moving"] = bool(flow_match["moving"])
        if flow_match.get("parked") is not None:
            record["parked"] = bool(flow_match["parked"])
        if flow_match.get("track_id") is not None:
            record["flow_track_id"] = flow_match.get("track_id")

    bbox_w = max(1, int(record["bbox"][2]) - int(record["bbox"][0]))
    bbox_h = max(1, int(record["bbox"][3]) - int(record["bbox"][1]))
    known_h, known_w, _ = get_class_dims(record["class"], record["subclass"])
    if moving_detection_is_implausible(
        canonical_class=str(record["class"]),
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        confidence=float(record["confidence"]),
        real_h_est=float(record["real_h_est"]),
        real_w_est=float(record["real_w_est"]),
        known_h=known_h,
        known_w=known_w,
    ):
        return None

    closure = projection_closure_metrics(record, calib)
    if closure is not None:
        record["projection_closure"] = {
            "h_ratio": round(float(closure["h_ratio"]), 4),
            "w_ratio": round(float(closure["w_ratio"]), 4),
            "center_dx_norm": round(float(closure["center_dx_norm"]), 4),
            "center_dy_norm": round(float(closure["center_dy_norm"]), 4),
        }

    if det.canonical_class in STATIC_UPRIGHT_CLASSES:
        if det.provided_position_blender is None:
            if static_semantic_is_grossly_implausible(
                record,
                calib=calib,
                frame_w=frame_w,
                frame_h=frame_h,
            ):
                return None
    else:
        if detection_is_implausible_after_lift(
            record,
            calib=calib,
            frame_w=frame_w,
            frame_h=frame_h,
            track_len=2,
        ):
            return None

    return record


def collect_frame_detections(
    frame_idx: int,
    det_index: Dict[int, Dict[str, Any]],
    tl_index: Dict[int, Dict[str, Any]],
    flow_index: Dict[int, Dict[str, Any]],
    fused_store: LazyFrameJsonStore,
    depth_map: Optional[np.ndarray],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    objects: List[Dict[str, Any]] = []
    traffic_lights: List[Dict[str, Any]] = []
    fused_frame = fused_store.get(frame_idx) if fused_store else None
    flow_frame = flow_index.get(frame_idx)

    aggregated_det_frame = det_index.get(frame_idx)
    aggregated_tl_frame = tl_index.get(frame_idx)
    aggregated_objects = list((aggregated_det_frame or {}).get("detections", []))
    aggregated_lights = list((aggregated_tl_frame or {}).get("traffic_lights", []))

    fused_objects, fused_tl = standardise_fused_frame(fused_frame) if fused_frame else ([], [])
    object_candidates: List[StandardDetection] = []
    traffic_light_candidates: List[StandardDetection] = []

    for std in fused_objects:
        append_or_merge_standard_detection(object_candidates, std)
    for std in fused_tl:
        append_or_merge_standard_detection(traffic_light_candidates, std)

    for raw in aggregated_objects:
        std = standardise_aggregated_detection(raw, source="aggregated_object")
        if std is None:
            continue
        if std.canonical_class == "traffic_light":
            append_or_merge_standard_detection(traffic_light_candidates, std)
            continue
        append_or_merge_standard_detection(object_candidates, std)

    for raw in aggregated_lights:
        std = standardise_aggregated_detection(raw, source="aggregated_traffic_light")
        if std is None:
            continue
        std.canonical_class = "traffic_light"
        std.subtype = "traffic_light"
        std.tl_color = raw.get("signal_color", raw.get("color", "unknown"))
        std.tl_color_conf = raw.get("state_confidence", raw.get("color_confidence"))
        std.tl_state = raw.get("signal_state")
        std.tl_shape = raw.get("signal_shape")
        std.tl_decision_source = raw.get("decision_source")
        append_or_merge_standard_detection(traffic_light_candidates, std)

    for std in object_candidates:
        record = assemble_standard_detection(
            std,
            depth_map,
            calib,
            frame_w,
            frame_h,
            flow_match=match_flow_to_detection(std, flow_frame),
        )
        if record is not None:
            objects.append(record)

    for std in traffic_light_candidates:
        record = assemble_standard_detection(
            std,
            depth_map,
            calib,
            frame_w,
            frame_h,
            flow_match=match_flow_to_detection(std, flow_frame),
        )
        if record is not None:
            traffic_lights.append(record)

    return objects, traffic_lights, fused_frame


def extract_lanes_and_road(
    frame_idx: int,
    lane_index: Dict[int, Dict[str, Any]],
    fused_frame: Optional[Dict[str, Any]],
    depth_map: Optional[np.ndarray],
    calib: CalibData,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    lanes: List[Dict[str, Any]] = []
    road: Dict[str, Any] = {}
    frame_h = int(depth_map.shape[0]) if isinstance(depth_map, np.ndarray) and depth_map.ndim >= 2 else int(round(max(1.0, 2.0 * float(calib.cy))))

    def _optional_triplet(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        try:
            return [round(float(value[0]), 4), round(float(value[1]), 4), round(float(value[2]), 4)]
        except (TypeError, ValueError):
            return None

    def _marking_bbox(marking: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        bbox = marking.get("bbox", [])
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _is_crosswalk_candidate(marking: Dict[str, Any]) -> bool:
        if str(marking.get("marking_type", "road_marking")) != "road_marking":
            return False
        if str(marking.get("color", "unknown")) != "white":
            return False
        bbox = _marking_bbox(marking)
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = int(marking.get("area_px", 0) or 0)
        aspect = float(w) / max(float(h), 1.0)
        return bool(
            area >= 1400
            and aspect >= 2.0
            and y2 >= int(0.48 * frame_h)
        )

    def _upgrade_crosswalk_markings(markings: List[Dict[str, Any]]) -> None:
        candidate_indices = [idx for idx, marking in enumerate(markings) if _is_crosswalk_candidate(marking)]
        if len(candidate_indices) < 2:
            return
        for idx in candidate_indices:
            marking = markings[idx]
            bbox = _marking_bbox(marking)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            rect_img = [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]
            rect_3d = lift_lane_points_to_3d(rect_img, calib, depth_map)
            if len(rect_3d) >= 4:
                marking["contour_img"] = rect_img
                marking["contour_3d"] = rect_3d
            marking["marking_type"] = "crosswalk"

    if frame_idx in lane_index:
        lane_frame = lane_index[frame_idx]
        lane_entries = list(lane_frame.get("lanes", []))
        derived_markings: List[Dict[str, Any]] = []
        if not lane_entries:
            for inst in lane_frame.get("lane_instances", []):
                class_name = str(inst.get("class_name", ""))
                if class_name == "road-sign-line":
                    bbox = [int(v) for v in inst.get("bbox", [0, 0, 0, 0])[:4]]
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox
                        if x2 > x1 and y2 > y1:
                            rect_img = [
                                [float(x1), float(y1)],
                                [float(x2), float(y1)],
                                [float(x2), float(y2)],
                                [float(x1), float(y2)],
                            ]
                            contour_3d = lift_lane_points_to_3d(rect_img, calib, depth_map)
                            if len(contour_3d) >= 3:
                                derived_markings.append(
                                    {
                                        "id": int(inst.get("id", len(derived_markings))),
                                        "marking_type": "road_marking",
                                        "color": "white",
                                        "confidence": round(float(inst.get("confidence", 0.0) or 0.0), 4),
                                        "bbox": bbox,
                                        "area_px": int(max(x2 - x1, 0) * max(y2 - y1, 0)),
                                        "contour_img": rect_img,
                                        "contour_3d": contour_3d,
                                        "direction": None,
                                    }
                                )
                    continue
                contour = np.asarray(inst.get("contour_img", []), dtype=np.float32)
                points_2d: List[List[float]] = []
                if contour.ndim == 2 and contour.shape[0] >= 4:
                    ys = contour[:, 1]
                    xs = contour[:, 0]
                    y_lo = int(np.percentile(ys, 5))
                    y_hi = int(np.percentile(ys, 95))
                    if (y_hi - y_lo) >= 12:
                        band = max(2, int(round((y_hi - y_lo) / 24.0)))
                        last_y = -10**9
                        for y_ref in np.linspace(y_lo, y_hi, 24):
                            select = np.abs(ys - y_ref) <= band
                            if int(np.count_nonzero(select)) < 2:
                                continue
                            y_med = int(round(float(np.median(ys[select]))))
                            if y_med <= last_y:
                                continue
                            x_med = int(round(float(np.median(xs[select]))))
                            points_2d.append([x_med, y_med])
                            last_y = y_med
                if len(points_2d) < 2:
                    bbox = [int(v) for v in inst.get("bbox", [0, 0, 0, 0])[:4]]
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox
                        cx = 0.5 * (x1 + x2)
                        points_2d = [[cx, float(y1)], [cx, float(y2)]]
                if len(points_2d) < 2:
                    continue
                lane_entries.append(
                    {
                        "id": inst.get("id", len(lane_entries)),
                        "lane_type": inst.get("lane_type", "solid"),
                        "line_class": class_name or inst.get("lane_type", "solid"),
                        "color": inst.get("paint_color", "unknown"),
                        "confidence": inst.get("confidence"),
                        "color_confidence": inst.get("paint_confidence", inst.get("color_confidence")),
                        "avg_hsv": _optional_triplet(inst.get("avg_hsv")),
                        "avg_ycrcb": _optional_triplet(inst.get("avg_ycrcb")),
                        "curve_points_img": points_2d,
                        "poly_coeffs": None,
                    }
                )

        for lane in lane_entries:
            points_2d = lane.get("curve_points_img", [])
            points_3d = lift_lane_points_to_3d(points_2d, calib, depth_map)
            if len(points_3d) < 2:
                continue
            lane_record = {
                "id": lane.get("id", 0),
                "lane_type": lane.get("lane_type", "solid"),
                "line_class": lane.get("line_class", lane.get("lane_type", "solid")),
                "lane_color": lane.get("color", "unknown"),
                "confidence": lane.get("confidence"),
                "points_2d": [[float(p[0]), float(p[1])] for p in points_2d],
                "points_3d": points_3d,
                "poly_coeffs": lane.get("poly_coeffs"),
            }
            color_conf = lane.get("color_confidence", lane.get("paint_confidence"))
            if color_conf is not None:
                try:
                    lane_record["lane_color_confidence"] = round(float(color_conf), 4)
                except (TypeError, ValueError):
                    pass
            avg_hsv = _optional_triplet(lane.get("avg_hsv"))
            if avg_hsv is not None:
                lane_record["avg_hsv"] = avg_hsv
            avg_ycrcb = _optional_triplet(lane.get("avg_ycrcb"))
            if avg_ycrcb is not None:
                lane_record["avg_ycrcb"] = avg_ycrcb
            lanes.append(lane_record)

        road_info = lane_frame.get("road", {})
        if road_info:
            contours_img = road_info.get("contours_img", [])
            contours_3d: List[List[List[float]]] = []
            for contour in contours_img:
                pts3d = lift_lane_points_to_3d(contour, calib, depth_map)
                if len(pts3d) >= 3:
                    contours_3d.append(pts3d)
            markings = []
            for marking in road_info.get("markings", []):
                contour_img = marking.get("contour_img", [])
                contour_3d = lift_lane_points_to_3d(contour_img, calib, depth_map)
                if len(contour_3d) < 3:
                    continue
                markings.append(
                    {
                        "id": int(marking.get("id", len(markings))),
                        "marking_type": str(marking.get("marking_type", "road_marking")),
                        "color": str(marking.get("color", "unknown")),
                        "confidence": round(float(marking.get("confidence", 0.0)), 4),
                        "bbox": [int(v) for v in marking.get("bbox", [0, 0, 0, 0])[:4]],
                        "contour_img": [[float(p[0]), float(p[1])] for p in contour_img],
                        "contour_3d": contour_3d,
                        "direction": marking.get("direction"),
                    }
                )
            markings.extend(derived_markings)
            _upgrade_crosswalk_markings(markings)
            road = {
                "area_frac": round(float(road_info.get("area_frac", 0.0)), 4),
                "contours_img": contours_img,
                "contours_3d": contours_3d,
                "markings": markings,
            }
        elif derived_markings:
            _upgrade_crosswalk_markings(derived_markings)
            road = {
                "area_frac": 0.0,
                "contours_img": [],
                "contours_3d": [],
                "markings": derived_markings,
            }

    elif fused_frame is not None:
        for lane_id, lane in enumerate(fused_frame.get("lanes", [])):
            points_3d = fused_ground_points_to_blender(lane.get("points_3d", []))
            if len(points_3d) < 2:
                continue
            lane_record = {
                "id": lane_id,
                "lane_type": lane.get("lane_type", lane.get("type", "solid")),
                "line_class": lane.get("line_class", lane.get("lane_type", lane.get("type", "solid"))),
                "lane_color": lane.get("color", lane.get("paint_color", "unknown")),
                "points_2d": [],
                "points_3d": points_3d,
                "poly_coeffs": lane.get("poly_coeffs"),
            }
            if lane.get("confidence") is not None:
                lane_record["confidence"] = lane.get("confidence")
            color_conf = lane.get("color_confidence", lane.get("paint_confidence"))
            if color_conf is not None:
                try:
                    lane_record["lane_color_confidence"] = round(float(color_conf), 4)
                except (TypeError, ValueError):
                    pass
            avg_hsv = _optional_triplet(lane.get("avg_hsv"))
            if avg_hsv is not None:
                lane_record["avg_hsv"] = avg_hsv
            avg_ycrcb = _optional_triplet(lane.get("avg_ycrcb"))
            if avg_ycrcb is not None:
                lane_record["avg_ycrcb"] = avg_ycrcb
            lanes.append(lane_record)

        road_info = fused_frame.get("road", {})
        contours_3d = []
        for contour in road_info.get("contours_3d", []):
            pts3d = fused_ground_points_to_blender(contour)
            if len(pts3d) >= 3:
                contours_3d.append(pts3d)
        markings = []
        for marking in road_info.get("markings", []):
            contour_3d = fused_ground_points_to_blender(marking.get("contour_3d", []))
            if len(contour_3d) < 3:
                continue
            markings.append(
                {
                    "id": int(marking.get("id", len(markings))),
                    "marking_type": str(marking.get("marking_type", "road_marking")),
                    "color": str(marking.get("color", "unknown")),
                    "confidence": round(float(marking.get("confidence", 0.0)), 4),
                    "bbox": [int(v) for v in marking.get("bbox", [0, 0, 0, 0])[:4]],
                    "contour_img": [],
                    "contour_3d": contour_3d,
                    "direction": marking.get("direction"),
                }
            )
        _upgrade_crosswalk_markings(markings)
        road = {
            "area_frac": round(float(road_info.get("area_frac", 0.0)), 4),
            "contours_img": [],
            "contours_3d": contours_3d,
            "markings": markings,
        }

    return lanes, road


# Tracking / temporal stabilisation
# ============================================================================

def track_cost(det: Dict[str, Any], state: TrackState, frame_gap: int) -> Optional[float]:
    if det["class"] != state.canonical_class:
        return None

    pos = det.get("position_blender", [0.0, 0.0, 0.0])
    dx = float(pos[0]) - state.position[0]
    dy = float(pos[1]) - state.position[1]
    dz = float(pos[2]) - state.position[2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    depth = float(det.get("depth_m", pos[0]))
    max_dist = 0.9 + 0.15 * max(depth, state.depth_m) + 0.45 * max(0, frame_gap - 1)
    iou = bbox_iou(det["bbox"], state.bbox)
    source_relation = _track_token_relation(det.get("source_track_id"), state.source_track_id)
    flow_relation = _track_token_relation(det.get("flow_track_id"), state.flow_track_id)
    veh3d_relation = _track_token_relation(det.get("vehicle_3d_track_id"), state.vehicle_3d_track_id)

    if dist > max_dist and iou < 0.03:
        return None
    if source_relation < 0 and veh3d_relation <= 0 and iou < 0.08 and dist > 3.8:
        return None

    cost = dist + 0.6 * max(0, frame_gap - 1) - 1.2 * iou
    if source_relation > 0:
        cost -= 2.0
    elif source_relation < 0:
        cost += 0.65
    if flow_relation > 0:
        cost -= 0.55
    elif flow_relation < 0:
        cost += 0.18
    if veh3d_relation > 0:
        cost -= 1.35
    elif veh3d_relation < 0:
        cost += 0.35

    det_subtype = str(det.get("subclass", det["class"]))
    if det_subtype == state.subtype:
        cost -= 0.08
    return cost


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def motion_direction_from_blender(dx: float, dy: float, moving_thresh: float = 0.12) -> str:
    mag = math.hypot(dx, dy)
    if mag < moving_thresh:
        return "stationary"

    angle = math.degrees(math.atan2(dy, dx))
    bins = [
        (-22.5, 22.5, "forward"),
        (22.5, 67.5, "forward-left"),
        (67.5, 112.5, "left"),
        (112.5, 157.5, "back-left"),
        (157.5, 180.0, "backward"),
        (-180.0, -157.5, "backward"),
        (-157.5, -112.5, "back-right"),
        (-112.5, -67.5, "right"),
        (-67.5, -22.5, "forward-right"),
    ]
    for lo, hi, label in bins:
        if lo <= angle < hi:
            return label
    return "unknown"


def upright_semantic_facing_yaw(position_blender: Sequence[float]) -> float:
    if len(position_blender) < 2:
        return 0.0
    px = float(position_blender[0])
    py = float(position_blender[1])
    radial = math.hypot(px, py)
    if radial <= 1e-5:
        return 0.0
    return wrap_angle(math.atan2(-py, -px))


def track_motion_samples(
    positions: Sequence[Sequence[float]],
    fps: float,
) -> Tuple[List[List[float]], List[float], List[float]]:
    if not positions:
        return [], [], []

    dt = 1.0 / max(float(fps), 1e-3)
    vectors: List[List[float]] = []
    speeds: List[float] = []
    accelerations: List[float] = []

    for idx, pos in enumerate(positions):
        prev_pos = positions[idx - 1] if idx > 0 else None
        next_pos = positions[idx + 1] if idx + 1 < len(positions) else None

        if prev_pos is not None and next_pos is not None:
            dx = (float(next_pos[0]) - float(prev_pos[0])) / (2.0 * dt)
            dy = (float(next_pos[1]) - float(prev_pos[1])) / (2.0 * dt)
        elif next_pos is not None:
            dx = (float(next_pos[0]) - float(pos[0])) / dt
            dy = (float(next_pos[1]) - float(pos[1])) / dt
        elif prev_pos is not None:
            dx = (float(pos[0]) - float(prev_pos[0])) / dt
            dy = (float(pos[1]) - float(prev_pos[1])) / dt
        else:
            dx = 0.0
            dy = 0.0

        speed = math.hypot(dx, dy)
        vectors.append([round(dx, 4), round(dy, 4), 0.0])
        speeds.append(speed)

    for idx, speed in enumerate(speeds):
        prev_speed = speeds[idx - 1] if idx > 0 else speed
        next_speed = speeds[idx + 1] if idx + 1 < len(speeds) else speed
        if idx > 0 and idx + 1 < len(speeds):
            accel = (next_speed - prev_speed) / (2.0 * dt)
        elif idx + 1 < len(speeds):
            accel = (next_speed - speed) / dt
        elif idx > 0:
            accel = (speed - prev_speed) / dt
        else:
            accel = 0.0
        accelerations.append(accel)

    return vectors, speeds, accelerations


def rear_is_visible_from_ego(
    position_blender: Sequence[float],
    yaw_rad: float,
) -> bool:
    if len(position_blender) < 2:
        return True
    px = float(position_blender[0])
    py = float(position_blender[1])
    radial = math.hypot(px, py)
    if radial < 1.0:
        return True
    to_ego_x = -px / radial
    to_ego_y = -py / radial
    forward_x = math.cos(float(yaw_rad))
    forward_y = math.sin(float(yaw_rad))
    return (forward_x * to_ego_x + forward_y * to_ego_y) <= -0.18


def infer_track_yaws(history: List[Dict[str, Any]]) -> List[float]:
    measured_yaws: List[Optional[float]] = []
    for det in history:
        measured = det.get("vehicle_3d_yaw_rad")
        if measured is None:
            measured = det.get("source_yaw_rad")
        if measured is None and det.get("yaw_source") in {"source_detection", "vehicle_3d"}:
            measured = det.get("yaw_rad")
        measured_yaws.append(float(measured) if measured is not None else None)

    def nearest_measured(idx: int) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
        prev_hit: Optional[Tuple[int, float]] = None
        next_hit: Optional[Tuple[int, float]] = None
        for j in range(idx - 1, -1, -1):
            val = measured_yaws[j]
            if val is not None:
                prev_hit = (j, float(val))
                break
        for j in range(idx + 1, len(measured_yaws)):
            val = measured_yaws[j]
            if val is not None:
                next_hit = (j, float(val))
                break
        return prev_hit, next_hit

    def blend_yaw(a: float, b: float, alpha: float) -> float:
        delta = wrap_angle(float(b) - float(a))
        return wrap_angle(float(a) + float(alpha) * delta)

    def local_motion_heading(idx: int) -> Optional[float]:
        pos = history[idx].get("position_blender", [0.0, 0.0, 0.0])
        prev_pos = history[idx - 1].get("position_blender") if idx > 0 else None
        next_pos = history[idx + 1].get("position_blender") if idx + 1 < len(history) else None
        if prev_pos is not None and next_pos is not None:
            dx = float(next_pos[0]) - float(prev_pos[0])
            dy = float(next_pos[1]) - float(prev_pos[1])
        elif next_pos is not None:
            dx = float(next_pos[0]) - float(pos[0])
            dy = float(next_pos[1]) - float(pos[1])
        elif prev_pos is not None:
            dx = float(pos[0]) - float(prev_pos[0])
            dy = float(pos[1]) - float(prev_pos[1])
        else:
            return None
        mag = math.hypot(dx, dy)
        if mag <= 0.18:
            return None
        return float(math.atan2(dy, dx))

    yaws: List[float] = []
    for idx, det in enumerate(history):
        measured_yaw = measured_yaws[idx]
        prev_yaw = yaws[-1] if yaws else None
        motion_heading = local_motion_heading(idx) if det["class"] in ROAD_CONTEXT_CLASSES else None
        if measured_yaw is not None:
            chosen = wrap_angle(float(measured_yaw))
            if det["class"] in ROAD_CONTEXT_CLASSES:
                candidates = [chosen, wrap_angle(chosen + math.pi)]
                refs: List[Tuple[float, float]] = []
                if prev_yaw is not None:
                    refs.append((float(prev_yaw), 0.75))
                if motion_heading is not None:
                    refs.append((float(motion_heading), 1.00))
                if refs:
                    def score(candidate: float) -> float:
                        return sum(weight * abs(wrap_angle(candidate - ref)) for ref, weight in refs)
                    chosen = min(candidates, key=score)
            yaws.append(float(chosen))
            continue

        prev_measured, next_measured = nearest_measured(idx)
        if det["class"] in ROAD_CONTEXT_CLASSES:
            if prev_measured is not None and next_measured is not None:
                prev_idx, prev_val = prev_measured
                next_idx, next_val = next_measured
                span = max(1, next_idx - prev_idx)
                if span <= 18:
                    alpha = float(idx - prev_idx) / float(span)
                    yaws.append(blend_yaw(prev_val, next_val, alpha))
                    continue
            if prev_measured is not None and (idx - prev_measured[0]) <= 8:
                yaws.append(float(prev_measured[1]))
                continue
            if next_measured is not None and (next_measured[0] - idx) <= 5:
                yaws.append(float(next_measured[1]))
                continue

        pos = det.get("position_blender", [0.0, 0.0, 0.0])
        prev_pos = history[idx - 1].get("position_blender") if idx > 0 else None
        next_pos = history[idx + 1].get("position_blender") if idx + 1 < len(history) else None

        candidates: List[Tuple[float, float]] = []
        for other in (prev_pos, next_pos):
            if other is None:
                continue
            dx = float(other[0]) - float(pos[0])
            dy = float(other[1]) - float(pos[1])
            mag = math.hypot(dx, dy)
            if mag > 0.08:
                candidates.append((mag, math.atan2(dy, dx)))

        flow_mag = 0.0
        if det["class"] in FLOW_YAW_CLASSES:
            flow_mag = float(det.get("flow_magnitude_px", 0.0) or 0.0)
        if candidates and det["class"] in MOVING_CLASSES:
            candidate_yaw = max(candidates, key=lambda item: item[0])[1]
            if prev_yaw is None:
                yaw = candidate_yaw
            else:
                same_dir = wrap_angle(candidate_yaw)
                flipped = wrap_angle(candidate_yaw + math.pi)
                same_delta = abs(wrap_angle(same_dir - prev_yaw))
                flipped_delta = abs(wrap_angle(flipped - prev_yaw))
                yaw = same_dir if same_delta <= flipped_delta else flipped

                # When optical flow says motion is weak, keep the prior heading
                # instead of allowing a noisy 180-degree flip.
                if flow_mag < 1.2 and abs(wrap_angle(yaw - prev_yaw)) > 0.45:
                    yaw = prev_yaw
                elif flow_mag < 2.2 and abs(wrap_angle(yaw - prev_yaw)) > 1.05:
                    yaw = prev_yaw
        elif det["class"] in STATIC_UPRIGHT_CLASSES:
            yaw = upright_semantic_facing_yaw(det.get("position_blender", [0.0, 0.0, 0.0]))
        elif prev_yaw is not None and det["class"] in MOVING_CLASSES:
            yaw = prev_yaw
        else:
            yaw = 0.0
        yaws.append(float(yaw))

    if not yaws:
        return []

    smoothed = [wrap_angle(yaws[0])]
    for yaw in yaws[1:]:
        prev = smoothed[-1]
        cur = wrap_angle(yaw)
        while cur - prev > math.pi:
            cur -= 2.0 * math.pi
        while cur - prev < -math.pi:
            cur += 2.0 * math.pi
        smoothed.append(cur)

    return [wrap_angle(val) for val in smoothed]


def smooth_track_positions(history: List[Dict[str, Any]], window_radius: int = 1) -> List[List[float]]:
    smoothed: List[List[float]] = []
    for idx in range(len(history)):
        lo = max(0, idx - window_radius)
        hi = min(len(history), idx + window_radius + 1)
        window_positions = [
            history[j].get("position_blender", [0.0, 0.0, 0.0])
            for j in range(lo, hi)
        ]
        xs = [float(pos[0]) for pos in window_positions]
        ys = [float(pos[1]) for pos in window_positions]
        zs = [float(pos[2]) for pos in window_positions]
        smoothed.append(
            [
                round(float(statistics.median(xs)), 4),
                round(float(statistics.median(ys)), 4),
                round(float(statistics.median(zs)), 4),
            ]
        )
    return smoothed


def center_bias_for_detection(det: Dict[str, Any]) -> float:
    dims = det.get("dims_m", [1.5, 1.8, 4.5])
    length_m = float(dims[2]) * float(det.get("scale", 1.0))
    canonical_class = str(det.get("class", "car"))

    if canonical_class == "car":
        return float(np.clip(0.40 * length_m, 1.2, 2.2))
    if canonical_class == "truck":
        return float(np.clip(0.42 * length_m, 2.0, 3.8))
    if canonical_class == "bus":
        return float(np.clip(0.42 * length_m, 2.2, 4.2))
    if canonical_class in {"motorcycle", "bicycle"}:
        return float(np.clip(0.32 * length_m, 0.4, 0.9))
    if canonical_class == "pedestrian":
        return 0.10
    return 0.0


def shift_position_away_from_camera(
    position: Sequence[float],
    center_bias_m: float,
) -> List[float]:
    px = float(position[0])
    py = float(position[1])
    pz = float(position[2])
    if center_bias_m <= 1e-6:
        return [round(px, 4), round(py, 4), round(pz, 4)]

    radial_mag = math.hypot(px, py)
    if radial_mag < 1e-4:
        radial_x, radial_y = 1.0, 0.0
    else:
        radial_x = px / radial_mag
        radial_y = py / radial_mag

    return [
        round(px + center_bias_m * radial_x, 4),
        round(py + center_bias_m * radial_y, 4),
        round(pz, 4),
    ]


def assign_tracks(assembled_frames: List[Dict[str, Any]], key: str, fps: float) -> None:
    next_track_id = 1
    active_tracks: Dict[int, TrackState] = {}
    history: Dict[int, List[Dict[str, Any]]] = {}

    for frame_ordinal, frame in enumerate(assembled_frames):
        detections = frame.get(key, [])
        candidate_pairs: List[Tuple[float, int, int]] = []

        live_tracks = {
            tid: state
            for tid, state in active_tracks.items()
            if frame_ordinal - state.frame_idx <= 4
        }

        for det_idx, det in enumerate(detections):
            for track_id, state in live_tracks.items():
                cost = track_cost(det, state, frame_ordinal - state.frame_idx)
                if cost is None:
                    continue
                candidate_pairs.append((cost, det_idx, track_id))

        assigned_det: set[int] = set()
        assigned_track: set[int] = set()
        for _, det_idx, track_id in sorted(candidate_pairs):
            if det_idx in assigned_det or track_id in assigned_track:
                continue
            detections[det_idx]["track_id"] = track_id
            assigned_det.add(det_idx)
            assigned_track.add(track_id)

        for det_idx, det in enumerate(detections):
            if det_idx not in assigned_det:
                det["track_id"] = next_track_id
                next_track_id += 1

            pos = det.get("position_blender", [0.0, 0.0, 0.0])
            track_id = int(det["track_id"])
            active_tracks[track_id] = TrackState(
                track_id=track_id,
                frame_idx=frame_ordinal,
                canonical_class=str(det["class"]),
                subtype=str(det.get("subclass", det["class"])),
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                bbox=[int(v) for v in det["bbox"]],
                depth_m=float(det.get("depth_m", pos[0])),
                source_track_id=det.get("source_track_id"),
                flow_track_id=det.get("flow_track_id"),
                vehicle_3d_track_id=det.get("vehicle_3d_track_id"),
            )
            history.setdefault(track_id, []).append(det)

    for track_id, detections in history.items():
        canonical_class = str(detections[0]["class"])
        subtype = str(detections[0].get("subclass", canonical_class))
        stable_subtype = choose_stable_vehicle_subtype(canonical_class, detections)
        if stable_subtype is not None:
            subtype = stable_subtype
            for det in detections:
                det["subclass"] = stable_subtype
        seed = f"{canonical_class}:{subtype}:{track_id}"
        asset = choose_asset(canonical_class, subtype, seed)

        scales = [float(det.get("scale", 1.0)) for det in detections]
        median_scale = float(statistics.median(scales)) if scales else 1.0
        yaws = infer_track_yaws(detections)
        smoothed_positions = smooth_track_positions(detections)
        centered_positions = [
            shift_position_away_from_camera(pos, center_bias_for_detection(det))
            for det, pos in zip(detections, smoothed_positions)
        ]
        motion_vectors, motion_speeds, motion_accels = track_motion_samples(centered_positions, fps)
        kinematic_votes: List[bool] = []

        for idx, (det, yaw, smoothed_pos, centered_position) in enumerate(
            zip(detections, yaws, smoothed_positions, centered_positions)
        ):
            center_bias_m = center_bias_for_detection(det)
            motion_vec = motion_vectors[idx] if idx < len(motion_vectors) else [0.0, 0.0, 0.0]
            speed_mps = motion_speeds[idx] if idx < len(motion_speeds) else 0.0
            accel_mps2 = motion_accels[idx] if idx < len(motion_accels) else 0.0
            if canonical_class in STATIC_UPRIGHT_CLASSES:
                motion_vec = [0.0, 0.0, 0.0]
                speed_mps = 0.0
                accel_mps2 = 0.0
                kinematic_moving = False
            else:
                kinematic_moving = speed_mps >= (
                    0.65 if canonical_class in ROAD_CONTEXT_CLASSES else 0.35
                )
            kinematic_votes.append(kinematic_moving)

            moving_flag = det.get("moving")
            parked_flag = det.get("parked")
            if canonical_class in STATIC_UPRIGHT_CLASSES:
                moving_flag = False
                parked_flag = True
            if moving_flag is None:
                moving_flag = kinematic_moving
            if parked_flag is None:
                parked_flag = not bool(moving_flag)

            direction = str(det.get("motion_direction", "") or "").strip().lower()
            if direction in {"", "unknown"}:
                direction = motion_direction_from_blender(float(motion_vec[0]), float(motion_vec[1]))
            braking = False
            if canonical_class in ROAD_CONTEXT_CLASSES:
                if accel_mps2 <= -0.8 and speed_mps >= 0.75:
                    braking = True
                elif idx > 0 and bool(kinematic_votes[idx - 1]) and not bool(kinematic_moving):
                    braking = True
                elif idx > 0 and float(motion_speeds[idx - 1]) - speed_mps >= 0.85:
                    braking = True

            rear_visible = rear_is_visible_from_ego(centered_position, yaw)
            visual_brake = bool(det.get("visual_brake_lights_on", False)) if rear_visible else False
            indicator_left = bool(det.get("visual_indicator_left_on", False)) if rear_visible else False
            indicator_right = bool(det.get("visual_indicator_right_on", False)) if rear_visible else False
            if canonical_class in ROAD_CONTEXT_CLASSES:
                braking = bool(braking or visual_brake)
            turn_signal = None
            if indicator_left and indicator_right:
                turn_signal = "hazard"
            elif indicator_left:
                turn_signal = "left"
            elif indicator_right:
                turn_signal = "right"

            det["uid"] = f"{canonical_class}_{track_id:04d}"
            det["asset"] = asset
            det["scale"] = round(median_scale, 4)
            det["yaw_rad"] = round(float(yaw), 4)
            if canonical_class in STATIC_UPRIGHT_CLASSES and det.get("source_yaw_rad") is None:
                det["yaw_source"] = "ego_facing_semantic"
            det["position_surface_blender"] = list(smoothed_pos)
            det["center_bias_m"] = round(center_bias_m, 4)
            det["position_blender"] = centered_position
            det["motion_vector_blender"] = motion_vec
            det["speed_mps"] = round(float(speed_mps), 4)
            det["accel_mps2"] = round(float(accel_mps2), 4)
            det["motion_direction"] = direction
            det["moving"] = bool(moving_flag)
            det["parked"] = bool(parked_flag)
            det["rear_visible_from_ego"] = bool(rear_visible)
            det["brake_lights_on"] = bool(braking and rear_visible)
            det["indicator_left_on"] = bool(indicator_left)
            det["indicator_right_on"] = bool(indicator_right)
            det["turn_signal"] = turn_signal
            if canonical_class in ROAD_CONTEXT_CLASSES and direction != "stationary":
                det["motion_source"] = (
                    "optical_flow"
                    if det.get("speed_px") is not None or det.get("flow_magnitude_px") is not None
                    else "track_kinematics"
                )


def refine_relative_vehicle_motion(assembled_frames: List[Dict[str, Any]]) -> None:
    for frame in assembled_frames:
        vehicles = [
            det for det in frame.get("objects", [])
            if str(det.get("class")) in ROAD_CONTEXT_CLASSES
            and isinstance(det.get("motion_vector_blender"), list)
            and len(det.get("motion_vector_blender", [])) >= 2
        ]
        if not vehicles:
            continue

        use_cohort_baseline = len(vehicles) >= 2
        if use_cohort_baseline:
            dxs = [float(det["motion_vector_blender"][0]) for det in vehicles]
            dys = [float(det["motion_vector_blender"][1]) for det in vehicles]
            cohort_dx = statistics.median(dxs)
            cohort_dy = statistics.median(dys)
        else:
            cohort_dx = 0.0
            cohort_dy = 0.0

        for det in vehicles:
            raw_vec = det.get("motion_vector_blender", [0.0, 0.0, 0.0])
            rel_dx = float(raw_vec[0]) - float(cohort_dx)
            rel_dy = float(raw_vec[1]) - float(cohort_dy)
            rel_speed = math.hypot(rel_dx, rel_dy)
            det["motion_vector_relative_blender"] = [
                round(rel_dx, 4),
                round(rel_dy, 4),
                0.0,
            ]
            det["speed_relative_mps"] = round(float(rel_speed), 4)

            has_flow_motion = det.get("speed_px") is not None or det.get("flow_magnitude_px") is not None
            if not has_flow_motion and use_cohort_baseline:
                if rel_speed <= 0.40:
                    det["moving"] = False
                    det["parked"] = True
                    det["motion_direction"] = "stationary"
                    det["motion_source"] = "track_residual"
                elif rel_speed >= 0.75:
                    det["moving"] = True
                    det["parked"] = False
                    det["motion_direction"] = motion_direction_from_blender(rel_dx, rel_dy)
                    det["motion_source"] = "track_residual"
            elif not has_flow_motion:
                det["motion_source"] = "track_kinematics"
            elif bool(det.get("moving", False)):
                det["motion_source"] = "optical_flow+track_residual"


def road_context_bounds(frame: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    points: List[Tuple[float, float]] = []

    for contour in frame.get("road", {}).get("contours_3d", []):
        for point in contour:
            if len(point) < 2:
                continue
            x = float(point[0])
            y = float(point[1])
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

    for lane in frame.get("lanes", []):
        for point in lane.get("points_3d", []):
            if len(point) < 2:
                continue
            x = float(point[0])
            y = float(point[1])
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

    if len(points) < 3:
        return None

    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return min(xs), max(xs), min(ys), max(ys)


def prune_objects_outside_scene_context(
    assembled_frames: List[Dict[str, Any]],
    frame_w: int,
) -> None:
    track_lengths: Dict[str, int] = {}
    for frame in assembled_frames:
        for det in frame.get("objects", []):
            uid = str(det.get("uid", ""))
            if uid:
                track_lengths[uid] = track_lengths.get(uid, 0) + 1

    removed = 0
    for frame in assembled_frames:
        bounds = road_context_bounds(frame)
        if bounds is None:
            continue

        min_x, max_x, min_y, max_y = bounds
        filtered_objects: List[Dict[str, Any]] = []

        for det in frame.get("objects", []):
            cls = str(det.get("class", "unknown"))
            if cls not in ROAD_CONTEXT_CLASSES:
                filtered_objects.append(det)
                continue

            pos = det.get("position_blender", [0.0, 0.0, 0.0])
            if len(pos) < 2:
                filtered_objects.append(det)
                continue

            x = float(pos[0])
            y = float(pos[1])
            dims = det.get("dims_m", [1.5, 1.8, 4.5])
            width_m = float(dims[1]) if len(dims) >= 2 else 1.8
            lateral_margin = max(0.9, 0.75 * width_m)
            lateral_overrun = max(min_y - y, y - max_y, 0.0)

            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                x1, _, x2, _ = [int(v) for v in bbox[:4]]
            else:
                x1, x2 = 0, 0
            touches_image_edge = x1 <= 2 or x2 >= (frame_w - 2)

            track_len = track_lengths.get(str(det.get("uid", "")), 1)
            outside_lateral_corridor = y < (min_y - lateral_margin) or y > (max_y + lateral_margin)
            shallow_without_road = x < max(min_x - 0.5, 6.0)
            severe_lateral_violation = lateral_overrun > max(2.2, 1.2 * width_m)

            if severe_lateral_violation:
                removed += 1
                continue

            if outside_lateral_corridor and (touches_image_edge or track_len <= 2 or shallow_without_road):
                removed += 1
                continue

            filtered_objects.append(det)

        frame["objects"] = filtered_objects

    if removed:
        print(f"[assembler] Pruned {removed} object placements outside scene context.")


def visible_pose_keypoint_count(keypoints: Any, conf_thresh: float = 0.20) -> int:
    if not isinstance(keypoints, list):
        return 0
    count = 0
    for kp in keypoints:
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        conf = float(kp[2]) if len(kp) >= 3 else 1.0
        if conf >= conf_thresh:
            count += 1
    return count


def interpolate_pose_keypoints(
    keypoints_a: Sequence[Sequence[float]],
    keypoints_b: Sequence[Sequence[float]],
    alpha: float,
) -> List[List[float]]:
    out: List[List[float]] = []
    n = max(len(keypoints_a), len(keypoints_b))
    for idx in range(n):
        kp_a = list(keypoints_a[idx]) if idx < len(keypoints_a) else []
        kp_b = list(keypoints_b[idx]) if idx < len(keypoints_b) else []
        conf_a = float(kp_a[2]) if len(kp_a) >= 3 else (1.0 if len(kp_a) >= 2 else 0.0)
        conf_b = float(kp_b[2]) if len(kp_b) >= 3 else (1.0 if len(kp_b) >= 2 else 0.0)
        if len(kp_a) >= 2 and len(kp_b) >= 2 and conf_a >= 0.20 and conf_b >= 0.20:
            x = (1.0 - alpha) * float(kp_a[0]) + alpha * float(kp_b[0])
            y = (1.0 - alpha) * float(kp_a[1]) + alpha * float(kp_b[1])
            conf = (1.0 - alpha) * conf_a + alpha * conf_b
            out.append([round(x, 3), round(y, 3), round(conf, 4)])
        elif len(kp_a) >= 2 and conf_a >= 0.20:
            out.append([float(kp_a[0]), float(kp_a[1]), round(conf_a, 4)])
        elif len(kp_b) >= 2 and conf_b >= 0.20:
            out.append([float(kp_b[0]), float(kp_b[1]), round(conf_b, 4)])
        else:
            out.append([0.0, 0.0, 0.0])
    return out


def propagate_pedestrian_keypoints(
    assembled_frames: List[Dict[str, Any]],
    max_gap_frames: int = 18,
) -> None:
    tracks: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for frame in assembled_frames:
        frame_idx = int(frame.get("frame_idx", 0))
        for det in frame.get("objects", []):
            if str(det.get("class", "unknown")) != "pedestrian":
                continue
            uid = str(det.get("uid", ""))
            if not uid:
                continue
            tracks.setdefault(uid, []).append((frame_idx, det))

    propagated = 0

    def pose3d_count(det: Dict[str, Any]) -> int:
        pts = det.get("pose_3d_local") or []
        count = 0
        for kp in pts[:17]:
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                continue
            conf = float(kp[3]) if len(kp) >= 4 else 1.0
            if conf >= 0.18:
                count += 1
        return count

    def clone_pose3d(det: Dict[str, Any]) -> Optional[List[List[float]]]:
        pts = det.get("pose_3d_local") or []
        if not isinstance(pts, list) or not pts:
            return None
        return [list(kp) for kp in pts]

    for uid, history in tracks.items():
        anchors = [
            (idx, frame_idx, det)
            for idx, (frame_idx, det) in enumerate(history)
            if visible_pose_keypoint_count(det.get("keypoints")) >= 6 or pose3d_count(det) >= 6
        ]
        if not anchors:
            continue

        for idx, (frame_idx, det) in enumerate(history):
            if visible_pose_keypoint_count(det.get("keypoints")) >= 6 or pose3d_count(det) >= 6:
                continue

            prev_anchor = next(((a_idx, a_frame, a_det) for a_idx, a_frame, a_det in reversed(anchors) if a_idx < idx), None)
            next_anchor = next(((a_idx, a_frame, a_det) for a_idx, a_frame, a_det in anchors if a_idx > idx), None)

            new_keypoints: Optional[List[List[float]]] = None
            new_pose3d: Optional[List[List[float]]] = None
            pose_backend: Optional[str] = None
            sprite_path: Optional[str] = None
            pose_source = None
            pose_anchor_frame = None

            if prev_anchor is not None and next_anchor is not None:
                prev_idx, prev_frame, prev_det = prev_anchor
                next_idx, next_frame, next_det = next_anchor
                if (frame_idx - prev_frame) <= max_gap_frames and (next_frame - frame_idx) <= max_gap_frames:
                    span = max(1, next_frame - prev_frame)
                    alpha = float(frame_idx - prev_frame) / float(span)
                    prev_kps = prev_det.get("keypoints", [])
                    next_kps = next_det.get("keypoints", [])
                    if prev_kps and next_kps:
                        new_keypoints = interpolate_pose_keypoints(prev_kps, next_kps, alpha)
                    prev_pose3d = clone_pose3d(prev_det)
                    next_pose3d = clone_pose3d(next_det)
                    if prev_pose3d and next_pose3d and len(prev_pose3d) == len(next_pose3d):
                        interp: List[List[float]] = []
                        for a, b in zip(prev_pose3d, next_pose3d):
                            if len(a) < 3 or len(b) < 3:
                                interp.append(list(a if len(a) >= len(b) else b))
                                continue
                            conf_a = float(a[3]) if len(a) >= 4 else 1.0
                            conf_b = float(b[3]) if len(b) >= 4 else 1.0
                            interp.append([
                                round((1.0 - alpha) * float(a[0]) + alpha * float(b[0]), 4),
                                round((1.0 - alpha) * float(a[1]) + alpha * float(b[1]), 4),
                                round((1.0 - alpha) * float(a[2]) + alpha * float(b[2]), 4),
                                round((1.0 - alpha) * conf_a + alpha * conf_b, 4),
                            ])
                        new_pose3d = interp
                    pose_backend = str(prev_det.get("pose_backend") or next_det.get("pose_backend") or "") or None
                    sprite_path = str(prev_det.get("pymaf_sprite_path") or next_det.get("pymaf_sprite_path") or "") or None
                    pose_source = "track_interpolated_pose"
                    pose_anchor_frame = [int(prev_frame), int(next_frame)]
            if new_keypoints is None and new_pose3d is None and prev_anchor is not None:
                _, prev_frame, prev_det = prev_anchor
                if (frame_idx - prev_frame) <= max_gap_frames:
                    prev_kps = prev_det.get("keypoints", [])
                    if prev_kps:
                        new_keypoints = [list(kp) for kp in prev_kps]
                    prev_pose3d = clone_pose3d(prev_det)
                    if prev_pose3d:
                        new_pose3d = prev_pose3d
                    pose_backend = str(prev_det.get("pose_backend") or "") or None
                    sprite_path = str(prev_det.get("pymaf_sprite_path") or "") or None
                    pose_source = "track_propagated_pose"
                    pose_anchor_frame = int(prev_frame)
            if new_keypoints is None and new_pose3d is None and next_anchor is not None:
                _, next_frame, next_det = next_anchor
                if (next_frame - frame_idx) <= max_gap_frames:
                    next_kps = next_det.get("keypoints", [])
                    if next_kps:
                        new_keypoints = [list(kp) for kp in next_kps]
                    next_pose3d = clone_pose3d(next_det)
                    if next_pose3d:
                        new_pose3d = next_pose3d
                    pose_backend = str(next_det.get("pose_backend") or "") or None
                    sprite_path = str(next_det.get("pymaf_sprite_path") or "") or None
                    pose_source = "track_propagated_pose"
                    pose_anchor_frame = int(next_frame)

            if (new_keypoints is not None and visible_pose_keypoint_count(new_keypoints) >= 6) or (new_pose3d is not None and pose3d_count({"pose_3d_local": new_pose3d}) >= 6):
                if new_keypoints is not None:
                    det["keypoints"] = new_keypoints
                if new_pose3d is not None:
                    det["pose_3d_local"] = new_pose3d
                if pose_backend:
                    det["pose_backend"] = pose_backend
                if sprite_path:
                    det["pymaf_sprite_path"] = sprite_path
                det["pose_source"] = pose_source
                det["pose_anchor_frame_idx"] = pose_anchor_frame
                propagated += 1

    if propagated:
        print(f"[assembler] Propagated pedestrian pose to {propagated} additional tracked frame(s).")


def gate_stationary_semantics(
    assembled_frames: List[Dict[str, Any]],
    calib: CalibData,
    frame_w: int,
    frame_h: int,
) -> None:
    """
    Enforce short confirmation and frame-level plausibility on upright semantics.

    Traffic lights and road signs are the most sensitive to bad monocular
    lifting because their bbox often represents only the elevated visible head.
    This pass therefore confirms tracks across time and suppresses individual
    implausible frames instead of stretching one bad pose through the sequence.
    """
    track_lengths: Dict[str, int] = {}
    for frame in assembled_frames:
        for key in ("traffic_lights", "objects"):
            for det in frame.get(key, []):
                cls = str(det.get("class", "unknown"))
                if cls not in STATIC_UPRIGHT_CLASSES:
                    continue
                uid = str(det.get("uid", ""))
                if uid:
                    track_lengths[uid] = track_lengths.get(uid, 0) + 1

    removed = 0
    for frame in assembled_frames:
        for key in ("traffic_lights", "objects"):
            filtered: List[Dict[str, Any]] = []
            for det in frame.get(key, []):
                cls = str(det.get("class", "unknown"))
                if cls not in STATIC_UPRIGHT_CLASSES:
                    filtered.append(det)
                    continue

                uid = str(det.get("uid", ""))
                track_len = track_lengths.get(uid, 1)
                if detection_is_implausible_after_lift(
                    det,
                    calib=calib,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    track_len=track_len,
                ):
                    removed += 1
                    continue
                filtered.append(det)
            frame[key] = filtered

    if removed:
        print(f"[assembler] Suppressed {removed} unstable upright semantic placement(s).")


# ============================================================================
# Assembly
# ============================================================================

def assemble_scene(
    scene: str,
    view: str,
    det_data: Optional[Dict[str, Any]],
    tl_data: Optional[Dict[str, Any]],
    lane_data: Optional[Dict[str, Any]],
    flow_data: Optional[Dict[str, Any]],
    fused_store: LazyFrameJsonStore,
    depth_store: LazyDepthStore,
    calib: CalibData,
    video_path: Optional[Path],
    source_paths: Dict[str, Optional[Path]],
    max_frames: Optional[int] = None,
    pose_data: Optional[Dict[str, Any]] = None,
    vehicle_3d_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    det_index = build_frame_index((det_data or {}).get("frames", []))
    tl_index = build_frame_index((tl_data or {}).get("frames", []))
    lane_index = build_frame_index((lane_data or {}).get("frames", []))
    flow_index = build_flow_index(flow_data)
    pose_index = build_frame_index((pose_data or {}).get("frames", []))
    veh3d_index = build_frame_index((vehicle_3d_data or {}).get("frames", []))

    frame_indices: set[int] = set()
    frame_indices |= set(det_index.keys())
    frame_indices |= set(tl_index.keys())
    frame_indices |= set(lane_index.keys())
    frame_indices |= depth_store.keys()
    frame_indices |= fused_store.keys()

    if not frame_indices:
        print("[assembler] ERROR — no frames found in any source.")
        return {}

    sorted_frame_indices = sorted(frame_indices)
    if max_frames is not None:
        sorted_frame_indices = sorted_frame_indices[:max_frames]

    frame_h, frame_w = 720, 1280
    depth_shape = depth_store.sample_shape()
    if depth_shape is not None:
        frame_h, frame_w = depth_shape
    elif fused_store:
        sample = fused_store.get(sorted_frame_indices[0])
        if sample and "image_hw" in sample and len(sample["image_hw"]) >= 2:
            frame_h = int(sample["image_hw"][0])
            frame_w = int(sample["image_hw"][1])

    fps = infer_fps(video_path, det_data, tl_data, lane_data)
    blender_cam = export_blender_camera(calib, frame_w, frame_h)
    output_layout = scene_output_layout(scene, create=False)
    video_reader = SequentialVideoFrameReader(
        video_path,
        fallback_frame_dirs=[
            output_layout.renders / ".collage_aux",
            output_layout.renders,
            output_layout.frames,
        ],
    )

    print(f"[assembler] Assembling {len(sorted_frame_indices)} frames …")
    assembled_frames: List[Dict[str, Any]] = []

    for ordinal, frame_idx in enumerate(sorted_frame_indices):
        if ordinal % 100 == 0:
            print(f"  [{ordinal:5d}/{len(sorted_frame_indices)}] frame={frame_idx}", end="\r", flush=True)

        depth_map = depth_store.get(frame_idx)

        timestamp_s = frame_idx / fps
        if frame_idx in lane_index and lane_index[frame_idx].get("timestamp_s") is not None:
            timestamp_s = float(lane_index[frame_idx]["timestamp_s"])
        elif frame_idx in det_index and det_index[frame_idx].get("timestamp_s") is not None:
            timestamp_s = float(det_index[frame_idx]["timestamp_s"])
        elif frame_idx in tl_index and tl_index[frame_idx].get("timestamp_s") is not None:
            timestamp_s = float(tl_index[frame_idx]["timestamp_s"])

        frame_bgr = video_reader.get(frame_idx) if video_reader else None

        objects, traffic_lights, fused_frame = collect_frame_detections(
            frame_idx=frame_idx,
            det_index=det_index,
            tl_index=tl_index,
            flow_index=flow_index,
            fused_store=fused_store,
            depth_map=depth_map,
            calib=calib,
            frame_w=frame_w,
            frame_h=frame_h,
        )

        lanes, road = extract_lanes_and_road(
            frame_idx=frame_idx,
            lane_index=lane_index,
            fused_frame=fused_frame,
            depth_map=depth_map,
            calib=calib,
        )

        if depth_map is not None:
            valid = depth_map[np.isfinite(depth_map) & (depth_map > 0.1)]
            depth_stats = {
                "min_m": round(float(np.min(valid)), 3) if valid.size else None,
                "max_m": round(float(np.max(valid)), 3) if valid.size else None,
                "mean_m": round(float(np.mean(valid)), 3) if valid.size else None,
                "median_m": round(float(np.median(valid)), 3) if valid.size else None,
            }
        else:
            depth_stats = dict((fused_frame or {}).get("depth_stats", {}))

        # ── Merge pose keypoints into pedestrian objects ─────────────────
        pose_frame = pose_index.get(frame_idx)
        if pose_frame:
            pose_peds = pose_frame.get("pedestrians", [])
            for obj in objects:
                if obj.get("class") != "pedestrian":
                    continue
                obj_bbox = obj.get("bbox", [0, 0, 0, 0])
                best_score, best_kps, best_pose = -1e9, None, None
                ox1, oy1, ox2, oy2 = [float(v) for v in obj_bbox[:4]]
                ocx = 0.5 * (ox1 + ox2)
                ocy = 0.5 * (oy1 + oy2)
                ow = max(ox2 - ox1, 1.0)
                oh = max(oy2 - oy1, 1.0)
                for pp in pose_peds:
                    pp_bbox = pp.get("bbox", [0, 0, 0, 0])
                    iou = bbox_iou(obj_bbox, pp_bbox)
                    if len(pp_bbox) >= 4:
                        px1, py1, px2, py2 = [float(v) for v in pp_bbox[:4]]
                        pcx = 0.5 * (px1 + px2)
                        pcy = 0.5 * (py1 + py2)
                    else:
                        pcx, pcy = ocx, ocy
                    center_cost = math.hypot((pcx - ocx) / ow, (pcy - ocy) / oh)
                    visible_kps = visible_pose_keypoint_count(pp.get("keypoints"), conf_thresh=0.18)
                    score = 2.2 * float(iou) - 0.45 * float(center_cost) + 0.03 * float(visible_kps)
                    if score > best_score:
                        best_score = score
                        best_kps = pp.get("keypoints")
                        best_pose = pp
                if best_pose is not None:
                    obj["pose_backend"] = (best_pose or {}).get("pose_backend")
                pose_3d_local = (best_pose or {}).get("pose_3d_local")
                pymaf_sprite_path = str((best_pose or {}).get("pymaf_sprite_path", "") or "").strip()
                pymaf_like = bool(pymaf_sprite_path) or (isinstance(pose_3d_local, list) and len(pose_3d_local) >= 6)
                if best_score >= 0.08 or (best_pose is not None and pymaf_like and best_score >= -0.02):
                    if best_kps:
                        obj["keypoints"] = best_kps
                    if isinstance(pose_3d_local, list) and pose_3d_local:
                        obj["pose_3d_local"] = pose_3d_local
                    if pymaf_sprite_path:
                        obj["pymaf_sprite_path"] = pymaf_sprite_path
                    smpl_pose = best_pose.get("smpl_pose")
                    if isinstance(smpl_pose, list) and smpl_pose:
                        obj["smpl_pose"] = smpl_pose
                    smpl_betas = best_pose.get("smpl_betas")
                    if isinstance(smpl_betas, list) and smpl_betas:
                        obj["smpl_betas"] = smpl_betas
                    pymaf_track_id = best_pose.get("pymaf_track_id")
                    if pymaf_track_id is not None:
                        obj["pymaf_track_id"] = pymaf_track_id
                    obj["pose_source"] = (best_pose or {}).get("source", "pose_match")

        # ── Merge 3D vehicle data into vehicle objects ──────────────────
        veh3d_frame = veh3d_index.get(frame_idx)
        if veh3d_frame:
            veh3d_list = veh3d_frame.get("vehicles_3d", [])
            for obj in objects:
                if obj.get("class") not in ROAD_CONTEXT_CLASSES:
                    continue
                obj_bbox = obj.get("bbox", [0, 0, 0, 0])
                best_iou, best_v3d = 0.0, None
                for v3d in veh3d_list:
                    v_bbox = v3d.get("bbox_2d", [0, 0, 0, 0])
                    iou = bbox_iou(obj_bbox, v_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_v3d = v3d
                if best_iou >= 0.30 and best_v3d:
                    if best_v3d.get("position_3d") and len(best_v3d.get("position_3d", [])) >= 3:
                        try:
                            v3d_pos = [
                                float(best_v3d["position_3d"][0]),
                                float(best_v3d["position_3d"][1]),
                                float(best_v3d["position_3d"][2]),
                            ]
                            if (
                                0.2 < v3d_pos[0] < MAX_VALID_DEPTH_M
                                and abs(v3d_pos[1]) <= max(18.0, 0.90 * v3d_pos[0] + 6.0)
                            ):
                                reconciled = reconcile_vehicle_3d_position(
                                    obj,
                                    v3d_pos,
                                    calib=calib,
                                    frame_w=frame_w,
                                    frame_h=frame_h,
                                    v3d_depth=float(best_v3d["depth_m"]) if best_v3d.get("depth_m") is not None else None,
                                    v3d_dims=best_v3d.get("dimensions_3d"),
                                )
                                if reconciled is not None:
                                    obj["vehicle_3d_position_blender"] = list(reconciled["position_blender"])
                                    obj["vehicle_3d_depth_m"] = round(float(reconciled["depth_m"]), 4)
                                    obj["vehicle_3d_position_source"] = str(reconciled["position_source"])
                                    if obj.get("position_source") is None:
                                        obj["position_source"] = "bbox_depth"
                                    closure = reconciled.get("projection_closure")
                                    if isinstance(closure, dict):
                                        obj["vehicle_3d_projection_closure"] = {
                                            "h_ratio": round(float(closure["h_ratio"]), 4),
                                            "w_ratio": round(float(closure["w_ratio"]), 4),
                                            "center_dx_norm": round(float(closure["center_dx_norm"]), 4),
                                            "center_dy_norm": round(float(closure["center_dy_norm"]), 4),
                                        }
                                    consistency = reconciled.get("position_consistency")
                                    if isinstance(consistency, dict):
                                        obj["position_consistency"] = consistency
                        except (TypeError, ValueError, IndexError):
                            pass
                    if best_v3d.get("orientation_rad") is not None:
                        obj["yaw_rad"] = round(float(best_v3d["orientation_rad"]), 4)
                        obj["vehicle_3d_yaw_rad"] = round(float(best_v3d["orientation_rad"]), 4)
                        obj["yaw_source"] = "vehicle_3d"
                    if best_v3d.get("id") is not None:
                        obj["vehicle_3d_track_id"] = best_v3d.get("id")
                    if best_v3d.get("dimensions_3d"):
                        obj["dims_m"] = [round(float(d), 3) for d in best_v3d["dimensions_3d"]]
                    if best_v3d.get("bbox_3d_corners"):
                        obj["bbox_3d_corners"] = best_v3d["bbox_3d_corners"]
                    if best_v3d.get("subclass"):
                        obj["subclass"] = best_v3d["subclass"]

        if frame_bgr is not None:
            for obj in objects:
                if obj.get("class") not in ROAD_CONTEXT_CLASSES:
                    continue
                obj.update(infer_vehicle_signal_state(frame_bgr, obj))

        assembled_frames.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp_s": round(float(timestamp_s), 4),
                "objects": objects,
                "traffic_lights": traffic_lights,
                "lanes": lanes,
                "road": road,
                "depth_stats": depth_stats,
            }
        )

    print(f"\n[assembler] Track association on {len(assembled_frames)} frames …")
    assign_tracks(assembled_frames, "objects", fps=fps)
    assign_tracks(assembled_frames, "traffic_lights", fps=fps)
    propagate_pedestrian_keypoints(assembled_frames)
    refine_relative_vehicle_motion(assembled_frames)
    prune_objects_outside_scene_context(assembled_frames, frame_w)
    gate_stationary_semantics(assembled_frames, calib, frame_w, frame_h)
    video_reader.close()

    object_counts: Dict[str, int] = {}
    for frame in assembled_frames:
        for det in frame["objects"] + frame["traffic_lights"]:
            object_counts[det["class"]] = object_counts.get(det["class"], 0) + 1

    return {
        "meta": {
            "scene": scene,
            "view": view,
            "total_frames": len(assembled_frames),
            "frame_w": frame_w,
            "frame_h": frame_h,
            "fps": round(float(fps), 4),
            "video_path": resolve_path(video_path),
            "depth_npz_path": resolve_path(source_paths.get("depth_npz")),
            "frame_json_dir": resolve_path(source_paths.get("frame_json_dir")),
            "camera": blender_cam,
            "calib": {
                "fx": calib.fx,
                "fy": calib.fy,
                "cx": calib.cx,
                "cy": calib.cy,
                "camera_height_m": calib.camera_height_m,
                "pitch_rad": calib.pitch_rad,
            },
            "coordinate_frame": {
                "origin": "ego_vehicle_base_center",
                "axes": {
                    "x": "forward",
                    "y": "left",
                    "z": "up",
                },
                "legacy_scene_positions": "camera_centric",
                "blender_scene_offset_source": "meta.ego_vehicle.scene_world_offset_blender",
            },
            "ego_vehicle": build_ego_vehicle_meta(view, calib),
            "depth_geometry": {
                "mesh_stride_px": 36,
                "crop_top_frac": 0.42,
                "crop_bottom_frac": 0.30,
                "bbox_margin_px": 28,
                "min_depth_m": 6.0,
                "foreground_depth_m": 7.5,
                "foreground_row_start_frac": 0.56,
                "foreground_bottom_boost_m": 10.0,
                "max_depth_m": 38.0,
            },
            "render_policy": {
                "clean_synthetic_only": True,
                "allow_background_plate": False,
                "allow_source_textures": False,
                "allow_reference_collage": False,
                "allow_pose_sprites": False
            },
            "sources": {
                "detections_json": resolve_path(source_paths.get("det_json")),
                "traffic_lights_json": resolve_path(source_paths.get("tl_json")),
                "lanes_json": resolve_path(source_paths.get("lane_json")),
                "depth_npz": resolve_path(source_paths.get("depth_npz")),
                "flow_json": resolve_path(source_paths.get("flow_json")),
                "pose_json": resolve_path(source_paths.get("pose_json")),
                "vehicle_3d_json": resolve_path(source_paths.get("vehicle_3d_json")),
                "frame_json_dir": resolve_path(source_paths.get("frame_json_dir")),
            },
            "object_counts": object_counts,
        },
        "frames": assembled_frames,
    }


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble a Blender-ready scene description for one sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", default="scene1", help="Scene folder name under P3Data/Sequences/")
    parser.add_argument("--view", default="front", choices=["front", "back", "left", "right"])
    parser.add_argument("--data-root", default="P3Data", help="Root of the P3Data directory")
    parser.add_argument("--renders-dir", default=None, help="Preferred renders directory to search first")
    parser.add_argument("--frame-json-dir", default=None, help="Optional directory of per-frame fused JSONs")
    parser.add_argument("--det-json", default=None, help="Optional aggregated object detections JSON")
    parser.add_argument("--tl-json", default=None, help="Optional aggregated traffic-lights JSON")
    parser.add_argument("--lane-json", default=None, help="Optional lane + road JSON")
    parser.add_argument("--depth-npz", default=None, help="Optional depth npz path")
    parser.add_argument(
        "--out",
        default=None,
        help="Output assembled scene JSON (default: output/scene_data/<scene>/scene_assembled.json)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit to the first N frames")
    parser.add_argument("--force-reload-calib", action="store_true")
    args = parser.parse_args()

    args.scene = infer_scene_name(args.scene, args.frame_json_dir, args.det_json, args.tl_json, args.lane_json, args.depth_npz)
    data_root = Path(args.data_root).resolve()
    renders_dir = Path(args.renders_dir).resolve() if args.renders_dir else (_THIS / "renders" / args.scene)
    output_layout = scene_output_layout(args.scene, create=True)
    discovered = discover_paths(args.scene, args.view, data_root, renders_dir)

    source_paths: Dict[str, Optional[Path]] = {
        "frame_json_dir": Path(args.frame_json_dir).resolve() if args.frame_json_dir else discovered["frame_json_dir"],
        "det_json": Path(args.det_json).resolve() if args.det_json else discovered["det_json"],
        "tl_json": Path(args.tl_json).resolve() if args.tl_json else discovered["tl_json"],
        "lane_json": Path(args.lane_json).resolve() if args.lane_json else discovered["lane_json"],
        "depth_npz": Path(args.depth_npz).resolve() if args.depth_npz else discovered["depth_npz"],
        "flow_json": discovered["flow_json"],
        "pose_json": discovered.get("pose_json"),
        "vehicle_3d_json": discovered.get("vehicle_3d_json"),
        "video": discovered["video"],
    }

    out_path = (
        Path(args.out).resolve()
        if args.out
        else (output_layout.scene_data / "scene_assembled.json")
    )

    print("\n" + "═" * 72)
    print("  RBE549 / CS549 P3 — Scene Assembler")
    print(f"  scene={args.scene}  view={args.view}")
    print(f"  data_root     = {data_root}")
    print(f"  renders_dir   = {renders_dir}")
    print(f"  frame_json    = {source_paths['frame_json_dir'] or 'NOT FOUND'}")
    print(f"  detections    = {source_paths['det_json'] or 'NOT FOUND'}")
    print(f"  traffic lights= {source_paths['tl_json'] or 'NOT FOUND'}")
    print(f"  lanes / road  = {source_paths['lane_json'] or 'NOT FOUND'}")
    print(f"  depth npz     = {source_paths['depth_npz'] or 'NOT FOUND'}")
    print(f"  optical flow  = {source_paths['flow_json'] or 'NOT FOUND'}")
    print(f"  pose keypts   = {source_paths['pose_json'] or 'NOT FOUND'}")
    print(f"  vehicle 3D    = {source_paths['vehicle_3d_json'] or 'NOT FOUND'}")
    print(f"  video         = {source_paths['video'] or 'NOT FOUND'}")
    print(f"  output        = {out_path}")
    print("═" * 72 + "\n")

    det_data = load_json(source_paths["det_json"], "detections")
    tl_data = load_json(source_paths["tl_json"], "traffic_lights")
    lane_data = load_json(source_paths["lane_json"], "lane+road")
    flow_data = load_json(source_paths["flow_json"], "optical_flow")
    pose_data = load_json(source_paths.get("pose_json"), "pose_keypoints")
    vehicle_3d_data = load_json(source_paths.get("vehicle_3d_json"), "vehicle_3d")
    fused_store = load_frame_json_store(source_paths["frame_json_dir"])
    depth_store = load_depth_store(source_paths["depth_npz"])

    print(f"\n[assembler] Loading calibration for '{args.view}' …")
    calib = load_calibration(args.view, force_reload=args.force_reload_calib)
    print(f"  {calib}")

    scene_data = assemble_scene(
        scene=args.scene,
        view=args.view,
        det_data=det_data,
        tl_data=tl_data,
        lane_data=lane_data,
        flow_data=flow_data,
        fused_store=fused_store,
        depth_store=depth_store,
        calib=calib,
        video_path=source_paths["video"],
        source_paths=source_paths,
        max_frames=args.max_frames,
        pose_data=pose_data,
        vehicle_3d_data=vehicle_3d_data,
    )

    depth_store.close()

    if not scene_data:
        print("[assembler] FATAL — no scene data produced.")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene_data, indent=2))
    if not args.out:
        mirror_stage_output(out_path, args.scene, "scene_data", out_path.name)
    size_mb = out_path.stat().st_size / (1024 * 1024)

    frames = scene_data.get("frames", [])
    total_objects = sum(len(frame.get("objects", [])) for frame in frames)
    total_tl = sum(len(frame.get("traffic_lights", [])) for frame in frames)
    total_lanes = sum(len(frame.get("lanes", [])) for frame in frames)
    depth_frames = sum(1 for frame in frames if frame.get("depth_stats"))

    print(f"\n[assembler] Scene JSON written → {out_path} ({size_mb:.1f} MB)")
    print(f"  Frames          : {len(frames)}")
    print(f"  Objects         : {total_objects}")
    print(f"  Traffic lights  : {total_tl}")
    print(f"  Lane polylines  : {total_lanes}")
    print(f"  Depth maps used : {depth_frames}")
    print(
        "\n  Next step:\n"
        f"    blender --background --python blender.py -- --scene-json {out_path}"
    )


if __name__ == "__main__":
    main()
