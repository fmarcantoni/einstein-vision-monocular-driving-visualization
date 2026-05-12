"""
calibration.py
==============

Camera calibration, projection, and coordinate-conversion utilities for the
autonomous-driving pipeline.

Responsibilities
----------------
* load per-view intrinsics from ``P3Data/Calib/<view>/intrinsics.json``
* provide stable defaults when calibration files are missing
* undistort frames consistently across the perception modules
* convert pixels and depth into camera/world/Blender coordinates
* export Blender camera metadata compatible with ``scene_assembler.py`` and ``blender.py``

Design notes
------------
This project assumes a simple vehicle-centric camera model with the camera at
approximately ``(0, 0, 1.5 m)`` and near-zero pitch when no stronger metadata
is available.  Those assumptions are encoded explicitly here so every
downstream stage shares the same geometry convention instead of re-implementing
its own fallback behavior.
"""

from __future__ import annotations

import json
import math
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

_THIS_DIR  = Path(__file__).parent.resolve()
_CALIB_DIR = _THIS_DIR / "P3Data" / "Calib"

_VALID_VIEWS: Tuple[str, ...] = ("front", "back", "left", "right")

# ── Project-report assumptions (Section I / Checkpoint 1) ────────────────────
#   "the camera is located at (0, 0, 1.5) m in the world frame"
#   zero pitch assumed throughout
_DEFAULT_CAMERA_HEIGHT_M = 1.5
_DEFAULT_PITCH_RAD = 0.0

# ── Fallback intrinsics — Tesla Model S front camera, 1280×720 ───────────────
#   fx ≈ 910 px  (matches ~70° horizontal FoV at 1280 px wide)
_FALLBACK_K: Dict[str, np.ndarray] = {
    "front": np.array([[910.,   0., 640.],
                       [  0., 910., 360.],
                       [  0.,   0.,   1.]], dtype=np.float64),
    "back":  np.array([[820.,   0., 640.],
                       [  0., 820., 360.],
                       [  0.,   0.,   1.]], dtype=np.float64),
    "left":  np.array([[820.,   0., 640.],
                       [  0., 820., 360.],
                       [  0.,   0.,   1.]], dtype=np.float64),
    "right": np.array([[820.,   0., 640.],
                       [  0., 820., 360.],
                       [  0.,   0.,   1.]], dtype=np.float64),
}


# ══════════════════════════════════════════════════════════════════════════════
# CalibData
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class CalibData:
    """Intrinsics + extrinsic assumptions for one camera view."""

    view: str
    K: np.ndarray   # (3,3) float64
    dist: np.ndarray   # (1,N) float64 — zeros when unknown
    fx: float
    fy: float
    cx: float
    cy: float
    source: str = "unknown"
    camera_height_m: float = _DEFAULT_CAMERA_HEIGHT_M
    pitch_rad: float = _DEFAULT_PITCH_RAD

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "view": self.view,
            "source": self.source,
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "camera_height_m": self.camera_height_m,
            "pitch_rad": self.pitch_rad,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CalibData":
        dist_raw = d.get("dist", [[0., 0., 0., 0., 0.]])
        return cls(
            view=d["view"],
            K=np.array(d["K"], dtype=np.float64),
            dist=np.array(dist_raw, dtype=np.float64),
            fx=float(d["fx"]), fy=float(d["fy"]),
            cx=float(d["cx"]), cy=float(d["cy"]),
            source=d.get("source", "cached"),
            camera_height_m=float(d.get("camera_height_m", _DEFAULT_CAMERA_HEIGHT_M)),
            pitch_rad=float(d.get("pitch_rad", _DEFAULT_PITCH_RAD)),
        )

    def __str__(self) -> str:
        return (f"CalibData({self.view}, fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, src={self.source})")


# ══════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(view: str) -> Path:
    return _CALIB_DIR / view / "intrinsics.json"


def _load_cached(view: str) -> Optional[CalibData]:
    p = _cache_path(view)
    if not p.exists():
        return None
    try:
        calib = CalibData.from_dict(json.loads(p.read_text()))
        # Basic sanity on the focal lengths
        if not (50 < calib.fx < 20_000 and 50 < calib.fy < 20_000):
            print(f"[calibration] '{view}': cached K looks wrong — ignoring.")
            return None
        return calib
    except Exception as exc:
        print(f"[calibration] Cache parse error for '{view}': {exc}")
        return None


def _save_cache(calib: CalibData) -> None:
    folder = _cache_path(calib.view).parent
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "intrinsics.json"
    path.write_text(json.dumps(calib.to_dict(), indent=2))
    print(f"[calibration] Cached → {path.relative_to(_THIS_DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# Public: load_calibration / load_all_calibrations
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(
    view: str,
    force_reload: bool = False,
    camera_height_m: float = _DEFAULT_CAMERA_HEIGHT_M,
    pitch_rad: float = _DEFAULT_PITCH_RAD,
) -> CalibData:
    """
    Return a CalibData for *view*.

    Resolution order
    ----------------
    1. intrinsics.json cache in  P3Data/Calib/<view>/  (fast path)
    2. Hard-coded fallback defaults

    The cache is written after step 2 so that subsequent calls hit step 1.

    Parameters
    ----------
    view : one of "front" | "back" | "left" | "right"
    force_reload : skip the cache and re-derive from defaults
    camera_height_m : override the stored camera height (metres)
    pitch_rad : override the stored pitch angle (radians)
    """
    if view not in _VALID_VIEWS:
        raise ValueError(f"Unknown view '{view}'. Choose from {_VALID_VIEWS}")

    # ── 1. Cache ──────────────────────────────────────────────────────────────
    if not force_reload:
        cached = _load_cached(view)
        if cached is not None:
            # Allow runtime overrides without mutating the cache
            if camera_height_m != _DEFAULT_CAMERA_HEIGHT_M:
                cached.camera_height_m = camera_height_m
            if pitch_rad != _DEFAULT_PITCH_RAD:
                cached.pitch_rad = pitch_rad
            print(f"[calibration] '{view}' loaded from cache  —  {cached}")
            return cached

    # ── 2. Fallback defaults (report: camera at (0,0,1.5 m), zero pitch) ─────
    K  = _FALLBACK_K.get(view, _FALLBACK_K["front"]).copy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    dist   = np.zeros((1, 5), dtype=np.float64)

    calib = CalibData(
        view=view, K=K, dist=dist,
        fx=fx, fy=fy, cx=cx, cy=cy,
        source="fallback_default",
        camera_height_m=camera_height_m,
        pitch_rad=pitch_rad,
    )
    print(f"[calibration] '{view}' using fallback defaults  —  {calib}")
    _save_cache(calib)
    return calib


def load_all_calibrations(
    force_reload: bool = False,
    camera_heights: Optional[Dict[str, float]] = None,
    pitch_angles:   Optional[Dict[str, float]] = None,
) -> Dict[str, CalibData]:
    """Load all four views.  Returns only those that succeed."""
    result: Dict[str, CalibData] = {}
    for view in _VALID_VIEWS:
        h = (camera_heights or {}).get(view, _DEFAULT_CAMERA_HEIGHT_M)
        p = (pitch_angles   or {}).get(view, _DEFAULT_PITCH_RAD)
        try:
            result[view] = load_calibration(view, force_reload, h, p)
        except Exception as exc:
            print(f"[calibration] WARNING — skipping '{view}': {exc}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Undistortion
# ══════════════════════════════════════════════════════════════════════════════

# Module-level cache: (view, W, H, alpha) → (map1, map2)
_map_cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}


def get_undistort_maps(
    calib: CalibData, W: int, H: int, alpha: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cv2 undistortion remap tables."""
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        calib.K, calib.dist, (W, H), alpha, (W, H))
    m1, m2 = cv2.initUndistortRectifyMap(
        calib.K, calib.dist, None, new_K, (W, H), cv2.CV_16SC2)
    return m1, m2


def undistort_fast(
    frame: np.ndarray, calib: CalibData, alpha: float = 0.0
) -> np.ndarray:
    """
    Undistort using pre-computed remap tables.
    First call per (view, W, H, alpha) computes; subsequent calls are O(1).
    Preferred for all video-processing loops.
    """
    H, W = frame.shape[:2]
    key = (calib.view, W, H, alpha)
    if key not in _map_cache:
        _map_cache[key] = get_undistort_maps(calib, W, H, alpha)
    m1, m2 = _map_cache[key]
    return cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)


def undistort_frame(
    frame: np.ndarray, calib: CalibData, alpha: float = 0.0
) -> np.ndarray:
    """Single-shot undistortion (no map cache). Use undistort_fast for video."""
    H, W = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        calib.K, calib.dist, (W, H), alpha, (W, H))
    out = cv2.undistort(frame, calib.K, calib.dist, None, new_K)
    if alpha == 0.0:
        x, y, w, h = roi
        if w > 0 and h > 0:
            out = cv2.resize(out[y:y+h, x:x+w], (W, H))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2-D → 3-D projection
# ══════════════════════════════════════════════════════════════════════════════

def pixel_to_camera_ray(
    u: float, v: float, calib: CalibData
) -> Tuple[float, float, float]:
    """Normalised ray in camera space (Z = 1 plane)."""
    return (
        (u - calib.cx) / calib.fx,
        (v - calib.cy) / calib.fy,
        1.0,
    )


def pixel_to_world(
    u: float, v: float, depth_m: float, calib: CalibData
) -> Tuple[float, float, float]:
    """
    Back-project pixel (u, v) to camera-space 3-D using metric depth.

    Camera convention: Z = forward, X = right, Y = down.

    Parameters
    ----------
    u, v : pixel coordinates
    depth_m : metric depth along the Z (optical) axis, in metres
    calib : CalibData for this view

    Returns
    -------
    (X, Y, Z) in camera space, metres
    """
    Z = float(depth_m)
    X = (float(u) - calib.cx) * Z / calib.fx
    Y = (float(v) - calib.cy) * Z / calib.fy
    return X, Y, Z


def pixel_to_ground(
    u: float, v: float, calib: CalibData,
    camera_height_m: Optional[float] = None,
    pitch_rad: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Intersect the ray through pixel (u, v) with the ground plane (Y_world = 0).
    Returns camera-space (X, Y, Z) in metres, or None if ray is parallel/behind.

    This is used by the lane module to map 2-D lane pixels to 3-D positions
    under the flat-ground assumption.
    """
    h = camera_height_m if camera_height_m is not None else calib.camera_height_m
    pitch = pitch_rad if pitch_rad is not None else calib.pitch_rad

    rx, ry, rz = pixel_to_camera_ray(u, v, calib)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # Transform ray direction into world frame (Y-up)
    ray_xw =  rx
    ray_yw = -sp * rz - cp * ry
    ray_zw =  cp * rz - sp * ry

    if abs(ray_yw) < 1e-9:
        return None   # Ray is horizontal — no intersection
    t = -h / ray_yw
    if t <= 0:
        return None   # Intersection is behind the camera

    return rx * t, ry * t, rz * t


# ══════════════════════════════════════════════════════════════════════════════
# Coordinate-system helpers
# ══════════════════════════════════════════════════════════════════════════════

def camera_to_blender(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """
    Camera space (Z-fwd, X-right, Y-down)  →  Blender world (X-fwd, Y-left, Z-up).
    """
    return float(Z), -float(X), max(0.0, -float(Y))


def blender_to_camera(bx: float, by: float, bz: float) -> Tuple[float, float, float]:
    """Inverse of camera_to_blender."""
    return -by, -bz, bx


# ══════════════════════════════════════════════════════════════════════════════
# Depth map → point cloud
# ══════════════════════════════════════════════════════════════════════════════

def depth_map_to_pointcloud(
    depth_map: np.ndarray,
    calib: CalibData,
    max_depth_m: float = 80.0,
    min_depth_m: float = 0.5,
    stride: int = 4,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    Convert a (H, W) depth map to a (N, 3) float32 camera-space XYZ array.

    Parameters
    ----------
    depth_map  : raw depth values (metric metres when depth_scale=1.0)
    depth_scale : scalar applied to raw depth values before use
    stride      : sub-sample every Nth pixel for speed

    Returns
    -------
    (N, 3) float32 in camera space [X, Y, Z], metres
    """
    H, W = depth_map.shape[:2]
    depth = depth_map.astype(np.float32) * depth_scale

    cols = np.arange(0, W, stride, dtype=np.float32)
    rows = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    zz = depth[::stride, ::stride]
    mask = (zz > min_depth_m) & (zz < max_depth_m)

    uu_v, vv_v, zz_v = uu[mask], vv[mask], zz[mask]
    if uu_v.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    XX = (uu_v - calib.cx) * zz_v / calib.fx
    YY = (vv_v - calib.cy) * zz_v / calib.fy
    return np.stack([XX, YY, zz_v], axis=-1).astype(np.float32)


def depth_map_to_pointcloud_rgb(
    depth_map: np.ndarray,
    frame_bgr: np.ndarray,
    calib: CalibData,
    max_depth_m: float = 80.0,
    min_depth_m: float = 0.5,
    stride: int = 4,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    (H, W) depth + BGR frame → (N, 6) float32  [X, Y, Z, R, G, B].
    Useful for coloured point-cloud export (e.g. PLY / Blender).
    """
    H, W = depth_map.shape[:2]
    depth = depth_map.astype(np.float32) * depth_scale
    if frame_bgr.shape[:2] != (H, W):
        frame_bgr = cv2.resize(frame_bgr, (W, H))

    cols = np.arange(0, W, stride, dtype=np.float32)
    rows = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(cols, rows)
    zz  = depth[::stride, ::stride]
    bgr = frame_bgr[::stride, ::stride].astype(np.float32)
    mask = (zz > min_depth_m) & (zz < max_depth_m)

    uu_v, vv_v, zz_v = uu[mask], vv[mask], zz[mask]
    b_v = bgr[..., 0][mask]
    g_v = bgr[..., 1][mask]
    r_v = bgr[..., 2][mask]
    if uu_v.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    XX = (uu_v - calib.cx) * zz_v / calib.fx
    YY = (vv_v - calib.cy) * zz_v / calib.fy
    return np.stack([XX, YY, zz_v, r_v, g_v, b_v], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline integration: enrich detections with 3-D positions
# ══════════════════════════════════════════════════════════════════════════════

def detections_to_3d(
    detections: list,
    depth_map: np.ndarray,
    calib: CalibData,
    depth_scale: float = 1.0,
) -> list:
    """
    Enrich DetectionResult / dict objects **in-place** with 3-D positions.

    For each detection the median depth of the inner 50 % of the bounding box
    is used as the metric distance, then back-projected via pixel_to_world.
    Both dataclass (det.bbox, det.depth_m, det.position_3d) and dict
    (det["bbox"], det["depth_m"], det["position_3d"]) forms are accepted.

    Parameters
    ----------
    detections : list of DetectionResult or dicts with a "bbox" key
    depth_map : (H, W) depth array (metric metres when depth_scale=1.0)
    calib : CalibData for this frame
    depth_scale : multiply raw depth values to get metres

    Returns
    -------
    The same list, modified in-place.
    """
    H, W = depth_map.shape[:2]
    for det in detections:
        bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
        if bbox is None:
            continue

        x1 = max(0, min(int(bbox[0]), W - 1))
        y1 = max(0, min(int(bbox[1]), H - 1))
        x2 = max(0, min(int(bbox[2]), W - 1))
        y2 = max(0, min(int(bbox[3]), H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        # Use the central 50% of the bounding box to avoid background bleed
        cy_lo = int(0.25 * (y2 - y1)) + y1
        cy_hi = int(0.75 * (y2 - y1)) + y1
        cx_lo = int(0.25 * (x2 - x1)) + x1
        cx_hi = int(0.75 * (x2 - x1)) + x1
        inner = depth_map[cy_lo:cy_hi, cx_lo:cx_hi].astype(np.float32)
        valid = inner[inner > 0]
        if valid.size == 0:
            continue

        depth_val = float(np.median(valid)) * depth_scale
        if depth_val <= 0:
            continue

        uc = (x1 + x2) / 2.0
        vc = (y1 + y2) / 2.0
        X, Y, Z = pixel_to_world(uc, vc, depth_val, calib)
        bx, by, bz = camera_to_blender(X, Y, Z)

        if hasattr(det, "depth_m"):          # dataclass branch
            det.depth_m = Z
            det.position_3d = [bx, by, bz]
        else:                                 # dict branch
            det["depth_m"] = round(Z, 3)
            det["position_3d"] = [round(bx, 3), round(by, 3), round(bz, 3)]

    return detections


# ══════════════════════════════════════════════════════════════════════════════
# Blender export helpers
# ══════════════════════════════════════════════════════════════════════════════

def export_blender_camera(
    calib: CalibData,
    W: int, H: int,
    out_path: Optional[Union[str, Path]] = None,
    sensor_width_mm: float = 36.0,
) -> Dict:
    """
    Compute Blender camera parameters that match the physical camera.

    Returns a dict suitable for JSON export; optionally writes it to disk.
    """
    focal_mm = calib.fx * sensor_width_mm / W
    shift_x  = (calib.cx - W / 2.0) / W
    shift_y  = (calib.cy - H / 2.0) / H

    pitch = calib.pitch_rad
    cp, sp = math.cos(pitch), math.sin(pitch)
    R = [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]

    d = {
        "view":                    calib.view,
        "resolution":              {"W": W, "H": H},
        "sensor_width_mm":         sensor_width_mm,
        "focal_length_mm":         round(focal_mm, 4),
        "shift_x":                 round(shift_x, 6),
        "shift_y":                 round(shift_y, 6),
        "clip_start":              0.1,
        "clip_end":                200.0,
        "camera_height_m":         calib.camera_height_m,
        "pitch_rad":               calib.pitch_rad,
        "location_blender":        [0.0, 0.0, calib.camera_height_m],
        "rotation_matrix_blender": R,
        "K":                       calib.K.tolist(),
        "dist":                    calib.dist.tolist(),
    }
    if out_path is not None:
        Path(out_path).write_text(json.dumps(d, indent=2))
        print(f"[calibration] Blender camera params → {out_path}")
    return d


def blender_camera_script(
    calib: CalibData, W: int, H: int, cam_name: str = "PipelineCam"
) -> str:
    """Return a ready-to-paste Blender Python snippet that creates the camera."""
    p = export_blender_camera(calib, W, H)
    return f"""
import bpy, mathutils, math
cam_data = bpy.data.cameras.new("{cam_name}_data")
cam_obj  = bpy.data.objects.new("{cam_name}", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj
cam_data.lens         = {p['focal_length_mm']:.4f}
cam_data.sensor_width = {p['sensor_width_mm']:.2f}
cam_data.shift_x      = {p['shift_x']:.6f}
cam_data.shift_y      = {p['shift_y']:.6f}
cam_data.clip_start   = {p['clip_start']}
cam_data.clip_end     = {p['clip_end']}
bpy.context.scene.render.resolution_x = {W}
bpy.context.scene.render.resolution_y = {H}
cam_obj.location = mathutils.Vector({p['location_blender']})
cam_obj.rotation_euler = mathutils.Euler(
    (math.pi/2 + {p['pitch_rad']:.6f}, 0, 0), 'XYZ')
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _put_label(img: np.ndarray, text: str, pos: Tuple[int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, 0.9, (0, 0, 0),     3, cv2.LINE_AA)
    cv2.putText(img, text, pos,                  font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def visualise_undistortion(
    frame: np.ndarray, calib: CalibData, alpha: float = 0.0, width: int = 0
) -> np.ndarray:
    """Return a side-by-side BGR comparison of the original and undistorted frame."""
    undist = undistort_frame(frame, calib, alpha)
    H, W = frame.shape[:2]
    tw = width if width > 0 else W
    th = int(H * tw / W)
    left  = cv2.resize(frame,  (tw, th))
    right = cv2.resize(undist, (tw, th))
    _put_label(left,  "Original",    (10, 30))
    _put_label(right, "Undistorted", (10, 30))
    return np.hstack([left, right])


def visualise_pointcloud_bev(
    cloud: np.ndarray,
    calib: CalibData,
    canvas_size: int   = 600,
    max_range_m: float = 50.0,
    point_radius: int  = 2,
) -> np.ndarray:
    """Top-down bird-eye-view render of a camera-space XYZ point cloud."""
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if cloud.shape[0] == 0:
        _put_label(canvas, "No points", (10, 30))
        return canvas

    X, Z = cloud[:, 0], cloud[:, 2]
    mask = (Z > 0) & (Z < max_range_m) & (np.abs(X) < max_range_m * 0.5)
    X, Z = X[mask], Z[mask]
    if X.size == 0:
        _put_label(canvas, "No points in range", (10, 30))
        return canvas

    px = ((X / max_range_m) * (canvas_size / 2) + canvas_size / 2).astype(int)
    py = (canvas_size - (Z / max_range_m) * canvas_size).astype(int)
    norm = np.clip(Z / max_range_m, 0, 1)
    hsv = np.zeros((len(Z), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = ((1 - norm) * 120).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 220
    colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)

    for i in range(len(px)):
        if 0 <= px[i] < canvas_size and 0 <= py[i] < canvas_size:
            cv2.circle(canvas, (int(px[i]), int(py[i])), point_radius,
                       (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), -1)

    # Ego vehicle marker
    ego_x, ego_y = canvas_size // 2, canvas_size - 20
    cv2.rectangle(canvas, (ego_x - 8, ego_y - 15), (ego_x + 8, ego_y + 5), (200, 200, 200), -1)

    # Distance grid lines
    for d in range(10, int(max_range_m), 10):
        y = int(canvas_size - (d / max_range_m) * canvas_size)
        cv2.line(canvas, (0, y), (canvas_size, y), (40, 40, 40), 1)
        cv2.putText(canvas, f"{d}m", (5, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.line(canvas, (canvas_size // 2, 0), (canvas_size // 2, canvas_size), (40, 40, 40), 1)
    _put_label(canvas, f"BEV [{calib.view}]  {len(X)} pts", (8, 22))
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# __main__ — quick self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv

    print("\n" + "═" * 68)
    print("  RBE549 / CS549 P3 — Einstein Vision — Calibration Module")
    if force:
        print("  (force-reload: rebuilding from fallback defaults)")
    print("═" * 68 + "\n")

    all_c = load_all_calibrations(force_reload=force)

    print(f"\n  {'View':<8}  {'fx':>9}  {'fy':>9}  {'cx':>9}  {'cy':>9}  source")
    print("  " + "─" * 62)
    for v, c in all_c.items():
        print(f"  {v:<8}  {c.fx:>9.2f}  {c.fy:>9.2f}  "
              f"{c.cx:>9.2f}  {c.cy:>9.2f}  {c.source}")

    # Sanity checks on front camera
    c0 = all_c["front"]

    ray = pixel_to_camera_ray(c0.cx, c0.cy, c0)
    print(f"\n  Ray at principal pt: {ray}  (expect ≈ (0, 0, 1))")

    pt = pixel_to_world(c0.cx + 50, c0.cy, 10.0, c0)
    print(f"  pixel_to_world (cx+50, cy, 10 m): {pt}")
    print(f"  → camera_to_blender: {camera_to_blender(*pt)}")

    pg = pixel_to_ground(c0.cx, c0.cy + 200, c0)
    print(f"  pixel_to_ground (cx, cy+200): {pg}")

    params = export_blender_camera(c0, 1280, 720)
    print(f"\n  Blender focal length: {params['focal_length_mm']:.2f} mm")
    print(f"  Camera height: {params['camera_height_m']} m  pitch: {params['pitch_rad']} rad")