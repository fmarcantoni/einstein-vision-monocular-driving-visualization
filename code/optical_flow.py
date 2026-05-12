"""
optical_flow.py
===============

Dense optical-flow estimation for motion analysis in the driving pipeline.

Responsibilities
----------------
* estimate per-pixel motion between consecutive frames
* provide object-level motion summaries inside detection bounding boxes
* convert image-space motion into approximate metric velocity when calibration and depth are available
* export optional visualizations and compressed ``.npz`` flow archives

Backend strategy
----------------
``RAFT`` is the preferred backend because it is the strongest dense-flow model available in the stack. 
``Farneback`` is preserved as a CPU fallback so motion estimation remains available even without the full PyTorch model path.

Output contract
---------------
All public flow fields are returned as ``(H, W, 2) float32`` arrays in pixel units.  
Higher-level helpers build on top of that primitive representation rather than changing it, which keeps downstream integration predictable.
"""

from __future__ import annotations

import os
import time
import warnings
import dataclasses
import json
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union
)

import cv2
import numpy as np

from project_setup import infer_scene_name, mirror_stage_output, scene_output_layout

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def reencode_for_vscode(src_video: Path, dst_video: Path) -> bool:
    if not ffmpeg_available():
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst_video),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False


class SafeVideoWriter:
    """
    Write temporary MJPG AVI output, then re-encode to H.264 MP4 when possible.

    The side-by-side flow visualisation is wider than the source frames, and
    some OpenCV MP4 codec combinations are unreliable for those dimensions.
    This keeps the flow stage aligned with the safer video strategy used by the
    lane and traffic-light modules.
    """

    def __init__(self, requested_output: Path, fps: float, width: int, height: int) -> None:
        self.requested_output = requested_output
        requested_output.parent.mkdir(parents=True, exist_ok=True)
        self.temp_video = requested_output.with_suffix(".tmp.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = None
        writer_errors: List[str] = []

        for backend_name, backend in [
            ("CAP_OPENCV_MJPEG", getattr(cv2, "CAP_OPENCV_MJPEG", None)),
            ("CAP_FFMPEG", getattr(cv2, "CAP_FFMPEG", None)),
            ("default", None),
        ]:
            try:
                if backend is None:
                    candidate = cv2.VideoWriter(str(self.temp_video), fourcc, fps, (width, height))
                else:
                    candidate = cv2.VideoWriter(str(self.temp_video), backend, fourcc, fps, (width, height))
                if candidate is not None and candidate.isOpened():
                    self.writer = candidate
                    break
                if candidate is not None:
                    candidate.release()
            except cv2.error as exc:
                writer_errors.append(f"{backend_name}: {exc}")

        if self.writer is None:
            detail = f" Details: {' | '.join(writer_errors)}" if writer_errors else ""
            raise RuntimeError(f"Could not open MJPG AVI writer.{detail}")

        print(f"[SafeVideoWriter] Writing temp AVI -> {self.temp_video}")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            raise RuntimeError("SafeVideoWriter is not open.")
        self.writer.write(frame)

    def close(self) -> Path:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.requested_output.suffix.lower() == ".mp4" and reencode_for_vscode(self.temp_video, self.requested_output):
            try:
                self.temp_video.unlink(missing_ok=True)
            except Exception:
                pass
            print(f"[SafeVideoWriter] Re-encoded -> {self.requested_output}")
            return self.requested_output

        fallback = self.requested_output.with_suffix(".avi")
        if fallback.exists():
            fallback.unlink()
        self.temp_video.rename(fallback)
        print(f"[SafeVideoWriter] Using AVI fallback -> {fallback}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Magnitude threshold (pixels/frame) above which an object is "moving"
_MOTION_THRESHOLD_PX = 1.5

# Arrow visualisation grid spacing (pixels)
_ARROW_GRID_STEP = 20

# Maximum arrow length drawn (pixels) — longer vectors are clamped
_ARROW_MAX_DRAW_LEN = 25

# HSV wheel max magnitude for colour saturation (pixels/frame)
_FLOW_VIS_MAX_MAG = 20.0

# RAFT model variants to try, in order of preference
_RAFT_VARIANTS = ["raft_large", "raft_small"]

STATIC_OBJECT_CLASSES = {"traffic_light", "stop_sign"}
_RESIDUAL_MOVING_THRESH = 1.15
_RESIDUAL_PARKED_THRESH = 0.55
_CENTER_MOVING_THRESH = 0.45
_CENTER_PARKED_THRESH = 0.20
_CONF_MOVING_THRESH = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# FlowResult  — per-object motion dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class FlowResult:
    """
    Motion information for a single detected object in one frame pair.

    Fields
    ------
    det_id : matches DetectionResult.id (or dict["id"])
    bbox : [x1, y1, x2, y2] pixel bounding box
    mean_dx : mean horizontal flow inside bbox (pixels/frame), + = right
    mean_dy : mean vertical   flow inside bbox (pixels/frame), + = down
    magnitude : sqrt(mean_dx² + mean_dy²)  pixels/frame
    angle_deg : flow direction, degrees CCW from right (0 = rightward)
    is_moving : True when magnitude > _MOTION_THRESHOLD_PX
    velocity_mps: estimated metric speed (m/s); None until filled by estimate_velocity() or align_flow_to_detections()

    JSON-compatible via .to_dict().
    """
    det_id: int
    bbox: List[int]
    mean_dx: float
    mean_dy: float
    magnitude: float
    angle_deg: float
    is_moving: bool
    velocity_mps: Optional[float] = None
    class_name: Optional[str] = None
    moving: Optional[bool] = None
    parked: Optional[bool] = None
    direction: Optional[str] = None
    speed_px: Optional[float] = None
    motion_confidence: Optional[float] = None
    residual_dx: Optional[float] = None
    residual_dy: Optional[float] = None
    track_id: Optional[Any] = None
    flow_track_id: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "det_id": self.det_id,
            "bbox": self.bbox,
            "mean_dx": round(self.mean_dx,  3),
            "mean_dy": round(self.mean_dy,  3),
            "magnitude": round(self.magnitude, 3),
            "angle_deg": round(self.angle_deg, 2),
            "is_moving": self.is_moving,
            "velocity_mps": round(self.velocity_mps, 3)
                            if self.velocity_mps is not None else None,
            "class_name": self.class_name,
            "moving": self.moving,
            "parked": self.parked,
            "direction": self.direction,
            "speed_px": round(self.speed_px, 3) if self.speed_px is not None else None,
            "motion_confidence": round(self.motion_confidence, 3) if self.motion_confidence is not None else None,
            "residual_dx": round(self.residual_dx, 3) if self.residual_dx is not None else None,
            "residual_dy": round(self.residual_dy, 3) if self.residual_dy is not None else None,
            "track_id": self.track_id,
            "flow_track_id": self.flow_track_id,
        }


VEHICLE_MOTION_CLASSES = {"car", "truck", "motorcycle", "bicycle"}


def _canonical_det_class(det: Any) -> str:
    raw = getattr(det, "class", None)
    if raw is None and isinstance(det, dict):
        raw = det.get("class") or det.get("class_name")
    return str(raw or "unknown").strip().lower()


def _is_vehicle_detection(det: Any) -> bool:
    return _canonical_det_class(det) in VEHICLE_MOTION_CLASSES


def _det_bbox(det: Any) -> Optional[List[int]]:
    bbox = getattr(det, "bbox", None)
    if bbox is None and isinstance(det, dict):
        bbox = det.get("bbox")
    if bbox is None or len(bbox) < 4:
        return None
    return [int(v) for v in list(bbox)[:4]]


def _det_id(det: Any) -> int:
    raw = getattr(det, "id", None)
    if raw is None and isinstance(det, dict):
        raw = det.get("id", 0)
    return int(raw or 0)


def _det_track_id(det: Any):
    raw = getattr(det, "track_id", None)
    if raw is None and isinstance(det, dict):
        raw = det.get("track_id")
    return raw


def _bbox_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [int(v) for v in box_b[:4]]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-6)


def center_of(box: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def estimate_background_motion_from_flow(
    flow: np.ndarray,
    boxes: List[Sequence[int]],
    margin: int = 6,
    sample_step: int = 8,
) -> np.ndarray:
    H, W = flow.shape[:2]
    mask = np.ones((H, W), dtype=bool)
    for bbox in boxes:
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(W, x2 + margin)
        y2 = min(H, y2 + margin)
        mask[y1:y2, x1:x2] = False

    sampled_mask = mask[::sample_step, ::sample_step]
    sampled_flow = flow[::sample_step, ::sample_step]
    valid = sampled_flow[sampled_mask]
    if valid.size == 0:
        valid = flow.reshape(-1, 2)
    if valid.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.median(valid, axis=0).astype(np.float32)


def direction_from_vec(dx: float, dy: float, mag_thresh: float = 0.75) -> str:
    mag = float(np.hypot(dx, dy))
    if mag < mag_thresh:
        return "stationary"

    ang = float(np.degrees(np.arctan2(-dy, dx)))
    bins = [
        (-22.5, 22.5, "right"),
        (22.5, 67.5, "up-right"),
        (67.5, 112.5, "up"),
        (112.5, 157.5, "up-left"),
        (157.5, 180.0, "left"),
        (-180.0, -157.5, "left"),
        (-157.5, -112.5, "down-left"),
        (-112.5, -67.5, "down"),
        (-67.5, -22.5, "down-right"),
    ]
    for lo, hi, name in bins:
        if lo <= ang < hi:
            return name
    return "unknown"


def _motion_confidence(
    residual_dx: float,
    residual_dy: float,
    center_dx: float,
    center_dy: float,
    local_mag: float,
) -> float:
    residual_mag = float(np.hypot(residual_dx, residual_dy))
    center_mag = float(np.hypot(center_dx, center_dy))

    residual_term = np.clip(residual_mag / 3.0, 0.0, 1.0)
    center_term = np.clip(center_mag / 2.0, 0.0, 1.0)
    activity_term = np.clip(local_mag / 4.0, 0.0, 1.0)

    agree_term = 0.0
    if residual_mag > 1e-6 and center_mag > 1e-6:
        v1 = np.array([residual_dx, residual_dy], dtype=np.float32) / residual_mag
        v2 = np.array([center_dx, center_dy], dtype=np.float32) / center_mag
        agree = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        agree_term = 0.5 * (agree + 1.0)

    conf = 0.45 * residual_term + 0.25 * center_term + 0.20 * agree_term + 0.10 * activity_term
    return float(np.clip(conf, 0.0, 1.0))


@dataclasses.dataclass
class VehicleMotionTrackState:
    flow_track_id: int
    class_name: str
    bbox: List[int]
    center: np.ndarray
    last_seen: int
    residuals: deque = dataclasses.field(default_factory=lambda: deque(maxlen=12))
    center_drifts: deque = dataclasses.field(default_factory=lambda: deque(maxlen=12))
    speed_hist: deque = dataclasses.field(default_factory=lambda: deque(maxlen=12))
    conf_hist: deque = dataclasses.field(default_factory=lambda: deque(maxlen=12))
    motion_votes: deque = dataclasses.field(default_factory=lambda: deque(maxlen=8))
    direction_votes: deque = dataclasses.field(default_factory=lambda: deque(maxlen=8))
    speed_px: float = 0.0
    moving: bool = False
    parked: bool = False
    direction: str = "unknown"
    motion_confidence: float = 0.0
    track_id: Optional[Any] = None


class VehicleMotionSmoother:
    def __init__(
        self,
        parked_speed_thresh: float = _RESIDUAL_MOVING_THRESH,
        parked_history: int = 4,
        max_age: int = 10,
        center_dist_thresh: float = 80.0,
        iou_thresh: float = 0.15,
    ) -> None:
        self.parked_speed_thresh = parked_speed_thresh
        self.parked_history = parked_history
        self.max_age = max_age
        self.center_dist_thresh = center_dist_thresh
        self.iou_thresh = iou_thresh
        self.frame_counter = 0
        self.tracks: Dict[int, VehicleMotionTrackState] = {}
        self.next_track_id = 1

    def _cleanup(self) -> None:
        dead = [tid for tid, tr in self.tracks.items() if (self.frame_counter - tr.last_seen) > self.max_age]
        for tid in dead:
            del self.tracks[tid]

    def _match_track(self, det: Any) -> Optional[VehicleMotionTrackState]:
        bbox = _det_bbox(det)
        cls_name = _canonical_det_class(det)
        if bbox is None:
            return None
        center = center_of(bbox)
        det_track_id = _det_track_id(det)

        if det_track_id is not None:
            for tr in self.tracks.values():
                if tr.class_name == cls_name and tr.track_id == det_track_id:
                    return tr

        best_tid = None
        best_score = -1e9
        for tid, tr in self.tracks.items():
            if tr.class_name != cls_name:
                continue
            iou = _bbox_iou(bbox, tr.bbox)
            cdist = float(np.linalg.norm(center - tr.center))
            if iou < self.iou_thresh and cdist > self.center_dist_thresh:
                continue
            score = iou - 0.002 * cdist
            if score > best_score:
                best_score = score
                best_tid = tid
        return self.tracks.get(best_tid) if best_tid is not None else None

    def annotate(self, detections: List[Dict[str, Any]], flow: np.ndarray) -> None:
        self.frame_counter += 1
        self._cleanup()
        all_boxes = [_det_bbox(det) for det in detections]
        all_boxes = [b for b in all_boxes if b is not None]
        background_vec = estimate_background_motion_from_flow(flow, all_boxes)

        for det in detections:
            bbox = _det_bbox(det)
            cls_name = _canonical_det_class(det)
            if bbox is None:
                continue

            local_dx, local_dy = get_bbox_flow(bbox, flow, inner_fraction=0.6)
            local_mag = float(np.hypot(local_dx, local_dy))

            det["motion_vec"] = [round(local_dx, 3), round(local_dy, 3)]
            det["mean_dx"] = round(local_dx, 3)
            det["mean_dy"] = round(local_dy, 3)
            det["magnitude_px"] = round(local_mag, 3)
            det["angle_deg"] = round(float(np.degrees(np.arctan2(local_dy, local_dx))), 2)

            if cls_name not in VEHICLE_MOTION_CLASSES:
                det["moving"] = None
                det["parked"] = None
                det["direction"] = None
                det["speed_px"] = None
                det["motion_confidence"] = None
                det["flow_track_id"] = None
                det["residual_vec"] = None
                det["is_moving"] = False
                continue

            center = center_of(bbox)
            tr = self._match_track(det)
            if tr is None:
                tr = VehicleMotionTrackState(
                    flow_track_id=self.next_track_id,
                    class_name=cls_name,
                    bbox=list(bbox),
                    center=center.copy(),
                    last_seen=self.frame_counter,
                    track_id=_det_track_id(det),
                )
                self.tracks[self.next_track_id] = tr
                self.next_track_id += 1

            center_dx = float(center[0] - tr.center[0])
            center_dy = float(center[1] - tr.center[1])
            residual_dx = float(local_dx - background_vec[0])
            residual_dy = float(local_dy - background_vec[1])
            residual_mag = float(np.hypot(residual_dx, residual_dy))
            conf = _motion_confidence(
                residual_dx=residual_dx,
                residual_dy=residual_dy,
                center_dx=center_dx,
                center_dy=center_dy,
                local_mag=local_mag,
            )

            tr.last_seen = self.frame_counter
            tr.bbox = list(bbox)
            tr.center = center.copy()
            tr.track_id = _det_track_id(det)

            tr.residuals.append(np.array([residual_dx, residual_dy], dtype=np.float32))
            tr.center_drifts.append(np.array([center_dx, center_dy], dtype=np.float32))
            tr.speed_hist.append(residual_mag)
            tr.conf_hist.append(conf)

            smooth_residual = np.mean(np.stack(list(tr.residuals), axis=0), axis=0)
            smooth_center = np.mean(np.stack(list(tr.center_drifts), axis=0), axis=0)
            smooth_residual_dx, smooth_residual_dy = float(smooth_residual[0]), float(smooth_residual[1])
            smooth_center_dx, smooth_center_dy = float(smooth_center[0]), float(smooth_center[1])
            smooth_residual_mag = float(np.hypot(smooth_residual_dx, smooth_residual_dy))
            smooth_center_mag = float(np.hypot(smooth_center_dx, smooth_center_dy))
            smooth_conf = float(np.mean(tr.conf_hist))

            direction = direction_from_vec(smooth_residual_dx, smooth_residual_dy, mag_thresh=0.5)
            moving_evidence = (
                smooth_residual_mag >= max(self.parked_speed_thresh, _RESIDUAL_MOVING_THRESH)
                and smooth_center_mag >= _CENTER_MOVING_THRESH
                and smooth_conf >= _CONF_MOVING_THRESH
            ) or (
                smooth_residual_mag >= 1.75 and smooth_conf >= 0.55
            )
            parked_evidence = (
                smooth_residual_mag <= _RESIDUAL_PARKED_THRESH
                and smooth_center_mag <= _CENTER_PARKED_THRESH
            )

            if moving_evidence:
                tr.motion_votes.append("moving")
            elif parked_evidence:
                tr.motion_votes.append("parked")
            else:
                tr.motion_votes.append("moving" if tr.moving else "parked")

            tr.direction_votes.append(direction)
            moving_votes = sum(v == "moving" for v in tr.motion_votes)
            parked_votes = sum(v == "parked" for v in tr.motion_votes)

            if moving_evidence and len(tr.motion_votes) <= 2:
                tr.moving = True
                tr.parked = False
            elif parked_evidence and len(tr.motion_votes) <= 2 and smooth_conf <= 0.35:
                tr.moving = False
                tr.parked = True
            elif moving_votes >= max(3, len(tr.motion_votes) // 2):
                tr.moving = True
                tr.parked = False
            elif parked_votes >= max(self.parked_history, len(tr.motion_votes) // 2):
                tr.moving = False
                tr.parked = True

            valid_dirs = [d for d in tr.direction_votes if d != "stationary"]
            tr.direction = max(valid_dirs, key=valid_dirs.count) if valid_dirs else ("stationary" if tr.parked else direction)
            if tr.parked:
                tr.direction = "stationary"

            tr.speed_px = smooth_residual_mag
            tr.motion_confidence = smooth_conf

            det["moving"] = bool(tr.moving)
            det["parked"] = bool(tr.parked)
            det["direction"] = tr.direction
            det["speed_px"] = round(tr.speed_px, 4)
            det["motion_confidence"] = round(tr.motion_confidence, 4)
            det["flow_track_id"] = int(tr.flow_track_id)
            det["residual_vec"] = [round(smooth_residual_dx, 3), round(smooth_residual_dy, 3)]


# ─────────────────────────────────────────────────────────────────────────────
# OpticalFlowEstimator
# ─────────────────────────────────────────────────────────────────────────────

class OpticalFlowEstimator:
    """
    Dense optical flow estimator.

    Tries RAFT (torchvision) first; falls back to Farneback (OpenCV, CPU-only).

    Parameters
    ----------
    device : "auto" | "cuda" | "mps" | "cpu"
    model_type : "raft" | "farneback" | "auto"
    """

    def __init__(
        self,
        device: str = "auto",
        model_type: str = "auto",
    ) -> None:
        self.device = _resolve_device(device)
        self._backend = None     # "raft" | "farneback"
        self._model = None

        if model_type in ("raft", "auto"):
            if self._try_load_raft():
                return

        if model_type in ("farneback", "auto"):
            self._backend = "farneback"
            print("[OpticalFlowEstimator] Using Farneback (OpenCV, CPU)")
            return

        raise RuntimeError("[OpticalFlowEstimator] No backend could be loaded.")

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _try_load_raft(self) -> bool:
        try:
            import torch
            from torchvision.models.optical_flow import (
                raft_large, raft_small,
                Raft_Large_Weights, Raft_Small_Weights,
            )
        except ImportError as e:
            print(f"[OpticalFlowEstimator] RAFT unavailable ({e}) — "
                  "install: pip install torchvision")
            return False

        for variant in _RAFT_VARIANTS:
            try:
                print(f"[OpticalFlowEstimator] Loading RAFT ({variant}) …")
                if variant == "raft_large":
                    model = raft_large(weights=Raft_Large_Weights.DEFAULT)
                else:
                    model = raft_small(weights=Raft_Small_Weights.DEFAULT)
                model.eval()
                if self.device != "cpu":
                    model = model.to(self.device)
                self._model = model
                self._backend = "raft"
                self._raft_variant = variant
                print(f"[OpticalFlowEstimator] RAFT ({variant}) ready  "
                      f"device='{self.device}'")
                return True
            except Exception as exc:
                print(f"[OpticalFlowEstimator] RAFT ({variant}) failed: {exc}")

        return False

    # ── Core inference ────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend or "none"

    def estimate(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Compute dense optical flow from prev_bgr → curr_bgr.

        Parameters
        ----------
        prev_bgr : (H, W, 3) uint8 — previous frame
        curr_bgr : (H, W, 3) uint8 — current  frame

        Returns
        -------
        flow : (H, W, 2) float32
            flow[y, x, 0] = dx (pixels, + rightward)
            flow[y, x, 1] = dy (pixels, + downward)
        """
        if self._backend == "raft":
            return self._estimate_raft(prev_bgr, curr_bgr)
        return self._estimate_farneback(prev_bgr, curr_bgr)

    def _estimate_raft(
        self,
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
    ) -> np.ndarray:
        import torch

        H, W = prev_bgr.shape[:2]

        def to_tensor(bgr: np.ndarray) -> "torch.Tensor":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            return t.unsqueeze(0)  # (1, 3, H, W)

        t1 = to_tensor(prev_bgr)
        t2 = to_tensor(curr_bgr)

        # RAFT requires H and W to be divisible by 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h or pad_w:
            t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h))
            t2 = torch.nn.functional.pad(t2, (0, pad_w, 0, pad_h))

        if self.device != "cpu":
            t1 = t1.to(self.device)
            t2 = t2.to(self.device)

        with torch.no_grad():
            # RAFT returns a list of refinement predictions; take the last
            preds = self._model(t1, t2)
            flow = preds[-1]  # (1, 2, H_pad, W_pad)

        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()  # (H_pad, W_pad, 2)

        # Crop back to original size
        if pad_h or pad_w:
            flow_np = flow_np[:H, :W]

        return flow_np.astype(np.float32)

    @staticmethod
    def _estimate_farneback(
        prev_bgr: np.ndarray,
        curr_bgr: np.ndarray,
    ) -> np.ndarray:
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            None,
            pyr_scale=0.5,
            # ``levels=5`` produced unstable flow magnitudes on the project
            # videos and made downstream visualisation unexpectedly slow.
            # ``levels=3`` is materially more stable here while remaining fast.
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow.astype(np.float32)  # (H, W, 2)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualise(
        self,
        flow: np.ndarray,
        frame_bgr: Optional[np.ndarray] = None,
        draw_arrows: bool = True,
        highlight_motion: bool = True,
        detections: Optional[list] = None,
        flow_results: Optional[List[FlowResult]] = None,
    ) -> np.ndarray:
        """
        Render a BGR visualisation of the flow field.

        Parameters
        ----------
        flow : (H, W, 2) flow array
        frame_bgr : if given, blend flow HSV over this frame
        draw_arrows : overlay sparse arrow grid
        highlight_motion : tint moving regions green
        detections : list of DetectionResult / dicts — draws bboxes
        flow_results : list of FlowResult — draws per-object motion vectors

        Returns
        -------
        (H, W, 3) uint8 BGR annotated frame
        """
        vis = flow_to_rgb(flow, alpha=0.6, overlay_frame=frame_bgr)

        if highlight_motion:
            _draw_motion_highlight(vis, flow)

        if draw_arrows:
            _draw_flow_arrows(vis, flow)

        if flow_results:
            _draw_flow_results(vis, flow_results)
        elif detections:
            # Draw bare bboxes if no flow_results yet
            for det in detections:
                bbox = getattr(det, "bbox", None) or (
                    det.get("bbox") if isinstance(det, dict) else None)
                if bbox:
                    cv2.rectangle(vis,
                                  (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  (200, 200, 200), 1)

        return vis

    # ── Video processing ──────────────────────────────────────────────────────

    def stream_video(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[
        Tuple[int, float, np.ndarray, np.ndarray, np.ndarray],
        None, None,
    ]:
        """
        Lazy generator — yields one processed frame pair at a time.

        Yields
        ------
        (frame_idx, timestamp_s, prev_bgr, curr_bgr, flow)
            frame_idx : 0-based index of the *processed* frame pair
            timestamp_s : wall-clock position of curr_bgr in the source video
            prev_bgr : previous BGR frame (H, W, 3)
            curr_bgr : current  BGR frame (H, W, 3)
            flow : dense flow prev→curr, (H, W, 2) float32, pixels

        Example
        -------
            for idx, ts, prev, curr, flow in estimator.stream_video("drive.mp4"):
                dets = detector.detect(curr)
                align_flow_to_detections(dets, flow, fps=30.0, calib=calib)
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        processed = 0
        src_idx = 0
        prev_frame: Optional[np.ndarray] = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                if prev_frame is None:
                    prev_frame = frame
                    src_idx += 1
                    continue

                ts = src_idx / src_fps
                flow = self.estimate(prev_frame, frame)
                yield processed, ts, prev_frame, frame, flow

                prev_frame = frame
                processed += 1
                src_idx += 1

                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        out_video: Optional[str] = "renders/flow_output.mp4",
        out_json: Optional[str] = "renders/flow_results.json",
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        layout: str = "side_by_side",
        frame_hook: Optional[
            Callable[
                [np.ndarray, np.ndarray, np.ndarray, list],
                None,
            ]
        ] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a full video file end-to-end.

        Parameters
        ----------
        video_path : path to input video
        out_video : annotated output video; None = skip
        out_json : JSON with per-frame flow stats; None = skip
        frame_skip : process every Nth source frame
        max_frames : stop after N processed frame pairs
        layout : "side_by_side" | "overlay" | "flow_only"
        frame_hook : callable(prev_bgr, curr_bgr, flow, detections=[]) → None
                      called after flow estimation, before writing.
                      Pass detections to fill motion_vec / velocity in-place.

        Returns
        -------
        Dict[int, np.ndarray]  —  {pair_idx: flow (H, W, 2) float32}
        """
        return VideoFlowProcessor(self).run(
            video_path=video_path,
            out_video=out_video,
            out_json=out_json,
            frame_skip=frame_skip,
            max_frames=max_frames,
            layout=layout,
            frame_hook=frame_hook,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VideoFlowProcessor
# ─────────────────────────────────────────────────────────────────────────────

class VideoFlowProcessor:
    """
    End-to-end video optical flow processor.

    Reads consecutive frame pairs, runs flow estimation, composes annotated
    output video (original | flow-HSV side-by-side by default), and saves
    a JSON summary of per-frame statistics.
    """

    def __init__(self, estimator: OpticalFlowEstimator) -> None:
        self.estimator = estimator

    # ── I/O helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _open_writer(path: Path, fps: float, W: int, H: int) -> SafeVideoWriter:
        return SafeVideoWriter(path, fps, W, H)

    @staticmethod
    def _make_canvas(
        frame_bgr: np.ndarray,
        flow: np.ndarray,
        layout: str,
    ) -> np.ndarray:
        """Compose the output frame according to layout."""
        flow_vis = flow_to_rgb(flow, alpha=0.65, overlay_frame=frame_bgr)
        _draw_flow_arrows(flow_vis, flow)

        if layout == "flow_only":
            return flow_vis
        if layout == "overlay":
            return cv2.addWeighted(frame_bgr, 0.45, flow_vis, 0.55, 0)

        # side_by_side (default)
        return np.hstack([frame_bgr, flow_vis])

    @staticmethod
    def _draw_hud(
        canvas: np.ndarray,
        pair_idx: int,
        proc_fps: float,
        flow: np.ndarray,
    ) -> None:
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        lines = [
            f"Pair   : {pair_idx}",
            f"FPS    : {proc_fps:5.1f}",
            f"Fl med : {float(np.median(mag)):.2f} px",
            f"Fl max : {float(mag.max()):.2f} px",
        ]
        x0, y0, lh = 8, 22, 20
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 18),
                      (x0 + 178, y0 + lh * len(lines) + 8),
                      (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x0, y0 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (210, 230, 255), 1, cv2.LINE_AA)

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        video_path: str,
        out_video: Optional[str] = "renders/flow_output.mp4",
        out_json: Optional[str] = "renders/flow_results.json",
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        layout: str = "side_by_side",
        frame_hook: Optional[Callable] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a video end-to-end.

        Returns
        -------
        Dict[int, np.ndarray]  —  pair_idx → flow (H, W, 2) float32
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {src}")

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = src_fps / max(frame_skip, 1)
        out_W = src_W * 2 if layout == "side_by_side" else src_W

        writer: Optional[SafeVideoWriter] = None
        if out_video is not None:
            p = Path(out_video)
            p.parent.mkdir(parents=True, exist_ok=True)
            writer = self._open_writer(p, out_fps, out_W, src_H)

        if out_json is not None:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*66}")
        print(f"  VideoFlowProcessor.run()")
        print(f"  Input   : {src}  ({src_W}×{src_H} @ {src_fps:.1f} fps, "
              f"~{src_total} frames)")
        print(f"  Backend : {self.estimator.backend}  "
              f"device={self.estimator.device}")
        print(f"  Layout  : {layout}  →  {out_W}×{src_H}")
        if out_video: print(f"  Video   : {out_video}")
        if out_json:  print(f"  JSON    : {out_json}")
        print(f"  Skip    : every {frame_skip} frame(s)  →  ~{out_fps:.1f} fps")
        print(f"{'='*66}\n")

        flow_store: Dict[int, np.ndarray] = {}
        json_records: List[Dict] = []
        fps_window: List[float] = []
        processed = 0
        src_idx = 0
        prev_frame: Optional[np.ndarray] = None
        t_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                if prev_frame is None:
                    prev_frame = frame
                    src_idx += 1
                    continue

                ts = src_idx / src_fps
                t0 = time.time()

                # ── Flow estimation ──────────────────────────────────────────
                flow = self.estimator.estimate(prev_frame, frame)
                flow_store[processed] = flow

                # ── Optional hook ────────────────────────────────────────────
                detections: list = []
                if frame_hook is not None:
                    frame_hook(prev_frame, frame, flow, detections)

                # ── JSON record ──────────────────────────────────────────────
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
                record: Dict[str, Any] = {
                    "pair_idx": processed,
                    "frame_idx": int(src_idx),
                    "timestamp_s": round(ts, 4),
                    "median_flow_px": round(float(np.median(mag)), 3),
                    "max_flow_px": round(float(mag.max()), 3),
                    "object_flows": [],
                }
                # Attach per-object flows if detections were enriched by hook
                for det in detections:
                    mv = (getattr(det, "motion_vec", None)
                          or (det.get("motion_vec") if isinstance(det, dict) else None))
                    if mv is not None:
                        record["object_flows"].append(_build_object_flow_record(det, flow))
                json_records.append(record)

                # ── Compose output frame ─────────────────────────────────────
                if writer is not None:
                    canvas = self._make_canvas(frame, flow, layout)
                    if detections:
                        _draw_flow_results(
                            canvas if layout == "flow_only" else
                            canvas[:, src_W:] if layout == "side_by_side" else canvas,
                            _detections_to_flow_results(detections, flow),
                        )
                    fps_window.append(time.time() - t0)
                    if len(fps_window) > 20:
                        fps_window.pop(0)
                    proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)
                    self._draw_hud(canvas, processed, proc_fps, flow)
                    writer.write(canvas)

                # ── Progress ─────────────────────────────────────────────────
                prev_frame = frame
                processed += 1
                src_idx += 1

                if processed % 10 == 0 or processed == 1:
                    pct = src_idx / src_total * 100 if src_total > 0 else 0
                    print(
                        f"  src={src_idx:5d}/~{src_total}  ({pct:5.1f}%)  "
                        f"med_flow={float(np.median(mag)):.2f}px",
                        end="\r", flush=True,
                    )

                if max_frames is not None and processed >= max_frames:
                    print(f"\n  Reached max_frames={max_frames} — stopping.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            cap.release()
            if writer is not None:
                final_video = writer.close()
                out_video = str(final_video)

        # ── Write JSON ────────────────────────────────────────────────────────
        if out_json is not None and json_records:
            with open(out_json, "w") as f:
                json.dump({
                    "source": str(src),
                    "pairs_written": processed,
                    "backend": self.estimator.backend,
                    "frames": json_records,
                }, f, indent=2)
            print(f"\n  Flow JSON saved → {out_json}")

        elapsed = time.time() - t_start
        print(f"\n{'='*66}")
        print(f"  Done.")
        print(f"  Pairs processed  : {processed}")
        print(f"  Wall time        : {elapsed:.1f}s  "
              f"({processed / max(elapsed, 1e-6):.1f} fps avg)")
        if out_video:  print(f"  Output video     : {out_video}")
        if out_json:   print(f"  JSON output      : {out_json}")
        print(f"{'='*66}")

        return flow_store

    # ── Frame-directory variant ───────────────────────────────────────────────

    def run_on_frames(
        self,
        frame_dir: str,
        out_video: Optional[str] = "renders/flow_output.mp4",
        out_json: Optional[str] = "renders/flow_results.json",
        fps: float = 15.0,
        max_frames: Optional[int] = None,
        layout: str = "side_by_side",
        frame_hook: Optional[Callable] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a directory of JPEG frames instead of a video file.
        Mirrors VideoDepthProcessor.run_on_frames in interface.
        """
        frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
        if not frame_paths:
            frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG frames found in {frame_dir}")
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames + 1]  # +1 for the seed frame

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        H, W  = first.shape[:2]
        out_W = W * 2 if layout == "side_by_side" else W

        writer = None
        if out_video is not None:
            p = Path(out_video)
            p.parent.mkdir(parents=True, exist_ok=True)
            writer = self._open_writer(p, fps, out_W, H)

        if out_json is not None:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[VideoFlowProcessor] Processing {len(frame_paths)-1} pairs "
              f"from {frame_dir}")

        flow_store: Dict[int, np.ndarray] = {}
        json_records: List[Dict] = []
        prev_frame = first

        for idx in range(1, len(frame_paths)):
            curr_frame = cv2.imread(str(frame_paths[idx]))
            if curr_frame is None:
                continue

            flow = self.estimator.estimate(prev_frame, curr_frame)
            flow_store[idx - 1] = flow

            detections: list = []
            if frame_hook is not None:
                frame_hook(prev_frame, curr_frame, flow, detections)

            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            json_records.append({
                "pair_idx": idx - 1,
                "frame_idx": idx,
                "timestamp_s": round((idx - 1) / fps, 4),
                "median_flow_px": round(float(np.median(mag)), 3),
                "max_flow_px": round(float(mag.max()), 3),
                "object_flows": [
                    _build_object_flow_record(det, flow)
                    for det in detections
                ] if detections else [],
            })

            if writer is not None:
                canvas = self._make_canvas(curr_frame, flow, layout)
                self._draw_hud(canvas, idx - 1, 0.0, flow)
                writer.write(canvas)

            prev_frame = curr_frame
            if (idx) % 10 == 0:
                print(f"  [{idx:4d}/{len(frame_paths)-1}]", end="\r", flush=True)

        if writer is not None:
            writer.release()

        if out_json is not None and json_records:
            with open(out_json, "w") as f:
                json.dump({"source": str(frame_dir), "frames": json_records},
                          f, indent=2)

        print(f"\n[VideoFlowProcessor] Finished  {len(flow_store)} pairs  "
              f"→  {out_video or '(no video)'}")
        return flow_store


# ─────────────────────────────────────────────────────────────────────────────
# Module-level public helpers
# ─────────────────────────────────────────────────────────────────────────────

def flow_to_rgb(
    flow: np.ndarray,
    max_magnitude: float = _FLOW_VIS_MAX_MAG,
    alpha: float = 0.0,
    overlay_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert a (H, W, 2) flow field to an HSV-wheel BGR image.

    Hue encodes direction (0° = rightward, 90° = downward, …)
    Value encodes magnitude (bright = fast, dark = still)
    Saturation is fixed at 255.

    Parameters
    ----------
    flow : (H, W, 2) float32 pixels/frame
    max_magnitude : flow magnitude that maps to full brightness
    alpha : if > 0, blend the flow vis over overlay_frame
    overlay_frame : BGR frame to blend onto (required when alpha > 0)

    Returns
    -------
    (H, W, 3) uint8 BGR
    """
    dx, dy = flow[..., 0], flow[..., 1]
    mag, angle = cv2.cartToPolar(dx, dy, angleInDegrees=False)

    # Direction → hue  (cv2 hue range 0–179)
    hue = (angle * (179.0 / (2.0 * np.pi))).astype(np.uint8)

    # Magnitude → value
    val = np.clip(mag / max(max_magnitude, 1e-6), 0.0, 1.0)
    value = (val * 255).astype(np.uint8)

    hsv = np.stack([hue, np.full_like(hue, 255), value], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if alpha > 0 and overlay_frame is not None:
        if overlay_frame.shape[:2] != bgr.shape[:2]:
            overlay_frame = cv2.resize(overlay_frame, (bgr.shape[1], bgr.shape[0]))
        bgr = cv2.addWeighted(overlay_frame, 1.0 - alpha, bgr, alpha, 0)

    return bgr


def get_bbox_flow(
    bbox: Union[List[int], Tuple[int, ...]],
    flow: np.ndarray,
    inner_fraction: float = 0.6,
) -> Tuple[float, float]:
    """
    Compute the mean optical flow vector inside a bounding box.

    Parameters
    ----------
    bbox : [x1, y1, x2, y2]
    flow : (H, W, 2) float32
    inner_fraction : use the central fraction of the box to reduce background contamination at edges

    Returns
    -------
    (mean_dx, mean_dy) in pixels/frame.  (0, 0) if the box is degenerate.
    """
    H, W = flow.shape[:2]
    x1 = max(0, min(int(bbox[0]), W - 1))
    y1 = max(0, min(int(bbox[1]), H - 1))
    x2 = max(0, min(int(bbox[2]), W - 1))
    y2 = max(0, min(int(bbox[3]), H - 1))

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0

    mx = int((x2 - x1) * (1.0 - inner_fraction) / 2)
    my = int((y2 - y1) * (1.0 - inner_fraction) / 2)
    ix1, iy1 = x1 + mx, y1 + my
    ix2, iy2 = x2 - mx, y2 - my
    if ix2 <= ix1: ix1, ix2 = x1, x2
    if iy2 <= iy1: iy1, iy2 = y1, y2

    patch = flow[iy1:iy2, ix1:ix2]
    return float(patch[..., 0].mean()), float(patch[..., 1].mean())


def estimate_velocity(
    mean_dx: float,
    mean_dy: float,
    depth_m: float,
    fps: float,
    fx: float,
) -> float:
    """
    Convert a pixel displacement to an approximate metric speed (m/s).

    Formula (pinhole camera, lateral approximation):
        velocity_mps = sqrt(dx² + dy²) / fx * Z * fps

    Parameters
    ----------
    mean_dx, mean_dy : mean pixel displacement over one frame interval
    depth_m : metric depth of the object (from depth_estimation.py)
    fps : source video frame-rate
    fx : horizontal focal length in pixels (from CalibData.fx)

    Returns
    -------
    Estimated speed in m/s.  Accuracy depends on how well the depth
    estimate reflects the true distance.
    """
    if depth_m <= 0 or fx <= 0 or fps <= 0:
        return 0.0
    pixel_disp = np.sqrt(mean_dx ** 2 + mean_dy ** 2)
    return float(pixel_disp / fx * depth_m * fps)


def align_flow_to_detections(
    detections: list,
    flow: np.ndarray,
    fps: float = 30.0,
    calib=None,                    # CalibData — provides fx for velocity
    inner_fraction: float = 0.6,
) -> List[FlowResult]:
    """
    Enrich each detection in-place with optical flow information.

    For each detection:
      - Computes mean flow vector inside the bounding box.
      - Flags as moving when magnitude exceeds _MOTION_THRESHOLD_PX.
      - Estimates metric velocity when depth_m and calib are available.
      - Writes results back to det.motion_vec / det["motion_vec"] etc.

    Supports both DetectionResult dataclasses and plain dicts.

    Parameters
    ----------
    detections : list of DetectionResult or dicts with "bbox" key
    flow : (H, W, 2) float32 dense flow map
    fps : source video frame-rate (needed for velocity)
    calib : CalibData — provides fx for velocity estimation
    inner_fraction : passed to get_bbox_flow

    Returns
    -------
    List[FlowResult] — one entry per detection (same order).
    """
    results: List[FlowResult] = []

    for det in detections:
        det_id = (
            getattr(det, "id", None)
            or (det.get("id") if isinstance(det, dict) else 0)
            or 0
        )
        bbox = (
            getattr(det, "bbox", None)
            or (det.get("bbox") if isinstance(det, dict) else None)
        )
        if bbox is None:
            continue

        dx, dy = get_bbox_flow(bbox, flow, inner_fraction)
        mag = float(np.sqrt(dx ** 2 + dy ** 2))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        moving = mag > _MOTION_THRESHOLD_PX

        # Velocity estimation requires depth + focal length
        vel: Optional[float] = None
        depth_m = (
            getattr(det, "depth_m", None)
            or (det.get("depth_m") if isinstance(det, dict) else None)
        )
        if depth_m is not None and calib is not None:
            vel = estimate_velocity(dx, dy, depth_m, fps, calib.fx)

        det_direction = (
            getattr(det, "direction", None)
            if not isinstance(det, dict)
            else det.get("direction")
        )
        det_speed_px = (
            getattr(det, "speed_px", None)
            if not isinstance(det, dict)
            else det.get("speed_px")
        )
        det_motion_conf = (
            getattr(det, "motion_confidence", None)
            if not isinstance(det, dict)
            else det.get("motion_confidence")
        )
        det_moving = (
            getattr(det, "moving", None)
            if not isinstance(det, dict)
            else det.get("moving")
        )
        det_parked = (
            getattr(det, "parked", None)
            if not isinstance(det, dict)
            else det.get("parked")
        )
        det_flow_track_id = (
            getattr(det, "flow_track_id", None)
            if not isinstance(det, dict)
            else det.get("flow_track_id")
        )
        residual_vec = (
            getattr(det, "residual_vec", None)
            if not isinstance(det, dict)
            else det.get("residual_vec")
        )
        residual_dx = (
            float(residual_vec[0])
            if isinstance(residual_vec, list) and len(residual_vec) >= 2 and residual_vec[0] is not None
            else None
        )
        residual_dy = (
            float(residual_vec[1])
            if isinstance(residual_vec, list) and len(residual_vec) >= 2 and residual_vec[1] is not None
            else None
        )

        fr = FlowResult(
            det_id=det_id,
            bbox=list(bbox),
            mean_dx=round(dx, 3),
            mean_dy=round(dy, 3),
            magnitude=round(mag, 3),
            angle_deg=round(angle, 2),
            is_moving=moving,
            velocity_mps=round(vel, 3) if vel is not None else None,
            class_name=_canonical_det_class(det),
            moving=bool(det_moving) if det_moving is not None else moving,
            parked=bool(det_parked) if det_parked is not None else (not moving),
            direction=det_direction,
            speed_px=det_speed_px,
            motion_confidence=det_motion_conf,
            residual_dx=residual_dx,
            residual_dy=residual_dy,
            track_id=_det_track_id(det),
            flow_track_id=det_flow_track_id,
        )
        results.append(fr)

        # Write back to detection object
        motion_vec = [round(dx, 3), round(dy, 3)]
        if hasattr(det, "motion_vec"):
            det.motion_vec = motion_vec
            det.is_moving = moving
            det.velocity_mps = fr.velocity_mps
        elif isinstance(det, dict):
            det["motion_vec"] = motion_vec
            det["is_moving"] = moving
            det["velocity_mps"] = fr.velocity_mps

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience singleton
# ─────────────────────────────────────────────────────────────────────────────

_estimator_singleton: Optional[OpticalFlowEstimator] = None


def compute_flow(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    device: str = "auto",
    model_type: str = "auto",
) -> np.ndarray:
    """
    Module-level singleton wrapper.
    Initialises OpticalFlowEstimator on first call; reuses it thereafter.

    Returns
    -------
    (H, W, 2) float32 flow in pixels/frame.
    """
    global _estimator_singleton
    if _estimator_singleton is None:
        _estimator_singleton = OpticalFlowEstimator(
            device=device, model_type=model_type)
    return _estimator_singleton.estimate(prev_bgr, curr_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Internal draw helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_flow_arrows(
    canvas: np.ndarray,
    flow: np.ndarray,
    step: int = _ARROW_GRID_STEP,
    max_len: int = _ARROW_MAX_DRAW_LEN,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """
    Overlay a sparse grid of flow arrows on canvas in-place.
    Arrow length is proportional to flow magnitude, capped at max_len.
    Only arrows with magnitude > 0.5 px are drawn.
    """
    H, W = flow.shape[:2]
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            dx = float(flow[y, x, 0])
            dy = float(flow[y, x, 1])
            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 0.5:
                continue
            scale = min(max_len / (mag + 1e-6), max_len)
            ex = int(x + dx * scale)
            ey = int(y + dy * scale)
            ex = max(0, min(ex, W - 1))
            ey = max(0, min(ey, H - 1))
            cv2.arrowedLine(
                canvas, (x, y), (ex, ey),
                color, thickness, tipLength=0.35,
            )


def _draw_motion_highlight(
    canvas: np.ndarray,
    flow:   np.ndarray,
    threshold: float = _MOTION_THRESHOLD_PX,
    alpha:     float = 0.25,
) -> None:
    """
    Tint moving pixels with a green overlay (in-place).
    Moving = flow magnitude above threshold.
    """
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    mask = (mag > threshold).astype(np.uint8)
    if mask.sum() == 0:
        return

    # Dilate slightly to cover object boundaries cleanly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    green = np.zeros_like(canvas)
    green[mask > 0] = (0, 200, 80)   # BGR green
    cv2.addWeighted(canvas, 1.0, green, alpha, 0, canvas)


def _draw_flow_results(
    canvas: np.ndarray,
    flow_results: List[FlowResult],
) -> None:
    """
    Draw per-object bounding boxes, motion arrows, and velocity labels
    on canvas in-place.

    Moving vehicles: blue box + green direction arrow.
    Parked vehicles: white box.
    Unknown/static objects: grey box.
    """
    for fr in flow_results:
        if str(fr.class_name or "").strip().lower() not in VEHICLE_MOTION_CLASSES:
            continue
        x1, y1, x2, y2 = fr.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        moving = fr.moving if fr.moving is not None else fr.is_moving
        parked = fr.parked if fr.parked is not None else (not moving)
        direction = str(fr.direction or ("stationary" if parked else "unknown"))
        if moving:
            color = (255, 120, 40)
            arrow_color = (0, 230, 60)
        elif parked:
            color = (245, 245, 245)
            arrow_color = color
        else:
            color = (130, 130, 130)
            arrow_color = color

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        if moving:
            # Motion arrow from box centre — scale to be readable
            arrow_scale = min(40.0 / (fr.magnitude + 1e-6), 6.0)
            ex = int(cx + fr.mean_dx * arrow_scale)
            ey = int(cy + fr.mean_dy * arrow_scale)
            H, W = canvas.shape[:2]
            ex = max(0, min(ex, W - 1))
            ey = max(0, min(ey, H - 1))
            cv2.arrowedLine(canvas, (cx, cy), (ex, ey),
                            arrow_color, 2, tipLength=0.3)

        status = "MOV" if moving else ("PRK" if parked else "UNK")
        parts = [f"#{fr.det_id}"]
        if fr.class_name:
            parts.append(str(fr.class_name))
        parts.append(status)
        if moving:
            parts.append(direction)
        if fr.speed_px is not None:
            parts.append(f"{float(fr.speed_px):.1f}px")
        else:
            parts.append(f"{fr.magnitude:.1f}px")
        if fr.motion_confidence is not None:
            parts.append(f"c={float(fr.motion_confidence):.2f}")
        if fr.velocity_mps is not None:
            parts.append(f"{fr.velocity_mps:.1f}m/s")
        label = "  ".join(parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
        lx = x1
        ly = max(y1 - 4, th + bl + 2)
        cv2.rectangle(canvas, (lx, ly - th - bl - 2),
                      (lx + tw + 4, ly + 2), color, cv2.FILLED)
        cv2.putText(canvas, label, (lx + 2, ly),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def _detections_to_flow_results(
    detections: list,
    flow: np.ndarray,
) -> List[FlowResult]:
    """
    Build FlowResult objects from detections that have already been enriched
    by align_flow_to_detections.  Falls back to computing flow on the fly.
    """
    results = []
    for det in detections:
        det_id = (getattr(det, "id", None) or (det.get("id") if isinstance(det, dict) else 0) or 0)
        bbox = (getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None))
        if bbox is None:
            continue
        cls_name = _canonical_det_class(det)
        if cls_name not in VEHICLE_MOTION_CLASSES:
            continue

        mv = (getattr(det, "motion_vec", None)
              or (det.get("motion_vec") if isinstance(det, dict) else None))
        if mv is not None:
            dx, dy = mv[0], mv[1]
        else:
            dx, dy = get_bbox_flow(bbox, flow)

        mag = float(np.sqrt(dx**2 + dy**2))
        angle = float(np.degrees(np.arctan2(dy, dx)))
        vel = (getattr(det, "velocity_mps", None) or (det.get("velocity_mps") if isinstance(det, dict) else None))
        residual_vec = (getattr(det, "residual_vec", None) if not isinstance(det, dict) else det.get("residual_vec"))

        results.append(FlowResult(
            det_id=det_id, bbox=list(bbox),
            mean_dx=dx, mean_dy=dy,
            magnitude=mag, angle_deg=angle,
            is_moving=mag > _MOTION_THRESHOLD_PX,
            velocity_mps=vel,
            class_name=cls_name,
            moving=(getattr(det, "moving", None) if not isinstance(det, dict) else det.get("moving")),
            parked=(getattr(det, "parked", None) if not isinstance(det, dict) else det.get("parked")),
            direction=(getattr(det, "direction", None) if not isinstance(det, dict) else det.get("direction")),
            speed_px=(getattr(det, "speed_px", None) if not isinstance(det, dict) else det.get("speed_px")),
            motion_confidence=(getattr(det, "motion_confidence", None) if not isinstance(det, dict) else det.get("motion_confidence")),
            residual_dx=(float(residual_vec[0]) if isinstance(residual_vec, list) and len(residual_vec) >= 2 else None),
            residual_dy=(float(residual_vec[1]) if isinstance(residual_vec, list) and len(residual_vec) >= 2 else None),
            track_id=_det_track_id(det),
            flow_track_id=(getattr(det, "flow_track_id", None) if not isinstance(det, dict) else det.get("flow_track_id")),
        ))
    return results


def _build_object_flow_record(det: Any, flow: np.ndarray) -> Dict[str, Any]:
    det_id = (getattr(det, "id", None) or (det.get("id") if isinstance(det, dict) else 0) or 0)
    bbox = (getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None) or [0, 0, 0, 0])
    cls_name = _canonical_det_class(det)
    if cls_name not in VEHICLE_MOTION_CLASSES:
        return {
            "det_id": det_id,
            "bbox": list(bbox),
            "class": cls_name,
            "track_id": _det_track_id(det),
            "flow_track_id": None,
            "motion_vec": [0.0, 0.0],
            "mean_dx": 0.0,
            "mean_dy": 0.0,
            "magnitude_px": 0.0,
            "angle_deg": 0.0,
            "velocity_mps": None,
            "moving": False,
            "parked": None,
            "direction": None,
            "speed_px": None,
            "motion_confidence": None,
            "residual_vec": None,
        }
    motion_vec = (getattr(det, "motion_vec", None) or (det.get("motion_vec") if isinstance(det, dict) else None))
    if motion_vec is not None and len(motion_vec) >= 2:
        dx, dy = float(motion_vec[0]), float(motion_vec[1])
    else:
        dx, dy = get_bbox_flow(bbox, flow)
    magnitude_px = float(np.sqrt(dx ** 2 + dy ** 2))
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    residual_vec = (getattr(det, "residual_vec", None) or (det.get("residual_vec") if isinstance(det, dict) else None))
    return {
        "det_id": det_id,
        "bbox": list(bbox),
        "class": cls_name,
        "track_id": _det_track_id(det),
        "flow_track_id": (getattr(det, "flow_track_id", None) if not isinstance(det, dict) else det.get("flow_track_id")),
        "motion_vec": [round(dx, 3), round(dy, 3)],
        "mean_dx": round(dx, 3),
        "mean_dy": round(dy, 3),
        "magnitude_px": round(magnitude_px, 3),
        "angle_deg": round(angle_deg, 2),
        "velocity_mps": (getattr(det, "velocity_mps", None) if not isinstance(det, dict) else det.get("velocity_mps")),
        "moving": (getattr(det, "moving", None) if not isinstance(det, dict) else det.get("moving")),
        "parked": (getattr(det, "parked", None) if not isinstance(det, dict) else det.get("parked")),
        "direction": (getattr(det, "direction", None) if not isinstance(det, dict) else det.get("direction")),
        "speed_px": (getattr(det, "speed_px", None) if not isinstance(det, dict) else det.get("speed_px")),
        "motion_confidence": (getattr(det, "motion_confidence", None) if not isinstance(det, dict) else det.get("motion_confidence")),
        "residual_vec": residual_vec,
    }


def discover_detection_json(scene: Optional[str], explicit_path: Optional[str] = None) -> Optional[Path]:
    """Return the best available aggregated detection JSON for optical-flow alignment."""
    if explicit_path:
        path = Path(explicit_path).resolve()
        return path if path.exists() else None

    layout = scene_output_layout(scene, create=False)
    candidates = [
        layout.detections / "detections.json",
        layout.legacy_repo_renders / "detections.json",
        layout.legacy_detections / "detections.json",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def load_detection_frame_index(path: Optional[Path]) -> Dict[int, List[Dict[str, Any]]]:
    """Load aggregated detections and index them by ``frame_idx``."""
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"[flow] WARNING: failed to parse detections JSON {path}: {exc}")
        return {}

    index: Dict[int, List[Dict[str, Any]]] = {}
    for frame in data.get("frames", []):
        try:
            frame_idx = int(frame.get("frame_idx"))
        except (TypeError, ValueError):
            continue
        detections = frame.get("detections", [])
        if isinstance(detections, list):
            index[frame_idx] = [dict(det) for det in detections if isinstance(det, dict)]
    return index



def compute_scene_flow(video_path, out_json):
    cap = cv2.VideoCapture(str(video_path))
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    flow_registry = {}
    f_idx = 1 # Start from second frame

    print(f"[flow] Computing optical flow for motion priors...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Farneback algorithm for dense flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate global dx, dy to help the assembler determine ego-motion/heading
        avg_dx = float(np.mean(flow[..., 0]))
        avg_dy = float(np.mean(flow[..., 1]))
        
        flow_registry[f_idx] = {"dx": avg_dx, "dy": avg_dy}
        
        prev_gray = curr_gray
        f_idx += 1

    with open(out_json, "w") as f:
        json.dump(flow_registry, f, indent=2)
    print(f"[flow] Saved motion data to {out_json}")

# ─────────────────────────────────────────────────────────────────────────────
# __main__  — standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Optical flow estimation — video or frame pair",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video",  type=str,
                           help="Input video file (.mp4 etc.)")
    src_group.add_argument("--frames", type=str,
                           help="Directory of JPEG frames")
    src_group.add_argument("--pair",   type=str, nargs=2,
                           metavar=("PREV", "CURR"),
                           help="Two image paths for a single frame pair")

    parser.add_argument("--scene",      default=None,
                        help="Optional explicit scene id (e.g. scene1); otherwise inferred from the input path")
    parser.add_argument("--detections-json", default=None,
                        help="Optional aggregated detections JSON used to export per-object flow records; auto-discovered from the scene when omitted")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--model-type", default="auto",
                        choices=["auto", "raft", "farneback"])
    parser.add_argument("--out",        default=None,
                        help="Output path (video: .mp4; pair: .jpg)")
    parser.add_argument("--out-json",   default=None,
                        help="(video) Save per-frame flow stats as JSON")
    parser.add_argument("--layout",     default="side_by_side",
                        choices=["side_by_side", "overlay", "flow_only"])
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps",        type=float, default=15.0,
                        help="Output FPS for frames-directory mode")
    parser.add_argument("--parked-speed-thresh", type=float, default=_RESIDUAL_MOVING_THRESH,
                        help="Residual motion threshold in px/step above which a vehicle is considered moving.")
    parser.add_argument("--parked-history", type=int, default=4,
                        help="Number of low-motion observations needed before a vehicle is marked parked.")

    args = parser.parse_args()

    print("\n" + "═" * 64)
    print("  RBE549 / CS549 P3 — Einstein Vision - Optical Flow Module")
    print("═" * 64 + "\n")

    estimator = OpticalFlowEstimator(
        device=args.device, model_type=args.model_type)
    print(f"  Backend : {estimator.backend}")
    print(f"  Device  : {estimator.device}\n")
    scene_name = infer_scene_name(args.scene, args.video, args.frames, args.out, args.out_json)
    output_layout = scene_output_layout(scene_name, create=True)
    detections_json_path = discover_detection_json(scene_name, args.detections_json)
    detection_index = load_detection_frame_index(detections_json_path)
    if detections_json_path is not None:
        print(f"  Detections : {detections_json_path}")
    else:
        print("  Detections : none (object_flows will be empty)")

    # ── Single frame pair ─────────────────────────────────────────────────────
    if args.pair:
        prev = cv2.imread(args.pair[0])
        curr = cv2.imread(args.pair[1])
        if prev is None or curr is None:
            print("  ERROR: cannot read one or both images.")
            sys.exit(1)

        flow = estimator.estimate(prev, curr)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
        print(f"  Flow range: {mag.min():.2f} – {mag.max():.2f} px/frame")
        print(f"  Flow median: {float(np.median(mag)):.2f} px/frame")

        vis = estimator.visualise(flow, curr, draw_arrows=True, highlight_motion=True)
        out = args.out or "/tmp/flow_pair.jpg"
        cv2.imwrite(out, vis)
        print(f"  Saved → {out}")

    # ── Video file ────────────────────────────────────────────────────────────
    elif args.video:
        out_video = args.out or str((output_layout.flow / "flow_output.mp4").resolve())
        out_json = args.out_json or str((output_layout.flow / "flow_results.json").resolve())
        pair_state = {"pair_idx": 0}
        motion_smoother = VehicleMotionSmoother(
            parked_speed_thresh=args.parked_speed_thresh,
            parked_history=args.parked_history,
        )

        def frame_hook(prev_frame, curr_frame, flow, detections):
            current_frame_idx = (pair_state["pair_idx"] + 1) * max(args.frame_skip, 1)
            pair_state["pair_idx"] += 1
            frame_detections = detection_index.get(current_frame_idx, [])
            if not frame_detections:
                return
            working = [dict(det) for det in frame_detections]
            align_flow_to_detections(working, flow)
            motion_smoother.annotate(working, flow)
            detections.extend(working)

        estimator.process_video(
            args.video,
            out_video=out_video,
            out_json=out_json,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            layout=args.layout,
            frame_hook=frame_hook if detection_index else None,
        )

    # ── Frame directory ───────────────────────────────────────────────────────
    elif args.frames:
        out_video = args.out or str((output_layout.flow / "flow_output.mp4").resolve())
        out_json = args.out_json or str((output_layout.flow / "flow_results.json").resolve())
        pair_state = {"pair_idx": 0}
        motion_smoother = VehicleMotionSmoother(
            parked_speed_thresh=args.parked_speed_thresh,
            parked_history=args.parked_history,
        )

        def frame_hook(prev_frame, curr_frame, flow, detections):
            current_frame_idx = pair_state["pair_idx"] + 1
            pair_state["pair_idx"] += 1
            frame_detections = detection_index.get(current_frame_idx, [])
            if not frame_detections:
                return
            working = [dict(det) for det in frame_detections]
            align_flow_to_detections(working, flow)
            motion_smoother.annotate(working, flow)
            detections.extend(working)

        VideoFlowProcessor(estimator).run_on_frames(
            args.frames,
            out_video=out_video,
            out_json=out_json,
            fps=args.fps,
            max_frames=args.max_frames,
            layout=args.layout,
            frame_hook=frame_hook if detection_index else None,
        )

    if args.video or args.frames:
        if not args.out:
            mirror_stage_output(out_video, scene_name, "flow", Path(out_video).name)
        if not args.out_json:
            mirror_stage_output(out_json, scene_name, "flow", Path(out_json).name)
