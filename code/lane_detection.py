"""
lane_detection.py
=================

Combined lane-line and drivable-area segmentation stage for the project.

Responsibilities
----------------
* infer lane geometry and drivable-road support from each frame
* merge the lane and road predictions into one conservative scene-context output used later by ``scene_assembler.py``
* export annotated videos and a JSON representation suitable for 3-D lifting

Implementation notes
--------------------
The active implementation uses the DebuggerCafe-style torchvision Mask R-CNN
lane-instance model as the primary lane backend, with the earlier SegNet
lane model and U-Net drivable-area model preserved as fallbacks.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import cv2
import json
import time
import math
import shutil
import argparse
import dataclasses
import subprocess
import importlib.util
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    NP_RANK_WARNING = np.RankWarning
except AttributeError:
    try:
        from numpy.exceptions import RankWarning as NP_RANK_WARNING
    except Exception:
        try:
            from numpy.polynomial.polyutils import RankWarning as NP_RANK_WARNING
        except Exception:
            NP_RANK_WARNING = RuntimeWarning

from omonsun_maskrcnn import OmonsunMaskRCNNRefiner, resolve_omonsun_weights
from project_setup import infer_scene_name, mirror_stage_output, scene_output_layout


# =============================================================================
# 1.  External repo path setup — use the cloned repos under ./external
# =============================================================================

EXTERNAL_DIR = (Path(__file__).parent / "external").resolve()

P2_REPO_DIR = (EXTERNAL_DIR / "Object_Detection_Classification_-_Ford_Otosan_Intern_P2").resolve()
P2_LANE_SEG_DIR = (P2_REPO_DIR / "src" / "Lane_Segmentation").resolve()
P2_LANE_CKPT = (P2_REPO_DIR / "models" / "best_line_model.pt").resolve()

P1_REPO_DIR = (EXTERNAL_DIR / "Freespace_Segmentation-Ford_Otosan_Intern").resolve()
P1_ROAD_SEG_DIR = (P1_REPO_DIR / "src").resolve()
P1_ROAD_CKPT = (P1_REPO_DIR / "models" / "Unet_2.pt").resolve()


def _load_module_from_file(module_name: str, file_path: Path):
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

segnet_mod = None
RepoSegNet = None
if P2_LANE_SEG_DIR.exists():
    if str(P2_LANE_SEG_DIR) not in sys.path:
        sys.path.insert(0, str(P2_LANE_SEG_DIR))
    segnet_mod = _load_module_from_file("_external_p2_segnet", P2_LANE_SEG_DIR / "SegNet.py")
    if segnet_mod is not None:
        RepoSegNet = getattr(segnet_mod, "SegNet", None)

# The lane checkpoint was saved from module name `line_SegNet`; alias the whole
# external SegNet module so older pickled checkpoints can recover the private
# `_Encoder` / `_Decoder` classes they were serialized with.
if segnet_mod is not None and "line_SegNet" not in sys.modules:
    alias_mod = type(sys)("line_SegNet")
    for name in ("SegNet", "_Encoder", "_Decoder"):
        if hasattr(segnet_mod, name):
            setattr(alias_mod, name, getattr(segnet_mod, name))
    sys.modules["line_SegNet"] = alias_mod

RepoRoadUNet = None
if P1_ROAD_SEG_DIR.exists():
    if str(P1_ROAD_SEG_DIR) not in sys.path:
        sys.path.insert(0, str(P1_ROAD_SEG_DIR))
    road_unet_mod = _load_module_from_file("_external_p1_unet", P1_ROAD_SEG_DIR / "Unet_1.py")
    if road_unet_mod is not None:
        RepoRoadUNet = getattr(road_unet_mod, "FoInternNet", None)

_repo_unet_available = RepoRoadUNet is not None


# =============================================================================
# 2.  Inline U-Net for road segmentation (P1-compatible)
#     Architecture exactly mirrors recepayddogdu/Freespace_Segmentation UNet_1.py
# =============================================================================

class _DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm2d → ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RoadUNet(nn.Module):
    """
    U-Net for binary freespace segmentation, compatible with the
    Freespace_Segmentation-Ford_Otosan_Intern P1 checkpoint.

    Input  : (B, 3, H, W)  float32
    Output : (B, n_classes, H, W)  raw logits  (softmax applied externally)

    Default n_classes=2  (0=background, 1=freespace/road)
    """

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder
        self.conv1 = _DoubleConv(3, 64)
        self.conv2 = _DoubleConv(64, 128)
        self.conv3 = _DoubleConv(128, 256)
        self.conv4 = _DoubleConv(256, 512)

        # Bottleneck
        self.conv5 = _DoubleConv(512, 1024)

        # Decoder
        self.conv6 = _DoubleConv(1024 + 512, 512)
        self.conv7 = _DoubleConv(512 + 256, 256)
        self.conv8 = _DoubleConv(256 + 128, 128)
        self.conv9 = _DoubleConv(128 + 64,  64)

        # Output
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.conv1(x)
        c2 = self.conv2(self.maxpool(c1))
        c3 = self.conv3(self.maxpool(c2))
        c4 = self.conv4(self.maxpool(c3))

        # Bottleneck
        c5 = self.conv5(self.maxpool(c4))

        # Decoder (skip connections)
        x = self.conv6(torch.cat([self.upsample(c5), c4], dim=1))
        x = self.conv7(torch.cat([self.upsample(x), c3], dim=1))
        x = self.conv8(torch.cat([self.upsample(x), c2], dim=1))
        x = self.conv9(torch.cat([self.upsample(x), c1], dim=1))

        logits = self.out_conv(x)
        return nn.Softmax(dim=1)(logits)


# =============================================================================
# 3.  Device helper
# =============================================================================

def resolve_device(requested: str = "auto") -> str:
    if requested == "cpu":
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


# =============================================================================
# 4.  VS Code-compatible video writer
# =============================================================================

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def reencode_for_vscode(src_video: Path, dst_video: Path) -> bool:
    if not ffmpeg_available():
        return False
    cmd = [
        "ffmpeg", "-y",
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
    """Writes to a temp MJPG AVI, then re-encodes to H.264 MP4 via ffmpeg."""

    def __init__(
        self,
        requested_output: Path,
        fps: float,
        width: int,
        height: int,
        vscode_compatible: bool = True,
    ) -> None:
        self.requested_output = requested_output
        self.vscode_compatible = vscode_compatible
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
                return self.requested_output
        fallback = self.requested_output.with_suffix(".avi")
        if fallback.exists():
            fallback.unlink()
        self.temp_video.rename(fallback)
        return fallback


# =============================================================================
# 5.  Data structures
# =============================================================================

@dataclasses.dataclass
class LaneCurve:
    """A single fitted lane line (solid or dashed)."""

    id: int
    lane_type: str               # "solid" | "dashed"
    line_class: str              # detailed class, e.g. solid-line / divider-line / dotted-line
    color: str                   # "white" | "yellow" | "unknown"
    pixel_count: int
    confidence: float
    curve_points_img: List[List[int]]   # (N, 2) [x, y] in original frame coords
    poly_coeffs: Optional[List[float]]  # 2nd-degree poly  x = f(y)
    color_confidence: float = 0.0
    avg_hsv: Optional[List[float]] = None
    avg_ycrcb: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "lane_type": self.lane_type,
            "line_class": self.line_class,
            "color": self.color,
            "pixel_count": self.pixel_count,
            "confidence": round(self.confidence, 4),
            "color_confidence": round(self.color_confidence, 4),
            "curve_points_img": self.curve_points_img,
            "poly_coeffs": (
                [round(v, 6) for v in self.poly_coeffs]
                if self.poly_coeffs is not None else None
            ),
            "avg_hsv": (
                [round(float(v), 3) for v in self.avg_hsv]
                if self.avg_hsv is not None else None
            ),
            "avg_ycrcb": (
                [round(float(v), 3) for v in self.avg_ycrcb]
                if self.avg_ycrcb is not None else None
            ),
        }


@dataclasses.dataclass
class LaneInstanceResult:
    """Detailed Mask R-CNN lane / marking instance used for rendering and JSON."""

    id: int
    class_name: str
    confidence: float
    bbox: List[int]
    pixel_count: int
    lane_type: str
    paint_color: str = "unknown"
    paint_confidence: float = 0.0
    avg_hsv: Optional[List[float]] = None
    avg_ycrcb: Optional[List[float]] = None
    contour_img: List[List[int]] = dataclasses.field(default_factory=list)
    mask_bool: np.ndarray = dataclasses.field(repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), dtype=bool))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "class_name": str(self.class_name),
            "confidence": round(float(self.confidence), 4),
            "bbox": [int(v) for v in self.bbox],
            "pixel_count": int(self.pixel_count),
            "lane_type": str(self.lane_type),
            "paint_color": str(self.paint_color),
            "paint_confidence": round(float(self.paint_confidence), 4),
            "avg_hsv": (
                [round(float(v), 3) for v in self.avg_hsv]
                if self.avg_hsv is not None else None
            ),
            "avg_ycrcb": (
                [round(float(v), 3) for v in self.avg_ycrcb]
                if self.avg_ycrcb is not None else None
            ),
            "contour_img": self.contour_img,
        }


@dataclasses.dataclass
class RoadMarking:
    """Road-surface marking candidate constrained to the drivable area."""

    id: int
    marking_type: str            # "arrow" | "road_marking"
    color: str                   # "white" | "yellow" | "unknown"
    area_px: int
    confidence: float
    bbox: List[int]
    contour_img: List[List[int]]
    direction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "marking_type": self.marking_type,
            "color": self.color,
            "area_px": int(self.area_px),
            "confidence": round(self.confidence, 4),
            "bbox": [int(v) for v in self.bbox],
            "contour_img": self.contour_img,
            "direction": self.direction,
        }


@dataclasses.dataclass
class RoadResult:
    """Drivable freespace output for one frame."""

    mask_bool: np.ndarray                # (H, W) bool  — True = drivable
    area_frac: float                     # fraction of image that is road [0–1]
    contours_img: List[List[List[int]]]  # list of polygon point-lists (for Blender)
    markings: List[RoadMarking] = dataclasses.field(default_factory=list)
    proc_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "area_frac": round(self.area_frac, 4),
            "proc_ms": round(self.proc_ms, 2),
            # contours: each entry is a closed polygon [[x,y], ...]
            "contours_img": self.contours_img,
            "markings": [m.to_dict() for m in self.markings],
        }


_MASKRCNN_LANE_CLASS_NAMES: Dict[int, str] = {
    1: "divider-line",
    2: "dotted-line",
    3: "double-line",
    4: "random-line",
    5: "road-sign-line",
    6: "solid-line",
}
_MASKRCNN_LANE_RENDER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "divider-line": (80, 230, 120),
    "dotted-line": (0, 255, 255),
    "double-line": (110, 255, 170),
    "random-line": (0, 180, 255),
    "road-sign-line": (90, 220, 90),
    "solid-line": (255, 70, 220),
}
_MASKRCNN_AUTO_CANDIDATES = (
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line_ft/model_30.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line_ft/model_25.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line_ft/model_20.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line_ft/model_15.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line/model_15.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line/model_20.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line/model_25.pth",
    "external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training/road_line/model_30.pth",
    "model_15.pth",
    "maskrcnn_lane_model.pth",
    "lane_maskrcnn.pth",
    "road_line_maskrcnn.pth",
    "mask_rcnn_road_lane.h5",
    "road_lane_mask_rcnn.h5",
    "mask_rcnn_lane.h5",
    "lane_mask_rcnn.h5",
)

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    _maskrcnn_available = True
except Exception:
    maskrcnn_resnet50_fpn_v2 = None
    _maskrcnn_available = False


def _resolve_optional_maskrcnn_weights(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    if str(requested).lower() in {"none", "off", "disable", "disabled"}:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser().resolve()
        return str(path) if path.exists() else requested
    for candidate in _MASKRCNN_AUTO_CANDIDATES:
        if Path(candidate).exists():
            return str(Path(candidate).resolve())
    for local_debuggercafe_training_dir in (
        Path("external/Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/training"),
    ):
        if local_debuggercafe_training_dir.exists():
            all_local_pths: List[Path] = []
            for run_dir in sorted(local_debuggercafe_training_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                all_local_pths.extend(sorted(run_dir.glob("model_*.pth")))
            if all_local_pths:
                all_local_pths = sorted(all_local_pths, key=lambda p: (p.parent.name, p.stat().st_mtime, p.name))
                return str(all_local_pths[-1].resolve())
    omonsun_auto = resolve_omonsun_weights("auto")
    if omonsun_auto:
        return omonsun_auto
    return None


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        out[new_key] = value
    return out


class MaskRCNNLaneRefiner:
    """
    Optional Mask R-CNN instance-segmentation refiner for lane instances and
    road-surface signs/markings, following the DebuggerCafe tutorial structure.

    Expected classes:
      1 divider-line
      2 dotted-line
      3 double-line
      4 random-line
      5 road-sign-line
      6 solid-line
    """

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        score_threshold: float = 0.80,
        mask_threshold: float = 0.45,
        max_dets: int = 32,
    ) -> None:
        if not _maskrcnn_available or maskrcnn_resnet50_fpn_v2 is None:
            raise ImportError("torchvision Mask R-CNN is not available in this environment.")

        self.requested_device = resolve_device(device)
        self.device = self.requested_device
        if self.device == "mps":
            warnings.warn(
                "[MaskRCNNLaneRefiner] torchvision Mask R-CNN inference on MPS can promote tensors "
                "to float64, which MPS does not support. Falling back to CPU for this refiner."
            )
            self.device = "cpu"
        self.score_threshold = float(np.clip(score_threshold, 0.05, 0.99))
        self.mask_threshold = float(np.clip(mask_threshold, 0.05, 0.95))
        self.max_dets = int(max(1, max_dets))
        self.weights_path = str(weights_path)

        self.model = maskrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            num_classes=1 + len(_MASKRCNN_LANE_CLASS_NAMES),
        )
        try:
            ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.weights_path, map_location=self.device)

        if isinstance(ckpt, dict):
            state = None
            for key in ("model", "state_dict", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
            if state is None:
                state = ckpt
        else:
            state = ckpt

        state = _strip_module_prefix(state)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(
            f"[MaskRCNNLaneRefiner] Loaded {Path(self.weights_path).name} "
            f"device='{self.device}' threshold={self.score_threshold:.2f}"
        )

    def infer(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        channels = rgb.shape[2] if rgb.ndim == 3 else 1
        image_bytes = bytearray(rgb.tobytes())
        tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).clone()
        tensor = tensor.view(height, width, channels).permute(2, 0, 1).contiguous()
        tensor = tensor.to(device=self.device, dtype=torch.float32).div(255.0)

        with torch.no_grad():
            output = self.model([tensor])[0]

        scores = output.get("scores")
        labels = output.get("labels")
        boxes = output.get("boxes")
        masks = output.get("masks")
        if scores is None or labels is None or boxes is None or masks is None:
            return []

        scores_list = scores.detach().cpu().tolist()
        labels_list = labels.detach().cpu().tolist()
        boxes_list = boxes.detach().cpu().tolist()
        mask_tensor = masks.detach().cpu()

        instances: List[Dict[str, Any]] = []
        for idx in range(min(len(scores_list), self.max_dets)):
            score = float(scores_list[idx])
            if score < self.score_threshold:
                continue
            class_name = _MASKRCNN_LANE_CLASS_NAMES.get(int(labels_list[idx]))
            if class_name is None:
                continue
            mask_bool = np.asarray(
                (mask_tensor[idx, 0] >= self.mask_threshold).to(torch.uint8).tolist(),
                dtype=bool,
            )
            if int(np.count_nonzero(mask_bool)) < 40:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in boxes_list[idx]]
            instances.append(
                {
                    "class_name": class_name,
                    "score": score,
                    "bbox": [x1, y1, x2, y2],
                    "mask": mask_bool,
                }
            )
        return _nms_mask_instances(instances)


# =============================================================================
# 6.  Shared preprocessing
# =============================================================================

def preprocess_frame(
    frame_bgr: np.ndarray,
    input_width: int,
    input_height: int,
    device: str,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Shared preprocessing for both P1 and P2 models:
      1. NORM_MINMAX normalisation  (matches both repos' training pipelines)
      2. Resize to (input_width, input_height) with INTER_NEAREST
      3. float32 CHW tensor on *device*

    Returns
    -------
    tensor  : (1, 3, input_height, input_width)  float32
    orig_wh : (orig_W, orig_H) — for inverse-mapping back to original coords
    """
    orig_H, orig_W = frame_bgr.shape[:2]
    zeros_img = np.zeros((1920, 1208), dtype=np.uint8)
    norm_img = cv2.normalize(frame_bgr, zeros_img, 0, 255, cv2.NORM_MINMAX)
    resized = cv2.resize(norm_img, (input_width, input_height), interpolation=cv2.INTER_NEAREST)
    height, width = resized.shape[:2]
    channels = resized.shape[2] if resized.ndim == 3 else 1
    image_bytes = bytearray(resized.tobytes())
    tensor = torch.frombuffer(image_bytes, dtype=torch.uint8).clone()
    tensor = tensor.view(height, width, channels).permute(2, 0, 1).contiguous()
    tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor, (orig_W, orig_H)


# =============================================================================
# 7.  Lane fitting helpers 
# =============================================================================

def fit_lane_curve_from_mask(
    mask: np.ndarray,
    degree: int = 2,
    min_pixels: int = 40,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pixels:
        return None, np.empty((0, 2), dtype=np.int32)

    y_span = float(np.max(ys) - np.min(ys))
    if y_span < 25:
        return None, np.empty((0, 2), dtype=np.int32)

    order = np.argsort(ys)
    ys = ys[order].astype(np.float32)
    xs = xs[order].astype(np.float32)

    y_mean = np.mean(ys)
    y_std = np.std(ys)
    if y_std < 1e-6:
        return None, np.empty((0, 2), dtype=np.int32)

    y_norm = (ys - y_mean) / y_std
    residual_thresh = float(np.clip(0.04 * max(np.ptp(xs), 20.0) + 4.0, 4.5, 12.0))
    sample_size = int(np.clip(max(degree + 2, len(xs) // 14), degree + 2, 20))
    rng_seed = int((len(xs) * 1315423911 + int(np.mean(ys)) * 2654435761) % (2**32 - 1))
    rng = np.random.default_rng(rng_seed)
    best_coeffs = None
    best_inliers = None
    best_score = -1.0
    max_trials = int(np.clip(len(xs) // 10, 24, 64))

    for _ in range(max_trials):
        try:
            pick = rng.choice(len(xs), size=sample_size, replace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("error", NP_RANK_WARNING)
                trial = np.polyfit(y_norm[pick], xs[pick], deg=degree)
        except (ValueError, np.linalg.LinAlgError, NP_RANK_WARNING):
            continue

        residual = np.abs(np.polyval(trial, y_norm) - xs)
        inliers = residual <= residual_thresh
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < max(degree + 4, int(0.42 * len(xs))):
            continue
        score = float(inlier_count) - 0.18 * float(np.mean(residual[inliers]))
        if score > best_score:
            best_score = score
            best_coeffs = trial
            best_inliers = inliers

    fit_mask = best_inliers
    if fit_mask is None:
        fit_mask = np.ones(len(xs), dtype=bool)
    if int(np.count_nonzero(fit_mask)) < max(degree + 4, int(0.35 * len(xs))):
        return None, np.empty((0, 2), dtype=np.int32)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", NP_RANK_WARNING)
            coeffs_norm = np.polyfit(y_norm[fit_mask], xs[fit_mask], deg=degree)
    except (np.linalg.LinAlgError, NP_RANK_WARNING):
        if best_coeffs is None:
            return None, np.empty((0, 2), dtype=np.int32)
        coeffs_norm = best_coeffs

    y_samples = np.linspace(
        np.percentile(ys[fit_mask], 5), np.percentile(ys[fit_mask], 98), 40
    ).astype(np.float32)
    y_samples_norm = (y_samples - y_mean) / y_std
    x_samples = np.polyval(coeffs_norm, y_samples_norm)
    pts = np.round(np.stack([x_samples, y_samples], axis=1)).astype(np.int32)

    return coeffs_norm.astype(np.float32), pts


def _sample_lane_centerline_from_mask(
    mask: np.ndarray,
    *,
    num_samples: int = 28,
    min_points: int = 6,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) < max(min_points * 4, 24):
        return np.empty((0, 2), dtype=np.int32)

    y_lo = int(np.percentile(ys, 5))
    y_hi = int(np.percentile(ys, 98))
    if (y_hi - y_lo) < 20:
        return np.empty((0, 2), dtype=np.int32)

    band = max(2, int(round((y_hi - y_lo) / max(num_samples, 1))))
    points: List[List[int]] = []
    last_y = -10**9
    for y_ref in np.linspace(y_lo, y_hi, num_samples):
        select = np.abs(ys - y_ref) <= band
        if int(np.count_nonzero(select)) < 3:
            continue
        y_med = int(round(float(np.median(ys[select]))))
        if y_med <= last_y:
            continue
        x_med = int(round(float(np.median(xs[select]))))
        points.append([x_med, y_med])
        last_y = y_med

    if len(points) < min_points:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(points, dtype=np.int32)


def _fit_poly_coeffs_from_curve_points(
    curve_pts: np.ndarray,
    degree: int = 2,
) -> Optional[np.ndarray]:
    if curve_pts is None or len(curve_pts) < max(degree + 2, 6):
        return None

    ys = curve_pts[:, 1].astype(np.float32)
    xs = curve_pts[:, 0].astype(np.float32)
    y_std = float(np.std(ys))
    if y_std < 1e-6:
        return None
    y_norm = (ys - float(np.mean(ys))) / y_std
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", NP_RANK_WARNING)
            coeffs = np.polyfit(y_norm, xs, deg=degree)
    except (ValueError, np.linalg.LinAlgError, NP_RANK_WARNING):
        return None
    return coeffs.astype(np.float32)


def connected_lane_components(
    class_mask: np.ndarray,
    min_area: int = 80,
) -> List[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        class_mask.astype(np.uint8), connectivity=8
    )
    masks = []
    for idx in range(1, n):
        area = stats[idx, cv2.CC_STAT_AREA]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        if area < min_area:
            continue
        if h < 20:
            continue
        if w > 3 * h:
            continue
        masks.append((labels == idx).astype(np.uint8))
    return masks


def _mask_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _compute_lane_component_metrics(
    mask: np.ndarray,
    curve_pts: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    H, W = mask.shape[:2]
    bbox = _mask_bbox(mask)
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    pixel_count = int(np.count_nonzero(mask))
    bbox_area = max(w * h, 1)
    extent = float(pixel_count) / float(bbox_area)
    elongation = max(float(w) / float(h), float(h) / float(w))

    contour = _mask_contour(mask)
    orientation_deg = _fit_component_orientation_deg(contour) if contour is not None else 90.0
    road_support = 1.0
    if road_mask is not None:
        dilated = cv2.dilate(
            mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        ).astype(bool)
        support_pixels = dilated & road_mask.astype(bool)
        denom = max(int(np.count_nonzero(dilated)), 1)
        road_support = float(np.count_nonzero(support_pixels)) / float(denom)

    top_x = float(curve_pts[0][0]) if len(curve_pts) > 0 else float(x1 + x2) * 0.5
    bottom_x = float(curve_pts[-1][0]) if len(curve_pts) > 0 else float(x1 + x2) * 0.5
    curve_y_span = float(np.ptp(curve_pts[:, 1])) if len(curve_pts) >= 2 else float(h)

    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "w": float(w),
        "h": float(h),
        "pixel_count": float(pixel_count),
        "extent": float(extent),
        "elongation": float(elongation),
        "orientation_deg": float(orientation_deg),
        "road_support": float(road_support),
        "top_frac": float(y1) / float(max(H, 1)),
        "bottom_frac": float(y2) / float(max(H, 1)),
        "y_span": float(h),
        "curve_y_span": float(curve_y_span),
        "top_x": float(top_x),
        "bottom_x": float(bottom_x),
        "center_x": float(0.5 * (x1 + x2)),
        "center_y": float(0.5 * (y1 + y2)),
    }


def _lane_component_is_plausible(
    metrics: Dict[str, float],
    lane_type: str,
    frame_shape: Tuple[int, int],
) -> bool:
    H, W = frame_shape[:2]
    y_span = float(metrics["y_span"])
    curve_y_span = float(metrics["curve_y_span"])
    pixel_count = float(metrics["pixel_count"])
    elongation = float(metrics["elongation"])
    extent = float(metrics["extent"])
    orientation_deg = float(metrics["orientation_deg"])
    road_support = float(metrics["road_support"])
    bottom_frac = float(metrics["bottom_frac"])
    top_frac = float(metrics["top_frac"])
    w = float(metrics["w"])
    h = float(metrics["h"])

    if metrics["center_x"] < 0.01 * float(W) or metrics["center_x"] > 0.99 * float(W):
        return False
    if extent <= 0.01 or extent >= 0.92:
        return False

    score = 0.0
    score += np.clip((pixel_count - (55.0 if lane_type == "dashed" else 80.0)) / 3400.0, 0.0, 1.0) * 0.22
    score += np.clip((curve_y_span - 42.0) / (0.28 * float(H)), 0.0, 1.0) * 0.22
    score += np.clip((elongation - 1.25) / 5.0, 0.0, 1.0) * 0.16
    score += np.clip((road_support - 0.20) / 0.65, 0.0, 1.0) * 0.22
    score += np.clip(1.0 - abs(90.0 - orientation_deg) / 65.0, 0.0, 1.0) * 0.10
    score += np.clip((bottom_frac - 0.25) / 0.55, 0.0, 1.0) * 0.08

    if lane_type == "dashed" and y_span < 36.0 and pixel_count < 140.0:
        return False
    if lane_type == "solid" and y_span < 54.0 and pixel_count < 160.0:
        return False
    if bottom_frac < 0.32 and y_span < 0.12 * float(H):
        return False
    if top_frac > 0.86 and y_span < 0.10 * float(H):
        return False
    if w > 3.1 * max(h, 1.0) and y_span < 0.20 * float(H):
        return False

    return bool(score >= (0.34 if lane_type == "dashed" else 0.38))


def _lane_component_is_viable_relaxed(
    metrics: Dict[str, float],
    lane_type: str,
    frame_shape: Tuple[int, int],
) -> bool:
    H, W = frame_shape[:2]
    if metrics["center_x"] < 0.02 * float(W) or metrics["center_x"] > 0.98 * float(W):
        return False
    if metrics["extent"] <= 0.003 or metrics["extent"] >= 0.98:
        return False

    y_span = float(metrics["y_span"])
    curve_y_span = float(metrics["curve_y_span"])
    pixel_count = float(metrics["pixel_count"])
    road_support = float(metrics["road_support"])
    orientation_deg = float(metrics["orientation_deg"])
    bottom_frac = float(metrics["bottom_frac"])

    min_pixels = 70.0 if lane_type == "solid" else 45.0
    min_span = max(24.0, 0.055 * float(H))
    if pixel_count < min_pixels:
        return False
    if y_span < min_span and curve_y_span < min_span:
        return False
    if bottom_frac < 0.18 and curve_y_span < 0.10 * float(H):
        return False
    if road_support < 0.02 and bottom_frac < 0.45:
        return False
    if abs(90.0 - orientation_deg) > 72.0 and curve_y_span < 0.14 * float(H):
        return False
    return True


def _lane_y_overlap_ratio(a: Dict[str, float], b: Dict[str, float]) -> float:
    overlap = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    denom = max(min(a["y_span"], b["y_span"]), 1.0)
    return float(overlap / denom)


def _lane_candidates_redundant(
    a: Dict[str, Any],
    b: Dict[str, Any],
    frame_shape: Tuple[int, int],
) -> bool:
    H, W = frame_shape[:2]
    ma = a["metrics"]
    mb = b["metrics"]
    y_overlap = _lane_y_overlap_ratio(ma, mb)
    if y_overlap < 0.42:
        return False
    if abs(ma["orientation_deg"] - mb["orientation_deg"]) > 18.0:
        return False

    bottom_dx = abs(ma["bottom_x"] - mb["bottom_x"])
    top_dx = abs(ma["top_x"] - mb["top_x"])
    center_dx = abs(ma["center_x"] - mb["center_x"])
    same_track = (
        bottom_dx <= max(85.0, 0.065 * float(W))
        and top_dx <= max(140.0, 0.11 * float(W))
        and center_dx <= max(110.0, 0.085 * float(W))
    )
    if not same_track:
        return False

    if a["lane"].lane_type == b["lane"].lane_type:
        return True
    return y_overlap >= 0.62 and bottom_dx <= max(60.0, 0.05 * float(W))


def _lane_candidate_score(entry: Dict[str, Any]) -> float:
    lane = entry["lane"]
    metrics = entry["metrics"]
    return (
        1.3 * float(lane.pixel_count)
        + 3.0 * float(metrics["curve_y_span"])
        + 180.0 * float(metrics["road_support"])
        + 140.0 * float(lane.color_confidence)
        + 45.0 * float(metrics["elongation"])
    )


def _suppress_redundant_lane_candidates(
    candidates: List[Dict[str, Any]],
    frame_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    ordered = sorted(candidates, key=_lane_candidate_score, reverse=True)
    for cand in ordered:
        if any(_lane_candidates_redundant(cand, other, frame_shape) for other in kept):
            continue
        kept.append(cand)
    kept.sort(key=lambda item: item["metrics"]["bottom_x"])
    return kept


# =============================================================================
# 8.  Road contour extraction (for Blender / scene assembler)
# =============================================================================

def extract_road_contours(
    road_mask: np.ndarray,           # (H, W) bool
    min_area: int = 2000,
    epsilon_frac: float = 0.005,       # Douglas-Peucker simplification
) -> List[List[List[int]]]:
    """
    Find external contours of the drivable area mask, simplified for JSON export.

    Returns a list of polygons; each polygon is a list of [x, y] int pairs.
    The largest contour (main road surface) is always first.
    """
    mask_u8 = road_mask.astype(np.uint8) * 255

    # Light morphological cleanup to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        eps = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)          # simplify
        poly = approx.reshape(-1, 2).tolist()              # [[x,y], ...]
        polys.append((area, poly))

    # Sort largest first (main road surface)
    polys.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in polys]


def classify_paint_color_from_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    fallback: str = "white",
    road_mask: Optional[np.ndarray] = None,
) -> Tuple[str, float]:
    """
    Estimate whether a lane/marking mask is painted white or yellow.

    The decision is made from a paint-core sample plus a local background ring.
    Using local contrast makes the colour estimate much more stable at night,
    where sodium / warm storefront lighting can otherwise push white paint
    toward yellow in raw HSV space.
    """
    mask_u8 = mask.astype(np.uint8)
    mask_bool = mask_u8.astype(bool)
    if not mask_bool.any():
        return "unknown", 0.0

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

    core = cv2.erode(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    if int(np.count_nonzero(core)) < max(20, int(0.30 * np.count_nonzero(mask_bool))):
        core = mask_bool

    lightness = lab[:, :, 0].astype(np.float32)
    core_vals = lightness[core]
    if core_vals.size == 0:
        return "unknown", 0.0
    bright_thresh = float(np.percentile(core_vals, 55))
    paint = core & (lightness >= bright_thresh)
    if int(np.count_nonzero(paint)) < max(16, int(0.25 * np.count_nonzero(core))):
        paint = core

    outer = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ).astype(bool)
    inner = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    ).astype(bool)
    ring = outer & (~inner) & (~mask_bool)
    if road_mask is not None:
        ring &= cv2.dilate(
            road_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        ).astype(bool)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    h_hls = hls[:, :, 0].astype(np.float32)
    l_hls = hls[:, :, 1].astype(np.float32)
    s_hls = hls[:, :, 2].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    paint_h = h[paint]
    paint_s = s[paint]
    paint_v = v[paint]
    paint_h_hls = h_hls[paint]
    paint_l_hls = l_hls[paint]
    paint_s_hls = s_hls[paint]
    paint_l = lightness[paint]
    paint_a = a[paint]
    paint_b = b[paint]
    paint_y = y[paint]
    paint_cr = cr[paint]
    paint_cb = cb[paint]
    if paint_h.size == 0:
        return "unknown", 0.0

    mean_h = float(np.mean(paint_h))
    mean_s = float(np.mean(paint_s))
    mean_v = float(np.mean(paint_v))
    mean_y = float(np.mean(paint_y))
    mean_cr = float(np.mean(paint_cr))
    mean_cb = float(np.mean(paint_cb))

    if int(np.count_nonzero(ring)) >= 24:
        bg_y = float(np.median(y[ring]))
        bg_l = float(np.median(lightness[ring]))
        bg_cr = float(np.median(cr[ring]))
        bg_cb = float(np.median(cb[ring]))
        bg_a = float(np.median(a[ring]))
        bg_b = float(np.median(b[ring]))
    else:
        bg_y = float(np.percentile(paint_y, 25))
        bg_l = float(np.percentile(paint_l, 25))
        bg_cr = float(np.median(paint_cr))
        bg_cb = float(np.median(paint_cb))
        bg_a = float(np.median(paint_a))
        bg_b = float(np.median(paint_b))

    delta_y = float(np.median(paint_y) - bg_y)
    delta_l = float(np.median(paint_l) - bg_l)
    delta_cr = float(np.median(paint_cr) - bg_cr)
    delta_cb = float(np.median(paint_cb) - bg_cb)
    delta_a = float(np.median(paint_a) - bg_a)
    delta_b = float(np.median(paint_b) - bg_b)
    paint_chroma = np.sqrt((paint_a - 128.0) ** 2 + (paint_b - 128.0) ** 2)

    white_hsv_mask = (
        (paint_s <= max(72.0, float(np.percentile(paint_s, 45))))
        & (paint_v >= max(148.0, float(np.percentile(paint_v, 45))))
    )
    white_hls_mask = (
        (paint_l_hls >= max(152.0, float(np.percentile(paint_l_hls, 45))))
        & (paint_s_hls <= max(120.0, float(np.percentile(paint_s_hls, 60))))
    )
    yellow_hsv_mask = (
        (paint_h >= 12.0) & (paint_h <= 42.0)
        & (paint_s >= 55.0)
        & (paint_v >= 90.0)
    )
    yellow_hls_mask = (
        (paint_h_hls >= 10.0) & (paint_h_hls <= 44.0)
        & (paint_l_hls >= 72.0)
        & (paint_s_hls >= 40.0)
    )
    white_lab_mask = (
        (paint_l >= max(154.0, float(np.percentile(paint_l, 45))))
        & (paint_chroma <= max(16.0, float(np.percentile(paint_chroma, 60))))
        & (paint_b <= max(150.0, bg_b + 14.0))
    )
    yellow_lab_mask = (
        (paint_b >= max(142.0, bg_b + 7.0))
        & (paint_a >= max(128.0, bg_a + 1.5))
        & (paint_l >= 96.0)
    )
    white_ycrcb_mask = (
        (paint_y >= max(150.0, float(np.percentile(paint_y, 45))))
        & (np.abs(paint_cr - 128.0) <= 18.0)
        & (np.abs(paint_cb - 128.0) <= 18.0)
    )
    yellow_ycrcb_mask = (
        (paint_y >= 88.0)
        & (paint_cr >= max(130.0, bg_cr + 0.5))
        & ((paint_cr - paint_cb) >= max(6.0, (bg_cr - bg_cb) + 2.0))
    )

    white_ratio = (
        0.28 * float(np.mean(white_hsv_mask))
        + 0.22 * float(np.mean(white_hls_mask))
        + 0.28 * float(np.mean(white_lab_mask))
        + 0.22 * float(np.mean(white_ycrcb_mask))
    )
    yellow_ratio = (
        0.26 * float(np.mean(yellow_hsv_mask))
        + 0.18 * float(np.mean(yellow_hls_mask))
        + 0.32 * float(np.mean(yellow_lab_mask))
        + 0.24 * float(np.mean(yellow_ycrcb_mask))
    )
    warm_hue_ratio = float(np.mean((paint_h >= 10.0) & (paint_h <= 45.0)))
    low_sat_ratio = float(np.mean(paint_s <= 72.0))
    bright_ratio = float(np.mean(paint_v >= max(120.0, float(np.percentile(paint_v, 45)))))
    chroma_ratio = float(np.mean((paint_s >= 45.0) & (paint_v >= 90.0)))
    hsv_white_mean_score = (
        np.clip((180.0 - mean_s) / 110.0, 0.0, 1.0)
        * np.clip((mean_v - 130.0) / 95.0, 0.0, 1.0)
    )
    hsv_yellow_mean_score = (
        np.clip(1.0 - abs(mean_h - 27.0) / 16.0, 0.0, 1.0)
        * np.clip((mean_s - 48.0) / 110.0, 0.0, 1.0)
        * np.clip((mean_v - 88.0) / 110.0, 0.0, 1.0)
    )
    ycrcb_white_mean_score = (
        np.clip((mean_y - 140.0) / 80.0, 0.0, 1.0)
        * np.clip(1.0 - ((abs(mean_cr - 128.0) + abs(mean_cb - 128.0)) / 44.0), 0.0, 1.0)
    )
    ycrcb_yellow_mean_score = (
        np.clip((mean_y - 90.0) / 110.0, 0.0, 1.0)
        * np.clip((mean_cr - 128.0) / 18.0, 0.0, 1.0)
        * np.clip((128.0 - mean_cb) / 18.0, 0.0, 1.0)
    )
    neutral_chroma_ratio = float(np.mean((np.abs(paint_cr - 128.0) <= 18.0) & (np.abs(paint_cb - 128.0) <= 18.0)))
    warm_ycrcb_ratio = float(np.mean((paint_cr >= 130.0) & ((paint_cr - paint_cb) >= 6.0)))

    white_score = 0.0
    white_score += 0.30 * white_ratio
    white_score += 0.18 * low_sat_ratio
    white_score += 0.16 * np.clip((delta_l - 8.0) / 24.0, 0.0, 1.0)
    white_score += 0.08 * bright_ratio
    white_score += 0.06 * np.clip((8.0 - max(delta_b, 0.0)) / 8.0, 0.0, 1.0)
    white_score += 0.08 * hsv_white_mean_score
    white_score += 0.10 * ycrcb_white_mean_score
    white_score += 0.06 * neutral_chroma_ratio
    white_score += 0.04 * np.clip((delta_y - max(abs(delta_cr), abs(delta_cb))) / 18.0, 0.0, 1.0)
    white_score += 0.04 * float(mean_s <= 70.0 and mean_v >= 145.0)

    yellow_score = 0.0
    yellow_score += 0.28 * yellow_ratio
    yellow_score += 0.16 * warm_hue_ratio
    yellow_score += 0.18 * np.clip((delta_b - 6.0) / 20.0, 0.0, 1.0)
    yellow_score += 0.10 * np.clip((delta_a - 2.0) / 16.0, 0.0, 1.0)
    yellow_score += 0.08 * chroma_ratio
    yellow_score += 0.08 * hsv_yellow_mean_score
    yellow_score += 0.12 * ycrcb_yellow_mean_score
    yellow_score += 0.08 * warm_ycrcb_ratio
    yellow_score += 0.06 * np.clip(((delta_cr - delta_cb) - 2.0) / 12.0, 0.0, 1.0)
    yellow_score += 0.04 * float(12.0 <= mean_h <= 42.0 and mean_s >= 55.0 and mean_v >= 90.0)

    if yellow_score >= max(0.42, white_score + 0.08):
        return "yellow", float(yellow_score)
    if white_score >= max(0.40, yellow_score + 0.06):
        return "white", float(white_score)
    if yellow_ratio >= max(0.24, white_ratio + 0.06):
        return "yellow", float(min(0.92, 0.45 + 0.60 * yellow_ratio))
    if white_ratio >= max(0.22, yellow_ratio + 0.04):
        return "white", float(min(0.92, 0.45 + 0.60 * white_ratio))
    if delta_b >= 14.0 and chroma_ratio >= 0.18:
        return "yellow", float(min(0.95, 0.35 + 0.02 * delta_b))
    if delta_l >= 10.0 and low_sat_ratio >= 0.42:
        return "white", float(min(0.95, 0.30 + 0.02 * delta_l))
    return fallback, float(max(white_score, yellow_score))


def _average_color_space_from_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    conversion_code: int,
) -> Optional[List[float]]:
    mask_u8 = mask.astype(np.uint8)
    mask_bool = mask_u8.astype(bool)
    if not mask_bool.any():
        return None
    core = cv2.erode(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ).astype(bool)
    if int(np.count_nonzero(core)) < max(20, int(0.30 * np.count_nonzero(mask_bool))):
        core = mask_bool
    converted = cv2.cvtColor(frame_bgr, conversion_code)
    vals = converted[core]
    if vals.size == 0:
        vals = converted[mask_bool]
    if vals.size == 0:
        return None
    return [float(v) for v in np.mean(vals, axis=0).tolist()]


def average_hsv_from_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
) -> Optional[List[float]]:
    return _average_color_space_from_mask(frame_bgr, mask, cv2.COLOR_BGR2HSV)


def average_ycrcb_from_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
) -> Optional[List[float]]:
    return _average_color_space_from_mask(frame_bgr, mask, cv2.COLOR_BGR2YCrCb)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    if union <= 1e-6:
        return 0.0
    return inter / union


def _mask_containment(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.count_nonzero(a & b))
    denom = float(max(min(np.count_nonzero(a), np.count_nonzero(b)), 1))
    return inter / denom


def _nms_mask_instances(
    instances: List[Dict[str, Any]],
    iou_thresh: float = 0.38,
    contain_thresh: float = 0.76,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    ordered = sorted(instances, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    for inst in ordered:
        cls_name = str(inst.get("class_name"))
        mask = np.asarray(inst.get("mask"), dtype=bool)
        duplicate = False
        for prev in kept:
            if str(prev.get("class_name")) != cls_name:
                continue
            prev_mask = np.asarray(prev.get("mask"), dtype=bool)
            iou = _mask_iou(mask, prev_mask)
            contain = _mask_containment(mask, prev_mask)
            if iou >= iou_thresh or contain >= contain_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def _lane_type_from_maskrcnn_class(class_name: str) -> str:
    return "dashed" if str(class_name) == "dotted-line" else "solid"


def _default_line_class_for_lane_type(lane_type: str) -> str:
    return "dotted-line" if str(lane_type) == "dashed" else "solid-line"


def _fallback_lane_color_from_instance(
    class_name: str,
    mask_bool: np.ndarray,
    road_mask: Optional[np.ndarray],
) -> str:
    if class_name in {"divider-line", "double-line"}:
        return "yellow"

    if road_mask is not None:
        road_center = _road_center_x(road_mask)
        if road_center is not None:
            ys, xs = np.where(mask_bool)
            if xs.size > 0:
                bottom_band = xs[ys >= np.percentile(ys, 65)] if ys.size > 0 else xs
                x_ref = float(np.median(bottom_band)) if bottom_band.size > 0 else float(np.median(xs))
                return "yellow" if x_ref < road_center - 0.04 * float(mask_bool.shape[1]) else "white"
    return "white"


def _curve_mask_from_points(
    curve: LaneCurve,
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    H, W = frame_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.asarray(curve.curve_points_img, dtype=np.int32)
    if len(pts) >= 2:
        thickness = 8 if curve.lane_type == "solid" else 6
        cv2.polylines(
            mask,
            [pts.reshape(-1, 1, 2)],
            False,
            1,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return mask.astype(bool)


def _curve_duplicate(
    curve_a: LaneCurve,
    curve_b: LaneCurve,
    frame_shape: Tuple[int, int],
) -> bool:
    H, W = frame_shape[:2]
    pts_a = np.asarray(curve_a.curve_points_img, dtype=np.float32)
    pts_b = np.asarray(curve_b.curve_points_img, dtype=np.float32)
    if len(pts_a) < 2 or len(pts_b) < 2:
        return False
    top_dx = abs(float(pts_a[0, 0]) - float(pts_b[0, 0]))
    bottom_dx = abs(float(pts_a[-1, 0]) - float(pts_b[-1, 0]))
    y_overlap = max(0.0, min(float(np.max(pts_a[:, 1])), float(np.max(pts_b[:, 1]))) - max(float(np.min(pts_a[:, 1])), float(np.min(pts_b[:, 1]))))
    y_span = max(1.0, min(float(np.ptp(pts_a[:, 1])), float(np.ptp(pts_b[:, 1]))))
    return bool(
        curve_a.lane_type == curve_b.lane_type
        and top_dx <= max(70.0, 0.08 * float(W))
        and bottom_dx <= max(50.0, 0.06 * float(W))
        and y_overlap / y_span >= 0.35
    )


def _maskrcnn_class_color(class_name: str) -> Tuple[int, int, int]:
    return _MASKRCNN_LANE_RENDER_COLORS.get(str(class_name), (160, 220, 255))


def _mask_to_primary_contour(mask_bool: np.ndarray) -> List[List[int]]:
    mask_u8 = np.asarray(mask_bool, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.0025 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


def _lane_instance_from_maskrcnn_instance(
    frame_bgr: np.ndarray,
    mask_bool: np.ndarray,
    class_name: str,
    score: float,
    bbox: List[int],
    road_mask: Optional[np.ndarray],
) -> Optional[LaneInstanceResult]:
    pixel_count = int(np.count_nonzero(mask_bool))
    if pixel_count < 40:
        return None

    if class_name == "road-sign-line":
        lane_type = "marking"
        fallback_color = "white"
    else:
        lane_type = _lane_type_from_maskrcnn_class(class_name)
        fallback_color = _fallback_lane_color_from_instance(class_name, mask_bool, road_mask)

    paint_color, paint_conf = classify_paint_color_from_mask(
        frame_bgr,
        mask_bool.astype(np.uint8),
        fallback=fallback_color,
        road_mask=road_mask,
    )
    avg_hsv = average_hsv_from_mask(frame_bgr, mask_bool.astype(np.uint8))
    avg_ycrcb = average_ycrcb_from_mask(frame_bgr, mask_bool.astype(np.uint8))
    return LaneInstanceResult(
        id=-1,
        class_name=str(class_name),
        confidence=float(score),
        bbox=[int(v) for v in bbox[:4]],
        pixel_count=pixel_count,
        lane_type=lane_type,
        paint_color=str(paint_color),
        paint_confidence=float(paint_conf),
        avg_hsv=avg_hsv,
        avg_ycrcb=avg_ycrcb,
        contour_img=_mask_to_primary_contour(mask_bool),
        mask_bool=np.asarray(mask_bool, dtype=bool),
    )


def _lane_results_from_maskrcnn_instances(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray,
    maskrcnn_instances: List[Dict[str, Any]],
    frame_shape: Tuple[int, int],
) -> Tuple[np.ndarray, List[LaneCurve], List[LaneInstanceResult]]:
    H, W = frame_shape[:2]
    lane_pred = np.zeros((H, W), dtype=np.uint8)
    curves: List[LaneCurve] = []
    lane_instances: List[LaneInstanceResult] = []

    sorted_instances = sorted(
        maskrcnn_instances,
        key=lambda inst: float(inst.get("score", 0.0)),
        reverse=True,
    )
    for inst in sorted_instances:
        class_name = str(inst.get("class_name") or "")
        if class_name not in set(_MASKRCNN_LANE_CLASS_NAMES.values()):
            continue
        inst_mask = np.asarray(inst.get("mask"), dtype=bool)
        if inst_mask.shape[:2] != (H, W) or not inst_mask.any():
            continue
        bbox = [int(round(v)) for v in (inst.get("bbox") or [0, 0, 0, 0])[:4]]
        score = float(inst.get("score", 0.0))

        lane_instance = _lane_instance_from_maskrcnn_instance(
            frame_bgr=frame_bgr,
            mask_bool=inst_mask,
            class_name=class_name,
            score=score,
            bbox=bbox,
            road_mask=road_mask,
        )
        if lane_instance is None:
            continue
        lane_instance.id = len(lane_instances)
        lane_instances.append(lane_instance)

        if class_name == "road-sign-line":
            continue

        lane_pred[inst_mask] = 2 if lane_instance.lane_type == "dashed" else 1

        curve = _lane_curve_from_maskrcnn_instance(
            frame_bgr=frame_bgr,
            mask_bool=inst_mask,
            class_name=class_name,
            score=score,
            road_mask=road_mask,
        )
        if curve is None:
            continue
        if any(_curve_duplicate(curve, prev, (H, W)) for prev in curves):
            continue
        curve.id = len(curves)
        curves.append(curve)

    curves.sort(key=lambda curve: _lane_track_key(curve)[2])
    for lane_id, curve in enumerate(curves):
        curve.id = lane_id
    return lane_pred, curves, lane_instances


def _lane_curve_from_maskrcnn_instance(
    frame_bgr: np.ndarray,
    mask_bool: np.ndarray,
    class_name: str,
    score: float,
    road_mask: Optional[np.ndarray],
) -> Optional[LaneCurve]:
    mask_u8 = mask_bool.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if int(np.count_nonzero(mask_u8)) < 60:
        return None

    used_relaxed_curve = False
    coeffs, curve_pts = fit_lane_curve_from_mask(mask_u8, degree=2, min_pixels=40)
    if coeffs is None or len(curve_pts) < 6:
        curve_pts = _sample_lane_centerline_from_mask(mask_u8)
        coeffs = _fit_poly_coeffs_from_curve_points(curve_pts, degree=2)
        used_relaxed_curve = True
    if len(curve_pts) < 6:
        return None

    lane_type = _lane_type_from_maskrcnn_class(class_name)
    metrics = _compute_lane_component_metrics(mask_u8, curve_pts, road_mask=road_mask)
    plausible = _lane_component_is_plausible(metrics, lane_type, frame_shape=mask_u8.shape)
    if not plausible and not used_relaxed_curve:
        curve_pts = _sample_lane_centerline_from_mask(mask_u8)
        coeffs = _fit_poly_coeffs_from_curve_points(curve_pts, degree=2)
        if len(curve_pts) >= 6:
            metrics = _compute_lane_component_metrics(mask_u8, curve_pts, road_mask=road_mask)
            used_relaxed_curve = True
            plausible = _lane_component_is_viable_relaxed(metrics, lane_type, frame_shape=mask_u8.shape)
    elif not plausible:
        plausible = _lane_component_is_viable_relaxed(metrics, lane_type, frame_shape=mask_u8.shape)
    if not plausible:
        return None

    fallback_color = _fallback_lane_color_from_instance(class_name, mask_bool, road_mask)
    color, color_conf = classify_paint_color_from_mask(
        frame_bgr,
        mask_u8,
        fallback=fallback_color,
        road_mask=road_mask,
    )
    avg_hsv = average_hsv_from_mask(frame_bgr, mask_u8)
    avg_ycrcb = average_ycrcb_from_mask(frame_bgr, mask_u8)
    pixel_count = int(np.count_nonzero(mask_u8))
    confidence = min(
        0.99,
        0.28 + 0.45 * float(score) + 0.12 * float(metrics["road_support"]) + 0.00045 * float(pixel_count),
    )
    if used_relaxed_curve or coeffs is None:
        confidence *= 0.88
    return LaneCurve(
        id=-1,
        lane_type=lane_type,
        line_class=str(class_name),
        color=color,
        pixel_count=pixel_count,
        confidence=float(confidence),
        curve_points_img=curve_pts.tolist(),
        poly_coeffs=coeffs.tolist() if coeffs is not None else None,
        color_confidence=float(max(color_conf, 0.35 + 0.30 * float(score))),
        avg_hsv=avg_hsv,
        avg_ycrcb=avg_ycrcb,
    )


def _road_mask_from_maskrcnn_instances(
    maskrcnn_instances: List[Dict[str, Any]],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    H, W = frame_shape[:2]
    road_mask = np.zeros((H, W), dtype=bool)
    allowed_labels = {
        "road",
        "road-area",
        "road_area",
        "road-roads",
        "roads",
        "drivable",
        "drivable-area",
        "drivable_area",
    }
    for inst in maskrcnn_instances:
        class_name = str(inst.get("class_name") or "").strip().lower()
        if class_name not in allowed_labels:
            continue
        inst_mask = np.asarray(inst.get("mask"), dtype=bool)
        if inst_mask.shape[:2] != (H, W):
            continue
        road_mask |= inst_mask
    if np.count_nonzero(road_mask) == 0:
        return road_mask
    road_mask = cv2.morphologyEx(
        road_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=2,
    ).astype(bool)
    return road_mask


def _fit_component_orientation_deg(contour: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        return 90.0
    try:
        vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx_scalar = float(np.asarray(vx).reshape(-1)[0])
        vy_scalar = float(np.asarray(vy).reshape(-1)[0])
        return float(abs(np.degrees(np.arctan2(vy_scalar, vx_scalar))))
    except cv2.error:
        return 90.0


def _bbox_gap(box_a: List[int], box_b: List[int]) -> Tuple[int, int]:
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [int(v) for v in box_b[:4]]
    gap_x = max(0, max(bx1 - ax2, ax1 - bx2))
    gap_y = max(0, max(by1 - ay2, ay1 - by2))
    return gap_x, gap_y


def _union_bbox(boxes: List[List[int]]) -> List[int]:
    xs1 = [int(box[0]) for box in boxes]
    ys1 = [int(box[1]) for box in boxes]
    xs2 = [int(box[2]) for box in boxes]
    ys2 = [int(box[3]) for box in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _cluster_marking_components(components: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not components:
        return []

    visited = [False] * len(components)
    clusters: List[List[Dict[str, Any]]] = []

    def linked(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        gap_x, gap_y = _bbox_gap(a["bbox"], b["bbox"])
        max_dim = max(a["bbox"][2] - a["bbox"][0], a["bbox"][3] - a["bbox"][1],
                      b["bbox"][2] - b["bbox"][0], b["bbox"][3] - b["bbox"][1])
        center_a = np.asarray(a["center"], dtype=np.float32)
        center_b = np.asarray(b["center"], dtype=np.float32)
        center_dist = float(np.linalg.norm(center_a - center_b))
        orientation_a = float(a.get("orientation_deg", 90.0))
        orientation_b = float(b.get("orientation_deg", 90.0))
        orientation_diff = abs(orientation_a - orientation_b)
        orientation_diff = min(orientation_diff, abs(180.0 - orientation_diff))
        aligned = (
            orientation_diff <= 38.0
            or min(float(a.get("elongation", 1.0)), float(b.get("elongation", 1.0))) <= 2.2
        )
        close = (
            center_dist <= max(72.0, 0.95 * float(max_dim))
            or (gap_x <= max(24, int(round(0.45 * max_dim))) and gap_y <= max(24, int(round(0.45 * max_dim))))
        )
        same_band = (
            abs(float(center_a[1] - center_b[1])) <= max(54.0, 0.85 * float(max_dim))
            and abs(float(center_a[0] - center_b[0])) <= max(110.0, 1.45 * float(max_dim))
        )
        return bool(aligned and close and same_band)

    for root_idx in range(len(components)):
        if visited[root_idx]:
            continue
        queue = [root_idx]
        visited[root_idx] = True
        cluster: List[Dict[str, Any]] = []
        while queue:
            idx = queue.pop()
            cluster.append(components[idx])
            for nxt in range(len(components)):
                if visited[nxt]:
                    continue
                if linked(components[idx], components[nxt]):
                    visited[nxt] = True
                    queue.append(nxt)
        clusters.append(cluster)

    return clusters


def _cluster_looks_like_lane_residual(cluster: List[Dict[str, Any]]) -> bool:
    if not cluster:
        return False

    boxes = [comp["bbox"] for comp in cluster]
    cluster_box = _union_bbox(boxes)
    bw = max(cluster_box[2] - cluster_box[0], 1)
    bh = max(cluster_box[3] - cluster_box[1], 1)
    cluster_aspect = max(float(bw) / float(bh), float(bh) / float(bw))

    orientations = [float(comp["orientation_deg"]) for comp in cluster]
    verticalish = sum(55.0 <= angle <= 125.0 for angle in orientations)
    elongated = sum(float(comp["elongation"]) >= 5.0 for comp in cluster)

    if len(cluster) == 1:
        comp = cluster[0]
        return bool(comp["elongation"] >= 6.0 and 55.0 <= comp["orientation_deg"] <= 125.0)

    if verticalish == len(cluster) and elongated >= max(1, len(cluster) - 1) and bh >= 2.4 * bw:
        return True

    centers_x = np.asarray([comp["center"][0] for comp in cluster], dtype=np.float32)
    centers_y = np.asarray([comp["center"][1] for comp in cluster], dtype=np.float32)
    spread_x = float(np.ptp(centers_x)) if len(cluster) > 1 else 0.0
    spread_y = float(np.ptp(centers_y)) if len(cluster) > 1 else 0.0
    if verticalish == len(cluster) and spread_y > 1.8 * max(spread_x, 1.0) and cluster_aspect >= 2.8:
        return True

    return False


def _contour_edge_count(contour: np.ndarray) -> int:
    peri = max(cv2.arcLength(contour, True), 1.0)
    approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
    return int(len(approx))


def _infer_arrow_direction(contour: np.ndarray) -> str:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return "forward"
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    ref_pts = hull if len(hull) >= 3 else pts
    centroid = np.mean(ref_pts, axis=0)
    dists = np.linalg.norm(ref_pts - centroid[None, :], axis=1)
    tip = ref_pts[int(np.argmax(dists))]
    delta = tip - centroid
    if abs(float(delta[0])) > 1.15 * abs(float(delta[1])):
        return "right" if float(delta[0]) > 0 else "left"
    return "forward" if float(delta[1]) < 0 else "backward"


def _contour_pointedness(contour: np.ndarray) -> float:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return 1.0
    centroid = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)
    denom = max(float(np.percentile(dists, 75)), 1e-3)
    return float(np.max(dists) / denom)


def _contour_shape_signature(contour: np.ndarray) -> Dict[str, float]:
    area = float(max(cv2.contourArea(contour), 1.0))
    peri = float(max(cv2.arcLength(contour, True), 1.0))
    approx = cv2.approxPolyDP(contour, 0.015 * peri, True).reshape(-1, 2).astype(np.float32)
    if approx.shape[0] < 3:
        approx = contour.reshape(-1, 2).astype(np.float32)

    if approx.shape[0] >= 2:
        seg_vecs = np.roll(approx, -1, axis=0) - approx
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        poly_peri = float(np.sum(seg_lengths))
        straight_edges = int(np.sum(seg_lengths >= max(10.0, 0.08 * peri)))
    else:
        seg_lengths = np.asarray([], dtype=np.float32)
        poly_peri = peri
        straight_edges = 0

    turn_angles: List[float] = []
    if approx.shape[0] >= 3:
        for idx in range(approx.shape[0]):
            prev_pt = approx[idx - 1]
            curr_pt = approx[idx]
            next_pt = approx[(idx + 1) % approx.shape[0]]
            v1 = prev_pt - curr_pt
            v2 = next_pt - curr_pt
            denom = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-6)
            cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
            turn_angles.append(float(np.degrees(np.arccos(cosang))))

    hull = cv2.convexHull(contour)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    hull_edges = int(len(cv2.approxPolyDP(hull, 0.018 * max(cv2.arcLength(hull, True), 1.0), True)))
    max_defect_depth = 0.0
    defect_count = 0
    if hull_indices is not None and len(hull_indices) >= 3:
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is not None:
            for defect in defects[:, 0, :]:
                defect_count += 1
                max_defect_depth = max(max_defect_depth, float(defect[3]) / 256.0)

    circularity = float((4.0 * math.pi * area) / max(peri * peri, 1e-6))
    rounded_edge_ratio = float(np.clip((peri - poly_peri) / peri, 0.0, 1.0))
    acute_turns = int(sum(angle <= 105.0 for angle in turn_angles))
    strong_turns = int(sum(angle <= 135.0 for angle in turn_angles))
    return {
        "straight_edges": float(straight_edges),
        "rounded_edge_ratio": rounded_edge_ratio,
        "acute_turns": float(acute_turns),
        "strong_turns": float(strong_turns),
        "hull_edges": float(hull_edges),
        "circularity": circularity,
        "max_defect_depth": float(max_defect_depth),
        "defect_count": float(defect_count),
    }


def _marking_iou(mark_a: RoadMarking, mark_b: RoadMarking) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in mark_a.bbox[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in mark_b.bbox[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-6 else 0.0


def _dedupe_markings(markings: List[RoadMarking]) -> List[RoadMarking]:
    kept: List[RoadMarking] = []
    ordered = sorted(
        markings,
        key=lambda item: (
            item.marking_type == "arrow",
            item.confidence,
            item.area_px,
        ),
        reverse=True,
    )
    for mark in ordered:
        if any(
            _marking_iou(mark, other) >= 0.35
            and mark.marking_type == other.marking_type
            for other in kept
        ):
            continue
        kept.append(mark)
    for idx, mark in enumerate(kept):
        mark.id = idx
    return kept


def _road_marking_from_mask(
    frame_bgr: np.ndarray,
    union_mask: np.ndarray,
    road_mask: np.ndarray,
    lane_mask: np.ndarray,
    road_dist: np.ndarray,
    lane_clear: np.ndarray,
    road_area: float,
    min_area: int = 140,
    cluster_size: int = 1,
    base_confidence: float = 0.0,
) -> Optional[RoadMarking]:
    H, W = road_mask.shape[:2]
    merged_u8 = (union_mask.astype(np.uint8) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(merged_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < float(min_area) or area > min(26000.0, 0.09 * road_area):
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w < 18 or h < 16 or y < int(0.24 * H):
        return None

    road_support = float(np.count_nonzero(union_mask & road_mask.astype(np.uint8))) / max(
        float(np.count_nonzero(union_mask)),
        1.0,
    )
    lane_overlap = float(np.count_nonzero(union_mask & lane_mask.astype(np.uint8))) / max(
        float(np.count_nonzero(union_mask)),
        1.0,
    )
    if road_support < 0.88 or lane_overlap > 0.10:
        return None

    aspect = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))
    bbox_area = max(float(w * h), 1.0)
    extent = area / bbox_area
    if extent < 0.04 or extent > 0.88:
        return None

    hull = cv2.convexHull(cnt)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = area / hull_area
    contour_edges = _contour_edge_count(cnt)
    hull_edges = _contour_edge_count(hull)
    concavity = max(0, contour_edges - hull_edges)
    pointedness = _contour_pointedness(cnt)
    shape_sig = _contour_shape_signature(cnt)
    straight_edges = float(shape_sig["straight_edges"])
    rounded_edge_ratio = float(shape_sig["rounded_edge_ratio"])
    acute_turns = float(shape_sig["acute_turns"])
    strong_turns = float(shape_sig["strong_turns"])
    max_defect_depth = float(shape_sig["max_defect_depth"])
    circularity = float(shape_sig["circularity"])

    m = cv2.moments(cnt)
    if abs(m.get("m00", 0.0)) > 1e-6:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
    else:
        cx = float(x + 0.5 * w)
        cy = float(y + 0.5 * h)
    ix = int(np.clip(round(cx), 0, W - 1))
    iy = int(np.clip(round(cy), 0, H - 1))
    road_margin = float(road_dist[iy, ix]) if road_dist.size else 0.0
    lane_margin = float(lane_clear[iy, ix]) if lane_clear.size else 0.0
    if road_margin < 12.0 or lane_margin < 10.0:
        return None

    color, color_conf = classify_paint_color_from_mask(
        frame_bgr,
        union_mask,
        fallback="white",
        road_mask=road_mask,
    )

    arrow_like = (
        6 <= contour_edges <= 14
        and 4 <= hull_edges <= 8
        and 1 <= concavity <= 4
        and 0.10 <= extent <= 0.62
        and 0.48 <= solidity <= 0.92
        and 28 <= w <= 320
        and 24 <= h <= 240
        and aspect <= 3.8
        and pointedness >= 1.28
        and area >= 720.0
        and road_support >= 0.92
        and lane_overlap <= 0.06
        and straight_edges >= 3.0
        and acute_turns >= 1.0
        and strong_turns >= 3.0
        and 0.03 <= rounded_edge_ratio <= 0.34
        and max_defect_depth >= 2.0
        and circularity <= 0.62
        and (cluster_size >= 2 or area >= 1200.0 or base_confidence >= 0.86)
    )
    marking_like = (
        area >= 1200.0
        and (cluster_size >= 2 or base_confidence >= 0.82)
        and 0.08 <= extent <= 0.80
        and 0.24 <= solidity <= 1.0
        and aspect <= 6.5
        and road_support >= 0.92
        and lane_overlap <= 0.05
        and (straight_edges >= 2.0 or rounded_edge_ratio >= 0.10)
        and strong_turns >= 2.0
        and circularity <= 0.80
    )
    direction = _infer_arrow_direction(cnt) if arrow_like else None
    if direction == "backward":
        arrow_like = False
        direction = None
    if not arrow_like and not marking_like:
        return None

    confidence = 0.20
    confidence += 0.24 * min(area / 2200.0, 1.0)
    confidence += 0.10 * min(cluster_size / 4.0, 1.0)
    confidence += 0.14 * min(max(concavity, 1) / 4.0, 1.0)
    confidence += 0.10 * min(max(pointedness - 1.0, 0.0) / 0.35, 1.0)
    confidence += 0.12 * max(0.0, 1.0 - min(abs(extent - 0.34) / 0.34, 1.0))
    confidence += 0.10 * max(0.0, 1.0 - min(abs(solidity - 0.72) / 0.72, 1.0))
    confidence += 0.08 * min(float(color_conf), 1.0)
    confidence += 0.08 * min(straight_edges / 5.0, 1.0)
    confidence += 0.06 * min(max(rounded_edge_ratio, 0.0) / 0.22, 1.0)
    confidence += 0.18 * min(float(base_confidence), 1.0)
    if arrow_like:
        confidence += 0.12
    confidence = float(np.clip(confidence, 0.05, 0.99))

    approx = cv2.approxPolyDP(cnt, 0.012 * max(cv2.arcLength(cnt, True), 1.0), True)
    return RoadMarking(
        id=-1,
        marking_type="arrow" if arrow_like else "road_marking",
        color=color,
        area_px=int(round(area)),
        confidence=confidence,
        bbox=[int(x), int(y), int(x + w), int(y + h)],
        contour_img=approx.reshape(-1, 2).astype(int).tolist(),
        direction=direction,
    )


def _build_marking_candidate_mask(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray,
    lane_pred: np.ndarray,
) -> np.ndarray:
    H, W = road_mask.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    road_vals = gray_eq[road_mask]
    if road_vals.size == 0:
        return np.zeros_like(road_mask, dtype=np.uint8)

    sat = hsv[:, :, 1]
    bright_floor = max(132.0, float(np.percentile(road_vals, 74)))
    road_dilated = cv2.dilate(
        road_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    ).astype(bool)
    road_dist = cv2.distanceTransform((road_dilated.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    road_core = road_dilated & (road_dist >= 7.0)
    adaptive = cv2.adaptiveThreshold(
        gray_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -8,
    )
    edges = cv2.Canny(gray_eq, threshold1=55, threshold2=150, apertureSize=3, L2gradient=True)

    bright_paint = (gray_eq >= bright_floor) & (sat <= 118)
    adaptive_paint = (adaptive > 0) & (gray_eq >= bright_floor - 18) & (sat <= 140)
    lane_u8 = cv2.dilate(
        (lane_pred > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    ) * 255
    interior_u8 = (road_core.astype(np.uint8) * 255)

    paint_seed = cv2.bitwise_and(((bright_paint | adaptive_paint).astype(np.uint8) * 255), interior_u8)
    paint_seed = cv2.bitwise_and(paint_seed, cv2.bitwise_not(lane_u8))
    edge_seed = cv2.bitwise_and(
        cv2.dilate(cv2.bitwise_and(edges, interior_u8), np.ones((3, 3), np.uint8), iterations=1),
        cv2.dilate(paint_seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1),
    )
    candidate = cv2.bitwise_or(
        edge_seed,
        cv2.bitwise_and(
            paint_seed,
            cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1),
        ),
    )
    candidate[: max(1, int(round(0.26 * H))), :] = 0

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return candidate


def detect_road_markings(
    frame_bgr: np.ndarray,
    road_mask: np.ndarray,
    lane_pred: np.ndarray,
    maskrcnn_instances: Optional[List[Dict[str, Any]]] = None,
    min_area: int = 140,
    max_markings: int = 10,
) -> List[RoadMarking]:
    """
    Detect bright painted road markings inside the drivable-area mask.

    This is aimed at road-surface symbols such as directional arrows, while
    remaining conservative enough not to flood the scene with lane pixels.
    """
    if not road_mask.any():
        return []

    H, W = road_mask.shape[:2]
    mask_u8 = _build_marking_candidate_mask(frame_bgr, road_mask, lane_pred)
    lane_mask = lane_pred > 0
    road_dist = cv2.distanceTransform((road_mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    lane_clear = cv2.distanceTransform((~lane_mask).astype(np.uint8) * 255, cv2.DIST_L2, 5)
    road_area = max(float(np.count_nonzero(road_mask)), 1.0)

    candidates: List[RoadMarking] = []
    if maskrcnn_instances:
        for inst in maskrcnn_instances:
            if str(inst.get("class_name")) != "road-sign-line":
                continue
            inst_mask = np.asarray(inst.get("mask"), dtype=np.uint8)
            if inst_mask.shape[:2] != road_mask.shape[:2]:
                continue
            marking = _road_marking_from_mask(
                frame_bgr=frame_bgr,
                union_mask=inst_mask,
                road_mask=road_mask,
                lane_mask=lane_mask,
                road_dist=road_dist,
                lane_clear=lane_clear,
                road_area=road_area,
                min_area=min_area,
                cluster_size=1,
                base_confidence=float(inst.get("score", 0.0)),
            )
            if marking is not None:
                candidates.append(marking)

    filled_from_edges = np.zeros_like(mask_u8)
    edge_contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in edge_contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area) or area > min(26000.0, 0.08 * road_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 14 or h < 14 or y < int(0.24 * H):
            continue
        cv2.drawContours(filled_from_edges, [cnt], -1, 255, thickness=cv2.FILLED)
    if np.count_nonzero(filled_from_edges) > 0:
        mask_u8 = filled_from_edges

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    components: List[Dict[str, Any]] = []
    for label_idx in range(1, n):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area < float(min_area):
            continue
        if area > min(14000.0, 0.05 * road_area):
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if w < 12 or h < 12:
            continue
        if y < int(0.24 * H):
            continue

        comp_mask = np.zeros((H, W), dtype=np.uint8)
        comp_mask[labels == label_idx] = 1
        component_u8 = (comp_mask[y : y + h, x : x + w] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cnt[:, 0, 0] += x
        cnt[:, 0, 1] += y
        orientation_deg = _fit_component_orientation_deg(cnt)
        elongation = max(float(w) / max(float(h), 1.0), float(h) / max(float(w), 1.0))

        components.append(
            {
                "area": area,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "center": [float(centroids[label_idx][0]), float(centroids[label_idx][1])],
                "mask": comp_mask,
                "contour": cnt,
                "orientation_deg": orientation_deg,
                "elongation": elongation,
            }
        )

    for cluster in _cluster_marking_components(components):
        if _cluster_looks_like_lane_residual(cluster):
            continue

        union_mask = np.zeros((H, W), dtype=np.uint8)
        for comp in cluster:
            union_mask |= comp["mask"].astype(np.uint8)
        marking = _road_marking_from_mask(
            frame_bgr=frame_bgr,
            union_mask=union_mask,
            road_mask=road_mask,
            lane_mask=lane_mask,
            road_dist=road_dist,
            lane_clear=lane_clear,
            road_area=road_area,
            min_area=min_area,
            cluster_size=len(cluster),
            base_confidence=0.0,
        )
        if marking is not None:
            candidates.append(marking)

    return _dedupe_markings(candidates)[:max_markings]


def _lane_track_key(curve: LaneCurve) -> Tuple[str, float, float]:
    pts = np.asarray(curve.curve_points_img, dtype=np.float32)
    if pts.size == 0:
        return curve.lane_type, 0.0, 0.0
    return curve.lane_type, float(pts[0, 0]), float(pts[-1, 0])


def _lane_track_distance(curve: LaneCurve, track: Dict[str, Any]) -> float:
    lane_type, top_x, bottom_x = _lane_track_key(curve)
    if lane_type != track["lane_type"]:
        return float("inf")
    misses = float(track.get("misses", 0))
    miss_penalty = 1.0 + 0.15 * misses
    return (abs(top_x - track["top_x"]) + 1.4 * abs(bottom_x - track["bottom_x"])) * miss_penalty


def _marking_center(marking: RoadMarking) -> Tuple[float, float]:
    x1, y1, x2, y2 = marking.bbox
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def _marking_track_distance(marking: RoadMarking, track: Dict[str, Any]) -> float:
    if marking.marking_type != track["marking_type"]:
        return float("inf")
    cx, cy = _marking_center(marking)
    misses = float(track.get("misses", 0))
    miss_penalty = 1.0 + 0.20 * misses
    return (abs(cx - track["cx"]) + 1.2 * abs(cy - track["cy"])) * miss_penalty


def _road_center_x(road_mask: np.ndarray) -> Optional[float]:
    H, W = road_mask.shape[:2]
    rows = [int(round(0.82 * H)), int(round(0.88 * H)), int(round(0.94 * H))]
    xs: List[float] = []
    for y in rows:
        y = int(np.clip(y, 0, H - 1))
        row = np.where(road_mask[y])[0]
        if row.size >= 20:
            xs.append(float(np.mean(row)))
    if not xs:
        return None
    return float(np.mean(xs))


def _ema_curve_points(prev_pts: List[List[int]], curr_pts: List[List[int]], alpha: float = 0.55) -> List[List[int]]:
    if not prev_pts:
        return [list(map(int, pt[:2])) for pt in curr_pts]
    if not curr_pts:
        return [list(map(int, pt[:2])) for pt in prev_pts]
    prev = np.asarray(prev_pts, dtype=np.float32)
    curr = np.asarray(curr_pts, dtype=np.float32)
    if prev.shape != curr.shape:
        return [list(map(int, pt[:2])) for pt in curr_pts]
    blended = alpha * curr + (1.0 - alpha) * prev
    return np.round(blended).astype(int).tolist()


# =============================================================================
# 9.  Lane Segmentation Pipeline  (P2 SegNet)
# =============================================================================

class LaneSegmentationPipeline:
    """
    Thin wrapper around the P2 SegNet checkpoint.
    Outputs per-pixel lane class (0=bg, 1=solid, 2=dashed) and fitted LaneCurve objects.
    """

    PALETTE = {
        1: (0, 0, 255),       # solid  — red
        2: (38, 255, 255),    # dashed — cyan-yellow
    }

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        input_width: int = 224,
        input_height: int = 224,
    ) -> None:
        self.device = resolve_device(device)
        self.input_width = input_width
        self.input_height = input_height

        print(f"[LaneSegmentationPipeline] Loading checkpoint: {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location=self.device)

        if isinstance(ckpt, nn.Module):
            self.model = ckpt.to(self.device)
            print("[LaneSegmentationPipeline] Loaded full model object.")
        elif isinstance(ckpt, dict):
            state = None
            for key in ("model", "state_dict", "model_state_dict"):
                if key in ckpt:
                    state = ckpt[key]
                    break
            if state is None:
                state = ckpt

            if RepoSegNet is not None:
                self.model = RepoSegNet(num_classes=3).to(self.device)
            else:
                raise ImportError(
                    "[LaneSegmentationPipeline] Cannot import SegNet from the external P2 repo. "
                    f"Expected under {P2_LANE_SEG_DIR}."
                )
            self.model.load_state_dict(state, strict=False)
            print("[LaneSegmentationPipeline] Loaded state_dict into P2 SegNet (3 classes).")
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

        self.model.eval()

    def infer(
        self,
        frame_bgr: np.ndarray,
        tensor: torch.Tensor,
        orig_wh: Tuple[int, int],
        road_mask: Optional[np.ndarray] = None,
        debug_classes: bool = False,
    ) -> Tuple[np.ndarray, List[LaneCurve]]:
        """
        Run lane inference on a pre-processed tensor.

        Parameters
        ----------
        frame_bgr : original BGR frame (for visualisation only)
        tensor : (1, 3, H_in, W_in) already on device
        orig_wh : (orig_W, orig_H)
        debug_classes: print unique predicted class ids (first frame only)

        Returns
        -------
        pred_full : (orig_H, orig_W) uint8 — 0/1/2 class map
        curves : List[LaneCurve]
        """
        orig_W, orig_H = orig_wh

        with torch.no_grad():
            logits = self.model(tensor)
            pred_tensor = torch.argmax(logits, dim=1)[0].detach().to(torch.uint8).cpu()
            pred = np.asarray(pred_tensor.tolist(), dtype=np.uint8)

        pred_full = cv2.resize(pred, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)
        pred_filtered = np.zeros_like(pred_full, dtype=np.uint8)

        if debug_classes:
            print(f"  [Lane] unique pred classes: {np.unique(pred_full)}")

        candidates: List[Dict[str, Any]] = []

        for cls_id, lane_type in [(1, "solid"), (2, "dashed")]:
            class_mask = (pred_full == cls_id).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)

            for comp in connected_lane_components(class_mask, min_area=80):
                ys, xs = np.where(comp > 0)
                if len(xs) < 80:
                    continue

                coeffs, curve_pts = fit_lane_curve_from_mask(comp, degree=2, min_pixels=40)
                if coeffs is None or len(curve_pts) < 6:
                    continue

                metrics = _compute_lane_component_metrics(comp, curve_pts, road_mask=road_mask)
                if not _lane_component_is_plausible(metrics, lane_type, frame_shape=pred_full.shape):
                    continue

                fallback_color = "yellow" if metrics["bottom_x"] < 0.58 * pred_full.shape[1] else "white"
                lane_color, color_conf = classify_paint_color_from_mask(
                    frame_bgr,
                    comp,
                    fallback=fallback_color,
                    road_mask=road_mask,
                )
                lane = LaneCurve(
                    id=-1,
                    lane_type=lane_type,
                    line_class=_default_line_class_for_lane_type(lane_type),
                    color=lane_color,
                    pixel_count=int(len(xs)),
                    confidence=min(0.99, 0.40 + 0.0025 * len(xs) + 0.20 * metrics["road_support"]),
                    curve_points_img=curve_pts.tolist(),
                    poly_coeffs=coeffs.tolist(),
                    color_confidence=color_conf,
                    avg_hsv=average_hsv_from_mask(frame_bgr, comp),
                    avg_ycrcb=average_ycrcb_from_mask(frame_bgr, comp),
                )
                candidates.append(
                    {
                        "lane": lane,
                        "mask": comp.astype(bool),
                        "class_id": cls_id,
                        "metrics": metrics,
                    }
                )

        kept = _suppress_redundant_lane_candidates(candidates, frame_shape=pred_full.shape)
        if not kept and candidates:
            kept = sorted(candidates, key=_lane_candidate_score, reverse=True)[:2]
        curves: List[LaneCurve] = []
        for lane_id, entry in enumerate(kept):
            lane = entry["lane"]
            lane.id = lane_id
            curves.append(lane)
            pred_filtered[entry["mask"]] = int(entry["class_id"])

        return pred_filtered, curves


# =============================================================================
# 10.  Road Segmentation Pipeline  (P1 UNet)
# =============================================================================

class RoadSegmentationPipeline:
    """
    Wrapper around the P1 UNet checkpoint for drivable freespace segmentation.

    Outputs a binary mask (bool) at original frame resolution plus extracted
    contour polygons ready for the scene assembler / Blender renderer.

    Checkpoint compatibility
    ------------------------
    Supports both:
      • Full model object (torch.save(model, path))
      • State dict (torch.save(model.state_dict(), path))
        with optional "model" / "state_dict" / "model_state_dict" wrapper key.

    If the local P1 repo is present, attempts to import its UNet first; falls
    back to the inline RoadUNet defined in this file.
    """

    ROAD_COLOR = (0, 200, 80)   # BGR  — bright green overlay

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        input_width: int = 224,
        input_height: int = 224,
        n_classes: int = 2,
    ) -> None:
        self.device = resolve_device(device)
        self.input_width = input_width
        self.input_height = input_height
        self.n_classes = n_classes

        print(f"[RoadSegmentationPipeline] Loading checkpoint: {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location=self.device)

        if isinstance(ckpt, nn.Module):
            self.model = ckpt.to(self.device)
            print("[RoadSegmentationPipeline] Loaded full model object.")
        elif isinstance(ckpt, dict):
            state = None
            for key in ("model", "state_dict", "model_state_dict"):
                if key in ckpt:
                    state = ckpt[key]
                    break
            if state is None:
                state = ckpt

            # Try importing the repo's UNet first
            UNetClass = self._try_import_repo_unet()
            if UNetClass is None:
                UNetClass = RoadUNet
                print("[RoadSegmentationPipeline] Using inline RoadUNet.")
            else:
                print("[RoadSegmentationPipeline] Using external P1 FoInternNet.")

            self.model = self._build_model(UNetClass).to(self.device)
            self.model.load_state_dict(state, strict=False)
            print(f"[RoadSegmentationPipeline] Loaded state_dict (n_classes={n_classes}).")
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt)}")

        self.model.eval()

    @staticmethod
    def _try_import_repo_unet() -> Optional[type]:
        """Return the FoInternNet class from the cloned P1 repo when available."""
        if not _repo_unet_available:
            return None
        return RepoRoadUNet

    def _build_model(self, model_cls: type) -> nn.Module:
        try:
            return model_cls(
                input_size=(self.input_height, self.input_width),
                n_classes=self.n_classes,
            )
        except TypeError:
            return model_cls(n_classes=self.n_classes)

    def infer(
        self,
        tensor: torch.Tensor,
        orig_wh: Tuple[int, int],
        min_road_area: int = 2000,
        debug_classes: bool = False,
    ) -> RoadResult:
        """
        Run road inference on a pre-processed tensor.

        Parameters
        ----------
        tensor : (1, 3, H_in, W_in) already on device
        orig_wh : (orig_W, orig_H)
        min_road_area : minimum contour area to include in output polygons
        debug_classes : print unique predicted class ids (first frame only)

        Returns
        -------
        RoadResult
        """
        orig_W, orig_H = orig_wh
        t0 = time.time()

        with torch.no_grad():
            out = self.model(tensor)               # softmax probabilities
            pred_tensor = torch.argmax(out, dim=1)[0].detach().to(torch.uint8).cpu()
            pred = np.asarray(pred_tensor.tolist(), dtype=np.uint8)

        pred_full = cv2.resize(pred, (orig_W, orig_H), interpolation=cv2.INTER_NEAREST)

        if debug_classes:
            print(f"  [Road] unique pred classes: {np.unique(pred_full)}")

        # Class 1 = freespace / drivable area  (P1 training convention)
        road_mask = pred_full == 1
        area_frac = float(road_mask.mean())
        contours = extract_road_contours(road_mask, min_area=min_road_area)
        proc_ms = (time.time() - t0) * 1000.0

        return RoadResult(
            mask_bool = road_mask,
            area_frac = area_frac,
            contours_img = contours,
            proc_ms = proc_ms,
        )


# =============================================================================
# 11.  Combined pipeline
# =============================================================================

class CombinedPipeline:
    """
    Runs lane and road segmentation simultaneously on every frame.

    Both models share the same preprocessing tensor to avoid double work.
    The combined JSON output feeds directly into scene_assembler.py and
    blender_renderer.py.

    Parameters
    ----------
    lane_checkpoint : path to SegNet P2 checkpoint (.pt)
    road_checkpoint : path to UNet  P1 checkpoint (.pt)
    device : "auto" | "cuda" | "mps" | "cpu"
    input_width : model input width  (default 224, matches both repos)
    input_height : model input height (default 224)
    road_n_classes : number of output classes for road model (default 2)
    """

    def __init__(
        self,
        lane_checkpoint: str,
        road_checkpoint: str,
        device: str = "auto",
        input_width: int = 224,
        input_height: int = 224,
        road_n_classes: int = 2,
        maskrcnn_backend: str = "torchvision",
        maskrcnn_weights: Optional[str] = "auto",
        maskrcnn_score_threshold: float = 0.65,
        maskrcnn_mask_threshold: float = 0.45,
        maskrcnn_max_dets: int = 64,
    ) -> None:
        self.device = resolve_device(device)
        self.input_width = input_width
        self.input_height = input_height

        self.lane_pipe = LaneSegmentationPipeline(
            checkpoint = lane_checkpoint,
            device = self.device,
            input_width = input_width,
            input_height = input_height,
        )
        self.road_pipe = RoadSegmentationPipeline(
            checkpoint = road_checkpoint,
            device = self.device,
            input_width = input_width,
            input_height = input_height,
            n_classes = road_n_classes,
        )
        self._frame_ordinal = 0
        self._lane_tracks: List[Dict[str, Any]] = []
        self._marking_tracks: List[Dict[str, Any]] = []
        self._last_lane_instances: List[LaneInstanceResult] = []
        self.maskrcnn_backend = str(maskrcnn_backend or "torchvision").lower()
        self.maskrcnn_weights_path = _resolve_optional_maskrcnn_weights(maskrcnn_weights)
        self.maskrcnn_refiner: Optional[Any] = None
        if self.maskrcnn_weights_path:
            try:
                if self.maskrcnn_backend == "omonsun" or self.maskrcnn_weights_path.endswith(".h5"):
                    self.maskrcnn_refiner = OmonsunMaskRCNNRefiner(
                        weights_path=self.maskrcnn_weights_path,
                        score_threshold=maskrcnn_score_threshold,
                        mask_threshold=maskrcnn_mask_threshold,
                        max_dets=maskrcnn_max_dets,
                    )
                else:
                    self.maskrcnn_refiner = MaskRCNNLaneRefiner(
                        weights_path=self.maskrcnn_weights_path,
                        device=self.device,
                        score_threshold=maskrcnn_score_threshold,
                        mask_threshold=maskrcnn_mask_threshold,
                        max_dets=maskrcnn_max_dets,
                    )
            except Exception as exc:
                warnings.warn(
                    "[CombinedPipeline] Mask R-CNN refiner could not be initialized; "
                    f"continuing with SegNet/UNet only. ({exc})"
                )
                self.maskrcnn_refiner = None

        print(f"\n[CombinedPipeline] Ready  device='{self.device}'  "
              f"input={input_width}×{input_height}")
        if self.maskrcnn_refiner is not None:
            print(
                f"[CombinedPipeline] Primary lane refiner/backend: "
                f"{Path(self.maskrcnn_weights_path).name}"
            )
        else:
            print("[CombinedPipeline] Primary lane refiner/backend: unavailable (using fallback repos)")

    def _infer_maskrcnn_instances(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if self.maskrcnn_refiner is None:
            return []
        try:
            return self.maskrcnn_refiner.infer(frame_bgr)
        except Exception as exc:
            warnings.warn(
                "[CombinedPipeline] Mask R-CNN inference failed on this frame; "
                f"falling back to SegNet/UNet only. ({exc})"
            )
            return []

    def _stabilize_lanes(
        self,
        curves: List[LaneCurve],
        road_mask: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> List[LaneCurve]:
        H, W = frame_shape[:2]
        road_center = _road_center_x(road_mask)
        prev_tracks = sorted(self._lane_tracks, key=lambda item: float(item.get("bottom_x", 0.0)))
        next_tracks: List[Dict[str, Any]] = []
        stabilized: List[LaneCurve] = []
        used_prev: set[int] = set()

        for curve in sorted(curves, key=lambda item: _lane_track_key(item)[2]):
            best_track = None
            best_track_idx = -1
            best_dist = float("inf")
            for track_idx, track in enumerate(prev_tracks):
                if track_idx in used_prev:
                    continue
                dist = _lane_track_distance(curve, track)
                if dist < best_dist:
                    best_dist = dist
                    best_track = track
                    best_track_idx = track_idx

            match_ok = best_track is not None and best_dist <= max(95.0, 0.09 * float(W))
            lane_type, top_x, bottom_x = _lane_track_key(curve)
            if match_ok:
                used_prev.add(best_track_idx)
                hits = int(best_track["hits"]) + 1
                color_votes = dict(best_track["color_votes"])
                smoothed_points = _ema_curve_points(best_track.get("curve_points_img", []), curve.curve_points_img)
            else:
                hits = 1
                color_votes = {"white": 0.0, "yellow": 0.0, "unknown": 0.0}
                smoothed_points = curve.curve_points_img

            color_votes[curve.color] = color_votes.get(curve.color, 0.0) + max(curve.color_confidence, 0.25)
            stable_color = max(color_votes.items(), key=lambda item: item[1])[0]
            curve.color = stable_color
            if match_ok and curve.color_confidence < 0.55:
                curve.color_confidence = min(0.95, max(curve.color_confidence, best_track.get("color_confidence", 0.0) * 0.92))
            curve.curve_points_img = smoothed_points

            if road_center is not None and curve.color_confidence < 0.58:
                if bottom_x < road_center - 0.06 * float(W) and lane_type in {"solid", "dashed"}:
                    curve.color = "yellow"
                elif bottom_x >= road_center + 0.02 * float(W):
                    curve.color = "white"

            pts = np.asarray(curve.curve_points_img, dtype=np.float32)
            curve_span = float(np.ptp(pts[:, 1])) if len(pts) >= 2 else 0.0
            keep = (
                hits >= 2
                or curve.pixel_count >= 420
                or curve_span >= 0.12 * float(H)
            )
            if keep:
                curve.confidence = min(0.99, max(curve.confidence, 0.45 + 0.08 * min(hits, 4)))
                stabilized.append(curve)

            next_tracks.append(
                {
                    "lane_type": lane_type,
                    "top_x": top_x,
                    "bottom_x": bottom_x,
                    "hits": hits,
                    "misses": 0,
                    "color_votes": color_votes,
                    "color_confidence": curve.color_confidence,
                    "curve_points_img": curve.curve_points_img,
                    "lane_type_out": curve.lane_type,
                    "line_class": curve.line_class,
                    "color": curve.color,
                    "pixel_count": curve.pixel_count,
                    "confidence": curve.confidence,
                    "poly_coeffs": curve.poly_coeffs,
                    "avg_hsv": curve.avg_hsv,
                    "avg_ycrcb": curve.avg_ycrcb,
                }
            )

        # Carry strong lanes for a short time when the current frame is weak.
        if len(stabilized) < 2:
            for track_idx, track in enumerate(prev_tracks):
                if track_idx in used_prev:
                    continue
                misses = int(track.get("misses", 0)) + 1
                if int(track.get("hits", 0)) < 3 or misses > 2:
                    continue
                carry_curve = LaneCurve(
                    id=-1,
                    lane_type=str(track.get("lane_type_out", track["lane_type"])),
                    line_class=str(track.get("line_class", _default_line_class_for_lane_type(track.get("lane_type_out", track["lane_type"])))),
                    color=str(track.get("color", "white")),
                    pixel_count=int(track.get("pixel_count", 0)),
                    confidence=float(max(0.35, float(track.get("confidence", 0.6)) * 0.90)),
                    curve_points_img=[list(map(int, pt[:2])) for pt in track.get("curve_points_img", [])],
                    poly_coeffs=track.get("poly_coeffs"),
                    color_confidence=float(max(0.45, float(track.get("color_confidence", 0.5)) * 0.92)),
                    avg_hsv=track.get("avg_hsv"),
                    avg_ycrcb=track.get("avg_ycrcb"),
                )
                if len(carry_curve.curve_points_img) >= 2:
                    stabilized.append(carry_curve)
                    next_tracks.append(
                        {
                            **track,
                            "misses": misses,
                        }
                    )
                if len(stabilized) >= 3:
                    break

        stabilized.sort(key=lambda item: _lane_track_key(item)[2])
        for idx, curve in enumerate(stabilized):
            curve.id = idx

        self._lane_tracks = sorted(next_tracks, key=lambda item: float(item.get("bottom_x", 0.0)))[:8]
        return stabilized

    def _stabilize_markings(
        self,
        markings: List[RoadMarking],
        lane_pred: np.ndarray,
        road_mask: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> List[RoadMarking]:
        H, W = frame_shape[:2]
        lane_mask = lane_pred > 0
        road_dist = cv2.distanceTransform((road_mask.astype(np.uint8) * 255), cv2.DIST_L2, 5)
        lane_clear = cv2.distanceTransform((~lane_mask).astype(np.uint8) * 255, cv2.DIST_L2, 5)
        prev_tracks = self._marking_tracks
        next_tracks: List[Dict[str, Any]] = []
        stabilized: List[RoadMarking] = []
        used_prev: set[int] = set()

        for marking in markings:
            best_track = None
            best_track_idx = -1
            best_dist = float("inf")
            for track_idx, track in enumerate(prev_tracks):
                if track_idx in used_prev:
                    continue
                dist = _marking_track_distance(marking, track)
                if dist < best_dist:
                    best_dist = dist
                    best_track = track
                    best_track_idx = track_idx

            match_ok = best_track is not None and best_dist <= max(130.0, 0.12 * float(W) + 0.10 * float(H))
            cx, cy = _marking_center(marking)
            ix = int(np.clip(round(cx), 0, W - 1))
            iy = int(np.clip(round(cy), 0, H - 1))
            road_margin = float(road_dist[iy, ix]) if road_dist.size else 0.0
            lane_margin = float(lane_clear[iy, ix]) if lane_clear.size else 0.0
            hits = int(best_track["hits"]) + 1 if match_ok else 1
            if match_ok:
                used_prev.add(best_track_idx)
            if match_ok and not marking.direction and best_track.get("direction"):
                marking.direction = str(best_track["direction"])

            keep = (
                (marking.marking_type == "arrow" and ((hits >= 2 and marking.confidence >= 0.72) or marking.confidence >= 0.92))
                or (
                    marking.marking_type == "road_marking"
                    and (
                        (hits >= 3 and marking.confidence >= 0.74)
                        or (hits >= 2 and marking.area_px >= 1800 and marking.confidence >= 0.78)
                    )
                )
            )
            if road_margin < 8.0 or lane_margin < 7.0:
                keep = False
            if marking.marking_type == "road_marking" and marking.area_px < 700:
                keep = False
            if keep:
                stabilized.append(marking)

            next_tracks.append(
                {
                    "marking_type": marking.marking_type,
                    "cx": cx,
                    "cy": cy,
                    "hits": hits,
                    "misses": 0,
                    "direction": marking.direction,
                }
            )

        # Carry recently confirmed symbols briefly if they momentarily drop out.
        for track_idx, track in enumerate(prev_tracks):
            if track_idx in used_prev:
                continue
            misses = int(track.get("misses", 0)) + 1
            mark_type = track.get("marking_type")
            hit_req = 3 if mark_type == "arrow" else 4
            miss_cap = 1 if mark_type == "arrow" else 1
            if int(track.get("hits", 0)) < hit_req or misses > miss_cap:
                continue
            next_tracks.append({**track, "misses": misses})

        self._marking_tracks = next_tracks[:12]
        stabilized.sort(key=lambda item: (item.marking_type == "arrow", item.confidence, item.area_px), reverse=True)
        for idx, marking in enumerate(stabilized):
            marking.id = idx
        return stabilized

    # ── Core inference ────────────────────────────────────────────────────────

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        debug_classes: bool = False,
    ) -> Tuple[np.ndarray, List[LaneCurve], RoadResult, np.ndarray]:
        """
        Run both models on a single BGR frame.

        Returns
        -------
        lane_pred : (H, W) uint8  — lane class map (0/1/2)
        curves : List[LaneCurve]
        road_result: RoadResult
        vis : annotated BGR frame
        """
        t0 = time.time()

        # Shared preprocessing — tensor is reused for both models
        tensor, orig_wh = preprocess_frame(
            frame_bgr, self.input_width, self.input_height, self.device
        )

        road_result = self.road_pipe.infer(
            tensor, orig_wh, debug_classes=debug_classes
        )
        maskrcnn_instances = self._infer_maskrcnn_instances(frame_bgr)
        maskrcnn_road = _road_mask_from_maskrcnn_instances(maskrcnn_instances, frame_bgr.shape)
        if np.count_nonzero(maskrcnn_road) > 0:
            road_result.mask_bool = road_result.mask_bool | maskrcnn_road
            road_result.area_frac = float(road_result.mask_bool.mean())
            road_result.contours_img = extract_road_contours(road_result.mask_bool)
        maskrcnn_lane_pred, maskrcnn_curves, lane_instances = _lane_results_from_maskrcnn_instances(
            frame_bgr=frame_bgr,
            road_mask=road_result.mask_bool,
            maskrcnn_instances=maskrcnn_instances,
            frame_shape=frame_bgr.shape,
        )
        use_maskrcnn_primary = bool(maskrcnn_curves) or bool(np.any(maskrcnn_lane_pred > 0))
        if use_maskrcnn_primary:
            lane_pred = maskrcnn_lane_pred
            curves = maskrcnn_curves
        else:
            lane_pred, curves = self.lane_pipe.infer(
                frame_bgr,
                tensor,
                orig_wh,
                road_mask=road_result.mask_bool,
                debug_classes=debug_classes,
            )
        road_result.markings = detect_road_markings(
            frame_bgr=frame_bgr,
            road_mask=road_result.mask_bool,
            lane_pred=lane_pred,
            maskrcnn_instances=maskrcnn_instances,
        )
        curves = self._stabilize_lanes(curves, road_mask=road_result.mask_bool, frame_shape=frame_bgr.shape)
        road_result.markings = self._stabilize_markings(
            road_result.markings,
            lane_pred=lane_pred,
            road_mask=road_result.mask_bool,
            frame_shape=frame_bgr.shape,
        )

        lane_pred_stable = np.zeros_like(lane_pred, dtype=np.uint8)
        for curve in curves:
            pts = np.asarray(curve.curve_points_img, dtype=np.int32)
            if len(pts) < 2:
                continue
            thickness = 8 if curve.lane_type == "solid" else 6
            cv2.polylines(
                lane_pred_stable,
                [pts.reshape(-1, 1, 2)],
                False,
                1 if curve.lane_type == "solid" else 2,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
        if np.any(lane_pred_stable > 0):
            lane_pred = lane_pred_stable

        road_result.proc_ms = (time.time() - t0) * 1000.0
        self._last_lane_instances = lane_instances
        vis = self.draw(frame_bgr, lane_pred, curves, road_result, lane_instances=lane_instances)
        self._frame_ordinal += 1
        return lane_pred, curves, road_result, vis

    # ── Visualisation ─────────────────────────────────────────────────────────

    def draw(
        self,
        frame_bgr: np.ndarray,
        lane_pred: np.ndarray,
        curves: List[LaneCurve],
        road_result: RoadResult,
        lane_instances: Optional[List[LaneInstanceResult]] = None,
        alpha_road: float = 0.30,
        alpha_lane: float = 0.45,
    ) -> np.ndarray:
        """
        Render combined overlays onto a copy of *frame_bgr*.

        Layer order (bottom → top)
        --------------------------
        1. Road mask  — translucent green fill
        2. Lane mask  — translucent per-class tint
        3. Road contour outlines
        4. Fitted lane polylines + labels
        5. HUD strip (top-left)
        """
        vis = frame_bgr.copy()
        H, W = vis.shape[:2]
        lane_instances = lane_instances or []
        has_maskrcnn_lane_instances = any(inst.class_name != "road-sign-line" for inst in lane_instances)
        has_maskrcnn_marking_instances = any(inst.class_name == "road-sign-line" for inst in lane_instances)

        # ── 1. Road mask overlay ──────────────────────────────────────────────
        if road_result.mask_bool.any():
            overlay = vis.copy()
            overlay[road_result.mask_bool] = RoadSegmentationPipeline.ROAD_COLOR
            cv2.addWeighted(overlay, alpha_road, vis, 1.0 - alpha_road, 0, vis)

        # ── 2. Lane overlay ───────────────────────────────────────────────────
        if lane_instances:
            for inst in lane_instances:
                color = _maskrcnn_class_color(inst.class_name)
                overlay = vis.copy()
                overlay[inst.mask_bool] = color
                cv2.addWeighted(overlay, 0.52, vis, 0.48, 0, vis)
                if inst.contour_img:
                    contour = np.asarray(inst.contour_img, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [contour], True, color, 2, cv2.LINE_AA)

                x1, y1, x2, y2 = inst.bbox
                label = f"{inst.class_name} {inst.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
                tx = int(np.clip(x1, 0, max(W - tw - 6, 0)))
                ty = int(np.clip(y1 - 6, th + 4, max(H - 2, th + 4)))
                bg = vis.copy()
                cv2.rectangle(bg, (tx - 2, ty - th - 4), (tx + tw + 4, ty + 2), color, cv2.FILLED)
                cv2.addWeighted(bg, 0.55, vis, 0.45, 0, vis)
                cv2.putText(
                    vis,
                    label,
                    (tx, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )
        if not has_maskrcnn_lane_instances:
            lane_overlay = vis.copy()
            solid_mask = lane_pred == 1
            dashed_mask = lane_pred == 2
            if solid_mask.any():
                lane_overlay[solid_mask] = LaneSegmentationPipeline.PALETTE[1]
            if dashed_mask.any():
                lane_overlay[dashed_mask] = LaneSegmentationPipeline.PALETTE[2]
            if solid_mask.any() or dashed_mask.any():
                cv2.addWeighted(lane_overlay, alpha_lane, vis, 1.0 - alpha_lane, 0, vis)

        # ── 3. Road contour outlines ──────────────────────────────────────────
        for poly in road_result.contours_img:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], True,
                          (0, 255, 80), 2, cv2.LINE_AA)

        # ── 4. Fitted lane polylines + labels ─────────────────────────────────
        if curves:
            for lane in curves:
                if lane.color == "yellow":
                    color = (0, 215, 255) if lane.lane_type == "solid" else (0, 245, 255)
                elif lane.color == "white":
                    color = (245, 245, 245) if lane.lane_type == "solid" else (220, 220, 220)
                else:
                    color = LaneSegmentationPipeline.PALETTE[
                        1 if lane.lane_type == "solid" else 2
                    ]
                pts = np.array(lane.curve_points_img, dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(
                        vis, [pts.reshape(-1, 1, 2)],
                        False, color, 4, cv2.LINE_AA
                    )
                if len(pts) > 0:
                    mid = pts[len(pts) // 2]
                    label = f"[{lane.id}] {lane.color} {lane.lane_type}"
                    if lane.line_class and lane.line_class not in {lane.lane_type, _default_line_class_for_lane_type(lane.lane_type)}:
                        label += f" {lane.line_class}"
                    cv2.putText(
                        vis, label,
                        (int(mid[0]), max(int(mid[1]) - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA,
                    )

        # ── 5. Road markings on the drivable area ────────────────────────────
        if not has_maskrcnn_marking_instances:
            for marking in road_result.markings:
                contour = np.array(marking.contour_img, dtype=np.int32)
                if contour.shape[0] >= 3:
                    cv2.polylines(
                        vis,
                        [contour.reshape(-1, 1, 2)],
                        True,
                        (255, 120, 0) if marking.marking_type == "arrow" else (180, 90, 255),
                        2,
                        cv2.LINE_AA,
                    )
                x1, y1, x2, y2 = marking.bbox
                label = f"{marking.marking_type}:{marking.color}"
                if marking.direction:
                    label += f":{marking.direction}"
                cv2.putText(
                    vis,
                    label,
                    (int(x1), max(int(y1) - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 200, 80) if marking.marking_type == "arrow" else (220, 160, 255),
                    1,
                    cv2.LINE_AA,
                )

        # ── 6. HUD ────────────────────────────────────────────────────────────
        n_solid = sum(1 for c in curves if c.lane_type == "solid")
        n_dashed = sum(1 for c in curves if c.lane_type == "dashed")
        n_white = sum(1 for c in curves if c.color == "white")
        n_yellow = sum(1 for c in curves if c.color == "yellow")
        n_arrows = sum(1 for m in road_result.markings if m.marking_type == "arrow")
        maskrcnn_lane_count = sum(1 for inst in lane_instances if inst.class_name != "road-sign-line")
        maskrcnn_mark_count = sum(1 for inst in lane_instances if inst.class_name == "road-sign-line")
        hud_lines = [
            f"Lanes  : {len(curves)}  (solid={n_solid}  dashed={n_dashed})",
            f"MaskRCNN: lane={maskrcnn_lane_count}  marking={maskrcnn_mark_count}",
            f"Colors : white={n_white}  yellow={n_yellow}",
            f"Road   : {road_result.area_frac * 100:.1f}%  "
            f"({len(road_result.contours_img)} region(s))",
            f"Marks  : {len(road_result.markings)}  (arrows={n_arrows})",
            f"Proc   : {road_result.proc_ms:.0f} ms",
        ]
        x0, y0, lh = 8, 22, 20
        panel_h = lh * len(hud_lines) + 8
        overlay2 = vis.copy()
        cv2.rectangle(overlay2, (x0 - 4, y0 - 18),
                      (x0 + 360, y0 + panel_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.55, vis, 0.45, 0, vis)
        for i, line in enumerate(hud_lines):
            cv2.putText(
                vis, line, (x0, y0 + i * lh),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (220, 230, 255), 1, cv2.LINE_AA,
            )

        return vis

    # ── Video processing ──────────────────────────────────────────────────────

    def run_video(
        self,
        video_path: str,
        out_video: str,
        out_json: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        vscode_compatible: bool = True,
        debug_classes: bool = False,
        alpha_road: float = 0.30,
        alpha_lane: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Process an entire video file through both segmentation pipelines.

        Parameters
        ----------
        video_path : input video (any OpenCV-readable format)
        out_video : annotated output MP4 path
        out_json : per-frame JSON output path
        frame_skip : process every Nth source frame (1 = every frame)
        max_frames : stop after N processed frames  (None = all)
        vscode_compatible : re-encode to H.264 via ffmpeg for VS Code preview
        debug_classes : print raw class ids on frame 0

        Returns
        -------
        List of per-frame record dicts (same as written to out_json)
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {src}")

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = src_fps / max(frame_skip, 1)

        print(f"\n{'='*72}")
        print(f"  CombinedPipeline.run_video()")
        print(f"  Input  : {src}  ({src_W}×{src_H} @ {src_fps:.1f} fps, "
              f"~{total_frames} frames)")
        print(f"  Skip   : every {frame_skip} frame(s)  →  ~{out_fps:.1f} fps out")
        print(f"  Video  : {out_video}")
        print(f"  JSON   : {out_json}")
        print(f"{'='*72}\n")

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        writer = SafeVideoWriter(
            Path(out_video), out_fps, src_W, src_H,
            vscode_compatible=vscode_compatible,
        )

        records: List[Dict[str, Any]] = []
        fps_window: List[float] = []
        src_idx = 0
        written = 0
        t_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                t0 = time.time()
                is_first = written == 0

                lane_pred, curves, road_result, vis = self.infer_frame(
                    frame, debug_classes=debug_classes and is_first
                )

                record: Dict[str, Any] = {
                    "frame_idx":   written,
                    "timestamp_s": round(src_idx / src_fps, 4),
                    # ── Lane output ───────────────────────────────────────────
                    "lane_mask_classes": {
                        "background": 0,
                        "solid": 1,
                        "dashed": 2,
                    },
                    "lanes": [c.to_dict() for c in curves],
                    "lane_instances": [inst.to_dict() for inst in self._last_lane_instances],
                    # ── Road output ───────────────────────────────────────────
                    "road": road_result.to_dict(),
                }
                records.append(record)
                writer.write(vis)

                elapsed_frame = time.time() - t0
                fps_window.append(elapsed_frame)
                if len(fps_window) > 30:
                    fps_window.pop(0)
                proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

                written += 1
                src_idx += 1

                if written % 20 == 0 or written == 1:
                    pct = src_idx / total_frames * 100.0 if total_frames > 0 else 0.0
                    print(
                        f"  frame {src_idx:5d}/~{total_frames} ({pct:5.1f}%)  "
                        f"{proc_fps:5.1f} fps  "
                        f"lanes={len(curves):2d}  "
                        f"road={road_result.area_frac*100:.0f}%",
                        end="\r", flush=True,
                    )

                if max_frames is not None and written >= max_frames:
                    print(f"\nReached max_frames={max_frames} — stopping.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            cap.release()
            final_video = writer.close()

        # ── Write JSON ────────────────────────────────────────────────────────
        with open(out_json, "w") as f:
            json.dump(
                {
                    "source": str(src),
                    "frames_written": written,
                    "final_video": str(final_video),
                    "models": {
                        "lane_primary": (
                            f"Mask R-CNN [{self.maskrcnn_backend}] ({Path(self.maskrcnn_weights_path).name})"
                            if self.maskrcnn_refiner is not None and self.maskrcnn_weights_path
                            else "SegNet fallback (P2)"
                        ),
                        "lane_fallback": "SegNet (P2 — solid/dashed, 3-class)",
                        "road_fallback": "UNet  (P1 — freespace, 2-class)",
                        "lane_refiner": (
                            f"Mask R-CNN [{self.maskrcnn_backend}] ({Path(self.maskrcnn_weights_path).name})"
                            if self.maskrcnn_refiner is not None and self.maskrcnn_weights_path
                            else None
                        ),
                    },
                    "frames": records,
                },
                f, indent=2,
            )

        elapsed = time.time() - t_start
        print(f"\n{'='*72}")
        print("  Done.")
        print(f"  Frames processed : {written}")
        print(f"  Wall time        : {elapsed:.1f}s  "
              f"({written / max(elapsed, 1e-6):.1f} fps avg)")
        print(f"  Output video     : {final_video}")
        print(f"  JSON output      : {out_json}")
        print(f"{'='*72}")

        return records


# =============================================================================
# 12.  Module-level singleton convenience wrappers
#      (for direct use in scene_assembler.py)
# =============================================================================

_combined_singleton: Optional[CombinedPipeline] = None


def init_combined_pipeline(
    lane_checkpoint: str,
    road_checkpoint: str,
    device: str = "auto",
    input_width: int = 224,
    input_height: int = 224,
    maskrcnn_backend: str = "torchvision",
    maskrcnn_weights: Optional[str] = "auto",
    maskrcnn_score_threshold: float = 0.65,
    maskrcnn_mask_threshold: float = 0.45,
    maskrcnn_max_dets: int = 64,
) -> CombinedPipeline:
    """
    Initialise (or replace) the module-level singleton.
    Call once at startup from scene_assembler.py:

        from lane_road_segmentation import init_combined_pipeline, segment_frame
        init_combined_pipeline("models/lane.pt", "models/road.pt")
        lane_pred, curves, road, vis = segment_frame(frame_bgr)
    """
    global _combined_singleton
    _combined_singleton = CombinedPipeline(
        lane_checkpoint = lane_checkpoint,
        road_checkpoint = road_checkpoint,
        device = device,
        input_width = input_width,
        input_height = input_height,
        maskrcnn_backend = maskrcnn_backend,
        maskrcnn_weights = maskrcnn_weights,
        maskrcnn_score_threshold = maskrcnn_score_threshold,
        maskrcnn_mask_threshold = maskrcnn_mask_threshold,
        maskrcnn_max_dets = maskrcnn_max_dets,
    )
    return _combined_singleton


def segment_frame(
    frame_bgr: np.ndarray,
) -> Tuple[np.ndarray, List[LaneCurve], RoadResult, np.ndarray]:
    """
    Run combined segmentation using the module-level singleton.
    init_combined_pipeline() must be called first.

    Returns
    -------
    lane_pred : (H, W) uint8
    curves : List[LaneCurve]
    road : RoadResult
    vis : annotated BGR frame
    """
    if _combined_singleton is None:
        raise RuntimeError(
            "Call init_combined_pipeline(lane_checkpoint, road_checkpoint) first."
        )
    return _combined_singleton.infer_frame(frame_bgr)


# =============================================================================
# 13.  CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined lane + road segmentation (primary Mask R-CNN + fallback SegNet/UNet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=(
            "/Users/pippo/Desktop/Computer_Vision/Group10_p3/P3Data/Sequences"
            "/scene1/Undist/2023-02-14_11-04-07-front_undistort.mp4"
        ),
        help="Input video path",
    )
    parser.add_argument(
        "--lane-ckpt",
        type=str,
        default=str(P2_LANE_CKPT),
        help="Path to SegNet (P2) lane checkpoint",
    )
    parser.add_argument(
        "--road-ckpt",
        type=str,
        default=str(P1_ROAD_CKPT),
        help="Path to UNet (P1) road checkpoint",
    )
    parser.add_argument("--scene",        type=str,   default=None,
                        help="Optional explicit scene id (e.g. scene1); otherwise inferred from the input video path")
    parser.add_argument("--out-video",    type=str,   default=None,
                        help="Annotated combined output video path; defaults to output/<scene>/lanes/combined.mp4")
    parser.add_argument("--out-json",     type=str,   default=None,
                        help="Combined lane+road JSON path; defaults to output/<scene>/lanes/combined.json")
    parser.add_argument("--device",       type=str,   default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--frame-skip",   type=int,   default=1)
    parser.add_argument("--max-frames",   type=int,   default=None)
    parser.add_argument("--input-width",  type=int,   default=224)
    parser.add_argument("--input-height", type=int,   default=224)
    parser.add_argument("--road-classes", type=int,   default=2,
                        help="Number of output classes for road model (2 or 3)")
    parser.add_argument("--alpha-road",   type=float, default=0.30,
                        help="Road overlay blend weight [0–1]")
    parser.add_argument("--alpha-lane",   type=float, default=0.45,
                        help="Lane overlay blend weight [0–1]")
    parser.add_argument(
        "--maskrcnn-backend",
        type=str,
        default="torchvision",
        choices=["omonsun", "torchvision"],
        help="Optional lane-instance backend. 'torchvision' uses the DebuggerCafe/torchvision Mask R-CNN lane pipeline; 'omonsun' expects Matterport/Keras .h5 weights.",
    )
    parser.add_argument(
        "--maskrcnn-weights",
        type=str,
        default="auto",
        help=(
            "Optional Mask R-CNN weights for lane-instance / road-mark refinement. "
            "Use 'auto' to prefer the local DebuggerCafe/torchvision .pth checkpoint and fall back to Omonsun-style .h5 logs. "
            "Use 'none' to disable Mask R-CNN refinement."
        ),
    )
    parser.add_argument(
        "--maskrcnn-score-threshold",
        type=float,
        default=0.65,
        help="Confidence threshold for optional Mask R-CNN refinement",
    )
    parser.add_argument(
        "--maskrcnn-mask-threshold",
        type=float,
        default=0.45,
        help="Mask binarization threshold for optional Mask R-CNN refinement",
    )
    parser.add_argument(
        "--maskrcnn-max-dets",
        type=int,
        default=64,
        help="Maximum number of Mask R-CNN detections per frame",
    )
    parser.add_argument("--debug-classes", action="store_true",
                        help="Print raw class ids for first frame")
    parser.add_argument("--no-vscode-compatible", action="store_true",
                        help="Skip ffmpeg re-encode (output stays as AVI)")

    args = parser.parse_args()

    print("\n" + "═" * 72)
    print("  RBE549 / CS549 P3 — Einstein Vision - Combined Lane + Road Segmentation")
    print("  Lane : Mask R-CNN  (DebuggerCafe / torchvision primary)")
    print("         SegNet      (P2 fallback)")
    print("  Road : UNet        (P1 drivable-area fallback)")
    print("═" * 72 + "\n")

    pipe = CombinedPipeline(
        lane_checkpoint = args.lane_ckpt,
        road_checkpoint = args.road_ckpt,
        device = args.device,
        input_width = args.input_width,
        input_height = args.input_height,
        road_n_classes = args.road_classes,
        maskrcnn_backend = args.maskrcnn_backend,
        maskrcnn_weights = args.maskrcnn_weights,
        maskrcnn_score_threshold = args.maskrcnn_score_threshold,
        maskrcnn_mask_threshold = args.maskrcnn_mask_threshold,
        maskrcnn_max_dets = args.maskrcnn_max_dets,
    )

    scene_name = infer_scene_name(args.scene, args.video, args.out_video, args.out_json)
    output_layout = scene_output_layout(scene_name, create=True)
    out_video = str(Path(args.out_video).resolve()) if args.out_video else str((output_layout.lanes / "combined.mp4").resolve())
    out_json = str(Path(args.out_json).resolve()) if args.out_json else str((output_layout.lanes / "combined.json").resolve())

    pipe.run_video(
        video_path = args.video,
        out_video = out_video,
        out_json = out_json,
        frame_skip = args.frame_skip,
        max_frames = args.max_frames,
        vscode_compatible = not args.no_vscode_compatible,
        debug_classes = args.debug_classes,
        alpha_road = args.alpha_road,
        alpha_lane = args.alpha_lane,
    )

    if not args.out_video:
        mirror_stage_output(out_video, scene_name, "lanes", Path(out_video).name)
    if not args.out_json:
        mirror_stage_output(out_json, scene_name, "lanes", Path(out_json).name)
