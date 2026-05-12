"""
traffic_light_detector.py
=========================

Traffic-light detection and color-state estimation for the project pipeline.

Responsibilities
----------------
* detect traffic-light candidates in image space
* classify the active light color (red / yellow / green / unknown)
* maintain lightweight temporal stability for repeated observations
* emit JSON records that can be fused later by ``scene_assembler.py``

The module is designed to be robust on consumer hardware and therefore includes
device selection, fallback paths, and conservative color reasoning rather than
assuming access to a single heavyweight detector configuration.
"""

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
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Sequence
from collections import deque

import numpy as np

from project_setup import (
    infer_scene_name,
    mirror_stage_output,
    resolve_existing_artifact,
    scene_output_layout,
)


# =============================================================================
# 1. Device handling
# =============================================================================

def resolve_device(requested: str = "auto") -> str:
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


# =============================================================================
# 2. Result dataclass
# =============================================================================

@dataclasses.dataclass
class TrafficLightResult:
    id: int
    bbox: List[int]                  # [x1, y1, x2, y2]
    confidence: float
    color: str                       # red | yellow | green | unknown
    color_confidence: float
    bright_ratio: float
    centroid: List[int]
    signal_state: str = "unknown"
    signal_shape: str = "unknown"
    decision_source: str = "classical"
    detic_color_check: str = "unknown"
    detic_color_confidence: float = 0.0
    detic_color_agrees: Optional[bool] = None
    track_id: Optional[int] = None
    position_3d: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "color": self.color,
            "color_confidence": round(self.color_confidence, 4),
            "signal_color": self.color,
            "signal_shape": self.signal_shape,
            "signal_state": self.signal_state,
            "state_confidence": round(self.color_confidence, 4),
            "decision_source": self.decision_source,
            "detic_color_check": self.detic_color_check,
            "detic_color_confidence": round(self.detic_color_confidence, 4),
            "detic_color_agrees": self.detic_color_agrees,
            "bright_ratio": round(self.bright_ratio, 4),
            "centroid": self.centroid,
            "track_id": self.track_id,
            "position_3d": [round(v, 3) for v in self.position_3d]
                           if self.position_3d is not None else None,
        }


# =============================================================================
# 3. Generic helpers
# =============================================================================

def clamp_bbox(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    return x1, y1, x2, y2


def centroid_of(b: List[int]) -> Tuple[int, int]:
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def clip_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    return clamp_bbox(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), W, H)


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    W: int,
    H: int,
    scale: float = 1.05,
) -> Tuple[int, int, int, int]:
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(4.0, (x2 - x1) * scale)
    bh = max(4.0, (y2 - y1) * scale)
    return clip_box(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0, W, H)


def safe_crop(img: np.ndarray, bbox: Sequence[int]) -> Optional[np.ndarray]:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = clip_box(*bbox[:4], W, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms_boxes(boxes: List[List[int]], scores: List[float], iou_thresh: float = 0.30) -> List[int]:
    if not boxes:
        return []

    order = np.argsort(scores)[::-1].tolist()
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        remain = []
        for j in order:
            if bbox_iou(boxes[i], boxes[j]) < iou_thresh:
                remain.append(j)
        order = remain

    return keep


def merge_boxes_xyxy(boxes: List[List[int]], gap_px: int = 4) -> List[List[int]]:
    if not boxes:
        return []

    work = [list(map(int, bbox[:4])) for bbox in boxes]
    merged = True
    while merged:
        merged = False
        out: List[List[int]] = []
        used = [False] * len(work)

        for i in range(len(work)):
            if used[i]:
                continue
            x1, y1, x2, y2 = work[i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(work)):
                    if used[j]:
                        continue
                    a1, b1, a2, b2 = work[j]
                    overlaps_or_close = not (
                        a1 > x2 + gap_px
                        or a2 < x1 - gap_px
                        or b1 > y2 + gap_px
                        or b2 < y1 - gap_px
                    )
                    if overlaps_or_close:
                        x1 = min(x1, a1)
                        y1 = min(y1, b1)
                        x2 = max(x2, a2)
                        y2 = max(y2, b2)
                        used[j] = True
                        changed = True
                        merged = True
            out.append([x1, y1, x2, y2])
        work = out

    return work


def state_to_bgr(signal_state: str) -> Tuple[int, int, int]:
    if str(signal_state).startswith("red_"):
        return (0, 0, 255)
    if str(signal_state).startswith("yellow_"):
        return (0, 255, 255)
    if str(signal_state).startswith("green_"):
        return (0, 255, 0)
    return (255, 255, 255)


ALIASES = {
    "go": "green_circle",
    "goforward": "green_straight_arrow",
    "goleft": "green_left_arrow",
    "goright": "green_right_arrow",
    "right": "green_right_arrow",
    "left": "green_left_arrow",
    "stop": "red_circle",
    "stopleft": "red_left_arrow",
    "stopright": "red_right_arrow",
    "warning": "yellow_circle",
    "warningleft": "yellow_left_arrow",
    "warningright": "yellow_right_arrow",
    "go_forward": "green_straight_arrow",
    "go_left": "green_left_arrow",
    "go_right": "green_right_arrow",
    "stop_left": "red_left_arrow",
    "stop_right": "red_right_arrow",
    "warning_left": "yellow_left_arrow",
    "warning_right": "yellow_right_arrow",
    "green": "green_circle",
    "yellow": "yellow_circle",
    "red": "red_circle",
    "green_left": "green_left_arrow",
    "yellow_left": "yellow_left_arrow",
    "red_left": "red_left_arrow",
    "green_right": "green_right_arrow",
    "yellow_right": "yellow_right_arrow",
    "red_right": "red_right_arrow",
    "green_straight": "green_straight_arrow",
    "yellow_straight": "yellow_straight_arrow",
    "red_straight": "red_straight_arrow",
    "back_off_unknown": "back_off_unknown",
    "unknown": "unknown",
}


def normalize_class_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, key)


def parse_state(class_name: str) -> Dict[str, str]:
    norm = normalize_class_name(class_name)
    if norm in {"unknown", "back_off_unknown", ""}:
        return {
            "class_name": "unknown",
            "signal_color": "unknown",
            "signal_shape": "unknown",
            "signal_state": "unknown",
        }
    parts = norm.split("_")
    if len(parts) < 2:
        return {
            "class_name": norm,
            "signal_color": "unknown",
            "signal_shape": "unknown",
            "signal_state": norm,
        }
    return {
        "class_name": norm,
        "signal_color": parts[0],
        "signal_shape": "_".join(parts[1:]),
        "signal_state": norm,
    }


def enhance_tl_crop(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    if h < 4 or w < 4:
        return crop_bgr

    target_h = 96
    scale = max(1.0, target_h / float(h))
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    up = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    ycrcb = cv2.cvtColor(up, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    y = clahe.apply(y)
    ycrcb = cv2.merge([y, cr, cb])
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


# =============================================================================
# 4. VS Code friendly video output helpers
# =============================================================================

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
    """
    Writes temporary AVI first, then converts to H.264 MP4 if requested and ffmpeg exists.
    """

    def __init__(self, requested_output: Path, fps: float, width: int, height: int, vscode_compatible: bool = True):
        self.requested_output = requested_output
        self.vscode_compatible = vscode_compatible
        requested_output.parent.mkdir(parents=True, exist_ok=True)

        self.temp_video = requested_output.with_suffix(".tmp.avi")
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
            fallback.unlink()
        self.temp_video.rename(fallback)
        print(f"[SafeVideoWriter] Using AVI fallback -> {fallback}")
        return fallback


# =============================================================================
# 5. Optional Detic color checker
# =============================================================================

class DeticTrafficLightColorChecker:
    """
    Use Detic only as a secondary color cue on already cropped traffic-light heads.
    """

    VOCABULARY = (
        "red traffic light",
        "yellow traffic light",
        "green traffic light",
        "traffic light",
    )

    def __init__(
        self,
        repo_root: str,
        config_file: str,
        weights_path: str,
        *,
        python_exe: str = "auto",
        min_confidence: float = 0.18,
    ) -> None:
        from detic_scene_detector import DeticSceneDetector

        self.detector = DeticSceneDetector(
            repo_root=repo_root,
            config_file=config_file,
            weights_path=weights_path,
            python_exe=python_exe,
            device="cpu",
            min_confidence=min_confidence,
            vocabulary=self.VOCABULARY,
            pred_all_classes=False,
        )
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _label_to_color(raw_label: str) -> str:
        label = str(raw_label or "").strip().lower()
        if "red" in label:
            return "red"
        if "yellow" in label or "amber" in label:
            return "yellow"
        if "green" in label:
            return "green"
        return "unknown"

    def classify(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        variants = [crop_bgr, enhance_tl_crop(crop_bgr)]
        h, w = crop_bgr.shape[:2]
        if h >= 4 and w >= 4:
            pad = max(2, int(round(0.10 * max(h, w))))
            padded = cv2.copyMakeBorder(
                crop_bgr,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_REPLICATE,
            )
            variants.append(enhance_tl_crop(padded))

        best_color = "unknown"
        best_conf = 0.0
        for variant in variants:
            try:
                detections = self.detector.detect(variant)
            except Exception:
                continue
            for det in detections:
                color = self._label_to_color(det.get("raw_label"))
                conf = float(det.get("confidence", 0.0) or 0.0)
                if color == "unknown":
                    continue
                if conf > best_conf:
                    best_color = color
                    best_conf = conf

        return best_color, best_conf


# =============================================================================
# 6. Improved color classifier
# =============================================================================

class RobustTrafficLightColorClassifier:
    """
    Robust color classifier for mostly vertical US traffic lights.

    Key ideas:
    - crop center columns to suppress poles/background
    - upscale small ROIs
    - build bright-lamp mask
    - build red/yellow/green masks
    - use connected components
    - use vertical zone prior:
        red    top
        yellow middle
        green  bottom
    - if hue is weak, fallback to brightest-blob vertical zone
    """

    def __init__(self, min_side: int = 56):
        self.min_side = min_side

    @staticmethod
    def _upscale_if_small(img: np.ndarray, min_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        if min(h, w) >= min_side:
            return img
        s = float(min_side) / max(1, min(h, w))
        return cv2.resize(
            img,
            (int(round(w * s)), int(round(h * s))),
            interpolation=cv2.INTER_CUBIC
        )

    @staticmethod
    def _largest_component(mask: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        if mask is None or mask.size == 0 or np.count_nonzero(mask) == 0:
            return 0.0, (0.0, 0.0)

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return 0.0, (0.0, 0.0)

        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        area = float(stats[idx, cv2.CC_STAT_AREA])
        total = float(mask.shape[0] * mask.shape[1])
        cx, cy = cents[idx]
        return area / max(total, 1.0), (cx, cy)

    @staticmethod
    def _zone_score(y_norm: float, target: float, sigma: float = 0.18) -> float:
        return math.exp(-0.5 * ((y_norm - target) / sigma) ** 2)

    def classify(self, roi_bgr: np.ndarray) -> Tuple[str, float, float]:
        if roi_bgr is None or roi_bgr.size == 0:
            return "unknown", 0.0, 0.0

        h0, w0 = roi_bgr.shape[:2]
        if h0 < 4 or w0 < 4:
            return "unknown", 0.0, 0.0

        # Stronger center crop to suppress housing/poles/background
        x1 = int(0.28 * w0)
        x2 = int(0.72 * w0)
        y1 = int(0.05 * h0)
        y2 = int(0.95 * h0)
        roi = roi_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            return "unknown", 0.0, 0.0

        roi = self._upscale_if_small(roi, self.min_side)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # more forgiving thresholds
        v_thr = int(max(145, np.percentile(V, 75)))
        s_thr = int(max(25, np.percentile(S, 30)))

        # generic bright lamp evidence
        bright = ((V >= v_thr) & (S >= s_thr)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_DILATE, kernel)

        bright_ratio = float(np.count_nonzero(bright)) / max(bright.shape[0] * bright.shape[1], 1)
        if bright_ratio < 0.002:
            return "unknown", 0.0, bright_ratio

        # hue masks
        red1 = cv2.inRange(hsv, (0, s_thr, v_thr), (12, 255, 255))
        red2 = cv2.inRange(hsv, (165, s_thr, v_thr), (179, 255, 255))
        red = cv2.bitwise_or(red1, red2)

        yellow = cv2.inRange(hsv, (12, s_thr, v_thr), (42, 255, 255))

        green = cv2.inRange(hsv, (35, s_thr, v_thr), (100, 255, 255))

        # allow pale cyan-ish greens
        cyan_green = cv2.inRange(hsv, (85, max(15, s_thr // 2), v_thr), (110, 255, 255))
        green = cv2.bitwise_or(green, cyan_green)

        # only keep pixels that are also bright
        red = cv2.bitwise_and(red, bright)
        yellow = cv2.bitwise_and(yellow, bright)
        green = cv2.bitwise_and(green, bright)

        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)

        rr, (_, rcy) = self._largest_component(red)
        yr, (_, ycy) = self._largest_component(yellow)
        gr, (_, gcy) = self._largest_component(green)
        br, (_, bcy) = self._largest_component(bright)

        h, w = bright.shape[:2]

        red_score = rr * (0.7 + 0.3 * self._zone_score(rcy / max(h, 1), 0.18))
        yellow_score = yr * (0.7 + 0.3 * self._zone_score(ycy / max(h, 1), 0.50))
        green_score = gr * (0.7 + 0.3 * self._zone_score(gcy / max(h, 1), 0.82))

        scores = {
            "red": red_score,
            "yellow": yellow_score,
            "green": green_score,
        }

        best_color = max(scores, key=scores.get)
        best = scores[best_color]
        second = sorted(scores.values(), reverse=True)[1]

        # fallback: hue weak, but one bright blob has strong vertical cue
        if best < 0.01 and br > 0.003:
            y_norm = bcy / max(h, 1)
            zone_scores = {
                "red": br * self._zone_score(y_norm, 0.18),
                "yellow": br * self._zone_score(y_norm, 0.50),
                "green": br * self._zone_score(y_norm, 0.82),
            }
            best_color = max(zone_scores, key=zone_scores.get)
            best = zone_scores[best_color]
            second = sorted(zone_scores.values(), reverse=True)[1]

        conf = float(np.clip(0.75 * best + 0.25 * max(0.0, best - second) * 2.0, 0.0, 1.0))

        if bright_ratio < 0.002 or best < 0.006 or conf < 0.14:
            return "unknown", 0.0, bright_ratio

        return best_color, conf, bright_ratio


# =============================================================================
# 6. Hybrid state classification + temporal smoothing
# =============================================================================

class ClassicalTrafficLightAnalyzer:
    def __init__(self) -> None:
        self.kernel3 = np.ones((3, 3), np.uint8)

    def _bright_sat_mask(self, hsv: np.ndarray) -> np.ndarray:
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        return ((s > 75) & (v > 120)).astype(np.uint8) * 255

    def _color_masks(self, crop_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        bs = self._bright_sat_mask(hsv)

        red1 = cv2.inRange(hsv, (0, 95, 120), (12, 255, 255))
        red2 = cv2.inRange(hsv, (165, 95, 120), (179, 255, 255))
        red = cv2.bitwise_or(red1, red2)
        red = cv2.bitwise_and(red, bs)

        yellow = cv2.inRange(hsv, (15, 95, 130), (40, 255, 255))
        yellow = cv2.bitwise_and(yellow, bs)

        green = cv2.inRange(hsv, (40, 65, 95), (95, 255, 255))
        green = cv2.bitwise_and(green, bs)

        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, self.kernel3)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, self.kernel3)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, self.kernel3)

        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, self.kernel3)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, self.kernel3)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, self.kernel3)

        return {"red": red, "yellow": yellow, "green": green}

    def _shape_from_contour(self, contour: np.ndarray) -> Tuple[str, float]:
        area = cv2.contourArea(contour)
        perim = cv2.arcLength(contour, True)
        if perim <= 1e-6:
            return "unknown", 0.0

        x, y, w, h = cv2.boundingRect(contour)
        if w < 3 or h < 3:
            return "unknown", 0.0

        rect_area = max(1.0, float(w * h))
        circularity = 4.0 * np.pi * area / (perim * perim + 1e-6)
        extent = area / rect_area
        aspect = w / float(h + 1e-6)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return "unknown", 0.0
        cx = M["m10"] / M["m00"]
        box_cx = x + w / 2.0
        box_cy = y + h / 2.0

        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

        left_protrusion = box_cx - leftmost[0]
        right_protrusion = rightmost[0] - box_cx
        top_protrusion = box_cy - topmost[1]
        bottom_protrusion = bottommost[1] - box_cy

        if circularity > 0.52 and 0.62 <= aspect <= 1.40 and extent > 0.40:
            return "circle", float(min(1.0, circularity))

        if h > 1.20 * w and top_protrusion > bottom_protrusion:
            if abs(cx - box_cx) < 0.11 * w:
                return "straight_arrow", 0.72

        if left_protrusion > right_protrusion * 1.18 and cx < box_cx - 0.06 * w:
            return "left_arrow", 0.72
        if right_protrusion > left_protrusion * 1.18 and cx > box_cx + 0.06 * w:
            return "right_arrow", 0.72

        if circularity > 0.42 and extent > 0.32:
            return "circle", 0.45

        return "unknown", 0.0

    def find_heads(self, crop_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 6 or h * w < 80:
            return []

        color_masks = self._color_masks(crop_bgr)
        raw_boxes: List[List[int]] = []

        for mask in color_masks.values():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 6:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw < 2 or bh < 2:
                    continue
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w, x + bw + pad)
                y2 = min(h, y + bh + pad)
                if (x2 - x1) < 5 or (y2 - y1) < 6:
                    continue
                raw_boxes.append([x1, y1, x2, y2])

        if not raw_boxes:
            return []

        heads: List[Dict[str, Any]] = []
        for bbox in merge_boxes_xyxy(raw_boxes, gap_px=6):
            x1, y1, x2, y2 = bbox
            head_crop = crop_bgr[y1:y2, x1:x2].copy()
            if head_crop.size == 0:
                continue

            dom_color = "unknown"
            best_area = 0.0
            for color_name, mask in color_masks.items():
                local = mask[y1:y2, x1:x2]
                area = float(np.count_nonzero(local))
                if area > best_area:
                    best_area = area
                    dom_color = color_name

            heads.append(
                {
                    "bbox_local": [x1, y1, x2, y2],
                    "crop": head_crop,
                    "dominant_color": dom_color if best_area >= 4 else "unknown",
                }
            )

        return heads

    def analyze_head(
        self,
        head_crop_bgr: np.ndarray,
        color_hint: str,
    ) -> Tuple[bool, str, str, str, float]:
        if head_crop_bgr is None or head_crop_bgr.size == 0:
            return False, "unknown", "unknown", "unknown", 0.0

        masks = self._color_masks(head_crop_bgr)
        mask = masks.get(color_hint)
        if mask is None:
            return False, "unknown", "unknown", "unknown", 0.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "unknown", "unknown", "unknown", 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < 5:
            return False, "unknown", "unknown", "unknown", 0.0

        shape, shape_conf = self._shape_from_contour(contour)
        if shape == "unknown":
            return False, "unknown", "unknown", "unknown", 0.0

        h, w = head_crop_bgr.shape[:2]
        crop_area = max(1.0, float(h * w))
        lit_ratio = min(1.0, area / crop_area * 24.0)
        conf = 0.55 * shape_conf + 0.45 * lit_ratio
        if conf < 0.24:
            return False, "unknown", "unknown", "unknown", conf

        state = f"{color_hint}_{shape}"
        return True, color_hint, shape, state, float(conf)


class HybridTrafficLightStateClassifier:
    def __init__(
        self,
        classifier_model: Optional[Any],
        device: str = "cpu",
        classifier_imgsz: int = 320,
        classical_accept_conf: float = 0.50,
        classifier_accept_conf: float = 0.70,
        fallback_color_classifier: Optional[RobustTrafficLightColorClassifier] = None,
        detic_color_checker: Optional[DeticTrafficLightColorChecker] = None,
        detic_accept_conf: float = 0.28,
        detic_override_conf: float = 0.48,
    ) -> None:
        self.classifier = classifier_model
        self.device = device
        self.classifier_imgsz = classifier_imgsz
        self.classical_accept_conf = classical_accept_conf
        self.classifier_accept_conf = classifier_accept_conf
        self.classical = ClassicalTrafficLightAnalyzer()
        self.fallback_color_classifier = fallback_color_classifier or RobustTrafficLightColorClassifier()
        self.detic_color_checker = detic_color_checker
        self.detic_accept_conf = float(detic_accept_conf)
        self.detic_override_conf = float(max(detic_override_conf, detic_accept_conf))

    @staticmethod
    def bright_ratio(crop_bgr: np.ndarray) -> float:
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask = ((s > 60) & (v > 120)).astype(np.uint8)
        return float(np.count_nonzero(mask)) / max(mask.shape[0] * mask.shape[1], 1)

    def classify_crop(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        if self.classifier is None or crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        variants = [crop_bgr, enhance_tl_crop(crop_bgr)]
        h, w = crop_bgr.shape[:2]
        pad = max(2, int(round(0.12 * max(h, w))))
        padded = cv2.copyMakeBorder(
            crop_bgr, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE
        )
        variants.append(enhance_tl_crop(padded))

        best_name = "unknown"
        best_conf = 0.0
        for variant in variants:
            try:
                results = self.classifier.predict(
                    source=variant,
                    imgsz=self.classifier_imgsz,
                    device=self.device,
                    verbose=False,
                )
            except Exception:
                continue
            if not results:
                continue
            probs = results[0].probs
            if probs is None:
                continue

            top1 = int(probs.top1)
            top1conf = probs.top1conf
            conf = float(top1conf.item()) if hasattr(top1conf, "item") else float(top1conf)
            names = results[0].names
            raw_name = names[top1] if isinstance(names, dict) else names[top1]
            norm_name = normalize_class_name(raw_name)
            if conf > best_conf:
                best_conf = conf
                best_name = norm_name

        return best_name, best_conf

    @staticmethod
    def should_reject_classifier_state(state_name: str) -> bool:
        return state_name in {"unknown", "back_off_unknown"}

    def decide_head(self, head_crop: np.ndarray, color_hint: str) -> Optional[Dict[str, Any]]:
        if color_hint == "unknown":
            keep_cls, cls_color, cls_shape, cls_state, cls_conf = False, "unknown", "unknown", "unknown", 0.0
        else:
            keep_cls, cls_color, cls_shape, cls_state, cls_conf = self.classical.analyze_head(head_crop, color_hint)

        final_state = "unknown"
        final_color = "unknown"
        final_shape = "unknown"
        final_conf = 0.0
        decision_source = "rejected"
        detic_color = "unknown"
        detic_conf = 0.0

        if keep_cls and cls_conf >= self.classical_accept_conf:
            final_state = cls_state
            final_color = cls_color
            final_shape = cls_shape
            final_conf = cls_conf
            decision_source = "classical"
        else:
            pred_name, pred_conf = self.classify_crop(head_crop)

            if not self.should_reject_classifier_state(pred_name) and pred_conf >= self.classifier_accept_conf:
                parsed = parse_state(pred_name)
                final_state = parsed["signal_state"]
                final_color = parsed["signal_color"]
                final_shape = parsed["signal_shape"]
                final_conf = pred_conf
                decision_source = "classifier"
            elif keep_cls:
                final_state = cls_state
                final_color = cls_color
                final_shape = cls_shape
                final_conf = cls_conf
                decision_source = "classical_fallback"
            elif not self.should_reject_classifier_state(pred_name) and pred_conf >= max(0.50, self.classifier_accept_conf - 0.15):
                parsed = parse_state(pred_name)
                final_state = parsed["signal_state"]
                final_color = parsed["signal_color"]
                final_shape = parsed["signal_shape"]
                final_conf = pred_conf
                decision_source = "classifier_weak"
            else:
                fallback_color, fallback_conf, _ = self.fallback_color_classifier.classify(head_crop)
                if fallback_color != "unknown" and fallback_conf >= 0.18:
                    final_state = f"{fallback_color}_circle"
                    final_color = fallback_color
                    final_shape = "circle"
                    final_conf = fallback_conf
                    decision_source = "color_fallback"
                else:
                    return None

        if self.detic_color_checker is not None:
            try:
                detic_color, detic_conf = self.detic_color_checker.classify(head_crop)
            except Exception:
                detic_color, detic_conf = "unknown", 0.0

        detic_agrees: Optional[bool]
        if detic_color == "unknown":
            detic_agrees = None
        else:
            detic_agrees = detic_color == final_color

        if detic_color != "unknown":
            if final_color == "unknown":
                final_color = detic_color
                if final_shape == "unknown":
                    final_shape = "circle"
                final_state = f"{final_color}_{final_shape}"
                final_conf = max(final_conf, detic_conf)
                decision_source = "detic_color"
            elif detic_color == final_color and detic_conf >= self.detic_accept_conf:
                final_conf = max(final_conf, min(0.99, 0.5 * final_conf + 0.5 * detic_conf))
                decision_source = f"{decision_source}+detic"
            elif (
                detic_conf >= self.detic_override_conf
                and final_conf < max(self.classifier_accept_conf, 0.78)
            ):
                final_color = detic_color
                if final_shape == "unknown":
                    final_shape = "circle"
                final_state = f"{final_color}_{final_shape}"
                final_conf = max(final_conf, detic_conf)
                decision_source = f"{decision_source}+detic_override"

        return {
            "signal_state": final_state,
            "signal_color": final_color,
            "signal_shape": final_shape,
            "hybrid_score": float(final_conf),
            "decision_source": decision_source,
            "detic_color_check": detic_color,
            "detic_color_confidence": float(detic_conf),
            "detic_color_agrees": detic_agrees,
            "bright_ratio": self.bright_ratio(head_crop),
        }

    def classify_crop_heads(self, crop_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if crop_bgr is None or crop_bgr.size == 0:
            return []

        heads = self.classical.find_heads(crop_bgr)
        if not heads:
            heads = [
                {
                    "bbox_local": [0, 0, crop_bgr.shape[1], crop_bgr.shape[0]],
                    "crop": crop_bgr,
                    "dominant_color": "unknown",
                }
            ]

        results: List[Dict[str, Any]] = []
        for head in heads:
            head_result = self.decide_head(head["crop"], head["dominant_color"])
            if head_result is None:
                continue
            results.append(
                {
                    **head_result,
                    "bbox_local": [int(v) for v in head["bbox_local"]],
                }
            )
        return results


class SimpleTrafficLightTracker:
    def __init__(self, max_age: int = 16, dist_thresh: float = 60.0, iou_thresh: float = 0.05):
        self.max_age = max_age
        self.dist_thresh = dist_thresh
        self.iou_thresh = iou_thresh
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[TrafficLightResult]) -> List[TrafficLightResult]:
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["age"] += 1
            if self.tracks[tid]["age"] > self.max_age:
                del self.tracks[tid]

        used: set[int] = set()
        for det in detections:
            dc = tuple(det.centroid)
            best_tid = None
            best_cost = 1e9

            for tid, state in self.tracks.items():
                if tid in used:
                    continue
                iou = bbox_iou(det.bbox, state["bbox"])
                dist = math.hypot(dc[0] - state["centroid"][0], dc[1] - state["centroid"][1])
                if iou < self.iou_thresh and dist > self.dist_thresh:
                    continue
                cost = dist - 25.0 * iou
                if cost < best_cost:
                    best_cost = cost
                    best_tid = tid

            if best_tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": det.bbox,
                    "centroid": dc,
                    "age": 0,
                    "states": deque([det.signal_state], maxlen=6),
                    "scores": deque([det.color_confidence], maxlen=6),
                }
                det.track_id = tid
                used.add(tid)
                continue

            track = self.tracks[best_tid]
            track["bbox"] = det.bbox
            track["centroid"] = dc
            track["age"] = 0
            track["states"].append(det.signal_state)
            track["scores"].append(det.color_confidence)
            det.track_id = best_tid
            used.add(best_tid)

            votes: Dict[str, float] = {}
            for state_name, score in zip(track["states"], track["scores"]):
                votes[state_name] = votes.get(state_name, 0.0) + max(float(score), 0.05)

            best_state = max(votes.items(), key=lambda item: item[1])[0]
            parsed = parse_state(best_state)
            det.signal_state = parsed["signal_state"]
            det.signal_shape = parsed["signal_shape"]
            det.color = parsed["signal_color"]
            det.color_confidence = float(
                np.clip(votes.get(best_state, 0.0) / max(sum(votes.values()), 1e-6), 0.0, 1.0)
            )

        return detections


# =============================================================================
# 7. Detector
# =============================================================================

class TrafficLightDetector:
    TRAFFIC_LIGHT_CLASS_ID = 9

    def __init__(
        self,
        device: str = "auto",
        yolo_model: str = "auto",
        classifier_weights: str = "auto",
        detic_color_checker: str = "auto",
        detic_python: str = "auto",
        detic_repo: str = "auto",
        detic_config: str = "auto",
        detic_weights: str = "auto",
        detic_min_confidence: float = 0.18,
        det_conf_thresh: float = 0.22,
        detect_width: int = 960,
        upper_frac: float = 0.68,
        classifier_imgsz: int = 320,
        crop_scale: float = 1.35,
        classical_accept_conf: float = 0.50,
        classifier_accept_conf: float = 0.70,
        use_tracker: bool = True,
    ):
        self.device = resolve_device(device)
        self.det_conf_thresh = det_conf_thresh
        self.detect_width = detect_width
        self.upper_frac = upper_frac
        self.crop_scale = crop_scale
        self.color_classifier = RobustTrafficLightColorClassifier()
        self.tracker = SimpleTrafficLightTracker() if use_tracker else None

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required. Run: pip install ultralytics") from exc

        self.yolo_model_path = self._resolve_model_path(yolo_model)
        self.model = YOLO(self.yolo_model_path)
        self.classifier_weights = self._resolve_classifier_weights(classifier_weights)
        self.classifier_model = YOLO(self.classifier_weights) if self.classifier_weights else None
        self.detic_checker_mode = str(detic_color_checker or "auto").strip().lower()
        self.detic_color_checker = self._build_detic_color_checker(
            checker_mode=self.detic_checker_mode,
            python_exe=detic_python,
            repo_root=detic_repo,
            config_file=detic_config,
            weights_path=detic_weights,
            min_confidence=detic_min_confidence,
        )
        self.state_classifier = HybridTrafficLightStateClassifier(
            classifier_model=self.classifier_model,
            device=self.device,
            classifier_imgsz=classifier_imgsz,
            classical_accept_conf=classical_accept_conf,
            classifier_accept_conf=classifier_accept_conf,
            fallback_color_classifier=self.color_classifier,
            detic_color_checker=self.detic_color_checker,
        )

        print(f"[TrafficLightDetector] YOLO loaded on {self.device}")
        print(
            f"[TrafficLightDetector] model={self.yolo_model_path}, "
            f"classifier={self.classifier_weights or 'disabled'}, "
            f"detic_checker={'enabled' if self.detic_color_checker is not None else 'disabled'}, "
            f"detect_width={detect_width}, upper_frac={upper_frac}"
        )

    @staticmethod
    def _resolve_model_path(requested: str) -> str:
        if requested and requested != "auto":
            path = Path(requested).expanduser()
            return str(path.resolve()) if path.exists() else requested
        resolved = resolve_existing_artifact(("yolo11x.pt", "yolov8x.pt", "yolov8n.pt", "yolo26x.pt"))
        if resolved:
            return resolved
        return "yolo11x.pt"

    @staticmethod
    def _resolve_classifier_weights(requested: str) -> Optional[str]:
        if str(requested or "").strip().lower() == "none":
            return None
        if requested and requested != "auto":
            path = Path(requested).expanduser()
            return str(path.resolve()) if path.exists() else requested
        resolved = resolve_existing_artifact(("best_tl_classifier.pt",))
        return resolved

    @staticmethod
    def _resolve_detic_repo(requested: str) -> Optional[str]:
        if str(requested or "").strip().lower() == "none":
            return None
        if requested and requested != "auto":
            path = Path(requested).expanduser()
            return str(path.resolve()) if path.exists() else None
        return resolve_existing_artifact(("external/Detic",))

    @staticmethod
    def _resolve_detic_config(requested: str) -> Optional[str]:
        if str(requested or "").strip().lower() == "none":
            return None
        if requested and requested != "auto":
            path = Path(requested).expanduser()
            return str(path.resolve()) if path.exists() else None
        return resolve_existing_artifact(
            (
                "external/Detic/configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml",
                "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml",
            )
        )

    @staticmethod
    def _resolve_detic_weights(requested: str) -> Optional[str]:
        if str(requested or "").strip().lower() == "none":
            return None
        if requested and requested != "auto":
            path = Path(requested).expanduser()
            return str(path.resolve()) if path.exists() else None
        return resolve_existing_artifact(
            (
                "external/Detic/models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth",
                "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth",
            )
        )

    def _build_detic_color_checker(
        self,
        *,
        checker_mode: str,
        python_exe: str,
        repo_root: str,
        config_file: str,
        weights_path: str,
        min_confidence: float,
    ) -> Optional[DeticTrafficLightColorChecker]:
        if checker_mode == "off":
            return None

        repo = self._resolve_detic_repo(repo_root)
        config = self._resolve_detic_config(config_file)
        weights = self._resolve_detic_weights(weights_path)
        if not (repo and config and weights):
            if checker_mode == "on":
                print("[TrafficLightDetector] Detic checker requested but Detic assets were not resolved; continuing without it.")
            return None

        try:
            checker = DeticTrafficLightColorChecker(
                repo_root=repo,
                config_file=config,
                weights_path=weights,
                python_exe=python_exe,
                min_confidence=min_confidence,
            )
            print(f"[TrafficLightDetector] Detic color checker ready ({Path(weights).name})")
            return checker
        except Exception as exc:
            if checker_mode == "on":
                print(
                    "[TrafficLightDetector] Warning: could not initialise Detic color checker. "
                    f"Continuing without it. ({type(exc).__name__}: {exc}) "
                    "If Detic lives in a separate conda env, pass --detic-python /path/to/env/bin/python."
                )
            return None

    def _preprocess_for_detection(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        det_h = int(round(H * self.upper_frac))
        crop = frame_bgr[:det_h, :]

        if W == self.detect_width:
            return crop, 1.0, det_h

        scale = self.detect_width / float(W)
        new_h = int(round(det_h * scale))
        resized = cv2.resize(crop, (self.detect_width, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale, det_h

    def _raw_detect(self, det_img: np.ndarray):
        results = self.model(
            det_img,
            verbose=False,
            device=self.device,
            classes=[self.TRAFFIC_LIGHT_CLASS_ID],
            conf=self.det_conf_thresh,
            max_det=128,
        )

        out = []
        if len(results) == 0 or results[0].boxes is None:
            return out

        for box in results[0].boxes:
            cls_id = int(box.cls.item())
            if cls_id != self.TRAFFIC_LIGHT_CLASS_ID:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            out.append(([x1, y1, x2, y2], conf))
        return out

    def _geometry_filter(self, bbox: List[int], W: int, H: int) -> bool:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False

        area = bw * bh
        aspect = bw / max(bh, 1)

        if area > 0.018 * W * H:
            return False
        if aspect > 1.15:
            return False
        if y1 > 0.78 * H:
            return False
        if bh < 6 or bw < 4:
            return False

        return True

    def _lamp_presence_filter(self, roi_bgr: np.ndarray) -> bool:
        """
        Reject poles/signs by requiring at least some plausible bright lamp evidence.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False

        h, w = roi_bgr.shape[:2]
        if h < 4 or w < 4:
            return False

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        v_thr = int(max(145, np.percentile(V, 75)))
        s_thr = int(max(20, np.percentile(S, 25)))

        mask = ((S >= s_thr) & (V >= v_thr)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        ratio = float(np.count_nonzero(mask)) / max(h * w, 1)
        return ratio > 0.0015

    def detect(self, frame_bgr: np.ndarray) -> List[TrafficLightResult]:
        H, W = frame_bgr.shape[:2]
        det_img, scale, _ = self._preprocess_for_detection(frame_bgr)
        raw = self._raw_detect(det_img)

        boxes: List[List[int]] = []
        scores: List[float] = []

        for bbox, conf in raw:
            x1, y1, x2, y2 = bbox

            # map from detection image back to original frame
            x1 = int(round(x1 / scale))
            y1 = int(round(y1 / scale))
            x2 = int(round(x2 / scale))
            y2 = int(round(y2 / scale))

            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, W, H)
            bb = [x1, y1, x2, y2]

            if not self._geometry_filter(bb, W, H):
                continue

            roi = frame_bgr[y1:y2, x1:x2]
            if not self._lamp_presence_filter(roi):
                continue

            boxes.append(bb)
            scores.append(conf)

        keep = nms_boxes(boxes, scores, iou_thresh=0.28)

        detections: List[TrafficLightResult] = []
        for i in keep:
            bb = boxes[i]
            conf = scores[i]
            x1, y1, x2, y2 = bb

            crop_bbox = list(expand_box(x1, y1, x2, y2, W, H, scale=self.crop_scale))
            parent_crop = safe_crop(frame_bgr, crop_bbox)
            if parent_crop is None:
                continue

            crop_x1, crop_y1, _, _ = crop_bbox
            head_results = self.state_classifier.classify_crop_heads(parent_crop)
            if not head_results:
                head_results = [
                    {
                        "bbox_local": [0, 0, parent_crop.shape[1], parent_crop.shape[0]],
                        "signal_state": "unknown",
                        "signal_color": "unknown",
                        "signal_shape": "unknown",
                        "hybrid_score": 0.0,
                        "decision_source": "rejected",
                        "bright_ratio": 0.0,
                    }
                ]

            for head in head_results:
                lx1, ly1, lx2, ly2 = [int(v) for v in head["bbox_local"][:4]]
                gx1 = crop_x1 + lx1
                gy1 = crop_y1 + ly1
                gx2 = crop_x1 + lx2
                gy2 = crop_y1 + ly2
                head_bbox = list(clamp_bbox(gx1, gy1, gx2, gy2, W, H))
                cx, cy = centroid_of(head_bbox)
                detections.append(
                    TrafficLightResult(
                        id=len(detections),
                        bbox=head_bbox,
                        confidence=conf,
                        color=str(head.get("signal_color", "unknown")),
                        color_confidence=float(head.get("hybrid_score", 0.0)),
                        bright_ratio=float(head.get("bright_ratio", 0.0)),
                        centroid=[cx, cy],
                        signal_state=str(head.get("signal_state", "unknown")),
                        signal_shape=str(head.get("signal_shape", "unknown")),
                        decision_source=str(head.get("decision_source", "rejected")),
                        detic_color_check=str(head.get("detic_color_check", "unknown")),
                        detic_color_confidence=float(head.get("detic_color_confidence", 0.0)),
                        detic_color_agrees=head.get("detic_color_agrees"),
                    )
                )

        detections.sort(key=lambda d: (d.color_confidence, d.confidence), reverse=True)
        for i, d in enumerate(detections):
            d.id = i

        if self.tracker is not None:
            detections = self.tracker.update(detections)

        return detections

    def draw(self, frame_bgr: np.ndarray, detections: List[TrafficLightResult]) -> np.ndarray:
        vis = frame_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            c = state_to_bgr(det.signal_state)

            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.circle(vis, tuple(det.centroid), 3, c, -1)

            label = (
                f"[{det.id}] {det.signal_state} "
                f"d={det.confidence:.2f} s={det.color_confidence:.2f}"
            )
            if det.track_id is not None:
                label += f" T{det.track_id}"
            if det.decision_source:
                label += f" {det.decision_source}"
            if det.detic_color_check != "unknown":
                label += f" detic:{det.detic_color_check}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.50
            thick = 1
            (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
            ly = max(y1 - 4, th + bl + 2)

            cv2.rectangle(vis, (x1, ly - th - bl - 2), (x1 + tw + 4, ly + 2), c, cv2.FILLED)
            cv2.putText(vis, label, (x1 + 2, ly), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        return vis


# =============================================================================
# 8. JSON record builder
# =============================================================================

def build_frame_record(
    frame_idx: int,
    timestamp_s: float,
    detections: List[TrafficLightResult],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "frame_idx": frame_idx,
        "timestamp_s": round(timestamp_s, 4),
        "traffic_lights": [d.to_dict() for d in detections],
        "metadata": metadata or {},
    }


# =============================================================================
# 9. Pipeline
# =============================================================================

class TrafficLightPipeline:
    def __init__(self, detector: Optional[TrafficLightDetector] = None, device: str = "auto"):
        self.detector = detector or TrafficLightDetector(device=device)

    @staticmethod
    def draw_hud(frame: np.ndarray, frame_idx: int, proc_fps: float, n: int, total: int) -> None:
        lines = [
            f"Frame : {frame_idx}",
            f"Proc  : {proc_fps:5.1f} fps",
            f"This  : {n} traffic light(s)",
            f"Total : {total}",
        ]

        x0, y0, lh = 8, 22, 22
        panel_w = 260
        panel_h = lh * len(lines) + 8

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 18), (x0 + panel_w, y0 + panel_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x0, y0 + i * lh),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (210, 230, 255),
                1,
                cv2.LINE_AA
            )

    def run(
        self,
        video_path: str,
        out_video: str = "renders/traffic_lights.mp4",
        out_json: str = "renders/traffic_lights.json",
        max_frames: Optional[int] = None,
        frame_skip: int = 2,
        metadata: Optional[Dict[str, Any]] = None,
        vscode_compatible: bool = True,
    ) -> List[Dict[str, Any]]:
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {src}")

        out_video_path = Path(out_video)
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = src_fps / max(frame_skip, 1)

        print("\n" + "=" * 72)
        print("  TrafficLightPipeline.run()")
        print(f"  Input  : {src} ({W}x{H} @ {src_fps:.1f} fps, ~{total_frames} frames)")
        print(f"  Output : {out_video_path}")
        print(f"  JSON   : {out_json_path}")
        print(f"  Skip   : every {frame_skip} frame(s) -> ~{out_fps:.1f} fps output")
        print("=" * 72 + "\n")

        writer = SafeVideoWriter(out_video_path, out_fps, W, H, vscode_compatible=vscode_compatible)

        all_records: List[Dict[str, Any]] = []
        total_dets = 0
        written = 0
        src_idx = 0
        t_start = time.time()
        fps_window: List[float] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                ts = src_idx / src_fps
                t0 = time.time()

                detections = self.detector.detect(frame)
                total_dets += len(detections)

                rec = build_frame_record(written, ts, detections, metadata)
                all_records.append(rec)

                vis = self.detector.draw(frame, detections)

                fps_window.append(time.time() - t0)
                if len(fps_window) > 30:
                    fps_window.pop(0)
                proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

                self.draw_hud(vis, written, proc_fps, len(detections), total_dets)
                writer.write(vis)

                written += 1
                if written % 30 == 0 or written == 1:
                    pct = (src_idx / total_frames * 100) if total_frames > 0 else 0.0
                    print(
                        f"  frame {src_idx:5d} / ~{total_frames} ({pct:5.1f}%)  {proc_fps:5.1f} fps  lights={len(detections):2d} total={total_dets}",
                        end="\r",
                        flush=True
                    )

                src_idx += 1

                if max_frames is not None and written >= max_frames:
                    print(f"\n  Reached max_frames={max_frames} — stopping.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")

        finally:
            cap.release()
            final_video = writer.close()

        with open(out_json_path, "w") as f:
            json.dump(
                {
                    "source": str(src),
                    "frames_written": written,
                    "total_traffic_lights": total_dets,
                    "final_video": str(final_video),
                    "frames": all_records,
                },
                f,
                indent=2,
            )

        elapsed = time.time() - t_start
        print("\n\n" + "=" * 72)
        print("  Done.")
        print(f"  Frames processed     : {written}")
        print(f"  Traffic lights total : {total_dets}")
        print(f"  Wall time            : {elapsed:.1f}s ({written / max(elapsed, 1e-6):.1f} fps avg)")
        print(f"  Output video         : {final_video}")
        print(f"  JSON output          : {out_json_path}")
        print("=" * 72)

        return all_records

    def run_on_frames(
        self,
        frame_dir: str,
        out_video: str = "renders/traffic_lights.mp4",
        out_json: str = "renders/traffic_lights.json",
        fps: float = 15.0,
        max_frames: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vscode_compatible: bool = True,
    ) -> List[Dict[str, Any]]:
        frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
        if not frame_paths:
            frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG frames found in {frame_dir}")

        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")

        H, W = first.shape[:2]

        out_video_path = Path(out_video)
        out_json_path = Path(out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        writer = SafeVideoWriter(out_video_path, fps, W, H, vscode_compatible=vscode_compatible)

        all_records: List[Dict[str, Any]] = []
        total_dets = 0
        fps_window: List[float] = []
        start = time.time()

        for idx, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            t0 = time.time()
            detections = self.detector.detect(frame)
            total_dets += len(detections)

            rec = build_frame_record(idx, idx / fps, detections, metadata)
            all_records.append(rec)

            vis = self.detector.draw(frame, detections)

            fps_window.append(time.time() - t0)
            if len(fps_window) > 30:
                fps_window.pop(0)
            proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

            self.draw_hud(vis, idx, proc_fps, len(detections), total_dets)
            writer.write(vis)

        final_video = writer.close()

        with open(out_json_path, "w") as f:
            json.dump(
                {
                    "source": str(frame_dir),
                    "frames_written": len(all_records),
                    "total_traffic_lights": total_dets,
                    "final_video": str(final_video),
                    "frames": all_records,
                },
                f,
                indent=2,
            )

        elapsed = time.time() - start
        print(f"\nDone. {len(all_records)} frames in {elapsed:.1f}s -> {final_video}")
        return all_records


# =============================================================================
# 10. Singleton helper
# =============================================================================

_detector_singleton: Optional[TrafficLightDetector] = None

def detect_traffic_lights(frame_bgr: np.ndarray, device: str = "auto") -> List[TrafficLightResult]:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = TrafficLightDetector(device=device)
    return _detector_singleton.detect(frame_bgr)


# =============================================================================
# 11. CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved traffic light detection + color classification")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Input video path")
    src.add_argument("--frames", type=str, help="Directory of JPG frames")

    parser.add_argument("--scene", type=str, default=None,
                        help="Optional explicit scene id (e.g. scene1); otherwise inferred from the input path")
    parser.add_argument("--out-video", type=str, default=None,
                        help="Annotated output video path; defaults to output/<scene>/traffic_lights/traffic_lights.mp4")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Traffic-light JSON path; defaults to output/<scene>/traffic_lights/traffic_lights.json")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="auto",
        help="YOLO detector checkpoint. 'auto' prefers local weights/yolo11x.pt, then weights/yolov8x.pt, then weights/yolov8n.pt.",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default="auto",
        help="Traffic-light state classifier. 'auto' prefers weights/best_tl_classifier.pt.",
    )
    parser.add_argument(
        "--detic-color-checker",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Use Detic as a secondary traffic-light color checker. 'auto' enables it when a Detic-capable runtime can be resolved locally.",
    )
    parser.add_argument("--detic-repo", type=str, default="auto",
                        help="Detic repo root for the optional traffic-light color checker.")
    parser.add_argument("--detic-python", type=str, default="auto",
                        help="Optional Python executable from a Detic-capable environment used by the Detic traffic-light color checker.")
    parser.add_argument("--detic-config", type=str, default="auto",
                        help="Detic config path for the optional traffic-light color checker.")
    parser.add_argument("--detic-weights", type=str, default="auto",
                        help="Detic checkpoint for the optional traffic-light color checker.")
    parser.add_argument("--detic-min-confidence", type=float, default=0.18,
                        help="Minimum Detic confidence used by the traffic-light color checker.")
    parser.add_argument("--det-conf", type=float, default=0.22)
    parser.add_argument("--detect-width", type=int, default=960)
    parser.add_argument("--upper-frac", type=float, default=0.68)
    parser.add_argument("--classifier-imgsz", type=int, default=320)
    parser.add_argument("--crop-scale", type=float, default=1.35)
    parser.add_argument("--classical-accept-conf", type=float, default=0.50)
    parser.add_argument("--classifier-accept-conf", type=float, default=0.70)
    parser.add_argument("--no-tracker", action="store_true")
    parser.add_argument("--no-vscode-compatible", action="store_true")

    args = parser.parse_args()

    wants_detic_runtime = str(args.detic_color_checker or "auto").strip().lower() in {"auto", "on"}
    if wants_detic_runtime and os.environ.get("EV_DETIC_REEXEC") != "1":
        try:
            from detic_scene_detector import resolve_detic_python_executable
        except Exception:
            resolve_detic_python_executable = None
        if resolve_detic_python_executable is not None:
            detic_python = resolve_detic_python_executable(args.detic_python)
            if detic_python:
                current_python = str(Path(sys.executable).expanduser().resolve())
                target_python = str(Path(detic_python).expanduser().resolve())
                if current_python != target_python:
                    print(f"[main] Re-launching traffic_light_detector.py under Detic runtime: {target_python}")
                    env = dict(os.environ)
                    env["EV_DETIC_REEXEC"] = "1"
                    os.execve(target_python, [target_python, str(Path(__file__).resolve()), *sys.argv[1:]], env)

    try:
        from ultralytics import YOLO  # noqa
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"[main] device resolved -> {resolve_device(args.device)}")
    print(f"[main] ffmpeg available -> {ffmpeg_available()}")

    detector = TrafficLightDetector(
        device=args.device,
        yolo_model=args.yolo_model,
        classifier_weights=args.classifier_weights,
        detic_color_checker=args.detic_color_checker,
        detic_python=args.detic_python,
        detic_repo=args.detic_repo,
        detic_config=args.detic_config,
        detic_weights=args.detic_weights,
        detic_min_confidence=args.detic_min_confidence,
        det_conf_thresh=args.det_conf,
        detect_width=args.detect_width,
        upper_frac=args.upper_frac,
        classifier_imgsz=args.classifier_imgsz,
        crop_scale=args.crop_scale,
        classical_accept_conf=args.classical_accept_conf,
        classifier_accept_conf=args.classifier_accept_conf,
        use_tracker=not args.no_tracker,
    )

    pipe = TrafficLightPipeline(detector=detector)
    scene_name = infer_scene_name(args.scene, args.video, args.frames, args.out_video, args.out_json)
    output_layout = scene_output_layout(scene_name, create=True)
    out_video = str(Path(args.out_video).resolve()) if args.out_video else str((output_layout.traffic_lights / "traffic_lights.mp4").resolve())
    out_json = str(Path(args.out_json).resolve()) if args.out_json else str((output_layout.traffic_lights / "traffic_lights.json").resolve())

    if args.video:
        pipe.run(
            args.video,
            out_video=out_video,
            out_json=out_json,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
            vscode_compatible=not args.no_vscode_compatible,
        )
    else:
        pipe.run_on_frames(
            args.frames,
            out_video=out_video,
            out_json=out_json,
            fps=args.fps,
            max_frames=args.max_frames,
            vscode_compatible=not args.no_vscode_compatible,
        )

    if not args.out_video:
        mirror_stage_output(out_video, scene_name, "traffic_lights", Path(out_video).name)
    if not args.out_json:
        mirror_stage_output(out_json, scene_name, "traffic_lights", Path(out_json).name)
