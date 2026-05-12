"""
Dedicated robust traffic-sign detection and classification.

This script is intentionally sign-only. It supports either a focused Detic
backend or a YOLO backend, then applies lightweight color/geometry sanity
checks and temporal confirmation to suppress one-frame false positives in
video mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detic_scene_detector import DeticSceneDetector


_DEFAULT_SIGN_VOCABULARY = (
    "stop sign",
    "yield sign",
    "yield ahead sign",
    "speed limit sign",
    "speed limit 15 sign",
    "speed limit 20 sign",
    "speed limit 25 sign",
    "speed limit 30 sign",
    "speed limit 35 sign",
    "speed limit 40 sign",
    "speed limit 45 sign",
    "speed limit 50 sign",
    "speed limit 55 sign",
    "speed limit 60 sign",
    "speed limit 65 sign",
    "speed limit 70 sign",
    "pedestrian crossing sign",
    "crosswalk sign",
    "signal ahead sign",
    "merge sign",
    "keep right sign",
    "keep left sign",
    "one way sign",
    "do not enter sign",
    "school zone sign",
    "road work sign",
    "construction sign",
    "speed bump sign",
    "speed hump sign",
    "bump ahead sign",
    "speed breaker sign",
    "warning sign",
)

_DEFAULT_REFINEMENT_VOCABULARY = (
    "stop sign",
    "yield sign",
    "yield ahead sign",
    "speed limit 15 sign",
    "speed limit 20 sign",
    "speed limit 25 sign",
    "speed limit 30 sign",
    "speed limit 35 sign",
    "speed limit 40 sign",
    "speed limit 45 sign",
    "speed limit 50 sign",
    "speed limit 55 sign",
    "speed limit 60 sign",
    "speed limit 65 sign",
    "speed limit 70 sign",
    "pedestrian crossing sign",
    "crosswalk sign",
    "signal ahead sign",
    "merge sign",
    "keep right sign",
    "keep left sign",
    "one way sign",
    "do not enter sign",
    "school zone sign",
    "road work sign",
    "construction sign",
    "speed bump sign",
    "speed hump sign",
    "bump ahead sign",
    "speed breaker sign",
)

_BUMP_WARNING_LABELS = {
    "speed_bump_warning",
    "speed_hump_warning",
    "bump_ahead",
}

_YOLO_SIGN_MODEL_CANDIDATES = (
    "weights/best.pt",
    "weights/traffic_sign_best.pt",
    "weights/TrafficSignDetection_best.pt",
    "weights/yolo11_traffic_signs_best.pt",
    "weights/yolo11x.pt",
    "weights/yolo26x.pt",
    "weights/yolov8x.pt",
    "weights/yolov8n.pt",
)


@dataclasses.dataclass
class SignDetection:
    bbox: List[int]
    confidence: float
    raw_label: str
    sign_label: str
    region_name: str
    class_id: Optional[int] = None
    color_stats: Optional[Dict[str, float]] = None
    source: str = "detic"
    track_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "bbox": [int(v) for v in self.bbox[:4]],
            "confidence": float(self.confidence),
            "raw_label": str(self.raw_label),
            "sign_label": str(self.sign_label),
            "region_name": str(self.region_name),
            "source": str(self.source),
        }
        if self.class_id is not None:
            payload["class_id"] = int(self.class_id)
        if self.track_id is not None:
            payload["track_id"] = int(self.track_id)
        if self.color_stats:
            payload["color_stats"] = {k: float(v) for k, v in self.color_stats.items()}
        return payload


@dataclasses.dataclass
class SignTrack:
    track_id: int
    bbox: List[float]
    family: str
    hits: int
    misses: int
    score_ema: float
    last_seen_confidence: float
    last_frame_index: int
    label_votes: Dict[str, float] = dataclasses.field(default_factory=dict)
    raw_votes: Dict[str, float] = dataclasses.field(default_factory=dict)
    color_stats: Dict[str, float] = dataclasses.field(default_factory=dict)
    updated_this_frame: bool = False

    def best_label(self) -> str:
        if not self.label_votes:
            return self.family
        return max(self.label_votes.items(), key=lambda item: item[1])[0]

    def best_raw_label(self) -> str:
        if not self.raw_votes:
            return self.best_label()
        return max(self.raw_votes.items(), key=lambda item: item[1])[0]

    def update(self, det: SignDetection, frame_index: int, alpha: float) -> None:
        alpha = float(min(max(alpha, 0.0), 1.0))
        self.bbox = [
            (1.0 - alpha) * float(old) + alpha * float(new)
            for old, new in zip(self.bbox, det.bbox)
        ]
        self.score_ema = (1.0 - alpha) * float(self.score_ema) + alpha * float(det.confidence)
        self.last_seen_confidence = float(det.confidence)
        self.hits += 1
        self.misses = 0
        self.last_frame_index = int(frame_index)
        self.updated_this_frame = True
        self.label_votes[det.sign_label] = self.label_votes.get(det.sign_label, 0.0) + float(det.confidence)
        self.raw_votes[det.raw_label] = self.raw_votes.get(det.raw_label, 0.0) + float(det.confidence)
        if det.color_stats:
            for key, value in det.color_stats.items():
                old_value = float(self.color_stats.get(key, 0.0))
                self.color_stats[key] = (1.0 - alpha) * old_value + alpha * float(value)

    def as_detection(self) -> SignDetection:
        return SignDetection(
            bbox=[int(round(v)) for v in self.bbox[:4]],
            confidence=float(max(self.score_ema, self.last_seen_confidence)),
            raw_label=self.best_raw_label(),
            sign_label=self.best_label(),
            region_name="tracked",
            class_id=None,
            color_stats={k: float(v) for k, v in self.color_stats.items()},
            source="detic+tracking",
            track_id=int(self.track_id),
        )


class SignTracker:
    def __init__(
        self,
        *,
        match_iou: float = 0.35,
        confirm_hits: int = 2,
        max_missed: int = 2,
        smoothing_alpha: float = 0.55,
        bypass_confidence: float = 0.82,
    ) -> None:
        self.match_iou = float(match_iou)
        self.confirm_hits = int(max(1, confirm_hits))
        self.max_missed = int(max(0, max_missed))
        self.smoothing_alpha = float(min(max(smoothing_alpha, 0.0), 1.0))
        self.bypass_confidence = float(min(max(bypass_confidence, 0.0), 1.0))
        self.tracks: List[SignTrack] = []
        self._next_track_id = 1

    def update(self, frame_index: int, detections: Sequence[SignDetection]) -> List[SignDetection]:
        for track in self.tracks:
            track.updated_this_frame = False

        used_tracks: set[int] = set()
        ordered = sorted(detections, key=lambda det: float(det.confidence), reverse=True)
        for det in ordered:
            family = _label_family(det.sign_label)
            best_track: Optional[SignTrack] = None
            best_iou = 0.0
            for track in self.tracks:
                if track.track_id in used_tracks:
                    continue
                if track.family != family:
                    continue
                iou = _bbox_iou(track.bbox, det.bbox)
                if iou >= self.match_iou and iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is None:
                track = SignTrack(
                    track_id=self._next_track_id,
                    bbox=[float(v) for v in det.bbox[:4]],
                    family=family,
                    hits=1,
                    misses=0,
                    score_ema=float(det.confidence),
                    last_seen_confidence=float(det.confidence),
                    last_frame_index=int(frame_index),
                    label_votes={det.sign_label: float(det.confidence)},
                    raw_votes={det.raw_label: float(det.confidence)},
                    color_stats={k: float(v) for k, v in (det.color_stats or {}).items()},
                    updated_this_frame=True,
                )
                self.tracks.append(track)
                used_tracks.add(track.track_id)
                self._next_track_id += 1
                continue

            best_track.update(det, frame_index, self.smoothing_alpha)
            used_tracks.add(best_track.track_id)

        survivors: List[SignTrack] = []
        for track in self.tracks:
            if not track.updated_this_frame:
                track.misses += 1
            if track.misses <= self.max_missed:
                survivors.append(track)
        self.tracks = survivors

        outputs: List[SignDetection] = []
        for track in self.tracks:
            if track.last_frame_index != int(frame_index):
                continue
            if track.hits >= self.confirm_hits or track.last_seen_confidence >= self.bypass_confidence:
                outputs.append(track.as_detection())

        outputs.sort(key=lambda det: float(det.confidence), reverse=True)
        return outputs


class UltralyticsSignDetector:
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        min_confidence: float = 0.20,
        imgsz: int = 1280,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError(
                "YOLO backend requires ultralytics. Install it in the active environment."
            ) from exc

        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"YOLO model not found: {resolved}")
        self.model_path = resolved
        self.device = str(device or "cpu").strip()
        self.min_confidence = float(max(0.0, min(1.0, min_confidence)))
        self.imgsz = int(max(320, imgsz))
        self.model = YOLO(str(resolved))
        self.model_name = resolved.name.lower()

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        result = self.model(
            frame_bgr,
            verbose=False,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.min_confidence,
        )[0]
        names = result.names if hasattr(result, "names") else {}
        detections: List[Dict[str, Any]] = []
        for box in getattr(result, "boxes", []):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if conf < self.min_confidence:
                continue
            raw_label = str(names.get(cls_id, str(cls_id)))
            sign_label = _normalize_yolo_sign_label(raw_label, self.model_name)
            if not _is_sign_like_label(sign_label):
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "raw_label": raw_label,
                    "class_id": cls_id,
                    "sign_label": sign_label,
                    "source": "yolo",
                }
            )
        return detections


def _normalize_text(label: str) -> str:
    text = str(label or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_speed_limit_value(raw_label: str) -> Optional[int]:
    norm = _normalize_text(raw_label)
    if "speed" not in norm and "mph" not in norm and "kmh" not in norm:
        return None
    values: List[int] = []
    for token in re.findall(r"\d{1,3}", norm):
        try:
            value = int(token)
        except (TypeError, ValueError):
            continue
        if 5 <= value <= 120:
            values.append(value)
    return values[0] if values else None


def _normalize_yolo_sign_label(raw_label: str, model_name: str = "") -> str:
    norm = _normalize_text(raw_label)
    model_name = str(model_name or "").strip().lower()
    if model_name in {"trafic.pt", "traffic.pt"}:
        if norm.isdigit():
            try:
                value = int(norm)
            except ValueError:
                value = -1
            if 5 <= value <= 120:
                return f"speed_limit_{int(value)}"
        turkish_map = {
            "dur": "stop",
            "durak": "traffic_sign",
            "girisyok": "do_not_enter",
            "ilerisag": "keep_right",
            "ilerisol": "keep_left",
            "yayagecidi": "pedestrian_crossing",
            "sag": "keep_right",
            "sol": "keep_left",
            "yaya": "pedestrian_crossing",
        }
        if norm in turkish_map:
            return turkish_map[norm]
    return _normalize_sign_label(raw_label)


def _normalize_sign_label(raw_label: str) -> str:
    norm = _normalize_text(raw_label)
    if not norm:
        return "traffic_sign"

    speed_value = _parse_speed_limit_value(raw_label)
    if speed_value is not None:
        return f"speed_limit_{int(speed_value)}"

    if norm in {"speed_limit", "speed_limit_sign"}:
        return "speed_limit"
    if norm.startswith("stop") or "stop_sign" in norm:
        return "stop"
    if "yield_ahead" in norm or "yieldahead" in norm:
        return "yield_ahead"
    if norm.startswith("yield") or "yield_sign" in norm:
        return "yield"
    if "pedestrian_crossing" in norm or "crosswalk" in norm:
        return "pedestrian_crossing"
    if "signal_ahead" in norm:
        return "signal_ahead"
    if "merge" in norm:
        return "merge"
    if "keep_right" in norm or "keepright" in norm:
        return "keep_right"
    if "keep_left" in norm or "keepleft" in norm:
        return "keep_left"
    if "do_not_enter" in norm or "donotenter" in norm:
        return "do_not_enter"
    if "one_way" in norm or "oneway" in norm:
        return "one_way"
    if "school_zone" in norm:
        return "school_zone"
    if "road_work" in norm or "construction" in norm:
        return "road_work"
    if "speed_hump" in norm or "speed_breaker" in norm or "speed_cushion" in norm:
        return "speed_hump_warning"
    if "bump_ahead" in norm:
        return "bump_ahead"
    if "speed_bump" in norm:
        return "speed_bump_warning"
    if "warning_sign" in norm or "warning" == norm:
        return "warning_sign"
    return "traffic_sign"


def _is_sign_like_label(label: str) -> bool:
    label = str(label or "").strip()
    if not label:
        return False
    if label.startswith("speed_limit_") or label == "speed_limit":
        return True
    if label in {
        "traffic_sign",
        "warning_sign",
        "stop",
        "yield",
        "yield_ahead",
        "pedestrian_crossing",
        "signal_ahead",
        "merge",
        "keep_right",
        "keep_left",
        "one_way",
        "do_not_enter",
        "school_zone",
        "road_work",
        "speed_bump_warning",
        "speed_hump_warning",
        "bump_ahead",
    }:
        return True
    return False


def _label_family(label: str) -> str:
    label = str(label or "").strip()
    if label.startswith("speed_limit_") or label == "speed_limit":
        return "speed_limit"
    if label in _BUMP_WARNING_LABELS:
        return "bump_warning"
    return label


def _is_specific_label(label: str) -> bool:
    return str(label or "") not in {"", "traffic_sign", "warning_sign", "speed_limit"}


def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
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


def _clip_bbox(bbox: Sequence[int], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return [x1, y1, x2, y2]


def _crop_color_stats(image_bgr: np.ndarray, bbox: Sequence[int]) -> Dict[str, float]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    if x2 <= x1 or y2 <= y1:
        return {"red_ratio": 0.0, "white_ratio": 0.0, "yellow_ratio": 0.0, "black_ratio": 0.0}

    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = max(int(round(box_w * 0.08)), 1)
    pad_y = max(int(round(box_h * 0.08)), 1)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return {"red_ratio": 0.0, "white_ratio": 0.0, "yellow_ratio": 0.0, "black_ratio": 0.0}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_a = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
    red_b = cv2.inRange(hsv, (165, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red_a, red_b)
    yellow = cv2.inRange(hsv, (14, 60, 80), (40, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 165), (180, 55, 255))
    black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))

    total = float(max(crop.shape[0] * crop.shape[1], 1))
    return {
        "red_ratio": float(np.count_nonzero(red) / total),
        "yellow_ratio": float(np.count_nonzero(yellow) / total),
        "white_ratio": float(np.count_nonzero(white) / total),
        "black_ratio": float(np.count_nonzero(black) / total),
    }


def _extract_padded_crop(
    image_bgr: np.ndarray,
    bbox: Sequence[int],
    *,
    pad_ratio: float = 0.30,
) -> Tuple[np.ndarray, int, int]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    pad_x = max(int(round(box_w * float(pad_ratio))), 2)
    pad_y = max(int(round(box_h * float(pad_ratio))), 2)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return image_bgr[cy1:cy2, cx1:cx2], cx1, cy1


def _looks_like_speed_limit(det: SignDetection) -> bool:
    x1, y1, x2, y2 = det.bbox
    box_w = max(float(x2 - x1), 1.0)
    box_h = max(float(y2 - y1), 1.0)
    aspect = box_w / box_h
    stats = det.color_stats or {}
    white_ratio = float(stats.get("white_ratio", 0.0))
    black_ratio = float(stats.get("black_ratio", 0.0))
    red_ratio = float(stats.get("red_ratio", 0.0))
    yellow_ratio = float(stats.get("yellow_ratio", 0.0))
    return bool(
        0.35 <= aspect <= 1.25
        and white_ratio >= 0.10
        and black_ratio >= 0.012
        and red_ratio <= 0.12
        and yellow_ratio <= 0.18
    )


def _should_override_with_speed_limit(det: SignDetection) -> bool:
    return det.sign_label in {
        "traffic_sign",
        "warning_sign",
        "keep_left",
        "keep_right",
        "one_way",
        "merge",
        "signal_ahead",
    }


def _apply_heuristic_label_override(det: SignDetection) -> SignDetection:
    if _looks_like_speed_limit(det) and _should_override_with_speed_limit(det):
        det.sign_label = "speed_limit"
        det.raw_label = "heuristic_speed_limit"
        det.confidence = max(float(det.confidence), 0.58)
    return det


def _refine_sign_with_crop(
    image_bgr: np.ndarray,
    det: SignDetection,
    refiner: Optional[DeticSceneDetector],
) -> SignDetection:
    if refiner is None:
        return _apply_heuristic_label_override(det)

    crop, ox, oy = _extract_padded_crop(image_bgr, det.bbox, pad_ratio=0.35)
    if crop.size == 0:
        return _apply_heuristic_label_override(det)

    raw_candidates = refiner.detect(crop)
    best_label = det.sign_label
    best_raw = det.raw_label
    best_score = float(det.confidence)

    crop_h, crop_w = crop.shape[:2]
    crop_cx = crop_w * 0.5
    crop_cy = crop_h * 0.5
    for item in raw_candidates:
        raw_label = str(item.get("raw_label") or "")
        sign_label = _normalize_sign_label(raw_label)
        confidence = float(item.get("confidence", 0.0))
        if confidence <= 0.05:
            continue

        bx1, by1, bx2, by2 = [float(v) for v in item.get("bbox", [0, 0, 0, 0])[:4]]
        cx = 0.5 * (bx1 + bx2)
        cy = 0.5 * (by1 + by2)
        center_dist = ((cx - crop_cx) ** 2 + (cy - crop_cy) ** 2) ** 0.5
        max_dist = max((crop_w ** 2 + crop_h ** 2) ** 0.5 * 0.5, 1.0)
        centrality = max(0.0, 1.0 - (center_dist / max_dist))
        score = confidence * (0.70 + 0.30 * centrality)
        if _is_specific_label(sign_label):
            score *= 1.10
        if sign_label.startswith("speed_limit_"):
            score *= 1.18

        if score > best_score:
            best_score = score
            best_label = sign_label
            best_raw = raw_label

    if best_label != det.sign_label:
        det.sign_label = best_label
        det.raw_label = best_raw
        det.confidence = max(float(det.confidence), min(float(best_score), 0.95))

    if _looks_like_speed_limit(det) and _should_override_with_speed_limit(det):
        if best_label.startswith("speed_limit_"):
            det.sign_label = best_label
            det.raw_label = best_raw
        else:
            det.sign_label = "speed_limit"
            det.raw_label = "heuristic_speed_limit"
        det.confidence = max(float(det.confidence), 0.58)

    return det


def _adjust_confidence(
    *,
    det: SignDetection,
    frame_shape: Tuple[int, int, int],
) -> float:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = det.bbox
    box_w = max(int(x2 - x1), 1)
    box_h = max(int(y2 - y1), 1)
    area_ratio = float(box_w * box_h) / float(max(width * height, 1))
    aspect = float(box_w) / float(max(box_h, 1))
    center_y = float(y1 + y2) * 0.5 / float(max(height, 1))
    stats = det.color_stats or {}
    red_ratio = float(stats.get("red_ratio", 0.0))
    yellow_ratio = float(stats.get("yellow_ratio", 0.0))
    white_ratio = float(stats.get("white_ratio", 0.0))

    score = float(det.confidence)
    if box_w < 8 or box_h < 8:
        score *= 0.35
    if area_ratio < 2.5e-5:
        score *= 0.55
    if area_ratio > 0.08:
        score *= 0.30
    if center_y > 0.92:
        score *= 0.35
    elif center_y > 0.82:
        score *= 0.75

    label = det.sign_label
    if label.startswith("speed_limit_") or label == "speed_limit":
        if not (0.50 <= aspect <= 1.35):
            score *= 0.70
        if white_ratio < 0.10:
            score *= 0.58
        elif white_ratio > 0.18:
            score *= 1.08
    elif label in {"stop", "yield", "yield_ahead", "do_not_enter"}:
        if red_ratio < 0.03:
            score *= 0.55
        else:
            score *= 1.08
        if label == "yield" and not (0.55 <= aspect <= 1.50):
            score *= 0.75
    elif label in {"pedestrian_crossing", "signal_ahead", "merge", "school_zone", "road_work"} or label in _BUMP_WARNING_LABELS or label == "warning_sign":
        if yellow_ratio > 0.04:
            score *= 1.12
        elif white_ratio < 0.05 and red_ratio < 0.02:
            score *= 0.78
        if not (0.55 <= aspect <= 1.50):
            score *= 0.75
    elif label in {"keep_right", "keep_left", "one_way"}:
        if white_ratio < 0.05:
            score *= 0.82
        if white_ratio >= 0.12 and float(stats.get("black_ratio", 0.0)) >= 0.012 and red_ratio <= 0.12:
            score *= 0.62

    return float(max(0.0, min(score, 0.99)))


def _merge_detection(existing: SignDetection, candidate: SignDetection) -> SignDetection:
    conf_a = float(existing.confidence)
    conf_b = float(candidate.confidence)
    total = max(conf_a + conf_b, 1e-6)
    merged_bbox = [
        int(round((conf_a * float(existing.bbox[idx]) + conf_b * float(candidate.bbox[idx])) / total))
        for idx in range(4)
    ]
    if _is_specific_label(candidate.sign_label) and not _is_specific_label(existing.sign_label):
        chosen_label = candidate.sign_label
        chosen_raw = candidate.raw_label
    elif conf_b > conf_a:
        chosen_label = candidate.sign_label
        chosen_raw = candidate.raw_label
    else:
        chosen_label = existing.sign_label
        chosen_raw = existing.raw_label

    stats: Dict[str, float] = {}
    for key in set((existing.color_stats or {}).keys()) | set((candidate.color_stats or {}).keys()):
        old_v = float((existing.color_stats or {}).get(key, 0.0))
        new_v = float((candidate.color_stats or {}).get(key, 0.0))
        stats[key] = float((conf_a * old_v + conf_b * new_v) / total)

    return SignDetection(
        bbox=merged_bbox,
        confidence=float(max(conf_a, conf_b)),
        raw_label=chosen_raw,
        sign_label=chosen_label,
        region_name=f"{existing.region_name}+{candidate.region_name}",
        class_id=existing.class_id if existing.class_id is not None else candidate.class_id,
        color_stats=stats,
        source="detic+merged",
    )


def _merge_duplicate_detections(
    detections: Iterable[SignDetection],
    *,
    iou_threshold: float = 0.42,
) -> List[SignDetection]:
    merged: List[SignDetection] = []
    for det in sorted(detections, key=lambda item: float(item.confidence), reverse=True):
        match_idx = None
        for idx, keep in enumerate(merged):
            if _label_family(keep.sign_label) != _label_family(det.sign_label):
                continue
            if _bbox_iou(keep.bbox, det.bbox) >= iou_threshold:
                match_idx = idx
                break
        if match_idx is None:
            merged.append(det)
        else:
            merged[match_idx] = _merge_detection(merged[match_idx], det)
    merged.sort(key=lambda item: float(item.confidence), reverse=True)
    return merged


def _suppress_generic_signs(detections: Iterable[SignDetection]) -> List[SignDetection]:
    detections = list(detections)
    specifics = [det for det in detections if _is_specific_label(det.sign_label)]
    if not specifics:
        return detections

    refined: List[SignDetection] = []
    for det in detections:
        if _is_specific_label(det.sign_label):
            refined.append(det)
            continue
        suppress = False
        for keep in specifics:
            if _bbox_iou(det.bbox, keep.bbox) >= 0.35:
                suppress = True
                break
        if not suppress:
            refined.append(det)
    return refined


def _generate_search_regions(width: int, height: int, mode: str) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    width = int(width)
    height = int(height)
    mode = str(mode or "multi").strip().lower()
    if mode == "full":
        return [("full", (0, 0, width, height))]

    regions: List[Tuple[str, Tuple[int, int, int, int]]] = [
        ("full", (0, 0, width, height)),
        ("upper", (0, 0, width, int(round(height * 0.78)))),
        ("upper_left", (0, 0, int(round(width * 0.58)), int(round(height * 0.72)))),
        ("upper_center", (int(round(width * 0.18)), 0, int(round(width * 0.82)), int(round(height * 0.72)))),
        ("upper_right", (int(round(width * 0.42)), 0, width, int(round(height * 0.72)))),
    ]
    if mode == "multi":
        regions.extend(
            [
                ("roadside_left", (0, int(round(height * 0.05)), int(round(width * 0.40)), int(round(height * 0.92)))),
                ("roadside_right", (int(round(width * 0.60)), int(round(height * 0.05)), width, int(round(height * 0.92)))),
            ]
        )
    return regions


def _detect_signs_in_frame(
    frame_bgr: np.ndarray,
    detector: DeticSceneDetector,
    refiner_getter: Optional[Callable[[], Optional[DeticSceneDetector]]],
    *,
    proposal_floor: float,
    final_min_confidence: float,
    search_mode: str,
) -> List[SignDetection]:
    frame_h, frame_w = frame_bgr.shape[:2]
    proposals: List[SignDetection] = []
    for region_name, (x1, y1, x2, y2) in _generate_search_regions(frame_w, frame_h, search_mode):
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        raw = detector.detect(crop)
        for item in raw:
            raw_bbox = [int(v) for v in item.get("bbox", [0, 0, 0, 0])[:4]]
            gx1 = x1 + raw_bbox[0]
            gy1 = y1 + raw_bbox[1]
            gx2 = x1 + raw_bbox[2]
            gy2 = y1 + raw_bbox[3]
            clipped = _clip_bbox([gx1, gy1, gx2, gy2], frame_w, frame_h)
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue

            det = SignDetection(
                bbox=clipped,
                confidence=float(item.get("confidence", 0.0)),
                raw_label=str(item.get("raw_label") or ""),
                sign_label=str(item.get("sign_label") or _normalize_sign_label(str(item.get("raw_label") or ""))),
                region_name=region_name,
                class_id=item.get("class_id"),
                color_stats=_crop_color_stats(frame_bgr, clipped),
                source=str(item.get("source") or "detector"),
            )
            proposals.append(det)

    refiner: Optional[DeticSceneDetector] = None
    if proposals and refiner_getter is not None:
        refiner = refiner_getter()

    refined_proposals: List[SignDetection] = []
    for det in proposals:
        det = _refine_sign_with_crop(frame_bgr, det, refiner)
        det.confidence = _adjust_confidence(det=det, frame_shape=frame_bgr.shape)
        if det.confidence < proposal_floor:
            continue
        refined_proposals.append(det)

    merged = _merge_duplicate_detections(refined_proposals)
    merged = _suppress_generic_signs(merged)
    return [det for det in merged if float(det.confidence) >= float(final_min_confidence)]


def _draw_sign_detections(image_bgr: np.ndarray, detections: Sequence[SignDetection]) -> np.ndarray:
    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox[:4]]
        label = f"{det.sign_label} {float(det.confidence):.2f}"
        if det.track_id is not None:
            label = f"{label} id={int(det.track_id)}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, font, 0.54, 1)
        ty = max(y1 - 4, th + bl + 2)
        cv2.rectangle(vis, (x1, ty - th - bl - 2), (x1 + tw + 4, ty + 2), color, cv2.FILLED)
        cv2.putText(vis, label, (x1 + 2, ty), font, 0.54, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def _open_video_writer(output_path: Path, fps: float, width: int, height: int) -> Tuple[cv2.VideoWriter, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(max(fps, 1.0))
    candidates = [
        (output_path, "mp4v"),
        (output_path, "avc1"),
        (output_path.with_suffix(".avi"), "MJPG"),
        (output_path.with_suffix(".avi"), "XVID"),
    ]
    for path, codec in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (int(width), int(height)),
        )
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {output_path}")


def _resolve_vocabulary(custom_vocabulary: str) -> List[str]:
    if custom_vocabulary and str(custom_vocabulary).strip().lower() != "auto":
        values = [term.strip() for term in str(custom_vocabulary).split(",") if term.strip()]
        if values:
            return values
    return list(_DEFAULT_SIGN_VOCABULARY)


def _save_json(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_yolo_model_path(model_arg: str) -> str:
    model_arg = str(model_arg or "auto").strip()
    if model_arg and model_arg.lower() != "auto":
        path = Path(model_arg).expanduser()
        if path.exists():
            return str(path.resolve())
        return model_arg
    for candidate in _YOLO_SIGN_MODEL_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path.resolve())
    return str(Path("weights/best.pt").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust dedicated traffic-sign detector.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Input image path.")
    source_group.add_argument("--video", help="Input video path.")

    parser.add_argument("--backend", choices=("auto", "detic", "yolo"), default="yolo", help="Primary detector backend.")
    parser.add_argument("--detic-repo", default="external/Detic", help="Path to the Detic repo clone.")
    parser.add_argument(
        "--detic-config",
        default="external/Detic/configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml",
        help="Detic config YAML.",
    )
    parser.add_argument(
        "--detic-weights",
        default="external/Detic/models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth",
        help="Detic weights file.",
    )
    parser.add_argument("--yolo-model", default="auto", help="YOLO sign model path or name. 'auto' prefers local sign weights.")
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--device", default="cpu", help="Detic device, typically cpu or cuda.")
    parser.add_argument("--proposal-threshold", type=float, default=0.18, help="Lower internal proposal floor.")
    parser.add_argument("--min-confidence", type=float, default=0.42, help="Final detection confidence threshold.")
    parser.add_argument("--search-mode", choices=("full", "tiles", "multi"), default="multi", help="Search regions to use.")
    parser.add_argument(
        "--crop-refinement-mode",
        choices=("none", "heuristic", "detic"),
        default="heuristic",
        help="Use no crop refinement, heuristic-only refinement, or a second Detic crop classifier.",
    )
    parser.add_argument("--disable-crop-refinement", action="store_true", help="Disable second-pass crop classification.")
    parser.add_argument(
        "--refine-vocabulary",
        default="auto",
        help="Optional comma-separated refinement vocabulary. 'auto' uses a narrower sign classifier vocabulary.",
    )
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame in video mode.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames in video mode.")
    parser.add_argument("--confirm-hits", type=int, default=2, help="Frames needed before a track is shown.")
    parser.add_argument("--max-missed", type=int, default=2, help="How many processed frames a track may disappear.")
    parser.add_argument("--track-iou", type=float, default=0.35, help="IoU threshold for temporal association.")
    parser.add_argument("--track-alpha", type=float, default=0.55, help="BBox/confidence smoothing alpha.")
    parser.add_argument("--bypass-confidence", type=float, default=0.82, help="High-confidence tracks bypass confirm-hits.")
    parser.add_argument("--out-image", default=None, help="Annotated output image path.")
    parser.add_argument("--out-video", default=None, help="Annotated output video path.")
    parser.add_argument("--out-json", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--custom-vocabulary",
        default="auto",
        help="Optional comma-separated custom sign vocabulary. 'auto' uses the built-in vocabulary.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    refinement_mode = str(args.crop_refinement_mode).strip().lower()
    if bool(args.disable_crop_refinement):
        refinement_mode = "none"
    backend = str(args.backend or "auto").strip().lower()
    if backend == "auto":
        backend = "yolo"
    if backend == "yolo" and refinement_mode == "detic":
        print("[robust_traffic_sign_detector.py] Crop refinement mode 'detic' is only supported with backend=detic; using heuristic instead.")
        refinement_mode = "heuristic"

    detector: Any
    detector_info: Dict[str, Any] = {"backend": backend}

    if backend == "yolo":
        yolo_model_path = _resolve_yolo_model_path(args.yolo_model)
        print("[robust_traffic_sign_detector.py] Initializing primary YOLO sign detector...")
        detector = UltralyticsSignDetector(
            yolo_model_path,
            device=args.device,
            min_confidence=float(max(0.01, args.proposal_threshold)),
            imgsz=int(args.yolo_imgsz),
        )
        detector_info["yolo_model"] = yolo_model_path
        detector_info["yolo_imgsz"] = int(args.yolo_imgsz)
        print(f"[robust_traffic_sign_detector.py] Primary detector ready. model={Path(yolo_model_path).name}")
    else:
        print("[robust_traffic_sign_detector.py] Initializing primary Detic sign detector...")
        detector = DeticSceneDetector(
            repo_root=args.detic_repo,
            config_file=args.detic_config,
            weights_path=args.detic_weights,
            device=args.device,
            min_confidence=float(args.proposal_threshold),
            vocabulary=_resolve_vocabulary(args.custom_vocabulary),
        )
        detector_info["detic_repo"] = str(args.detic_repo)
        detector_info["detic_config"] = str(args.detic_config)
        detector_info["detic_weights"] = str(args.detic_weights)
        print("[robust_traffic_sign_detector.py] Primary detector ready.")

    print(f"[robust_traffic_sign_detector.py] Crop refinement mode: {refinement_mode}")

    refiner_holder: Dict[str, Optional[DeticSceneDetector]] = {"model": None}

    def _get_refiner() -> Optional[DeticSceneDetector]:
        if backend != "detic" or refinement_mode != "detic":
            return None
        if refiner_holder["model"] is None:
            print("[robust_traffic_sign_detector.py] Initializing crop refiner...")
            refiner_holder["model"] = DeticSceneDetector(
                repo_root=args.detic_repo,
                config_file=args.detic_config,
                weights_path=args.detic_weights,
                device=args.device,
                min_confidence=float(max(0.05, min(0.15, float(args.proposal_threshold)))),
                vocabulary=_resolve_vocabulary(args.refine_vocabulary)
                if str(args.refine_vocabulary).strip().lower() != "auto"
                else list(_DEFAULT_REFINEMENT_VOCABULARY),
            )
            print("[robust_traffic_sign_detector.py] Crop refiner ready.")
        return refiner_holder["model"]

    meta: Dict[str, Any] = {
        "backend": backend,
        "proposal_threshold": float(args.proposal_threshold),
        "min_confidence": float(args.min_confidence),
        "search_mode": str(args.search_mode),
        "crop_refinement_mode": refinement_mode,
    }
    if backend == "detic":
        meta["vocabulary"] = _resolve_vocabulary(args.custom_vocabulary)
    meta.update(detector_info)

    if args.image:
        image_path = Path(args.image).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        out_path = (
            Path(args.out_image).expanduser()
            if args.out_image
            else image_path.with_name(f"{image_path.stem}_robust_signs{image_path.suffix}")
        )

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        t0 = time.time()
        detections = _detect_signs_in_frame(
            image_bgr,
            detector,
            _get_refiner,
            proposal_floor=float(max(0.05, args.proposal_threshold)),
            final_min_confidence=float(args.min_confidence),
            search_mode=str(args.search_mode),
        )
        elapsed = time.time() - t0
        vis = _draw_sign_detections(image_bgr, detections)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), vis):
            raise RuntimeError(f"Could not write output image: {out_path}")

        payload = {
            "mode": "image",
            "input": str(image_path),
            "output": str(out_path),
            "elapsed_s": float(elapsed),
            "meta": meta,
            "detections": [det.to_dict() for det in detections],
        }
        _save_json(args.out_json, payload)

        print(f"[robust_traffic_sign_detector.py] image={image_path}")
        print(f"[robust_traffic_sign_detector.py] output={out_path}")
        print(
            f"[robust_traffic_sign_detector.py] detections={len(detections)} "
            f"elapsed_s={elapsed:.3f}"
        )
        return

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_path = (
        Path(args.out_video).expanduser()
        if args.out_video
        else video_path.with_name(f"{video_path.stem}_robust_signs.mp4")
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0.0:
        src_fps = 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_skip = max(1, int(args.frame_skip))
    out_fps = max(src_fps / float(frame_skip), 1.0)
    writer, final_out_path = _open_video_writer(out_path, out_fps, width, height)

    tracker = SignTracker(
        match_iou=float(args.track_iou),
        confirm_hits=int(args.confirm_hits),
        max_missed=int(args.max_missed),
        smoothing_alpha=float(args.track_alpha),
        bypass_confidence=float(args.bypass_confidence),
    )

    processed = 0
    read_idx = -1
    total_time = 0.0
    frame_records: List[Dict[str, Any]] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            read_idx += 1
            if read_idx % frame_skip != 0:
                continue

            t0 = time.time()
            frame_detections = _detect_signs_in_frame(
                frame_bgr,
                detector,
                _get_refiner,
                proposal_floor=float(max(0.05, args.proposal_threshold)),
                final_min_confidence=float(max(0.05, args.min_confidence * 0.70)),
                search_mode=str(args.search_mode),
            )
            stable_detections = [
                det
                for det in tracker.update(processed, frame_detections)
                if float(det.confidence) >= float(args.min_confidence)
            ]
            total_time += time.time() - t0

            vis = _draw_sign_detections(frame_bgr, stable_detections)
            writer.write(vis)

            frame_records.append(
                {
                    "processed_frame_index": int(processed),
                    "source_frame_index": int(read_idx),
                    "timestamp_s": float(read_idx / max(src_fps, 1.0)),
                    "detections": [det.to_dict() for det in stable_detections],
                    "raw_proposals": [det.to_dict() for det in frame_detections],
                }
            )

            processed += 1
            if processed <= 3 or processed % 10 == 0:
                avg = total_time / float(max(processed, 1))
                print(
                    f"[robust_traffic_sign_detector.py] processed={processed} "
                    f"frame_idx={read_idx} avg_s_per_frame={avg:.3f} "
                    f"visible={len(stable_detections)}"
                )

            if args.max_frames is not None and processed >= int(args.max_frames):
                break
    finally:
        cap.release()
        writer.release()

    avg = total_time / float(max(processed, 1))
    payload = {
        "mode": "video",
        "input": str(video_path),
        "output": str(final_out_path),
        "processed_frames": int(processed),
        "frame_skip": int(frame_skip),
        "avg_s_per_frame": float(avg),
        "meta": meta,
        "frames": frame_records,
    }
    _save_json(args.out_json, payload)

    print(f"[robust_traffic_sign_detector.py] video={video_path}")
    print(f"[robust_traffic_sign_detector.py] output={final_out_path}")
    print(
        f"[robust_traffic_sign_detector.py] processed_frames={processed} "
        f"frame_skip={frame_skip} avg_s_per_frame={avg:.3f}"
    )


if __name__ == "__main__":
    main(parse_args())
