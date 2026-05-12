"""
object_detection.py
===================

Primary object-detection stage for the autonomous-driving pipeline.

Responsibilities
----------------
* detect traffic participants and road furniture in image space
* normalize detections into a stable JSON schema used by downstream modules
* keep vehicle labels in the coarse production taxonomy (car, truck, bicycle, motorcycle)
* provide optional video and frame-sequence processing entry points
* preserve room for richer downstream enrichment such as depth, motion, and 3-D position lifting

Backend strategy
----------------
Detic is the preferred detector for broad scene understanding because it can
switch to a custom vocabulary at inference time and therefore cover both the
core traffic actors and long-tail road furniture in one pass. Ultralytics YOLO
is retained as the general fallback detector.

Traffic signs are handled slightly differently: Detic can already detect them,
but the pipeline can additionally refine compact sign detections with either a
DETR sign model, a frozen TensorFlow Faster R-CNN graph exported from the
`meng1994412/Traffic_Sign_Detection` repo, or a dedicated YOLO sign model.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import time
import argparse
import dataclasses
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np

from project_setup import (
    infer_scene_name,
    mirror_stage_output,
    resolve_existing_artifact,
    scene_output_layout,
)
from vehicle_subclassification import VehicleSubclassifier


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Constants & class map
# ─────────────────────────────────────────────────────────────────────────────

_TARGET_CLASSES = {
    "pedestrian",
    "bicycle",
    "car",
    "motorcycle",
    "truck",
    "traffic_light",
    "traffic_sign",
    "stop_sign",
    "speed_limit",
    "dustbin",
    "traffic_pole",
    "traffic_cone",
    "traffic_cylinder",
    "fire_hydrant",
    "brake_light",
    "indicator_light",
}

_VEHICLE_CLASSES = {"car", "truck", "bicycle", "motorcycle"}
_PRIMARY_SIGN_CLASSES = {"traffic_sign", "stop_sign", "speed_limit"}
_BUMP_RELATED_SIGN_LABELS = {
    "speed_bump_warning",
    "speed_hump_warning",
    "bump_ahead",
}
_ROAD_OBJECT_CLASSES = {
    "dustbin",
    "traffic_pole",
    "traffic_cone",
    "traffic_cylinder",
    "fire_hydrant",
    "brake_light",
    "indicator_light",
}
_MOTION_CRITICAL_CLASSES = {"pedestrian", "bicycle", "car", "motorcycle", "truck"}
_STATIC_ROAD_OBJECT_CLASSES = {
    "dustbin",
    "traffic_pole",
    "traffic_cone",
    "traffic_cylinder",
    "fire_hydrant",
}

_LABEL_ALIASES: Dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "cycle": "bicycle",
    "car": "car",
    "automobile": "car",
    "sedan": "car",
    "sedan_car": "car",
    "hatchback": "car",
    "hatchback_car": "car",
    "suv": "car",
    "pickup": "car",
    "pickup_truck": "car",
    "car": "car",
    "van": "truck",
    "truck": "truck",
    "lorry": "truck",
    "bus": "truck",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "traffic_light": "traffic_light",
    "traffic_light_red": "traffic_light",
    "traffic_light_green": "traffic_light",
    "traffic_light_yellow": "traffic_light",
    "traffic_light_red_left": "traffic_light",
    "traffic_light_red_right": "traffic_light",
    "traffic_light_green_left": "traffic_light",
    "traffic_light_green_right": "traffic_light",
    "traffic_light_yellow_left": "traffic_light",
    "traffic_light_yellow_right": "traffic_light",
    "traffic_signal": "traffic_light",
    "traffic_sign": "traffic_sign",
    "road_sign": "traffic_sign",
    "sign": "traffic_sign",
    "stop_sign": "stop_sign",
    "yield": "traffic_sign",
    "yield_sign": "traffic_sign",
    "warning_sign": "traffic_sign",
    "school_zone_sign": "traffic_sign",
    "one_way_sign": "traffic_sign",
    "do_not_enter_sign": "traffic_sign",
    "street_sign": "traffic_sign",
    "speed_limit": "speed_limit",
    "speed_limit_sign": "speed_limit",
    "speed_sign": "speed_limit",
    "dustbin": "dustbin",
    "trash_can": "dustbin",
    "trash_bin": "dustbin",
    "trash_container": "dustbin",
    "bin": "dustbin",
    "garbage_can": "dustbin",
    "trash_bin_can": "dustbin",
    "garbage_bin": "dustbin",
    "waste_bin": "dustbin",
    "waste_container": "dustbin",
    "wheelie_bin": "dustbin",
    "recycling_bin": "dustbin",
    "recycle_bin": "dustbin",
    "litter_bin": "dustbin",
    "rubbish_bin": "dustbin",
    "refuse_bin": "dustbin",
    "refuse_container": "dustbin",
    "traffic_pole": "traffic_pole",
    "sign_pole": "traffic_pole",
    "pole": "traffic_pole",
    "post": "traffic_pole",
    "traffic_cone": "traffic_cone",
    "cone": "traffic_cone",
    "construction_cone": "traffic_cone",
    "safety_cone": "traffic_cone",
    "orange_cone": "traffic_cone",
    "channelizer_cone": "traffic_cone",
    "traffic_pylon": "traffic_cone",
    "pylon": "traffic_cone",
    "traffic_cylinder": "traffic_cylinder",
    "bollard": "traffic_cylinder",
    "barrel": "traffic_cylinder",
    "barrel_bucket": "traffic_cylinder",
    "traffic_barrel": "traffic_cylinder",
    "road_barrel": "traffic_cylinder",
    "construction_barrel": "traffic_cylinder",
    "orange_barrel": "traffic_cylinder",
    "traffic_drum": "traffic_cylinder",
    "drum": "traffic_cylinder",
    "delineator_post": "traffic_cylinder",
    "fire_hydrant": "fire_hydrant",
    "hydrant": "fire_hydrant",
    "brake_light": "brake_light",
    "tail_light": "brake_light",
    "tail_lamp": "brake_light",
    "tail_lights": "brake_light",
    "brake_lamp": "brake_light",
    "indicator_light": "indicator_light",
    "indicator": "indicator_light",
    "turn_signal": "indicator_light",
    "turn_indicator": "indicator_light",
    "blinker": "indicator_light",
    "left_turn_signal": "indicator_light",
    "right_turn_signal": "indicator_light",
    "left_indicator": "indicator_light",
    "right_indicator": "indicator_light",
}

# Default confidence thresholds per class (safety-critical classes get lower bars)
_CONF_THRESHOLDS: Dict[str, float] = {
    "pedestrian": 0.55,
    "bicycle": 0.55,
    "car": 0.55,
    "motorcycle": 0.55,
    "truck": 0.55,
    "traffic_light": 0.55,
    "traffic_sign": 0.55,
    "stop_sign": 0.50,
    "speed_limit": 0.55,
    "dustbin": 0.45,
    "traffic_pole": 0.50,
    "traffic_cone": 0.35,
    "traffic_cylinder": 0.35,
    "fire_hydrant": 0.50,
    "brake_light": 0.35,
    "indicator_light": 0.35,
}

# BGR draw colours per class
_DRAW_COLORS: Dict[str, tuple] = {
    "pedestrian": (255, 140, 0),        # orange
    "bicycle": (200, 200, 0),           # teal
    "car": (0, 220, 0),                 # green
    "motorcycle": (0, 200, 255),        # yellow-ish
    "truck": (200, 0, 0),               # blue
    "traffic_light": (0, 255, 255),     # cyan
    "traffic_sign": (255, 255, 255),    # white
    "stop_sign": (0, 0, 255),           # red
    "speed_limit": (220, 220, 220),     # light gray
    "dustbin": (34, 139, 34),           # dark green
    "traffic_pole": (128, 128, 128),    # gray
    "traffic_cone": (0, 128, 255),      # orange-ish
    "traffic_cylinder": (20, 100, 255), # orange/red
    "fire_hydrant": (40, 40, 220),      # deep red
    "brake_light": (0, 64, 255),        # bright red/orange
    "indicator_light": (0, 191, 255),   # amber
}

_SIGN_LIKE_CLASSES = {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}
_COMPACT_SIGN_CLASSES = {"traffic_sign", "stop_sign", "speed_limit"}
_DISABLED_MODEL_STRINGS = {"none", "off", "disable", "disabled"}
_BASE_IMGSZ = 1280
_SIGN_TILE_IMGSZ = 1600
_SIGN_TILE_OVERLAP = 0.30
_HF_US_SIGN_MODEL_URL = "https://huggingface.co/cvtechniques/TrafficSignDetection/resolve/main/best.pt"
_LOCAL_SIGN_MODEL_CANDIDATES = (
    "best.pt",
    "traffic_sign_best.pt",
    "TrafficSignDetection_best.pt",
    "yolo11_traffic_signs_best.pt",
    "trafic.pt",
    "traffic.pt",
    "yolo26x-signs.pt",
    "yolo26l-signs.pt",
    "yolo26m-signs.pt",
    "yolo26_us_signs.pt",
    "yolo26-us-signs.pt",
    "us_road_signs_yolo26.pt",
)
_FASTER_RCNN_SIGN_MODEL_CANDIDATES = (
    "weights/frozen_inference_graph.pb",
    "weights/fronzen_inference_graph.pb",
    "frozen_inference_graph.pb",
    "fronzen_inference_graph.pb",
    "weights/lisa_faster_rcnn/frozen_inference_graph.pb",
    "weights/lisa_faster_rcnn/fronzen_inference_graph.pb",
)
_FASTER_RCNN_SIGN_LABEL_CANDIDATES = (
    "weights/classes.pbtxt",
    "weights/lisa_classes.pbtxt",
    "classes.pbtxt",
    "lisa_classes.pbtxt",
    "weights/lisa_faster_rcnn/classes.pbtxt",
)
_DETR_SIGN_MODEL_CANDIDATES = (
    "weights/traffic_sign_detr",
    "weights/detr_traffic_sign",
    "weights/detr_signs",
    "vision_transformers/runs/training/detr_resnet50_dc5_75e",
    "vision_transformers/runs/training/detr_resnet50_75e",
    "runs/training/detr_resnet50_dc5_75e",
    "runs/training/detr_resnet50_75e",
)
_RAW_DETR_ARTIFACT_CANDIDATES = (
    "external/Traffic_Sign_Detection_using_DETR/trained_weights/detr_resnet50_dc5_75e/best_model.pth",
    "external/Traffic_Sign_Detection_using_DETR/trained_weights/detr_resnet50_75e/best_model.pth",
    "external/Traffic_Sign_Detection_using_DETR/trained_weights/detr_resnet50_dc5_75e",
    "external/Traffic_Sign_Detection_using_DETR/trained_weights/detr_resnet50_75e",
)
_DETIC_REPO_CANDIDATES = (
    "external/Detic",
    "Detic",
)
_DETIC_CONFIG_REL_CANDIDATES = (
    "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml",
    "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
)
_DETIC_WEIGHT_CANDIDATES = (
    "weights/detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth",
    "weights/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
    "external/Detic/models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth",
    "external/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
)
_DEFAULT_DETIC_VOCABULARY = (
    "pedestrian",
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "passenger car",
    "sedan",
    "sedan car",
    "suv",
    "sport utility vehicle",
    "hatchback",
    "hatchback car",
    "pickup truck",
    "pickup",
    "truck",
    "lorry",
    "van",
    "bus",
    "traffic light",
    "red traffic light",
    "green traffic light",
    "yellow traffic light",
    "red traffic light right arrow",
    "red traffic light left arrow",
    "red traffic light straight arrow",
    "green traffic light right arrow",
    "green traffic light left arrow",
    "green traffic light straight arrow",
    "yellow traffic light right arrow",
    "yellow traffic light left arrow",
    "yellow traffic light straight arrow",
    "stop sign",
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
    "speed bump sign",
    "speed bump warning sign",
    "speed hump sign",
    "speed hump warning sign",
    "bump ahead sign",
    "speed breaker sign",
    "speed cushion sign",
    "yield sign",
    "yield ahead sign",
    "pedestrian crossing sign",
    "crosswalk sign",
    "school zone sign",
    "one way sign",
    "keep right sign",
    "keep left sign",
    "merge sign",
    "signal ahead sign",
    "do not enter sign",
    "road work sign",
    "construction sign",
    "warning sign",
    "traffic sign",
    "dustbin",
    "trash can",
    "trash bin",
    "trash container",
    "garbage can",
    "garbage bin",
    "waste bin",
    "recycling bin",
    "wheelie bin",
    "fire hydrant",
    "hydrant",
    "traffic pole",
    "sign pole",
    "pole",
    "traffic cone",
    "cone",
    "construction cone",
    "safety cone",
    "orange cone",
    "traffic pylon",
    "pylon",
    "bollard",
    "traffic barrel",
    "construction barrel",
    "road barrel",
    "traffic drum",
    "drum",
    "delineator post",
    "traffic cylinder",
    "brake light",
    "brake lamp",
    "tail light",
    "turn signal",
    "indicator light",
    "turn indicator",
    "left turn signal",
    "right turn signal",
    "left indicator",
    "right indicator",
)
_DETIC_TILED_CLASSES = {
    "traffic_light",
    "traffic_sign",
    "stop_sign",
    "speed_limit",
    "traffic_cone",
    "traffic_cylinder",
    "traffic_pole",
    "dustbin",
    "fire_hydrant",
    "brake_light",
    "indicator_light",
}
_DETIC_TILE_OVERLAP = 0.35
_DETIC_TILE_MIN_SIDE = 960
_ROAD_OBJECT_ZOOM_CLASSES = {
    "traffic_cone",
    "traffic_cylinder",
    "traffic_pole",
    "dustbin",
    "fire_hydrant",
}
_ROAD_OBJECT_ROI_TOP = 0.22
_ROAD_OBJECT_TILE_OVERLAP = 0.45
_ROAD_OBJECT_TILE_MIN_W = 480
_ROAD_OBJECT_TILE_MIN_H = 320
_DEFAULT_OUTPUT_MIN_CONFIDENCE = 0.70
_OUTPUT_CLASS_MIN_CONFIDENCE: Dict[str, float] = {
    "pedestrian": 0.55,
    "bicycle": 0.55,
    "car": 0.55,
    "motorcycle": 0.55,
    "truck": 0.55,
    "dustbin": 0.45,
    "traffic_pole": 0.45,
    "traffic_cone": 0.35,
    "traffic_cylinder": 0.35,
    "fire_hydrant": 0.5,
    "traffic_light": 0.50,
    "traffic_sign": 0.55,
    "stop_sign": 0.50,
    "speed_limit": 0.55,
}
_TEMPORAL_SMOOTH_ALPHA = 0.68
_TEMPORAL_TRACK_MAX_AGE = 6
_VEHICLE_SUBCLASS_MIN_CONFIDENCE = 0.44

# ─────────────────────────────────────────────────────────────────────────────
# 1b.  Vehicle subclassification heuristics
# ─────────────────────────────────────────────────────────────────────────────

def _subclassify_vehicle(cls_name: str, bbox: List[int]) -> str:
    # Keep the output taxonomy coarse and production-stable.
    return cls_name


def _normalize_label(label: str) -> str:
    text = str(label or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _canonicalize_label(raw_label: str) -> Optional[str]:
    norm = _normalize_label(raw_label)
    if norm in _LABEL_ALIASES:
        return _LABEL_ALIASES[norm]
    if norm in _TARGET_CLASSES:
        return norm
    return None


def _parse_speed_limit_value(raw_label: str) -> Optional[int]:
    norm = _normalize_label(raw_label)
    if not any(token in norm for token in ("speed", "limit", "mph", "kmh")):
        return None

    candidates: List[int] = []
    for token in re.findall(r"\d{1,3}", norm):
        try:
            value = int(token)
        except (TypeError, ValueError):
            continue
        if 5 <= value <= 120:
            candidates.append(value)
    return candidates[0] if candidates else None


def _canonicalize_detection_label(raw_label: str) -> Tuple[Optional[str], Optional[int]]:
    norm = _normalize_label(raw_label)
    speed_limit_value = _parse_speed_limit_value(raw_label)
    bump_like_tokens = (
        "speed_bump",
        "speed_hump",
        "speed_breaker",
        "speed_cushion",
        "road_bump",
        "road_hump",
        "raised_crosswalk",
        "raised_table",
        "sleeping_policeman",
        "bump_ahead",
    )
    if speed_limit_value is not None:
        return "speed_limit", speed_limit_value

    if (
        "traffic" in norm and ("light" in norm or "signal" in norm)
    ) or norm in {"red_light", "yellow_light", "green_light"}:
        return "traffic_light", None
    if norm in {"stop", "stop_sign", "regulatory_stop"} or norm.startswith("stop_sign"):
        return "stop_sign", None
    if norm.startswith("speed_limit") or norm.startswith("speedlimit") or (
        "speed" in norm and ("limit" in norm or "mph" in norm)
    ):
        return "speed_limit", None
    if any(token in norm for token in bump_like_tokens) and any(
        token in norm for token in ("sign", "warning", "ahead", "diamond", "advisory")
    ):
        return "traffic_sign", None
    if any(
        token in norm
        for token in (
            "yield",
            "one_way",
            "do_not_enter",
            "donotenter",
            "school_zone",
            "pedestrian_crossing",
            "crosswalk",
            "merge",
            "lane_ends",
            "intersection",
            "keep_left",
            "keep_right",
            "roundabout",
            "curve",
            "turn_only",
            "left_turn",
            "right_turn",
            "u_turn",
            "signal_ahead",
            "detour",
            "construction",
            "road_work",
            "children",
            "warning",
            "regulatory",
        )
    ):
        return "traffic_sign", None
    if any(token in norm for token in bump_like_tokens):
        return None, None
    if (
        "sign" in norm
        or norm.startswith("regulatory_")
        or norm.startswith("warning_")
        or norm.startswith("information_")
        or norm.startswith("complementary_")
    ):
        return "traffic_sign", None

    cls_name = _canonicalize_label(raw_label)
    if cls_name is None:
        return None, None
    return cls_name, None


def _normalize_sign_subclass_label(raw_label: str, canonical_class: Optional[str]) -> str:
    norm = _normalize_label(raw_label)
    if not norm:
        return str(canonical_class or "").strip()

    if canonical_class == "speed_limit":
        value = _parse_speed_limit_value(raw_label)
        if value is not None:
            return f"speed_limit_{int(value)}"
        return "speed_limit"
    if canonical_class == "stop_sign":
        return "stop"
    if any(
        token in norm
        for token in (
            "speed_bump",
            "speed_hump",
            "speed_breaker",
            "speed_cushion",
            "bump_ahead",
            "road_bump",
            "road_hump",
            "sleeping_policeman",
        )
    ):
        if "hump" in norm:
            return "speed_hump_warning"
        if "ahead" in norm and "speed" not in norm:
            return "bump_ahead"
        return "speed_bump_warning"

    direct_map = {
        "do_not_enter": "do_not_enter",
        "donotenter": "do_not_enter",
        "do_not_enter_sign": "do_not_enter",
        "keep_right": "keep_right",
        "keepright": "keep_right",
        "keep_left": "keep_left",
        "keepleft": "keep_left",
        "merge": "merge",
        "crosswalk_sign": "pedestrian_crossing",
        "pedestrian_crossing": "pedestrian_crossing",
        "pedestriancrossing": "pedestrian_crossing",
        "pedestrian_crossing_sign": "pedestrian_crossing",
        "signal_ahead": "signal_ahead",
        "signalahead": "signal_ahead",
        "school_zone_sign": "school_zone",
        "yield": "yield",
        "yield_ahead": "yield_ahead",
        "yeildahead": "yield_ahead",
        "yieldahead": "yield_ahead",
        "yield_sign": "yield",
        "one_way_sign": "one_way",
        "stop": "stop",
        "stop_sign": "stop",
        "warning_pedestrians": "pedestrian_crossing",
    }
    if norm in direct_map:
        return direct_map[norm]
    if norm.startswith("speed_limit") or norm.startswith("speedlimit"):
        value = _parse_speed_limit_value(raw_label)
        if value is not None:
            return f"speed_limit_{int(value)}"
    return norm


def _normalize_vehicle_subclass_label(raw_label: str) -> Optional[str]:
    norm = _normalize_label(raw_label)
    if norm in {"sedan", "sedan_car", "compact_sedan", "mid_size_sedan", "full_size_sedan"}:
        return "sedan"
    if norm in {"suv", "sport_utility_vehicle", "compact_suv", "crossover", "crossover_suv"}:
        return "suv"
    if norm in {"hatchback", "hatchback_car", "wagon", "estate_car", "liftback", "minivan"}:
        return "hatchback"
    if norm in {"pickup", "pickup_truck", "crew_cab", "double_cab", "extended_cab"}:
        return "pickup"
    return None


def _derive_subclass_label(
    cls_name: str,
    raw_label: Optional[str],
    bbox: List[int],
    sign_label: Optional[str] = None,
) -> str:
    norm = _normalize_label(raw_label or "")
    if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}:
        return str(sign_label or cls_name)
    if cls_name == "car":
        return _subclassify_vehicle(cls_name, bbox)
    if cls_name == "truck" and norm == "bus":
        return "bus"
    if cls_name == "traffic_light":
        if "red" in norm:
            return "traffic_light_red"
        if "green" in norm:
            return "traffic_light_green"
        if "yellow" in norm or "amber" in norm:
            return "traffic_light_yellow"
        return "traffic_light"
    if cls_name == "indicator_light":
        if "left" in norm:
            return "left_indicator"
        if "right" in norm:
            return "right_indicator"
        return "indicator_light"
    if cls_name == "traffic_cylinder" and norm == "bollard":
        return "bollard"
    return cls_name


def _parse_traffic_light_signal_label(raw_label: Optional[str]) -> Dict[str, str]:
    norm = _normalize_label(raw_label or "")
    if not norm:
        return {
            "signal_color": "unknown",
            "signal_shape": "unknown",
            "signal_state": "unknown",
        }

    color = "unknown"
    if "green" in norm or norm.startswith("go"):
        color = "green"
    elif "yellow" in norm or "amber" in norm or norm.startswith("warning"):
        color = "yellow"
    elif "red" in norm or norm.startswith("stop"):
        color = "red"

    shape = "unknown"
    if any(token in norm for token in ("left_arrow", "go_left", "left", "stopleft", "warningleft")):
        shape = "left_arrow"
    elif any(token in norm for token in ("right_arrow", "go_right", "right", "stopright", "warningright")):
        shape = "right_arrow"
    elif any(token in norm for token in ("straight_arrow", "go_forward", "forward", "straight")):
        shape = "straight_arrow"
    elif "circle" in norm or "traffic_light" in norm:
        shape = "circle"

    state = f"{color}_{shape}" if color != "unknown" and shape != "unknown" else (
        color if color != "unknown" else "unknown"
    )
    return {
        "signal_color": color,
        "signal_shape": shape,
        "signal_state": state,
    }


def _road_paint_mask(image_bgr: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), sigmaX=1.2, sigmaY=1.2)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    v_hi = max(168, int(np.percentile(v, 82)))
    s_lo = min(90, int(np.percentile(s, 60)))
    white_mask = (v >= v_hi) & (s <= s_lo)
    yellow_mask = (
        (h >= 10) & (h <= 45)
        & (s >= max(55, int(np.percentile(s, 55))))
        & (v >= max(96, int(np.percentile(v, 58))))
    )
    mask = (white_mask | yellow_mask).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _line_length(line: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    (x1, y1), (x2, y2) = line
    return float(math.hypot(x2 - x1, y2 - y1))


def _resolve_model_path(requested: str, candidates: List[str]) -> str:
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        return str(path.resolve()) if path.exists() else requested
    resolved = resolve_existing_artifact(candidates)
    if resolved:
        return resolved
    return candidates[0]


def _resolve_optional_sign_model_path(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        return str(path.resolve()) if path.exists() else requested
    resolved = resolve_existing_artifact(_LOCAL_SIGN_MODEL_CANDIDATES)
    if resolved:
        return resolved
    return _HF_US_SIGN_MODEL_URL


def _resolve_optional_faster_rcnn_model_path(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        return requested
    for candidate in _FASTER_RCNN_SIGN_MODEL_CANDIDATES:
        if Path(candidate).exists():
            return str(Path(candidate).resolve())
    return None


def _resolve_optional_faster_rcnn_labels_path(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        return requested
    for candidate in _FASTER_RCNN_SIGN_LABEL_CANDIDATES:
        if Path(candidate).exists():
            return str(Path(candidate).resolve())
    return None


def _resolve_optional_detr_model_path(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        if path.exists():
            return str(path.resolve())
        return requested
    for candidate in _DETR_SIGN_MODEL_CANDIDATES:
        path = Path(candidate)
        if _is_transformers_detr_checkpoint_path(path):
            return str(path.resolve())
    for candidate in _RAW_DETR_ARTIFACT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    return None


def _is_transformers_detr_checkpoint_path(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_processor = (path / "preprocessor_config.json").exists() or (path / "processor_config.json").exists()
    has_weights = (
        (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin.index.json").exists()
        or (path / "model.safetensors.index.json").exists()
    )
    return has_config and has_processor and has_weights


def _resolve_optional_raw_detr_artifact_path() -> Optional[str]:
    for candidate in _RAW_DETR_ARTIFACT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    return None


def _resolve_optional_detic_repo_path(requested: Optional[str]) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        if path.exists():
            return str(path.resolve())
        return requested
    for candidate in _DETIC_REPO_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    return None


def _resolve_optional_detic_config_path(
    requested: Optional[str],
    repo_path: Optional[str],
) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        if path.exists():
            return str(path.resolve())
        return requested
    if repo_path:
        repo = Path(repo_path)
        for rel_path in _DETIC_CONFIG_REL_CANDIDATES:
            candidate = repo / rel_path
            if candidate.exists():
                return str(candidate.resolve())
    return None


def _resolve_optional_detic_weights_path(
    requested: Optional[str],
    repo_path: Optional[str],
) -> Optional[str]:
    if requested is None:
        return None
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        if path.exists():
            return str(path.resolve())
        return requested
    for candidate in _DETIC_WEIGHT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    if repo_path:
        repo = Path(repo_path)
        for candidate in _DETIC_WEIGHT_CANDIDATES:
            if candidate.startswith("external/Detic/"):
                rel = candidate.split("external/Detic/", 1)[1]
                possible = repo / rel
                if possible.exists():
                    return str(possible.resolve())
    return None


def _parse_detic_vocabulary(
    custom_vocabulary: Optional[str],
    vocabulary_file: Optional[str],
) -> List[str]:
    if vocabulary_file and vocabulary_file not in {"", "auto"}:
        path = Path(vocabulary_file).expanduser()
        if path.exists():
            terms = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            if terms:
                return terms
    if custom_vocabulary and custom_vocabulary not in {"", "auto"}:
        terms = [term.strip() for term in str(custom_vocabulary).split(",") if term.strip()]
        if terms:
            return terms
    return list(_DEFAULT_DETIC_VOCABULARY)


def _canonicalize_trafic_pt_label(raw_label: str) -> Tuple[Optional[str], Optional[int]]:
    norm = _normalize_label(raw_label)
    if norm.isdigit():
        try:
            value = int(norm)
        except ValueError:
            value = -1
        if 5 <= value <= 120:
            return "speed_limit", value

    if norm in {"dur"}:
        return "stop_sign", None

    if norm in {"kirmizi", "sari", "yesil"}:
        return "traffic_light", None

    if norm in {
        "durak",
        "girisyok",
        "ilerisag",
        "ilerisol",
        "park",
        "parkyasak",
        "parkyasak2",
        "sag",
        "sol",
        "sagadonulmez",
        "soladonulmez",
        "yayagecidi",
        "tasitrafiginekapali",
        "yaya",
        "otobus",
        "bisikletli",
        "arac",
    }:
        return "traffic_sign", None

    return None, None


def _canonicalize_sign_label(
    raw_label: str,
    model_path: Optional[str],
) -> Tuple[Optional[str], Optional[int]]:
    model_name = Path(str(model_path or "")).name.lower()
    if model_name in {"trafic.pt", "traffic.pt"}:
        return _canonicalize_trafic_pt_label(raw_label)
    return _canonicalize_detection_label(raw_label)


def _format_sign_label(sign_label: str) -> str:
    label = str(sign_label or "").strip()
    if not label:
        return label
    if label.startswith("speed_limit_"):
        return label
    if label in _BUMP_RELATED_SIGN_LABELS:
        return "speed_bump_sign"
    return label


def _is_bump_related_sign_label(sign_label: Optional[str]) -> bool:
    label = _normalize_label(sign_label or "")
    if not label:
        return False
    return label in _BUMP_RELATED_SIGN_LABELS or (
        "speed_bump" in label
        or "speed_hump" in label
        or "speed_breaker" in label
        or "bump_ahead" in label
    )


def _bbox_iou(box_a: List[int], box_b: List[int]) -> float:
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


def _bbox_area(box: List[int]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_containment_ratio(inner_box: List[int], outer_box: List[int]) -> float:
    ix1 = max(float(inner_box[0]), float(outer_box[0]))
    iy1 = max(float(inner_box[1]), float(outer_box[1]))
    ix2 = min(float(inner_box[2]), float(outer_box[2]))
    iy2 = min(float(inner_box[3]), float(outer_box[3]))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    inner_area = max(_bbox_area(inner_box), 1.0)
    return float(inter / inner_area)


def _is_plausible_compact_sign(det: Dict[str, Any]) -> bool:
    if str(det.get("class")) not in _COMPACT_SIGN_CLASSES:
        return True
    x1, y1, x2, y2 = [int(v) for v in det["bbox"][:4]]
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    area = w * h
    aspect = w / max(h, 1)
    return area >= 64 and 0.25 <= aspect <= 4.0


def _refine_compact_sign_detections(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    non_signs = [det for det in detections if str(det.get("class")) not in _COMPACT_SIGN_CLASSES]
    refined: List[Dict[str, Any]] = []

    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        cls_name = str(det.get("class"))
        if cls_name in _COMPACT_SIGN_CLASSES and _is_plausible_compact_sign(det):
            by_class.setdefault(cls_name, []).append(det)

    for cls_name, cls_dets in by_class.items():
        candidates = sorted(
            cls_dets,
            key=lambda item: (_bbox_area(item["bbox"]), -float(item["confidence"])),
        )
        selected: List[Dict[str, Any]] = []
        for det in candidates:
            det_area = _bbox_area(det["bbox"])
            suppress = False
            for kept in selected:
                kept_area = _bbox_area(kept["bbox"])
                if kept_area <= 0.0:
                    continue
                if (
                    _bbox_containment_ratio(kept["bbox"], det["bbox"]) >= 0.88
                    and det_area >= kept_area * 1.30
                    and float(det["confidence"]) <= float(kept["confidence"]) + 0.08
                ):
                    suppress = True
                    break
            if not suppress:
                selected.append(det)
        refined.extend(selected)

    refined.extend(non_signs)
    refined.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return refined


def _is_specific_sign_detection(det: Dict[str, Any]) -> bool:
    cls_name = str(det.get("class") or "")
    sign_label = str(det.get("sign_label") or "")
    if cls_name in {"stop_sign", "speed_limit"}:
        return True
    if cls_name != "traffic_sign":
        return False
    return sign_label not in {"", "traffic_sign", "sign"}


def _suppress_generic_sign_duplicates(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    specific_signs = [det for det in detections if _is_specific_sign_detection(det)]
    if not specific_signs:
        return detections

    refined: List[Dict[str, Any]] = []
    for det in detections:
        if str(det.get("class")) != "traffic_sign":
            refined.append(det)
            continue
        if _is_specific_sign_detection(det):
            refined.append(det)
            continue

        suppress = False
        for specific in specific_signs:
            if specific is det:
                continue
            if (
                _bbox_iou(det["bbox"], specific["bbox"]) >= 0.45
                or _bbox_containment_ratio(det["bbox"], specific["bbox"]) >= 0.70
            ) and float(det.get("confidence", 0.0)) <= float(specific.get("confidence", 0.0)) + 0.10:
                suppress = True
                break
        if not suppress:
            refined.append(det)
    return refined


def _nms_by_class(
    detections: List[Dict[str, Any]],
    iou_thr: float = 0.55,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        by_class.setdefault(str(det["class"]), []).append(det)

    for cls_name, cls_dets in by_class.items():
        remaining = sorted(cls_dets, key=lambda item: float(item["confidence"]), reverse=True)
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                other
                for other in remaining
                if _bbox_iou(best["bbox"], other["bbox"]) < iou_thr
            ]
    kept.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return kept


def _merge_preferred_sign_detections(
    preferred: List[Dict[str, Any]],
    fallback: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Keep the preferred sign detections (for us: DETR) and only add fallback
    boxes (for us: Detic) when they are not duplicates of the preferred ones.
    """

    merged = list(preferred)
    for det in fallback:
        if str(det.get("class")) not in _PRIMARY_SIGN_CLASSES:
            merged.append(det)
            continue
        suppress = False
        for kept in preferred:
            if str(kept.get("class")) not in _PRIMARY_SIGN_CLASSES:
                continue
            same_label = str(det.get("sign_label") or "") == str(kept.get("sign_label") or "")
            if same_label and _bbox_iou(det["bbox"], kept["bbox"]) >= 0.20:
                suppress = True
                break
            if (
                _bbox_iou(det["bbox"], kept["bbox"]) >= 0.45
                or _bbox_containment_ratio(det["bbox"], kept["bbox"]) >= 0.72
            ):
                suppress = True
                break
        if not suppress:
            merged.append(det)
    return merged


def _passes_context_filter(
    det: Dict[str, Any],
    frame_shape: Tuple[int, int],
    detections: List[Dict[str, Any]],
    *,
    bump_sign_present: bool = False,
) -> bool:
    """
    Lightweight geometry/context gating to reduce Detic false positives for
    long-tail road furniture and tiny vehicle-light detections.
    """

    H, W = frame_shape[:2]
    cls_name = str(det.get("class") or "")
    x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])[:4]]
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    area = float(w * h)
    aspect = float(w) / float(max(h, 1))
    vehicle_boxes = [
        other["bbox"]
        for other in detections
        if str(other.get("class")) in _VEHICLE_CLASSES
    ]

    if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}:
        return area >= 36.0 and y1 <= int(0.90 * H)

    if cls_name == "traffic_light":
        return area >= 24.0 and y1 <= int(0.92 * H) and h >= max(10, int(0.75 * w))

    # if cls_name in {"brake_light", "indicator_light"}:
    #     if area < 9.0 or area > 0.02 * float(W * H) or y1 < int(0.12 * H):
    #         return False
    #     return any(
    #         _bbox_containment_ratio(det["bbox"], veh_box) >= 0.85
    #         or _bbox_iou(det["bbox"], veh_box) >= 0.02
    #         for veh_box in vehicle_boxes
    #     )

    if cls_name == "traffic_pole":
        return area >= 90.0 and y2 >= int(0.18 * H) and h >= max(22, int(1.8 * w))

    if cls_name == "traffic_cone":
        return (
            area >= 32.0
            and y2 >= int(0.22 * H)
            and h >= max(10, int(0.65 * w))
            and aspect <= 1.45
        )

    if cls_name == "traffic_cylinder":
        return (
            area >= 40.0
            and y2 >= int(0.22 * H)
            and h >= max(12, int(0.45 * w))
            and 0.20 <= aspect <= 1.9
        )

    if cls_name == "dustbin":
        edge_visible = x1 <= int(0.02 * W) or x2 >= int(0.98 * W)
        min_area = 36.0 if edge_visible else 56.0
        max_aspect = 3.0 if edge_visible else 2.5
        return area >= min_area and y2 >= int(0.15 * H) and 0.18 <= aspect <= max_aspect

    if cls_name == "fire_hydrant":
        return area >= 70.0 and y2 >= int(0.25 * H) and 0.25 <= aspect <= 1.8 and h >= max(16, int(0.04 * H))

    return True


def _context_filter_detections(
    detections: List[Dict[str, Any]],
    frame_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    bump_sign_present = any(
        str(det.get("class") or "") in _PRIMARY_SIGN_CLASSES
        and float(det.get("confidence", 0.0)) >= 0.55
        and _is_bump_related_sign_label(det.get("sign_label"))
        for det in detections
    )
    refined = [
        det
        for det in detections
        if _passes_context_filter(
            det,
            frame_shape,
            detections,
            bump_sign_present=bump_sign_present,
        )
    ]
    refined.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return refined


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DetectionResult  — the canonical output unit
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DetectionResult:
    """Single detected object in one frame."""
    id: int
    cls: str            # canonical class name
    bbox: List[int]     # [x1, y1, x2, y2]
    confidence: float
    subclass: str = ""  # coarse production label (kept aligned with cls)

    # Populated by downstream modules; None until filled
    depth_m: Optional[float] = None
    track_id: Optional[int] = None
    position_3d: Optional[List[float]] = None   # [bx, by, bz]
    keypoints: Optional[List[List[float]]] = None  # [[x,y,conf], ...] for pedestrian pose
    orientation_rad: Optional[float] = None        # yaw angle from 3D detection
    bbox_3d: Optional[List[List[float]]] = None    # projected 3D bbox corners [[x,y], ...]
    dimensions_3d: Optional[List[float]] = None    # [h, w, l] in metres
    speed_limit_value: Optional[int] = None
    sign_label: Optional[str] = None
    signal_color: Optional[str] = None
    signal_shape: Optional[str] = None
    signal_state: Optional[str] = None
    subclass_source: Optional[str] = None
    source: Optional[str] = None
    raw_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "class": self.cls,
            "subclass": self.subclass or self.cls,
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "depth_m": round(self.depth_m, 3) if self.depth_m is not None else None,
            "track_id": self.track_id,
            "position_3d": [round(v, 3) for v in self.position_3d]
                           if self.position_3d is not None else None,
        }
        if self.keypoints is not None:
            d["keypoints"] = [[round(v, 2) for v in kp] for kp in self.keypoints]
        if self.orientation_rad is not None:
            d["orientation_rad"] = round(self.orientation_rad, 4)
        if self.bbox_3d is not None:
            d["bbox_3d"] = [[round(v, 1) for v in pt] for pt in self.bbox_3d]
        if self.dimensions_3d is not None:
            d["dimensions_3d"] = [round(v, 3) for v in self.dimensions_3d]
        if self.speed_limit_value is not None:
            d["speed_limit_value"] = int(self.speed_limit_value)
        if self.sign_label:
            d["sign_label"] = str(self.sign_label)
        if self.signal_color:
            d["signal_color"] = str(self.signal_color)
        if self.signal_shape:
            d["signal_shape"] = str(self.signal_shape)
        if self.signal_state:
            d["signal_state"] = str(self.signal_state)
        if self.subclass_source:
            d["subclass_source"] = str(self.subclass_source)
        if self.source:
            d["source"] = str(self.source)
        if self.raw_label:
            d["raw_label"] = str(self.raw_label)
        return d


@dataclasses.dataclass
class TemporalTrack:
    track_id: int
    cls: str
    subclass: str
    bbox: List[float]
    confidence: float
    last_frame_idx: int
    hits: int = 1
    sign_label: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Device resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(requested: str = "auto") -> str:
    """Return the best available torch device string."""
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
# 4.  ObjectDetector
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Primary detector for the production pipeline.

    Parameters
    ----------
    device : "auto" | "cuda" | "mps" | "cpu"
    detector_backend : "auto" | "detic" | "yolo"
    yolo_model : ultralytics model name or path  (default: auto)
    sign_backend : "auto" | "detic" | "yolo" | "faster_rcnn" | "detr" | "ensemble"
    """

    def __init__(
        self,
        device: str = "auto",
        detector_backend: str = "auto",
        yolo_model: str = "auto",
        sign_model: Optional[str] = "auto",
        sign_backend: str = "auto",
        detic_python: Optional[str] = "auto",
        detic_repo: Optional[str] = "auto",
        detic_config: Optional[str] = "auto",
        detic_weights: Optional[str] = "auto",
        detic_custom_vocabulary: Optional[str] = "auto",
        detic_vocabulary_file: Optional[str] = "auto",
        detic_min_confidence: float = 0.30,
        faster_rcnn_model: Optional[str] = "auto",
        faster_rcnn_labels: Optional[str] = "auto",
        faster_rcnn_min_confidence: float = 0.55,
        faster_rcnn_max_dim: int = 1000,
        detr_model: Optional[str] = "auto",
        detr_min_confidence: float = 0.55,
        car_subclass_model: Optional[str] = "auto",
        imgsz: int = _BASE_IMGSZ,
        output_min_confidence: float = _DEFAULT_OUTPUT_MIN_CONFIDENCE,
    ) -> None:
        self.device = _resolve_device(device)
        self._target_classes = set(_TARGET_CLASSES)
        self._imgsz = int(max(640, imgsz))
        self._output_min_confidence = float(max(0.0, min(1.0, output_min_confidence)))
        self._detector_backend = str(detector_backend or "auto").strip().lower()
        self._sign_backend = str(sign_backend or "auto").strip().lower()
        self._detic_python = str(detic_python or "auto")
        if self._detector_backend not in {"auto", "detic", "yolo", "hybrid"}:
            raise ValueError("detector_backend must be one of: auto, detic, yolo, hybrid")
        if self._sign_backend not in {"auto", "detic", "yolo", "faster_rcnn", "detr", "ensemble"}:
            raise ValueError("sign_backend must be one of: auto, detic, yolo, faster_rcnn, detr, ensemble")
        self._detic_repo_path = _resolve_optional_detic_repo_path(detic_repo)
        self._detic_config_path = _resolve_optional_detic_config_path(detic_config, self._detic_repo_path)
        self._detic_weights_path = _resolve_optional_detic_weights_path(detic_weights, self._detic_repo_path)
        self._detic_vocabulary = _parse_detic_vocabulary(detic_custom_vocabulary, detic_vocabulary_file)
        self._detic_min_confidence = float(max(0.0, min(1.0, detic_min_confidence)))
        self._detic_detector = None
        if self._detector_backend in {"auto", "detic", "hybrid"}:
            if self._detic_repo_path and self._detic_config_path and self._detic_weights_path:
                try:
                    from detic_scene_detector import DeticSceneDetector

                    detic_device = "cuda" if self.device == "cuda" else "cpu"
                    if self.device != detic_device:
                        print(
                            f"[ObjectDetector] Detic on device='{self.device}' is not supported here; using '{detic_device}' instead."
                        )
                    self._detic_detector = DeticSceneDetector(
                        repo_root=self._detic_repo_path,
                        config_file=self._detic_config_path,
                        weights_path=self._detic_weights_path,
                        python_exe=self._detic_python,
                        device=detic_device,
                        min_confidence=self._detic_min_confidence,
                        vocabulary=self._detic_vocabulary,
                    )
                    print(
                        "[ObjectDetector] Detic loaded "
                        f"({Path(self._detic_weights_path).name}) "
                        f"config={Path(self._detic_config_path).name} "
                        f"vocab={len(self._detic_vocabulary)} class prompts"
                    )
                except Exception as exc:
                    if self._detector_backend == "detic":
                        raise RuntimeError(
                            "detector_backend='detic' was requested, but the Detic backend could not be loaded."
                        ) from exc
                    print(
                        "[ObjectDetector] Warning: could not load the Detic backend. "
                        "Continuing without it. "
                        f"({type(exc).__name__}: {exc}) "
                        "If you have a separate Detic conda env, pass --detic-python /path/to/env/bin/python."
                    )
                    self._detic_detector = None
            elif self._detector_backend == "detic":
                raise RuntimeError(
                    "detector_backend='detic' was requested but the Detic repo/config/weights could not be resolved."
                )

        self._yolo_model_path = _resolve_model_path(
            yolo_model,
            ["yolo26x.pt", "yolo26l.pt", "yolo26m.pt", "yolo26s.pt", "yolo26n.pt"],
        )
        self._sign_model_path = _resolve_optional_sign_model_path(sign_model)
        self._yolo = None
        self._sign_yolo = None
        need_yolo = (
            self._detector_backend in {"auto", "yolo", "hybrid"}
            or self._sign_backend in {"auto", "yolo", "ensemble"}
            or self._detic_detector is None
        )
        if need_yolo:
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(self._yolo_model_path)
                print(f"[ObjectDetector] YOLO loaded ({self._yolo_model_path})  device='{self.device}'")
                print(f"[ObjectDetector] Active YOLO model: {Path(self._yolo_model_path).name}")
                if self._sign_model_path and self._sign_backend in {"auto", "yolo", "ensemble"}:
                    try:
                        self._sign_yolo = YOLO(self._sign_model_path)
                        print(f"[ObjectDetector] Sign model: {Path(self._sign_model_path).name}")
                        if Path(self._sign_model_path).name.lower() in {"trafic.pt", "traffic.pt"}:
                            print(
                                "[ObjectDetector] Note: trafic.pt uses non-U.S. label names; the pipeline maps its numeric/sign labels "
                                "into stop_sign / speed_limit / traffic_sign where possible."
                            )
                    except Exception as exc:
                        print(
                            "[ObjectDetector] Warning: could not load the dedicated sign model. "
                            f"Continuing with the general YOLO detector only. ({type(exc).__name__}: {exc})"
                        )
                        self._sign_yolo = None
                        self._sign_model_path = None
            except ImportError as exc:
                if self._detector_backend == "yolo" or self._detic_detector is None or self._sign_backend in {"yolo", "ensemble"}:
                    raise ImportError(
                        "ultralytics is required for the YOLO fallback/sign backend. Install: pip install ultralytics"
                    ) from exc
                print(
                    "[ObjectDetector] Warning: ultralytics is unavailable, so YOLO fallback/sign refinement will be skipped."
                )

        self._effective_detector_backend = self._resolve_effective_detector_backend()
        if self._detector_backend == "auto":
            print(
                "[ObjectDetector] General backend priority (auto): "
                f"YOLO + Detic fusion -> Detic -> YOLO. Active='{self._effective_detector_backend}'."
            )
        elif self._effective_detector_backend != self._detector_backend:
            print(
                "[ObjectDetector] Requested detector backend "
                f"'{self._detector_backend}' unavailable; falling back to '{self._effective_detector_backend}'."
            )

        self._faster_rcnn_sign_model_path = _resolve_optional_faster_rcnn_model_path(faster_rcnn_model)
        self._faster_rcnn_sign_labels_path = _resolve_optional_faster_rcnn_labels_path(faster_rcnn_labels)
        self._faster_rcnn_sign_detector = None
        self._faster_rcnn_min_confidence = float(max(0.0, min(1.0, faster_rcnn_min_confidence)))
        self._faster_rcnn_max_dim = int(max(128, faster_rcnn_max_dim))
        if self._sign_backend in {"auto", "faster_rcnn", "ensemble"}:
            if self._faster_rcnn_sign_model_path and self._faster_rcnn_sign_labels_path:
                try:
                    from faster_rcnn_sign_detector import FasterRCNNSignDetector

                    self._faster_rcnn_sign_detector = FasterRCNNSignDetector(
                        self._faster_rcnn_sign_model_path,
                        self._faster_rcnn_sign_labels_path,
                        min_confidence=self._faster_rcnn_min_confidence,
                        resize_max_dim=self._faster_rcnn_max_dim,
                    )
                    print(
                        "[ObjectDetector] Faster R-CNN sign model: "
                        f"{Path(self._faster_rcnn_sign_model_path).name}"
                    )
                    print(
                        "[ObjectDetector] Note: the referenced LISA Faster R-CNN repo code ships a 3-class "
                        "label map (pedestrianCrossing, signalAhead, stop), despite the README claiming more classes."
                    )
                except Exception as exc:
                    print(
                        "[ObjectDetector] Warning: could not load the Faster R-CNN sign backend. "
                        f"Continuing without it. ({type(exc).__name__}: {exc})"
                    )
                    self._faster_rcnn_sign_detector = None
            elif self._sign_backend == "faster_rcnn":
                print(
                    "[ObjectDetector] Warning: sign_backend='faster_rcnn' was requested but no "
                    "frozen_inference_graph.pb / classes.pbtxt pair was found."
                )

        self._detr_sign_model_path = _resolve_optional_detr_model_path(detr_model)
        self._raw_detr_artifact_path = None
        if self._detr_sign_model_path:
            detr_path = Path(self._detr_sign_model_path)
            if detr_path.exists() and not _is_transformers_detr_checkpoint_path(detr_path):
                self._raw_detr_artifact_path = str(detr_path.resolve())
        self._detr_sign_detector = None
        self._detr_min_confidence = float(max(0.0, min(1.0, detr_min_confidence)))
        if self._sign_backend in {"auto", "detr", "ensemble"} and self._detr_sign_model_path:
            try:
                from detr_sign_detector import DetrSignDetector

                detr_device = "cuda" if self.device == "cuda" else "cpu"
                if self.device != detr_device:
                    print(
                        f"[ObjectDetector] DETR on device='{self.device}' is not supported here; using '{detr_device}' instead."
                    )
                self._detr_sign_detector = DetrSignDetector(
                    self._detr_sign_model_path,
                    device=detr_device,
                    min_confidence=self._detr_min_confidence,
                )
                print(
                    "[ObjectDetector] DETR sign model: "
                    f"{Path(self._detr_sign_model_path).name}"
                )
                if self._raw_detr_artifact_path:
                    print(
                        "[ObjectDetector] DETR source resolved from local trained_weights artifacts: "
                        f"{self._raw_detr_artifact_path}"
                    )
            except Exception as exc:
                print(
                    "[ObjectDetector] Warning: could not load the DETR sign backend. "
                    f"Continuing without it. ({type(exc).__name__}: {exc})"
                )
                self._detr_sign_detector = None
        elif self._sign_backend == "detr":
            print(
                "[ObjectDetector] Warning: sign_backend='detr' was requested but no DETR checkpoint was found."
            )
        model_names = getattr(self._yolo, "names", {}) if self._yolo is not None else {}
        canonical_model_classes = {
            cls_name
            for raw_name in model_names.values()
            for cls_name, _ in [_canonicalize_detection_label(raw_name)]
            if cls_name is not None
        }
        sign_names = getattr(self._sign_yolo, "names", {}) if self._sign_yolo is not None else {}
        canonical_sign_classes = {
            cls_name
            for raw_name in (sign_names or {}).values()
            for cls_name, _ in [_canonicalize_sign_label(raw_name, self._sign_model_path)]
            if cls_name is not None
        }
        if self._yolo is not None and not (
            canonical_model_classes & {"traffic_sign", "stop_sign", "speed_limit"}
            or canonical_sign_classes & {"traffic_sign", "stop_sign", "speed_limit"}
        ):
            print(
                "[ObjectDetector] Warning: current YOLO weights do not advertise any recognizable road-sign labels. "
                "The pipeline will try local best.pt-style sign weights first and otherwise the HF TrafficSignDetection model."
            )

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

        self._effective_sign_backend = self._resolve_effective_sign_backend()
        if self._sign_backend == "auto":
            print(
                "[ObjectDetector] Sign backend priority (auto): "
                f"best.pt/YOLO -> DETR -> Faster R-CNN -> Detic. Active='{self._effective_sign_backend}'."
            )
        elif self._effective_sign_backend != self._sign_backend:
            print(
                "[ObjectDetector] Requested sign backend "
                f"'{self._sign_backend}' unavailable; falling back to '{self._effective_sign_backend}'."
            )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _has_dedicated_sign_yolo(self) -> bool:
        return self._sign_yolo is not None

    def _resolve_effective_detector_backend(self) -> str:
        available = {
            "detic": self._detic_detector is not None,
            "yolo": self._yolo is not None,
        }
        if self._detector_backend == "auto":
            if available["detic"] and available["yolo"]:
                return "hybrid"
            if available["detic"]:
                return "detic"
            if available["yolo"]:
                return "yolo"
            raise RuntimeError("No object-detection backend is available.")
        if self._detector_backend == "hybrid":
            if available["detic"] and available["yolo"]:
                return "hybrid"
            if available["detic"]:
                return "detic"
            if available["yolo"]:
                return "yolo"
            raise RuntimeError("No object-detection backend is available.")
        if available.get(self._detector_backend):
            return self._detector_backend
        if available["detic"]:
            return "detic"
        if available["yolo"]:
            return "yolo"
        raise RuntimeError("No object-detection backend is available.")

    def _resolve_effective_sign_backend(self) -> str:
        if self._sign_backend == "ensemble":
            return "ensemble"

        available = {
            "detr": self._detr_sign_detector is not None,
            "detic": self._detic_detector is not None,
            "faster_rcnn": self._faster_rcnn_sign_detector is not None,
            "yolo": self._has_dedicated_sign_yolo()
            or (self._effective_detector_backend != "detic" and self._yolo is not None),
        }

        fallback_order = ["yolo", "detr", "faster_rcnn", "detic"]
        if self._sign_backend == "auto":
            for backend in fallback_order:
                if available.get(backend):
                    return backend
            return "none" if self._effective_detector_backend == "detic" else "yolo"

        if available.get(self._sign_backend):
            return self._sign_backend

        if self._sign_backend == "detr":
            for backend in ("yolo", "faster_rcnn", "detic"):
                if available.get(backend):
                    return backend
        if self._sign_backend == "detic":
            for backend in ("faster_rcnn", "yolo"):
                if available.get(backend):
                    return backend
        if self._sign_backend == "faster_rcnn":
            for backend in ("yolo", "detic"):
                if available.get(backend):
                    return backend
        return "none" if self._effective_detector_backend == "detic" else "yolo"

    def backend_info(self) -> Dict[str, Any]:
        return {
            "detector_requested": self._detector_backend,
            "detector_active": self._effective_detector_backend,
            "sign_requested": self._sign_backend,
            "sign_active": self._effective_sign_backend,
            "detic_repo": self._detic_repo_path,
            "detic_config": self._detic_config_path,
            "detic_weights": self._detic_weights_path,
            "detic_python": self._detic_python,
            "detr_model": self._detr_sign_model_path,
            "faster_rcnn_model": self._faster_rcnn_sign_model_path,
            "sign_yolo_model": self._sign_model_path,
            "general_yolo_model": self._yolo_model_path,
            "output_min_confidence": self._output_min_confidence,
            "detic_min_confidence": self._detic_min_confidence,
            "detr_min_confidence": self._detr_min_confidence,
            "faster_rcnn_min_confidence": self._faster_rcnn_min_confidence,
        }

    def class_catalog(self) -> Dict[str, Any]:
        return {
            "target_classes": sorted(self._target_classes),
            "vehicle_classes": sorted(_VEHICLE_CLASSES),
            "sign_classes": sorted(_PRIMARY_SIGN_CLASSES),
            "sign_subclasses": sorted(
                {
                    "stop",
                    "yield",
                    "yield_ahead",
                    "merge",
                    "pedestrian_crossing",
                    "signal_ahead",
                    "keep_left",
                    "keep_right",
                    "do_not_enter",
                    "one_way",
                    "school_zone",
                    "speed_limit_15",
                    "speed_limit_20",
                    "speed_limit_25",
                    "speed_limit_30",
                    "speed_limit_35",
                    "speed_limit_40",
                    "speed_limit_45",
                    "speed_limit_50",
                    "speed_limit_55",
                    "speed_limit_60",
                    "speed_limit_65",
                    "speed_bump_warning",
                    "speed_hump_warning",
                    "bump_ahead",
                }
            ),
            "road_object_classes": sorted(_ROAD_OBJECT_CLASSES),
            "vehicle_subtypes": ["sedan", "suv", "hatchback", "pickup"],
            "detic_vocabulary": list(self._detic_vocabulary),
        }

    def _handle_detic_runtime_failure(self, exc: Exception) -> None:
        message = (
            "[ObjectDetector] Warning: Detic failed during runtime and will be disabled for the rest of this run. "
            f"({type(exc).__name__}: {exc})"
        )
        print(message)
        self._detic_detector = None
        self._effective_detector_backend = self._resolve_effective_detector_backend()
        self._effective_sign_backend = self._resolve_effective_sign_backend()

    def _decode_yolo_result(
        self,
        result: Any,
        *,
        x_offset: int = 0,
        y_offset: int = 0,
        restrict_classes: Optional[set] = None,
        source: str = "yolo",
        model_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        names = result.names if hasattr(result, "names") else {}
        model_name = Path(model_path).name.lower() if model_path else ""
        for box in result.boxes:
            cls_id = int(box.cls.item())
            raw_label = names.get(cls_id, str(cls_id))
            cls_name, speed_limit_value = _canonicalize_sign_label(raw_label, model_path)
            if cls_name is None or cls_name not in self._target_classes:
                continue
            if restrict_classes is not None and cls_name not in restrict_classes:
                continue
            conf = float(box.conf.item())
            if conf < _CONF_THRESHOLDS.get(cls_name, 0.40):
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            out.append(
                {
                    "bbox": [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset],
                    "class": cls_name,
                    "confidence": conf,
                    "source": source,
                    "raw_label": str(raw_label),
                    "speed_limit_value": speed_limit_value,
                    "sign_label": _normalize_sign_subclass_label(raw_label, cls_name)
                    if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}
                    else None,
                }
            )
        return out

    def _yolo_infer(
        self,
        image_bgr: np.ndarray,
        *,
        model: Any,
        imgsz: int,
    ) -> Any:
        return model(
            image_bgr,
            verbose=False,
            device=self.device,
            imgsz=int(max(640, imgsz)),
        )[0]

    def _tiled_sign_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        H, W = frame_bgr.shape[:2]
        tile_w = min(W, max(720, int(round(0.78 * W))))
        tile_h = min(H, max(480, int(round(0.90 * H))))
        if tile_w >= W and tile_h >= H:
            return []

        step_x = max(1, int(round(tile_w * (1.0 - _SIGN_TILE_OVERLAP))))
        step_y = max(1, int(round(tile_h * (1.0 - _SIGN_TILE_OVERLAP))))

        xs = list(range(0, max(W - tile_w + 1, 1), step_x))
        ys = list(range(0, max(H - tile_h + 1, 1), step_y))
        if not xs or xs[-1] != W - tile_w:
            xs.append(max(0, W - tile_w))
        if not ys or ys[-1] != H - tile_h:
            ys.append(max(0, H - tile_h))

        out: List[Dict[str, Any]] = []
        model = self._sign_yolo if self._sign_yolo is not None else self._yolo
        source = "yolo_sign" if self._sign_yolo is not None else "yolo"
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                crop = frame_bgr[y0 : y0 + tile_h, x0 : x0 + tile_w]
                if crop.size == 0:
                    continue
                result = self._yolo_infer(crop, model=model, imgsz=max(self._imgsz, _SIGN_TILE_IMGSZ))
                out.extend(
                    self._decode_yolo_result(
                        result,
                        x_offset=int(x0),
                        y_offset=int(y0),
                        restrict_classes=_SIGN_LIKE_CLASSES,
                        source=source,
                        model_path=self._sign_model_path if self._sign_yolo is not None else self._yolo_model_path,
                    )
                )
        return out

    def _faster_rcnn_sign_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._faster_rcnn_sign_detector is None:
            return []

        out: List[Dict[str, Any]] = []
        for det in self._faster_rcnn_sign_detector.detect(frame_bgr):
            raw_label = str(det.get("raw_label") or "")
            cls_name, speed_limit_value = _canonicalize_sign_label(
                raw_label,
                self._faster_rcnn_sign_model_path,
            )
            if cls_name is None or cls_name not in self._target_classes:
                continue
            conf = float(det.get("confidence", 0.0))
            min_conf = max(
                _CONF_THRESHOLDS.get(cls_name, 0.40),
                self._faster_rcnn_min_confidence,
            )
            if conf < min_conf:
                continue
            out.append(
                {
                    "bbox": [int(v) for v in det["bbox"][:4]],
                    "class": cls_name,
                    "confidence": conf,
                    "source": "faster_rcnn_sign",
                    "raw_label": raw_label,
                    "speed_limit_value": speed_limit_value,
                    "sign_label": _normalize_sign_subclass_label(raw_label, cls_name)
                    if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}
                    else None,
                }
            )
        return out

    def _detr_sign_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._detr_sign_detector is None:
            return []

        out: List[Dict[str, Any]] = []
        for det in self._detr_sign_detector.detect(frame_bgr):
            raw_label = str(det.get("raw_label") or "")
            cls_name, speed_limit_value = _canonicalize_sign_label(
                raw_label,
                self._detr_sign_model_path,
            )
            if cls_name is None or cls_name not in self._target_classes:
                continue
            conf = float(det.get("confidence", 0.0))
            min_conf = max(
                _CONF_THRESHOLDS.get(cls_name, 0.35),
                self._detr_min_confidence,
            )
            if conf < min_conf:
                continue
            out.append(
                {
                    "bbox": [int(v) for v in det["bbox"][:4]],
                    "class": cls_name,
                    "confidence": conf,
                    "source": "detr_sign",
                    "raw_label": raw_label,
                    "speed_limit_value": speed_limit_value,
                    "sign_label": _normalize_sign_subclass_label(raw_label, cls_name)
                    if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}
                    else None,
                }
            )
        return out

    def _detic_frame_detections(
        self,
        frame_bgr: np.ndarray,
        *,
        restrict_classes: Optional[set] = None,
        include_tiled: bool = True,
    ) -> List[Dict[str, Any]]:
        if self._detic_detector is None:
            return []

        out = self._decode_detic_detections(
            self._detic_detector.detect(frame_bgr),
            source="detic",
            restrict_classes=restrict_classes,
        )
        if include_tiled:
            out.extend(self._detic_tiled_detections(frame_bgr, restrict_classes=restrict_classes))
            out.extend(self._detic_zoomed_road_object_detections(frame_bgr, restrict_classes=restrict_classes))
        return out

    def _detic_sign_fallback_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        return self._detic_frame_detections(
            frame_bgr,
            restrict_classes=set(_PRIMARY_SIGN_CLASSES),
            include_tiled=True,
        )

    def _dedicated_sign_ensemble_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Preferred sign stack for production:
        dedicated sign YOLO + DETR + Faster R-CNN, without Detic unless it is
        explicitly selected as the sign backend.
        """
        preferred = self._detr_sign_detections(frame_bgr)
        preferred = _merge_preferred_sign_detections(preferred, self._tiled_sign_detections(frame_bgr))
        preferred = _merge_preferred_sign_detections(preferred, self._faster_rcnn_sign_detections(frame_bgr))
        return preferred

    def _supplemental_sign_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._effective_sign_backend == "none":
            return []
        if self._effective_sign_backend == "yolo":
            return self._tiled_sign_detections(frame_bgr)
        if self._effective_sign_backend == "detic":
            return self._detic_sign_fallback_detections(frame_bgr)
        if self._effective_sign_backend == "faster_rcnn":
            return self._faster_rcnn_sign_detections(frame_bgr)
        if self._effective_sign_backend == "detr":
            return self._detr_sign_detections(frame_bgr)
        if self._effective_sign_backend == "ensemble":
            return self._dedicated_sign_ensemble_detections(frame_bgr)
        return []

    def _decode_detic_detections(
        self,
        detic_detections: List[Dict[str, Any]],
        *,
        x_offset: int = 0,
        y_offset: int = 0,
        source: str = "detic",
        restrict_classes: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for det in detic_detections:
            raw_label = str(det.get("raw_label") or "")
            cls_name, speed_limit_value = _canonicalize_detection_label(raw_label)
            if cls_name is None or cls_name not in self._target_classes:
                continue
            if restrict_classes is not None and cls_name not in restrict_classes:
                continue
            conf = float(det.get("confidence", 0.0))
            min_conf = max(_CONF_THRESHOLDS.get(cls_name, 0.25), self._detic_min_confidence)
            if conf < min_conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in det["bbox"][:4]]
            out.append(
                {
                    "bbox": [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset],
                    "class": cls_name,
                    "confidence": conf,
                    "source": source,
                    "raw_label": raw_label,
                    "speed_limit_value": speed_limit_value,
                    "sign_label": _normalize_sign_subclass_label(raw_label, cls_name)
                    if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}
                    else None,
                }
            )
        return out

    def _detic_tiled_detections(
        self,
        frame_bgr: np.ndarray,
        *,
        restrict_classes: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        if self._detic_detector is None:
            return []

        H, W = frame_bgr.shape[:2]
        tile_w = min(W, max(_DETIC_TILE_MIN_SIDE, int(round(0.65 * W))))
        tile_h = min(H, max(_DETIC_TILE_MIN_SIDE, int(round(0.82 * H))))
        if tile_w >= W and tile_h >= H:
            return []

        step_x = max(1, int(round(tile_w * (1.0 - _DETIC_TILE_OVERLAP))))
        step_y = max(1, int(round(tile_h * (1.0 - _DETIC_TILE_OVERLAP))))

        xs = list(range(0, max(W - tile_w + 1, 1), step_x))
        ys = list(range(0, max(H - tile_h + 1, 1), step_y))
        if not xs or xs[-1] != W - tile_w:
            xs.append(max(0, W - tile_w))
        if not ys or ys[-1] != H - tile_h:
            ys.append(max(0, H - tile_h))

        out: List[Dict[str, Any]] = []
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                crop = frame_bgr[y0 : y0 + tile_h, x0 : x0 + tile_w]
                if crop.size == 0:
                    continue
                out.extend(
                    self._decode_detic_detections(
                        self._detic_detector.detect(crop),
                        x_offset=int(x0),
                        y_offset=int(y0),
                        source="detic_tile",
                        restrict_classes=(
                            _DETIC_TILED_CLASSES
                            if restrict_classes is None
                            else set(_DETIC_TILED_CLASSES).intersection(set(restrict_classes))
                        ),
                    )
                )
        return out

    def _detic_zoomed_road_object_detections(
        self,
        frame_bgr: np.ndarray,
        *,
        restrict_classes: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        if self._detic_detector is None:
            return []

        zoom_classes = set(_ROAD_OBJECT_ZOOM_CLASSES)
        if restrict_classes is not None:
            zoom_classes &= set(restrict_classes)
        if not zoom_classes:
            return []

        H, W = frame_bgr.shape[:2]
        y_origin = max(0, min(H - 1, int(round(_ROAD_OBJECT_ROI_TOP * H))))
        roi = frame_bgr[y_origin:H, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 160 or roi_w < 240:
            return []

        out = self._decode_detic_detections(
            self._detic_detector.detect(roi),
            x_offset=0,
            y_offset=int(y_origin),
            source="detic_road_zoom",
            restrict_classes=zoom_classes,
        )

        tile_w = min(roi_w, max(_ROAD_OBJECT_TILE_MIN_W, int(round(0.46 * roi_w))))
        tile_h = min(roi_h, max(_ROAD_OBJECT_TILE_MIN_H, int(round(0.72 * roi_h))))
        if tile_w >= roi_w and tile_h >= roi_h:
            return out

        step_x = max(1, int(round(tile_w * (1.0 - _ROAD_OBJECT_TILE_OVERLAP))))
        step_y = max(1, int(round(tile_h * (1.0 - _ROAD_OBJECT_TILE_OVERLAP))))

        xs = list(range(0, max(roi_w - tile_w + 1, 1), step_x))
        ys = list(range(0, max(roi_h - tile_h + 1, 1), step_y))
        if not xs or xs[-1] != roi_w - tile_w:
            xs.append(max(0, roi_w - tile_w))
        if not ys or ys[-1] != roi_h - tile_h:
            ys.append(max(0, roi_h - tile_h))

        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                crop = roi[y0 : y0 + tile_h, x0 : x0 + tile_w]
                if crop.size == 0:
                    continue
                out.extend(
                    self._decode_detic_detections(
                        self._detic_detector.detect(crop),
                        x_offset=int(x0),
                        y_offset=int(y_origin + y0),
                        source="detic_road_zoom_tile",
                        restrict_classes=zoom_classes,
                    )
                )

        return out

    def _base_detic_detections(
        self,
        frame_bgr: np.ndarray,
        *,
        restrict_classes: Optional[set] = None,
        fallback_on_yolo: bool = False,
    ) -> List[Dict[str, Any]]:
        if self._detic_detector is None:
            return []
        try:
            base_restrict_classes = restrict_classes
            if base_restrict_classes is None and self._effective_sign_backend in {"detr", "faster_rcnn"}:
                base_restrict_classes = set(self._target_classes) - set(_PRIMARY_SIGN_CLASSES)
            return self._detic_frame_detections(
                frame_bgr,
                restrict_classes=base_restrict_classes,
                include_tiled=True,
            )
        except Exception as exc:
            if self._detector_backend == "detic" and self._yolo is None:
                raise RuntimeError("Detic failed during runtime and no fallback detector is available.") from exc
            self._handle_detic_runtime_failure(exc)
            if fallback_on_yolo and self._effective_detector_backend == "yolo":
                return self._yolo_detections(frame_bgr)
            return []

    def _detic_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        out = self._base_detic_detections(frame_bgr, fallback_on_yolo=True)
        out.extend(self._supplemental_sign_detections(frame_bgr))
        return out

    def _base_yolo_detections(
        self,
        frame_bgr: np.ndarray,
        *,
        restrict_classes: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Run the general YOLO detector and normalize its detections."""
        if self._yolo is None:
            return []
        result = self._yolo_infer(frame_bgr, model=self._yolo, imgsz=self._imgsz)
        base_restrict_classes = restrict_classes
        if base_restrict_classes is None and self._effective_sign_backend in {"detr", "detic", "faster_rcnn", "none"}:
            base_restrict_classes = set(self._target_classes) - set(_SIGN_LIKE_CLASSES)
        return self._decode_yolo_result(
            result,
            source="yolo",
            model_path=self._yolo_model_path,
            restrict_classes=base_restrict_classes,
        )

    def _yolo_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO and normalize detections into canonical class labels."""
        out = self._base_yolo_detections(frame_bgr)
        out.extend(self._supplemental_sign_detections(frame_bgr))
        return out

    def _hybrid_detections(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Production-oriented fusion:
        YOLO anchors motion-critical actors while Detic fills long-tail road
        furniture coverage and sign fallback runs exactly once.
        """
        yolo_general = self._base_yolo_detections(
            frame_bgr,
            restrict_classes=set(self._target_classes) - set(_PRIMARY_SIGN_CLASSES),
        )
        detic_support = self._base_detic_detections(
            frame_bgr,
            restrict_classes=set(self._target_classes) - set(_MOTION_CRITICAL_CLASSES) - set(_PRIMARY_SIGN_CLASSES),
        )
        out: List[Dict[str, Any]] = []
        out.extend(yolo_general)
        out.extend(detic_support)
        out.extend(self._supplemental_sign_detections(frame_bgr))
        return out

    def _apply_vehicle_subclassification(
        self,
        frame_bgr: np.ndarray,
        detections: List[DetectionResult],
    ) -> None:
        car_indices = [
            idx
            for idx, det in enumerate(detections)
            if det.cls == "car"
        ]
        if not car_indices:
            return
        predictions = []
        if self._vehicle_subclassifier is not None:
            predictions = self._vehicle_subclassifier.classify_car_boxes(
                frame_bgr,
                [detections[idx].bbox for idx in car_indices],
            )
        else:
            predictions = [None] * len(car_indices)

        for idx, pred in zip(car_indices, predictions):
            det = detections[idx]
            detic_subclass = _normalize_vehicle_subclass_label(det.raw_label or "")
            model_subclass = None
            model_conf = 0.0
            if (
                pred is not None
                and pred.subtype in {"sedan", "suv", "hatchback", "pickup"}
                and float(pred.confidence) >= _VEHICLE_SUBCLASS_MIN_CONFIDENCE
            ):
                model_subclass = pred.subtype
                model_conf = float(pred.confidence)

            if model_subclass and detic_subclass:
                if model_subclass == detic_subclass:
                    detections[idx].subclass = model_subclass
                    detections[idx].subclass_source = "car_model+detic"
                elif model_conf >= 0.68:
                    detections[idx].subclass = model_subclass
                    detections[idx].subclass_source = "car_model_over_detic"
                else:
                    detections[idx].subclass = detic_subclass
                    detections[idx].subclass_source = "detic_over_car_model"
            elif model_subclass:
                detections[idx].subclass = model_subclass
                detections[idx].subclass_source = "car_model"
            elif detic_subclass in {"sedan", "suv", "hatchback", "pickup"}:
                detections[idx].subclass = detic_subclass
                detections[idx].subclass_source = "detic_label"
            else:
                detections[idx].subclass = "car"
                detections[idx].subclass_source = detections[idx].subclass_source or "coarse_class"

    @staticmethod
    def _clamp_bbox(x1, y1, x2, y2, W, H):
        return (
            max(0, min(x1, W - 1)),
            max(0, min(y1, H - 1)),
            max(0, min(x2, W - 1)),
            max(0, min(y2, H - 1)),
        )

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[DetectionResult]:
        """
        Detect objects in a single BGR frame.

        Parameters
        ----------
        frame_bgr : H×W×3 uint8 BGR image

        Returns
        -------
        List of DetectionResult (sorted by confidence desc)
        """
        H, W = frame_bgr.shape[:2]
        if self._effective_detector_backend == "hybrid":
            raw = _nms_by_class(self._hybrid_detections(frame_bgr))
        elif self._effective_detector_backend == "detic":
            raw = _nms_by_class(self._detic_detections(frame_bgr))
        else:
            raw = _nms_by_class(self._yolo_detections(frame_bgr))
        raw = _refine_compact_sign_detections(raw)
        raw = _suppress_generic_sign_duplicates(raw)
        raw = _nms_by_class(raw)
        raw = _context_filter_detections(raw, (H, W))

        detections: List[DetectionResult] = []
        for idx, det in enumerate(raw):
            x1, y1, x2, y2 = det["bbox"]
            x1, y1, x2, y2 = self._clamp_bbox(x1, y1, x2, y2, W, H)
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = [x1, y1, x2, y2]
            cls_name = str(det["class"])
            tl_signal = _parse_traffic_light_signal_label(det.get("raw_label")) if cls_name == "traffic_light" else None
            subclass = _derive_subclass_label(
                cls_name,
                det.get("raw_label"),
                bbox,
                sign_label=det.get("sign_label"),
            )
            detections.append(DetectionResult(
                id=idx,
                cls=cls_name,
                bbox=bbox,
                confidence=float(det["confidence"]),
                subclass=subclass,
                speed_limit_value=det.get("speed_limit_value"),
                sign_label=det.get("sign_label"),
                signal_color=(tl_signal or {}).get("signal_color"),
                signal_shape=(tl_signal or {}).get("signal_shape"),
                signal_state=(tl_signal or {}).get("signal_state"),
                subclass_source="label_hint" if cls_name != "car" and subclass and subclass != cls_name else None,
                source=det.get("source"),
                raw_label=det.get("raw_label"),
            ))

        # Stable sort: highest confidence first
        detections.sort(key=lambda d: d.confidence, reverse=True)

        self._apply_vehicle_subclassification(frame_bgr, detections)

        detections = [
            det
            for det in detections
            if det.confidence >= min(
                self._output_min_confidence,
                _OUTPUT_CLASS_MIN_CONFIDENCE.get(det.cls, self._output_min_confidence),
            )
        ]

        # Re-assign sequential IDs after sorting
        for i, d in enumerate(detections):
            d.id = i

        return detections

    def draw(
        self,
        frame_bgr: np.ndarray,
        detections: List[DetectionResult],
        show_track_id: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes + labels on a copy of frame_bgr.

        Label format:  [id] class  conf  dist?  track?
        """
        vis = frame_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _DRAW_COLORS.get(det.cls, (200, 200, 200))

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Ground-contact dot (bottom-centre)
            cx = (x1 + x2) // 2
            cv2.circle(vis, (cx, y2), 4, color, -1)

            # Label — show subclass when it differs from class
            display_cls = det.subclass if (det.subclass and det.subclass != det.cls) else det.cls
            if det.sign_label:
                display_cls = _format_sign_label(det.sign_label)
            elif det.cls == "speed_limit" and det.speed_limit_value is not None:
                display_cls = f"speed_{int(det.speed_limit_value)}"
            parts = [f"[{det.id}]", display_cls, f"{det.confidence:.2f}"]
            if det.depth_m is not None:
                parts.append(f"{det.depth_m:.1f}m")
            if show_track_id and det.track_id is not None:
                parts.append(f"T{det.track_id}")
            label = "  ".join(parts)

            (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(y1 - 4, th + bl + 2)
            cv2.rectangle(
                vis,
                (x1, label_y - th - bl - 2),
                (x1 + tw + 4, label_y + 2),
                color, cv2.FILLED,
            )
            cv2.putText(
                vis, label, (x1 + 2, label_y),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )

        return vis


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Frame-level JSON structure
# ─────────────────────────────────────────────────────────────────────────────

def build_frame_record(
    frame_idx: int,
    timestamp_s: float,
    detections: List[DetectionResult],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the JSON record for a single frame.

    Schema
    ------
    {
      "frame_idx":  int,
      "timestamp_s": float,
      "detections": [ DetectionResult.to_dict(), ... ],
      "metadata":   { ... }   ← optional, e.g. depth scale, calib info
    }
    """
    return {
        "frame_idx":   frame_idx,
        "timestamp_s": round(timestamp_s, 4),
        "detections":  [d.to_dict() for d in detections],
        "metadata":    metadata or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Safer video output helpers for macOS / VS Code
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
    """
    Write to temporary MJPG AVI first, then optionally re-encode to MP4.

    This avoids the fragile codec negotiation path that often fails on macOS
    OpenCV builds when writing MP4 directly.
    """

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
# 7.  DetectionPipeline  — end-to-end video processor
# ─────────────────────────────────────────────────────────────────────────────

class DetectionPipeline:
    """
    Full pipeline:  video file  →  annotated output video  +  JSON detections.

    Downstream modules (depth, tracking, 3-D lifting) can be injected via the
    ``frame_hook`` callable which receives the raw frame and detections and may
    mutate detection fields (depth_m, track_id, position_3d) in-place before
    the frame is written to the output video.

    Parameters
    ----------
    detector : ObjectDetector instance  (created with defaults if None)
    device : device string passed to ObjectDetector when auto-creating
    frame_hook : optional callable(frame_bgr, List[DetectionResult]) → None
                 Called after detection, before visualisation.
                 Use to integrate depth / tracking / 3-D lifting.
    """

    def __init__(
        self,
        detector: Optional[ObjectDetector] = None,
        device: str = "auto",
        frame_hook = None,
    ) -> None:
        self.detector = detector or ObjectDetector(device=device)
        self.frame_hook = frame_hook
        self._tracks: List[TemporalTrack] = []
        self._next_track_id: int = 1

    @staticmethod
    def _track_match_iou(det: DetectionResult) -> float:
        if det.cls in {"brake_light", "indicator_light"}:
            return 0.12
        if det.cls in {"traffic_cone", "traffic_cylinder", "traffic_pole", "fire_hydrant", "dustbin"}:
            return 0.10
        if det.cls in _ROAD_OBJECT_CLASSES or det.cls in _PRIMARY_SIGN_CLASSES or det.cls == "traffic_light":
            return 0.22
        return 0.35

    @staticmethod
    def _required_confirmation_hits(det: DetectionResult) -> int:
        if det.cls in {"brake_light", "indicator_light"}:
            return 3
        if det.cls in _STATIC_ROAD_OBJECT_CLASSES:
            return 1
        if det.cls in _ROAD_OBJECT_CLASSES or det.cls in _PRIMARY_SIGN_CLASSES or det.cls == "traffic_light":
            return 2
        return 1

    def _temporal_stabilize(
        self,
        detections: List[DetectionResult],
        frame_idx: int,
    ) -> List[DetectionResult]:
        alpha = float(_TEMPORAL_SMOOTH_ALPHA)
        recent_tracks = [
            track
            for track in self._tracks
            if (frame_idx - track.last_frame_idx) <= _TEMPORAL_TRACK_MAX_AGE
        ]
        updated_tracks: List[TemporalTrack] = []
        used_track_ids: set = set()
        stabilized: List[DetectionResult] = []

        for det in detections:
            best_track: Optional[TemporalTrack] = None
            best_iou = 0.0
            for track in recent_tracks:
                if track.track_id in used_track_ids:
                    continue
                if track.cls != det.cls:
                    continue
                if det.cls in _PRIMARY_SIGN_CLASSES:
                    track_label = str(track.sign_label or "")
                    det_label = str(det.sign_label or "")
                    if track_label and det_label and track_label != det_label:
                        continue
                iou = _bbox_iou([int(v) for v in track.bbox], det.bbox)
                if iou >= self._track_match_iou(det) and iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is None:
                det.track_id = self._next_track_id
                updated_tracks.append(
                    TemporalTrack(
                        track_id=self._next_track_id,
                        cls=det.cls,
                        subclass=det.subclass or det.cls,
                        bbox=[float(v) for v in det.bbox],
                        confidence=float(det.confidence),
                        last_frame_idx=frame_idx,
                        hits=1,
                        sign_label=det.sign_label,
                    )
                )
                self._next_track_id += 1
            else:
                used_track_ids.add(best_track.track_id)
                smoothed_bbox: List[int] = []
                for prev_v, cur_v in zip(best_track.bbox, det.bbox):
                    smoothed_bbox.append(int(round((1.0 - alpha) * float(prev_v) + alpha * float(cur_v))))
                det.bbox = smoothed_bbox
                det.confidence = float((1.0 - alpha) * float(best_track.confidence) + alpha * float(det.confidence))
                det.track_id = best_track.track_id
                best_track.bbox = [float(v) for v in det.bbox]
                best_track.confidence = float(det.confidence)
                best_track.last_frame_idx = frame_idx
                best_track.hits += 1
                if det.subclass and det.subclass != det.cls:
                    best_track.subclass = det.subclass
                det.subclass = best_track.subclass or det.subclass
                if det.sign_label:
                    best_track.sign_label = det.sign_label
                updated_tracks.append(best_track)

            track_for_det = updated_tracks[-1]
            required_hits = self._required_confirmation_hits(det)
            if det.confidence < self.detector._output_min_confidence:
                continue
            if track_for_det.hits < required_hits and det.confidence < 0.90:
                continue
            stabilized.append(det)

        for track in recent_tracks:
            if track.track_id in used_track_ids:
                continue
            if (frame_idx - track.last_frame_idx) <= _TEMPORAL_TRACK_MAX_AGE:
                updated_tracks.append(track)

        self._tracks = updated_tracks
        stabilized.sort(key=lambda d: d.confidence, reverse=True)
        for idx, det in enumerate(stabilized):
            det.id = idx
        return stabilized

    # ── video utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _open_writer(path: Path, fps: float, W: int, H: int) -> "SafeVideoWriter":
        return SafeVideoWriter(path, fps, W, H, vscode_compatible=True)

    @staticmethod
    def _draw_hud(
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        n_dets: int,
        total_dets: int,
    ) -> None:
        """Overlay a compact HUD in the top-left corner (in-place)."""
        lines = [
            f"Frame : {frame_idx}",
            f"Proc  : {fps:5.1f} fps",
            f"This  : {n_dets} object(s)",
            f"Total : {total_dets}",
        ]
        x0, y0, lh = 8, 22, 22
        panel_w, panel_h = 200, lh * len(lines) + 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 18),
                      (x0 + panel_w, y0 + panel_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x0, y0 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                        (210, 230, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_legend(frame: np.ndarray) -> None:
        """Draw a class-colour legend strip at the bottom (in-place)."""
        H, W = frame.shape[:2]
        strip_y = H - 28
        cv2.rectangle(frame, (0, strip_y - 4), (W, H), (20, 20, 20), cv2.FILLED)
        lx = 8
        for cls_name, bgr in _DRAW_COLORS.items():
            label = cls_name.replace("_", " ")
            cv2.circle(frame, (lx + 7, strip_y + 10), 6, bgr, -1)
            cv2.putText(frame, label, (lx + 18, strip_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, bgr, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            lx += tw + 30

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        video_path: str,
        out_video: str = "renders/detection_output.mp4",
        out_json: str = "renders/detections.json",
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a video file end-to-end.

        Parameters
        ----------
        video_path : path to input .mp4 (or any OpenCV-readable format)
        out_video : path for the annotated output video
        out_json : path for the per-frame JSON detections file
        max_frames : stop after this many *processed* frames  (None = all)
        frame_skip : process every Nth source frame  (1 = every frame)
        metadata : dict stored in every frame record's "metadata" field

        Returns
        -------
        List of per-frame dicts (same structure written to out_json)
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {src}")

        out_video_path = Path(out_video)
        out_json_path = Path(out_json)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = src_fps / max(frame_skip, 1)

        print(f"\n{'='*64}")
        print(f"  DetectionPipeline.run()")
        print(f"  Input  : {src}  ({src_W}×{src_H} @ {src_fps:.1f} fps, ~{src_total} fr.)")
        print(f"  Output : {out_video_path}")
        print(f"  JSON   : {out_json_path}")
        print(f"  Skip   : every {frame_skip} frame(s)  →  ~{out_fps:.1f} fps output")
        print(f"{'='*64}\n")

        writer = self._open_writer(out_video_path, out_fps, src_W, src_H)
        final_video = out_video_path

        all_records: List[Dict[str, Any]] = []
        total_dets = 0
        written = 0
        src_idx = 0           # source frame counter
        t_start = time.time()
        fps_window: List[float] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Frame-skip logic
                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                ts = src_idx / src_fps
                t0 = time.time()

                # ── Detect ──────────────────────────────────────────────────
                detections = self.detector.detect(frame)
                detections = self._temporal_stabilize(detections, written)

                # ── Downstream hook (depth / tracking / 3-D) ────────────────
                if self.frame_hook is not None:
                    self.frame_hook(frame, detections)

                # ── JSON record ─────────────────────────────────────────────
                record = build_frame_record(written, ts, detections, metadata)
                all_records.append(record)
                total_dets += len(detections)

                # ── Visualise ────────────────────────────────────────────────
                vis = self.detector.draw(frame, detections)
                self._draw_legend(vis)

                # Rolling FPS
                fps_window.append(time.time() - t0)
                if len(fps_window) > 30:
                    fps_window.pop(0)
                proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

                self._draw_hud(vis, written, proc_fps, len(detections), total_dets)
                writer.write(vis)

                # Progress
                written += 1
                if written % 30 == 0 or written == 1:
                    elapsed = time.time() - t_start
                    pct = src_idx / src_total * 100 if src_total > 0 else 0
                    print(f"  frame {src_idx:5d} / ~{src_total}  "
                          f"({pct:5.1f}%)  "
                          f"{proc_fps:5.1f} fps  "
                          f"dets={len(detections):2d}  "
                          f"total={total_dets}",
                          end="\r", flush=True)

                src_idx += 1

                if max_frames is not None and written >= max_frames:
                    print(f"\n  Reached max_frames={max_frames}  — stopping.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")

        finally:
            cap.release()
            final_video = writer.close()

        # ── Write JSON ───────────────────────────────────────────────────────
        with open(out_json_path, "w") as f:
            json.dump(
                {
                    "source": str(src),
                    "backend_info": self.detector.backend_info(),
                    "class_catalog": self.detector.class_catalog(),
                    "frames_written": written,
                    "total_detections": total_dets,
                    "final_video": str(final_video),
                    "frames": all_records,
                },
                f, indent=2,
            )

        elapsed_total = time.time() - t_start
        print(f"\n\n{'='*64}")
        print(f"  Done.")
        print(f"  Frames processed : {written}")
        print(f"  Detections total : {total_dets}")
        print(f"  Wall time        : {elapsed_total:.1f}s  "
              f"({written / max(elapsed_total, 1e-6):.1f} fps avg)")
        print(f"  Output video     : {final_video}")
        print(f"  JSON output      : {out_json_path}")
        print(f"{'='*64}")

        return all_records

    def run_on_frames(
        self,
        frame_dir: str,
        out_video: str = "renders/detection_output.mp4",
        out_json: str = "renders/detections.json",
        fps: float = 15.0,
        max_frames: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a directory of JPEG frames instead of a video file.

        Parameters
        ----------
        frame_dir : directory containing frame_XXXXX.jpg files
        fps : frame-rate for the output video
        """
        frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
        if not frame_paths:
            frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG frames found in {frame_dir}")

        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        # Determine frame size from first image
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        H, W = first.shape[:2]

        out_video_path = Path(out_video)
        out_json_path = Path(out_json)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)

        writer = self._open_writer(out_video_path, fps, W, H)
        final_video = out_video_path

        all_records: List[Dict[str, Any]] = []
        total_dets = 0
        t_start = time.time()
        fps_window: List[float] = []

        print(f"\n[DetectionPipeline] Processing {len(frame_paths)} frames from {frame_dir}")

        for idx, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            ts = idx / fps
            t0 = time.time()

            detections = self.detector.detect(frame)
            detections = self._temporal_stabilize(detections, idx)

            if self.frame_hook is not None:
                self.frame_hook(frame, detections)

            record = build_frame_record(idx, ts, detections, metadata)
            all_records.append(record)
            total_dets += len(detections)

            vis = self.detector.draw(frame, detections)
            self._draw_legend(vis)

            fps_window.append(time.time() - t0)
            if len(fps_window) > 30:
                fps_window.pop(0)
            proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

            self._draw_hud(vis, idx, proc_fps, len(detections), total_dets)
            writer.write(vis)

            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1:4d}/{len(frame_paths)}]  "
                      f"{proc_fps:5.1f} fps  dets={len(detections)}  "
                      f"total={total_dets}",
                      end="\r", flush=True)

        final_video = writer.close()

        with open(out_json_path, "w") as f:
            json.dump(
                {
                    "source": str(frame_dir),
                    "backend_info": self.detector.backend_info(),
                    "class_catalog": self.detector.class_catalog(),
                    "frames_written": len(all_records),
                    "total_detections": total_dets,
                    "final_video": str(final_video),
                    "frames": all_records,
                },
                f, indent=2,
            )

        elapsed = time.time() - t_start
        print(f"\n\n[DetectionPipeline] Finished  {len(all_records)} frames  "
              f"{elapsed:.1f}s  →  {final_video}")
        return all_records


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Convenience singleton wrappers  (plug-and-play for other modules)
# ─────────────────────────────────────────────────────────────────────────────

_detector_singleton: Optional[ObjectDetector] = None


def detect_objects(
    frame_bgr: np.ndarray,
    device: str = "auto",
) -> List[DetectionResult]:
    """
    Module-level singleton wrapper.
    Initialises ObjectDetector on first call, reuses it thereafter.

    Returns
    -------
    List[DetectionResult]  — plug into depth / tracking modules directly.
    """
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = ObjectDetector(device=device)
    return _detector_singleton.detect(frame_bgr)


def detections_to_json(detections: List[DetectionResult]) -> str:
    """Serialize a detection list to a compact JSON string."""
    return json.dumps([d.to_dict() for d in detections], indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  __main__  — CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous driving object detection pipeline"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video",  type=str, help="Input video path (.mp4)")
    source_group.add_argument("--frames", type=str, help="Directory of JPEG frames")

    parser.add_argument("--scene",       default=None,
                        help="Optional explicit scene id (e.g. scene1); otherwise inferred from input path")
    parser.add_argument("--out-video",   default=None,
                        help="Annotated output video path; defaults to output/<scene>/detections/detection_output.mp4")
    parser.add_argument("--out-json",    default=None,
                        help="Detection JSON path; defaults to output/<scene>/detections/detections.json")
    parser.add_argument("--device",      default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-frames",  type=int, default=None)
    parser.add_argument("--frame-skip",  type=int, default=1,
                        help="Process every Nth frame (video mode only)")
    parser.add_argument("--fps",         type=float, default=15.0,
                        help="Output FPS (frames mode only)")
    parser.add_argument("--detector-backend", default="auto",
                        choices=["auto", "detic", "yolo", "hybrid"],
                        help="General object detector. 'auto' prefers YOLO+Detic fusion when both are available, then falls back to Detic or YOLO alone.")
    parser.add_argument("--yolo-model",  default="auto",
                        help="YOLO model name or path. 'auto' resolves to YOLO26 weights/models only.")
    parser.add_argument("--sign-model",  default="auto",
                        help="Optional dedicated sign model. 'auto' prefers local best.pt-style checkpoints and otherwise uses the HF TrafficSignDetection weights for tiled sign detection.")
    parser.add_argument("--sign-backend", default="auto",
                        choices=["auto", "detic", "yolo", "faster_rcnn", "detr", "ensemble"],
                        help="Traffic-sign refinement backend. 'auto' prefers DETR, then Detic fallback, then Faster R-CNN, then dedicated sign YOLO when available.")
    parser.add_argument("--detic-repo", default="auto",
                        help="Path to the facebookresearch/Detic repo clone. 'auto' looks under external/Detic.")
    parser.add_argument("--detic-python", default="auto",
                        help="Optional Python executable from a Detic-capable environment. 'auto' searches common conda env locations such as ~/anaconda3/envs/detic/bin/python.")
    parser.add_argument("--detic-config", default="auto",
                        help="Detic config YAML. 'auto' searches for the recommended R50/SwinB configs inside the Detic repo.")
    parser.add_argument("--detic-weights", default="auto",
                        help="Detic .pth weights file. 'auto' searches common local paths.")
    parser.add_argument("--detic-custom-vocabulary", default="auto",
                        help="Comma-separated Detic custom vocabulary. 'auto' uses the built-in traffic-scene vocabulary.")
    parser.add_argument("--detic-vocabulary-file", default="auto",
                        help="Optional text file with one Detic vocabulary term per line.")
    parser.add_argument("--detic-min-confidence", type=float, default=0.30,
                        help="Minimum confidence for Detic detections.")
    parser.add_argument("--faster-rcnn-model", default="auto",
                        help="Optional Faster R-CNN frozen graph (.pb) trained on LISA. 'auto' searches common local paths.")
    parser.add_argument("--faster-rcnn-labels", default="auto",
                        help="Optional Faster R-CNN label map (.pbtxt). 'auto' searches common local paths.")
    parser.add_argument("--faster-rcnn-min-confidence", type=float, default=0.55,
                        help="Minimum confidence for Faster R-CNN sign detections.")
    parser.add_argument("--faster-rcnn-max-dim", type=int, default=1000,
                        help="Resize the longest side to at most this value before Faster R-CNN inference, matching the repo's image/video scripts.")
    parser.add_argument("--detr-model", default="auto",
                        help="Optional DETR traffic-sign checkpoint. Supports a Transformers export, Hugging Face model id, or the raw trained_weights artifacts under external/Traffic_Sign_Detection_using_DETR. 'auto' searches common local paths.")
    parser.add_argument("--detr-min-confidence", type=float, default=0.55,
                        help="Minimum confidence for DETR sign detections.")
    parser.add_argument("--car-subclass-model", default="auto",
                        help="Optional car-subclassification model. 'auto' uses the local HF cache when available; 'none' disables it.")
    parser.add_argument("--min-confidence", type=float, default=_DEFAULT_OUTPUT_MIN_CONFIDENCE,
                        help="Final confidence floor applied to rendered/JSON detections after backend-specific filtering and temporal smoothing.")
    parser.add_argument("--imgsz",       type=int, default=_BASE_IMGSZ,
                        help="Base YOLO inference size; tiled sign pass automatically uses a larger size.")

    args = parser.parse_args()

    wants_detic_runtime = (
        args.detector_backend in {"auto", "detic", "hybrid"}
        or args.sign_backend in {"auto", "detic", "ensemble"}
    )
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
                    print(f"[main] Re-launching object_detection.py under Detic runtime: {target_python}")
                    env = dict(os.environ)
                    env["EV_DETIC_REEXEC"] = "1"
                    os.execve(target_python, [target_python, str(Path(__file__).resolve()), *sys.argv[1:]], env)

    print(f"[main] device resolved → {_resolve_device(args.device)}")

    detector = ObjectDetector(
        device=args.device,
        detector_backend=args.detector_backend,
        yolo_model=args.yolo_model,
        sign_model=args.sign_model,
        sign_backend=args.sign_backend,
        detic_python=args.detic_python,
        detic_repo=args.detic_repo,
        detic_config=args.detic_config,
        detic_weights=args.detic_weights,
        detic_custom_vocabulary=args.detic_custom_vocabulary,
        detic_vocabulary_file=args.detic_vocabulary_file,
        detic_min_confidence=args.detic_min_confidence,
        faster_rcnn_model=args.faster_rcnn_model,
        faster_rcnn_labels=args.faster_rcnn_labels,
        faster_rcnn_min_confidence=args.faster_rcnn_min_confidence,
        faster_rcnn_max_dim=args.faster_rcnn_max_dim,
        detr_model=args.detr_model,
        detr_min_confidence=args.detr_min_confidence,
        car_subclass_model=args.car_subclass_model,
        output_min_confidence=args.min_confidence,
        imgsz=args.imgsz,
    )
    pipe = DetectionPipeline(detector=detector)
    scene_name = infer_scene_name(args.scene, args.video, args.frames, args.out_video, args.out_json)
    output_layout = scene_output_layout(scene_name, create=True)
    out_video = str(Path(args.out_video).resolve()) if args.out_video else str((output_layout.detections / "detection_output.mp4").resolve())
    out_json = str(Path(args.out_json).resolve()) if args.out_json else str((output_layout.detections / "detections.json").resolve())

    if args.video:
        pipe.run(
            args.video,
            out_video=out_video,
            out_json=out_json,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
    else:
        pipe.run_on_frames(
            args.frames,
            out_video=out_video,
            out_json=out_json,
            fps=args.fps,
            max_frames=args.max_frames,
        )

    if not args.out_video:
        mirror_stage_output(out_video, scene_name, "detections", Path(out_video).name)
    if not args.out_json:
        mirror_stage_output(out_json, scene_name, "detections", Path(out_json).name)
