"""
Standalone Detic traffic-sign tester.

Purpose
-------
Quickly test whether Detic can robustly detect traffic signs on a single image
or a short video clip without running the full object-detection pipeline.

This script focuses only on traffic-sign prompts and performs a small amount of
post-processing so overlapping prompts such as "traffic sign", "speed limit
sign", and "speed limit 25 sign" collapse into more useful labels.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detic_scene_detector import DeticSceneDetector


_DEFAULT_SIGN_VOCABULARY = (
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
    "yield sign",
    "yield ahead sign",
    "pedestrian crossing sign",
    "crosswalk sign",
    "signal ahead sign",
    "merge sign",
    "keep right sign",
    "keep left sign",
    "one way sign",
    "do not enter sign",
    "speed bump sign",
    "speed hump sign",
    "bump ahead sign",
    "traffic sign",
)


def _normalize_label(label: str) -> str:
    text = str(label or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _parse_speed_limit_value(raw_label: str) -> Optional[int]:
    norm = _normalize_label(raw_label)
    if not any(token in norm for token in ("speed", "limit", "mph", "kmh")):
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


def _normalize_sign_label(raw_label: str) -> str:
    norm = _normalize_label(raw_label)
    if not norm:
        return "traffic_sign"

    value = _parse_speed_limit_value(raw_label)
    if value is not None:
        return f"speed_limit_{int(value)}"

    if norm in {"stop", "stop_sign"} or norm.startswith("stop_sign"):
        return "stop"
    if "yield_ahead" in norm or "yieldahead" in norm:
        return "yield_ahead"
    if norm.startswith("yield") or "yield_sign" in norm:
        return "yield"
    if "pedestrian_crossing" in norm or "crosswalk" in norm:
        return "pedestrian_crossing"
    if "signal_ahead" in norm or "signalahead" in norm:
        return "signal_ahead"
    if "merge" in norm:
        return "merge"
    if "keep_right" in norm or "keepright" in norm:
        return "keep_right"
    if "keep_left" in norm or "keepleft" in norm:
        return "keep_left"
    if "do_not_enter" in norm or "donotenter" in norm:
        return "do_not_enter"
    if "one_way" in norm:
        return "one_way"
    if "school_zone" in norm:
        return "school_zone"
    if any(token in norm for token in ("speed_hump", "speed_breaker", "speed_cushion")):
        return "speed_hump_warning"
    if "bump_ahead" in norm:
        return "bump_ahead"
    if "speed_bump" in norm:
        return "speed_bump_warning"
    if "road_work" in norm or "construction" in norm:
        return "road_work"
    return "traffic_sign"


def _format_sign_label(label: str) -> str:
    norm = str(label or "").strip()
    if not norm:
        return "traffic_sign"
    return norm


def _is_specific_sign_label(label: str) -> bool:
    label = str(label or "").strip()
    return label not in {"", "traffic_sign", "warning_sign"}


def _bbox_iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
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


def _suppress_generic_duplicates(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    specific = [det for det in detections if _is_specific_sign_label(det.get("sign_label", ""))]
    if not specific:
        return detections
    refined: List[Dict[str, Any]] = []
    for det in detections:
        if _is_specific_sign_label(det.get("sign_label", "")):
            refined.append(det)
            continue
        suppress = False
        for keep in specific:
            if _bbox_iou(det["bbox"], keep["bbox"]) >= 0.40:
                suppress = True
                break
        if not suppress:
            refined.append(det)
    return refined


def _nms_by_sign_label(
    detections: List[Dict[str, Any]],
    iou_thr: float = 0.45,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for det in detections:
        grouped.setdefault(str(det.get("sign_label") or "traffic_sign"), []).append(det)

    kept: List[Dict[str, Any]] = []
    for _, group in grouped.items():
        remaining = sorted(group, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            remaining = [
                other for other in remaining
                if _bbox_iou(best["bbox"], other["bbox"]) < iou_thr
            ]
    kept.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return kept


def _decode_sign_detections(
    raw_detections: List[Dict[str, Any]],
    *,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = []
    for det in raw_detections:
        confidence = float(det.get("confidence", 0.0))
        if confidence < min_confidence:
            continue
        raw_label = str(det.get("raw_label") or "")
        sign_label = _normalize_sign_label(raw_label)
        decoded.append(
            {
                "bbox": [int(v) for v in det.get("bbox", [0, 0, 0, 0])[:4]],
                "confidence": confidence,
                "raw_label": raw_label,
                "sign_label": sign_label,
                "class_id": det.get("class_id"),
            }
        )
    decoded = _nms_by_sign_label(decoded)
    decoded = _suppress_generic_duplicates(decoded)
    return decoded


def _draw_sign_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.56
    thickness = 1
    color = (0, 255, 255)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"][:4]]
        label = f"{_format_sign_label(det['sign_label'])} {float(det['confidence']):.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1 - 4, th + bl + 2)
        cv2.rectangle(
            vis,
            (x1, label_y - th - bl - 2),
            (x1 + tw + 4, label_y + 2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            vis,
            label,
            (x1 + 2, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return vis


def _open_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> Tuple[cv2.VideoWriter, Path]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Detic traffic-sign tester.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Input image path.")
    source_group.add_argument("--video", help="Input video path.")
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
    parser.add_argument("--device", default="cpu", help="Detic device, typically cpu or cuda.")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame in video mode.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum processed frames in video mode.")
    parser.add_argument("--out-image", default=None, help="Annotated output image path.")
    parser.add_argument("--out-video", default=None, help="Annotated output video path.")
    parser.add_argument(
        "--custom-vocabulary",
        default="auto",
        help="Optional comma-separated custom sign vocabulary. 'auto' uses the built-in sign vocabulary.",
    )
    return parser.parse_args()


def _resolve_vocabulary(custom_vocabulary: str) -> List[str]:
    if custom_vocabulary and custom_vocabulary != "auto":
        terms = [term.strip() for term in str(custom_vocabulary).split(",") if term.strip()]
        if terms:
            return terms
    return list(_DEFAULT_SIGN_VOCABULARY)


def main(args: argparse.Namespace) -> None:
    detector = DeticSceneDetector(
        repo_root=args.detic_repo,
        config_file=args.detic_config,
        weights_path=args.detic_weights,
        device=args.device,
        min_confidence=float(args.min_confidence),
        vocabulary=_resolve_vocabulary(args.custom_vocabulary),
    )

    if args.image:
        image_path = Path(args.image).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        out_path = (
            Path(args.out_image).expanduser()
            if args.out_image
            else image_path.with_name(f"{image_path.stem}_detic_signs{image_path.suffix}")
        )
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        t0 = time.time()
        detections = _decode_sign_detections(
            detector.detect(image_bgr),
            min_confidence=float(args.min_confidence),
        )
        elapsed = time.time() - t0
        vis = _draw_sign_detections(image_bgr, detections)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), vis):
            raise RuntimeError(f"Could not write output image: {out_path}")
        print(f"[detic_sign_detector.py] image={image_path}")
        print(f"[detic_sign_detector.py] output={out_path}")
        print(f"[detic_sign_detector.py] detections={len(detections)} elapsed_s={elapsed:.3f}")
        for det in detections:
            print(
                f"  - {det['sign_label']} conf={float(det['confidence']):.3f} "
                f"raw='{det['raw_label']}' bbox={det['bbox']}"
            )
        return

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    out_path = (
        Path(args.out_video).expanduser()
        if args.out_video
        else video_path.with_name(f"{video_path.stem}_detic_signs.mp4")
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

    read_idx = -1
    processed = 0
    total_time = 0.0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            read_idx += 1
            if read_idx % frame_skip != 0:
                continue

            t0 = time.time()
            detections = _decode_sign_detections(
                detector.detect(frame_bgr),
                min_confidence=float(args.min_confidence),
            )
            total_time += time.time() - t0
            vis = _draw_sign_detections(frame_bgr, detections)
            writer.write(vis)
            processed += 1

            if processed % 10 == 0:
                avg = total_time / float(max(processed, 1))
                print(
                    f"[detic_sign_detector.py] processed={processed} "
                    f"frame_idx={read_idx} avg_s_per_frame={avg:.3f}"
                )

            if args.max_frames is not None and processed >= int(args.max_frames):
                break
    finally:
        cap.release()
        writer.release()

    avg = total_time / float(max(processed, 1))
    print(f"[detic_sign_detector.py] video={video_path}")
    print(f"[detic_sign_detector.py] output={final_out_path}")
    print(
        f"[detic_sign_detector.py] processed_frames={processed} "
        f"frame_skip={frame_skip} avg_s_per_frame={avg:.3f}"
    )


if __name__ == "__main__":
    main(parse_args())
