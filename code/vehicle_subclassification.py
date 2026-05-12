"""
vehicle_subclassification.py
============================

Shared helpers for fine-grained car subtype classification.

The current pipeline only needs a coarse renderable taxonomy for cars:
``sedan``, ``suv``, ``hatchback``, and ``pickup``.  This helper loads a
fine-grained Hugging Face image classifier and collapses its detailed labels
into those four production subtypes.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HF_CAR_SUBCLASS_MODEL_ID = "rodrigoruiz/image-classifier-car-models-3"
CAR_SUBTYPE_CHOICES: Tuple[str, ...] = ("sedan", "suv", "hatchback", "pickup")
_DISABLED_MODEL_STRINGS = {"none", "off", "disable", "disabled"}
_LOCAL_MODEL_CANDIDATES: Tuple[str, ...] = (
    "weights/car_subclass_model",
    "weights/car_subclassifier",
    "weights/car_models_classifier",
    "weights/vehicle_subclassifier",
)

_DEFAULT_MIN_CROP_SIDE_PX = 36
_DEFAULT_MIN_CROP_AREA_PX = 42 * 42
_DEFAULT_BATCH_SIZE = 8


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class VehicleSubtypePrediction:
    subtype: str
    confidence: float
    raw_label: Optional[str] = None


def _normalize_label(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def map_fine_label_to_car_subtype(label: str) -> Optional[str]:
    norm = _normalize_label(label)
    if not norm:
        return None

    if any(token in norm for token in ("pickup", "crew_cab", "extended_cab", "regular_cab", "club_cab", "double_cab")):
        return "pickup"
    if any(token in norm for token in ("suv", "crossover")):
        return "suv"
    if any(token in norm for token in ("hatchback", "wagon", "minivan", "van", "liftback")):
        return "hatchback"
    if any(token in norm for token in ("sedan", "coupe", "convertible", "cabriolet", "roadster", "fastback")):
        return "sedan"

    # The model is fine-grained and largely passenger-car-focused, so an
    # unmapped passenger-car label is still best rendered as a sedan.
    return "sedan"


def _resolve_device(requested: str = "cpu") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps":
        mps_ok = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
        return "mps" if mps_ok else "cpu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_ok = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
        if mps_ok:
            return "mps"
    return "cpu"


def _crop_signature(image_bgr: np.ndarray) -> str:
    thumb = cv2.resize(image_bgr, (48, 48), interpolation=cv2.INTER_AREA)
    quantized = np.clip((thumb.astype(np.float32) / 16.0).round(), 0, 15).astype(np.uint8)
    return hashlib.sha1(quantized.tobytes()).hexdigest()


def _expand_bbox(
    bbox: Sequence[int],
    frame_w: int,
    frame_h: int,
    pad_frac: float = 0.08,
) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    px = max(2, int(round(w * pad_frac)))
    py = max(2, int(round(h * pad_frac)))
    return [
        max(0, x1 - px),
        max(0, y1 - py),
        min(frame_w - 1, x2 + px),
        min(frame_h - 1, y2 + py),
    ]


class VehicleSubclassifier:
    def __init__(
        self,
        model_name: str = "auto",
        device: str = "cpu",
        min_crop_side_px: int = _DEFAULT_MIN_CROP_SIDE_PX,
        min_crop_area_px: int = _DEFAULT_MIN_CROP_AREA_PX,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        disabled = str(model_name or "").strip().lower() in _DISABLED_MODEL_STRINGS
        self.model_name = "" if disabled else self._resolve_model_name(model_name)
        self.device = _resolve_device(device)
        self.min_crop_side_px = int(max(16, min_crop_side_px))
        self.min_crop_area_px = int(max(16 * 16, min_crop_area_px))
        self.batch_size = int(max(1, batch_size))
        self.allow_remote_downloads = _truthy_env("EV_ALLOW_REMOTE_MODEL_DOWNLOADS")
        self._processor = None
        self._model = None
        self._torch = None
        self._id2label: Dict[int, str] = {}
        self._available = False
        self._warned = bool(disabled)
        self._cache: Dict[str, VehicleSubtypePrediction] = {}

    @staticmethod
    def _resolve_model_name(model_name: Optional[str]) -> str:
        if model_name not in {"", "auto", None}:
            return str(model_name)
        for candidate in _LOCAL_MODEL_CANDIDATES:
            path = Path(candidate).expanduser()
            if path.exists():
                return str(path.resolve())
        return HF_CAR_SUBCLASS_MODEL_ID

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return bool(self._available)

    def _ensure_loaded(self) -> None:
        if self._available or self._warned:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self._torch = torch
            load_kwargs = {}
            if not self.allow_remote_downloads:
                model_path = Path(self.model_name).expanduser()
                if not model_path.exists():
                    load_kwargs["local_files_only"] = True

            self._processor = AutoImageProcessor.from_pretrained(self.model_name, **load_kwargs)
            self._model = AutoModelForImageClassification.from_pretrained(self.model_name, **load_kwargs)
            self._model.eval()
            if self.device != "cpu":
                self._model.to(self.device)
            self._id2label = {
                int(k): str(v)
                for k, v in getattr(self._model.config, "id2label", {}).items()
            }
            if not self._id2label:
                self._id2label = {
                    int(i): str(v)
                    for i, v in enumerate(getattr(self._model.config, "labels", []) or [])
                }
            self._available = True
            print(f"[VehicleSubclassifier] Loaded {self.model_name} on {self.device}")
        except Exception as exc:
            self._warned = True
            print(
                "[VehicleSubclassifier] Warning: could not load the HF car-subclassification model. "
                "Continuing without fine-grained car subtypes. "
                "Set EV_ALLOW_REMOTE_MODEL_DOWNLOADS=1 if you want this helper to fetch missing HF files. "
                f"({type(exc).__name__}: {exc})"
            )

    def _predict_batch(self, crops_bgr: Sequence[np.ndarray]) -> List[VehicleSubtypePrediction]:
        self._ensure_loaded()
        if not self._available or self._processor is None or self._model is None or self._torch is None:
            return [VehicleSubtypePrediction("sedan", 0.0, None) for _ in crops_bgr]

        rgb_images = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops_bgr]
        encoded = self._processor(images=rgb_images, return_tensors="pt")
        if self.device != "cpu":
            encoded = {
                key: value.to(self.device)
                if hasattr(value, "to")
                else value
                for key, value in encoded.items()
            }

        with self._torch.no_grad():
            outputs = self._model(**encoded)
            probs = self._torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

        predictions: List[VehicleSubtypePrediction] = []
        for row in probs:
            coarse_scores = {key: 0.0 for key in CAR_SUBTYPE_CHOICES}
            top_idx = int(np.argmax(row))
            raw_label = self._id2label.get(top_idx)

            for idx, prob in enumerate(row.tolist()):
                subtype = map_fine_label_to_car_subtype(self._id2label.get(int(idx), ""))
                if subtype is None:
                    continue
                coarse_scores[subtype] += float(prob)

            best_subtype = max(coarse_scores.items(), key=lambda item: item[1])[0]
            predictions.append(
                VehicleSubtypePrediction(
                    subtype=best_subtype,
                    confidence=float(coarse_scores[best_subtype]),
                    raw_label=raw_label,
                )
            )
        return predictions

    def classify_car_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: Sequence[Sequence[int]],
    ) -> List[VehicleSubtypePrediction]:
        self._ensure_loaded()
        if not boxes_xyxy:
            return []
        if not self._available:
            return [VehicleSubtypePrediction("sedan", 0.0, None) for _ in boxes_xyxy]

        H, W = frame_bgr.shape[:2]
        prepared: List[Tuple[int, str, np.ndarray]] = []
        outputs: List[Optional[VehicleSubtypePrediction]] = [None] * len(boxes_xyxy)

        for idx, bbox in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = _expand_bbox(bbox, W, H)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                outputs[idx] = VehicleSubtypePrediction("sedan", 0.0, None)
                continue
            h, w = crop.shape[:2]
            if min(h, w) < self.min_crop_side_px or (h * w) < self.min_crop_area_px:
                outputs[idx] = VehicleSubtypePrediction("sedan", 0.0, None)
                continue
            signature = _crop_signature(crop)
            cached = self._cache.get(signature)
            if cached is not None:
                outputs[idx] = cached
                continue
            prepared.append((idx, signature, crop))

        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            predictions = self._predict_batch([crop for _, _, crop in batch])
            for (idx, signature, _), pred in zip(batch, predictions):
                self._cache[signature] = pred
                outputs[idx] = pred

        return [
            pred if pred is not None else VehicleSubtypePrediction("sedan", 0.0, None)
            for pred in outputs
        ]
