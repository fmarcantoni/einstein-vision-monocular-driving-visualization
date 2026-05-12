"""
Optional DETR traffic-sign backend.

This adapter supports two DETR inference paths:

1. A standard Hugging Face Transformers DETR checkpoint directory or model id.
2. The local raw training checkpoints saved under
   ``external/Traffic_Sign_Detection_using_DETR/trained_weights/...``.

The second path reconstructs a small DETR-style inference module that matches
the original checkpoint layout closely enough for runtime use inside this
project, so the pipeline can consume the locally trained sign weights without
requiring a separate export step.
"""

from __future__ import annotations

import copy
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class _PositionEmbeddingSine:
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000) -> None:
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)

    def __call__(self, tensor, torch):
        b, _, h, w = tensor.shape
        y_embed = torch.arange(h, device=tensor.device, dtype=tensor.dtype).view(1, h, 1).expand(b, h, w)
        x_embed = torch.arange(w, device=tensor.device, dtype=tensor.dtype).view(1, 1, w).expand(b, h, w)

        eps = 1e-6
        y_embed = (y_embed + 0.5) / max(float(h), eps) * (2.0 * np.pi)
        x_embed = (x_embed + 0.5) / max(float(w), eps) * (2.0 * np.pi)

        dim_t = torch.arange(self.num_pos_feats, device=tensor.device, dtype=tensor.dtype)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _build_raw_detr_components(torch, torchvision, *, dc5: bool, class_outputs: int):
    import torch.nn as nn

    class BackboneHolder(nn.Module):
        def __init__(self, body) -> None:
            super().__init__()
            self.body = body

        def forward(self, x):
            x = self.body.conv1(x)
            x = self.body.bn1(x)
            x = self.body.relu(x)
            x = self.body.maxpool(x)
            x = self.body.layer1(x)
            x = self.body.layer2(x)
            x = self.body.layer3(x)
            x = self.body.layer4(x)
            return x

    class MLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
            super().__init__()
            dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            self.layers = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
            )

        def forward(self, x):
            for idx, layer in enumerate(self.layers):
                x = layer(x)
                if idx < len(self.layers) - 1:
                    x = torch.relu(x)
            return x

    class EncoderLayer(nn.Module):
        def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, src, pos=None):
            q = k = src if pos is None else src + pos
            src2 = self.self_attn(q, k, value=src, need_weights=False)[0]
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
            src = self.norm2(src + self.dropout2(src2))
            return src

    class DecoderLayer(nn.Module):
        def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)

        def forward(self, tgt, memory, pos=None, query_pos=None):
            q = k = tgt if query_pos is None else tgt + query_pos
            tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
            q = tgt if query_pos is None else tgt + query_pos
            k = memory if pos is None else memory + pos
            tgt2 = self.multihead_attn(query=q, key=k, value=memory, need_weights=False)[0]
            tgt = self.norm2(tgt + self.dropout2(tgt2))
            tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
            tgt = self.norm3(tgt + self.dropout3(tgt2))
            return tgt

    class Encoder(nn.Module):
        def __init__(self, layer, num_layers: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

        def forward(self, src, pos=None):
            output = src
            for layer in self.layers:
                output = layer(output, pos=pos)
            return output

    class Decoder(nn.Module):
        def __init__(self, layer, num_layers: int, norm) -> None:
            super().__init__()
            self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
            self.norm = norm

        def forward(self, tgt, memory, pos=None, query_pos=None):
            output = tgt
            for layer in self.layers:
                output = layer(output, memory, pos=pos, query_pos=query_pos)
            return self.norm(output)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_model = 256
            nhead = 8
            dim_feedforward = 2048
            dropout = 0.1
            encoder_layer = EncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            decoder_layer = DecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            self.encoder = Encoder(encoder_layer, num_layers=6)
            self.decoder = Decoder(decoder_layer, num_layers=6, norm=nn.LayerNorm(d_model))

        def forward(self, src, query_embed, pos):
            memory = self.encoder(src, pos=pos)
            tgt = torch.zeros_like(query_embed)
            hs = self.decoder(tgt, memory, pos=pos, query_pos=query_embed)
            return hs.transpose(0, 1)

    class DetrCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            body = torchvision.models.resnet50(
                weights=None,
                replace_stride_with_dilation=[False, False, bool(dc5)],
            )
            self.backbone = nn.ModuleList([BackboneHolder(body)])
            self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)
            self.query_embed = nn.Embedding(100, 256)
            self.transformer = Transformer()
            self.class_embed = nn.Linear(256, int(class_outputs))
            self.bbox_embed = MLP(256, 256, 4, 3)
            self._position_embedding = _PositionEmbeddingSine(128)

        def forward(self, images):
            features = self.backbone[0](images)
            src = self.input_proj(features)
            pos = self._position_embedding(src, torch)

            b, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos = pos.flatten(2).permute(2, 0, 1)
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)

            hs = self.transformer(src, query_embed=query_embed, pos=pos)
            logits = self.class_embed(hs)
            boxes = self.bbox_embed(hs).sigmoid()
            return logits, boxes

    class DetrWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = DetrCore()

        def forward(self, images):
            return self.model(images)

    return DetrWrapper(), BackboneHolder, MLP, EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer


class DetrSignDetector:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        min_confidence: float = 0.5,
    ) -> None:
        try:
            import torch
        except Exception as exc:
            raise ImportError("torch is required for the DETR sign backend.") from exc

        self._torch = torch
        self.device = str(device)
        self.min_confidence = float(min_confidence)
        self.model_name_or_path = model_name_or_path
        self.processor = None
        self.model = None
        self._raw_model = None
        self._raw_mode = False
        self.runtime_mode = "unknown"
        self._raw_sign_ids: List[int] = []
        self.category_index: Dict[int, str] = {}

        resolved_path = Path(str(model_name_or_path)).expanduser()
        if resolved_path.exists():
            raw_ckpt, raw_config = self._maybe_detect_raw_training_checkpoint(resolved_path)
            if raw_ckpt is not None:
                self._init_raw_checkpoint_detector(raw_ckpt, raw_config)
                return

        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except Exception as exc:
            raise ImportError(
                "Transformers and torch are required for the Hugging Face DETR sign backend."
            ) from exc

        self.processor = AutoImageProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.runtime_mode = "transformers"

        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.category_index = {int(k): str(v) for k, v in id2label.items()}

    def _safe_torch_load(self, checkpoint_path: Path) -> Dict[str, Any]:
        return self._torch.load(str(checkpoint_path), map_location="cpu")

    def _maybe_detect_raw_training_checkpoint(
        self,
        candidate_path: Path,
    ) -> Tuple[Path | None, Dict[str, Any] | None]:
        checkpoint_path: Path | None = None
        if candidate_path.is_file() and candidate_path.suffix == ".pth":
            checkpoint_path = candidate_path
        elif candidate_path.is_dir():
            for name in ("best_model.pth", "last_model_state.pth"):
                possible = candidate_path / name
                if possible.exists():
                    checkpoint_path = possible
                    break
        if checkpoint_path is None:
            return None, None

        try:
            payload = self._safe_torch_load(checkpoint_path)
        except Exception:
            return checkpoint_path, None
        if not isinstance(payload, dict):
            return checkpoint_path, None
        if "model_state_dict" not in payload:
            return None, None
        config = payload.get("config")
        return checkpoint_path, config if isinstance(config, dict) else None

    def _init_raw_checkpoint_detector(
        self,
        checkpoint_path: Path,
        raw_config: Optional[Dict[str, Any]],
    ) -> None:
        try:
            import torchvision
        except Exception as exc:
            raise ImportError(
                "torchvision is required to run the raw DETR training checkpoints."
            ) from exc

        payload = self._safe_torch_load(checkpoint_path)
        state = payload.get("model_state_dict") or {}
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"Raw DETR checkpoint '{checkpoint_path}' does not contain a usable model_state_dict.")

        class_head = state.get("model.class_embed.weight")
        if class_head is None:
            raise RuntimeError(
                f"Raw DETR checkpoint '{checkpoint_path}' is missing 'model.class_embed.weight'."
            )

        class_outputs = int(class_head.shape[0])
        dc5 = "dc5" in str(checkpoint_path).lower()
        raw_model, *_ = _build_raw_detr_components(
            self._torch,
            torchvision,
            dc5=dc5,
            class_outputs=class_outputs,
        )

        missing, unexpected = raw_model.load_state_dict(state, strict=False)
        non_trivial_missing = [
            key for key in missing
            if "num_batches_tracked" not in key
            and "_position_embedding" not in key
            and not key.endswith(".fc.weight")
            and not key.endswith(".fc.bias")
        ]
        tolerated_unexpected = {"out.weight", "out.bias"}
        unexpected = [key for key in unexpected if key not in tolerated_unexpected]
        if unexpected:
            raise RuntimeError(
                "Raw DETR checkpoint load produced unexpected parameter names: "
                + ", ".join(sorted(unexpected)[:12])
            )
        if non_trivial_missing:
            raise RuntimeError(
                "Raw DETR checkpoint load is incomplete. Missing parameters: "
                + ", ".join(sorted(non_trivial_missing)[:12])
            )

        raw_model.to(self.device)
        raw_model.eval()
        self._raw_model = raw_model
        self._raw_mode = True
        self.runtime_mode = "raw_checkpoint"

        class_names = list((raw_config or {}).get("CLASSES") or [])
        if not class_names:
            class_names = [
                "__background__",
                "keepRight",
                "merge",
                "pedestrianCrossing",
                "signalAhead",
                "speedLimit25",
                "speedLimit35",
                "stop",
                "yield",
                "yieldAhead",
            ]
        self.category_index = {
            int(idx): str(name)
            for idx, name in enumerate(class_names)
            if idx > 0
        }
        self._raw_sign_ids = sorted(self.category_index.keys())

    def _prepare_raw_input(self, frame_bgr: np.ndarray):
        orig_h, orig_w = frame_bgr.shape[:2]
        target_short = 800.0
        max_side = 1333.0
        scale = min(target_short / float(max(min(orig_h, orig_w), 1)), max_side / float(max(orig_h, orig_w)))
        scale = float(max(scale, 1e-6))
        resized_w = max(1, int(round(orig_w * scale)))
        resized_h = max(1, int(round(orig_h * scale)))
        resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        mean = self._torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = self._torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0).to(self.device), (orig_h, orig_w), (resized_h, resized_w)

    def _detect_raw_checkpoint(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._raw_model is None:
            return []

        image_tensor, (orig_h, orig_w), (resized_h, resized_w) = self._prepare_raw_input(frame_bgr)
        with self._torch.no_grad():
            logits, boxes = self._raw_model(image_tensor)

        logits = logits[0]
        boxes = boxes[0]
        probs = self._torch.softmax(logits, dim=-1)
        no_object_index = int(probs.shape[-1] - 1)

        detections: List[Dict[str, Any]] = []
        for query_idx in range(int(probs.shape[0])):
            if not self._raw_sign_ids:
                continue
            sign_scores = probs[query_idx, self._raw_sign_ids]
            best_idx = int(self._torch.argmax(sign_scores).item())
            class_id = int(self._raw_sign_ids[best_idx])
            confidence = float(sign_scores[best_idx].item())
            no_object_conf = float(probs[query_idx, no_object_index].item())
            if confidence < self.min_confidence or confidence <= no_object_conf:
                continue

            cx, cy, w, h = [float(v) for v in boxes[query_idx].tolist()]
            x1 = (cx - 0.5 * w) * float(resized_w)
            y1 = (cy - 0.5 * h) * float(resized_h)
            x2 = (cx + 0.5 * w) * float(resized_w)
            y2 = (cy + 0.5 * h) * float(resized_h)

            x1 = int(round(np.clip(x1 * (float(orig_w) / float(max(resized_w, 1))), 0, orig_w - 1)))
            y1 = int(round(np.clip(y1 * (float(orig_h) / float(max(resized_h, 1))), 0, orig_h - 1)))
            x2 = int(round(np.clip(x2 * (float(orig_w) / float(max(resized_w, 1))), 0, orig_w - 1)))
            y2 = int(round(np.clip(y2 * (float(orig_h) / float(max(resized_h, 1))), 0, orig_h - 1)))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "raw_label": self.category_index.get(class_id, str(class_id)),
                    "class_id": class_id,
                }
            )
        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._raw_mode:
            return self._detect_raw_checkpoint(frame_bgr)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self._torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.min_confidence,
            target_sizes=[(height, width)],
        )[0]

        boxes = results.get("boxes")
        scores = results.get("scores")
        labels = results.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        detections: List[Dict[str, Any]] = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(label.item() if hasattr(label, "item") else label)
            raw_label = self.category_index.get(class_id, str(class_id))
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score.item() if hasattr(score, "item") else score),
                    "raw_label": raw_label,
                    "class_id": class_id,
                }
            )
        return detections


def _draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    *,
    use_pipeline_labels: bool = True,
) -> np.ndarray:
    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    color = (0, 255, 255)

    canonicalize = None
    normalize = None
    format_label = None
    if use_pipeline_labels:
        try:
            from object_detection import (
                _canonicalize_sign_label,
                _format_sign_label,
                _normalize_sign_subclass_label,
            )

            canonicalize = _canonicalize_sign_label
            normalize = _normalize_sign_subclass_label
            format_label = _format_sign_label
        except Exception:
            canonicalize = None

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])[:4]]
        raw_label = str(det.get("raw_label") or "")
        confidence = float(det.get("confidence", 0.0))
        label = raw_label or "sign"

        if canonicalize is not None and normalize is not None and format_label is not None:
            cls_name, speed_limit_value = canonicalize(raw_label, "")
            if cls_name in {"traffic_sign", "stop_sign", "speed_limit"}:
                label = format_label(normalize(raw_label, cls_name))
            elif cls_name:
                label = str(cls_name)
            elif speed_limit_value is not None:
                label = f"speed_limit_{int(speed_limit_value)}"

        display = f"{label} {confidence:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        (tw, th), bl = cv2.getTextSize(display, font, font_scale, thickness)
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
            display,
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
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (int(width), int(height)))
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DETR sign detector on a single image or a short video.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--image",
        help="Input image path.",
    )
    source_group.add_argument(
        "--video",
        help="Input video path.",
    )
    parser.add_argument(
        "--model",
        default="external/Traffic_Sign_Detection_using_DETR/trained_weights/detr_resnet50_dc5_75e/best_model.pth",
        help="DETR checkpoint path. Defaults to the local best_model.pth.",
    )
    parser.add_argument(
        "--out-image",
        default=None,
        help="Annotated output image path. Defaults next to the input image.",
    )
    parser.add_argument(
        "--out-video",
        default=None,
        help="Annotated output video path. Defaults next to the input video.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.40,
        help="Minimum confidence threshold for drawing detections.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of processed frames in video mode.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame in video mode.",
    )
    args = parser.parse_args()

    detector = DetrSignDetector(
        args.model,
        device=args.device,
        min_confidence=args.min_confidence,
    )
    print(
        f"[detr_sign_detector.py] runtime_mode={detector.runtime_mode} "
        f"device={args.device} min_confidence={args.min_confidence:.2f}"
    )

    if args.image:
        image_path = Path(args.image).expanduser()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        out_path = (
            Path(args.out_image).expanduser()
            if args.out_image
            else image_path.with_name(f"{image_path.stem}_detr_signs{image_path.suffix}")
        )

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        t0 = time.time()
        detections = detector.detect(image_bgr)
        elapsed = time.time() - t0
        vis = _draw_detections(image_bgr, detections)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), vis)
        if not ok:
            raise RuntimeError(f"Could not write output image: {out_path}")

        print(f"[detr_sign_detector.py] model={args.model}")
        print(f"[detr_sign_detector.py] image={image_path}")
        print(f"[detr_sign_detector.py] output={out_path}")
        print(f"[detr_sign_detector.py] detections={len(detections)} elapsed_s={elapsed:.3f}")
        for det in detections:
            print(
                f"  - {det.get('raw_label', 'sign')} "
                f"conf={float(det.get('confidence', 0.0)):.3f} "
                f"bbox={det.get('bbox')}"
            )
    else:
        video_path = Path(args.video).expanduser()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        out_path = (
            Path(args.out_video).expanduser()
            if args.out_video
            else video_path.with_name(f"{video_path.stem}_detr_signs.mp4")
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

        frame_idx = -1
        processed = 0
        total_time = 0.0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                t0 = time.time()
                detections = detector.detect(frame_bgr)
                total_time += time.time() - t0
                vis = _draw_detections(frame_bgr, detections)
                writer.write(vis)
                processed += 1

                if processed <= 3 or processed % 10 == 0:
                    avg = total_time / float(max(processed, 1))
                    print(
                        f"[detr_sign_detector.py] processed={processed} "
                        f"frame_idx={frame_idx} avg_s_per_frame={avg:.3f}"
                    )

                if args.max_frames is not None and processed >= int(args.max_frames):
                    break
        finally:
            cap.release()
            writer.release()

        avg = total_time / float(max(processed, 1))
        print(f"[detr_sign_detector.py] model={args.model}")
        print(f"[detr_sign_detector.py] video={video_path}")
        print(f"[detr_sign_detector.py] output={final_out_path}")
        print(
            f"[detr_sign_detector.py] processed_frames={processed} "
            f"frame_skip={frame_skip} avg_s_per_frame={avg:.3f}"
        )
