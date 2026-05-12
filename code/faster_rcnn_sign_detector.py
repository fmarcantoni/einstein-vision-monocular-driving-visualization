"""
Optional Faster R-CNN traffic-sign backend.

This adapter is intentionally small and only implements the inference path
used by the meng1994412/Traffic_Sign_Detection repo: load a frozen TensorFlow
graph, parse a `.pbtxt` label map, and return bounding boxes/classes/scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import cv2
import numpy as np


def parse_label_map_pbtxt(path: str | Path) -> Dict[int, str]:
    text = Path(path).read_text(encoding="utf-8")
    mapping: Dict[int, str] = {}
    for block in re.findall(r"item\s*\{(.*?)\}", text, flags=re.S):
        id_match = re.search(r"\bid\s*:\s*(\d+)", block)
        name_match = re.search(r"\b(?:display_name|name)\s*:\s*['\"]([^'\"]+)['\"]", block)
        if not id_match or not name_match:
            continue
        mapping[int(id_match.group(1))] = name_match.group(1).strip()
    return mapping


class FasterRCNNSignDetector:
    """
    Lightweight wrapper around a TensorFlow frozen inference graph.

    The expected exported model format is `frozen_inference_graph.pb`, matching
    the original TensorFlow Object Detection API export path used by the repo.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        *,
        min_confidence: float = 0.5,
        resize_max_dim: int = 1000,
    ) -> None:
        try:
            import tensorflow as tf  # type: ignore
        except Exception as exc:
            raise ImportError(
                "TensorFlow is required for the Faster R-CNN sign backend."
            ) from exc

        tf.compat.v1.disable_eager_execution()
        self._tf = tf.compat.v1
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.labels_path = str(Path(labels_path).expanduser().resolve())
        self.min_confidence = float(min_confidence)
        self.resize_max_dim = int(max(128, resize_max_dim))
        self.category_index = parse_label_map_pbtxt(self.labels_path)

        self.graph = self._tf.Graph()
        with self.graph.as_default():
            graph_def = self._tf.GraphDef()
            with self._tf.gfile.GFile(self.model_path, "rb") as fh:
                graph_def.ParseFromString(fh.read())
            self._tf.import_graph_def(graph_def, name="")

        self.sess = self._tf.Session(graph=self.graph)
        self.image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
        self.boxes_tensor = self.graph.get_tensor_by_name("detection_boxes:0")
        self.scores_tensor = self.graph.get_tensor_by_name("detection_scores:0")
        self.classes_tensor = self.graph.get_tensor_by_name("detection_classes:0")
        self.num_detections_tensor = self.graph.get_tensor_by_name("num_detections:0")

    def close(self) -> None:
        if getattr(self, "sess", None) is not None:
            self.sess.close()
            self.sess = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _prepare_image(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        orig_h, orig_w = frame_bgr.shape[:2]
        scale = 1.0
        resized = frame_bgr
        longest_side = max(orig_h, orig_w)
        if longest_side > self.resize_max_dim:
            scale = self.resize_max_dim / float(longest_side)
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batched = np.expand_dims(rgb, axis=0)
        return {
            "image": batched,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "resized_h": resized.shape[0],
            "resized_w": resized.shape[1],
        }

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        payload = self._prepare_image(frame_bgr)
        boxes, scores, labels, _ = self.sess.run(
            [
                self.boxes_tensor,
                self.scores_tensor,
                self.classes_tensor,
                self.num_detections_tensor,
            ],
            feed_dict={self.image_tensor: payload["image"]},
        )

        boxes = np.squeeze(boxes, axis=0)
        scores = np.squeeze(scores, axis=0)
        labels = np.squeeze(labels, axis=0)

        orig_h = int(payload["orig_h"])
        orig_w = int(payload["orig_w"])
        resized_h = int(payload["resized_h"])
        resized_w = int(payload["resized_w"])
        scale_x = orig_w / float(max(resized_w, 1))
        scale_y = orig_h / float(max(resized_h, 1))

        detections: List[Dict[str, Any]] = []
        for box, score, label in zip(boxes, scores, labels):
            conf = float(score)
            if conf < self.min_confidence:
                continue

            start_y, start_x, end_y, end_x = [float(v) for v in box.tolist()]
            x1 = int(round(start_x * resized_w * scale_x))
            y1 = int(round(start_y * resized_h * scale_y))
            x2 = int(round(end_x * resized_w * scale_x))
            y2 = int(round(end_y * resized_h * scale_y))
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(label)
            raw_label = self.category_index.get(class_id, str(class_id))
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "raw_label": raw_label,
                    "class_id": class_id,
                }
            )
        return detections
