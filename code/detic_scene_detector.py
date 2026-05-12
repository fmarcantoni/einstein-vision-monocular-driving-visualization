"""
Optional Detic scene detector backend.

This adapter wraps the official facebookresearch/Detic custom-vocabulary demo
flow so the main pipeline can request broad open-vocabulary scene detections
without hard-coding Detectron2/Detic imports at module import time.
"""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _candidate_detic_python_paths() -> List[Path]:
    candidates: List[Path] = []

    env_python = str(os.environ.get("DETIC_PYTHON", "") or "").strip()
    if env_python:
        candidates.append(Path(env_python).expanduser())

    conda_prefix = str(os.environ.get("CONDA_PREFIX", "") or "").strip()
    conda_env = str(os.environ.get("CONDA_DEFAULT_ENV", "") or "").strip().lower()
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser()
        if conda_env == "detic":
            candidates.append(prefix / "bin" / "python")
        for sibling_name in ("detic",):
            sibling = prefix.parent / sibling_name
            candidates.append(sibling / "bin" / "python")

    home = Path.home()
    for root in ("anaconda3", "miniconda3", "mambaforge", "miniforge3"):
        candidates.append(home / root / "envs" / "detic" / "bin" / "python")

    seen: set[str] = set()
    ordered: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            ordered.append(candidate.resolve())
    return ordered


def resolve_detic_python_executable(requested: str = "auto") -> Optional[str]:
    text = str(requested or "auto").strip()
    if text and text.lower() != "auto":
        path = Path(text).expanduser()
        return str(path.resolve()) if path.exists() else None
    candidates = _candidate_detic_python_paths()
    if not candidates:
        return None
    return str(candidates[0])


def _site_packages_for_python(python_exe: Path) -> List[Path]:
    try:
        probe = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import sys; [print(p) for p in sys.path if 'site-packages' in p]",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return []

    paths: List[Path] = []
    for line in probe.stdout.splitlines():
        text = str(line).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            paths.append(path.resolve())
    return paths


def _ensure_external_detic_runtime_on_sys_path() -> Optional[Path]:
    for python_exe in _candidate_detic_python_paths():
        for site_path in _site_packages_for_python(python_exe):
            site_str = str(site_path)
            if site_str not in sys.path:
                sys.path.insert(0, site_str)
        try:
            import detectron2  # noqa: F401
            import fvcore  # noqa: F401
            return python_exe
        except Exception:
            continue
    return None


class DeticSceneDetector:
    def __init__(
        self,
        repo_root: str,
        config_file: str,
        weights_path: str,
        *,
        python_exe: str = "auto",
        device: str = "cpu",
        min_confidence: float = 0.25,
        vocabulary: Sequence[str],
        pred_all_classes: bool = False,
    ) -> None:
        repo = Path(repo_root).expanduser().resolve()
        config_path = Path(config_file).expanduser().resolve()
        weights = Path(weights_path).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"Detic repo not found: {repo}")
        if not config_path.exists():
            raise FileNotFoundError(f"Detic config not found: {config_path}")
        if not weights.exists():
            raise FileNotFoundError(f"Detic weights not found: {weights}")

        cache_root = (repo / ".cache" / "runtime").resolve()
        clip_cache = (cache_root / "clip").resolve()
        torch_cache = (cache_root / "torch").resolve()
        hf_cache = (cache_root / "huggingface").resolve()
        mpl_cache = (cache_root / "matplotlib").resolve()
        classifier_cache = (cache_root / "classifiers").resolve()
        for path in (cache_root, clip_cache, torch_cache, hf_cache, mpl_cache, classifier_cache):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
        os.environ.setdefault("TORCH_HOME", str(torch_cache))
        os.environ.setdefault("HF_HOME", str(hf_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))
        os.environ.setdefault("DETIC_CLIP_CACHE", str(clip_cache))
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

        repo_str = str(repo)
        centernet_str = str((repo / "third_party" / "CenterNet2").resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        if centernet_str not in sys.path:
            sys.path.insert(0, centernet_str)

        self.runtime_python: Optional[Path] = None
        requested_python = str(python_exe or "auto").strip()
        if requested_python not in {"", "auto"}:
            probe_path = Path(requested_python).expanduser()
            if probe_path.exists():
                for site_path in _site_packages_for_python(probe_path):
                    site_str = str(site_path)
                    if site_str not in sys.path:
                        sys.path.insert(0, site_str)
                self.runtime_python = probe_path.resolve()
        else:
            self.runtime_python = _ensure_external_detic_runtime_on_sys_path()

        try:
            import torch
            from detectron2.data import MetadataCatalog
            from detectron2.config import get_cfg
            from detectron2.engine.defaults import DefaultPredictor
            from centernet.config import add_centernet_config
            from detic.config import add_detic_config
            from detic.modeling.utils import reset_cls_test
            from detic.predictor import BUILDIN_CLASSIFIER, BUILDIN_METADATA_PATH, get_clip_embeddings
        except Exception as exc:
            raise ImportError(
                "Detic requires detectron2 plus the official Detic repo with submodules."
            ) from exc

        self._torch = torch
        self.device = "cuda" if str(device).strip().lower() == "cuda" and torch.cuda.is_available() else "cpu"
        self.min_confidence = float(max(0.0, min(1.0, min_confidence)))
        self.repo_root = repo
        self.config_file = config_path
        self.weights_path = weights
        self.vocabulary = [str(term).strip() for term in vocabulary if str(term).strip()]
        if not self.vocabulary:
            raise ValueError("Detic vocabulary cannot be empty.")

        builtin_vocabulary: Optional[str] = None
        if len(self.vocabulary) == 1:
            token = self.vocabulary[0].strip().lower()
            for prefix in ("__builtin__:", "builtin:"):
                if token.startswith(prefix):
                    builtin_vocabulary = token.split(":", 1)[1].strip()
                    break

        classifier_source: Any

        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(weights)
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.min_confidence
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.min_confidence
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.min_confidence
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cat_freq_path = str(getattr(cfg.MODEL.ROI_BOX_HEAD, "CAT_FREQ_PATH", "") or "").strip()
        if cat_freq_path and not Path(cat_freq_path).expanduser().is_absolute():
            cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str((repo / cat_freq_path).resolve())
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = not bool(pred_all_classes)
        cfg.freeze()

        if builtin_vocabulary:
            if builtin_vocabulary not in BUILDIN_CLASSIFIER or builtin_vocabulary not in BUILDIN_METADATA_PATH:
                raise ValueError(
                    f"Unsupported built-in Detic vocabulary: {builtin_vocabulary}. "
                    f"Available: {sorted(set(BUILDIN_CLASSIFIER).intersection(BUILDIN_METADATA_PATH))}"
                )
            metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[builtin_vocabulary])
            builtin_classes = [str(term).strip() for term in getattr(metadata, "thing_classes", []) if str(term).strip()]
            if not builtin_classes:
                raise RuntimeError(f"Built-in Detic vocabulary '{builtin_vocabulary}' did not expose any classes.")
            self.vocabulary = builtin_classes
            classifier_path = Path(BUILDIN_CLASSIFIER[builtin_vocabulary])
            if not classifier_path.is_absolute():
                classifier_path = (repo / classifier_path).resolve()
            if not classifier_path.exists():
                raise FileNotFoundError(f"Built-in Detic classifier not found: {classifier_path}")
            classifier_source = str(classifier_path)
            print(
                "[DeticSceneDetector] Using built-in Detic classifier "
                f"'{builtin_vocabulary}' with {len(self.vocabulary)} classes"
            )
        else:
            vocab_key = "\n".join(self.vocabulary)
            vocab_hash = hashlib.sha1(vocab_key.encode("utf-8")).hexdigest()[:16]
            classifier_cache_path = classifier_cache / f"custom_vocab_{vocab_hash}.npy"
            if classifier_cache_path.exists():
                classifier_source = str(classifier_cache_path)
                print(
                    "[DeticSceneDetector] Reusing cached CLIP classifier "
                    f"({classifier_cache_path.name}) for {len(self.vocabulary)} prompts"
                )
            else:
                print(
                    "[DeticSceneDetector] Building CLIP classifier cache "
                    f"for {len(self.vocabulary)} prompts"
                )
                classifier_tensor = get_clip_embeddings(self.vocabulary)
                np.save(
                    str(classifier_cache_path),
                    classifier_tensor.permute(1, 0).contiguous().cpu().numpy(),
                )
                del classifier_tensor
                gc.collect()
                classifier_source = str(classifier_cache_path)
                print(
                    "[DeticSceneDetector] Saved CLIP classifier cache "
                    f"to {classifier_cache_path}"
                )

        self.predictor = DefaultPredictor(cfg)
        reset_cls_test(self.predictor.model, classifier_source, len(self.vocabulary))
        self.category_index = {idx: label for idx, label in enumerate(self.vocabulary)}

    def detect(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        outputs = self.predictor(frame_bgr)
        instances = outputs.get("instances")
        if instances is None:
            return []

        cpu_device = self._torch.device("cpu")
        instances = instances.to(cpu_device)
        if not instances.has("scores") or not instances.has("pred_boxes") or not instances.has("pred_classes"):
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        labels = instances.pred_classes.numpy()
        height, width = frame_bgr.shape[:2]

        detections: List[Dict[str, Any]] = []
        for box, score, label_idx in zip(boxes, scores, labels):
            conf = float(score)
            if conf < self.min_confidence:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(label_idx)
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
