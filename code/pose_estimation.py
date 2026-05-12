"""
pose_estimation.py
==================

Pedestrian pose estimation stage for the autonomous-driving pipeline.

Responsibilities
----------------
* detect pedestrians and estimate 17-keypoint COCO pose per frame
* output JSON with per-frame keypoints consumable by scene_assembler.py
* provide standalone video output for visual debugging

Backend
-------
The stage is PyMAF-aware but only uses a true PyMAF backend when a local
PyMAF/SMPL installation is available. Otherwise it falls back to tracked
Ultralytics YOLO pose estimates. A local/custom ``YOLO26`` pose checkpoint is
always preferred in `auto` mode. If no local YOLO26 pose weights are present,
the official Ultralytics model name is used directly so the runtime still
targets YOLO26 rather than falling through to an older family.

Output schema
-------------
The JSON mirrors the detection JSON schema with an added ``keypoints`` field::

    {
      "source": "...",
      "frames_written": N,
      "total_detections": N,
      "frames": [
        {
          "frame_idx": 0,
          "timestamp_s": 0.0,
          "pedestrians": [
            {
              "id": 0,
              "bbox": [x1, y1, x2, y2],
              "confidence": 0.87,
              "keypoints": [[x, y, conf], ...],   # 17 COCO keypoints
              "keypoint_names": ["nose", "left_eye", ...]
            }
          ]
        }
      ]
    }
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import json
import time
import math
import argparse
import shutil
import subprocess
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Sequence

import cv2
import numpy as np

from object_detection import ObjectDetector
from project_setup import (
    infer_scene_name,
    mirror_stage_output,
    resolve_existing_artifact,
    scene_output_layout,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton edges for visualisation (pairs of keypoint indices)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                      # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12),                            # torso
    (11, 12),                                    # hips
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]

# Keypoint colours (BGR) — left=blue, right=green, center=yellow
_KP_COLORS = [
    (0, 255, 255),   # nose
    (255, 0, 0),     # left_eye
    (0, 255, 0),     # right_eye
    (255, 0, 0),     # left_ear
    (0, 255, 0),     # right_ear
    (255, 128, 0),   # left_shoulder
    (0, 200, 0),     # right_shoulder
    (255, 128, 0),   # left_elbow
    (0, 200, 0),     # right_elbow
    (255, 128, 0),   # left_wrist
    (0, 200, 0),     # right_wrist
    (255, 0, 128),   # left_hip
    (0, 128, 255),   # right_hip
    (255, 0, 128),   # left_knee
    (0, 128, 255),   # right_knee
    (255, 0, 128),   # left_ankle
    (0, 128, 255),   # right_ankle
]

CONF_THRESHOLD = 0.40
KP_CONF_THRESHOLD = 0.30
PROPOSAL_CONF_THRESHOLD = 0.30
POSE_BASE_IMGSZ = 1280
POSE_CROP_IMGSZ = 960
POSE_CROP_PAD_FRAC = 0.18

PYMAF_J24_TO_COCO = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]
PYMAF_REQUIRED_MODULES = (
    "yacs",
    "joblib",
    "smplx",
    "trimesh",
    "pyrender",
    "ultralytics",
    "timm",
    "einops",
)
PYMAF_REQUIRED_FILES = (
    "smpl_mean_params.npz",
    "mesh_downsampling.npz",
    "J_regressor_extra.npy",
    "SMPL_NEUTRAL.pkl",
    "SMPL_MALE.pkl",
    "SMPL_FEMALE.pkl",
)
PYMAF_FILE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "SMPL_NEUTRAL.pkl": (
        "SMPL_NEUTRAL.pkl",
        "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
        "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
        "basicModel_neutral_lbs_10_207_0_v1.1.0.pkl",
    ),
    "SMPL_MALE.pkl": (
        "SMPL_MALE.pkl",
        "basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
        "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
        "basicModel_m_lbs_10_207_0_v1.1.0.pkl",
    ),
    "SMPL_FEMALE.pkl": (
        "SMPL_FEMALE.pkl",
        "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
        "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
        "basicModel_f_lbs_10_207_0_v1.1.0.pkl",
    ),
}


def _path_from_setting(value: str) -> Optional[Path]:
    text = str(value or "").strip()
    if not text or text.lower() == "auto":
        return None
    return Path(text).expanduser().resolve()


def _resolve_pymaf_python(requested: str = "auto") -> str:
    explicit = str(requested or "").strip()
    if explicit and explicit.lower() != "auto":
        return explicit
    env_python = str(os.environ.get("PYMAF_PYTHON", "")).strip()
    if env_python:
        return env_python
    return sys.executable


def _resolve_pymaf_repo_dir(requested: str = "auto") -> Optional[Path]:
    explicit = _path_from_setting(requested)
    if explicit is not None:
        return explicit if explicit.exists() else None

    candidates = [
        Path.cwd() / "external" / "PyMAF",
        Path(__file__).resolve().parent / "external" / "PyMAF",
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "demo.py").exists():
            return candidate.resolve()
    return None


def _python_has_modules(python_exe: str, modules: Tuple[str, ...]) -> Tuple[bool, List[str]]:
    probe = (
        "import importlib.util, json, sys\n"
        "mods = sys.argv[1:]\n"
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
        "print(json.dumps(missing))\n"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe, *modules],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        missing = json.loads(proc.stdout.strip() or "[]")
        return len(missing) == 0, [str(m) for m in missing]
    except Exception:
        return False, list(modules)


def _python_runtime_import_check(python_exe: str) -> Tuple[bool, str]:
    probe = (
        "import os, sys\n"
        "if sys.platform.startswith('darwin') and os.environ.get('PYOPENGL_PLATFORM', '').lower() == 'egl':\n"
        "    os.environ.pop('PYOPENGL_PLATFORM', None)\n"
        "import json\n"
        "details = {}\n"
        "import cv2, numpy, torch, joblib, trimesh, smplx, pyrender, ultralytics\n"
        "details['numpy'] = getattr(numpy, '__version__', '')\n"
        "details['torch'] = getattr(torch, '__version__', '')\n"
        "details['cv2'] = getattr(cv2, '__version__', '')\n"
        "print(json.dumps(details))\n"
    )
    try:
        proc = subprocess.run(
            [python_exe, "-c", probe],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True, proc.stdout.strip()
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or repr(exc)
        return False, detail
    except Exception as exc:
        return False, repr(exc)


def _recursive_find(root: Optional[Path], name: str) -> Optional[Path]:
    if root is None or not root.exists():
        return None
    direct = root / name
    if direct.exists():
        return direct.resolve()
    try:
        for match in root.rglob(name):
            if match.is_file():
                return match.resolve()
    except Exception:
        return None
    return None


def _resolve_pymaf_assets(
    repo_dir: Optional[Path],
    checkpoint: str = "auto",
    data_dir: str = "auto",
) -> Dict[str, Any]:
    explicit_ckpt = _path_from_setting(checkpoint)
    explicit_data = _path_from_setting(data_dir)

    checkpoint_candidates: List[Path] = []
    if explicit_ckpt is not None:
        checkpoint_candidates.append(explicit_ckpt)
    checkpoint_candidates.extend([
        Path.cwd() / "weights" / "PyMAF_model_checkpoint.pt",
        Path.cwd() / "weights" / "pymaf_model_checkpoint.pt",
    ])
    if repo_dir is not None:
        checkpoint_candidates.extend([
            repo_dir / "data" / "pretrained_model" / "PyMAF_model_checkpoint.pt",
            repo_dir / "weights" / "PyMAF_model_checkpoint.pt",
        ])
    checkpoint_path = next((p.resolve() for p in checkpoint_candidates if p.exists()), None)

    search_roots: List[Path] = []
    if explicit_data is not None and explicit_data.exists():
        search_roots.append(explicit_data)
    if repo_dir is not None:
        search_roots.extend([repo_dir / "data", repo_dir, repo_dir.parent])
    search_roots.extend([Path.cwd(), Path.cwd() / "weights"])

    resolved: Dict[str, Optional[Path]] = {"checkpoint": checkpoint_path}
    for name in PYMAF_REQUIRED_FILES:
        match = None
        aliases = PYMAF_FILE_ALIASES.get(name, (name,))
        for alias in aliases:
            for root in search_roots:
                match = _recursive_find(root, alias)
                if match is not None:
                    break
            if match is not None:
                break
        resolved[name] = match

    smpl_dir = None
    for root in search_roots:
        candidate = root / "smpl"
        if candidate.exists():
            smpl_dir = candidate.resolve()
            break
    if smpl_dir is None and resolved["SMPL_NEUTRAL.pkl"] is not None:
        smpl_dir = Path(resolved["SMPL_NEUTRAL.pkl"]).parent.resolve()
    resolved["smpl_dir"] = smpl_dir

    missing = [
        name for name in (
            "checkpoint",
            "smpl_mean_params.npz",
            "mesh_downsampling.npz",
            "J_regressor_extra.npy",
            "SMPL_NEUTRAL.pkl",
        )
        if resolved.get(name) is None
    ]
    return resolved | {"missing": missing}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def _prepare_pymaf_runtime_tree(repo_dir: Path, assets: Dict[str, Any]) -> None:
    data_dir = repo_dir / "data"
    pretrained_dir = data_dir / "pretrained_model"
    smpl_dir = data_dir / "smpl"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = assets.get("checkpoint")
    if isinstance(checkpoint_path, Path) and checkpoint_path.exists():
        _link_or_copy(checkpoint_path, pretrained_dir / "PyMAF_model_checkpoint.pt")

    for name in ("smpl_mean_params.npz", "mesh_downsampling.npz", "J_regressor_extra.npy"):
        src = assets.get(name)
        if isinstance(src, Path) and src.exists():
            _link_or_copy(src, data_dir / name)

    smpl_root = assets.get("smpl_dir")
    if isinstance(smpl_root, Path) and smpl_root.exists():
        for name in ("SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"):
            src = assets.get(name) or (smpl_root / name)
            if isinstance(src, Path) and src.exists():
                _link_or_copy(src, smpl_dir / name)


def _inspect_smpl_model_file(model_path: Optional[Path]) -> Tuple[bool, str]:
    if model_path is None or not model_path.exists():
        return False, "SMPL_NEUTRAL.pkl is missing."
    resolved_name = model_path.resolve().name.lower()
    if resolved_name.startswith("basicmodel_neutral_lbs_10_207_0_v1."):
        return True, ""
    try:
        with open(model_path, "rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        if not isinstance(payload, dict):
            return False, f"SMPL model at '{model_path}' is not a dict payload."
        v_template = np.asarray(payload.get("v_template"))
        j_regressor = np.asarray(payload.get("J_regressor"))
        vertex_count = int(v_template.shape[0]) if v_template.ndim >= 2 else -1
        joint_rows = int(j_regressor.shape[0]) if j_regressor.ndim >= 2 else -1
        if vertex_count == 10475 or joint_rows == 55:
            return (
                False,
                f"'{model_path}' appears to be an SMPL-X model ({vertex_count} vertices / {joint_rows} joints). "
                "PyMAF_model_checkpoint.pt expects a real SMPL neutral model (6890 vertices / 24 joints) "
                "under data/smpl/SMPL_NEUTRAL.pkl.",
            )
        if vertex_count != 6890 or joint_rows < 24:
            return (
                False,
                f"'{model_path}' does not look like the expected SMPL neutral model "
                f"(found {vertex_count} vertices / {joint_rows} joints).",
            )
        return True, ""
    except Exception as exc:
        return False, f"Failed to inspect '{model_path}': {exc}"


def _pymaf_bbox_to_xyxy(bbox: Any) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 3:
        return [0, 0, 0, 0]
    cx = float(bbox[0])
    cy = float(bbox[1])
    side = float(bbox[2])
    half = 0.5 * side
    return [
        int(round(cx - half)),
        int(round(cy - half)),
        int(round(cx + half)),
        int(round(cy + half)),
    ]


def _pymaf_joints_to_coco_local(joints_3d: Any) -> List[List[float]]:
    arr = np.asarray(joints_3d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 17 or arr.shape[1] < 3:
        return []

    if arr.shape[0] >= 49:
        arr = arr[-24:][PYMAF_J24_TO_COCO]
    elif arr.shape[0] >= 24:
        arr = arr[:24][PYMAF_J24_TO_COCO]
    else:
        arr = arr[:17]

    # PyMAF joints arrive in camera-style coordinates where the image Y axis
    # points downward. Our pedestrian rig uses (forward, left, up), so the
    # vertical component must be flipped before we ground-align the pose.
    local = np.stack([arr[:, 2], -arr[:, 0], -arr[:, 1]], axis=1)

    hip_pts = [local[idx] for idx in (11, 12) if idx < local.shape[0]]
    pelvis = np.mean(np.stack(hip_pts, axis=0), axis=0) if hip_pts else np.mean(local, axis=0)
    local = local - pelvis[None, :]

    ground_indices = [15, 16, 13, 14]
    valid_ground = [local[idx, 2] for idx in ground_indices if idx < local.shape[0]]
    ground_z = float(min(valid_ground)) if valid_ground else float(np.min(local[:, 2]))
    local[:, 2] -= ground_z

    return [
        [round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4), 0.98]
        for pt in local
    ]


def _approximate_2d_keypoints_from_local_pose(
    local_pose: List[List[float]],
    bbox_xyxy: Sequence[int],
) -> List[List[float]]:
    if len(bbox_xyxy) < 4 or not local_pose:
        return []
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)

    arr = np.asarray([kp[:3] for kp in local_pose if len(kp) >= 3], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return []

    lat = arr[:, 1]
    up = arr[:, 2]
    lat_span = max(float(np.max(lat) - np.min(lat)), 1e-3)
    up_span = max(float(np.max(up) - np.min(up)), 1e-3)

    out: List[List[float]] = []
    for kp in local_pose[: len(COCO_KEYPOINT_NAMES)]:
        if len(kp) < 3:
            out.append([cx, y2, 0.0])
            continue
        _, lateral, vertical = [float(v) for v in kp[:3]]
        px = cx + (lateral / lat_span) * min(0.58 * bw, 0.42 * bh)
        py = y2 - (vertical / up_span) * (0.95 * bh)
        conf = float(kp[3]) if len(kp) >= 4 else 0.75
        out.append([round(px, 2), round(py, 2), round(conf, 4)])
    return out


class PyMAFAdapter:
    def __init__(
        self,
        python_exe: str = "auto",
        repo_dir: str = "auto",
        checkpoint: str = "auto",
        data_dir: str = "auto",
        detector: str = "yolov8",
        tracking_method: str = "bbox",
    ) -> None:
        self.python_exe = _resolve_pymaf_python(python_exe)
        self.repo_dir = _resolve_pymaf_repo_dir(repo_dir)
        self.assets = _resolve_pymaf_assets(self.repo_dir, checkpoint=checkpoint, data_dir=data_dir)
        self.detector = str(detector or "yolov8")
        self.tracking_method = str(tracking_method or "bbox")
        self.ready, self.reason = self._check_ready()

    def _check_ready(self) -> Tuple[bool, str]:
        if self.repo_dir is None:
            return False, "PyMAF repo not found under external/PyMAF."
        ok_modules, missing_modules = _python_has_modules(self.python_exe, PYMAF_REQUIRED_MODULES)
        if not ok_modules:
            install_hint = "Install them in the PyMAF env, e.g. pip install " + " ".join(missing_modules)
            return False, (
                f"PyMAF runtime missing Python modules in '{self.python_exe}': "
                f"{', '.join(missing_modules)}. {install_hint}"
            )
        missing_assets = list(self.assets.get("missing", []))
        if missing_assets:
            smplx_hint = ""
            if "SMPL_NEUTRAL.pkl" in missing_assets:
                smplx_candidate = None
                if self.repo_dir is not None:
                    smplx_candidate = self.repo_dir / "data" / "smplx" / "SMPLX_NEUTRAL.pkl"
                if smplx_candidate is not None and smplx_candidate.exists():
                    smplx_hint = (
                        f" Found SMPL-X at '{smplx_candidate}', but this PyMAF setup still needs "
                        "SMPL under data/smpl/SMPL_NEUTRAL.pkl."
                    )
            return False, (
                f"PyMAF runtime missing support files: {', '.join(missing_assets)}.{smplx_hint}"
            )
        smpl_ok, smpl_detail = _inspect_smpl_model_file(self.assets.get("SMPL_NEUTRAL.pkl"))
        if not smpl_ok:
            return False, f"PyMAF SMPL model validation failed. {smpl_detail}"
        ok_runtime, runtime_detail = _python_runtime_import_check(self.python_exe)
        if not ok_runtime:
            return False, (
                "PyMAF scientific stack is broken in "
                f"'{self.python_exe}'. "
                "The PyMAF runtime could not import its core Python stack. "
                f"Runtime import error: {runtime_detail}"
            )
        return True, ""

    def _demo_output_dir(self, cache_root: Path, video_path: Path) -> Path:
        return cache_root / "pymaf" / video_path.stem

    def _export_pickle_to_json(self, pkl_path: Path, json_path: Path) -> None:
        export_code = r"""
import json, pickle, sys
from pathlib import Path
try:
    import joblib
except Exception:
    joblib = None

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
if joblib is not None:
    data = joblib.load(src)
else:
    with src.open('rb') as f:
        data = pickle.load(f)

def conv(obj):
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): conv(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [conv(v) for v in obj]
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return obj

trimmed = {}
for key, value in data.items():
    trimmed[str(key)] = {
        'frame_ids': conv(value.get('frame_ids', [])),
        'bboxes': conv(value.get('bboxes', [])),
        'joints3d': conv(value.get('joints3d', [])),
        'pose': conv(value.get('pose', [])),
        'betas': conv(value.get('betas', [])),
        'orig_cam': conv(value.get('orig_cam', [])),
    }

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open('w') as f:
    json.dump(trimmed, f)
"""
        subprocess.run(
            [self.python_exe, "-c", export_code, str(pkl_path), str(json_path)],
            check=True,
            cwd=str(self.repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def run_video(
        self,
        video_path: str,
        cache_root: Path,
        max_frames: Optional[int] = None,
        render: bool = True,
    ) -> Dict[str, Any]:
        if not self.ready or self.repo_dir is None:
            return {"frames": {}, "rendered_video": None}

        src_video = Path(video_path).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        _prepare_pymaf_runtime_tree(self.repo_dir, self.assets)

        demo_out = self._demo_output_dir(cache_root, src_video)
        pkl_path = demo_out / "output.pkl"
        json_path = demo_out / "output_tracks.json"
        rendered_video = demo_out / f"{src_video.stem}_result.mp4"

        need_run = (not pkl_path.exists()) or (render and not rendered_video.exists())
        if need_run:
            cmd = [
                self.python_exe,
                "demo.py",
                "--vid_file", str(src_video),
                "--output_folder", str(demo_out.parent),
                "--checkpoint", str(self.assets["checkpoint"]),
                "--tracking_method", self.tracking_method,
                "--detector", self.detector,
                "--model_batch_size", "8",
                "--tracker_batch_size", "8",
            ]
            if not render:
                cmd.append("--no_render")
            env = os.environ.copy()
            if sys.platform.startswith("darwin") and str(env.get("PYOPENGL_PLATFORM", "")).strip().lower() == "egl":
                env.pop("PYOPENGL_PLATFORM", None)
            subprocess.run(
                cmd,
                check=True,
                cwd=str(self.repo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        if not json_path.exists() or json_path.stat().st_mtime < pkl_path.stat().st_mtime:
            self._export_pickle_to_json(pkl_path, json_path)

        with json_path.open("r") as f:
            tracklets = json.load(f)

        frames: Dict[int, List[Dict[str, Any]]] = {}
        for track_id, track_data in tracklets.items():
            frame_ids = track_data.get("frame_ids") or []
            bboxes = track_data.get("bboxes") or []
            joints3d = track_data.get("joints3d") or []
            poses = track_data.get("pose") or []
            betas = track_data.get("betas") or []
            cams = track_data.get("orig_cam") or []
            count = min(len(frame_ids), len(bboxes), len(joints3d))
            for idx in range(count):
                frame_idx = int(frame_ids[idx])
                if max_frames is not None and frame_idx >= int(max_frames):
                    continue
                bbox_xyxy = _pymaf_bbox_to_xyxy(bboxes[idx])
                local_pose = _pymaf_joints_to_coco_local(joints3d[idx])
                approx_keypoints = _approximate_2d_keypoints_from_local_pose(local_pose, bbox_xyxy)
                frames.setdefault(frame_idx, []).append(
                    {
                        "id": int(track_id),
                        "bbox": bbox_xyxy,
                        "confidence": 0.98,
                        "keypoints": approx_keypoints,
                        "keypoint_names": COCO_KEYPOINT_NAMES,
                        "pose_backend": "pymaf",
                        "source": "pymaf",
                        "pose_3d_local": local_pose,
                        "smpl_pose": poses[idx] if idx < len(poses) else [],
                        "smpl_betas": betas[idx] if idx < len(betas) else [],
                        "pymaf_track_id": int(track_id),
                        "pymaf_orig_cam": cams[idx] if idx < len(cams) else [],
                    }
                )
        return {
            "frames": frames,
            "rendered_video": str(rendered_video) if rendered_video.exists() else None,
        }


def _resolve_pose_model_path(requested: str) -> str:
    if requested and requested != "auto":
        path = Path(requested).expanduser()
        return str(path.resolve()) if path.exists() else requested
    resolved = resolve_existing_artifact(
        ("yolo26x-pose.pt", "yolo26l-pose.pt", "yolo26m-pose.pt", "yolo26s-pose.pt", "yolo26n-pose.pt")
    )
    if resolved:
        return resolved
    return "yolo26x-pose.pt"


def _resolve_pose_backend(requested: str = "auto", pymaf_ready: bool = False) -> str:
    backend = str(requested or "auto").strip().lower()
    if backend == "pymaf":
        return "pymaf" if pymaf_ready else "pymaf"
    if backend == "yolo":
        return "yolo"
    return "pymaf" if pymaf_ready else "yolo"


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


def _visible_keypoint_count(keypoints: List[List[float]]) -> int:
    return sum(1 for kp in keypoints if len(kp) >= 3 and float(kp[2]) >= KP_CONF_THRESHOLD)


def _pose_quality_score(ped: Dict[str, Any]) -> float:
    return float(ped.get("confidence", 0.0)) + 0.05 * float(_visible_keypoint_count(ped.get("keypoints", [])))


def _expand_bbox_xyxy(bbox: Sequence[Any], frame_w: int, frame_h: int, pad_frac: float = 0.16) -> List[int]:
    if len(bbox) < 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(2.0, pad_frac * bw)
    pad_y = max(2.0, pad_frac * bh)
    return [
        int(max(0, min(frame_w - 1, round(x1 - pad_x)))),
        int(max(0, min(frame_h - 1, round(y1 - pad_y)))),
        int(max(0, min(frame_w, round(x2 + pad_x)))),
        int(max(0, min(frame_h, round(y2 + pad_y)))),
    ]


def _compose_pymaf_visual(
    frame_bgr: np.ndarray,
    pymaf_render_bgr: Optional[np.ndarray],
    pedestrians: List[Dict[str, Any]],
) -> Optional[np.ndarray]:
    if pymaf_render_bgr is None or pymaf_render_bgr.shape[:2] != frame_bgr.shape[:2]:
        return None

    H, W = frame_bgr.shape[:2]
    vis = frame_bgr.copy()
    any_overlay = False

    for ped in pedestrians:
        bbox = ped.get("pymaf_bbox") or ped.get("bbox") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = _expand_bbox_xyxy(bbox, W, H, pad_frac=0.18)
        if x2 <= x1 or y2 <= y1:
            continue

        frame_roi = frame_bgr[y1:y2, x1:x2]
        pymaf_roi = pymaf_render_bgr[y1:y2, x1:x2]
        if frame_roi.size == 0 or pymaf_roi.size == 0:
            continue

        diff = cv2.absdiff(pymaf_roi, frame_roi)
        mask = np.max(diff, axis=2) >= 18
        if not np.any(mask):
            continue
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        roi = vis[y1:y2, x1:x2]
        roi[mask] = pymaf_roi[mask]
        vis[y1:y2, x1:x2] = roi
        any_overlay = True

    return vis if any_overlay else None


def _extract_pymaf_sprite_bgra(
    frame_bgr: np.ndarray,
    pymaf_render_bgr: Optional[np.ndarray],
    bbox_xyxy: Sequence[Any],
) -> Optional[np.ndarray]:
    if pymaf_render_bgr is None or pymaf_render_bgr.shape[:2] != frame_bgr.shape[:2]:
        return None
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _expand_bbox_xyxy(bbox_xyxy, W, H, pad_frac=0.20)
    if x2 <= x1 or y2 <= y1:
        return None

    frame_roi = frame_bgr[y1:y2, x1:x2]
    pymaf_roi = pymaf_render_bgr[y1:y2, x1:x2]
    if frame_roi.size == 0 or pymaf_roi.size == 0:
        return None

    diff = cv2.absdiff(pymaf_roi, frame_roi)
    diff_gray = np.max(diff, axis=2).astype(np.uint8)
    render_gray = cv2.cvtColor(pymaf_roi, cv2.COLOR_BGR2GRAY)
    render_sat = cv2.cvtColor(pymaf_roi, cv2.COLOR_BGR2HSV)[..., 1]

    mask = (
        ((diff_gray >= 14) & (render_gray >= 26))
        | ((render_gray >= 182) & (diff_gray >= 8))
        | ((render_sat >= 40) & (render_gray >= 24) & (diff_gray >= 10))
    ).astype(np.uint8)
    if int(mask.sum()) <= 10:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        keep = np.zeros_like(mask)
        cx_ref = 0.5 * float(mask.shape[1])
        cy_ref = 0.56 * float(mask.shape[0])
        best_idx = None
        best_score = -1e18
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < 18:
                continue
            x = float(stats[label_idx, cv2.CC_STAT_LEFT])
            y = float(stats[label_idx, cv2.CC_STAT_TOP])
            w = float(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            tall_bonus = min(h / max(w, 1.0), 4.0)
            center_penalty = math.hypot((cx - cx_ref) / max(mask.shape[1], 1), (cy - cy_ref) / max(mask.shape[0], 1))
            score = float(area) + 18.0 * tall_bonus - 55.0 * center_penalty
            if score > best_score:
                best_score = score
                best_idx = label_idx
        if best_idx is not None:
            keep[labels == int(best_idx)] = 1
        mask = keep
    if int(mask.sum()) <= 18:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None

    tight_x1 = max(0, int(xs.min()) - 2)
    tight_y1 = max(0, int(ys.min()) - 2)
    tight_x2 = min(mask.shape[1], int(xs.max()) + 3)
    tight_y2 = min(mask.shape[0], int(ys.max()) + 3)
    if tight_x2 <= tight_x1 or tight_y2 <= tight_y1:
        return None

    crop_bgr = pymaf_roi[tight_y1:tight_y2, tight_x1:tight_x2].copy()
    crop_alpha = (mask[tight_y1:tight_y2, tight_x1:tight_x2] * 255).astype(np.uint8)
    if crop_bgr.size == 0 or crop_alpha.size == 0 or int(np.count_nonzero(crop_alpha)) <= 10:
        return None

    crop_alpha = cv2.GaussianBlur(crop_alpha, (3, 3), 0)
    crop_alpha = np.where(crop_alpha >= 12, crop_alpha, 0).astype(np.uint8)
    if int(np.count_nonzero(crop_alpha)) <= 10:
        return None

    crop_rgb = np.zeros_like(crop_bgr)
    valid_mask = crop_alpha > 0
    crop_rgb[valid_mask] = crop_bgr[valid_mask]
    crop_bgra = np.zeros((crop_bgr.shape[0], crop_bgr.shape[1], 4), dtype=np.uint8)
    crop_bgra[..., :3] = crop_rgb
    crop_bgra[..., 3] = crop_alpha
    return crop_bgra


def _attach_pymaf_sprite_paths(
    frame_bgr: np.ndarray,
    pymaf_render_bgr: Optional[np.ndarray],
    pedestrians: List[Dict[str, Any]],
    sprite_dir: Path,
    frame_idx: int,
) -> None:
    if pymaf_render_bgr is None:
        return
    sprite_dir.mkdir(parents=True, exist_ok=True)
    for ped in pedestrians:
        if str(ped.get("pose_backend", "")).strip().lower() != "pymaf":
            continue
        bbox = ped.get("pymaf_bbox") or ped.get("bbox") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        sprite = _extract_pymaf_sprite_bgra(frame_bgr, pymaf_render_bgr, bbox)
        if sprite is None:
            continue
        sprite_token = ped.get("pymaf_track_id", ped.get("id", "ped"))
        sprite_path = sprite_dir / f"{int(frame_idx):06d}_{str(sprite_token)}.png"
        if cv2.imwrite(str(sprite_path), sprite):
            ped["pymaf_sprite_path"] = str(sprite_path.resolve())


def _load_gate_pedestrian_index(gate_json_path: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
    if not gate_json_path:
        return {}
    path = Path(gate_json_path).expanduser()
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            data = json.load(f)
    except Exception:
        return {}

    frames = data.get("frames", []) if isinstance(data, dict) else []
    index: Dict[int, List[Dict[str, Any]]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_idx = int(frame.get("frame_idx", len(index)))
        detections = frame.get("detections", []) if isinstance(frame.get("detections", []), list) else []
        ped_boxes: List[Dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            if str(det.get("class", "")).strip().lower() != "pedestrian":
                continue
            bbox = det.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            conf = float(det.get("confidence", 0.0) or 0.0)
            if conf < PROPOSAL_CONF_THRESHOLD:
                continue
            ped_boxes.append({
                "id": det.get("id", len(ped_boxes)),
                "bbox": [int(v) for v in bbox[:4]],
                "confidence": conf,
                "keypoints": [],
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "source": "detic_pedestrian_gate",
                "pose_backend": "pymaf",
            })
        if ped_boxes:
            index[frame_idx] = ped_boxes
    return index


# ─────────────────────────────────────────────────────────────────────────────
# Device resolver
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
# PoseEstimator
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Ultralytics YOLO based pedestrian pose estimator.

    Parameters
    ----------
    device : "auto" | "cuda" | "mps" | "cpu"
    model_path : ultralytics model name or path (default: auto)
    """

    def __init__(
        self,
        device: str = "auto",
        backend: str = "auto",
        model_path: str = "auto",
        proposal_model: str = "auto",
        pymaf_python: str = "auto",
        pymaf_repo: str = "auto",
        pymaf_checkpoint: str = "auto",
        pymaf_data: str = "auto",
        imgsz: int = POSE_BASE_IMGSZ,
        crop_imgsz: int = POSE_CROP_IMGSZ,
        use_detector_proposals: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self._pymaf = PyMAFAdapter(
            python_exe=pymaf_python,
            repo_dir=pymaf_repo,
            checkpoint=pymaf_checkpoint,
            data_dir=pymaf_data,
        )
        self.backend = _resolve_pose_backend(backend, pymaf_ready=self._pymaf.ready)
        self.model_path = _resolve_pose_model_path(model_path)
        self.imgsz = int(max(640, imgsz))
        self.crop_imgsz = int(max(640, crop_imgsz))
        if str(backend).strip().lower() == "pymaf" and not self._pymaf.ready:
            raise RuntimeError(f"[PoseEstimator] PyMAF requested but unavailable. {self._pymaf.reason}")
        elif self.backend == "pymaf":
            print(f"[PoseEstimator] PyMAF video backend ready via '{self._pymaf.python_exe}' using repo '{self._pymaf.repo_dir}'.")
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            print(f"[PoseEstimator] backend='{self.backend}'  YOLO pose model loaded ({self.model_path})  device='{self.device}'")
        except ImportError as exc:
            raise ImportError("ultralytics is required.  pip install ultralytics") from exc

        self._proposal_detector: Optional[ObjectDetector] = None
        if use_detector_proposals:
            try:
                self._proposal_detector = ObjectDetector(
                    device=device,
                    detector_backend="detic",
                    yolo_model=proposal_model,
                    sign_model=None,
                    imgsz=max(self.imgsz, 1280),
                )
                print("[PoseEstimator] DETIC-guided pedestrian proposals enabled.")
            except Exception as exc:
                try:
                    self._proposal_detector = ObjectDetector(
                        device=device,
                        detector_backend="auto",
                        yolo_model=proposal_model,
                        sign_model=None,
                        imgsz=max(self.imgsz, 1280),
                    )
                    print(
                        "[PoseEstimator] Detector-guided pedestrian proposals enabled "
                        f"with fallback backend ({type(exc).__name__}: {exc})."
                    )
                except Exception as fallback_exc:
                    print(f"[PoseEstimator] Detector-guided proposals unavailable — {fallback_exc}")

    @staticmethod
    def _merge_pymaf_frame(
        pedestrians: List[Dict[str, Any]],
        pymaf_entries: List[Dict[str, Any]],
        require_pymaf_match: bool = False,
    ) -> List[Dict[str, Any]]:
        if not pymaf_entries:
            return [] if require_pymaf_match else pedestrians

        merged: List[Dict[str, Any]] = []
        used_pymaf: set[int] = set()

        for ped in pedestrians:
            best_idx = None
            best_score = -1e9
            for idx, pymaf_ped in enumerate(pymaf_entries):
                if idx in used_pymaf:
                    continue
                iou = _bbox_iou(ped.get("bbox", [0, 0, 0, 0]), pymaf_ped.get("bbox", [0, 0, 0, 0]))
                pb = ped.get("bbox", [0, 0, 0, 0])
                qb = pymaf_ped.get("bbox", [0, 0, 0, 0])
                if len(pb) >= 4 and len(qb) >= 4:
                    pcx = 0.5 * (float(pb[0]) + float(pb[2]))
                    pcy = 0.5 * (float(pb[1]) + float(pb[3]))
                    qcx = 0.5 * (float(qb[0]) + float(qb[2]))
                    qcy = 0.5 * (float(qb[1]) + float(qb[3]))
                    pw = max(float(pb[2]) - float(pb[0]), 1.0)
                    ph = max(float(pb[3]) - float(pb[1]), 1.0)
                    center_cost = float(np.hypot((pcx - qcx) / pw, (pcy - qcy) / ph))
                else:
                    center_cost = 10.0
                score = 2.3 * float(iou) - 0.45 * center_cost
                if score > best_score:
                    best_score = score
                    best_idx = idx

            merged_ped = dict(ped)
            if best_idx is not None and best_score >= 0.05:
                pymaf_ped = pymaf_entries[best_idx]
                used_pymaf.add(best_idx)
                gate_bbox = list(merged_ped.get("bbox", []))
                merged_ped.update({
                    "bbox": pymaf_ped.get("bbox", merged_ped.get("bbox", [])),
                    "gate_bbox": gate_bbox,
                    "pymaf_bbox": pymaf_ped.get("bbox", merged_ped.get("bbox", [])),
                    "pose_backend": "pymaf",
                    "keypoints": pymaf_ped.get("keypoints", merged_ped.get("keypoints", [])),
                    "keypoint_names": pymaf_ped.get("keypoint_names", merged_ped.get("keypoint_names", COCO_KEYPOINT_NAMES)),
                    "pose_3d_local": pymaf_ped.get("pose_3d_local", []),
                    "smpl_pose": pymaf_ped.get("smpl_pose", []),
                    "smpl_betas": pymaf_ped.get("smpl_betas", []),
                    "pymaf_track_id": pymaf_ped.get("pymaf_track_id"),
                    "pymaf_orig_cam": pymaf_ped.get("pymaf_orig_cam", []),
                    "source": pymaf_ped.get("source", merged_ped.get("source", "pymaf")),
                    "confidence": max(
                        float(merged_ped.get("confidence", 0.0) or 0.0),
                        float(pymaf_ped.get("confidence", 0.0) or 0.0),
                    ),
                })
                merged.append(merged_ped)
            elif not require_pymaf_match:
                merged.append(merged_ped)

        merged.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        for idx, ped in enumerate(merged):
            ped["id"] = idx
        return merged

    def detect_pedestrian_proposals(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._proposal_detector is None:
            return []

        out: List[Dict[str, Any]] = []
        proposals = [
            det for det in self._proposal_detector.detect(frame_bgr)
            if det.cls == "pedestrian" and float(det.confidence) >= PROPOSAL_CONF_THRESHOLD
        ]
        detector_backend = str(getattr(self._proposal_detector, "_detector_backend", "auto")).strip().lower()
        source_name = "detic_pedestrian_gate" if detector_backend == "detic" else "pedestrian_gate"

        for idx, det in enumerate(proposals):
            out.append({
                "id": det.id if det.id is not None else idx,
                "bbox": list(det.bbox),
                "confidence": round(float(det.confidence), 4),
                "keypoints": [],
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "source": source_name,
                "pose_backend": self.backend,
            })
        return out

    def _run_pose_model(self, image_bgr: np.ndarray, imgsz: int) -> List[Dict[str, Any]]:
        results = self._model(
            image_bgr,
            verbose=False,
            device=self.device,
            classes=[0],
            imgsz=int(max(640, imgsz)),
        )

        pedestrians: List[Dict[str, Any]] = []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return pedestrians

        for idx, box in enumerate(result.boxes):
            conf = float(box.conf.item())
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

            keypoints_list: List[List[float]] = []
            if result.keypoints is not None and idx < len(result.keypoints):
                kp_data = result.keypoints[idx]
                kp_array = kp_data.data[0].cpu().numpy()
                for kp in kp_array:
                    keypoints_list.append([float(kp[0]), float(kp[1]), float(kp[2])])

            pedestrians.append({
                "id": idx,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "keypoints": keypoints_list,
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "pose_backend": self.backend,
            })

        return pedestrians

    def _estimate_from_proposals(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if self._proposal_detector is None:
            return []

        H, W = frame_bgr.shape[:2]
        out: List[Dict[str, Any]] = []
        proposals = [
            det for det in self._proposal_detector.detect(frame_bgr)
            if det.cls == "pedestrian" and float(det.confidence) >= PROPOSAL_CONF_THRESHOLD
        ]

        for det in proposals:
            x1, y1, x2, y2 = det.bbox
            pad_x = int(round((x2 - x1) * POSE_CROP_PAD_FRAC))
            pad_y = int(round((y2 - y1) * POSE_CROP_PAD_FRAC))
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(W, x2 + pad_x)
            cy2 = min(H, y2 + pad_y)
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            candidates = self._run_pose_model(crop, self.crop_imgsz)
            if not candidates:
                continue

            best = None
            best_score = -1e9
            for cand in candidates:
                bx1, by1, bx2, by2 = cand["bbox"]
                global_bbox = [bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1]
                score = _pose_quality_score(cand) + 0.75 * _bbox_iou(global_bbox, det.bbox)
                if score > best_score:
                    best_score = score
                    best = cand

            if best is None:
                continue

            keypoints = []
            for kp in best.get("keypoints", []):
                if len(kp) < 3:
                    continue
                keypoints.append([float(kp[0]) + cx1, float(kp[1]) + cy1, float(kp[2])])

            out.append({
                "id": det.id,
                "bbox": list(det.bbox),
                "confidence": round(max(float(det.confidence), float(best.get("confidence", 0.0))), 4),
                "keypoints": keypoints,
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "source": "detector_guided_pose",
                "pose_backend": self.backend,
            })

        return out

    @staticmethod
    def _merge_pose_estimates(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        flat = [ped for group in groups for ped in group]
        flat.sort(key=_pose_quality_score, reverse=True)

        for ped in flat:
            keep = True
            for idx, existing in enumerate(merged):
                if _bbox_iou(existing["bbox"], ped["bbox"]) >= 0.55:
                    if _pose_quality_score(ped) > _pose_quality_score(existing):
                        merged[idx] = ped
                    keep = False
                    break
            if keep:
                merged.append(ped)

        merged.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        for idx, ped in enumerate(merged):
            ped["id"] = idx
        return merged

    def estimate(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect pedestrians and estimate keypoints.

        Returns list of dicts with keys: id, bbox, confidence, keypoints, keypoint_names.
        """
        full_frame = self._run_pose_model(frame_bgr, self.imgsz)
        proposal_guided = self._estimate_from_proposals(frame_bgr)
        return self._merge_pose_estimates(proposal_guided, full_frame)

    def draw(self, frame_bgr: np.ndarray, pedestrians: List[Dict[str, Any]]) -> np.ndarray:
        """Draw skeleton overlays on the frame."""
        vis = frame_bgr.copy()

        for ped in pedestrians:
            x1, y1, x2, y2 = ped["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 140, 0), 2)

            label = f"ped {ped['confidence']:.2f}"
            backend = str(ped.get("pose_backend", "")).strip()
            if backend:
                label += f" {backend}"
            cv2.putText(vis, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1, cv2.LINE_AA)

            kps = ped.get("keypoints", [])
            if not kps:
                continue

            # Draw skeleton edges
            for (i, j) in SKELETON_EDGES:
                if i >= len(kps) or j >= len(kps):
                    continue
                xi, yi, ci = kps[i]
                xj, yj, cj = kps[j]
                if ci < KP_CONF_THRESHOLD or cj < KP_CONF_THRESHOLD:
                    continue
                cv2.line(vis, (int(xi), int(yi)), (int(xj), int(yj)),
                         (200, 200, 200), 2, cv2.LINE_AA)

            # Draw keypoints
            for k, kp in enumerate(kps):
                px, py, c = kp
                if c < KP_CONF_THRESHOLD:
                    continue
                color = _KP_COLORS[k] if k < len(_KP_COLORS) else (255, 255, 255)
                cv2.circle(vis, (int(px), int(py)), 4, color, -1)

        return vis


# ─────────────────────────────────────────────────────────────────────────────
# Safer video output helpers for macOS / VS Code
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
# PosePipeline — video processor
# ─────────────────────────────────────────────────────────────────────────────

class PosePipeline:
    """
    End-to-end video processor for pedestrian pose estimation.

    Outputs an annotated video and a JSON with per-frame keypoints.
    """

    def __init__(self, estimator: Optional[PoseEstimator] = None, device: str = "auto") -> None:
        self.estimator = estimator or PoseEstimator(device=device)

    @staticmethod
    def _open_writer(path: Path, fps: float, W: int, H: int) -> "SafeVideoWriter":
        return SafeVideoWriter(path, fps, W, H, vscode_compatible=True)

    def run(
        self,
        video_path: str,
        out_video: str,
        out_json: str,
        max_frames: Optional[int] = None,
        frame_skip: int = 1,
        gate_detections_json: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        print(f"  PosePipeline.run()")
        print(f"  Input  : {src}  ({src_W}x{src_H} @ {src_fps:.1f} fps)")
        print(f"  Output : {out_video_path}")
        print(f"  JSON   : {out_json_path}")
        print(f"{'='*64}\n")

        writer = self._open_writer(out_video_path, out_fps, src_W, src_H)
        final_video = out_video_path
        pymaf_frame_map: Dict[int, List[Dict[str, Any]]] = {}
        pymaf_render_cap: Optional[cv2.VideoCapture] = None
        sprite_dir = out_json_path.parent / "pose_sprites"
        gate_index = _load_gate_pedestrian_index(gate_detections_json)
        if gate_index:
            print(f"[PosePipeline] Loaded detector gate for {len(gate_index)} frame(s) from {gate_detections_json}.")

        if getattr(self.estimator, "backend", "yolo") == "pymaf":
            cache_root = out_json_path.parent
            try:
                pymaf_result = self.estimator._pymaf.run_video(
                    video_path=str(src),
                    cache_root=cache_root,
                    max_frames=max_frames,
                    render=True,
                )
                pymaf_frame_map = dict(pymaf_result.get("frames", {}))
                pymaf_render_video = pymaf_result.get("rendered_video")
                if pymaf_render_video:
                    pymaf_render_cap = cv2.VideoCapture(str(pymaf_render_video))
                    if not pymaf_render_cap.isOpened():
                        pymaf_render_cap.release()
                        pymaf_render_cap = None
                print(f"[PosePipeline] Loaded PyMAF pose tracks for {len(pymaf_frame_map)} frame(s).")
            except Exception as exc:
                pymaf_frame_map = {}
                pymaf_render_cap = None
                print(f"[PosePipeline] PyMAF video inference unavailable for this run; continuing with YOLO pose only. {exc}")

        all_records: List[Dict[str, Any]] = []
        total_dets = 0
        written = 0
        src_idx = 0
        t_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pymaf_vis = None
                if pymaf_render_cap is not None:
                    ok_vis, vis_frame = pymaf_render_cap.read()
                    if ok_vis and vis_frame is not None and vis_frame.size:
                        pymaf_vis = vis_frame

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                ts = src_idx / src_fps

                pymaf_entries = pymaf_frame_map.get(src_idx, []) if pymaf_frame_map else []
                if getattr(self.estimator, "backend", "yolo") == "pymaf":
                    detector_pedestrians = gate_index.get(src_idx)
                    if detector_pedestrians is None:
                        detector_pedestrians = self.estimator.detect_pedestrian_proposals(frame)
                    if pymaf_entries:
                        pedestrians = self.estimator._merge_pymaf_frame(
                            detector_pedestrians,
                            pymaf_entries,
                            require_pymaf_match=True,
                        )
                    else:
                        pedestrians = []
                else:
                    pedestrians = self.estimator.estimate(frame)
                    if pymaf_entries:
                        pedestrians = self.estimator._merge_pymaf_frame(
                            pedestrians,
                            pymaf_entries,
                        )
                if getattr(self.estimator, "backend", "yolo") == "pymaf" and pymaf_vis is not None and pedestrians:
                    _attach_pymaf_sprite_paths(frame, pymaf_vis, pedestrians, sprite_dir, src_idx)
                all_records.append({
                    "frame_idx": src_idx,
                    "timestamp_s": round(ts, 4),
                    "pedestrians": pedestrians,
                })
                total_dets += len(pedestrians)

                vis = None
                if getattr(self.estimator, "backend", "yolo") == "pymaf" and pymaf_vis is not None:
                    vis = _compose_pymaf_visual(frame, pymaf_vis, pedestrians)
                if vis is None:
                    vis = self.estimator.draw(frame, pedestrians)
                hud = f"Frame {written}  Peds: {len(pedestrians)}  Total: {total_dets}"
                cv2.putText(vis, hud, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 230, 255), 1, cv2.LINE_AA)

                writer.write(vis)
                written += 1
                src_idx += 1

                if written % 30 == 0:
                    elapsed = time.time() - t_start
                    fps_est = written / max(elapsed, 1e-6)
                    pct = src_idx / src_total * 100 if src_total > 0 else 0
                    print(f"  frame {src_idx:5d}/{src_total}  ({pct:5.1f}%)  "
                          f"{fps_est:5.1f} fps  peds={len(pedestrians)}",
                          end="\r", flush=True)

                if max_frames is not None and written >= max_frames:
                    print(f"\n  Reached max_frames={max_frames}.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            cap.release()
            if pymaf_render_cap is not None:
                pymaf_render_cap.release()
            final_video = writer.close()

        with open(out_json_path, "w") as f:
            json.dump({
                "source": str(src),
                "pose_backend": str(getattr(self.estimator, "backend", "yolo")),
                "frames_written": written,
                "total_detections": total_dets,
                "final_video": str(final_video),
                "frames": all_records,
            }, f, indent=2)

        elapsed_total = time.time() - t_start
        print(f"\n\n{'='*64}")
        print(f"  Done.  {written} frames, {total_dets} pedestrians")
        print(f"  Wall time: {elapsed_total:.1f}s ({written/max(elapsed_total,1e-6):.1f} fps)")
        print(f"  Output: {final_video}")
        print(f"  JSON:   {out_json_path}")
        print(f"{'='*64}")

        return all_records


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian pose estimation pipeline")
    parser.add_argument("--video", type=str, required=True, help="Input video path (.mp4)")
    parser.add_argument("--scene", default=None,
                        help="Scene id (e.g. scene10); inferred from path if omitted")
    parser.add_argument("--out-video", default=None,
                        help="Annotated output video path")
    parser.add_argument("--out-json", default=None,
                        help="Pose JSON path")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--model", default="auto",
                        help="YOLO pose model name or path. 'auto' resolves to YOLO26 pose weights/models only.")
    parser.add_argument("--backend", default="auto", choices=["auto", "yolo", "pymaf"],
                        help="Pose backend preference. 'pymaf' uses the external PyMAF repo when its Python environment and SMPL assets are available.")
    parser.add_argument("--pymaf-python", default="auto",
                        help="Python interpreter used to run the external PyMAF repo. Defaults to $PYMAF_PYTHON or the current interpreter.")
    parser.add_argument("--pymaf-repo", default="auto",
                        help="Path to the cloned PyMAF repository. 'auto' resolves external/PyMAF.")
    parser.add_argument("--pymaf-checkpoint", default="auto",
                        help="PyMAF checkpoint path. 'auto' searches weights/ and the repo data tree.")
    parser.add_argument("--pymaf-data", default="auto",
                        help="Optional PyMAF data root containing smpl/, smpl_mean_params.npz, mesh_downsampling.npz, etc.")
    parser.add_argument("--proposal-model", default="auto",
                        help="YOLO detector model used to generate pedestrian proposals for cropped pose inference.")
    parser.add_argument("--imgsz", type=int, default=POSE_BASE_IMGSZ,
                        help="Base full-frame pose inference size.")
    parser.add_argument("--crop-imgsz", type=int, default=POSE_CROP_IMGSZ,
                        help="Pose inference size for detector-guided pedestrian crops.")
    parser.add_argument("--no-detector-proposals", action="store_true",
                        help="Disable detector-guided pedestrian crop proposals and run pose on the full frame only.")
    parser.add_argument("--gate-detections-json", default="auto",
                        help="Pedestrian detection JSON from object_detection.py used to gate PyMAF poses. 'auto' resolves to output/<scene>/detections/detections.json.")

    args = parser.parse_args()

    scene_name = infer_scene_name(args.scene, args.video, args.out_video, args.out_json)
    output_layout = scene_output_layout(scene_name, create=True)

    out_video = (
        str(Path(args.out_video).resolve()) if args.out_video
        else str((output_layout.detections / "pose_output.mp4").resolve())
    )
    out_json = (
        str(Path(args.out_json).resolve()) if args.out_json
        else str((output_layout.detections / "pose_keypoints.json").resolve())
    )
    gate_detections_json = None
    if str(args.gate_detections_json).strip().lower() == "auto":
        gate_candidate = output_layout.detections / "detections.json"
        if gate_candidate.exists():
            gate_detections_json = str(gate_candidate.resolve())
    elif str(args.gate_detections_json).strip():
        gate_detections_json = str(Path(args.gate_detections_json).expanduser().resolve())

    estimator = PoseEstimator(
        device=args.device,
        backend=args.backend,
        model_path=args.model,
        proposal_model=args.proposal_model,
        pymaf_python=args.pymaf_python,
        pymaf_repo=args.pymaf_repo,
        pymaf_checkpoint=args.pymaf_checkpoint,
        pymaf_data=args.pymaf_data,
        imgsz=args.imgsz,
        crop_imgsz=args.crop_imgsz,
        use_detector_proposals=not args.no_detector_proposals,
    )
    pipe = PosePipeline(estimator=estimator)
    pipe.run(
        args.video,
        out_video=out_video,
        out_json=out_json,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        gate_detections_json=gate_detections_json,
    )

    if not args.out_video:
        mirror_stage_output(out_video, scene_name, "detections", Path(out_video).name)
    if not args.out_json:
        mirror_stage_output(out_json, scene_name, "detections", Path(out_json).name)
