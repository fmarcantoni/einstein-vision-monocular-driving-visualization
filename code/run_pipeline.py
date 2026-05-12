"""
run_pipeline.py
===============

Offline-first end-to-end runner for the project pipeline.

This orchestrates the existing stage scripts in a reproducible order and keeps
their individual CLIs intact. By default it:

* uses the local project weights before any online model names
* reuses cached depth archives instead of forcing a model download
* uses Farneback flow for reliable CPU/offline execution
* skips stages whose primary outputs already exist unless ``--overwrite`` is set
"""

from __future__ import annotations

import argparse
import cv2
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

from project_setup import CONFIG, resolve_existing_artifact, scene_output_layout


def _normalise_executable_path(raw_path: str) -> str:
    """
    Return an absolute executable path without dereferencing virtualenv symlinks.

    ``Path.resolve()`` follows symlinks, which can turn ``.venv/bin/python`` into
    the system interpreter and silently drop the environment we actually want.
    """
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.absolute()
    if not path.exists():
        raise FileNotFoundError(f"Executable not found: {path}")
    return str(path)


def _resolve_base_python(raw_path: str) -> str:
    raw = str(raw_path or "auto").strip()
    if raw and raw.lower() != "auto":
        return _normalise_executable_path(raw)

    candidates = (
        Path(".venv/bin/python"),
        Path(".venv_mrcnn_cpu/bin/python"),
        Path(sys.executable),
    )
    for candidate in candidates:
        try:
            if candidate.exists():
                return _normalise_executable_path(str(candidate))
        except Exception:
            continue
    return _normalise_executable_path(sys.executable)


def _resolve_stage_python(
    raw_path: str,
    *,
    stage: str,
    base_python: str,
    detector_backend: str = "auto",
    prefer_detic: bool = False,
) -> str:
    raw = str(raw_path or "auto").strip()
    if raw and raw.lower() != "auto":
        return _normalise_executable_path(raw)

    candidates: list[Path | str] = []
    detic_candidates = [
        Path.home() / "anaconda3" / "envs" / "detic" / "bin" / "python",
        Path.home() / "miniconda3" / "envs" / "detic" / "bin" / "python",
        Path("detic/bin/python"),
        Path(".venv_detic/bin/python"),
    ]

    if stage == "object":
        if detector_backend in {"auto", "detic"}:
            candidates.extend(detic_candidates)
        candidates.extend(
            [
                Path(".venv_mrcnn_cpu/bin/python"),
                Path(".venv/bin/python"),
                base_python,
            ]
        )
    elif stage == "traffic":
        if prefer_detic:
            candidates.extend(detic_candidates)
        candidates.extend(
            [
                Path(".venv/bin/python"),
                Path(".venv_mrcnn_cpu/bin/python"),
                base_python,
            ]
        )
    elif stage == "lane":
        candidates.extend(
            [
                Path(".venv/bin/python"),
                Path(".venv_mrcnn_cpu/bin/python"),
                base_python,
            ]
        )
    else:
        candidates.extend(
            [
                Path(".venv/bin/python"),
                Path(".venv_mrcnn_cpu/bin/python"),
                base_python,
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalised = _normalise_executable_path(str(candidate))
        except Exception:
            continue
        if normalised in seen:
            continue
        seen.add(normalised)
        return normalised
    return _normalise_executable_path(base_python)


def _discover_video(scene: str, view: str, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        return path

    undist_dir = CONFIG["SEQUENCES_DIR"] / scene / "Undist"
    patterns = (
        f"*{view}*undistort*.mp4",
        f"*{view}*.mp4",
        "*.mp4",
    )
    for pattern in patterns:
        matches = sorted(undist_dir.glob(pattern))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"No undistorted video found under {undist_dir}")


def _discover_depth_npz(scene: str) -> Optional[Path]:
    layout = scene_output_layout(scene, create=False)
    candidates = (
        layout.depth / "depth_maps.npz",
        layout.depth / "depth_frames.npz",
        CONFIG["SEQUENCES_DIR"] / scene / "Depth" / "depth_maps.npz",
        CONFIG["SEQUENCES_DIR"] / scene / "Depth" / "depth_frames.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _append_common_video_args(
    cmd: list[str],
    *,
    scene: str,
    video: Path,
    device: str,
    max_frames: Optional[int],
    frame_skip: int,
) -> list[str]:
    cmd.extend(["--scene", scene, "--video", str(video), "--device", device, "--frame-skip", str(frame_skip)])
    if max_frames is not None:
        cmd.extend(["--max-frames", str(max_frames)])
    return cmd


def _run_command(cmd: list[str]) -> None:
    print(f"\n[pipeline] $ {' '.join(shlex.quote(part) for part in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def _skip_or_run(
    *,
    primary_output: Path,
    overwrite: bool,
    cmd: list[str],
    label: str,
) -> None:
    if primary_output.exists() and not overwrite:
        print(f"[pipeline] Reusing existing {label}: {primary_output}")
        return
    _run_command(cmd)
    if not primary_output.exists():
        raise RuntimeError(f"{label} did not produce the expected output: {primary_output}")


def _json_frame_count(json_path: Path) -> int:
    try:
        payload = json.loads(json_path.read_text())
    except Exception:
        return 0
    frames = payload.get("frames", [])
    return len(frames) if isinstance(frames, list) else 0


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def _lane_output_is_usable(
    lane_json: Path,
    *,
    video_path: Path,
    frame_skip: int,
    max_frames: Optional[int],
) -> bool:
    if not lane_json.exists():
        return False
    actual = _json_frame_count(lane_json)
    if actual <= 0:
        return False
    if max_frames is not None:
        expected = max(1, int(max_frames))
    else:
        total_frames = _video_frame_count(video_path)
        expected = max(1, int(math.ceil(total_frames / max(frame_skip, 1)))) if total_frames > 0 else 0
    if expected <= 0:
        return actual >= 12
    return actual >= max(12, int(0.65 * expected))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full project pipeline with offline-safe defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene", default="scene1", help="Scene id under P3Data/Sequences/")
    parser.add_argument("--view", default="front", choices=["front", "back", "left", "right"])
    parser.add_argument("--video", default=None, help="Optional explicit undistorted video path")
    parser.add_argument("--python", default="auto", help="Base Python interpreter. 'auto' prefers local project envs.")
    parser.add_argument("--object-python", default="auto", help="Interpreter for object_detection.py. 'auto' prefers the Detic env when requested/available.")
    parser.add_argument("--traffic-python", default="auto", help="Interpreter for traffic_light_detector.py.")
    parser.add_argument("--lane-python", default="auto", help="Interpreter for lane_detection.py.")
    parser.add_argument("--pose-python", default="auto", help="Interpreter for pose_estimation.py.")
    parser.add_argument("--pose-backend", default="auto", choices=["auto", "yolo", "pymaf"],
                        help="Pose backend preference for pose_estimation.py.")
    parser.add_argument("--pymaf-python", default="auto", help="Python interpreter used for the external PyMAF repo.")
    parser.add_argument("--pymaf-repo", default="auto", help="Path to the cloned PyMAF repo.")
    parser.add_argument("--pymaf-checkpoint", default="auto", help="PyMAF checkpoint path.")
    parser.add_argument("--pymaf-data", default="auto", help="PyMAF data root containing SMPL assets and support files.")
    parser.add_argument("--depth-python", default="auto", help="Interpreter for depth_estimation.py.")
    parser.add_argument("--flow-python", default="auto", help="Interpreter for optical_flow.py.")
    parser.add_argument("--collision-python", default="auto", help="Interpreter for collision_detection.py.")
    parser.add_argument("--vehicle3d-python", default="auto", help="Interpreter for vehicle_3d_detection.py.")
    parser.add_argument("--assemble-python", default="auto", help="Interpreter for scene_assembler.py.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--frame-skip", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional stage frame cap for debugging")
    parser.add_argument("--overwrite", action="store_true", help="Recompute stages even when outputs already exist")
    parser.add_argument(
        "--depth-mode",
        default="reuse",
        choices=["reuse", "recompute", "skip"],
        help="Depth policy. 'reuse' only uses an existing depth npz; 'recompute' runs depth_estimation.py.",
    )
    parser.add_argument(
        "--detector-backend",
        default="auto",
        choices=["yolo", "auto", "detic"],
        help="General object detector backend for object_detection.py",
    )
    parser.add_argument(
        "--sign-backend",
        default="auto",
        choices=["auto", "yolo", "faster_rcnn", "detr", "ensemble"],
        help="Traffic-sign backend for object_detection.py",
    )
    parser.add_argument("--sign-model", default="auto", help="Optional dedicated YOLO sign model path; 'auto' resolves local weights.")
    parser.add_argument("--detic-repo", default="auto", help="Detic repo root for object_detection.py.")
    parser.add_argument("--detic-config", default="auto", help="Detic config path for object_detection.py.")
    parser.add_argument("--detic-weights", default="auto", help="Detic checkpoint path for object_detection.py.")
    parser.add_argument("--detr-model", default="auto", help="DETR sign checkpoint path for object_detection.py.")
    parser.add_argument("--traffic-detector-model", default="auto", help="YOLO detector checkpoint for traffic_light_detector.py.")
    parser.add_argument("--traffic-classifier-weights", default="auto", help="Traffic-light state classifier for traffic_light_detector.py.")
    parser.add_argument("--traffic-detic-color-checker", default="auto", choices=["auto", "on", "off"],
                        help="Use Detic as a secondary traffic-light color checker.")
    parser.add_argument("--traffic-detic-repo", default="auto", help="Detic repo root for traffic_light_detector.py.")
    parser.add_argument("--traffic-detic-config", default="auto", help="Detic config path for traffic_light_detector.py.")
    parser.add_argument("--traffic-detic-weights", default="auto", help="Detic checkpoint path for traffic_light_detector.py.")
    parser.add_argument("--traffic-detic-min-confidence", type=float, default=0.18,
                        help="Minimum Detic confidence for traffic-light color checking.")
    parser.add_argument(
        "--flow-model",
        default="farneback",
        choices=["farneback", "raft", "auto"],
        help="Optical flow backend for optical_flow.py",
    )
    parser.add_argument("--parked-speed-thresh", type=float, default=1.15, help="Residual motion threshold for parked vs moving vehicle classification in optical_flow.py.")
    parser.add_argument("--parked-history", type=int, default=4, help="Number of stable low-motion observations before a vehicle is marked parked.")
    parser.add_argument("--warn-max-distance-m", type=float, default=22.0, help="Primary collision warning distance in metres.")
    parser.add_argument(
        "--deepbox-backend",
        default="skhadem",
        choices=["geometry", "auto", "cersar", "lzccccc", "skhadem"],
        help="3D vehicle box backend for vehicle_3d_detection.py",
    )
    parser.add_argument("--cersar-repo", default=None, help="Optional path to external/3D_detection.")
    parser.add_argument("--cersar-weights", default="auto", help="Path to cersar 3D_detection weights (.h5).")
    parser.add_argument("--skhadem-repo", default=None, help="Optional path to external/3D-BoundingBox.")
    parser.add_argument("--skhadem-weights", default="auto", help="Path to skhadem 3D-BoundingBox weights (.pk/.pkl).")
    parser.add_argument(
        "--car-subclass-model",
        default="auto",
        help="Car subtype model for object_detection.py and vehicle_3d_detection.py. Use 'none' to disable.",
    )
    parser.add_argument("--lane-maskrcnn-backend", default="torchvision", choices=["torchvision", "omonsun"])
    parser.add_argument("--lane-maskrcnn-weights", default="auto")
    parser.add_argument("--lane-maskrcnn-score-threshold", type=float, default=0.70)
    parser.add_argument("--lane-maskrcnn-mask-threshold", type=float, default=0.45)
    parser.add_argument("--lane-maskrcnn-max-dets", type=int, default=64)
    parser.add_argument("--skip-traffic-lights", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--skip-collision", action="store_true")
    parser.add_argument("--skip-vehicle-3d", action="store_true")
    parser.add_argument("--render", action="store_true", help="Run Blender after scene assembly")
    parser.add_argument("--blender-exe", default=str(CONFIG["BLENDER_EXECUTABLE"]), help="Blender executable path")
    args = parser.parse_args()

    scene = str(args.scene).strip().lower()
    python_exe = _resolve_base_python(args.python)
    object_python = _resolve_stage_python(
        args.object_python,
        stage="object",
        base_python=python_exe,
        detector_backend=args.detector_backend,
    )
    traffic_python = _resolve_stage_python(
        args.traffic_python,
        stage="traffic",
        base_python=python_exe,
        prefer_detic=str(args.traffic_detic_color_checker).strip().lower() != "off",
    )
    lane_python = _resolve_stage_python(args.lane_python, stage="lane", base_python=python_exe)
    pose_python = _resolve_stage_python(args.pose_python, stage="pose", base_python=python_exe)
    depth_python = _resolve_stage_python(args.depth_python, stage="depth", base_python=python_exe)
    flow_python = _resolve_stage_python(args.flow_python, stage="flow", base_python=python_exe)
    collision_python = _resolve_stage_python(args.collision_python, stage="collision", base_python=python_exe)
    vehicle3d_python = _resolve_stage_python(args.vehicle3d_python, stage="vehicle3d", base_python=python_exe)
    assemble_python = _resolve_stage_python(args.assemble_python, stage="assemble", base_python=python_exe)
    video_path = _discover_video(scene, args.view, args.video)
    layout = scene_output_layout(scene, create=True)

    if (
        str(args.traffic_python).strip().lower() == "auto"
        and str(args.traffic_detic_color_checker).strip().lower() != "off"
        and object_python
    ):
        traffic_python = object_python

    yolo_model = resolve_existing_artifact(("yolo26x.pt", "yolo11x.pt", "yolov8x.pt", "yolov8n.pt")) or "yolo26x.pt"
    sign_model = None if str(args.sign_model).strip().lower() == "none" else (
        resolve_existing_artifact(("best.pt", "traffic_sign_best.pt")) if str(args.sign_model).strip().lower() == "auto" else str(args.sign_model)
    )
    depth_npz = _discover_depth_npz(scene)

    det_json = layout.detections / "detections.json"
    tl_json = layout.traffic_lights / "traffic_lights.json"
    lane_json = layout.lanes / "combined.json"
    pose_json = layout.detections / "pose_keypoints.json"
    flow_json = layout.flow / "flow_results.json"
    collision_json = layout.detections / "collision_detection.json"
    veh3d_json = layout.detections / "vehicle_3d_detections.json"
    assembled_json = layout.scene_data / "scene_assembled.json"

    print("\n" + "=" * 72)
    print("  Project Pipeline Runner")
    print(f"  scene        : {scene}")
    print(f"  view         : {args.view}")
    print(f"  video        : {video_path}")
    print(f"  python(base) : {python_exe}")
    print(f"  python(obj)  : {object_python}")
    print(f"  python(lane) : {lane_python}")
    print(f"  python(coll) : {collision_python}")
    print(f"  python(3d)   : {vehicle3d_python}")
    print(f"  device       : {args.device}")
    print(f"  frame_skip   : {args.frame_skip}")
    print(f"  max_frames   : {args.max_frames if args.max_frames is not None else 'all'}")
    print(f"  cached depth : {depth_npz or 'none'}")
    print("=" * 72, flush=True)

    det_cmd = _append_common_video_args(
        [object_python, "object_detection.py"],
        scene=scene,
        video=video_path,
        device=args.device,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )
    det_cmd.extend(
        [
            "--detector-backend",
            args.detector_backend,
            "--sign-backend",
            args.sign_backend,
            "--yolo-model",
            yolo_model,
            "--car-subclass-model",
            args.car_subclass_model,
            "--detic-repo",
            args.detic_repo,
            "--detic-config",
            args.detic_config,
            "--detic-weights",
            args.detic_weights,
            "--detr-model",
            args.detr_model,
        ]
    )
    if sign_model:
        det_cmd.extend(["--sign-model", sign_model])
    det_cmd.extend(["--out-json", str(det_json), "--out-video", str(layout.detections / "detection_output.mp4")])
    _skip_or_run(primary_output=det_json, overwrite=args.overwrite, cmd=det_cmd, label="object detections")

    if not args.skip_traffic_lights:
        tl_cmd = _append_common_video_args(
            [traffic_python, "traffic_light_detector.py"],
            scene=scene,
            video=video_path,
            device=args.device,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        tl_cmd.extend([
            "--yolo-model",
            args.traffic_detector_model,
            "--classifier-weights",
            args.traffic_classifier_weights,
            "--detic-color-checker",
            args.traffic_detic_color_checker,
            "--detic-repo",
            (args.traffic_detic_repo if str(args.traffic_detic_repo).strip().lower() != "auto" else args.detic_repo),
            "--detic-config",
            (args.traffic_detic_config if str(args.traffic_detic_config).strip().lower() != "auto" else args.detic_config),
            "--detic-weights",
            (args.traffic_detic_weights if str(args.traffic_detic_weights).strip().lower() != "auto" else args.detic_weights),
            "--detic-min-confidence",
            str(args.traffic_detic_min_confidence),
            "--out-json",
            str(tl_json),
            "--out-video",
            str(layout.traffic_lights / "traffic_lights.mp4"),
        ])
        _skip_or_run(primary_output=tl_json, overwrite=args.overwrite, cmd=tl_cmd, label="traffic lights")

    lane_cmd = _append_common_video_args(
        [lane_python, "lane_detection.py"],
        scene=scene,
        video=video_path,
        device=args.device,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )
    lane_cmd.extend(
        [
            "--maskrcnn-backend",
            args.lane_maskrcnn_backend,
            "--maskrcnn-weights",
            args.lane_maskrcnn_weights,
            "--maskrcnn-score-threshold",
            str(args.lane_maskrcnn_score_threshold),
            "--maskrcnn-mask-threshold",
            str(args.lane_maskrcnn_mask_threshold),
            "--maskrcnn-max-dets",
            str(args.lane_maskrcnn_max_dets),
            "--out-json",
            str(lane_json),
            "--out-video",
            str(layout.lanes / "combined.mp4"),
        ]
    )
    lane_overwrite = bool(args.overwrite)
    if not lane_overwrite and lane_json.exists():
        if not _lane_output_is_usable(
            lane_json,
            video_path=video_path,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
        ):
            print(f"[pipeline] Existing lane output looks incomplete; recomputing: {lane_json}")
            lane_overwrite = True
    _skip_or_run(primary_output=lane_json, overwrite=lane_overwrite, cmd=lane_cmd, label="lane and road segmentation")

    if not args.skip_pose:
        pose_cmd = _append_common_video_args(
            [pose_python, "pose_estimation.py"],
            scene=scene,
            video=video_path,
            device=args.device,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        pose_cmd.extend([
            "--backend", args.pose_backend,
            "--pymaf-python", args.pymaf_python,
            "--pymaf-repo", args.pymaf_repo,
            "--pymaf-checkpoint", args.pymaf_checkpoint,
            "--pymaf-data", args.pymaf_data,
            "--model", "auto",
            "--proposal-model", yolo_model,
            "--out-json", str(pose_json),
            "--out-video", str(layout.detections / "pose_output.mp4"),
        ])
        _skip_or_run(primary_output=pose_json, overwrite=args.overwrite, cmd=pose_cmd, label="pedestrian pose")

    if args.depth_mode == "recompute":
        depth_npz = layout.depth / "depth_maps.npz"
        depth_cmd = _append_common_video_args(
            [depth_python, "depth_estimation.py"],
            scene=scene,
            video=video_path,
            device=args.device,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        depth_cmd.extend(["--out", str(layout.depth / "depth_output.mp4"), "--out-npz", str(depth_npz)])
        _skip_or_run(primary_output=depth_npz, overwrite=args.overwrite, cmd=depth_cmd, label="depth maps")
    elif args.depth_mode == "skip":
        depth_npz = None
    elif depth_npz is None:
        print("[pipeline] No cached depth archive found; continuing without running depth_estimation.py.", flush=True)

    if not args.skip_flow:
        flow_cmd = _append_common_video_args(
            [flow_python, "optical_flow.py"],
            scene=scene,
            video=video_path,
            device=args.device,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        flow_cmd.extend(
            [
                "--model-type",
                args.flow_model,
                "--detections-json",
                str(det_json),
                "--parked-speed-thresh",
                str(args.parked_speed_thresh),
                "--parked-history",
                str(args.parked_history),
                "--out",
                str(layout.flow / "flow_output.mp4"),
                "--out-json",
                str(flow_json),
            ]
        )
        _skip_or_run(primary_output=flow_json, overwrite=args.overwrite, cmd=flow_cmd, label="optical flow")

    if not args.skip_collision:
        collision_cmd = _append_common_video_args(
            [collision_python, "collision_detection.py"],
            scene=scene,
            video=video_path,
            device=args.device,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        collision_cmd.extend(
            [
                "--detections-json",
                str(det_json),
                "--warn-max-distance-m",
                str(args.warn_max_distance_m),
                "--out-video",
                str(layout.detections / "collision_output.mp4"),
                "--out-json",
                str(collision_json),
            ]
        )
        if depth_npz is not None and Path(depth_npz).exists():
            collision_cmd.extend(["--depth-npz", str(depth_npz)])
        _skip_or_run(primary_output=collision_json, overwrite=args.overwrite, cmd=collision_cmd, label="collision warnings")

    if not args.skip_vehicle_3d:
        veh3d_cmd = _append_common_video_args(
            [vehicle3d_python, "vehicle_3d_detection.py"],
            scene=scene,
            video=video_path,
            device=args.device,
            max_frames=args.max_frames,
            frame_skip=args.frame_skip,
        )
        veh3d_cmd.extend(
            [
                "--view",
                args.view,
                "--yolo-model",
                yolo_model,
                "--car-subclass-model",
                args.car_subclass_model,
                "--deepbox-backend",
                args.deepbox_backend,
                "--cersar-repo",
                args.cersar_repo or "auto",
                "--cersar-weights",
                args.cersar_weights,
                "--skhadem-weights",
                args.skhadem_weights,
                "--detections-json",
                str(det_json),
                "--out-json",
                str(veh3d_json),
                "--out-video",
                str(layout.detections / "vehicle_3d_output.mp4"),
            ]
        )
        if args.skhadem_repo:
            veh3d_cmd.extend(["--skhadem-repo", args.skhadem_repo])
        if depth_npz is not None:
            veh3d_cmd.extend(["--depth-npz", str(depth_npz)])
        _skip_or_run(primary_output=veh3d_json, overwrite=args.overwrite, cmd=veh3d_cmd, label="vehicle 3D detections")

    assemble_cmd = [assemble_python, "scene_assembler.py", "--scene", scene, "--view", args.view, "--det-json", str(det_json), "--lane-json", str(lane_json), "--out", str(assembled_json)]
    if args.max_frames is not None:
        assemble_cmd.extend(["--max-frames", str(args.max_frames)])
    if tl_json.exists() and not args.skip_traffic_lights:
        assemble_cmd.extend(["--tl-json", str(tl_json)])
    if collision_json.exists() and not args.skip_collision:
        assemble_cmd.extend(["--collision-json", str(collision_json)])
    if pose_json.exists() and not args.skip_pose:
        # scene_assembler auto-discovers pose JSON, but we keep the path explicit in the run summary by mirroring its discovery layout
        pass
    if depth_npz is not None and Path(depth_npz).exists():
        assemble_cmd.extend(["--depth-npz", str(depth_npz)])
    _skip_or_run(primary_output=assembled_json, overwrite=args.overwrite, cmd=assemble_cmd, label="assembled scene")

    if args.render:
        blender_exe = Path(args.blender_exe).expanduser()
        if not blender_exe.exists():
            raise FileNotFoundError(f"Blender executable not found: {blender_exe}")
        render_cmd = [
            str(blender_exe.resolve()),
            "--background",
            "--python",
            "blender.py",
            "--",
            "--scene-json",
            str(assembled_json),
            "--camera-mode",
            "chase",
            "--export-reference-frames",
            "--compose-collage",
            "--render",
        ]
        _run_command(render_cmd)

    manifest_path = layout.scene_data / "pipeline_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "scene": scene,
                "view": args.view,
                "video": str(video_path),
                "device": args.device,
                "base_python": python_exe,
                "stage_pythons": {
                    "object_detection": object_python,
                    "traffic_lights": traffic_python,
                    "lane_detection": lane_python,
                    "pose_estimation": pose_python,
                    "depth_estimation": depth_python,
                    "optical_flow": flow_python,
                    "collision_detection": collision_python,
                    "vehicle_3d_detection": vehicle3d_python,
                    "scene_assembler": assemble_python,
                },
                "backends": {
                    "detector_backend": args.detector_backend,
                    "sign_backend": args.sign_backend,
                    "traffic_detector_model": args.traffic_detector_model,
                    "traffic_classifier_weights": args.traffic_classifier_weights,
                    "traffic_detic_color_checker": args.traffic_detic_color_checker,
                    "flow_model": args.flow_model,
                    "warn_max_distance_m": args.warn_max_distance_m,
                    "deepbox_backend": args.deepbox_backend,
                    "lane_maskrcnn_backend": args.lane_maskrcnn_backend,
                },
                "artifacts": {
                    "detections_json": str(det_json),
                    "traffic_lights_json": str(tl_json),
                    "lane_json": str(lane_json),
                    "pose_json": str(pose_json),
                    "flow_json": str(flow_json),
                    "collision_json": str(collision_json),
                    "vehicle_3d_json": str(veh3d_json),
                    "depth_npz": str(depth_npz) if depth_npz is not None else None,
                    "assembled_json": str(assembled_json),
                },
            },
            f,
            indent=2,
        )

    print("\n[pipeline] Finished.", flush=True)
    print(f"[pipeline] detections   : {det_json}", flush=True)
    if not args.skip_traffic_lights:
        print(f"[pipeline] traffic      : {tl_json}", flush=True)
    print(f"[pipeline] lanes        : {lane_json}", flush=True)
    if not args.skip_pose:
        print(f"[pipeline] pose         : {pose_json}", flush=True)
    if not args.skip_flow:
        print(f"[pipeline] flow         : {flow_json}", flush=True)
    if not args.skip_collision:
        print(f"[pipeline] collision    : {collision_json}", flush=True)
    if not args.skip_vehicle_3d:
        print(f"[pipeline] vehicle_3d   : {veh3d_json}", flush=True)
    print(f"[pipeline] scene        : {assembled_json}", flush=True)
    print(f"[pipeline] manifest     : {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
