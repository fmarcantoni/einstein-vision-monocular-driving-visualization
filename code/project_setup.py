"""
project_setup.py
================

Shared path, scene-output, and compatibility helpers for the project.

Purpose
-------
The perception modules, scene assembler, and Blender renderer all need a
consistent answer to the same questions:

* which ``sceneX`` is being processed?
* where should canonical outputs be written?
* which legacy paths should still be mirrored for backward compatibility?

This module centralises that logic so every stage follows the same output
layout and the same compatibility rules.

Canonical output layout
-----------------------
The production layout is scene-rooted under ``output/<scene>/``::

    output/scene1/detections/
    output/scene1/traffic_lights/
    output/scene1/lanes/
    output/scene1/depth/
    output/scene1/flow/
    output/scene1/scene_data/
    output/scene1/renders/
    output/scene1/videos/
    output/scene1/frames/

"""

from __future__ import annotations

import glob
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


_THIS = Path(__file__).parent.resolve()
_IS_WINDOWS = sys.platform.startswith("win")
_IS_MAC = sys.platform == "darwin"
_SCENE_RE = re.compile(r"(scene\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Central path configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "PROJECT_ROOT": _THIS,
    "DATA_ROOT": _THIS / "P3Data",
    "SEQUENCES_DIR": _THIS / "P3Data" / "Sequences",
    "CALIB_DIR": _THIS / "P3Data" / "Calib",
    "ASSETS_DIR": _THIS / "P3Data" / "Assets",
    "OUTPUT_ROOT": _THIS / "output",
    "LEGACY_DETECTIONS_ROOT": _THIS / "output" / "detections",
    "LEGACY_DEPTH_ROOT": _THIS / "output" / "depth",
    "LEGACY_FRAMES_ROOT": _THIS / "output" / "frames",
    "LEGACY_RENDERS_ROOT": _THIS / "output" / "renders",
    "LEGACY_SCENE_DATA_ROOT": _THIS / "output" / "scene_data",
    "LEGACY_VIDEOS_ROOT": _THIS / "output" / "videos",
    "LEGACY_REPO_RENDERS_ROOT": _THIS / "renders",
    "CAMERA_VIEWS": ["front", "back", "left", "right"],
    "TARGET_FPS": 15,
    "BLENDER_EXECUTABLE": Path(
        r"C:/Program Files/Blender Foundation/Blender 3.6/blender.exe"
        if _IS_WINDOWS
        else (
            "/Applications/Blender.app/Contents/MacOS/Blender"
            if _IS_MAC
            else "/usr/bin/blender"
        )
    ),
}

SEQUENCE_NAMES = [f"scene{i}" for i in range(1, 14)]


@dataclass(frozen=True)
class SceneOutputLayout:
    """Resolved canonical and legacy output roots for one scene."""

    scene: str
    root: Path
    detections: Path
    traffic_lights: Path
    lanes: Path
    depth: Path
    flow: Path
    scene_data: Path
    renders: Path
    videos: Path
    frames: Path
    legacy_detections: Path
    legacy_depth: Path
    legacy_frames: Path
    legacy_output_renders: Path
    legacy_scene_data: Path
    legacy_videos: Path
    legacy_repo_renders: Path


def normalise_scene_name(value: Optional[str]) -> Optional[str]:
    """Return a normalised ``sceneX`` string, or ``None`` if unavailable."""
    if not value:
        return None
    match = _SCENE_RE.search(str(value))
    if not match:
        return None
    return match.group(1).lower()


def infer_scene_name(scene: Optional[str] = None, *hints: object) -> str:
    """
    Infer a ``sceneX`` identifier from an explicit scene or any path-like hint.

    The resolver prefers the explicit ``scene`` argument, then scans the
    supplied hints from left to right.  If nothing matches, ``scene1`` is used
    as a conservative default so scripts remain runnable without extra flags.
    """
    explicit = normalise_scene_name(scene)
    if explicit:
        return explicit

    for hint in hints:
        if hint is None:
            continue
        match = normalise_scene_name(str(hint))
        if match:
            return match
    return "scene1"


def scene_output_layout(scene: Optional[str] = None, *hints: object, create: bool = False) -> SceneOutputLayout:
    """Resolve the canonical and legacy output directories for one scene."""
    resolved_scene = infer_scene_name(scene, *hints)
    root = CONFIG["OUTPUT_ROOT"] / resolved_scene
    layout = SceneOutputLayout(
        scene=resolved_scene,
        root=root,
        detections=root / "detections",
        traffic_lights=root / "traffic_lights",
        lanes=root / "lanes",
        depth=root / "depth",
        flow=root / "flow",
        scene_data=root / "scene_data",
        renders=root / "renders",
        videos=root / "videos",
        frames=root / "frames",
        legacy_detections=CONFIG["LEGACY_DETECTIONS_ROOT"] / resolved_scene,
        legacy_depth=CONFIG["LEGACY_DEPTH_ROOT"] / resolved_scene,
        legacy_frames=CONFIG["LEGACY_FRAMES_ROOT"] / resolved_scene,
        legacy_output_renders=CONFIG["LEGACY_RENDERS_ROOT"] / resolved_scene,
        legacy_scene_data=CONFIG["LEGACY_SCENE_DATA_ROOT"] / resolved_scene,
        legacy_videos=CONFIG["LEGACY_VIDEOS_ROOT"] / resolved_scene,
        legacy_repo_renders=CONFIG["LEGACY_REPO_RENDERS_ROOT"] / resolved_scene,
    )
    if create:
        ensure_scene_layout(layout)
    return layout


def candidate_artifact_paths(
    candidate: str | Path,
    *,
    extra_roots: Optional[Sequence[str | Path]] = None,
) -> list[Path]:
    """
    Return local filesystem locations worth probing for *candidate*.

    This keeps stage scripts offline-friendly by checking the project root and
    ``weights/`` before falling back to remote model names.
    """
    raw = Path(candidate).expanduser()
    roots = [CONFIG["PROJECT_ROOT"]]
    if extra_roots:
        roots.extend(Path(root).expanduser() for root in extra_roots)

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        for root in roots:
            candidates.append(root / raw)
            if not raw.parts or raw.parts[0] != "weights":
                candidates.append(root / "weights" / raw.name)

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_existing_artifact(
    candidates: Sequence[str | Path],
    *,
    extra_roots: Optional[Sequence[str | Path]] = None,
) -> Optional[str]:
    """Return the first existing local path for *candidates*, else ``None``."""
    for candidate in candidates:
        for path in candidate_artifact_paths(candidate, extra_roots=extra_roots):
            if path.exists():
                return str(path.resolve())
    return None


def ensure_dir(path: Path) -> Path:
    """Create *path* if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for *path* if needed and return *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_scene_layout(layout: SceneOutputLayout) -> SceneOutputLayout:
    """Create every canonical and legacy directory for a scene."""
    canonical_dirs = [
        layout.root,
        layout.detections,
        layout.traffic_lights,
        layout.lanes,
        layout.depth,
        layout.flow,
        layout.scene_data,
        layout.renders,
        layout.videos,
        layout.frames,
    ]
    legacy_dirs = [
        layout.legacy_detections,
        layout.legacy_depth,
        layout.legacy_frames,
        layout.legacy_output_renders,
        layout.legacy_scene_data,
        layout.legacy_videos,
        layout.legacy_repo_renders,
    ]
    for path in canonical_dirs + legacy_dirs:
        ensure_dir(path)
    for view in CONFIG["CAMERA_VIEWS"]:
        ensure_dir(layout.frames / view)
        ensure_dir(layout.legacy_frames / view)
    return layout


def canonical_stage_path(
    scene: Optional[str],
    stage: str,
    filename: str | Path,
    *hints: object,
    create_parent: bool = True,
) -> Path:
    """Return the canonical ``output/<scene>/<stage>/<filename>`` path."""
    layout = scene_output_layout(scene, *hints, create=create_parent)
    stage_dir = getattr(layout, stage)
    path = stage_dir / Path(filename)
    return ensure_parent(path) if create_parent else path


def legacy_stage_paths(
    scene: Optional[str],
    stage: str,
    filename: str | Path,
    *hints: object,
) -> list[Path]:
    """
    Return legacy mirror paths for a canonical stage output.

    The mapping is intentionally small and deterministic so each script can
    write once to the canonical path and mirror only the compatibility files
    that older tools expect.
    """
    layout = scene_output_layout(scene, *hints, create=False)
    rel = Path(filename)
    if stage == "detections":
        return [
            layout.legacy_detections / rel,
            layout.legacy_repo_renders / rel,
        ]
    if stage == "traffic_lights":
        return [layout.legacy_repo_renders / rel]
    if stage == "lanes":
        return [layout.legacy_repo_renders / rel]
    if stage == "depth":
        return [
            layout.legacy_depth / rel,
            layout.legacy_repo_renders / rel,
        ]
    if stage == "flow":
        return [layout.legacy_repo_renders / rel]
    if stage == "scene_data":
        return [layout.legacy_scene_data / rel]
    if stage == "renders":
        return [layout.legacy_output_renders / rel]
    if stage == "videos":
        return [layout.legacy_videos / rel]
    if stage == "frames":
        return [layout.legacy_frames / rel]
    return []


def _safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except OSError:
        pass


def mirror_output_file(
    source: str | Path,
    destinations: Sequence[str | Path],
    *,
    prefer_symlink: bool = True,
) -> list[Path]:
    """
    Mirror one output file to one or more compatibility paths.

    Symlinks are preferred for large outputs.  When a symlink cannot be created
    on the current platform or filesystem, the file is copied instead.
    """
    src = Path(source).resolve()
    if not src.exists():
        return []

    mirrored: list[Path] = []
    for raw_dst in destinations:
        dst = Path(raw_dst)
        if not dst.is_absolute():
            dst = (_THIS / dst).resolve()
        if dst == src:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        _safe_unlink(dst)
        try:
            if prefer_symlink:
                os.symlink(src, dst)
            else:
                raise OSError
        except OSError:
            shutil.copy2(src, dst)
        mirrored.append(dst)
    return mirrored


def mirror_stage_output(
    source: str | Path,
    scene: Optional[str],
    stage: str,
    filename: str | Path,
    *hints: object,
    prefer_symlink: bool = True,
) -> list[Path]:
    """Mirror one canonical stage output to its legacy compatibility paths."""
    targets = legacy_stage_paths(scene, stage, filename, *hints)
    return mirror_output_file(source, targets, prefer_symlink=prefer_symlink)


def setup_directories() -> list[Path]:
    """
    Create canonical and legacy directories for every known scene.

    Returns
    -------
    list[Path]
        Sorted list of directories that had to be created.
    """
    before = {path.resolve() for path in CONFIG["OUTPUT_ROOT"].glob("**/*") if path.is_dir()} if CONFIG["OUTPUT_ROOT"].exists() else set()
    ensure_dir(CONFIG["OUTPUT_ROOT"])
    for legacy_root in (
        CONFIG["LEGACY_DETECTIONS_ROOT"],
        CONFIG["LEGACY_DEPTH_ROOT"],
        CONFIG["LEGACY_FRAMES_ROOT"],
        CONFIG["LEGACY_RENDERS_ROOT"],
        CONFIG["LEGACY_SCENE_DATA_ROOT"],
        CONFIG["LEGACY_VIDEOS_ROOT"],
        CONFIG["LEGACY_REPO_RENDERS_ROOT"],
    ):
        ensure_dir(legacy_root)
    for scene in SEQUENCE_NAMES:
        ensure_scene_layout(scene_output_layout(scene, create=True))
    after = {path.resolve() for path in CONFIG["OUTPUT_ROOT"].glob("**/*") if path.is_dir()}
    created = sorted(after - before)
    return created


def get_sequence_video_path(seq_name: str, view: str) -> Path | None:
    """
    Return the undistorted video for ``seq_name`` and ``view``, if available.
    """
    undist_dir = CONFIG["SEQUENCES_DIR"] / seq_name / "Undist"
    pattern = str(undist_dir / f"*{view}*.mp4")
    matches = glob.glob(pattern)
    if not matches:
        print(
            f"[WARNING] No undistorted video found for {seq_name}/{view} in {undist_dir}"
        )
        return None
    if len(matches) > 1:
        print(
            f"[WARNING] Multiple videos matched for {seq_name}/{view}; using first: {matches[0]}"
        )
    return Path(matches[0]).resolve()


def get_sequence_frame_dir(seq_name: str, view: str) -> Path:
    """Return the canonical extracted-frame directory for one scene/view."""
    layout = scene_output_layout(seq_name, create=True)
    return layout.frames / view


def get_detection_path(seq_name: str, frame_idx: int) -> Path:
    """Return the legacy per-frame detection JSON path expected by old tools."""
    layout = scene_output_layout(seq_name, create=True)
    return layout.legacy_detections / f"frame_{frame_idx:05d}.json"


if __name__ == "__main__":
    print("=" * 72)
    print("  RBE549 P3 — Einstein Vision — Project Setup")
    print("=" * 72)
    print("\n[CONFIG] Key paths:")
    for key, value in CONFIG.items():
        print(f"  {key:<24} = {value}")

    print("\n[SETUP] Creating canonical + legacy output directories …")
    newly_created = setup_directories()
    if newly_created:
        print(f"\n  Created {len(newly_created)} directories:")
        for path in newly_created[:60]:
            print(f"    {path}")
        if len(newly_created) > 60:
            print(f"    … {len(newly_created) - 60} more")
    else:
        print("\n  All directories already existed.")

    sample_scene = "scene1"
    sample_layout = scene_output_layout(sample_scene, create=True)
    print("\n[VERIFY] Sample canonical layout:")
    print(f"  scene root     = {sample_layout.root}")
    print(f"  detections     = {sample_layout.detections}")
    print(f"  scene_data     = {sample_layout.scene_data}")
    print(f"  renders        = {sample_layout.renders}")
    print(f"  legacy renders = {sample_layout.legacy_repo_renders}")

    sample_view = "front"
    print(f"\n[VERIFY] Video lookup for {sample_scene}/{sample_view}:")
    print(f"  {get_sequence_video_path(sample_scene, sample_view)}")
