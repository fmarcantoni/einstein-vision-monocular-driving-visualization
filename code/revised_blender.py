"""
blender.py
==========

Blender scene construction and rendering backend for the autonomous-driving
simulation pipeline.

Purpose
-------
This script consumes the compact scene description emitted by
``scene_assembler.py`` and turns it into a Blender scene that can be rendered
frame-by-frame or exported as a full animation.  The goal is not only to place
detected objects in roughly correct 3-D positions, but to do so in a way that
is stable over long sequences and resilient to noisy upstream perception data.

Pipeline responsibilities
-------------------------
1. Recreate the front camera using calibrated intrinsics and a consistent
   vehicle-centric coordinate frame.
2. Build persistent scene collections for the camera, ground reconstruction,
   traffic objects, lanes, road surface, and reusable imported asset templates.
3. Convert per-frame depth maps into a conservative near-field shell while
   masking out road regions, tracked objects, and unstable ego-foreground
   geometry that would otherwise create artifacts near the bottom of the image.
4. Import assets from ``P3Data/Assets`` once, sanitize their geometry, and
   instance them across the animation timeline with deterministic transforms.
5. Render a clean synthetic output sequence and optionally encode it to MP4,
   using ``output/<scene>/renders`` and ``output/<scene>/videos`` as the
   canonical default destinations.

Design notes
------------
Two classes of long-horizon artifacts are handled explicitly here:

* Depth-shell artifacts:
  close-range monocular depth is unreliable around the ego vehicle, so the
  shell is cropped and filtered aggressively near the image bottom.
* Asset accumulation artifacts:
  some source ``.blend`` files contain multiple disconnected meshes or helper
  geometry.  Asset templates are therefore sanitized so only the coherent main
  cluster remains before instancing begins.
* Elevated semantic artifacts:
  signs and traffic lights are expected to arrive from
  ``scene_assembler.py`` only after class-aware geometric validation, and the
  renderer keeps their materials matte to avoid bright synthetic slices.

Entry point
-----------
Run from Blender in background mode, for example::

    blender --background --python blender.py -- --scene-json <scene.json> --render
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import bpy
import mathutils

try:
    import numpy as np
except ImportError:  # Blender builds typically include numpy, but keep a fallback.
    np = None


_SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_SCRIPT_DIR))

from project_setup import infer_scene_name, mirror_stage_output, scene_output_layout


# ============================================================================
# Constants
# ============================================================================

COL_ROOT = "FSD_Scene"
COL_BG = "Background"
COL_DEPTH = "Depth"
COL_OBJECTS = "Objects"
COL_TRAFFIC = "Traffic"
COL_LANES = "Lanes"
COL_ROAD = "Road"
COL_MARKINGS = "Markings"
COL_CAMERA = "Camera"
COL_EGO = "EgoVehicle"
COL_TEMPLATES = "Templates"

MAX_USEFUL_DEPTH_M = 80.0
MAX_GROUND_LATERAL_M = 24.0
MAX_GROUND_HEIGHT_M = 8.0
DEFAULT_DEPTH_TOP_CROP_FRAC = 0.18

CLASS_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "car": (0.10, 0.34, 0.78, 1.0),
    "truck": (0.25, 0.25, 0.25, 1.0),
    "motorcycle": (0.75, 0.12, 0.55, 1.0),
    "bicycle": (0.10, 0.62, 0.42, 1.0),
    "pedestrian": (0.88, 0.40, 0.10, 1.0),
    "traffic_light": (0.20, 0.20, 0.20, 1.0),
    "traffic_sign": (0.88, 0.88, 0.88, 1.0),
    "stop_sign": (0.85, 0.10, 0.10, 1.0),
    "speed_limit": (0.85, 0.85, 0.85, 1.0),
    "dustbin": (0.16, 0.45, 0.22, 1.0),
    "traffic_pole": (0.55, 0.55, 0.55, 1.0),
    "traffic_cone": (0.82, 0.30, 0.05, 1.0),
    "traffic_cylinder": (0.96, 0.48, 0.10, 1.0),
    "fire_hydrant": (0.82, 0.08, 0.08, 1.0),
    "speed_bump": (0.94, 0.94, 0.88, 1.0),
}

CLASS_PRIMITIVE_DIMS: Dict[str, Tuple[float, float, float]] = {
    "car": (4.50, 1.90, 1.50),
    "truck": (8.00, 2.50, 3.40),
    "motorcycle": (2.20, 0.80, 1.20),
    "bicycle": (1.80, 0.55, 1.10),
    "pedestrian": (0.55, 0.55, 1.75),
    "traffic_light": (0.45, 0.45, 4.20),
    "traffic_sign": (0.15, 0.65, 2.10),
    "stop_sign": (0.15, 0.75, 2.20),
    "speed_limit": (0.15, 0.65, 2.20),
    "dustbin": (0.75, 0.75, 1.10),
    "traffic_pole": (0.20, 0.20, 3.50),
    "traffic_cone": (0.38, 0.38, 0.72),
    "traffic_cylinder": (0.40, 0.40, 1.00),
    "fire_hydrant": (0.55, 0.55, 0.90),
    "speed_bump": (0.95, 3.20, 0.10),
}

TL_COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (1.0, 0.02, 0.02),
    "yellow": (1.0, 0.82, 0.06),
    "green": (0.05, 0.95, 0.08),
    "unknown": (0.55, 0.55, 0.55),
}

TL_EMISSION_STRENGTH: Dict[str, float] = {
    "red": 12.0,
    "yellow": 10.5,
    "green": 11.5,
    "unknown": 0.0,
}

LANE_COLORS: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {
    ("white", "solid"): (0.96, 0.96, 0.96, 1.0),
    ("white", "dashed"): (0.95, 0.95, 0.95, 1.0),
    ("yellow", "solid"): (0.98, 0.82, 0.08, 1.0),
    ("yellow", "dashed"): (0.98, 0.82, 0.08, 1.0),
    ("unknown", "solid"): (0.96, 0.96, 0.96, 1.0),
    ("unknown", "dashed"): (0.98, 0.82, 0.08, 1.0),
}

ROAD_MARKING_COLORS: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {
    ("white", "road_marking"): (0.95, 0.95, 0.93, 1.0),
    ("yellow", "road_marking"): (0.96, 0.82, 0.10, 1.0),
    ("unknown", "road_marking"): (0.94, 0.94, 0.92, 1.0),
    ("white", "arrow"): (0.97, 0.97, 0.95, 1.0),
    ("yellow", "arrow"): (0.98, 0.84, 0.12, 1.0),
    ("unknown", "arrow"): (0.96, 0.96, 0.94, 1.0),
}

PED_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

PED_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)

PED_DEFAULT_POSE: Dict[int, Tuple[float, float]] = {
    0: (0.00, 0.96),
    1: (-0.05, 0.98),
    2: (0.05, 0.98),
    3: (-0.10, 0.95),
    4: (0.10, 0.95),
    5: (-0.34, 0.82),
    6: (0.34, 0.82),
    7: (-0.46, 0.66),
    8: (0.46, 0.66),
    9: (-0.52, 0.48),
    10: (0.52, 0.48),
    11: (-0.18, 0.54),
    12: (0.18, 0.54),
    13: (-0.15, 0.28),
    14: (0.15, 0.28),
    15: (-0.12, 0.02),
    16: (0.12, 0.02),
}

POSE_JOINT_RADIUS_M = 0.085
POSE_BONE_RADIUS_M = 0.052
POSE_CONFIDENCE_THRESH = 0.20


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    try:
        sep = sys.argv.index("--")
        script_argv = sys.argv[sep + 1 :]
    except ValueError:
        script_argv = []

    parser = argparse.ArgumentParser(
        description="Blender scene builder for the autonomous driving pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene-json",
        default="output/scene_data/scene1/scene_assembled.json",
        help="Path to scene_assembled.json from scene_assembler.py",
    )
    parser.add_argument("--assets-dir", default="P3Data/Assets", help="Root of P3Data/Assets/")
    parser.add_argument("--out-blend", default=None, help="Optional output .blend path")
    parser.add_argument("--render-dir", default=None, help="Optional directory for rendered PNG frames")
    parser.add_argument("--renderer", default="EEVEE", choices=["EEVEE", "CYCLES"])
    parser.add_argument("--samples", type=int, default=48, help="Render samples")
    parser.add_argument(
        "--camera-mode",
        choices=["chase", "dashcam"],
        default="chase",
        help="Render camera viewpoint: 'chase' shows the ego car in 3rd-person (EinsteinVision style); "
        "'dashcam' uses the calibrated front-camera intrinsics for a 1st-person POV.",
    )
    parser.add_argument("--chase-distance", type=float, default=8.5, help="Chase camera distance behind ego (m)")
    parser.add_argument("--chase-height", type=float, default=4.8, help="Chase camera height above ground (m)")
    parser.add_argument("--chase-pitch-deg", type=float, default=22.0, help="Chase camera downward pitch (deg)")
    parser.add_argument("--start-frame", type=int, default=None, help="Source frame index to start from")
    parser.add_argument("--end-frame", type=int, default=None, help="Source frame index to end at")
    parser.add_argument("--fps", type=float, default=None, help="Override the timeline FPS")
    parser.add_argument("--render", action="store_true", help="Render the selected frame range")
    parser.add_argument("--no-assets", action="store_true", help="Skip `.blend` asset import and use primitives only")
    parser.add_argument("--no-lanes", action="store_true", help="Skip lane geometry")
    parser.add_argument("--no-depth-shell", action="store_true", help="Skip the depth-driven near-field shell")
    parser.add_argument(
        "--use-background-plate",
        action="store_true",
        help="Render the source video as a camera-aligned backdrop (debug/reference mode)",
    )
    parser.add_argument("--no-background-plate", action="store_false", dest="use_background_plate", help=argparse.SUPPRESS)
    parser.add_argument(
        "--use-source-textures",
        action="store_true",
        help="Texture the depth shell from the source video instead of using synthetic materials",
    )
    parser.add_argument("--no-source-textures", action="store_false", dest="use_source_textures", help=argparse.SUPPRESS)
    parser.add_argument("--max-objects-per-frame", type=int, default=24, help="Limit imported objects per frame")
    parser.add_argument("--depth-stride", type=int, default=None, help="Override depth shell sampling stride in pixels")
    parser.add_argument("--depth-top-cut", type=float, default=None, help="Override depth shell top crop fraction")
    parser.add_argument("--depth-bottom-cut", type=float, default=None, help="Override depth shell bottom crop fraction")
    parser.add_argument("--depth-min-distance", type=float, default=None, help="Override minimum depth used for the depth shell")
    parser.add_argument(
        "--depth-foreground-distance",
        type=float,
        default=None,
        help="Override the scene-aware near-field depth floor used to suppress ego-foreground artifacts",
    )
    parser.add_argument(
        "--depth-foreground-row",
        type=float,
        default=None,
        help="Override the normalized image row where aggressive foreground suppression begins",
    )
    parser.add_argument(
        "--depth-foreground-boost",
        type=float,
        default=None,
        help="Override the extra depth floor added toward the image bottom",
    )
    parser.add_argument("--depth-bbox-margin", type=int, default=None, help="Override bbox masking margin for the depth shell")
    parser.add_argument(
        "--export-reference-frames",
        action="store_true",
        help="Extract source video frames into a hidden auxiliary directory used to build collage outputs",
    )
    parser.add_argument("--no-reference-frames", action="store_false", dest="export_reference_frames", help=argparse.SUPPRESS)
    parser.add_argument(
        "--compose-collage",
        action="store_true",
        help="Replace the main rendered PNG sequence with side-by-side source/render collage frames",
    )
    parser.add_argument("--no-collage", action="store_false", dest="compose_collage", help=argparse.SUPPRESS)
    parser.add_argument("--no-mp4", action="store_false", dest="encode_mp4", help="Skip encoding the rendered PNG sequence into an mp4")
    parser.set_defaults(
        use_background_plate=False,
        use_source_textures=True,
        export_reference_frames=True,
        compose_collage=True,
        encode_mp4=True,
    )
    return parser.parse_args(script_argv)


# ============================================================================
# Collection / scene helpers
# ============================================================================

def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if parent is None:
        parent = bpy.context.scene.collection
    if collection.name not in [child.name for child in parent.children]:
        parent.children.link(collection)
    return collection


def link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for coll in list(obj.users_collection):
        if coll != collection:
            coll.objects.unlink(obj)
    if obj.name not in [member.name for member in collection.objects]:
        collection.objects.link(obj)


def reset_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for coll in list(bpy.data.collections):
        if coll.name != "Collection":
            bpy.data.collections.remove(coll)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.movieclips,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def set_visibility_keyframe(obj: bpy.types.Object, frame: int, visible: bool) -> None:
    frame = max(1, int(frame))
    obj.hide_render = not visible
    obj.hide_viewport = not visible
    obj.keyframe_insert(data_path="hide_render", frame=frame)
    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
    set_keyframe_interpolation(obj, "hide_render", frame, "CONSTANT")
    set_keyframe_interpolation(obj, "hide_viewport", frame, "CONSTANT")


def set_keyframe_interpolation(
    obj: bpy.types.Object,
    data_path: str,
    frame: int,
    interpolation: str,
) -> None:
    animation_data = getattr(obj, "animation_data", None)
    action = getattr(animation_data, "action", None)
    if action is None:
        return

    fcurves = getattr(action, "fcurves", None)
    if fcurves is None:
        return

    target_frame = float(frame)
    for fcurve in fcurves:
        if fcurve.data_path != data_path:
            continue
        for keyframe in fcurve.keyframe_points:
            if abs(float(keyframe.co.x) - target_frame) < 0.5:
                keyframe.interpolation = interpolation


def contiguous_ranges(frames: Iterable[int]) -> List[Tuple[int, int]]:
    values = sorted(set(int(f) for f in frames))
    if not values:
        return []
    ranges: List[Tuple[int, int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev))
        start = prev = value
    ranges.append((start, prev))
    return ranges


# ============================================================================
# Materials / textures
# ============================================================================

def get_or_create_material(name: str) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    return material


def make_solid_material(
    name: str,
    color: Tuple[float, float, float, float],
    roughness: float = 0.6,
    metallic: float = 0.0,
    specular: float = 0.03,
) -> bpy.types.Material:
    material = get_or_create_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_emission_material(
    name: str,
    color: Tuple[float, float, float, float],
    *,
    strength: float = 1.0,
    roughness: float = 0.55,
) -> bpy.types.Material:
    material = get_or_create_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    add_shader = nodes.new("ShaderNodeAddShader")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    emission = nodes.new("ShaderNodeEmission")

    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.01
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = float(strength)

    links.new(bsdf.outputs["BSDF"], add_shader.inputs[0])
    links.new(emission.outputs["Emission"], add_shader.inputs[1])
    links.new(add_shader.outputs["Shader"], output.inputs["Surface"])
    return material


def get_movie_image(video_path: str) -> Optional[bpy.types.Image]:
    if not video_path or not Path(video_path).exists():
        return None
    try:
        image = bpy.data.images.load(video_path, check_existing=True)
    except RuntimeError:
        return None
    image.source = "MOVIE"
    return image


def get_static_image(image_path: str) -> Optional[bpy.types.Image]:
    if not image_path or not Path(image_path).exists():
        return None
    try:
        image = bpy.data.images.load(image_path, check_existing=True)
    except RuntimeError:
        return None
    image.source = "FILE"
    return image


def rasterize_svg_asset(svg_path: Path, size_px: int = 1400) -> Optional[Path]:
    if not svg_path.exists():
        return None

    cache_root = Path(tempfile.gettempdir()) / "group10_p3_svg_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    stat = svg_path.stat()
    key = hashlib.sha1(
        f"{svg_path.resolve()}::{stat.st_mtime_ns}::{size_px}".encode("utf-8")
    ).hexdigest()[:16]
    cached_png = cache_root / f"{svg_path.stem}_{key}.png"
    if cached_png.exists():
        return cached_png

    preview_dir = cache_root / f"preview_{key}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    commands: List[List[str]] = []
    qlmanage = shutil.which("qlmanage")
    if sys.platform == "darwin" and qlmanage:
        commands.append([qlmanage, "-t", "-s", str(int(size_px)), "-o", str(preview_dir), str(svg_path)])

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception:
            continue

        candidates = sorted(preview_dir.rglob("*.png"))
        if not candidates:
            continue
        newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        try:
            shutil.copy2(str(newest), str(cached_png))
            return cached_png
        except OSError:
            continue

    return None


def get_svg_backed_image(svg_path: Path, size_px: int = 1400) -> Optional[bpy.types.Image]:
    png_path = rasterize_svg_asset(svg_path, size_px=size_px)
    if png_path is not None:
        image = get_static_image(str(png_path))
        if image is not None:
            return image
    return get_static_image(str(svg_path))


def make_movie_material(
    name: str,
    image: bpy.types.Image,
    emission: bool,
) -> bpy.types.Material:
    material = get_or_create_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"
    tex.interpolation = "Linear"
    tex.image_user.frame_start = 1
    tex.image_user.frame_offset = 0
    tex.image_user.frame_duration = 100000
    tex.image_user.use_auto_refresh = True

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(uv.outputs["UV"], tex.inputs["Vector"])

    if emission:
        shader = nodes.new("ShaderNodeEmission")
        shader.inputs["Strength"].default_value = 1.0
        links.new(tex.outputs["Color"], shader.inputs["Color"])
        links.new(shader.outputs["Emission"], output.inputs["Surface"])
    else:
        shader = nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = 0.96
        shader.inputs["Specular IOR Level"].default_value = 0.01
        links.new(tex.outputs["Color"], shader.inputs["Base Color"])
        links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    material.use_backface_culling = False
    return material


def make_image_sign_material(
    name: str,
    image: bpy.types.Image,
    *,
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    use_alpha: bool = True,
    backface_culling: bool = False,
) -> bpy.types.Material:
    material = get_or_create_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"
    tex.interpolation = "Linear"

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"

    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (180, 80)
    shader.inputs["Roughness"].default_value = 0.84
    if "Specular IOR Level" in shader.inputs:
        shader.inputs["Specular IOR Level"].default_value = 0.015

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (420, 40)

    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], shader.inputs["Base Color"])
    if use_alpha and "Alpha" in tex.outputs:
        if "Alpha" in shader.inputs:
            links.new(tex.outputs["Alpha"], shader.inputs["Alpha"])
    else:
        shader.inputs["Base Color"].default_value = base_color
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    material.use_backface_culling = bool(backface_culling)
    if use_alpha and hasattr(material, "blend_method"):
        material.blend_method = "CLIP"
    if use_alpha and hasattr(material, "shadow_method"):
        material.shadow_method = "CLIP"
    return material


def make_image_sprite_material(
    name: str,
    image: bpy.types.Image,
    tint_rgba: Optional[Tuple[float, float, float, float]] = None,
) -> bpy.types.Material:
    material = get_or_create_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    tint = tint_rgba or (1.0, 1.0, 1.0, 1.0)

    try:
        if hasattr(image, "alpha_mode"):
            image.alpha_mode = "STRAIGHT"
    except Exception:
        pass

    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.extension = "CLIP"
    tex.interpolation = "Linear"

    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"

    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.location = (40, 80)
    multiply.blend_type = "MULTIPLY"
    multiply.inputs["Fac"].default_value = 1.0

    tint_rgb = nodes.new("ShaderNodeRGB")
    tint_rgb.location = (-160, 0)
    tint_rgb.outputs["Color"].default_value = tint

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (260, 110)
    emission.inputs["Strength"].default_value = 0.92

    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.location = (260, -40)

    mix_shader = nodes.new("ShaderNodeMixShader")
    mix_shader.location = (440, 40)

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (660, 40)

    links.new(uv.outputs["UV"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], multiply.inputs["Color1"])
    links.new(tint_rgb.outputs["Color"], multiply.inputs["Color2"])
    links.new(multiply.outputs["Color"], emission.inputs["Color"])
    if "Alpha" in tex.outputs:
        links.new(tex.outputs["Alpha"], mix_shader.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

    material.use_backface_culling = False
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "NONE"
    return material


def make_road_material() -> bpy.types.Material:
    material = get_or_create_material("M_Road")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.14, 0.14, 0.14)

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 16.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.72

    coarse_noise = nodes.new("ShaderNodeTexNoise")
    coarse_noise.inputs["Scale"].default_value = 36.0
    coarse_noise.inputs["Detail"].default_value = 8.0
    coarse_noise.inputs["Roughness"].default_value = 0.58

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.inputs["B"].default_value = (0.21, 0.21, 0.20, 1.0)
    mix.inputs["A"].default_value = (0.14, 0.14, 0.14, 1.0)

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.03
    bump.inputs["Distance"].default_value = 0.06

    roughness_scale = nodes.new("ShaderNodeMath")
    roughness_scale.operation = "MULTIPLY"
    roughness_scale.inputs[1].default_value = 0.05

    roughness_bias = nodes.new("ShaderNodeMath")
    roughness_bias.operation = "ADD"
    roughness_bias.inputs[1].default_value = 0.90

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.17, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.965
    bsdf.inputs["Specular IOR Level"].default_value = 0.01

    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], coarse_noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], mix.inputs["Factor"])
    links.new(coarse_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], roughness_scale.inputs[0])
    links.new(roughness_scale.outputs["Value"], roughness_bias.inputs[0])
    links.new(roughness_bias.outputs["Value"], bsdf.inputs["Roughness"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def make_road_marking_material(
    color_name: str,
    marking_type: str,
) -> bpy.types.Material:
    if marking_type == "crosswalk":
        marking_type = "road_marking"
    rgba = ROAD_MARKING_COLORS.get(
        (str(color_name), str(marking_type)),
        ROAD_MARKING_COLORS.get((str(color_name), "road_marking"), ROAD_MARKING_COLORS[("unknown", "road_marking")]),
    )
    roughness = 0.74 if marking_type == "arrow" else 0.79
    specular = 0.025 if color_name == "yellow" else 0.02
    return make_solid_material(
        f"M_RoadMarking_{color_name}_{marking_type}",
        rgba,
        roughness=roughness,
        metallic=0.0,
        specular=specular,
    )


def make_depth_shell_material() -> bpy.types.Material:
    material = get_or_create_material("M_DepthShellSynthetic")
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.16, 0.16, 0.16)

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.72

    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.inputs["Scale"].default_value = 11.0

    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.blend_type = "MIX"
    mix.inputs["Factor"].default_value = 0.25

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.035
    bump.inputs["Distance"].default_value = 0.08

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.26
    ramp.color_ramp.elements[0].color = (0.27, 0.29, 0.30, 1.0)
    ramp.color_ramp.elements[1].position = 0.84
    ramp.color_ramp.elements[1].color = (0.39, 0.41, 0.43, 1.0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.995
    bsdf.inputs["Specular IOR Level"].default_value = 0.005

    output = nodes.new("ShaderNodeOutputMaterial")

    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
    links.new(noise.outputs["Fac"], mix.inputs["A"])
    links.new(voronoi.outputs["Distance"], mix.inputs["B"])
    links.new(mix.outputs["Result"], ramp.inputs["Fac"])
    links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def canonicalise_tl_color(tl_color: str) -> str:
    value = str(tl_color or "unknown").strip().lower()
    if value in {"amber", "orange"}:
        return "yellow"
    if value not in TL_COLORS:
        return "unknown"
    return value


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _confidence_bucket(value: Optional[float]) -> int:
    return int(round(clamp01(float(value or 0.0)) * 10.0))


def _coerce_triplet(values: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float]]:
    if values is None:
        return None
    vals = list(values)[:3]
    if len(vals) < 3:
        return None
    try:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    except (TypeError, ValueError):
        return None


def resolve_lane_paint_rgba(
    lane_color: str,
    lane_type: str,
    avg_hsv: Optional[Sequence[float]] = None,
    avg_ycrcb: Optional[Sequence[float]] = None,
    color_confidence: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    lane_key = str(lane_color or "unknown").strip().lower()
    type_key = str(lane_type or "solid").strip().lower()
    base_rgba = LANE_COLORS.get(
        (lane_key, type_key),
        LANE_COLORS.get(("unknown", type_key), LANE_COLORS[("unknown", "solid")]),
    )
    base_rgb = base_rgba[:3]
    conf = clamp01(float(color_confidence or 0.0))

    hsv_triplet = _coerce_triplet(avg_hsv)
    ycrcb_triplet = _coerce_triplet(avg_ycrcb)
    sample_rgb = base_rgb

    if hsv_triplet is not None:
        sample_rgb = colorsys.hsv_to_rgb(
            clamp01(hsv_triplet[0] / 179.0),
            clamp01(hsv_triplet[1] / 255.0),
            clamp01(hsv_triplet[2] / 255.0),
        )

    y_norm = clamp01(
        (ycrcb_triplet[0] / 255.0)
        if ycrcb_triplet is not None
        else ((hsv_triplet[2] / 255.0) if hsv_triplet is not None else 0.78)
    )
    cr = float(ycrcb_triplet[1]) if ycrcb_triplet is not None else 128.0
    cb = float(ycrcb_triplet[2]) if ycrcb_triplet is not None else 128.0
    warm_bias = max(-1.0, min(1.0, ((128.0 - cb) / 52.0) + ((cr - 128.0) / 70.0)))
    neutral_chroma = clamp01(1.0 - (abs(cr - 128.0) + abs(cb - 128.0)) / 150.0)
    sat_norm = clamp01((hsv_triplet[1] / 255.0) if hsv_triplet is not None else 0.65)

    if lane_key == "white":
        target = (
            clamp01(0.84 + 0.14 * y_norm + 0.03 * max(0.0, warm_bias)),
            clamp01(0.84 + 0.14 * y_norm + 0.018 * max(0.0, warm_bias)),
            clamp01(0.83 + 0.13 * y_norm - 0.03 * max(0.0, warm_bias)),
        )
        sample_mix = 0.08 + 0.10 * conf
        target = tuple(
            clamp01((1.0 - sample_mix) * target[i] + sample_mix * sample_rgb[i])
            for i in range(3)
        )
        mix = 0.22 + 0.28 * max(conf, neutral_chroma * 0.75)
    elif lane_key == "yellow":
        target = (
            clamp01(0.80 + 0.16 * y_norm + 0.06 * max(0.0, warm_bias)),
            clamp01(0.63 + 0.17 * y_norm + 0.05 * sat_norm),
            clamp01(0.05 + 0.06 * (1.0 - y_norm)),
        )
        sample_mix = 0.22 + 0.18 * conf
        target = tuple(
            clamp01((1.0 - sample_mix) * target[i] + sample_mix * sample_rgb[i])
            for i in range(3)
        )
        mix = 0.34 + 0.30 * conf
    else:
        target = tuple(clamp01(v) for v in sample_rgb)
        mix = 0.16 + 0.20 * conf

    final_rgb = tuple(
        clamp01((1.0 - mix) * base_rgb[i] + mix * target[i])
        for i in range(3)
    )
    return (final_rgb[0], final_rgb[1], final_rgb[2], base_rgba[3])


def resolve_traffic_light_render_color(
    tl_color: Optional[str],
    tl_color_conf: Optional[float],
    detic_color_check: Optional[str],
    detic_color_conf: Optional[float],
    detic_color_agrees: Optional[bool],
) -> Tuple[str, float, float]:
    primary = canonicalise_tl_color(str(tl_color or "unknown"))
    checker = canonicalise_tl_color(str(detic_color_check or "unknown"))
    primary_conf = clamp01(float(tl_color_conf or 0.0))
    checker_conf = clamp01(float(detic_color_conf or 0.0))

    chosen = primary
    chosen_conf = primary_conf

    if primary == "unknown" and checker != "unknown" and checker_conf >= 0.12:
        chosen = checker
        chosen_conf = checker_conf
    elif checker != "unknown" and detic_color_agrees is False and checker_conf >= max(0.35, primary_conf + 0.15):
        chosen = checker
        chosen_conf = checker_conf
    elif checker != "unknown" and detic_color_agrees is True and primary_conf < 0.40:
        chosen = checker
        chosen_conf = max(primary_conf, checker_conf)

    agreement_bonus = 0.0
    if detic_color_agrees is True:
        agreement_bonus = 0.12
    elif detic_color_agrees is False:
        agreement_bonus = -0.05

    return chosen, chosen_conf, agreement_bonus


def make_traffic_signal_material(
    name: str,
    tl_color: str,
    *,
    confidence: float = 1.0,
    agreement_bonus: float = 0.0,
) -> bpy.types.Material:
    """Traffic-light material that visibly shows the detected signal state."""
    material = get_or_create_material(name)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (420, 0)
    add_shader = nodes.new("ShaderNodeAddShader")
    add_shader.location = (220, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-40, 80)
    emission = nodes.new("ShaderNodeEmission")
    emission.location = (-40, -90)

    tl_key = canonicalise_tl_color(tl_color)
    glow = TL_COLORS.get(tl_key, TL_COLORS["unknown"])
    conf = clamp01(confidence)
    emission_strength = TL_EMISSION_STRENGTH.get(tl_key, 0.0) * max(0.0, 0.42 + 0.48 * conf + float(agreement_bonus))
    base_tint = (
        0.015 + (0.10 + 0.04 * conf) * float(glow[0]),
        0.015 + (0.10 + 0.04 * conf) * float(glow[1]),
        0.015 + (0.10 + 0.04 * conf) * float(glow[2]),
        1.0,
    )

    bsdf.inputs["Base Color"].default_value = base_tint
    bsdf.inputs["Roughness"].default_value = 0.96
    bsdf.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.01

    emission.inputs["Color"].default_value = (glow[0], glow[1], glow[2], 1.0)
    emission.inputs["Strength"].default_value = emission_strength

    links.new(bsdf.outputs["BSDF"], add_shader.inputs[0])
    links.new(emission.outputs["Emission"], add_shader.inputs[1])
    links.new(add_shader.outputs["Shader"], output.inputs["Surface"])
    return material


def make_traffic_signal_housing_material() -> bpy.types.Material:
    return make_solid_material(
        "M_TrafficSignalHousing",
        (0.10, 0.10, 0.10, 1.0),
        roughness=0.88,
        metallic=0.02,
        specular=0.015,
    )


def make_traffic_signal_lens_material(
    color_name: str,
    *,
    active: bool,
    confidence: float = 1.0,
    agreement_bonus: float = 0.0,
) -> bpy.types.Material:
    tl_key = canonicalise_tl_color(color_name)
    glow = TL_COLORS.get(tl_key, TL_COLORS["unknown"])
    if active:
        return make_traffic_signal_material(
            f"M_TL_Lens_{tl_key}_active_{_confidence_bucket(confidence)}_{int(round((agreement_bonus + 0.10) * 10.0))}",
            tl_key,
            confidence=confidence,
            agreement_bonus=agreement_bonus,
        )

    dimmed = (
        0.015 + 0.055 * float(glow[0]),
        0.015 + 0.055 * float(glow[1]),
        0.015 + 0.055 * float(glow[2]),
        1.0,
    )
    return make_solid_material(
        f"M_TL_Lens_{tl_key}_idle",
        dimmed,
        roughness=0.55,
        metallic=0.0,
        specular=0.03,
    )


def make_vehicle_motion_material(status: str) -> bpy.types.Material:
    status_key = str(status or "unknown").strip().lower()
    color_map = {
        "moving": (0.05, 0.92, 0.18, 1.0),
        "parked": (0.97, 0.97, 0.97, 1.0),
        "unknown": (0.78, 0.78, 0.78, 1.0),
    }
    return make_solid_material(
        f"M_VehicleMotion_{status_key}",
        color_map.get(status_key, color_map["unknown"]),
        roughness=0.55,
        metallic=0.0,
        specular=0.015,
    )


def make_vehicle_body_material(status: str, alert_state: str = "none") -> bpy.types.Material:
    status_key = str(status or "unknown").strip().lower()
    alert_key = str(alert_state or "none").strip().lower()
    if alert_key == "collision":
        return make_solid_material(
            "M_VehicleBody_collision",
            (0.86, 0.12, 0.12, 1.0),
            roughness=0.36,
            metallic=0.15,
            specular=0.24,
        )
    if alert_key == "warning":
        return make_solid_material(
            "M_VehicleBody_warning",
            (0.86, 0.36, 0.08, 1.0),
            roughness=0.38,
            metallic=0.16,
            specular=0.25,
        )
    color_map = {
        "moving": (0.10, 0.31, 0.84, 1.0),
        "parked": (0.95, 0.96, 0.98, 1.0),
        "unknown": (0.68, 0.70, 0.74, 1.0),
    }
    roughness_map = {
        "moving": 0.34,
        "parked": 0.44,
        "unknown": 0.52,
    }
    return make_solid_material(
        f"M_VehicleBody_{status_key}",
        color_map.get(status_key, color_map["unknown"]),
        roughness=roughness_map.get(status_key, roughness_map["unknown"]),
        metallic=0.18 if status_key in {"moving", "parked"} else 0.08,
        specular=0.26,
    )


def make_vehicle_wheel_material() -> bpy.types.Material:
    return make_solid_material(
        "M_VehicleWheel",
        (0.04, 0.04, 0.04, 1.0),
        roughness=0.86,
        metallic=0.0,
        specular=0.02,
    )


def make_cone_stripe_material() -> bpy.types.Material:
    return make_solid_material(
        "M_TrafficConeStripe",
        (0.97, 0.97, 0.95, 1.0),
        roughness=0.74,
        metallic=0.0,
        specular=0.04,
    )


def make_brake_light_material(active: bool) -> bpy.types.Material:
    if active:
        return make_emission_material(
            "M_BrakeLightActive",
            (1.0, 0.06, 0.04, 1.0),
            strength=12.0,
            roughness=0.18,
        )
    return make_emission_material(
        "M_BrakeLightIdle",
        (0.28, 0.04, 0.04, 1.0),
        strength=0.45,
        roughness=0.42,
    )


def make_indicator_light_material(active: bool) -> bpy.types.Material:
    if active:
        return make_emission_material(
            "M_IndicatorLightActive",
            (1.0, 0.58, 0.06, 1.0),
            strength=10.5,
            roughness=0.18,
        )
    return make_emission_material(
        "M_IndicatorLightIdle",
        (0.34, 0.18, 0.05, 1.0),
        strength=0.35,
        roughness=0.42,
    )


def traffic_signal_shape_marker(signal_shape: Optional[str]) -> str:
    shape = str(signal_shape or "unknown").strip().lower()
    if shape == "left_arrow":
        return "<"
    if shape == "right_arrow":
        return ">"
    if shape == "straight_arrow":
        return "^"
    return ""


def vehicle_motion_direction_marker(direction: Optional[str]) -> str:
    value = str(direction or "unknown").strip().lower()
    mapping = {
        "right": "DIR R",
        "left": "DIR L",
        "up": "DIR U",
        "down": "DIR D",
        "up-right": "DIR UR",
        "up-left": "DIR UL",
        "down-right": "DIR DR",
        "down-left": "DIR DL",
        "stationary": "",
        "unknown": "",
    }
    return mapping.get(value, "")


# ============================================================================
# Camera / projection helpers
# ============================================================================

def _vector3(values: Optional[Sequence[float]], default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> mathutils.Vector:
    if values is None:
        return mathutils.Vector(default)
    vals = list(values)[:3]
    while len(vals) < 3:
        vals.append(0.0)
    try:
        return mathutils.Vector((float(vals[0]), float(vals[1]), float(vals[2])))
    except Exception:
        return mathutils.Vector(default)


def get_scene_world_offset(meta: Dict[str, Any]) -> mathutils.Vector:
    ego_meta = meta.get("ego_vehicle", {}) or {}
    return _vector3(ego_meta.get("scene_world_offset_blender"), (0.0, 0.0, 0.0))


def offset_blender_point(point: Sequence[float], world_offset: mathutils.Vector) -> Tuple[float, float, float]:
    vals = list(point)[:3]
    while len(vals) < 3:
        vals.append(0.0)
    return (
        float(vals[0]) + float(world_offset.x),
        float(vals[1]) + float(world_offset.y),
        float(vals[2]) + float(world_offset.z),
    )


def setup_camera(
    meta: Dict[str, Any],
    world_offset: Optional[mathutils.Vector] = None,
    camera_mode: str = "chase",
    chase_distance: float = 8.5,
    chase_height: float = 4.8,
    chase_pitch_deg: float = 22.0,
) -> bpy.types.Object:
    """
    Build the render camera.

    Two modes are supported:

    * ``dashcam``  – calibrated 1st-person POV using the front-camera
      intrinsics.  This matches the back-projection geometry exactly but
      hides the ego car from the rendered frame.
    * ``chase``    – 3rd-person camera positioned behind and above the ego
      vehicle (EinsteinVision style).  The ego vehicle is rendered in the
      foreground and all detections / lanes / road geometry are visible
      relative to it.  This is the default.
    """

    camera_meta = meta.get("camera", {})
    frame_w = int(meta.get("frame_w", 1280))
    frame_h = int(meta.get("frame_h", 720))
    calib_meta = meta.get("calib", {})
    camera_pitch = float(calib_meta.get("pitch_rad", camera_meta.get("pitch_rad", 0.0)))
    scene_offset = world_offset if world_offset is not None else mathutils.Vector((0.0, 0.0, 0.0))
    base_location = _vector3(
        camera_meta.get("location_blender"),
        (
            0.0,
            0.0,
            float(calib_meta.get("camera_height_m", camera_meta.get("camera_height_m", 1.5))),
        ),
    )

    cam_data = bpy.data.cameras.new("EgoCam_data")
    cam_obj = bpy.data.objects.new("EgoCamera", cam_data)
    link_to_collection(cam_obj, ensure_collection(COL_CAMERA, ensure_collection(COL_ROOT)))

    cam_data.clip_start = float(camera_meta.get("clip_start", 0.1))
    cam_data.clip_end = float(camera_meta.get("clip_end", 250.0))

    if camera_mode == "dashcam":
        # Calibrated 1st-person view: respect intrinsics + principal-point shift.
        cam_data.lens = float(camera_meta.get("focal_length_mm", 26.0))
        cam_data.sensor_width = float(camera_meta.get("sensor_width_mm", 36.0))
        cam_data.shift_x = float(camera_meta.get("shift_x", 0.0))
        cam_data.shift_y = float(camera_meta.get("shift_y", 0.0))
        cam_obj.location = base_location + scene_offset
        forward_dir = mathutils.Vector(
            (
                math.cos(camera_pitch),
                0.0,
                -math.sin(camera_pitch),
            )
        )
        if forward_dir.length < 1e-6:
            forward_dir = mathutils.Vector((1.0, 0.0, 0.0))
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = forward_dir.normalized().to_track_quat("-Z", "Y")
    else:
        # Chase camera: positioned behind and above the ego vehicle origin (0, 0, 0),
        # looking forward and slightly down so the ego car is in the foreground and
        # the road / detections are clearly visible in front of it.
        cam_data.lens = 35.0
        cam_data.sensor_width = 36.0
        cam_data.shift_x = 0.0
        cam_data.shift_y = 0.0
        chase_distance = max(2.0, float(chase_distance))
        chase_height = max(1.5, float(chase_height))
        pitch_rad = math.radians(float(chase_pitch_deg))
        cam_obj.location = mathutils.Vector(
            (-chase_distance, 0.0, chase_height)
        ) + scene_offset
        # Aim the camera at a point in front of the ego car so the host car
        # sits in the lower-third of the frame and the road dominates above.
        look_at = mathutils.Vector((6.0, 0.0, 0.6)) + scene_offset
        direction = (look_at - cam_obj.location).normalized()
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")

    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.resolution_x = frame_w
    scene.render.resolution_y = frame_h
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()

    print(
        f"[blender] Camera set: mode={camera_mode} {frame_w}x{frame_h}, "
        f"lens={cam_data.lens:.3f}mm, shift=({cam_data.shift_x:.5f}, {cam_data.shift_y:.5f}), "
        f"loc=({cam_obj.location.x:.2f}, {cam_obj.location.y:.2f}, {cam_obj.location.z:.2f})"
    )
    forward = cam_obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))
    print(f"[blender] Camera forward vector: ({forward.x:.3f}, {forward.y:.3f}, {forward.z:.3f})")
    return cam_obj


def pixel_to_blender(u: float, v: float, depth_m: float, calib_meta: Dict[str, Any]) -> Tuple[float, float, float]:
    fx = float(calib_meta["fx"])
    fy = float(calib_meta["fy"])
    cx = float(calib_meta["cx"])
    cy = float(calib_meta["cy"])
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    return float(depth_m), -float(X), max(0.0, -float(Y))


def clean_ground_points(
    points: Sequence[Sequence[float]],
    *,
    smooth_lateral: bool = False,
    allow_elevated: bool = False,
) -> List[Tuple[float, float, float]]:
    cleaned: List[Tuple[float, float, float]] = []
    max_z = MAX_GROUND_HEIGHT_M if allow_elevated else 2.5

    for point in points:
        if len(point) < 3:
            continue
        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        if x < 0.35 or x > MAX_USEFUL_DEPTH_M:
            continue
        if abs(y) > MAX_GROUND_LATERAL_M:
            continue
        if z < -1.0 or z > max_z:
            continue
        candidate = (x, y, z)
        if cleaned and math.dist(cleaned[-1], candidate) < 0.05:
            continue
        cleaned.append(candidate)

    if smooth_lateral and len(cleaned) >= 3:
        smoothed: List[Tuple[float, float, float]] = []
        for idx, (x, _, z) in enumerate(cleaned):
            lo = max(0, idx - 1)
            hi = min(len(cleaned), idx + 2)
            neighborhood = [cleaned[j][1] for j in range(lo, hi)]
            smoothed.append((x, float(statistics.median(neighborhood)), z))
        cleaned = smoothed

    return cleaned


def convex_hull_xy(points: Sequence[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    unique = sorted({(round(p[0], 5), round(p[1], 5), round(p[2], 5)) for p in points})
    if len(unique) <= 3:
        return [(float(x), float(y), float(z)) for x, y, z in unique]

    pts2d = [(float(x), float(y), float(z)) for x, y, z in unique]

    def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float, float]] = []
    for point in pts2d:
        while len(lower) >= 2 and cross(
            (lower[-2][0], lower[-2][1]),
            (lower[-1][0], lower[-1][1]),
            (point[0], point[1]),
        ) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: List[Tuple[float, float, float]] = []
    for point in reversed(pts2d):
        while len(upper) >= 2 and cross(
            (upper[-2][0], upper[-2][1]),
            (upper[-1][0], upper[-1][1]),
            (point[0], point[1]),
        ) <= 0.0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return pts2d
    return hull


def triangulate_xy_polygon(points: Sequence[Tuple[float, float, float]]) -> List[Tuple[int, int, int]]:
    """Triangulate an ordered XY contour without collapsing it to a convex hull.

    The road contours emitted by ``scene_assembler.py`` are already ordered and
    often intentionally concave around ramps, merges, and split shoulders.
    Turning them into a convex hull makes the road surface unrealistically
    bridge across empty space, which shows up as large floating polygons in the
    render.  We therefore tessellate the original contour directly and only
    fall back to a simple fan if Blender's triangulator cannot help.
    """
    if len(points) < 3:
        return []

    contour_2d = [mathutils.Vector((float(x), float(y))) for x, y, _ in points]
    faces: List[Tuple[int, int, int]] = []
    try:
        triangles = mathutils.geometry.tessellate_polygon([contour_2d])
    except Exception:
        triangles = []

    for tri in triangles:
        if len(tri) != 3 or len(set(tri)) != 3:
            continue
        a, b, c = [contour_2d[int(idx)] for idx in tri]
        area = abs(
            float(a.x) * (float(b.y) - float(c.y))
            + float(b.x) * (float(c.y) - float(a.y))
            + float(c.x) * (float(a.y) - float(b.y))
        ) * 0.5
        if area < 1e-5:
            continue
        faces.append((int(tri[0]), int(tri[1]), int(tri[2])))

    if faces:
        return faces

    # Conservative fallback: preserve the original contour order instead of
    # inventing a convex hull, even if triangulation is unavailable.
    return [
        (0, idx, idx + 1)
        for idx in range(1, len(points) - 1)
    ]


def create_background_plate(
    cam_obj: bpy.types.Object,
    video_path: str,
    collection: bpy.types.Collection,
) -> Optional[bpy.types.Object]:
    movie_image = get_movie_image(video_path)
    if movie_image is None:
        print(f"[blender] Background plate skipped, video not found: {video_path}")
        return None

    scene = bpy.context.scene
    corners = cam_obj.data.view_frame(scene=scene)
    target_depth = 150.0
    scale = target_depth / abs(corners[0].z)
    verts = [(corner.x * scale, corner.y * scale, corner.z * scale) for corner in corners]

    mesh = bpy.data.meshes.new("BackgroundPlateMesh")
    mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
    mesh.update()

    plate = bpy.data.objects.new("BackgroundPlate", mesh)
    plate.parent = cam_obj
    plate.matrix_parent_inverse = cam_obj.matrix_world.inverted()
    plate.location = (0.0, 0.0, 0.0)
    plate.rotation_euler = (0.0, 0.0, 0.0)
    plate.hide_select = True

    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vert_idx = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uv_coords[vert_idx]

    material = make_movie_material("M_BackgroundPlate", movie_image, emission=True)
    mesh.materials.clear()
    mesh.materials.append(material)

    link_to_collection(plate, collection)
    return plate


# ============================================================================
# Asset import / instancing
# ============================================================================

_TEMPLATE_CACHE: Dict[str, bpy.types.Object] = {}
_TEMPLATE_COLLECTION: Optional[bpy.types.Collection] = None


def select_only(obj: bpy.types.Object) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.update()


def local_bbox_min_max(obj: bpy.types.Object) -> Tuple[mathutils.Vector, mathutils.Vector]:
    bbox = [mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector(
        (
            min((corner.x for corner in bbox), default=0.0),
            min((corner.y for corner in bbox), default=0.0),
            min((corner.z for corner in bbox), default=0.0),
        )
    )
    max_corner = mathutils.Vector(
        (
            max((corner.x for corner in bbox), default=0.0),
            max((corner.y for corner in bbox), default=0.0),
            max((corner.z for corner in bbox), default=0.0),
        )
    )
    return min_corner, max_corner


def world_bbox_min_max(obj: bpy.types.Object) -> Tuple[mathutils.Vector, mathutils.Vector]:
    """Return the object's world-space bounding-box limits."""
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_corner = mathutils.Vector(
        (
            min((corner.x for corner in bbox), default=0.0),
            min((corner.y for corner in bbox), default=0.0),
            min((corner.z for corner in bbox), default=0.0),
        )
    )
    max_corner = mathutils.Vector(
        (
            max((corner.x for corner in bbox), default=0.0),
            max((corner.y for corner in bbox), default=0.0),
            max((corner.z for corner in bbox), default=0.0),
        )
    )
    return min_corner, max_corner


def infer_asset_class(asset_rel: str, semantic_class: Optional[str] = None) -> str:
    if semantic_class:
        return str(semantic_class)
    lower = str(asset_rel).lower()
    if "traffic" in lower and "signal" in lower:
        return "traffic_light"
    if "stopsign" in lower or "stop_sign" in lower:
        return "stop_sign"
    if "speedlimit" in lower or "speed_limit" in lower:
        return "speed_limit"
    if "motorcycle" in lower:
        return "motorcycle"
    if "bicycle" in lower or "bike" in lower:
        return "bicycle"
    if "pickup" in lower:
        return "car"
    if "suv" in lower or "sedan" in lower or "hatchback" in lower:
        return "car"
    if "truck" in lower:
        return "truck"
    if "dustbin" in lower or "trash" in lower or "bin" in lower:
        return "dustbin"
    if "hydrant" in lower:
        return "fire_hydrant"
    if "cylinder" in lower or "bollard" in lower:
        return "traffic_cylinder"
    if "cone" in lower:
        return "traffic_cone"
    if "pole" in lower or "post" in lower:
        return "traffic_pole"
    if "pedestr" in lower:
        return "pedestrian"
    return "car"


def filter_imported_asset_objects(
    imported: Sequence[bpy.types.Object],
    asset_rel: str,
    preferred_class: Optional[str] = None,
) -> List[bpy.types.Object]:
    """
    Keep only the principal mesh objects of a loaded asset before joining.

    Some source ``.blend`` files contain helper geometry, far-away loose pieces,
    or support meshes that should never be rendered as part of the semantic
    asset.  This filter keeps the dominant coherent cluster and removes the
    obvious outliers before the mesh-level loose-part cleanup runs.
    """
    imported_objects = list(imported)
    asset_lower = str(asset_rel).lower()
    if preferred_class == "traffic_light" and "trafficsignal" in asset_lower:
        return list(imported_objects)

    if preferred_class in {"traffic_light", "traffic_cone", "traffic_cylinder", "traffic_pole", "dustbin", "fire_hydrant", "stop_sign", "speed_limit"}:
        aliases = {
            "traffic_light": ("traffic_signal", "signal"),
            "traffic_cone": ("cone",),
            "traffic_cylinder": ("cylinder", "bollard", "barrel"),
            "traffic_pole": ("pole", "post"),
            "dustbin": ("dustbin", "trash", "bin"),
            "fire_hydrant": ("hydrant",),
            "stop_sign": ("stop", "sign"),
            "speed_limit": ("speed", "limit"),
        }.get(preferred_class, ())
        if aliases:
            named = [obj for obj in imported_objects if any(alias in obj.name.lower() for alias in aliases)]
            if named:
                imported_objects = named

    if len(imported_objects) <= 1:
        return list(imported_objects)

    inferred_class = preferred_class or infer_asset_class(asset_rel)
    if inferred_class == "pedestrian":
        return list(imported_objects)
    prior_dims = CLASS_PRIMITIVE_DIMS.get(inferred_class, CLASS_PRIMITIVE_DIMS["car"])
    prior_max_dim = 4.5 * max(prior_dims)

    scored: List[Dict[str, Any]] = []
    for obj in imported_objects:
        if obj.data is None:
            continue
        min_corner, max_corner = world_bbox_min_max(obj)
        dims = max_corner - min_corner
        center = 0.5 * (min_corner + max_corner)
        face_count = len(getattr(obj.data, "polygons", []))
        volume_score = max(float(dims.x), 0.02) * max(float(dims.y), 0.02) * max(float(dims.z), 0.02)
        score = volume_score * max(face_count, 1)
        scored.append(
            {
                "obj": obj,
                "min": min_corner,
                "max": max_corner,
                "dims": dims,
                "center": center,
                "face_count": face_count,
                "score": score,
            }
        )

    if len(scored) <= 1:
        return [item["obj"] for item in scored]

    scored.sort(key=lambda item: (float(item["score"]), int(item["face_count"])), reverse=True)
    core = scored[0]
    core_dims = core["dims"]
    pad_x = max(0.35, 0.35 * float(core_dims.x))
    pad_y = max(0.35, 0.55 * float(core_dims.y))
    pad_z = max(0.35, 0.55 * float(core_dims.z))

    kept: List[bpy.types.Object] = []
    removed = 0
    for item in scored:
        dims = item["dims"]
        max_dim = max(float(dims.x), float(dims.y), float(dims.z))
        overlaps_core = (
            item["max"].x >= (core["min"].x - pad_x)
            and item["min"].x <= (core["max"].x + pad_x)
            and item["max"].y >= (core["min"].y - pad_y)
            and item["min"].y <= (core["max"].y + pad_y)
            and item["max"].z >= (core["min"].z - pad_z)
            and item["min"].z <= (core["max"].z + pad_z)
        )
        close_center = (item["center"] - core["center"]).length <= max(2.0, 0.85 * max(float(core_dims.x), float(core_dims.y), float(core_dims.z)))
        significant = float(item["score"]) >= 0.015 * float(core["score"])
        plausible_size = max_dim <= prior_max_dim

        if item is core or (plausible_size and significant and (overlaps_core or close_center)):
            kept.append(item["obj"])
            continue

        bpy.data.objects.remove(item["obj"], do_unlink=True)
        removed += 1

    if removed:
        print(f"[blender] Filtered {removed} imported helper object(s) from {asset_rel}")
    return kept


def transform_mesh_data(obj: bpy.types.Object, matrix: mathutils.Matrix) -> None:
    if obj.data is None:
        return
    obj.data.transform(matrix)
    obj.data.update()
    bpy.context.view_layer.update()


def component_min_max(
    verts: Sequence[Any],
) -> Tuple[mathutils.Vector, mathutils.Vector]:
    min_corner = mathutils.Vector(
        (
            min((vert.co.x for vert in verts), default=0.0),
            min((vert.co.y for vert in verts), default=0.0),
            min((vert.co.z for vert in verts), default=0.0),
        )
    )
    max_corner = mathutils.Vector(
        (
            max((vert.co.x for vert in verts), default=0.0),
            max((vert.co.y for vert in verts), default=0.0),
            max((vert.co.z for vert in verts), default=0.0),
        )
    )
    return min_corner, max_corner


def sanitize_template_loose_parts(obj: bpy.types.Object) -> None:
    if obj.data is None:
        return

    try:
        import bmesh
    except ImportError:
        return

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        if len(bm.verts) < 8 or len(bm.faces) < 4:
            return

        remaining = set(bm.verts)
        components: List[Dict[str, Any]] = []

        while remaining:
            seed = remaining.pop()
            stack = [seed]
            verts = [seed]
            faces = set(seed.link_faces)

            while stack:
                vert = stack.pop()
                for edge in vert.link_edges:
                    for nbr in edge.verts:
                        if nbr in remaining:
                            remaining.remove(nbr)
                            stack.append(nbr)
                            verts.append(nbr)
                    for face in edge.link_faces:
                        faces.add(face)

            min_corner, max_corner = component_min_max(verts)
            dims = max_corner - min_corner
            volume_score = max(float(dims.x), 0.01) * max(float(dims.y), 0.01) * max(float(dims.z), 0.2)
            center = 0.5 * (min_corner + max_corner)
            components.append(
                {
                    "verts": verts,
                    "faces": list(faces),
                    "min": min_corner,
                    "max": max_corner,
                    "dims": dims,
                    "center": center,
                    "score": volume_score,
                }
            )

        if len(components) <= 1:
            return

        components.sort(
            key=lambda comp: (float(comp["score"]), len(comp["faces"]), len(comp["verts"])),
            reverse=True,
        )
        core = components[0]
        core_min = core["min"]
        core_max = core["max"]
        core_dims = core_max - core_min

        pad_x = max(0.55, 0.35 * core_dims.x)
        pad_y = max(0.45, 0.90 * core_dims.y)
        pad_z = max(0.35, 0.75 * core_dims.z)

        to_delete = []
        removed = 0
        for comp in components[1:]:
            overlaps_core_bounds = (
                comp["max"].x >= (core_min.x - pad_x)
                and comp["min"].x <= (core_max.x + pad_x)
                and comp["max"].y >= (core_min.y - pad_y)
                and comp["min"].y <= (core_max.y + pad_y)
                and comp["max"].z >= (core_min.z - pad_z)
                and comp["min"].z <= (core_max.z + 0.55 * pad_z)
            )
            if overlaps_core_bounds:
                continue
            to_delete.extend(comp["verts"])
            removed += 1

        if to_delete:
            bmesh.ops.delete(bm, geom=to_delete, context="VERTS")
            bm.to_mesh(obj.data)
            obj.data.update()
            print(f"[blender] Sanitized {obj.name}: removed {removed} loose mesh island(s)")
    finally:
        bm.free()


def normalise_template_geometry(
    obj: bpy.types.Object,
    inferred_class: Optional[str] = None,
    asset_rel: Optional[str] = None,
) -> None:
    select_only(obj)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    asset_lower = str(asset_rel or "").lower()
    if not (inferred_class == "traffic_light" and "trafficsignal" in asset_lower):
        sanitize_template_loose_parts(obj)

    min_corner, max_corner = local_bbox_min_max(obj)
    x_dim = max_corner.x - min_corner.x
    y_dim = max_corner.y - min_corner.y

    if inferred_class in {"traffic_light", "stop_sign", "speed_limit"}:
        # Upright road furniture should keep its thin axis on local X so later
        # scale_x matches physical thickness while scale_y matches face width.
        if x_dim > y_dim:
            transform_mesh_data(obj, mathutils.Matrix.Rotation(-math.pi / 2.0, 4, "Z"))
            min_corner, max_corner = local_bbox_min_max(obj)
    elif y_dim > x_dim:
        transform_mesh_data(obj, mathutils.Matrix.Rotation(-math.pi / 2.0, 4, "Z"))
        min_corner, max_corner = local_bbox_min_max(obj)

    center_x = 0.5 * (min_corner.x + max_corner.x)
    center_y = 0.5 * (min_corner.y + max_corner.y)
    transform_mesh_data(
        obj,
        mathutils.Matrix.Translation((-center_x, -center_y, -min_corner.z)),
    )
    obj.location = (0.0, 0.0, 0.0)

    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    obj["_template_length"] = max(float(dims.x), 0.01)
    obj["_template_width"] = max(float(dims.y), 0.01)
    obj["_template_height"] = max(float(dims.z), 0.01)
    obj["_template_yaw_offset"] = 0.0


def template_geometry_is_plausible(obj: bpy.types.Object, asset_rel: str) -> bool:
    """Reject cached templates whose aspect ratios are wildly inconsistent with their class prior."""
    inferred_class = infer_asset_class(asset_rel)
    if inferred_class == "pedestrian":
        return True
    prior = CLASS_PRIMITIVE_DIMS.get(inferred_class, CLASS_PRIMITIVE_DIMS["car"])
    length = max(float(obj.get("_template_length", 1.0)), 0.01)
    width = max(float(obj.get("_template_width", 1.0)), 0.01)
    height = max(float(obj.get("_template_height", 1.0)), 0.01)

    prior_length, prior_width, prior_height = prior
    aspect_checks = [
        (length / width, prior_length / max(prior_width, 0.01)),
        (length / height, prior_length / max(prior_height, 0.01)),
        (width / height, prior_width / max(prior_height, 0.01)),
    ]

    for actual, expected in aspect_checks:
        ratio = actual / max(expected, 0.01)
        if ratio < 0.2 or ratio > 5.0:
            print(f"[blender] Rejecting asset template {asset_rel}: implausible aspect ratio after import")
            return False
    return True


def create_asset_template(asset_rel: str, assets_dir: Path) -> Optional[bpy.types.Object]:
    global _TEMPLATE_COLLECTION

    if asset_rel in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[asset_rel]

    asset_path = assets_dir / asset_rel
    if not asset_path.exists():
        return None

    if _TEMPLATE_COLLECTION is None:
        _TEMPLATE_COLLECTION = ensure_collection(COL_TEMPLATES, ensure_collection(COL_ROOT))

    try:
        with bpy.data.libraries.load(str(asset_path), link=False) as (src, dst):
            dst.objects = [name for name in src.objects]
    except Exception as exc:
        print(f"[blender] Failed to open asset {asset_rel}: {exc}")
        return None

    inferred_class = infer_asset_class(asset_rel)
    imported: List[bpy.types.Object] = []
    for obj in dst.objects:
        if obj is None:
            continue
        if inferred_class == "pedestrian":
            keep_type = obj.type in {"MESH", "CURVE", "SURFACE"}
        else:
            keep_type = obj.type == "MESH"
        if not keep_type:
            continue
        bpy.context.scene.collection.objects.link(obj)
        imported.append(obj)

    if not imported:
        return None

    imported = filter_imported_asset_objects(imported, asset_rel, preferred_class=inferred_class)
    if not imported:
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]

    if len(imported) > 1:
        bpy.ops.object.join()

    template = bpy.context.view_layer.objects.active
    template.name = f"TEMPLATE_{Path(asset_rel).stem}_{len(_TEMPLATE_CACHE):03d}"
    normalise_template_geometry(template, inferred_class, asset_rel)
    asset_lower = str(asset_rel).lower()
    if inferred_class == "traffic_light" and "trafficsignal" in asset_lower:
        template["_template_yaw_offset"] = 0.0
        template["_tl_head_bottom_frac"] = 0.10
        template["_tl_head_top_frac"] = 0.90
        template["_traffic_signal_preserve_scale"] = False
    elif inferred_class == "stop_sign" and "stopsign" in asset_lower:
        template["_template_yaw_offset"] = 0.0
        template["_face_bottom_frac"] = 0.72
        template["_face_top_frac"] = 0.94
    elif inferred_class in {"traffic_sign", "speed_limit"} and "speedlimitsign" in asset_lower:
        template["_template_yaw_offset"] = math.pi
        template["_face_bottom_frac"] = 0.66
        template["_face_top_frac"] = 0.94
    if not template_geometry_is_plausible(template, asset_rel):
        bpy.data.objects.remove(template, do_unlink=True)
        return None
    link_to_collection(template, _TEMPLATE_COLLECTION)
    template.hide_render = True
    template.hide_viewport = True
    _TEMPLATE_CACHE[asset_rel] = template
    return template


def instantiate_asset(
    asset_rel: str,
    assets_dir: Path,
    collection: bpy.types.Collection,
    instance_name: str,
    unique_mesh: bool = False,
) -> Optional[bpy.types.Object]:
    template = create_asset_template(asset_rel, assets_dir)
    if template is None:
        return None

    obj = template.copy()
    obj.name = instance_name
    obj.animation_data_clear()
    obj.hide_render = False
    obj.hide_viewport = False
    obj["_template_height"] = float(template.get("_template_height", 1.0))
    obj["_template_width"] = float(template.get("_template_width", 1.0))
    obj["_template_length"] = float(template.get("_template_length", 1.0))
    obj["_template_yaw_offset"] = float(template.get("_template_yaw_offset", 0.0))
    obj["_asset_rel"] = str(asset_rel)
    if "_face_bottom_frac" in template:
        obj["_face_bottom_frac"] = float(template.get("_face_bottom_frac", 0.0))
    if "_face_top_frac" in template:
        obj["_face_top_frac"] = float(template.get("_face_top_frac", 1.0))
    if "_tl_head_bottom_frac" in template:
        obj["_tl_head_bottom_frac"] = float(template.get("_tl_head_bottom_frac", 0.0))
    if "_tl_head_top_frac" in template:
        obj["_tl_head_top_frac"] = float(template.get("_tl_head_top_frac", 1.0))
    if "_traffic_signal_preserve_scale" in template:
        obj["_traffic_signal_preserve_scale"] = bool(template.get("_traffic_signal_preserve_scale", False))

    if unique_mesh and template.data is not None:
        obj.data = template.data.copy()
    else:
        obj.data = template.data

    link_to_collection(obj, collection)
    return obj


def _join_mesh_objects(objects: Sequence[bpy.types.Object], instance_name: str) -> Optional[bpy.types.Object]:
    mesh_objects = [obj for obj in objects if obj is not None and obj.type == "MESH"]
    if not mesh_objects:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = instance_name
    return joined


def create_primitive_fallback(
    cls: str,
    instance_name: str,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    dims = CLASS_PRIMITIVE_DIMS.get(cls, CLASS_PRIMITIVE_DIMS["car"])
    length, width, height = dims

    if cls in {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}:
        created: List[bpy.types.Object] = []
        if cls == "traffic_light":
            head_height = min(0.95, max(0.68, 0.24 * float(height)))
            head_width = max(0.24, 0.84 * float(width))
            head_depth = max(0.18, 0.72 * float(length))
            pole_height = max(1.8, float(height) - 0.78 * head_height)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, pole_height + 0.5 * head_height))
            head = bpy.context.active_object
            head.scale = (0.5 * head_depth, 0.5 * head_width, 0.5 * head_height)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            created.append(head)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.5 * pole_height))
            pole = bpy.context.active_object
            pole.scale = (0.045, 0.045, 0.5 * pole_height)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            created.append(pole)
        elif cls == "stop_sign":
            face_height = 0.78
            face_width = 0.78
            face_depth = 0.07
            pole_height = max(1.15, float(height) - 0.70 * face_height)

            bpy.ops.mesh.primitive_cylinder_add(
                vertices=8,
                radius=0.5 * face_width,
                depth=face_depth,
                rotation=(0.0, math.pi / 2.0, math.pi / 8.0),
                location=(0.0, 0.0, pole_height + 0.5 * face_height),
            )
            face = bpy.context.active_object
            face.scale = (1.0, 1.0, max(0.75, face_height / max(face_width, 0.01)))
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            created.append(face)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.5 * pole_height))
            pole = bpy.context.active_object
            pole.scale = (0.03, 0.03, 0.5 * pole_height)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            created.append(pole)
        else:
            face_height = min(max(0.62, 0.34 * float(height)), 0.95 if cls == "stop_sign" else 0.88)
            face_width = max(0.45, 0.92 * float(width))
            face_depth = max(0.07, 0.82 * float(length))
            pole_height = max(1.15, float(height) - 0.72 * face_height)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, pole_height + 0.5 * face_height))
            face = bpy.context.active_object
            face.scale = (0.5 * face_depth, 0.5 * face_width, 0.5 * face_height)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            created.append(face)

            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.5 * pole_height))
            pole = bpy.context.active_object
            pole.scale = (0.03, 0.03, 0.5 * pole_height)
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            created.append(pole)

        obj = _join_mesh_objects(created, instance_name)
        if obj is None:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, height * 0.5))
            obj = bpy.context.active_object
            obj.name = instance_name
        if cls == "traffic_light":
            obj["_tl_head_bottom_frac"] = round(float(pole_height) / max(float(height), 1e-6), 4)
            obj["_tl_head_top_frac"] = 1.0
        elif cls == "stop_sign":
            obj["_face_bottom_frac"] = round(float(pole_height) / max(float(height), 1e-6), 4)
            obj["_face_top_frac"] = round((float(pole_height) + float(face_height)) / max(float(height), 1e-6), 4)
            obj["_template_yaw_offset"] = 0.0
        elif cls == "speed_limit":
            obj["_face_bottom_frac"] = round(float(pole_height) / max(float(height), 1e-6), 4)
            obj["_face_top_frac"] = round((float(pole_height) + float(face_height)) / max(float(height), 1e-6), 4)
            obj["_template_yaw_offset"] = math.pi
    elif cls == "traffic_pole":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=18,
            radius=0.5 * max(length, width),
            depth=height,
            location=(0.0, 0.0, height * 0.5),
        )
    elif cls == "speed_bump":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, height * 0.5))
        obj = bpy.context.active_object
        obj.name = instance_name
        obj.scale = (length, width, height)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif cls == "traffic_cone":
        bpy.ops.mesh.primitive_cone_add(
            vertices=20,
            radius1=0.5 * max(length, width),
            radius2=max(0.02, 0.12 * max(length, width)),
            depth=height,
            location=(0.0, 0.0, height * 0.5),
        )
    elif cls == "traffic_cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=20,
            radius=0.5 * max(length, width),
            depth=height,
            location=(0.0, 0.0, height * 0.5),
        )
    elif cls == "fire_hydrant":
        created = []
        body_height = 0.62 * float(height)
        body_radius = 0.38 * max(length, width)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=20,
            radius=body_radius,
            depth=body_height,
            location=(0.0, 0.0, 0.5 * body_height),
        )
        created.append(bpy.context.active_object)

        top_height = max(0.16, 0.22 * float(height))
        top_radius = 0.28 * max(length, width)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=20,
            radius=top_radius,
            depth=top_height,
            location=(0.0, 0.0, body_height + 0.5 * top_height),
        )
        created.append(bpy.context.active_object)

        cap_radius = 0.22 * max(length, width)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16,
            ring_count=8,
            radius=cap_radius,
            location=(0.0, 0.0, body_height + top_height),
        )
        created.append(bpy.context.active_object)

        arm_radius = 0.13 * max(length, width)
        arm_length = 0.45 * float(width)
        arm_z = 0.52 * body_height
        for arm_sign in (-1.0, 1.0):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=16,
                radius=arm_radius,
                depth=arm_length,
                location=(0.0, arm_sign * 0.24 * float(width), arm_z),
                rotation=(math.pi / 2.0, 0.0, 0.0),
            )
            created.append(bpy.context.active_object)
        obj = _join_mesh_objects(created, instance_name)
        if obj is None:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=18,
                radius=0.5 * max(length, width),
                depth=height,
                location=(0.0, 0.0, 0.5 * height),
            )
            obj = bpy.context.active_object
            obj.name = instance_name
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, height * 0.5))
        obj = bpy.context.active_object
        obj.name = instance_name

    if cls not in {"traffic_pole", "traffic_cone", "traffic_cylinder", "traffic_light", "traffic_sign", "stop_sign", "speed_limit", "speed_bump", "fire_hydrant"}:
        obj.scale = (length, width, height)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    elif cls in {"traffic_pole", "traffic_cone", "traffic_cylinder"}:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    normalise_template_geometry(obj, cls)

    material = make_solid_material(
        f"M_Primitive_{cls}",
        CLASS_COLORS.get(cls, (0.5, 0.5, 0.5, 1.0)),
        roughness=0.72 if cls in {"car", "truck"} else 0.82,
        metallic=0.08 if cls in {"car", "truck"} else 0.0,
        specular=0.12 if cls in {"car", "truck"} else 0.02,
    )
    obj.data.materials.clear()
    obj.data.materials.append(material)

    link_to_collection(obj, collection)
    return obj


def _horizontal_thin_axis(dims: mathutils.Vector) -> str:
    return "x" if float(dims.x) <= float(dims.y) else "y"


def _camera_local_position(obj: bpy.types.Object) -> Optional[mathutils.Vector]:
    camera = getattr(bpy.context.scene, "camera", None)
    if camera is None:
        return None
    try:
        return obj.matrix_world.inverted() @ camera.matrix_world.translation
    except Exception:
        return None


def _camera_facing_face_transform(
    obj: bpy.types.Object,
    *,
    z_fraction: float = 0.7,
    outward_fraction: float = 0.03,
    roll_rad: float = 0.0,
    right_fraction: float = 0.0,
    up_fraction: float = 0.0,
) -> Optional[Tuple[mathutils.Vector, mathutils.Euler, float, float]]:
    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    if float(dims.z) <= 0.01:
        return None

    thin_axis = _horizontal_thin_axis(dims)
    center_x = 0.5 * (min_corner.x + max_corner.x)
    center_y = 0.5 * (min_corner.y + max_corner.y)
    face_bottom_frac = obj.get("_face_bottom_frac", None)
    face_top_frac = obj.get("_face_top_frac", None)
    head_bottom_frac = float(obj.get("_tl_head_bottom_frac", 0.0))
    head_top_frac = float(obj.get("_tl_head_top_frac", 1.0))
    if (
        face_bottom_frac is not None
        and face_top_frac is not None
        and 0.0 <= float(face_bottom_frac) < float(face_top_frac) <= 1.0
    ):
        face_bottom_z = float(min_corner.z) + float(face_bottom_frac) * float(dims.z)
        face_top_z = float(min_corner.z) + float(face_top_frac) * float(dims.z)
        face_height = max(0.12, face_top_z - face_bottom_z)
        center_z = face_bottom_z + max(0.05, min(0.95, float(z_fraction))) * face_height
    elif 0.0 <= head_bottom_frac < head_top_frac <= 1.0 and str(obj.name).lower().startswith("traffic_light"):
        head_bottom_z = float(min_corner.z) + head_bottom_frac * float(dims.z)
        head_top_z = float(min_corner.z) + head_top_frac * float(dims.z)
        head_height = max(0.12, head_top_z - head_bottom_z)
        center_z = head_bottom_z + max(0.05, min(0.95, float(z_fraction))) * head_height
        face_height = head_height
    else:
        center_z = float(min_corner.z) + max(0.05, min(0.95, float(z_fraction))) * float(dims.z)
        face_height = float(dims.z)
    outward = max(0.008, float(outward_fraction) * max(float(dims.x), float(dims.y), 0.1))
    camera_local = _camera_local_position(obj)

    if thin_axis == "x":
        normal_sign = -1.0 if camera_local is None or float(camera_local.x) <= 0.0 else 1.0
        x_face = float(min_corner.x) - outward if normal_sign < 0.0 else float(max_corner.x) + outward
        location = mathutils.Vector((x_face, center_y, center_z))
        horizontal = mathutils.Vector((0.0, normal_sign, 0.0))
        vertical = mathutils.Vector((0.0, 0.0, 1.0))
        normal = mathutils.Vector((normal_sign, 0.0, 0.0))
        location += horizontal.normalized() * (float(right_fraction) * max(float(dims.y), 0.08))
        location += vertical.normalized() * (float(up_fraction) * max(face_height, 0.08))
        rotation_matrix = mathutils.Matrix((horizontal, vertical, normal)).transposed()
        if abs(float(roll_rad)) > 1e-6:
            rotation_matrix = rotation_matrix @ mathutils.Matrix.Rotation(float(roll_rad), 3, "Z")
        rotation = rotation_matrix.to_euler("XYZ")
        return location, rotation, max(float(dims.y), 0.08), max(face_height, 0.08)

    normal_sign = -1.0 if camera_local is None or float(camera_local.y) <= 0.0 else 1.0
    y_face = float(min_corner.y) - outward if normal_sign < 0.0 else float(max_corner.y) + outward
    location = mathutils.Vector((center_x, y_face, center_z))
    horizontal = mathutils.Vector((-normal_sign, 0.0, 0.0))
    vertical = mathutils.Vector((0.0, 0.0, 1.0))
    normal = mathutils.Vector((0.0, normal_sign, 0.0))
    location += horizontal.normalized() * (float(right_fraction) * max(float(dims.x), 0.08))
    location += vertical.normalized() * (float(up_fraction) * max(face_height, 0.08))
    rotation_matrix = mathutils.Matrix((horizontal, vertical, normal)).transposed()
    if abs(float(roll_rad)) > 1e-6:
        rotation_matrix = rotation_matrix @ mathutils.Matrix.Rotation(float(roll_rad), 3, "Z")
    rotation = rotation_matrix.to_euler("XYZ")
    return location, rotation, max(float(dims.x), 0.08), max(face_height, 0.08)


def _ensure_text_overlay(
    parent: bpy.types.Object,
    child_suffix: str,
    body: str,
    material: bpy.types.Material,
    *,
    z_fraction: float,
    width_fraction: float,
    height_fraction: float,
    outward_fraction: float = 0.03,
    roll_rad: float = 0.0,
    right_fraction: float = 0.0,
    up_fraction: float = 0.0,
) -> Optional[bpy.types.Object]:
    child_name = f"{parent.name}_{child_suffix}"
    child = bpy.data.objects.get(child_name)

    if not body:
        if child is not None:
            child.hide_render = True
            child.hide_viewport = True
        return None

    if child is None or child.type != "FONT":
        if child is not None:
            bpy.data.objects.remove(child, do_unlink=True)
        text_data = bpy.data.curves.new(child_name, type="FONT")
        child = bpy.data.objects.new(child_name, text_data)
        child.hide_select = True
        target_collection = parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection
        link_to_collection(child, target_collection)
        child.parent = parent
        child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    else:
        text_data = child.data

    transform = _camera_facing_face_transform(
        parent,
        z_fraction=z_fraction,
        outward_fraction=outward_fraction,
        roll_rad=roll_rad,
        right_fraction=right_fraction,
        up_fraction=up_fraction,
    )
    if transform is None:
        child.hide_render = True
        child.hide_viewport = True
        return child

    location, rotation, face_width, face_height = transform
    text_data.body = str(body)
    if hasattr(text_data, "align_x"):
        text_data.align_x = "CENTER"
    if hasattr(text_data, "align_y"):
        text_data.align_y = "CENTER"
    if hasattr(text_data, "fill_mode"):
        text_data.fill_mode = "BOTH"
    text_data.extrude = max(0.003, 0.018 * min(face_width, face_height))
    text_data.bevel_depth = 0.0
    text_data.size = max(
        0.06,
        min(
            float(face_height) * float(height_fraction),
            float(face_width) * float(width_fraction) / max(len(str(body)), 1),
        ),
    )

    child.rotation_mode = "XYZ"
    child.location = location
    child.rotation_euler = rotation
    child.scale = (1.0, 1.0, 1.0)
    child.hide_render = False
    child.hide_viewport = False

    text_data.materials.clear()
    text_data.materials.append(material)
    return child


def _ensure_overlay_plane(
    parent: bpy.types.Object,
    child_suffix: str,
    material: bpy.types.Material,
) -> bpy.types.Object:
    child_name = f"{parent.name}_{child_suffix}"
    child = bpy.data.objects.get(child_name)

    if child is None or child.type != "MESH":
        if child is not None:
            bpy.data.objects.remove(child, do_unlink=True)
        mesh = bpy.data.meshes.new(f"{child_name}_Mesh")
        mesh.from_pydata(
            [
                (-0.5, -0.5, 0.0),
                (0.5, -0.5, 0.0),
                (0.5, 0.5, 0.0),
                (-0.5, 0.5, 0.0),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.update()
        child = bpy.data.objects.new(child_name, mesh)
        child.hide_select = True
        target_collection = parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection
        link_to_collection(child, target_collection)
        child.parent = parent
        child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    child.hide_render = False
    child.hide_viewport = False
    if child.data is not None and hasattr(child.data, "materials"):
        if hasattr(child.data, "uv_layers") and len(child.data.uv_layers) == 0:
            child.data.uv_layers.new(name="UVMap")
        child.data.materials.clear()
        child.data.materials.append(material)
    return child


def _ensure_overlay_disc(
    parent: bpy.types.Object,
    child_suffix: str,
    material: bpy.types.Material,
    *,
    segments: int = 28,
) -> bpy.types.Object:
    child_name = f"{parent.name}_{child_suffix}"
    child = bpy.data.objects.get(child_name)
    recreate = (
        child is None
        or child.type != "MESH"
        or not bool(child.get("_overlay_disc", False))
    )

    if recreate:
        if child is not None:
            bpy.data.objects.remove(child, do_unlink=True)
        mesh = bpy.data.meshes.new(f"{child_name}_Mesh")
        verts: List[Tuple[float, float, float]] = []
        uvs: List[Tuple[float, float]] = []
        for idx in range(max(12, int(segments))):
            theta = (2.0 * math.pi * float(idx)) / float(max(12, int(segments)))
            x = 0.5 * math.cos(theta)
            y = 0.5 * math.sin(theta)
            verts.append((x, y, 0.0))
            uvs.append((0.5 + x, 0.5 + y))
        mesh.from_pydata(verts, [], [tuple(range(len(verts)))])
        mesh.update()
        if len(mesh.uv_layers) == 0:
            mesh.uv_layers.new(name="UVMap")
        uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        if uv_layer is not None:
            for loop in mesh.loops:
                uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
        child = bpy.data.objects.new(child_name, mesh)
        child.hide_select = True
        child["_overlay_disc"] = True
        target_collection = parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection
        link_to_collection(child, target_collection)
        child.parent = parent
        child.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    child.hide_render = False
    child.hide_viewport = False
    if child.data is not None and hasattr(child.data, "materials"):
        child.data.materials.clear()
        child.data.materials.append(material)
    return child


def _ensure_overlay_cylinder(
    parent: bpy.types.Object,
    child_suffix: str,
    material: bpy.types.Material,
    *,
    vertices: int = 24,
) -> bpy.types.Object:
    child_name = f"{parent.name}_{child_suffix}"
    child = bpy.data.objects.get(child_name)
    recreate = (
        child is None
        or child.type != "MESH"
        or not bool(child.get("_overlay_cylinder", False))
    )

    if recreate:
        if child is not None:
            bpy.data.objects.remove(child, do_unlink=True)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=max(12, int(vertices)),
            radius=0.5,
            depth=1.0,
            location=(0.0, 0.0, 0.0),
        )
        child = bpy.context.active_object
        child.name = child_name
        child.hide_select = True
        child["_overlay_cylinder"] = True
        target_collection = parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection
        link_to_collection(child, target_collection)
        child.parent = parent
        child.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    child.hide_render = False
    child.hide_viewport = False
    if child.data is not None and hasattr(child.data, "materials"):
        child.data.materials.clear()
        child.data.materials.append(material)
    return child


def _set_plane_uvs(
    obj: bpy.types.Object,
    *,
    flip_u: bool = False,
    flip_v: bool = False,
) -> None:
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "uv_layers"):
        return
    if len(mesh.uv_layers) == 0:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
    if uv_layer is None or len(mesh.loops) < 4:
        return

    u0, u1 = (1.0, 0.0) if flip_u else (0.0, 1.0)
    v0, v1 = (1.0, 0.0) if flip_v else (0.0, 1.0)
    for loop_idx, uv in enumerate(((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
        uv_layer.data[loop_idx].uv = uv


def _image_aspect_ratio(image: Optional[bpy.types.Image]) -> Optional[float]:
    if image is None:
        return None
    try:
        width = float(image.size[0])
        height = float(image.size[1])
    except Exception:
        return None
    if width <= 1.0 or height <= 1.0:
        return None
    return width / height


def _ensure_face_overlay_plane(
    parent: bpy.types.Object,
    child_suffix: str,
    material: bpy.types.Material,
    *,
    z_fraction: float,
    width_fraction: float,
    height_fraction: float,
    outward_fraction: float = 0.03,
    aspect_ratio: Optional[float] = None,
    flip_u: bool = False,
    flip_v: bool = False,
    roll_rad: float = 0.0,
    right_fraction: float = 0.0,
    up_fraction: float = 0.0,
) -> Optional[bpy.types.Object]:
    transform = _camera_facing_face_transform(
        parent,
        z_fraction=z_fraction,
        outward_fraction=outward_fraction,
        roll_rad=roll_rad,
        right_fraction=right_fraction,
        up_fraction=up_fraction,
    )
    if transform is None:
        return None

    location, rotation, face_width, face_height = transform
    width_cap = max(0.06, float(face_width) * float(width_fraction))
    height_cap = max(0.06, float(face_height) * float(height_fraction))
    if aspect_ratio is not None and aspect_ratio > 0.05:
        width = min(width_cap, height_cap * float(aspect_ratio))
        height = max(0.05, width / float(aspect_ratio))
    else:
        width = width_cap
        height = height_cap

    child = _ensure_overlay_plane(parent, child_suffix, material)
    _set_plane_uvs(child, flip_u=flip_u, flip_v=flip_v)
    child.rotation_mode = "XYZ"
    child.location = location
    child.rotation_euler = rotation
    child.scale = (width, height, 1.0)
    return child


def _ensure_face_overlay_disc(
    parent: bpy.types.Object,
    child_suffix: str,
    material: bpy.types.Material,
    *,
    z_fraction: float,
    diameter_fraction: float,
    outward_fraction: float = 0.03,
    right_fraction: float = 0.0,
    up_fraction: float = 0.0,
) -> Optional[bpy.types.Object]:
    transform = _camera_facing_face_transform(
        parent,
        z_fraction=z_fraction,
        outward_fraction=outward_fraction,
        right_fraction=right_fraction,
        up_fraction=up_fraction,
    )
    if transform is None:
        return None

    location, rotation, face_width, face_height = transform
    diameter = max(0.06, min(face_width, face_height) * float(diameter_fraction))
    child = _ensure_overlay_disc(parent, child_suffix, material)
    child.rotation_mode = "XYZ"
    child.location = location
    child.rotation_euler = rotation
    child.scale = (diameter, diameter, 1.0)
    return child


def _hide_named_child(parent: bpy.types.Object, child_suffix: str) -> None:
    child = bpy.data.objects.get(f"{parent.name}_{child_suffix}")
    if child is None:
        return
    child.hide_render = True
    child.hide_viewport = True


def _ensure_arrow_overlay(
    parent: bpy.types.Object,
    child_suffix: str,
    material: bpy.types.Material,
) -> bpy.types.Object:
    child_name = f"{parent.name}_{child_suffix}"
    child = bpy.data.objects.get(child_name)

    if child is None or child.type != "MESH":
        if child is not None:
            bpy.data.objects.remove(child, do_unlink=True)
        mesh = bpy.data.meshes.new(f"{child_name}_Mesh")
        verts = [
            (0.0, -0.10, 0.0),
            (0.76, -0.10, 0.0),
            (0.76, -0.22, 0.0),
            (1.0, 0.0, 0.0),
            (0.76, 0.22, 0.0),
            (0.76, 0.10, 0.0),
            (0.0, 0.10, 0.0),
        ]
        mesh.from_pydata(verts, [], [(0, 1, 2, 3, 4, 5, 6)])
        mesh.update()
        child = bpy.data.objects.new(child_name, mesh)
        child.hide_select = True
        target_collection = parent.users_collection[0] if parent.users_collection else bpy.context.scene.collection
        link_to_collection(child, target_collection)
        child.parent = parent
        child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    child.hide_render = False
    child.hide_viewport = False
    if child.data is not None and hasattr(child.data, "materials"):
        child.data.materials.clear()
        child.data.materials.append(material)
    return child


def _vehicle_overlay_anchor(parent: bpy.types.Object) -> Tuple[mathutils.Vector, mathutils.Vector]:
    min_corner, max_corner = local_bbox_min_max(parent)
    dims = max_corner - min_corner
    center = mathutils.Vector(
        (
            0.5 * (float(min_corner.x) + float(max_corner.x)),
            0.5 * (float(min_corner.y) + float(max_corner.y)),
            float(min_corner.z),
        )
    )
    return center, dims


def _ensure_vehicle_status_plate(
    parent: bpy.types.Object,
    status: str,
) -> bpy.types.Object:
    material = make_vehicle_motion_material(status)
    child = _ensure_overlay_plane(parent, "StatusPlate", material)
    center, dims = _vehicle_overlay_anchor(parent)
    child.rotation_mode = "XYZ"
    child.location = mathutils.Vector((center.x, center.y, center.z + 0.03))
    child.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), "XYZ")
    child.scale = (
        max(0.26, 0.48 * float(dims.x)),
        max(0.18, 0.42 * float(dims.y)),
        1.0,
    )
    return child


def _ensure_vehicle_motion_arrow(
    parent: bpy.types.Object,
    *,
    relative_yaw_rad: float,
    arrow_length_m: float,
) -> bpy.types.Object:
    child = _ensure_arrow_overlay(parent, "MotionArrow", make_vehicle_motion_material("moving"))
    center, dims = _vehicle_overlay_anchor(parent)
    child.rotation_mode = "XYZ"
    child.location = mathutils.Vector((center.x, center.y, center.z + 0.035))
    child.rotation_euler = mathutils.Euler((0.0, 0.0, float(relative_yaw_rad)), "XYZ")
    child.scale = (
        max(0.6, float(arrow_length_m)),
        max(0.18, 0.18 * float(dims.y)),
        1.0,
    )
    return child


def _ensure_vehicle_brake_light(
    parent: bpy.types.Object,
    child_suffix: str,
    lateral_sign: float,
    active: bool,
) -> bpy.types.Object:
    child = _ensure_overlay_plane(parent, child_suffix, make_brake_light_material(active))
    min_corner, max_corner = local_bbox_min_max(parent)
    dims = max_corner - min_corner
    length_axis = "x" if float(dims.x) >= float(dims.y) else "y"
    center_x = 0.5 * (float(min_corner.x) + float(max_corner.x))
    center_y = 0.5 * (float(min_corner.y) + float(max_corner.y))
    # Rear-light overlays belong on the rear face of the vehicle body, not on
    # whichever side currently faces the render camera.
    face_sign = 1.0
    inboard = max(0.006, 0.018 * max(float(dims.x), float(dims.y)))
    z_pos = float(min_corner.z) + 0.47 * float(dims.z)
    child.rotation_mode = "XYZ"
    if length_axis == "x":
        rear_x = float(min_corner.x) + inboard if face_sign < 0.0 else float(max_corner.x) - inboard
        horizontal = mathutils.Vector((0.0, face_sign, 0.0))
        vertical = mathutils.Vector((0.0, 0.0, 1.0))
        normal = mathutils.Vector((face_sign, 0.0, 0.0))
        child.location = mathutils.Vector((rear_x, center_y, z_pos)) + horizontal * (0.39 * float(dims.y) * float(lateral_sign))
        child.scale = (
            max(0.09, 0.095 * float(dims.y)),
            max(0.055, 0.072 * float(dims.z)),
            1.0,
        )
    else:
        rear_y = float(min_corner.y) + inboard if face_sign < 0.0 else float(max_corner.y) - inboard
        horizontal = mathutils.Vector((-face_sign, 0.0, 0.0))
        vertical = mathutils.Vector((0.0, 0.0, 1.0))
        normal = mathutils.Vector((0.0, face_sign, 0.0))
        child.location = mathutils.Vector((center_x, rear_y, z_pos)) + horizontal * (0.39 * float(dims.x) * float(lateral_sign))
        child.scale = (
            max(0.09, 0.095 * float(dims.x)),
            max(0.055, 0.072 * float(dims.z)),
            1.0,
        )
    rotation_matrix = mathutils.Matrix((horizontal, vertical, normal)).transposed()
    child.rotation_euler = rotation_matrix.to_euler("XYZ")
    return child


def _ensure_vehicle_indicator_light(
    parent: bpy.types.Object,
    child_suffix: str,
    lateral_sign: float,
    active: bool,
) -> bpy.types.Object:
    child = _ensure_overlay_plane(parent, child_suffix, make_indicator_light_material(active))
    min_corner, max_corner = local_bbox_min_max(parent)
    dims = max_corner - min_corner
    length_axis = "x" if float(dims.x) >= float(dims.y) else "y"
    center_x = 0.5 * (float(min_corner.x) + float(max_corner.x))
    center_y = 0.5 * (float(min_corner.y) + float(max_corner.y))
    face_sign = 1.0
    inboard = max(0.005, 0.014 * max(float(dims.x), float(dims.y)))
    z_pos = float(min_corner.z) + 0.49 * float(dims.z)
    child.rotation_mode = "XYZ"
    if length_axis == "x":
        rear_x = float(min_corner.x) + inboard if face_sign < 0.0 else float(max_corner.x) - inboard
        horizontal = mathutils.Vector((0.0, face_sign, 0.0))
        vertical = mathutils.Vector((0.0, 0.0, 1.0))
        normal = mathutils.Vector((face_sign, 0.0, 0.0))
        child.location = mathutils.Vector((rear_x, center_y, z_pos)) + horizontal * (0.46 * float(dims.y) * float(lateral_sign))
        child.scale = (
            max(0.075, 0.085 * float(dims.y)),
            max(0.045, 0.060 * float(dims.z)),
            1.0,
        )
    else:
        rear_y = float(min_corner.y) + inboard if face_sign < 0.0 else float(max_corner.y) - inboard
        horizontal = mathutils.Vector((-face_sign, 0.0, 0.0))
        vertical = mathutils.Vector((0.0, 0.0, 1.0))
        normal = mathutils.Vector((0.0, face_sign, 0.0))
        child.location = mathutils.Vector((center_x, rear_y, z_pos)) + horizontal * (0.46 * float(dims.x) * float(lateral_sign))
        child.scale = (
            max(0.075, 0.085 * float(dims.x)),
            max(0.045, 0.060 * float(dims.z)),
            1.0,
        )
    rotation_matrix = mathutils.Matrix((horizontal, vertical, normal)).transposed()
    child.rotation_euler = rotation_matrix.to_euler("XYZ")
    return child


def _append_overlay_face(
    mesh: bpy.types.Mesh,
    coords: Sequence[Tuple[float, float, float]],
    uvs: Sequence[Tuple[float, float]],
    material_index: int,
) -> None:
    try:
        import bmesh
    except ImportError:
        return

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
        verts = [bm.verts.new(coord) for coord in coords]
        bm.verts.ensure_lookup_table()
        face = bm.faces.new(verts)
        face.material_index = material_index
        for loop, uv in zip(face.loops, uvs):
            loop[uv_layer].uv = uv
        bmesh.ops.recalc_face_normals(bm, faces=[face])
        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()


def _face_vertical_bounds(
    obj: bpy.types.Object,
    min_corner: mathutils.Vector,
    dims: mathutils.Vector,
    *,
    default_bottom_frac: float,
    default_top_frac: float,
) -> Tuple[float, float]:
    face_bottom_frac = obj.get("_face_bottom_frac", None)
    face_top_frac = obj.get("_face_top_frac", None)
    if (
        face_bottom_frac is not None
        and face_top_frac is not None
        and 0.0 <= float(face_bottom_frac) < float(face_top_frac) <= 1.0
    ):
        bottom_frac = float(face_bottom_frac)
        top_frac = float(face_top_frac)
    else:
        bottom_frac = float(default_bottom_frac)
        top_frac = float(default_top_frac)
    face_bottom_z = float(min_corner.z) + bottom_frac * float(dims.z)
    face_top_z = float(min_corner.z) + top_frac * float(dims.z)
    if face_top_z <= face_bottom_z:
        face_bottom_z = float(min_corner.z) + float(default_bottom_frac) * float(dims.z)
        face_top_z = float(min_corner.z) + float(default_top_frac) * float(dims.z)
    return face_bottom_z, face_top_z


def add_stop_sign_face_mesh(obj: bpy.types.Object, material_index: int) -> None:
    mesh = obj.data
    if mesh is None or mesh.get("_stop_sign_face_ready"):
        return

    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    if float(dims.z) <= 0.05:
        return

    thin_axis = _horizontal_thin_axis(dims)
    center_y = 0.5 * (min_corner.y + max_corner.y)
    center_x = 0.5 * (min_corner.x + max_corner.x)
    face_bottom_z, face_top_z = _face_vertical_bounds(
        obj,
        min_corner,
        dims,
        default_bottom_frac=0.64,
        default_top_frac=0.96,
    )
    face_height = max(0.18, face_top_z - face_bottom_z)
    horizontal_span = float(max(dims.x, dims.y))
    sign_size = max(0.18, min(0.98 * horizontal_span, 0.98 * face_height))
    half_side = 0.5 * sign_size
    sign_center_z = face_bottom_z + 0.52 * face_height
    corner_inset = 0.41421356237 * half_side
    offset = 0.035 * max(float(dims.x), float(dims.y), 0.1)

    def octagon_2d() -> List[Tuple[float, float]]:
        a = half_side
        c = corner_inset
        return [
            (-a + c, a),
            (a - c, a),
            (a, a - c),
            (a, -a + c),
            (a - c, -a),
            (-a + c, -a),
            (-a, -a + c),
            (-a, a - c),
        ]

    verts_2d = octagon_2d()
    uvs = [((x / (2.0 * half_side)) + 0.5, (z / (2.0 * half_side)) + 0.5) for x, z in verts_2d]

    if thin_axis == "x":
        x_front = float(max_corner.x) + offset
        x_back = float(min_corner.x) - offset
        face_front = [(x_front, center_y + y_off, sign_center_z + z_off) for y_off, z_off in verts_2d]
        face_back = [(x_back, center_y + y_off, sign_center_z + z_off) for y_off, z_off in reversed(verts_2d)]
    else:
        y_front = float(max_corner.y) + offset
        y_back = float(min_corner.y) - offset
        face_front = [(center_x + x_off, y_front, sign_center_z + z_off) for x_off, z_off in verts_2d]
        face_back = [(center_x + x_off, y_back, sign_center_z + z_off) for x_off, z_off in reversed(verts_2d)]

    _append_overlay_face(mesh, face_front, uvs, material_index)
    _append_overlay_face(mesh, face_back, uvs, material_index)
    mesh["_stop_sign_face_material_index"] = int(material_index)
    mesh["_stop_sign_face_ready"] = True


def add_rect_sign_face_mesh(
    obj: bpy.types.Object,
    material_index: int,
    *,
    width_fraction: float = 0.68,
    height_fraction: float = 0.34,
    z_fraction: float = 0.66,
    outward_fraction: float = 0.035,
    mesh_flag: str = "_rect_sign_face_ready",
) -> None:
    mesh = obj.data
    if mesh is None or mesh.get(mesh_flag):
        return

    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    if float(dims.z) <= 0.05:
        return

    thin_axis = _horizontal_thin_axis(dims)
    center_y = 0.5 * (min_corner.y + max_corner.y)
    center_x = 0.5 * (min_corner.x + max_corner.x)
    face_bottom_z, face_top_z = _face_vertical_bounds(
        obj,
        min_corner,
        dims,
        default_bottom_frac=0.58,
        default_top_frac=0.95,
    )
    face_height = max(0.18, face_top_z - face_bottom_z)
    sign_width = max(0.18, float(width_fraction) * float(max(dims.x, dims.y)))
    sign_height = max(0.18, float(height_fraction) * face_height)
    half_w = 0.5 * sign_width
    half_h = 0.5 * sign_height
    sign_center_z = face_bottom_z + float(z_fraction) * face_height
    offset = float(outward_fraction) * max(float(dims.x), float(dims.y), 0.1)
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    if thin_axis == "x":
        x_front = float(max_corner.x) + offset
        x_back = float(min_corner.x) - offset
        face_front = [
            (x_front, center_y - half_w, sign_center_z - half_h),
            (x_front, center_y + half_w, sign_center_z - half_h),
            (x_front, center_y + half_w, sign_center_z + half_h),
            (x_front, center_y - half_w, sign_center_z + half_h),
        ]
        face_back = [
            (x_back, center_y - half_w, sign_center_z + half_h),
            (x_back, center_y + half_w, sign_center_z + half_h),
            (x_back, center_y + half_w, sign_center_z - half_h),
            (x_back, center_y - half_w, sign_center_z - half_h),
        ]
    else:
        y_front = float(max_corner.y) + offset
        y_back = float(min_corner.y) - offset
        face_front = [
            (center_x - half_w, y_front, sign_center_z - half_h),
            (center_x + half_w, y_front, sign_center_z - half_h),
            (center_x + half_w, y_front, sign_center_z + half_h),
            (center_x - half_w, y_front, sign_center_z + half_h),
        ]
        face_back = [
            (center_x - half_w, y_back, sign_center_z + half_h),
            (center_x + half_w, y_back, sign_center_z + half_h),
            (center_x + half_w, y_back, sign_center_z - half_h),
            (center_x - half_w, y_back, sign_center_z - half_h),
        ]

    _append_overlay_face(mesh, face_front, uvs, material_index)
    _append_overlay_face(mesh, face_back, uvs, material_index)
    mesh[mesh_flag] = True


def add_traffic_signal_lens_quads(obj: bpy.types.Object) -> None:
    mesh = obj.data
    if mesh is None or mesh.get("_traffic_signal_lenses_ready"):
        return

    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    if float(dims.z) <= 0.05:
        return

    thin_axis = _horizontal_thin_axis(dims)
    center_y = 0.5 * (min_corner.y + max_corner.y)
    center_x = 0.5 * (min_corner.x + max_corner.x)
    lens_width = max(0.06, 0.62 * float(max(dims.x, dims.y)))
    head_bottom_frac = float(obj.get("_tl_head_bottom_frac", 0.0))
    head_top_frac = float(obj.get("_tl_head_top_frac", 1.0))
    head_bottom_z = float(min_corner.z) + max(0.0, min(0.98, head_bottom_frac)) * float(dims.z)
    head_top_z = float(min_corner.z) + max(head_bottom_frac + 0.02, min(1.0, head_top_frac)) * float(dims.z)
    head_height = max(0.18, head_top_z - head_bottom_z)
    lens_height = max(0.05, 0.18 * head_height)
    half_w = 0.5 * lens_width
    half_h = 0.5 * lens_height
    offset = 0.025 * max(float(dims.x), float(dims.y), 0.1)
    z_centers = {
        "red": head_bottom_z + 0.77 * head_height,
        "yellow": head_bottom_z + 0.51 * head_height,
        "green": head_bottom_z + 0.25 * head_height,
    }
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    slot_map = {"red": 1, "yellow": 2, "green": 3}

    def quads_for_band(z_center: float) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        z0 = z_center - half_h
        z1 = z_center + half_h
        if thin_axis == "x":
            x_front = float(max_corner.x) + offset
            x_back = float(min_corner.x) - offset
            y0 = center_y - half_w
            y1 = center_y + half_w
            front = [(x_front, y0, z0), (x_front, y1, z0), (x_front, y1, z1), (x_front, y0, z1)]
            back = [(x_back, y1, z0), (x_back, y0, z0), (x_back, y0, z1), (x_back, y1, z1)]
        else:
            y_front = float(max_corner.y) + offset
            y_back = float(min_corner.y) - offset
            x0 = center_x - half_w
            x1 = center_x + half_w
            front = [(x0, y_front, z0), (x1, y_front, z0), (x1, y_front, z1), (x0, y_front, z1)]
            back = [(x1, y_back, z0), (x0, y_back, z0), (x0, y_back, z1), (x1, y_back, z1)]
        return front, back

    for color_name, z_center in z_centers.items():
        front, back = quads_for_band(z_center)
        mat_index = slot_map[color_name]
        _append_overlay_face(mesh, front, uvs, mat_index)
        _append_overlay_face(mesh, back, uvs, mat_index)

    mesh["_traffic_signal_lenses_ready"] = True


def apply_traffic_light_material(
    obj: bpy.types.Object,
    frame: int,
    tl_color: str,
    tl_color_conf: Optional[float] = None,
    signal_shape: Optional[str] = None,
    detic_color_check: Optional[str] = None,
    detic_color_conf: Optional[float] = None,
    detic_color_agrees: Optional[bool] = None,
    bbox: Optional[Sequence[Any]] = None,
    tl_centroid: Optional[Sequence[Any]] = None,
) -> None:
    if obj.data is None:
        return

    mesh = obj.data
    tl_key, render_conf, agreement_bonus = resolve_traffic_light_render_color(
        tl_color=tl_color,
        tl_color_conf=tl_color_conf,
        detic_color_check=detic_color_check,
        detic_color_conf=detic_color_conf,
        detic_color_agrees=detic_color_agrees,
    )
    show_all_lamps = tl_key == "unknown"
    red_mat = make_traffic_signal_lens_material(
        "red",
        active=(show_all_lamps or tl_key == "red"),
        confidence=render_conf,
        agreement_bonus=agreement_bonus if tl_key == "red" else 0.0,
    )
    yellow_mat = make_traffic_signal_lens_material(
        "yellow",
        active=(show_all_lamps or tl_key == "yellow"),
        confidence=render_conf,
        agreement_bonus=agreement_bonus if tl_key == "yellow" else 0.0,
    )
    green_mat = make_traffic_signal_lens_material(
        "green",
        active=(show_all_lamps or tl_key == "green"),
        confidence=render_conf,
        agreement_bonus=agreement_bonus if tl_key == "green" else 0.0,
    )

    housing_mat = make_traffic_signal_housing_material()
    if len(mesh.materials) == 0 or str(mesh.materials[0].name).lower() != "m_trafficsignalhousing":
        mesh.materials.clear()
        mesh.materials.append(housing_mat)
        for poly in mesh.polygons:
            poly.material_index = 0

    lens_children = (
        _ensure_face_overlay_disc(
            obj,
            "TLRedLens",
            red_mat,
            z_fraction=0.76,
            diameter_fraction=0.31,
            outward_fraction=0.064,
        ),
        _ensure_face_overlay_disc(
            obj,
            "TLYellowLens",
            yellow_mat,
            z_fraction=0.50,
            diameter_fraction=0.31,
            outward_fraction=0.064,
        ),
        _ensure_face_overlay_disc(
            obj,
            "TLGreenLens",
            green_mat,
            z_fraction=0.24,
            diameter_fraction=0.31,
            outward_fraction=0.064,
        ),
    )
    lens_names = ("red", "yellow", "green")
    for color_name, child in zip(lens_names, lens_children):
        if child is not None:
            set_visibility_keyframe(child, frame, show_all_lamps or tl_key == color_name)

    active_z_fraction = {"red": 0.76, "yellow": 0.50, "green": 0.24}.get(tl_key, 0.50)
    if (
        bbox is not None
        and tl_centroid is not None
        and len(bbox) >= 4
        and len(tl_centroid) >= 2
    ):
        try:
            y1 = float(bbox[1])
            y2 = float(bbox[3])
            cy = float(tl_centroid[1])
            if y2 > y1 + 1.0:
                centroid_frac = 1.0 - ((cy - y1) / (y2 - y1))
                centroid_frac = max(0.12, min(0.88, centroid_frac))
                active_z_fraction = 0.30 * active_z_fraction + 0.70 * centroid_frac
        except (TypeError, ValueError):
            pass

    active_child = _ensure_face_overlay_disc(
        obj,
        "TLActiveState",
        make_solid_material("M_TL_ActiveHidden", (0.1, 0.1, 0.1, 1.0), roughness=1.0, metallic=0.0, specular=0.0),
        z_fraction=active_z_fraction if not show_all_lamps else 0.50,
        diameter_fraction=0.24,
        outward_fraction=0.092,
    )
    if active_child is not None:
        set_visibility_keyframe(active_child, frame, False)

    if tl_key != "unknown":
        arrow_material = make_traffic_signal_material(
            f"M_TL_ArrowOverlay_{tl_key}_{_confidence_bucket(render_conf)}_{int(round((agreement_bonus + 0.10) * 10.0))}",
            tl_key,
            confidence=max(render_conf, 0.35),
            agreement_bonus=agreement_bonus,
        )
    else:
        arrow_material = make_solid_material(
            "M_TL_ArrowOverlay_unknown",
            (0.92, 0.92, 0.92, 1.0),
            roughness=0.35,
            metallic=0.0,
            specular=0.02,
        )
    left_child = _ensure_text_overlay(
        obj,
        "TLArrowLeft",
        "<",
        arrow_material,
        z_fraction=0.53,
        width_fraction=0.24,
        height_fraction=0.10,
        outward_fraction=0.068,
    )
    right_child = _ensure_text_overlay(
        obj,
        "TLArrowRight",
        ">",
        arrow_material,
        z_fraction=0.53,
        width_fraction=0.24,
        height_fraction=0.10,
        outward_fraction=0.068,
    )
    straight_child = _ensure_text_overlay(
        obj,
        "TLArrowStraight",
        "^",
        arrow_material,
        z_fraction=0.53,
        width_fraction=0.24,
        height_fraction=0.10,
        outward_fraction=0.068,
    )
    arrow_marker = traffic_signal_shape_marker(signal_shape)
    if left_child is not None:
        set_visibility_keyframe(left_child, frame, arrow_marker == "<")
    if right_child is not None:
        set_visibility_keyframe(right_child, frame, arrow_marker == ">")
    if straight_child is not None:
        set_visibility_keyframe(straight_child, frame, arrow_marker == "^")


def apply_stop_sign_material(obj: bpy.types.Object, assets_dir: Path) -> None:
    if obj.data is None:
        return
    mesh = obj.data
    if not bool(mesh.get("_stop_sign_styled", False)) and len(mesh.materials) == 0:
        mesh.materials.append(
            make_solid_material(
                "M_StopSignBoard",
                (0.78, 0.08, 0.08, 1.0),
                roughness=0.76,
                metallic=0.0,
                specular=0.02,
            )
        )
    mesh["_stop_sign_styled"] = True

    image = get_static_image(str(assets_dir / "StopSignImage.png"))
    if image is not None:
        try:
            if hasattr(image, "alpha_mode"):
                image.alpha_mode = "STRAIGHT"
        except Exception:
            pass
        _ensure_face_overlay_plane(
            obj,
            "StopFace",
            make_image_sign_material(
                "M_StopSignFace",
                image,
                base_color=(1.0, 1.0, 1.0, 1.0),
                use_alpha=True,
                backface_culling=True,
            ),
            z_fraction=0.52,
            width_fraction=1.3,
            height_fraction=1.3,
            outward_fraction=0.038,
            aspect_ratio=_image_aspect_ratio(image),
            up_fraction=0.065,
        )
        _hide_named_child(obj, "StopText")
        return

    _hide_named_child(obj, "StopFace")
    stop_text_material = make_solid_material(
        "M_StopSignText",
        (0.97, 0.97, 0.97, 1.0),
        roughness=0.78,
        metallic=0.0,
        specular=0.01,
    )
    _ensure_text_overlay(
        obj,
        "StopText",
        "STOP",
        stop_text_material,
        z_fraction=0.50,
        width_fraction=0.70,
        height_fraction=0.18,
        outward_fraction=0.040,
        roll_rad=0.0,
        right_fraction=0.0,
        up_fraction=0.03,
    )


def apply_procedural_speed_limit_face(obj: bpy.types.Object) -> None:
    if obj.data is None:
        return

    mesh = obj.data
    if not mesh.get("_speed_limit_outer_face_ready"):
        outer_index = len(mesh.materials)
        mesh.materials.append(
            make_solid_material(
                "M_SpeedLimitBorder",
                (0.04, 0.04, 0.04, 1.0),
                roughness=0.72,
                metallic=0.0,
                specular=0.01,
            )
        )
        add_rect_sign_face_mesh(
            obj,
            outer_index,
            width_fraction=0.96,
            height_fraction=0.96,
            z_fraction=0.50,
            outward_fraction=0.032,
            mesh_flag="_speed_limit_outer_face_ready",
        )

    if not mesh.get("_speed_limit_inner_face_ready"):
        inner_index = len(mesh.materials)
        mesh.materials.append(
            make_solid_material(
                "M_SpeedLimitInnerFace",
                (0.98, 0.98, 0.97, 1.0),
                roughness=0.78,
                metallic=0.0,
                specular=0.01,
            )
        )
        add_rect_sign_face_mesh(
            obj,
            inner_index,
            width_fraction=0.88,
            height_fraction=0.88,
            z_fraction=0.50,
            outward_fraction=0.040,
            mesh_flag="_speed_limit_inner_face_ready",
        )


def apply_speed_limit_material(
    obj: bpy.types.Object,
    speed_limit_value: Optional[Any],
    sign_label: Optional[Any] = None,
    assets_dir: Optional[Path] = None,
) -> None:
    if obj.data is None:
        return

    mesh = obj.data
    if not bool(mesh.get("_speed_limit_styled", False)) and len(mesh.materials) == 0:
        mesh.materials.append(
            make_solid_material(
                "M_SpeedLimitBoard",
                (0.92, 0.93, 0.95, 1.0),
                roughness=0.82,
                metallic=0.0,
                specular=0.02,
            )
        )
    mesh["_speed_limit_styled"] = True

    blank_face_loaded = False
    face_material: Optional[bpy.types.Material] = None
    image = None
    if assets_dir is not None:
        image = get_svg_backed_image(assets_dir / "Speed_Limit_blank_sign.svg")
        if image is not None:
            try:
                if hasattr(image, "alpha_mode"):
                    image.alpha_mode = "STRAIGHT"
            except Exception:
                pass
            face_material = make_image_sign_material(
                "M_SpeedLimitBlankFace",
                image,
                base_color=(1.0, 1.0, 1.0, 1.0),
                use_alpha=True,
                backface_culling=True,
            )
            blank_face_loaded = True
    if face_material is None:
        face_material = make_solid_material(
            "M_SpeedLimitBlankFallback",
            (0.98, 0.98, 0.97, 1.0),
            roughness=0.78,
            metallic=0.0,
            specular=0.01,
        )

    _ensure_face_overlay_plane(
        obj,
        "SpeedLimitFace",
        face_material,
        z_fraction=0.50,
        width_fraction=0.94,
        height_fraction=1.5,
        outward_fraction=0.034,
        aspect_ratio=_image_aspect_ratio(image) if blank_face_loaded else None,
        roll_rad=0.0,
        up_fraction=0.205,
    )

    label = ""
    if speed_limit_value is not None:
        try:
            label = str(int(speed_limit_value))
        except (TypeError, ValueError):
            label = ""
    if not label and sign_label is not None:
        digits = "".join(ch for ch in str(sign_label) if ch.isdigit())
        if 1 <= len(digits) <= 3:
            label = digits

    text_material = make_solid_material(
        "M_SpeedLimitDigits",
        (0.05, 0.05, 0.05, 1.0),
        roughness=0.78,
        metallic=0.0,
        specular=0.01,
    )
    _ensure_text_overlay(
        obj,
        "LimitText",
        label,
        text_material,
        z_fraction=0.50 if blank_face_loaded else 0.47,
        width_fraction=0.75 if blank_face_loaded else 0.50,
        height_fraction=0.35 if blank_face_loaded else 0.24,
        outward_fraction=0.042 if blank_face_loaded else 0.040,
        roll_rad=0.0,
        up_fraction=0.001 if blank_face_loaded else 0.065,
    )

    _ensure_text_overlay(
        obj,
        "LimitHeadingTop",
        "" if blank_face_loaded else "SPEED",
        text_material,
        z_fraction=0.79,
        width_fraction=0.72,
        height_fraction=0.075,
        outward_fraction=0.086,
        roll_rad=0.0,
        up_fraction=0.01,
    )
    _ensure_text_overlay(
        obj,
        "LimitHeadingBottom",
        "" if blank_face_loaded else "LIMIT",
        text_material,
        z_fraction=0.68,
        width_fraction=0.72,
        height_fraction=0.075,
        outward_fraction=0.086,
        roll_rad=0.0,
        up_fraction=0.01,
    )


def apply_vehicle_motion_overlay(
    obj: bpy.types.Object,
    *,
    frame: int,
    moving: bool,
    parked: bool,
    direction: Optional[str] = None,
    motion_vector_blender: Optional[Sequence[float]] = None,
    motion_speed_mps: Optional[float] = None,
) -> None:
    status = "moving" if moving and not parked else "parked" if parked else "unknown"
    status_child = _ensure_vehicle_status_plate(obj, status)
    set_visibility_keyframe(status_child, frame, False)

    motion_vec = list(motion_vector_blender or [])[:2]
    if len(motion_vec) < 2:
        motion_vec = [0.0, 0.0]
    motion_mag = math.hypot(float(motion_vec[0]), float(motion_vec[1]))
    speed_hint = max(float(motion_speed_mps or 0.0), motion_mag)
    arrow_visible = bool(moving and not parked and speed_hint >= 0.20)

    if arrow_visible:
        world_yaw = math.atan2(float(motion_vec[1]), float(motion_vec[0])) if motion_mag >= 1e-5 else float(obj.rotation_euler.z)
        relative_yaw = wrap_angle(world_yaw - float(obj.rotation_euler.z))
        arrow_length = max(0.8, min(4.5, 0.55 * max(speed_hint, 1.0)))
        arrow_child = _ensure_vehicle_motion_arrow(
            obj,
            relative_yaw_rad=relative_yaw,
            arrow_length_m=arrow_length,
        )
        set_visibility_keyframe(arrow_child, frame, True)
    else:
        arrow_child = bpy.data.objects.get(f"{obj.name}_MotionArrow")
        if arrow_child is not None:
            set_visibility_keyframe(arrow_child, frame, False)


def apply_vehicle_state_material(
    obj: bpy.types.Object,
    *,
    moving: bool,
    parked: bool,
    alert_state: str = "none",
) -> None:
    status = "moving" if moving and not parked else "parked" if parked else "unknown"
    assign_single_material_recursive(obj, make_vehicle_body_material(status, alert_state))


def apply_vehicle_wheel_overlays(obj: bpy.types.Object) -> None:
    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    length_axis = "x" if float(dims.x) >= float(dims.y) else "y"
    body_length = float(dims.x) if length_axis == "x" else float(dims.y)
    body_width = float(dims.y) if length_axis == "x" else float(dims.x)
    wheel_diameter = max(0.10, min(0.18 * float(dims.z), 0.085 * max(body_length, 0.1)))
    wheel_depth = max(0.035, min(0.075 * max(body_width, 0.1), 0.42 * wheel_diameter))
    z_center = float(min_corner.z) + 0.52 * wheel_diameter
    longitudinal_offsets = (
        -0.26 * body_length,
        0.26 * body_length,
    )
    side_center = max(0.5 * wheel_depth + 0.01, 0.44 * body_width)
    wheel_material = make_vehicle_wheel_material()

    for long_index, longitudinal in enumerate(longitudinal_offsets):
        for side_index, side_sign in enumerate((-1.0, 1.0)):
            child = _ensure_overlay_cylinder(
                obj,
                f"Wheel_{long_index}_{side_index}",
                wheel_material,
            )
            child.rotation_mode = "XYZ"
            if length_axis == "x":
                x_center = 0.5 * (float(min_corner.x) + float(max_corner.x)) + float(longitudinal)
                y_center = 0.5 * (float(min_corner.y) + float(max_corner.y)) + float(side_sign) * float(side_center)
                child.location = (x_center, y_center, z_center)
                child.rotation_euler = mathutils.Euler((math.pi / 2.0, 0.0, 0.0), "XYZ")
            else:
                x_center = 0.5 * (float(min_corner.x) + float(max_corner.x)) + float(side_sign) * float(side_center)
                y_center = 0.5 * (float(min_corner.y) + float(max_corner.y)) + float(longitudinal)
                child.location = (x_center, y_center, z_center)
                child.rotation_euler = mathutils.Euler((0.0, math.pi / 2.0, 0.0), "XYZ")
            child.scale = (wheel_diameter, wheel_diameter, wheel_depth)
            child.hide_render = False
            child.hide_viewport = False


def hide_vehicle_wheel_overlays(obj: bpy.types.Object) -> None:
    for child in list(obj.children):
        if "_wheel_" not in str(child.name).lower():
            continue
        child.hide_render = True
        child.hide_viewport = True


def apply_traffic_cone_details(obj: bpy.types.Object) -> None:
    stripe = _ensure_overlay_cylinder(obj, "ConeStripe", make_cone_stripe_material())
    min_corner, max_corner = local_bbox_min_max(obj)
    dims = max_corner - min_corner
    stripe_diameter = max(0.10, 0.46 * max(float(dims.x), float(dims.y)))
    stripe_height = max(0.04, 0.14 * float(dims.z))
    stripe.location = (
        0.5 * (float(min_corner.x) + float(max_corner.x)),
        0.5 * (float(min_corner.y) + float(max_corner.y)),
        float(min_corner.z) + 0.56 * float(dims.z),
    )
    stripe.rotation_mode = "XYZ"
    stripe.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), "XYZ")
    stripe.scale = (stripe_diameter, stripe_diameter, stripe_height)
    stripe.hide_render = False
    stripe.hide_viewport = False


def apply_vehicle_brake_lights(
    obj: bpy.types.Object,
    *,
    frame: int,
    brake_lights_on: bool,
) -> None:
    left_child = _ensure_vehicle_brake_light(obj, "BrakeLeft", -1.0, brake_lights_on)
    right_child = _ensure_vehicle_brake_light(obj, "BrakeRight", 1.0, brake_lights_on)
    set_visibility_keyframe(left_child, frame, bool(brake_lights_on))
    set_visibility_keyframe(right_child, frame, bool(brake_lights_on))


def apply_vehicle_indicator_lights(
    obj: bpy.types.Object,
    *,
    frame: int,
    turn_signal: Optional[str],
    blink_period_frames: int = 10,
) -> None:
    signal = str(turn_signal or "").strip().lower()
    blink_on = bool(signal) and ((int(frame) // max(1, int(blink_period_frames // 2))) % 2 == 0)
    left_active = blink_on and signal in {"left", "hazard"}
    right_active = blink_on and signal in {"right", "hazard"}
    left_child = _ensure_vehicle_indicator_light(obj, "IndicatorLeft", -1.0, left_active)
    right_child = _ensure_vehicle_indicator_light(obj, "IndicatorRight", 1.0, right_active)
    set_visibility_keyframe(left_child, frame, left_active)
    set_visibility_keyframe(right_child, frame, right_active)


def apply_generic_traffic_sign_material(obj: bpy.types.Object) -> None:
    if obj.data is None:
        return
    mesh = obj.data
    if len(mesh.materials) == 0:
        mesh.materials.append(
            make_solid_material(
                "M_TrafficSignFallback",
                (0.95, 0.95, 0.94, 1.0),
                roughness=0.76,
                metallic=0.0,
                specular=0.02,
            )
        )


# ============================================================================
# Geometry builders
# ============================================================================

def create_lane_curve(
    points_3d: Sequence[Sequence[float]],
    lane_color: str,
    lane_type: str,
    avg_hsv: Optional[Sequence[float]],
    avg_ycrcb: Optional[Sequence[float]],
    color_confidence: Optional[float],
    lane_id: int,
    frame_idx: int,
    collection: bpy.types.Collection,
    world_offset: Optional[mathutils.Vector] = None,
) -> Optional[bpy.types.Object]:
    filtered = clean_ground_points(points_3d, smooth_lateral=True)
    if len(filtered) < 2:
        return None
    scene_offset = world_offset if world_offset is not None else mathutils.Vector((0.0, 0.0, 0.0))

    curve = bpy.data.curves.new(f"LaneCurve_{frame_idx:05d}_{lane_id}", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.035
    curve.bevel_resolution = 2
    curve.use_fill_caps = True

    spline = curve.splines.new("POLY")
    spline.points.add(len(filtered) - 1)
    for idx, point in enumerate(filtered):
        spline.points[idx].co = (
            point[0] + float(scene_offset.x),
            point[1] + float(scene_offset.y),
            point[2] + 0.02 + float(scene_offset.z),
            1.0,
        )

    obj = bpy.data.objects.new(f"Lane_{frame_idx:05d}_{lane_id}", curve)
    link_to_collection(obj, collection)

    mat_color = resolve_lane_paint_rgba(
        lane_color=lane_color,
        lane_type=lane_type,
        avg_hsv=avg_hsv,
        avg_ycrcb=avg_ycrcb,
        color_confidence=color_confidence,
    )
    lane_conf = clamp01(float(color_confidence or 0.0))
    material = make_solid_material(
        f"M_Lane_{lane_color}_{lane_type}_{_confidence_bucket(color_confidence)}_{int(round(mat_color[0] * 100))}_{int(round(mat_color[1] * 100))}_{int(round(mat_color[2] * 100))}",
        mat_color,
        roughness=0.90 - 0.10 * lane_conf,
        metallic=0.0,
        specular=0.010 + 0.012 * lane_conf,
    )
    obj.data.materials.clear()
    obj.data.materials.append(material)
    return obj


def create_road_surface(
    contours_3d: Sequence[Sequence[Sequence[float]]],
    frame_idx: int,
    collection: bpy.types.Collection,
    world_offset: Optional[mathutils.Vector] = None,
) -> Optional[bpy.types.Object]:
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, ...]] = []
    offset = 0
    scene_offset = world_offset if world_offset is not None else mathutils.Vector((0.0, 0.0, 0.0))

    for contour in contours_3d:
        if len(contour) < 3:
            continue
        pts = clean_ground_points(contour, smooth_lateral=False)
        if len(pts) >= 2 and math.dist(pts[0], pts[-1]) < 0.1:
            pts = pts[:-1]
        if len(pts) < 3:
            continue
        local_faces = triangulate_xy_polygon(pts)
        if not local_faces:
            continue

        shifted_pts = [
            (
                pt[0] + float(scene_offset.x),
                pt[1] + float(scene_offset.y),
                pt[2] + 0.006 + float(scene_offset.z),
            )
            for pt in pts
        ]
        verts.extend(shifted_pts)
        for face in local_faces:
            faces.append(tuple(offset + int(idx) for idx in face))
        offset += len(shifted_pts)

    if not verts or not faces:
        return None

    mesh = bpy.data.meshes.new(f"RoadMesh_{frame_idx:05d}")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(f"Road_{frame_idx:05d}", mesh)
    link_to_collection(obj, collection)
    mesh.materials.clear()
    mesh.materials.append(make_road_material())
    return obj


def create_road_marking_surface(
    contour_3d: Sequence[Sequence[float]],
    color: str,
    marking_type: str,
    marking_id: int,
    frame_idx: int,
    collection: bpy.types.Collection,
    world_offset: Optional[mathutils.Vector] = None,
) -> Optional[bpy.types.Object]:
    pts = clean_ground_points(contour_3d, smooth_lateral=False)
    if marking_type == "crosswalk":
        pts = convex_hull_xy(pts)
    if len(pts) < 3:
        return None
    if len(pts) >= 2 and math.dist(pts[0], pts[-1]) < 0.08:
        pts = pts[:-1]
    if len(pts) < 3:
        return None

    z_offset = 0.018 if marking_type == "arrow" else (0.012 if marking_type == "crosswalk" else 0.014)
    scene_offset = world_offset if world_offset is not None else mathutils.Vector((0.0, 0.0, 0.0))
    verts = [
        (
            pt[0] + float(scene_offset.x),
            pt[1] + float(scene_offset.y),
            pt[2] + z_offset + float(scene_offset.z),
        )
        for pt in pts
    ]
    face = tuple(range(len(verts)))

    mesh = bpy.data.meshes.new(f"RoadMarkingMesh_{frame_idx:05d}_{marking_id}")
    mesh.from_pydata(verts, [], [face])
    mesh.update()

    obj = bpy.data.objects.new(f"RoadMarking_{frame_idx:05d}_{marking_id}", mesh)
    link_to_collection(obj, collection)
    mesh.materials.clear()
    mesh.materials.append(make_road_marking_material(color, marking_type))
    return obj




def create_ego_vehicle_proxy(
    meta: Dict[str, Any],
    collection: bpy.types.Collection,
    assets_dir: Optional[Path] = None,
) -> Optional[bpy.types.Object]:
    """
    Place the host (ego) vehicle in the scene using a real .blend asset
    from ``P3Data/Assets`` whenever possible, falling back to a coarse
    procedural cube only if the asset cannot be loaded.

    The ego vehicle is anchored at the ego-base origin (0, 0, 0) and oriented
    so its forward axis matches the camera's forward axis (+X).  All
    detections, lanes and road geometry are already shifted into this
    ego-base frame by ``scene_world_offset_blender``, so the host car becomes
    the natural reference for the rest of the rendered scene.
    """

    ego_meta = meta.get("ego_vehicle", {}) or {}
    dims_meta = ego_meta.get("dims_m", {}) or {}
    length = float(dims_meta.get("length", 4.65))
    width = float(dims_meta.get("width", 1.85))
    height = float(dims_meta.get("height", 1.52))
    view = str(meta.get("view", ego_meta.get("view", "front"))).strip().lower()
    style = str(ego_meta.get("proxy_style", "hood")).strip().lower()
    asset_rel = str(ego_meta.get("asset", "Vehicles/SedanAndHatchback.blend")).strip()
    camera_mount = _vector3(
        ego_meta.get("camera_mount_blender"),
        (1.20, 0.0, float(meta.get("calib", {}).get("camera_height_m", 1.5))),
    )

    obj_name = "EgoVehicle"

    # ----------------------------------------------------------------
    # Preferred path: load the real Sedan/Hatchback asset from P3Data
    # ----------------------------------------------------------------
    if assets_dir is not None and asset_rel:
        ego_obj = instantiate_asset(
            asset_rel=asset_rel,
            assets_dir=assets_dir,
            collection=collection,
            instance_name=obj_name,
            unique_mesh=False,
        )
        if ego_obj is not None:
            base_length = max(float(ego_obj.get("_template_length", 1.0)), 0.01)
            base_width = max(float(ego_obj.get("_template_width", 1.0)), 0.01)
            base_height = max(float(ego_obj.get("_template_height", 1.0)), 0.01)
            scale_x = max(0.05, length / base_length)
            scale_y = max(0.05, width / base_width)
            scale_z = max(0.05, height / base_height)
            uniform = (scale_x + scale_y + scale_z) / 3.0
            blend = 0.4
            scale_x = (1.0 - blend) * uniform + blend * scale_x
            scale_y = (1.0 - blend) * uniform + blend * scale_y
            scale_z = (1.0 - blend) * uniform + blend * scale_z
            ego_obj.scale = (scale_x, scale_y, scale_z)
            ego_obj.location = (0.0, 0.0, 0.0)
            yaw_offset = float(ego_obj.get("_template_yaw_offset", 0.0))
            ego_obj.rotation_mode = "XYZ"
            ego_obj.rotation_euler = mathutils.Euler((0.0, 0.0, yaw_offset), "XYZ")
            ego_obj.hide_render = False
            ego_obj.hide_viewport = False
            ego_obj["is_ego_vehicle"] = True
            ego_obj["proxy_style"] = "asset"
            ego_obj["ego_asset"] = asset_rel
            # Paint the ego vehicle a distinctive Tesla-style blue so it is
            # immediately recognisable as the host car in the chase view.
            ego_paint = make_solid_material(
                "M_EgoVehicle_Blue",
                (0.06, 0.18, 0.62, 1.0),
                roughness=0.32,
                metallic=0.55,
                specular=0.55,
            )
            if ego_obj.data is not None and hasattr(ego_obj.data, "materials"):
                ego_obj.data.materials.clear()
                ego_obj.data.materials.append(ego_paint)
            print(f"[blender] Ego vehicle instanced from asset: {asset_rel}")
            return ego_obj
        print(f"[blender] WARNING — could not load ego asset {asset_rel}, using cube proxy.")

    # ----------------------------------------------------------------
    # Fallback: procedural cube proxy (legacy behaviour, last resort).
    # ----------------------------------------------------------------
    half_length = 0.5 * length
    half_width = 0.5 * width

    verts: List[Tuple[float, float, float]]
    faces: List[Tuple[int, ...]]

    if style == "rear_deck":
        rear = -half_length
        trunk_front = min(camera_mount.x - 0.20, rear + 1.10)
        z_low = max(0.30, 0.24 * height)
        z_top = max(0.72, min(camera_mount.z - 0.18, 0.78 * height))
        verts = [
            (trunk_front, -0.46 * half_width, z_low),
            (trunk_front, 0.46 * half_width, z_low),
            (rear, 0.50 * half_width, z_low + 0.05),
            (rear, -0.50 * half_width, z_low + 0.05),
            (trunk_front + 0.08, -0.38 * half_width, z_top),
            (trunk_front + 0.08, 0.38 * half_width, z_top),
            (rear + 0.18, 0.42 * half_width, z_top - 0.14),
            (rear + 0.18, -0.42 * half_width, z_top - 0.14),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
    elif style in {"left_side", "right_side"}:
        sign = 1.0 if style == "left_side" or view == "left" else -1.0
        side_outer = sign * half_width
        side_inner = sign * max(half_width - 0.18, 0.55)
        x_front = min(camera_mount.x + 1.35, half_length)
        x_rear = max(camera_mount.x - 1.10, -half_length + 0.25)
        z_low = max(0.28, 0.20 * height)
        z_high = max(0.78, min(camera_mount.z - 0.15, 0.84 * height))
        verts = [
            (x_rear, side_inner, z_low),
            (x_front, side_inner, z_low),
            (x_front, side_outer, z_low + 0.02),
            (x_rear, side_outer, z_low + 0.02),
            (x_rear + 0.12, side_inner, z_high),
            (x_front - 0.18, side_inner, z_high),
            (x_front - 0.12, side_outer, z_high - 0.06),
            (x_rear + 0.08, side_outer, z_high - 0.08),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
    else:
        hood_rear = min(camera_mount.x + 0.55, half_length - 0.30)
        hood_front = half_length
        z_low = max(0.28, 0.18 * height)
        z_top_rear = max(0.86, min(camera_mount.z - 0.20, 0.84 * height))
        z_top_front = max(0.62, 0.52 * height)
        verts = [
            (hood_rear - 0.08, -0.50 * half_width, z_low),
            (hood_rear - 0.08, 0.50 * half_width, z_low),
            (hood_front, 0.54 * half_width, z_low + 0.05),
            (hood_front, -0.54 * half_width, z_low + 0.05),
            (hood_rear, -0.40 * half_width, z_top_rear),
            (hood_rear, 0.40 * half_width, z_top_rear),
            (hood_front - 0.08, 0.46 * half_width, z_top_front),
            (hood_front - 0.08, -0.46 * half_width, z_top_front),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]

    mesh = bpy.data.meshes.new(f"{obj_name}Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(obj_name, mesh)
    link_to_collection(obj, collection)
    mesh.materials.clear()
    mesh.materials.append(
        make_solid_material(
            "M_EgoVehicle",
            (0.92, 0.92, 0.94, 1.0),
            roughness=0.55,
            metallic=0.08,
            specular=0.22,
        )
    )
    obj["is_ego_vehicle"] = True
    obj["proxy_style"] = style
    return obj


def expand_bbox(bbox: Sequence[int], margin: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width - 1, x2 + margin),
        min(height - 1, y2 + margin),
    )


def point_in_boxes(u: int, v: int, boxes: Sequence[Tuple[int, int, int, int]]) -> bool:
    for x1, y1, x2, y2 in boxes:
        if x1 <= u <= x2 and y1 <= v <= y2:
            return True
    return False


def point_in_polygon_2d(u: float, v: float, polygon: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    x_prev = float(polygon[-1][0])
    y_prev = float(polygon[-1][1])
    for point in polygon:
        x_cur = float(point[0])
        y_cur = float(point[1])
        denom = (y_prev - y_cur)
        if abs(denom) < 1e-9:
            x_prev, y_prev = x_cur, y_cur
            continue
        intersects = ((y_cur > v) != (y_prev > v)) and (
            u < (x_prev - x_cur) * (v - y_cur) / denom + x_cur
        )
        if intersects:
            inside = not inside
        x_prev, y_prev = x_cur, y_cur
    return inside


def point_in_polygons(u: int, v: int, polygons: Sequence[Sequence[Sequence[float]]]) -> bool:
    for polygon in polygons:
        if point_in_polygon_2d(float(u), float(v), polygon):
            return True
    return False


def sanitize_depth_shell_islands(obj: bpy.types.Object) -> None:
    """
    Remove tiny, low, or sliver-like connected components from a depth shell.

    The depth shell is only used as conservative background geometry.  Any
    small disconnected component near the ego vehicle is therefore more likely
    to be an artifact than a useful reconstructed surface.
    """
    if obj.data is None:
        return

    try:
        import bmesh
    except ImportError:
        return

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        if len(bm.faces) < 8:
            return

        remaining = set(bm.faces)
        components: List[Dict[str, Any]] = []

        while remaining:
            seed = remaining.pop()
            stack = [seed]
            faces = [seed]
            verts = set(seed.verts)

            while stack:
                face = stack.pop()
                for edge in face.edges:
                    for linked in edge.link_faces:
                        if linked in remaining:
                            remaining.remove(linked)
                            stack.append(linked)
                            faces.append(linked)
                            verts.update(linked.verts)

            min_corner, max_corner = component_min_max(list(verts))
            dims = max_corner - min_corner
            components.append(
                {
                    "faces": faces,
                    "verts": list(verts),
                    "min": min_corner,
                    "max": max_corner,
                    "dims": dims,
                    "center": 0.5 * (min_corner + max_corner),
                }
            )

        if len(components) <= 1:
            return

        to_delete = []
        removed = 0
        for comp in components:
            dims = comp["dims"]
            center = comp["center"]
            min_corner = comp["min"]
            max_corner = comp["max"]
            max_dim = max(float(dims.x), float(dims.y), float(dims.z))
            min_dim = max(min(float(dims.x), float(dims.y), float(dims.z)), 0.01)
            sliver_ratio = max_dim / min_dim
            face_count = len(comp["faces"])
            low_component = float(max_corner.z) < 1.05
            near_camera = float(center.x) < 18.0
            too_small = face_count < 18 or max_dim < 0.90
            thin_sliver = sliver_ratio > 16.0 and face_count < 220
            low_near_sliver = near_camera and low_component and (thin_sliver or max(float(dims.y), float(dims.z)) < 1.05)
            far_floating = float(center.x) > 42.0 and max(float(dims.y), float(dims.z)) < 2.6
            elevated_fragment = float(min_corner.z) > 1.8 and face_count < 420
            very_elevated_small = float(min_corner.z) > 2.6 and face_count < 900
            disconnected_midair = float(min_corner.z) > 1.45 and float(center.x) > 14.0 and face_count < 260

            if too_small or low_near_sliver or far_floating or elevated_fragment or very_elevated_small or disconnected_midair:
                to_delete.extend(comp["faces"])
                removed += 1

        if to_delete:
            bmesh.ops.delete(bm, geom=to_delete, context="FACES")
            bm.to_mesh(obj.data)
            obj.data.update()
            print(f"[blender] Sanitized {obj.name}: removed {removed} depth-shell component(s)")
    finally:
        bm.free()


def collect_ground_depths(frame_meta: Dict[str, Any]) -> List[float]:
    ground_depths: List[float] = []

    for contour in frame_meta.get("road", {}).get("contours_3d", []):
        for point in clean_ground_points(contour, smooth_lateral=False):
            ground_depths.append(float(point[0]))

    for marking in frame_meta.get("road", {}).get("markings", []):
        for point in clean_ground_points(marking.get("contour_3d", []), smooth_lateral=False):
            ground_depths.append(float(point[0]))

    for lane in frame_meta.get("lanes", []):
        for point in clean_ground_points(lane.get("points_3d", []), smooth_lateral=False):
            ground_depths.append(float(point[0]))

    return sorted(depth for depth in ground_depths if math.isfinite(depth) and depth > 0.5)


def infer_depth_shell_top_cut_px(
    frame_meta: Dict[str, Any],
    frame_h: int,
    fallback_top_px: int,
) -> int:
    road_polygons = list(frame_meta.get("road", {}).get("contours_img", []))
    road_ys: List[float] = []
    for polygon in road_polygons:
        for point in polygon:
            if len(point) >= 2:
                try:
                    road_ys.append(float(point[1]))
                except (TypeError, ValueError):
                    continue

    if not road_ys:
        return max(int(fallback_top_px), int(round(frame_h * 0.46)))

    road_top = min(road_ys)
    adaptive = int(round(max(float(fallback_top_px), road_top - 24.0, frame_h * 0.40)))
    return max(0, min(frame_h - 2, adaptive))


def infer_depth_shell_max_distance(
    frame_meta: Dict[str, Any],
    fallback_max_depth_m: float,
) -> float:
    ground_depths = collect_ground_depths(frame_meta)
    if len(ground_depths) < 8:
        return max(18.0, min(float(fallback_max_depth_m), 42.0))

    quantile_idx = min(len(ground_depths) - 1, max(0, int(round(0.90 * (len(ground_depths) - 1)))))
    road_far = float(ground_depths[quantile_idx])
    return max(18.0, min(float(fallback_max_depth_m), road_far + 10.0, 42.0))


def infer_depth_shell_foreground_distance(
    frame_meta: Dict[str, Any],
    fallback_depth_m: float,
) -> float:
    ground_depths = collect_ground_depths(frame_meta)
    if not ground_depths:
        return max(float(fallback_depth_m), 3.8)

    nearest = float(ground_depths[0])
    return max(float(fallback_depth_m), 3.8, min(6.0, nearest + 0.35))


def infer_ground_shell_clip_distance(
    frame_meta: Dict[str, Any],
    fallback_depth_m: float,
) -> float:
    ground_depths = collect_ground_depths(frame_meta)
    if not ground_depths:
        return max(float(fallback_depth_m) + 1.5, 6.0)

    nearest = float(ground_depths[0])
    return max(float(fallback_depth_m) + 1.2, 6.0, min(8.0, nearest + 1.75))


def depth_shell_row_floor(
    v: int,
    frame_h: int,
    base_depth_m: float,
    foreground_row_start: float,
    foreground_bottom_boost_m: float,
) -> float:
    depth_floor = float(base_depth_m)
    start = max(0.0, min(0.98, float(foreground_row_start)))
    if frame_h <= 1:
        return depth_floor

    v_norm = float(v) / float(frame_h - 1)
    if v_norm <= start:
        return depth_floor

    span = max(1e-6, 1.0 - start)
    t = min(1.0, max(0.0, (v_norm - start) / span))
    return depth_floor + float(foreground_bottom_boost_m) * (t ** 1.35)


def create_depth_shell(
    frame_idx: int,
    depth_map: "np.ndarray",
    frame_meta: Dict[str, Any],
    calib_meta: Dict[str, Any],
    movie_image: Optional[bpy.types.Image],
    use_source_textures: bool,
    collection: bpy.types.Collection,
    stride: int,
    crop_top_frac: float,
    crop_bottom_frac: float,
    min_depth_m: float,
    foreground_depth_m: float,
    foreground_row_start: float,
    foreground_bottom_boost_m: float,
    bbox_margin: int,
    max_depth_m: float,
    world_offset: Optional[mathutils.Vector] = None,
) -> Optional[bpy.types.Object]:
    if np is None:
        return None

    frame_h, frame_w = depth_map.shape[:2]
    top = int(max(0, min(frame_h - 2, round(frame_h * crop_top_frac))))
    top = infer_depth_shell_top_cut_px(frame_meta, frame_h, top)
    bottom = int(min(frame_h - 1, max(top + 2, round(frame_h * (1.0 - crop_bottom_frac)))))

    mask_boxes = [
        expand_bbox(det.get("bbox", [0, 0, 0, 0]), bbox_margin, frame_w, frame_h)
        for det in (frame_meta.get("objects", []) + frame_meta.get("traffic_lights", []))
        if det.get("bbox") is not None
    ]
    road_polygons = list(frame_meta.get("road", {}).get("contours_img", []))

    cols = list(range(0, frame_w, stride))
    rows = list(range(top, bottom, stride))
    if cols[-1] != frame_w - 1:
        cols.append(frame_w - 1)
    if not rows:
        return None
    if rows[-1] != bottom - 1:
        rows.append(bottom - 1)

    near_field_depth_floor = infer_depth_shell_foreground_distance(frame_meta, foreground_depth_m)
    ground_shell_clip_distance = infer_ground_shell_clip_distance(frame_meta, near_field_depth_floor)
    effective_max_depth_m = infer_depth_shell_max_distance(frame_meta, max_depth_m)
    lower_band_start = int(frame_h * 0.52)
    mid_band_start = int(frame_h * 0.40)

    verts: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    vert_ids = [[-1 for _ in cols] for _ in rows]
    sampled_depths = [[0.0 for _ in cols] for _ in rows]
    scene_offset = world_offset if world_offset is not None else mathutils.Vector((0.0, 0.0, 0.0))

    for row_idx, v in enumerate(rows):
        for col_idx, u in enumerate(cols):
            if point_in_boxes(u, v, mask_boxes):
                continue
            if road_polygons and point_in_polygons(u, v, road_polygons):
                continue
            depth_m = float(depth_map[v, u])
            depth_floor = depth_shell_row_floor(
                v=v,
                frame_h=frame_h,
                base_depth_m=max(float(min_depth_m), float(near_field_depth_floor)),
                foreground_row_start=foreground_row_start,
                foreground_bottom_boost_m=foreground_bottom_boost_m,
            )
            if not math.isfinite(depth_m) or depth_m < depth_floor or depth_m > effective_max_depth_m:
                continue
            neighborhood = depth_map[max(0, v - stride): min(frame_h, v + stride + 1), max(0, u - stride): min(frame_w, u + stride + 1)]
            valid_patch = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0.1) & (neighborhood < effective_max_depth_m + 8.0)]
            if valid_patch.size < 8:
                continue
            patch_p10 = float(np.percentile(valid_patch, 10.0))
            patch_p50 = float(np.percentile(valid_patch, 50.0))
            patch_p90 = float(np.percentile(valid_patch, 90.0))
            if abs(depth_m - patch_p50) > max(1.6, 0.12 * depth_m):
                continue
            if (patch_p90 - patch_p10) > max(2.2, 0.20 * depth_m):
                continue
            bx, by, bz = pixel_to_blender(float(u), float(v), depth_m, calib_meta)
            if bx < max(float(min_depth_m), 6.0):
                continue
            lateral_limit = min(MAX_GROUND_LATERAL_M, max(7.5, 0.42 * bx + 4.0))
            if abs(by) > lateral_limit:
                continue
            if bz < 0.0 or bz > max(18.0, 0.55 * bx + 8.0):
                continue
            if v >= lower_band_start and (bz < 1.20 or bx < (ground_shell_clip_distance + 3.5)):
                continue
            if v >= mid_band_start and bz < 0.85 and bx < max(15.0, ground_shell_clip_distance + 5.5):
                continue
            if bz < 0.55 and bx < max(18.0, ground_shell_clip_distance + 7.0):
                continue
            if v < int(frame_h * 0.58) and bz > max(4.5, 0.14 * bx + 2.8):
                continue
            vert_ids[row_idx][col_idx] = len(verts)
            verts.append((bx + float(scene_offset.x), by + float(scene_offset.y), bz + float(scene_offset.z)))
            sampled_depths[row_idx][col_idx] = depth_m
            uvs.append((u / max(frame_w - 1, 1), 1.0 - (v / max(frame_h - 1, 1))))

    faces: List[Tuple[int, int, int, int]] = []
    for row_idx in range(len(rows) - 1):
        for col_idx in range(len(cols) - 1):
            corners = (
                vert_ids[row_idx][col_idx],
                vert_ids[row_idx][col_idx + 1],
                vert_ids[row_idx + 1][col_idx + 1],
                vert_ids[row_idx + 1][col_idx],
            )
            if any(idx < 0 for idx in corners):
                continue

            depths = (
                sampled_depths[row_idx][col_idx],
                sampled_depths[row_idx][col_idx + 1],
                sampled_depths[row_idx + 1][col_idx + 1],
                sampled_depths[row_idx + 1][col_idx],
            )
            if max(depths) - min(depths) > max(1.1, 0.18 * statistics.median(depths)):
                continue
            pts = [mathutils.Vector(verts[idx]) for idx in corners]
            edge_lengths = [
                (pts[1] - pts[0]).length,
                (pts[2] - pts[1]).length,
                (pts[3] - pts[2]).length,
                (pts[0] - pts[3]).length,
                (pts[2] - pts[0]).length,
                (pts[3] - pts[1]).length,
            ]
            if max(edge_lengths) > max(5.0, 0.16 * statistics.median(depths)):
                continue
            tri_area_a = 0.5 * ((pts[1] - pts[0]).cross(pts[3] - pts[0])).length
            tri_area_b = 0.5 * ((pts[2] - pts[1]).cross(pts[3] - pts[1])).length
            if tri_area_a < 0.015 or tri_area_b < 0.015:
                continue
            faces.append(corners)

    if len(verts) < 4 or not faces:
        return None

    mesh = bpy.data.meshes.new(f"DepthShellMesh_{frame_idx:05d}")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vert_idx = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uvs[vert_idx]

    obj = bpy.data.objects.new(f"DepthShell_{frame_idx:05d}", mesh)
    link_to_collection(obj, collection)
    sanitize_depth_shell_islands(obj)
    if obj.data is None or len(obj.data.polygons) == 0:
        bpy.data.objects.remove(obj, do_unlink=True)
        return None
    mesh.materials.clear()
    if use_source_textures and movie_image is not None:
        mesh.materials.append(make_movie_material("M_DepthShell", movie_image, emission=False))
    else:
        mesh.materials.append(make_depth_shell_material())
    return obj


def camera_billboard_axes() -> Tuple[mathutils.Vector, mathutils.Vector]:
    camera = getattr(bpy.context.scene, "camera", None)
    right = mathutils.Vector((0.0, -1.0, 0.0))
    if camera is not None:
        right = camera.matrix_world.to_quaternion() @ mathutils.Vector((1.0, 0.0, 0.0))
    right.z = 0.0
    if right.length < 1e-6:
        right = mathutils.Vector((0.0, -1.0, 0.0))
    else:
        right.normalize()
    up = mathutils.Vector((0.0, 0.0, 1.0))
    return right, up


def create_pedestrian_pose_rig(
    instance_name: str,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    root = bpy.data.objects.new(instance_name, None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.10
    root.hide_render = True
    root.hide_viewport = True
    root["_pose_rig"] = True
    root["_template_height"] = 1.0
    root["_template_width"] = 1.0
    root["_template_length"] = 1.0
    root["_template_yaw_offset"] = 0.0
    link_to_collection(root, collection)

    joint_material = make_emission_material(
        "M_PedPoseJoint",
        (0.96, 0.98, 1.0, 1.0),
        strength=2.8,
        roughness=0.28,
    )
    bone_material = make_emission_material(
        "M_PedPoseBone",
        (0.90, 0.95, 1.0, 1.0),
        strength=1.9,
        roughness=0.30,
    )

    for idx in range(len(PED_KEYPOINT_NAMES)):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12,
            ring_count=8,
            radius=POSE_JOINT_RADIUS_M,
            location=(0.0, 0.0, 0.0),
        )
        joint = bpy.context.active_object
        joint.name = f"{instance_name}__joint_{idx:02d}"
        joint.parent = root
        joint.matrix_parent_inverse = root.matrix_world.inverted()
        joint.hide_render = True
        joint.hide_viewport = True
        joint["_pose_joint_index"] = idx
        joint.data.materials.clear()
        joint.data.materials.append(joint_material)
        link_to_collection(joint, collection)

    for idx_a, idx_b in PED_SKELETON_EDGES:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12,
            radius=POSE_BONE_RADIUS_M,
            depth=1.0,
            location=(0.0, 0.0, 0.0),
        )
        bone = bpy.context.active_object
        bone.name = f"{instance_name}__bone_{idx_a:02d}_{idx_b:02d}"
        bone.parent = root
        bone.matrix_parent_inverse = root.matrix_world.inverted()
        bone.rotation_mode = "QUATERNION"
        bone.hide_render = True
        bone.hide_viewport = True
        bone["_pose_edge_a"] = idx_a
        bone["_pose_edge_b"] = idx_b
        bone.data.materials.clear()
        bone.data.materials.append(bone_material)
        link_to_collection(bone, collection)

    return root


def pose_rig_members(root: bpy.types.Object) -> Tuple[Dict[int, bpy.types.Object], List[bpy.types.Object]]:
    joints: Dict[int, bpy.types.Object] = {}
    bones: List[bpy.types.Object] = []
    for child in root.children:
        joint_idx = child.get("_pose_joint_index", None)
        if joint_idx is not None:
            joints[int(joint_idx)] = child
            continue
        edge_a = child.get("_pose_edge_a", None)
        edge_b = child.get("_pose_edge_b", None)
        if edge_a is not None and edge_b is not None:
            bones.append(child)
    return joints, bones


def normalize_pedestrian_pose3d_local(det: Dict[str, Any]) -> Dict[int, mathutils.Vector]:
    pose_3d_local = det.get("pose_3d_local") or []
    if not isinstance(pose_3d_local, list) or not pose_3d_local:
        return {}

    local_points: Dict[int, mathutils.Vector] = {}
    for idx, kp in enumerate(pose_3d_local[: len(PED_KEYPOINT_NAMES)]):
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        conf = float(kp[3]) if len(kp) >= 4 else 1.0
        if conf < POSE_CONFIDENCE_THRESH:
            continue
        pt = mathutils.Vector((float(kp[0]), float(kp[1]), float(kp[2])))
        if not all(math.isfinite(v) for v in pt):
            continue
        local_points[idx] = pt

    if len(local_points) < 6:
        return {}

    hip_candidates = [local_points[idx] for idx in (11, 12) if idx in local_points]
    pelvis = sum(hip_candidates, mathutils.Vector((0.0, 0.0, 0.0))) / len(hip_candidates) if hip_candidates else sum(local_points.values(), mathutils.Vector((0.0, 0.0, 0.0))) / len(local_points)
    grounded: Dict[int, mathutils.Vector] = {idx: (pt - pelvis) for idx, pt in local_points.items()}

    foot_heights = [grounded[idx].z for idx in (15, 16, 13, 14) if idx in grounded]
    ground_z = min(foot_heights) if foot_heights else min(pt.z for pt in grounded.values())
    for idx in list(grounded.keys()):
        grounded[idx].z -= ground_z

    left_indices = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (1, 2), (3, 4)]
    for a, b in left_indices:
        if a in grounded and b not in grounded:
            mirrored = grounded[a].copy()
            mirrored.y *= -1.0
            grounded[b] = mirrored
        elif b in grounded and a not in grounded:
            mirrored = grounded[b].copy()
            mirrored.y *= -1.0
            grounded[a] = mirrored

    if 0 in grounded:
        nose = grounded[0]
        for src, dst, dx in ((0, 1, -0.04), (0, 2, 0.04), (1, 3, -0.03), (2, 4, 0.03)):
            if src in grounded and dst not in grounded:
                p = grounded[src].copy()
                p.y += dx
                grounded[dst] = p

    return grounded


def build_default_pedestrian_pose_offsets(
    target_height: float,
    target_width: float,
    right_axis: mathutils.Vector,
    up_axis: mathutils.Vector,
) -> Dict[int, mathutils.Vector]:
    offsets: Dict[int, mathutils.Vector] = {}
    width_scale = max(target_width * 1.15, 0.32)
    for idx, (lat_frac, height_frac) in PED_DEFAULT_POSE.items():
        offsets[idx] = right_axis * (lat_frac * width_scale) + up_axis * (height_frac * target_height)
    return offsets


def build_pedestrian_pose_offsets(det: Dict[str, Any]) -> Dict[int, mathutils.Vector]:
    dims_m = det.get("dims_m", [1.75, 0.55, 0.45])
    scale = float(det.get("scale", 1.0) or 1.0)
    target_height = max(float(dims_m[0]) * scale, 1.20)
    target_width = max(float(dims_m[1]) * scale, 0.35)
    right_axis, up_axis = camera_billboard_axes()
    offsets = build_default_pedestrian_pose_offsets(target_height, target_width, right_axis, up_axis)

    local_points = normalize_pedestrian_pose3d_local(det)
    if local_points:
        ys = [float(pt.y) for pt in local_points.values()]
        zs = [float(pt.z) for pt in local_points.values()]
        raw_height = max(max(zs) - min(zs), 0.45)
        raw_width = max(max(ys) - min(ys), 0.18)
        lateral_target = max(target_width * 1.55, 0.42)
        scale_h = max(0.65, min(2.80, target_height / raw_height))
        scale_w = max(0.60, min(2.40, lateral_target / raw_width))
        for idx, point in local_points.items():
            lateral_m = -float(point.y) * scale_w
            vertical_m = max(0.0, min(target_height * 1.02, float(point.z) * scale_h))
            offsets[idx] = right_axis * lateral_m + up_axis * vertical_m
        return offsets

    bbox = det.get("bbox", [])
    keypoints = det.get("keypoints") or []
    if len(bbox) >= 4 and keypoints:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bbox_w = max(x2 - x1, 1.0)
        bbox_h = max(y2 - y1, 1.0)

        torso_xs: List[float] = []
        foot_ys: List[float] = []
        confident_count = 0
        for idx, kp in enumerate(keypoints[: len(PED_KEYPOINT_NAMES)]):
            if len(kp) < 2:
                continue
            conf = float(kp[2]) if len(kp) >= 3 else 1.0
            if conf < POSE_CONFIDENCE_THRESH:
                continue
            confident_count += 1
            x = float(kp[0])
            y = float(kp[1])
            if idx in {0, 5, 6, 11, 12}:
                torso_xs.append(x)
            if idx in {13, 14, 15, 16}:
                foot_ys.append(y)

        if confident_count >= 6:
            ref_cx = statistics.mean(torso_xs) if torso_xs else 0.5 * (x1 + x2)
            ref_foot_y = max(foot_ys) if foot_ys else y2
            lateral_scale_m = max(target_width * 1.55, 0.42)

            for idx, kp in enumerate(keypoints[: len(PED_KEYPOINT_NAMES)]):
                if len(kp) < 2:
                    continue
                conf = float(kp[2]) if len(kp) >= 3 else 1.0
                if conf < POSE_CONFIDENCE_THRESH:
                    continue
                x = float(kp[0])
                y = float(kp[1])
                lateral_m = ((x - ref_cx) / bbox_w) * lateral_scale_m
                vertical_m = ((ref_foot_y - y) / bbox_h) * target_height
                vertical_m = max(0.0, min(target_height * 1.02, vertical_m))
                offsets[idx] = right_axis * lateral_m + up_axis * vertical_m
            return offsets

    return offsets


def pedestrian_pose_keypoint_count(det: Dict[str, Any]) -> int:
    local_points = normalize_pedestrian_pose3d_local(det)
    if local_points:
        return len(local_points)

    keypoints = det.get("keypoints") or []
    count = 0
    for kp in keypoints[: len(PED_KEYPOINT_NAMES)]:
        if len(kp) < 2:
            continue
        conf = float(kp[2]) if len(kp) >= 3 else 1.0
        if conf >= POSE_CONFIDENCE_THRESH:
            count += 1
    return count


def infer_pedestrian_body_pose_angles(det: Dict[str, Any]) -> Tuple[float, float]:
    local_points = normalize_pedestrian_pose3d_local(det)
    if len(local_points) >= 6:
        ls = local_points.get(5)
        rs = local_points.get(6)
        lh = local_points.get(11)
        rh = local_points.get(12)
        if ls is not None and rs is not None and lh is not None and rh is not None:
            shoulder = 0.5 * (ls + rs)
            hip = 0.5 * (lh + rh)
            torso = shoulder - hip
            pitch = max(-0.32, min(0.32, -0.45 * float(torso.x)))
            roll = max(-0.26, min(0.26, -0.65 * float(torso.y)))
            return roll, pitch

    keypoints = det.get("keypoints") or []
    bbox = det.get("bbox", [])
    if len(bbox) < 4 or pedestrian_pose_keypoint_count(det) < 6:
        return 0.0, 0.0

    bbox_w = max(float(bbox[2]) - float(bbox[0]), 1.0)
    bbox_h = max(float(bbox[3]) - float(bbox[1]), 1.0)

    def mean_points(indices: Sequence[int]) -> Optional[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for idx in indices:
            if idx >= len(keypoints):
                continue
            kp = keypoints[idx]
            if len(kp) < 2:
                continue
            conf = float(kp[2]) if len(kp) >= 3 else 1.0
            if conf < POSE_CONFIDENCE_THRESH:
                continue
            pts.append((float(kp[0]), float(kp[1])))
        if not pts:
            return None
        return (
            float(statistics.mean(pt[0] for pt in pts)),
            float(statistics.mean(pt[1] for pt in pts)),
        )

    shoulder_center = mean_points((5, 6))
    hip_center = mean_points((11, 12))
    foot_center = mean_points((15, 16))
    if shoulder_center is None or hip_center is None:
        return 0.0, 0.0

    torso_dx = (shoulder_center[0] - hip_center[0]) / bbox_w
    torso_height = max(1e-6, (hip_center[1] - shoulder_center[1]) / bbox_h)
    pitch = float(np.clip((0.34 - torso_height) * 1.35, -0.26, 0.26)) if np is not None else max(-0.26, min(0.26, (0.34 - torso_height) * 1.35))
    roll = float(np.clip(-torso_dx * 1.10, -0.32, 0.32)) if np is not None else max(-0.32, min(0.32, -torso_dx * 1.10))

    if foot_center is not None:
        hip_dx = (hip_center[0] - foot_center[0]) / bbox_w
        roll += max(-0.10, min(0.10, -hip_dx * 0.45))
    return roll, pitch


def update_pedestrian_pose_rig(root: bpy.types.Object, det: Dict[str, Any], frame: int) -> None:
    joints, bones = pose_rig_members(root)
    pose_backend = str(det.get("pose_backend", "")).strip().lower()
    sprite_available = bool(str(det.get("pymaf_sprite_path", "") or "").strip())
    pose_visible = pedestrian_pose_keypoint_count(det) >= 6 and not sprite_available
    offsets = build_pedestrian_pose_offsets(det) if pose_visible else {}

    for idx, joint in joints.items():
        offset = offsets.get(idx, mathutils.Vector((0.0, 0.0, 0.0)))
        joint.location = offset
        joint.scale = (1.0, 1.0, 1.0)
        set_visibility_keyframe(joint, frame, pose_visible)
        joint.keyframe_insert(data_path="location", frame=frame)
        joint.keyframe_insert(data_path="scale", frame=frame)
        set_keyframe_interpolation(joint, "location", frame, "LINEAR")
        set_keyframe_interpolation(joint, "scale", frame, "LINEAR")

    identity = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    for bone in bones:
        idx_a = int(bone.get("_pose_edge_a"))
        idx_b = int(bone.get("_pose_edge_b"))
        point_a = offsets.get(idx_a)
        point_b = offsets.get(idx_b)
        if point_a is None or point_b is None:
            set_visibility_keyframe(bone, frame, False)
            continue
        segment = point_b - point_a
        length = max(float(segment.length), 0.05)
        bone.location = 0.5 * (point_a + point_b)
        bone.rotation_quaternion = segment.normalized().to_track_quat("Z", "Y") if segment.length > 1e-6 else identity
        bone.scale = (1.0, 1.0, length)
        set_visibility_keyframe(bone, frame, pose_visible)
        bone.keyframe_insert(data_path="location", frame=frame)
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        bone.keyframe_insert(data_path="scale", frame=frame)
        set_keyframe_interpolation(bone, "location", frame, "LINEAR")
        set_keyframe_interpolation(bone, "rotation_quaternion", frame, "LINEAR")
        set_keyframe_interpolation(bone, "scale", frame, "LINEAR")


def create_pedestrian_actor(
    instance_name: str,
    collection: bpy.types.Collection,
    assets_dir: Optional[Path] = None,
    asset_rel: str = "",
) -> bpy.types.Object:
    root = create_pedestrian_pose_rig(instance_name, collection)
    root["_pedestrian_actor"] = True
    return root


def _ensure_pedestrian_sprite_plane(root: bpy.types.Object) -> bpy.types.Object:
    child = _ensure_overlay_plane(
        root,
        "PoseSprite",
        make_solid_material(
            "M_PedSpritePlaceholder",
            (1.0, 1.0, 1.0, 1.0),
            roughness=1.0,
            metallic=0.0,
            specular=0.0,
        ),
    )
    child["_pedestrian_sprite"] = True
    child.hide_select = True
    return child


def _pedestrian_billboard_rotation(world_location: mathutils.Vector) -> mathutils.Euler:
    camera = getattr(bpy.context.scene, "camera", None)
    right, up = camera_billboard_axes()
    normal = right.cross(up)
    if camera is not None:
        to_camera = camera.matrix_world.translation - world_location
        if normal.dot(to_camera) < 0.0:
            right = -right
            normal = right.cross(up)
    if normal.length < 1e-6:
        normal = mathutils.Vector((1.0, 0.0, 0.0))
    else:
        normal.normalize()
    rotation_matrix = mathutils.Matrix((right, up, normal)).transposed()
    return rotation_matrix.to_euler("XYZ")


def update_pedestrian_pose_sprite(root: bpy.types.Object, det: Dict[str, Any], frame: int) -> bool:
    sprite_path = str(det.get("pymaf_sprite_path", "") or "").strip()
    if not sprite_path:
        sprite = bpy.data.objects.get(f"{root.name}_PoseSprite")
        if sprite is not None:
            set_visibility_keyframe(sprite, frame, False)
        return False

    path = Path(sprite_path).expanduser()
    if not path.exists():
        sprite = bpy.data.objects.get(f"{root.name}_PoseSprite")
        if sprite is not None:
            set_visibility_keyframe(sprite, frame, False)
        return False

    try:
        image = bpy.data.images.load(str(path), check_existing=True)
    except Exception:
        sprite = bpy.data.objects.get(f"{root.name}_PoseSprite")
        if sprite is not None:
            set_visibility_keyframe(sprite, frame, False)
        return False

    sprite = _ensure_pedestrian_sprite_plane(root)
    _set_plane_uvs(sprite)

    dims_m = det.get("dims_m", [1.75, 0.55, 0.45])
    scale = float(det.get("scale", 1.0) or 1.0)
    target_height = max(float(dims_m[0]) * scale, 1.20)
    try:
        image_aspect = float(image.size[0]) / max(float(image.size[1]), 1.0)
    except Exception:
        image_aspect = 0.45
    sprite_width = max(0.18, min(target_height * 0.95, target_height * image_aspect))
    sprite_height = max(0.60, target_height * 1.02)

    alert_state = str(det.get("collision_alert_state", "none") or "none").strip().lower()
    if alert_state == "collision":
        tint_rgba = (1.0, 0.26, 0.26, 1.0)
    elif alert_state == "warning":
        tint_rgba = (1.0, 0.62, 0.18, 1.0)
    else:
        tint_rgba = (1.0, 1.0, 1.0, 1.0)

    sprite.location = mathutils.Vector((0.0, 0.0, 0.5 * sprite_height))
    sprite.rotation_mode = "XYZ"
    sprite.rotation_euler = _pedestrian_billboard_rotation(root.matrix_world.translation + sprite.location)
    sprite.scale = (sprite_width, sprite_height, 1.0)
    if sprite.data is not None and hasattr(sprite.data, "materials"):
        sprite.data.materials.clear()
        material_name = (
            f"M_PedSprite_{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:10]}_{alert_state}"
        )
        sprite.data.materials.append(make_image_sprite_material(material_name, image, tint_rgba=tint_rgba))
    set_visibility_keyframe(sprite, frame, True)
    sprite.keyframe_insert(data_path="location", frame=frame)
    sprite.keyframe_insert(data_path="rotation_euler", frame=frame)
    sprite.keyframe_insert(data_path="scale", frame=frame)
    set_keyframe_interpolation(sprite, "location", frame, "LINEAR")
    set_keyframe_interpolation(sprite, "rotation_euler", frame, "LINEAR")
    set_keyframe_interpolation(sprite, "scale", frame, "LINEAR")
    return True


def update_pedestrian_actor_body(root: bpy.types.Object, det: Dict[str, Any], frame: int) -> None:
    body = next((child for child in root.children if bool(child.get("_pedestrian_body", False))), None)
    if body is None:
        return

    dims_m = det.get("dims_m", [1.75, 0.55, 0.45])
    scale = float(det.get("scale", 1.0) or 1.0)
    target_height = max(float(dims_m[0]) * scale, 1.2)
    target_width = max(float(dims_m[1]) * scale, 0.35)
    target_length = max(float(dims_m[2]) * scale, 0.24)

    base_height = max(float(body.get("_template_height", 1.0)), 0.01)
    base_width = max(float(body.get("_template_width", 1.0)), 0.01)
    base_length = max(float(body.get("_template_length", 1.0)), 0.01)
    body.scale = (
        max(0.25, min(4.0, target_length / base_length)),
        max(0.25, min(4.0, target_width / base_width)),
        max(0.25, min(4.0, target_height / base_height)),
    )
    body.location = (0.0, 0.0, 0.0)

    yaw = float(det.get("yaw_rad") or 0.0) + float(body.get("_template_yaw_offset", 0.0))
    roll, pitch = infer_pedestrian_body_pose_angles(det)
    body.rotation_euler = mathutils.Euler((roll, pitch, yaw), "XYZ")
    body.keyframe_insert(data_path="location", frame=frame)
    body.keyframe_insert(data_path="rotation_euler", frame=frame)
    body.keyframe_insert(data_path="scale", frame=frame)
    set_keyframe_interpolation(body, "location", frame, "LINEAR")
    set_keyframe_interpolation(body, "rotation_euler", frame, "LINEAR")
    set_keyframe_interpolation(body, "scale", frame, "LINEAR")


def semantic_render_scale(cls: str, raw_scale: Any) -> float:
    scale = float(raw_scale or 1.0)
    if not math.isfinite(scale):
        scale = 1.0
    if cls in {"car", "truck", "motorcycle", "bicycle"}:
        return 1.0
    if cls == "traffic_light":
        return 1.0
    if cls in {"traffic_sign", "stop_sign", "speed_limit"}:
        return max(0.90, min(1.20, scale))
    return max(0.05, scale)


def detection_dims_m(det: Dict[str, Any]) -> Tuple[float, float, float]:
    dims = det.get("dims_m") or det.get("bbox_dims_m") or ()
    if isinstance(dims, (list, tuple)) and len(dims) >= 3:
        try:
            return float(dims[0]), float(dims[1]), float(dims[2])
        except (TypeError, ValueError):
            pass
    cls = str(det.get("class", "car"))
    primitive_dims = CLASS_PRIMITIVE_DIMS.get(cls, CLASS_PRIMITIVE_DIMS["car"])
    return float(primitive_dims[2]), float(primitive_dims[1]), float(primitive_dims[0])


def semantic_render_dims_m(det: Dict[str, Any]) -> Tuple[float, float, float]:
    cls = str(det.get("class", "car"))
    dims = det.get("dims_m") or ()
    bbox_dims = det.get("bbox_dims_m") or ()
    if cls == "traffic_light":
        source = bbox_dims if isinstance(bbox_dims, (list, tuple)) and len(bbox_dims) >= 3 else ()
    else:
        source = dims if isinstance(dims, (list, tuple)) and len(dims) >= 3 else bbox_dims
    if isinstance(source, (list, tuple)) and len(source) >= 3:
        try:
            return float(source[0]), float(source[1]), float(source[2])
        except (TypeError, ValueError):
            pass
    return detection_dims_m(det)


def semantic_render_location(det: Dict[str, Any]) -> Tuple[float, float, float]:
    pos = det.get("position_blender", [5.0, 0.0, 0.0])
    x = float(pos[0]) if len(pos) >= 1 else 5.0
    y = float(pos[1]) if len(pos) >= 2 else 0.0
    z = float(pos[2]) if len(pos) >= 3 else 0.0
    cls = str(det.get("class", "unknown"))
    if cls != "traffic_light":
        return x, y, z

    bbox_dims = det.get("bbox_dims_m") or ()
    if isinstance(bbox_dims, (list, tuple)) and len(bbox_dims) >= 1:
        try:
            head_h = max(0.05, float(bbox_dims[0]))
        except (TypeError, ValueError):
            head_h = 0.85
    else:
        head_h = 0.85
    visible_center = det.get("visible_center_blender")
    if isinstance(visible_center, (list, tuple)) and len(visible_center) >= 3:
        return (
            float(visible_center[0]),
            float(visible_center[1]),
            max(0.0, float(visible_center[2]) - 0.5 * head_h),
        )
    return x, y, z


def semantic_render_yaw(
    det: Dict[str, Any],
    world_location: Sequence[float],
) -> float:
    cls = str(det.get("class", "unknown"))
    base_yaw = float(det.get("yaw_rad") or 0.0)
    if cls not in {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}:
        return base_yaw

    x = float(world_location[0]) if len(world_location) >= 1 else 0.0
    y = float(world_location[1]) if len(world_location) >= 2 else 0.0
    radial = math.hypot(x, y)
    if radial <= 1e-5:
        return base_yaw

    facing_yaw = math.atan2(-y, -x)
    yaw_source = str(det.get("yaw_source", "") or "").strip().lower()
    if det.get("yaw_rad") is None or yaw_source in {"", "ego_facing_semantic", "track_inferred", "track_kinematics"}:
        return wrap_angle(facing_yaw)

    candidates = [wrap_angle(base_yaw), wrap_angle(base_yaw + math.pi)]
    return min(candidates, key=lambda cand: abs(wrap_angle(cand - facing_yaw)))


def assign_single_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "materials"):
        return
    mesh.materials.clear()
    mesh.materials.append(material)
    for poly in getattr(mesh, "polygons", []):
        poly.material_index = 0


def _is_vehicle_overlay_object(obj: bpy.types.Object) -> bool:
    name = str(getattr(obj, "name", "") or "").lower()
    overlay_tokens = (
        "_brake",
        "_indicator",
        "_motionarrow",
        "_statusplate",
        "_wheel_",
        "_conestripe",
        "_pose",
        "_tlarrow",
        "_tlredlens",
        "_tlyellowlens",
        "_tlgreenlens",
        "_tlactivestate",
        "_stopface",
        "_stoptext",
        "_speedlimitface",
        "_limittext",
        "_limitheading",
    )
    return any(token in name for token in overlay_tokens)


def assign_single_material_recursive(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    assign_single_material(obj, material)
    for child in list(getattr(obj, "children_recursive", [])):
        if child is None or _is_vehicle_overlay_object(child):
            continue
        assign_single_material(child, material)


def apply_semantic_roadside_material(obj: bpy.types.Object, cls: str) -> None:
    if cls not in {"traffic_cone", "traffic_cylinder", "dustbin", "fire_hydrant", "speed_bump"}:
        return
    base_color = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5, 1.0))
    material = make_solid_material(
        f"M_Semantic_{cls}",
        base_color,
        roughness=0.82 if cls != "fire_hydrant" else 0.68,
        metallic=0.0,
        specular=0.03,
    )
    assign_single_material(obj, material)
    if cls == "traffic_cone":
        apply_traffic_cone_details(obj)


def should_skip_due_to_ego_collision(
    det: Dict[str, Any],
    ego_dims_m: Tuple[float, float, float],
) -> bool:
    pos = det.get("position_blender", ())
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return False

    cls = str(det.get("class", "unknown"))
    if cls in {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}:
        return False

    dims_h, dims_w, dims_l = detection_dims_m(det)
    render_scale = semantic_render_scale(cls, det.get("scale", 1.0))
    obj_length = max(0.10, float(dims_l) * render_scale)
    obj_width = max(0.10, float(dims_w) * render_scale)
    obj_height = max(0.10, float(dims_h) * render_scale)

    x = float(pos[0])
    y = float(pos[1])
    z = float(pos[2]) if len(pos) >= 3 else 0.0

    ego_h, ego_w, ego_l = ego_dims_m
    small_roadside = cls in {"traffic_cone", "traffic_cylinder", "traffic_pole", "dustbin", "fire_hydrant"}
    forward_buffer = 2.2 if small_roadside else 0.35
    lateral_buffer = 0.65 if small_roadside else 0.12
    rear_buffer = 0.30
    vertical_buffer = 0.20

    longitudinal_overlap = (
        x <= (0.5 * ego_l + 0.5 * obj_length + forward_buffer)
        and x >= -(0.5 * ego_l + 0.5 * obj_length + rear_buffer)
    )
    lateral_overlap = abs(y) <= (0.5 * ego_w + 0.5 * obj_width + lateral_buffer)
    vertical_overlap = z <= (ego_h + obj_height + vertical_buffer)
    return bool(longitudinal_overlap and lateral_overlap and vertical_overlap)


# ============================================================================
# Object pool
# ============================================================================

class ObjectPool:
    def __init__(
        self,
        assets_dir: Path,
        objects_collection: bpy.types.Collection,
        traffic_collection: bpy.types.Collection,
        use_assets: bool,
        world_offset: Optional[Sequence[float]] = None,
        ego_dims_m: Optional[Sequence[float]] = None,
    ) -> None:
        self.assets_dir = assets_dir
        self.objects_collection = objects_collection
        self.traffic_collection = traffic_collection
        self.use_assets = use_assets
        self.world_offset = _vector3(world_offset, (0.0, 0.0, 0.0))
        ego_dims = ego_dims_m if ego_dims_m is not None else (1.52, 1.85, 4.65)
        self.ego_dims_m = tuple(float(v) for v in ego_dims[:3])
        self._objects: Dict[str, bpy.types.Object] = {}
        self._frames: Dict[str, List[int]] = {}

    def _new_object(self, det: Dict[str, Any]) -> bpy.types.Object:
        uid = det["uid"]
        cls = det.get("class", "car")
        asset_rel = det.get("asset", "")
        target_collection = self.traffic_collection if cls in {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"} else self.objects_collection

        obj = None
        if cls == "pedestrian":
            obj = create_pedestrian_actor(
                uid,
                target_collection,
                assets_dir=self.assets_dir if self.use_assets else None,
                asset_rel=str(det.get("asset", "") or ""),
            )
        elif self.use_assets and asset_rel:
            obj = instantiate_asset(
                asset_rel=asset_rel,
                assets_dir=self.assets_dir,
                collection=target_collection,
                instance_name=uid,
                unique_mesh=(cls in {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}),
            )
        elif cls in {"traffic_light", "traffic_sign", "stop_sign", "speed_limit"}:
            obj = create_primitive_fallback(cls, uid, target_collection)
        if obj is None:
            obj = create_primitive_fallback(cls, uid, target_collection)

        obj.hide_render = True
        obj.hide_viewport = True
        self._objects[uid] = obj
        self._frames[uid] = []
        return obj

    def get(self, det: Dict[str, Any]) -> bpy.types.Object:
        uid = det["uid"]
        if uid in self._objects:
            return self._objects[uid]
        return self._new_object(det)

    def place(self, det: Dict[str, Any], frame: int) -> None:
        if should_skip_due_to_ego_collision(det, self.ego_dims_m):
            return

        obj = self.get(det)
        uid = det["uid"]
        self._frames[uid].append(frame)

        position = semantic_render_location(det)
        obj.location = mathutils.Vector(offset_blender_point(position, self.world_offset))

        cls = det.get("class", "car")
        if bool(obj.get("_pose_rig", False)):
            obj.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), "XYZ")
            obj.scale = (1.0, 1.0, 1.0)
            update_pedestrian_pose_rig(obj, det, frame)
            update_pedestrian_pose_sprite(obj, det, frame)
            update_pedestrian_actor_body(obj, det, frame)
            obj.keyframe_insert(data_path="location", frame=frame)
            set_keyframe_interpolation(obj, "location", frame, "LINEAR")
            return

        yaw = semantic_render_yaw(det, obj.location) + float(obj.get("_template_yaw_offset", 0.0))
        obj.rotation_euler = mathutils.Euler((0.0, 0.0, yaw), "XYZ")

        dims_m = semantic_render_dims_m(det)
        obj_scale = semantic_render_scale(cls, det.get("scale", 1.0))
        target_height = float(dims_m[0]) * obj_scale
        target_width = float(dims_m[1]) * obj_scale
        target_length = float(dims_m[2]) * obj_scale

        base_height = max(float(obj.get("_template_height", 1.0)), 0.01)
        base_width = max(float(obj.get("_template_width", 1.0)), 0.01)
        base_length = max(float(obj.get("_template_length", 1.0)), 0.01)

        scale_x = max(0.05, min(12.0, target_length / base_length))
        scale_y = max(0.05, min(12.0, target_width / base_width))
        scale_z = max(0.05, min(12.0, target_height / base_height))

        if bool(obj.get("_traffic_signal_preserve_scale", False)):
            uniform_scale = max(0.05, min(12.0, target_height / base_height))
            scale_x = uniform_scale
            scale_y = uniform_scale
            scale_z = uniform_scale

        if cls in {"car", "truck", "motorcycle", "bicycle"}:
            mean_scale = (scale_x + scale_y + scale_z) / 3.0
            blend = 0.35
            scale_x = (1.0 - blend) * mean_scale + blend * scale_x
            scale_y = (1.0 - blend) * mean_scale + blend * scale_y
            scale_z = (1.0 - blend) * mean_scale + blend * scale_z

        obj.scale = (scale_x, scale_y, scale_z)

        if cls == "traffic_light":
            dims_m = semantic_render_dims_m(det)
            bbox_dims_m = det.get("bbox_dims_m") or []
            if len(dims_m) >= 1 and len(bbox_dims_m) >= 1:
                try:
                    total_h = max(0.25, float(dims_m[0]))
                    head_h = max(0.18, min(total_h * 0.82, float(bbox_dims_m[0])))
                    head_top_frac = 0.94
                    head_bottom_frac = max(0.18, min(0.92, head_top_frac - (head_h / total_h)))
                    obj["_tl_head_bottom_frac"] = round(head_bottom_frac, 4)
                    obj["_tl_head_top_frac"] = round(head_top_frac, 4)
                except (TypeError, ValueError):
                    pass

        if cls == "traffic_light":
            apply_traffic_light_material(
                obj,
                frame,
                str(det.get("tl_color", "unknown")),
                tl_color_conf=det.get("tl_color_conf"),
                signal_shape=str(det.get("tl_shape", "unknown")),
                detic_color_check=det.get("tl_detic_color_check"),
                detic_color_conf=det.get("tl_detic_color_conf"),
                detic_color_agrees=det.get("tl_detic_color_agrees"),
                bbox=det.get("bbox"),
                tl_centroid=det.get("tl_centroid"),
            )
        elif cls == "traffic_sign":
            apply_generic_traffic_sign_material(obj)
        elif cls == "stop_sign":
            apply_stop_sign_material(obj, self.assets_dir)
        elif cls == "speed_limit":
            apply_speed_limit_material(
                obj,
                det.get("speed_limit_value"),
                sign_label=det.get("sign_label"),
                assets_dir=self.assets_dir,
            )

        if cls in {"traffic_cone", "traffic_cylinder", "dustbin", "fire_hydrant", "speed_bump"}:
            apply_semantic_roadside_material(obj, str(cls))

        if cls in {"car", "truck", "motorcycle", "bicycle"}:
            apply_vehicle_state_material(
                obj,
                moving=bool(det.get("moving", False)),
                parked=bool(det.get("parked", False)),
                alert_state=str(det.get("collision_alert_state", "none") or "none"),
            )
            hide_vehicle_wheel_overlays(obj)
            apply_vehicle_motion_overlay(
                obj,
                frame=frame,
                moving=bool(det.get("moving", False)),
                parked=bool(det.get("parked", False)),
                direction=str(det.get("motion_direction", det.get("direction", "unknown"))),
                motion_vector_blender=det.get("motion_vector_relative_blender", det.get("motion_vector_blender")),
                motion_speed_mps=det.get("speed_relative_mps", det.get("speed_mps")),
            )
            apply_vehicle_brake_lights(
                obj,
                frame=frame,
                brake_lights_on=bool(det.get("brake_lights_on", False)),
            )
            apply_vehicle_indicator_lights(
                obj,
                frame=frame,
                turn_signal=det.get("turn_signal"),
            )

        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)
        set_keyframe_interpolation(obj, "location", frame, "LINEAR")
        set_keyframe_interpolation(obj, "rotation_euler", frame, "LINEAR")
        set_keyframe_interpolation(obj, "scale", frame, "LINEAR")

    def finalise(self, start_frame: int, end_frame: int) -> None:
        for uid, obj in self._objects.items():
            ranges = contiguous_ranges(self._frames.get(uid, []))
            if bool(obj.get("_pose_rig", False)):
                members = list(obj.children)
                for member in members:
                    set_visibility_keyframe(member, start_frame, False)
                if not ranges:
                    continue
                for first, last in ranges:
                    if first > start_frame:
                        for member in members:
                            set_visibility_keyframe(member, first - 1, False)
                    for member in members:
                        set_visibility_keyframe(member, first, True)
                        set_visibility_keyframe(member, last, True)
                    if last < end_frame:
                        for member in members:
                            set_visibility_keyframe(member, last + 1, False)
                continue

            members = [obj, *list(obj.children)]
            for member in members:
                set_visibility_keyframe(member, start_frame, False)
            if not ranges:
                continue
            for first, last in ranges:
                if first > start_frame:
                    for member in members:
                        set_visibility_keyframe(member, first - 1, False)
                for member in members:
                    set_visibility_keyframe(member, first, True)
                    set_visibility_keyframe(member, last, True)
                if last < end_frame:
                    for member in members:
                        set_visibility_keyframe(member, last + 1, False)


# ============================================================================
# Lighting / render
# ============================================================================

def setup_lighting(scene: bpy.types.Scene) -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = 5.0
    sun_data.angle = math.radians(0.55)
    if hasattr(sun_data, "use_shadow"):
        sun_data.use_shadow = True
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = mathutils.Euler((math.radians(48.0), math.radians(-12.0), math.radians(34.0)), "XYZ")

    fill_data = bpy.data.lights.new("SkyFill", type="SUN")
    fill_data.energy = 1.6
    fill_data.angle = math.radians(45.0)
    if hasattr(fill_data, "use_shadow"):
        fill_data.use_shadow = False
    fill = bpy.data.objects.new("SkyFill", fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.rotation_euler = mathutils.Euler((math.radians(20.0), math.radians(15.0), math.radians(-40.0)), "XYZ")

    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    sky = nodes.new("ShaderNodeTexSky")
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    background.inputs["Strength"].default_value = 1.10

    try:
        sky.sky_type = "HOSEK_WILKIE"
    except Exception:
        pass
    if "Sun Elevation" in sky.inputs:
        sky.inputs["Sun Elevation"].default_value = math.radians(34.0)
    if "Air" in sky.inputs:
        sky.inputs["Air"].default_value = 1.2
    if "Dust" in sky.inputs:
        sky.inputs["Dust"].default_value = 0.65
    if "Ozone" in sky.inputs:
        sky.inputs["Ozone"].default_value = 0.55

    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def configure_compositor(scene: bpy.types.Scene) -> bool:
    tree = getattr(scene, "node_tree", None)

    if tree is None:
        compositing = getattr(scene, "compositing", None)
        if compositing is not None:
            try:
                if hasattr(compositing, "use_nodes"):
                    compositing.use_nodes = True
            except Exception:
                pass
            tree = getattr(compositing, "node_tree", None)

    if tree is None:
        print("[blender] Compositor not available in this Blender build; skipping post effects.")
        return False

    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    render_layers = nodes.new("CompositorNodeRLayers")
    lens = nodes.new("CompositorNodeLensdist")
    lens.inputs["Distort"].default_value = 0.004
    lens.inputs["Dispersion"].default_value = 0.001

    color_balance = nodes.new("CompositorNodeColorBalance")
    color_balance.correction_method = "LIFT_GAMMA_GAIN"
    color_balance.lift = (0.988, 0.988, 1.0)
    color_balance.gamma = (0.995, 0.997, 1.0)
    color_balance.gain = (1.01, 1.008, 1.0)

    composite = nodes.new("CompositorNodeComposite")

    links.new(render_layers.outputs["Image"], lens.inputs["Image"])
    links.new(lens.outputs["Image"], color_balance.inputs["Image"])
    links.new(color_balance.outputs["Image"], composite.inputs["Image"])
    return True


def configure_render(
    scene: bpy.types.Scene,
    renderer: str,
    samples: int,
    render_dir: Optional[str],
    start_frame: int,
    end_frame: int,
) -> None:
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    if hasattr(scene.render, "motion_blur_shutter"):
        scene.render.motion_blur_shutter = 0.0
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True

    try:
        scene.view_settings.view_transform = "AgX"
    except Exception:
        pass
    try:
        scene.view_settings.look = "Medium Contrast"
    except Exception:
        pass
    if hasattr(scene.view_settings, "exposure"):
        scene.view_settings.exposure = -0.32
    if hasattr(scene.render, "dither_intensity"):
        scene.render.dither_intensity = 0.5

    if renderer == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "caustics_reflective"):
            scene.cycles.caustics_reflective = False
        if hasattr(scene.cycles, "caustics_refractive"):
            scene.cycles.caustics_refractive = False
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = max(samples, 16)
        if hasattr(scene.eevee, "use_motion_blur"):
            scene.eevee.use_motion_blur = False
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False
        if hasattr(scene.eevee, "use_ssr"):
            scene.eevee.use_ssr = False
        if hasattr(scene.eevee, "use_volumetric_lights"):
            scene.eevee.use_volumetric_lights = False
        if hasattr(scene.eevee, "shadow_cube_size"):
            scene.eevee.shadow_cube_size = "2048"
        if hasattr(scene.eevee, "shadow_cascade_size"):
            scene.eevee.shadow_cascade_size = "2048"

    if render_dir:
        render_path = Path(render_dir)
        render_path.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(render_path / "frame_######")
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"

    configure_compositor(scene)


def auxiliary_frame_dir(render_dir: str) -> Path:
    return Path(render_dir).resolve() / ".collage_aux"


def purge_auxiliary_frames(render_dir: str) -> None:
    out_dir = Path(render_dir).resolve()
    aux_dir = auxiliary_frame_dir(render_dir)
    if not out_dir.exists() and not aux_dir.exists():
        return

    removed = 0
    for base_dir in (out_dir, aux_dir):
        if not base_dir.exists():
            continue
        for pattern in ("frame_*_real.png", "frame_*_render.png", "frame_*_collage.png"):
            for path in base_dir.glob(pattern):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        try:
            if base_dir == aux_dir and not any(base_dir.iterdir()):
                base_dir.rmdir()
        except OSError:
            pass

    if removed:
        print(f"[blender] Removed {removed} stale auxiliary frames from {out_dir}")


def export_reference_frames(
    video_path: str,
    render_dir: str,
    source_start_frame: int,
    source_end_frame: int,
    output_start_frame: int,
) -> None:
    video = Path(video_path).resolve() if video_path else None
    if video is None or not video.exists():
        print("[blender] Reference-frame export skipped; source video not found.")
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[blender] Reference-frame export skipped; ffmpeg not available.")
        return

    out_dir = auxiliary_frame_dir(render_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(out_dir / "frame_%06d_real.png")
    select_expr = f"select=between(n\\,{int(source_start_frame)}\\,{int(source_end_frame)})"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-vf",
        select_expr,
        "-fps_mode",
        "vfr",
        "-start_number",
        str(int(output_start_frame)),
        output_pattern,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip().splitlines()[-1] if exc.stderr else str(exc)
        print(f"[blender] Reference-frame export failed: {message}")
        return

    line_count = len([line for line in result.stderr.splitlines() if "frame=" in line])
    print(
        f"[blender] Reference frames exported → {out_dir} "
        f"(source {source_start_frame}..{source_end_frame}, output start {output_start_frame}, lines={line_count})"
    )


def compose_render_collages(
    render_dir: str,
    start_frame: int,
    end_frame: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[blender] Collage export skipped; ffmpeg not available.")
        return False

    out_dir = Path(render_dir).resolve()
    aux_dir = auxiliary_frame_dir(render_dir)
    if not out_dir.exists():
        print("[blender] Collage export skipped; render directory does not exist.")
        return False

    first_real = aux_dir / f"frame_{int(start_frame):06d}_real.png"
    first_render = out_dir / f"frame_{int(start_frame):06d}.png"
    if not first_real.exists() or not first_render.exists():
        print("[blender] Collage export skipped; raw or render sequence is missing.")
        return False

    total_frames = max(0, int(end_frame) - int(start_frame) + 1)
    if total_frames <= 0:
        print("[blender] Collage export skipped; empty frame range.")
        return False

    aux_dir.mkdir(parents=True, exist_ok=True)
    raw_pattern = str(aux_dir / "frame_%06d_real.png")
    render_pattern = str(out_dir / "frame_%06d.png")
    collage_pattern = str(aux_dir / "frame_%06d_collage.png")
    filter_graph = (
        f"[0:v]scale={int(frame_width)}:{int(frame_height)}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={int(frame_width)}:{int(frame_height)}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[raw];"
        f"[1:v]scale={int(frame_width)}:{int(frame_height)}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={int(frame_width)}:{int(frame_height)}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[render];"
        "[raw][render]hstack=inputs=2[out]"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-start_number",
        str(int(start_frame)),
        "-i",
        raw_pattern,
        "-start_number",
        str(int(start_frame)),
        "-i",
        render_pattern,
        "-frames:v",
        str(total_frames),
        "-filter_complex",
        filter_graph,
        "-map",
        "[out]",
        "-start_number",
        str(int(start_frame)),
        collage_pattern,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip().splitlines()[-1] if exc.stderr else str(exc)
        print(f"[blender] Collage export failed: {message}")
        return False

    print(
        f"[blender] Collage frames composed → {out_dir} "
        f"(frames {start_frame}..{end_frame})"
    )
    return True


def promote_collage_frames(
    render_dir: str,
    start_frame: int,
    end_frame: int,
) -> int:
    out_dir = Path(render_dir).resolve()
    aux_dir = auxiliary_frame_dir(render_dir)
    promoted = 0

    for frame_no in range(int(start_frame), int(end_frame) + 1):
        render_path = out_dir / f"frame_{frame_no:06d}.png"
        archived_path = aux_dir / f"frame_{frame_no:06d}_render.png"
        collage_path = aux_dir / f"frame_{frame_no:06d}_collage.png"
        if not render_path.exists() or not collage_path.exists():
            continue
        try:
            archived_path.parent.mkdir(parents=True, exist_ok=True)
            if archived_path.exists():
                archived_path.unlink()
            render_path.rename(archived_path)
            collage_path.rename(render_path)
            promoted += 1
        except OSError:
            continue

    if promoted:
        print(
            f"[blender] Promoted {promoted} collage frames to the main render sequence "
            f"and moved auxiliary raw/render frames under {aux_dir.name}"
        )
    return promoted


def encode_render_video(
    render_dir: str,
    start_frame: int,
    fps: float,
) -> Optional[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[blender] MP4 export skipped; ffmpeg not available.")
        return None

    out_dir = Path(render_dir).resolve()
    if not out_dir.exists():
        print("[blender] MP4 export skipped; render directory does not exist.")
        return None

    output_path = out_dir / f"{out_dir.name}.mp4"
    input_pattern = str(out_dir / "frame_%06d.png")

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{float(fps):.6g}",
        "-start_number",
        str(int(start_frame)),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip().splitlines()[-1] if exc.stderr else str(exc)
        print(f"[blender] MP4 export failed: {message}")
        return None

    print(f"[blender] MP4 saved → {output_path}")
    return output_path


# ============================================================================
# Depth store
# ============================================================================

class LazyDepthStore:
    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path).resolve() if path else None
        self._npz = None
        if np is not None and self.path and self.path.exists():
            self._npz = np.load(str(self.path), allow_pickle=False)

    def get(self, frame_idx: int) -> Optional["np.ndarray"]:
        if self._npz is None:
            return None
        key = str(int(frame_idx))
        if key not in self._npz.files:
            return None
        return self._npz[key]

    def close(self) -> None:
        if self._npz is not None:
            self._npz.close()
            self._npz = None


# ============================================================================
# Main build
# ============================================================================

def build_scene(args: argparse.Namespace) -> None:
    scene_json_path = Path(args.scene_json).resolve()
    if not scene_json_path.exists():
        print(f"[blender] FATAL — scene JSON not found: {scene_json_path}")
        return

    with scene_json_path.open() as handle:
        scene_data = json.load(handle)

    meta = scene_data.get("meta", {})
    scene_name = infer_scene_name(
        str(meta.get("scene", "")),
        args.scene_json,
        args.render_dir,
        args.out_blend,
    )
    output_layout = scene_output_layout(scene_name, create=True)
    frames: List[Dict[str, Any]] = scene_data.get("frames", [])
    if not frames:
        print("[blender] FATAL — no frames in assembled scene.")
        return

    start_idx = args.start_frame if args.start_frame is not None else int(frames[0]["frame_idx"])
    end_idx = args.end_frame if args.end_frame is not None else int(frames[-1]["frame_idx"])
    selected_frames = [
        frame for frame in frames if start_idx <= int(frame["frame_idx"]) <= end_idx
    ]
    if not selected_frames:
        print("[blender] FATAL — selected frame range is empty.")
        return

    assets_dir = Path(args.assets_dir).resolve()
    use_assets = assets_dir.exists() and not args.no_assets
    render_dir = args.render_dir
    if render_dir is None and args.render:
        render_dir = str(output_layout.renders.resolve())
    requested_camera_mode = str(getattr(args, "camera_mode", "chase") or "chase").strip().lower()
    effective_camera_mode = "chase"
    if requested_camera_mode != "chase":
        print(
            f"[blender] Overriding camera_mode='{requested_camera_mode}' -> 'chase' "
            "so the ego vehicle always remains visible in the final output."
        )
    if not bool(getattr(args, "export_reference_frames", True)):
        print("[blender] Forcing reference-frame export on so rendered outputs always include raw comparisons.")
        args.export_reference_frames = True
    if not bool(getattr(args, "compose_collage", True)):
        print("[blender] Forcing collage composition on so saved render frames are always raw-vs-render side-by-sides.")
        args.compose_collage = True

    depth_cfg = meta.get("depth_geometry", {})
    depth_stride = int(args.depth_stride or depth_cfg.get("mesh_stride_px", 32))
    depth_stride = max(24, depth_stride)
    depth_top_cut = float(
        args.depth_top_cut
        if args.depth_top_cut is not None
        else depth_cfg.get("crop_top_frac", DEFAULT_DEPTH_TOP_CROP_FRAC)
    )
    depth_top_cut = max(depth_top_cut, 0.44)
    depth_bottom_cut = float(
        args.depth_bottom_cut
        if args.depth_bottom_cut is not None
        else depth_cfg.get("crop_bottom_frac", 0.22)
    )
    depth_bottom_cut = max(depth_bottom_cut, 0.28)
    depth_min_distance = float(
        args.depth_min_distance
        if args.depth_min_distance is not None
        else depth_cfg.get("min_depth_m", 2.5)
    )
    depth_min_distance = max(depth_min_distance, 5.0)
    depth_foreground_distance = float(
        args.depth_foreground_distance
        if args.depth_foreground_distance is not None
        else depth_cfg.get("foreground_depth_m", 3.8)
    )
    depth_foreground_distance = max(depth_foreground_distance, 6.5)
    depth_foreground_row = float(
        args.depth_foreground_row
        if args.depth_foreground_row is not None
        else depth_cfg.get("foreground_row_start_frac", 0.72)
    )
    depth_foreground_row = min(depth_foreground_row, 0.60)
    depth_foreground_boost = float(
        args.depth_foreground_boost
        if args.depth_foreground_boost is not None
        else depth_cfg.get("foreground_bottom_boost_m", 4.0)
    )
    depth_foreground_boost = max(depth_foreground_boost, 8.5)
    depth_bbox_margin = max(20, int(args.depth_bbox_margin or depth_cfg.get("bbox_margin_px", 12)))
    depth_max_distance = min(42.0, float(depth_cfg.get("max_depth_m", MAX_USEFUL_DEPTH_M)))
    fps = float(args.fps or meta.get("fps", 15.0))

    print("\n" + "═" * 72)
    print("  Blender Scene Builder")
    print(f"  scene_json    = {scene_json_path}")
    print(f"  frames        = {selected_frames[0]['frame_idx']}..{selected_frames[-1]['frame_idx']}")
    print(f"  assets_dir    = {assets_dir} (use_assets={use_assets})")
    print(f"  render_dir    = {render_dir or 'not rendering'}")
    print("═" * 72 + "\n")

    reset_scene()
    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(fps)))

    root = ensure_collection(COL_ROOT)
    background_col = ensure_collection(COL_BG, root)
    depth_col = ensure_collection(COL_DEPTH, root)
    object_col = ensure_collection(COL_OBJECTS, root)
    traffic_col = ensure_collection(COL_TRAFFIC, root)
    lane_col = ensure_collection(COL_LANES, root)
    road_col = ensure_collection(COL_ROAD, root)
    marking_col = ensure_collection(COL_MARKINGS, root)
    ego_col = ensure_collection(COL_EGO, root)

    world_offset = get_scene_world_offset(meta)
    cam_obj = setup_camera(
        meta,
        world_offset=world_offset,
        camera_mode=effective_camera_mode,
        chase_distance=float(getattr(args, "chase_distance", 8.5)),
        chase_height=float(getattr(args, "chase_height", 4.8)),
        chase_pitch_deg=float(getattr(args, "chase_pitch_deg", 22.0)),
    )
    setup_lighting(scene)
    create_ego_vehicle_proxy(meta, ego_col, assets_dir=assets_dir if use_assets else None)

    movie_image = None
    if args.use_background_plate or args.use_source_textures:
        movie_image = get_movie_image(str(meta.get("video_path", "")))
    if args.use_background_plate:
        create_background_plate(cam_obj, str(meta.get("video_path", "")), background_col)

    depth_store = LazyDepthStore(meta.get("depth_npz_path"))
    pool = ObjectPool(
        assets_dir=assets_dir,
        objects_collection=object_col,
        traffic_collection=traffic_col,
        use_assets=use_assets,
        world_offset=world_offset,
        ego_dims_m=(
            float(meta.get("ego_vehicle", {}).get("dims_m", {}).get("height", 1.52)),
            float(meta.get("ego_vehicle", {}).get("dims_m", {}).get("width", 1.85)),
            float(meta.get("ego_vehicle", {}).get("dims_m", {}).get("length", 4.65)),
        ),
    )

    lane_count = 0
    road_count = 0
    road_marking_count = 0
    depth_shell_count = 0

    for ordinal, frame_data in enumerate(selected_frames):
        frame_idx = int(frame_data["frame_idx"])
        blender_frame = frame_idx + 1
        scene.frame_set(blender_frame)

        if ordinal % 30 == 0:
            print(
                f"  [{ordinal:5d}/{len(selected_frames)}] frame={frame_idx:5d} "
                f"obj={len(frame_data.get('objects', [])):2d} "
                f"tl={len(frame_data.get('traffic_lights', [])):2d} "
                f"lanes={len(frame_data.get('lanes', [])):2d}",
                end="\r",
                flush=True,
            )

        if not args.no_depth_shell:
            depth_map = depth_store.get(frame_idx)
            if depth_map is not None:
                depth_obj = create_depth_shell(
                    frame_idx=frame_idx,
                    depth_map=depth_map,
                    frame_meta=frame_data,
                    calib_meta=meta.get("calib", {}),
                    movie_image=movie_image,
                    use_source_textures=args.use_source_textures,
                    collection=depth_col,
                    stride=depth_stride,
                    crop_top_frac=depth_top_cut,
                    crop_bottom_frac=depth_bottom_cut,
                    min_depth_m=depth_min_distance,
                    foreground_depth_m=depth_foreground_distance,
                    foreground_row_start=depth_foreground_row,
                    foreground_bottom_boost_m=depth_foreground_boost,
                    bbox_margin=depth_bbox_margin,
                    max_depth_m=depth_max_distance,
                    world_offset=world_offset,
                )
                if depth_obj is not None:
                    set_visibility_keyframe(depth_obj, blender_frame - 1, False)
                    set_visibility_keyframe(depth_obj, blender_frame, True)
                    set_visibility_keyframe(depth_obj, blender_frame + 1, False)
                    depth_shell_count += 1

        if not args.no_lanes:
            for lane in frame_data.get("lanes", []):
                lane_obj = create_lane_curve(
                    points_3d=lane.get("points_3d", []),
                    lane_color=str(lane.get("lane_color", "unknown")),
                    lane_type=str(lane.get("lane_type", "solid")),
                    avg_hsv=lane.get("avg_hsv"),
                    avg_ycrcb=lane.get("avg_ycrcb"),
                    color_confidence=lane.get("lane_color_confidence"),
                    lane_id=int(lane.get("id", 0)),
                    frame_idx=frame_idx,
                    collection=lane_col,
                    world_offset=world_offset,
                )
                if lane_obj is None:
                    continue
                set_visibility_keyframe(lane_obj, blender_frame - 1, False)
                set_visibility_keyframe(lane_obj, blender_frame, True)
                set_visibility_keyframe(lane_obj, blender_frame + 1, False)
                lane_count += 1

        if frame_data.get("road", {}).get("contours_3d"):
            road_obj = create_road_surface(
                contours_3d=frame_data["road"]["contours_3d"],
                frame_idx=frame_idx,
                collection=road_col,
                world_offset=world_offset,
            )
            if road_obj is not None:
                set_visibility_keyframe(road_obj, blender_frame - 1, False)
                set_visibility_keyframe(road_obj, blender_frame, True)
                set_visibility_keyframe(road_obj, blender_frame + 1, False)
                road_count += 1

        if not args.no_lanes:
            for marking in frame_data.get("road", {}).get("markings", []):
                marking_obj = create_road_marking_surface(
                    contour_3d=marking.get("contour_3d", []),
                    color=str(marking.get("color", "unknown")),
                    marking_type=str(marking.get("marking_type", "road_marking")),
                    marking_id=int(marking.get("id", 0)),
                    frame_idx=frame_idx,
                    collection=marking_col,
                    world_offset=world_offset,
                )
                if marking_obj is None:
                    continue
                set_visibility_keyframe(marking_obj, blender_frame - 1, False)
                set_visibility_keyframe(marking_obj, blender_frame, True)
                set_visibility_keyframe(marking_obj, blender_frame + 1, False)
                road_marking_count += 1

        detections = (
            frame_data.get("objects", [])[: args.max_objects_per_frame]
            + frame_data.get("traffic_lights", [])
        )
        for det in detections:
            if float(det.get("position_blender", [0.0])[0]) > MAX_USEFUL_DEPTH_M:
                continue
            pool.place(det, blender_frame)

    start_bl_frame = int(selected_frames[0]["frame_idx"]) + 1
    end_bl_frame = int(selected_frames[-1]["frame_idx"]) + 1
    pool.finalise(start_bl_frame, end_bl_frame)
    depth_store.close()

    print(
        f"\n\n[blender] Objects={len(pool._objects)}  "
        f"depth_shells={depth_shell_count}  lanes={lane_count}  roads={road_count}  markings={road_marking_count}"
    )

    configure_render(
        scene=scene,
        renderer=args.renderer,
        samples=args.samples,
        render_dir=render_dir,
        start_frame=start_bl_frame,
        end_frame=end_bl_frame,
    )

    out_blend = Path(args.out_blend).resolve() if args.out_blend else (output_layout.scene_data / "scene.blend")
    out_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    if args.out_blend is None:
        mirror_stage_output(out_blend, scene_name, "scene_data", out_blend.name)
    print(f"[blender] Scene saved → {out_blend}")

    if args.render:
        if render_dir:
            purge_auxiliary_frames(render_dir)
        if render_dir and (args.export_reference_frames or args.compose_collage):
            export_reference_frames(
                video_path=str(meta.get("video_path", "")),
                render_dir=render_dir,
                source_start_frame=start_idx,
                source_end_frame=end_idx,
                output_start_frame=start_bl_frame,
            )
        print(f"[blender] Rendering frames {start_bl_frame}..{end_bl_frame} …")
        bpy.ops.render.render(animation=True, write_still=False)
        print(f"[blender] Render complete → {render_dir}")
        if render_dir and args.compose_collage:
            collages_ok = compose_render_collages(
                render_dir=render_dir,
                start_frame=start_bl_frame,
                end_frame=end_bl_frame,
                frame_width=int(scene.render.resolution_x),
                frame_height=int(scene.render.resolution_y),
            )
            if collages_ok:
                promote_collage_frames(
                    render_dir=render_dir,
                    start_frame=start_bl_frame,
                    end_frame=end_bl_frame,
                )
        if render_dir and args.encode_mp4:
            mp4_path = encode_render_video(
                render_dir=render_dir,
                start_frame=start_bl_frame,
                fps=scene.render.fps,
            )
            if mp4_path is not None:
                mirror_stage_output(mp4_path, scene_name, "videos", mp4_path.name)

    print("[blender] Done.")


if __name__ == "__main__":
    build_scene(parse_args())
