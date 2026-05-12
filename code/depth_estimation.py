"""
depth_estimation.py
===================

Monocular depth-estimation stage for the autonomous-driving pipeline.

Responsibilities
----------------
* estimate dense depth for frames, videos, and extracted frame sequences
* provide object-level depth summaries for downstream lifting and tracking
* write compressed ``.npz`` archives that are consumed later by ``scene_assembler.py`` and ``blender.py``

Backend strategy
----------------
``ZoeDepth`` is the preferred backend because it predicts metric depth directly. 
``MiDaS`` is kept as a practical fallback so the pipeline remains usable on machines where ZoeDepth or its dependencies are unavailable.

Output format
---------------
All public depth maps are returned as ``float32`` arrays measured in metres, or best-effort metric approximations in the MiDaS fallback path.  
This keeps the interface consistent and simplifies downstream processing, at the cost of some extra post-processing.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np

from project_setup import infer_scene_name, mirror_stage_output, scene_output_layout

# Suppress noisy hub download messages
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Device helper (mirrors object_detection.py)
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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MIDAS_MODEL_TYPES = [
    "DPT_Large",       # highest accuracy, slowest
    "DPT_Hybrid",      # good balance
    "MiDaS_small",     # fastest, lowest accuracy
]

_ZOEDEPTH_CONFIGS = [
    ("isl-org/ZoeDepth", "ZoeD_N"),   # NYU + KITTI — best for driving
    ("isl-org/ZoeDepth", "ZoeD_K"),   # KITTI-only (indoor-less training)
]

# Depth range used when converting MiDaS relative → metric
_MIDAS_MAX_RANGE_M = 80.0
_MIDAS_MIN_RANGE_M =  0.3


# ─────────────────────────────────────────────────────────────────────────────
# DepthEstimator
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Monocular depth estimator.

    Tries ZoeDepth first (metric output, no post-processing needed).
    Falls back to MiDaS if ZoeDepth cannot be loaded.

    Parameters
    ----------
    device : "auto" | "cuda" | "mps" | "cpu"
    model_type : "zoedepth" | "midas" | "auto", "auto" → tries ZoeDepth, falls back to MiDaS
    midas_type : MiDaS model variant (only used when model_type="midas")
    """

    def __init__(
        self,
        device: str = "auto",
        model_type: str = "auto",
        midas_type: str = "DPT_Large",
    ) -> None:
        self.device = _resolve_device(device)
        self._backend = None    # "zoedepth" | "midas"
        self._model = None
        self._transform = None    # MiDaS only

        if model_type in ("zoedepth", "auto"):
            loaded = self._try_load_zoedepth()
            if loaded:
                return

        if model_type in ("midas", "auto"):
            loaded = self._try_load_midas(midas_type)
            if loaded:
                return

        raise RuntimeError(
            "[DepthEstimator] Could not load any depth model.\n"
            "  Install ZoeDepth deps:  pip install timm einops\n"
            "  Install MiDaS deps:     pip install timm"
        )

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _try_load_zoedepth(self) -> bool:
        try:
            import torch
            for repo, config in _ZOEDEPTH_CONFIGS:
                try:
                    print(f"[DepthEstimator] Loading ZoeDepth ({config}) …")
                    model = torch.hub.load(
                        repo, config,
                        pretrained=True,
                        verbose=False,
                        trust_repo=True,
                    )
                    model.eval()
                    if self.device != "cpu":
                        model = model.to(self.device)
                    self._model = model
                    self._backend = "zoedepth"
                    self._zoe_config = config
                    print(f"[DepthEstimator] ZoeDepth ({config}) ready  "
                          f"device='{self.device}'")
                    return True
                except Exception as exc:
                    print(f"[DepthEstimator] ZoeDepth ({config}) failed: {exc}")
            return False
        except ImportError:
            print("[DepthEstimator] torch not available — skipping ZoeDepth")
            return False

    def _try_load_midas(self, midas_type: str) -> bool:
        types_to_try = [midas_type] + [
            t for t in _MIDAS_MODEL_TYPES if t != midas_type
        ]
        try:
            import torch
            for mtype in types_to_try:
                try:
                    print(f"[DepthEstimator] Loading MiDaS ({mtype}) …")
                    model = torch.hub.load(
                        "intel-isl/MiDaS", mtype,
                        verbose=False,
                        trust_repo=True,
                    )
                    transforms = torch.hub.load(
                        "intel-isl/MiDaS", "transforms",
                        verbose=False,
                        trust_repo=True,
                    )
                    if "DPT" in mtype:
                        transform = transforms.dpt_transform
                    else:
                        transform = transforms.small_transform

                    model.eval()
                    if self.device != "cpu":
                        model = model.to(self.device)
                    self._model = model
                    self._transform = transform
                    self._backend = "midas"
                    self._midas_type = mtype
                    print(f"[DepthEstimator] MiDaS ({mtype}) ready  "
                          f"device='{self.device}'")
                    return True
                except Exception as exc:
                    print(f"[DepthEstimator] MiDaS ({mtype}) failed: {exc}")
            return False
        except ImportError:
            print("[DepthEstimator] torch not available — skipping MiDaS")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Which model is active: "zoedepth" or "midas"."""
        return self._backend or "none"

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate per-pixel depth for a single BGR frame.

        Parameters
        ----------
        frame_bgr : (H, W, 3) uint8 BGR image

        Returns
        -------
        depth_map : (H, W) float32 numpy array, metric metres.
                    ZoeDepth: direct metric output.
                    MiDaS: converted to approximate metres via sky-clip
                           heuristic (max = _MIDAS_MAX_RANGE_M).
        """
        if self._backend == "zoedepth":
            return self._estimate_zoedepth(frame_bgr)
        elif self._backend == "midas":
            return self._estimate_midas(frame_bgr)
        else:
            raise RuntimeError("[DepthEstimator] No model loaded.")

    def _estimate_zoedepth(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            depth = self._model.infer_pil(
                _numpy_to_pil(rgb),
                output_type="numpy",
            )

        # ZoeDepth returns (H, W) float32 in metres — resize if shape differs
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        return depth

    def _estimate_midas(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        input_batch = self._transform(rgb)
        if self.device != "cpu":
            input_batch = input_batch.to(self.device)

        with torch.no_grad():
            raw = self._model(input_batch)
            raw = torch.nn.functional.interpolate(
                raw.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

        return _midas_to_metric(raw)

    # ── Batch convenience ─────────────────────────────────────────────────────

    def estimate_batch(
        self, frames: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Estimate depth for a list of BGR frames.
        Runs sequentially (ZoeDepth does not expose a batched PIL API).
        """
        return [self.estimate(f) for f in frames]

    # ── Video processing ──────────────────────────────────────────────────────

    def stream_video(
        self,
        video_path: str,
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[int, float, np.ndarray, np.ndarray], None, None]:
        """
        Lazy generator — yields one processed frame at a time without buffering
        the full video in memory.  Useful for integrating depth into a larger
        pipeline loop.

        Yields
        ------
        (frame_idx, timestamp_s, frame_bgr, depth_map)
            frame_idx : 0-based index of the *processed* frame
            timestamp_s : wall-clock position in the source video (seconds)
            frame_bgr : original BGR frame (H, W, 3) uint8
            depth_map : metric depth    (H, W) float32, metres

        Parameters
        ----------
        video_path : path to any OpenCV-readable video file
        frame_skip : process every Nth source frame (1 = every frame)
        max_frames : stop after this many *processed* frames (None = all)
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        processed = 0
        src_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                ts = src_idx / src_fps
                depth = self.estimate(frame)
                yield processed, ts, frame, depth

                processed += 1
                src_idx += 1

                if max_frames is not None and processed >= max_frames:
                    break
        finally:
            cap.release()

    def process_video(
        self,
        video_path: str,
        out_video: str = "renders/depth_output.mp4",
        out_npz: Optional[str] = "renders/depth_maps.npz",
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        layout: str = "side_by_side",
        colormap: int = cv2.COLORMAP_MAGMA,
        frame_hook: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a full video file: estimate depth per frame, write an
        annotated output video, and optionally save all depth arrays.

        Parameters
        ----------
        video_path : path to input video (any OpenCV-readable format)
        out_video : path for the annotated output video
                      pass None to skip video writing
        out_npz : path for a compressed .npz archive of depth maps
                      keyed by frame index string ("0", "1", …)
                      pass None to skip saving depth maps
        frame_skip : process every Nth source frame (1 = every frame)
        max_frames : stop after this many processed frames (None = all)
        layout : output video layout —
                        "side_by_side"  original | depth colourmap  (default)
                        "overlay"       depth colourmap blended over original
                        "depth_only"    depth colourmap alone
        colormap : cv2.COLORMAP_* constant used for depth visualisation
        frame_hook : optional callable(frame_bgr, depth_map) → None
                      called after depth estimation, before writing.
                      Use it to draw extra annotations, run detections, etc.

        Returns
        -------
        Dict[int, np.ndarray]  —  {processed_frame_idx: depth_map (H,W) float32}
        """
        return VideoDepthProcessor(self).run(
            video_path=video_path,
            out_video=out_video,
            out_npz=out_npz,
            frame_skip=frame_skip,
            max_frames=max_frames,
            layout=layout,
            colormap=colormap,
            frame_hook=frame_hook,
        )

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualise(
        self,
        depth_map: np.ndarray,
        colormap: int = cv2.COLORMAP_MAGMA,
        min_depth: float = 0.0,
        max_depth: float = 0.0,
        overlay_frame: Optional[np.ndarray] = None,
        alpha: float = 0.55,
    ) -> np.ndarray:
        """
        Convert a metric depth map to a colourised BGR visualisation.

        Parameters
        ----------
        depth_map : (H, W) float32 metres
        colormap : OpenCV colormap constant (default: MAGMA)
        min_depth : clip min (0 = auto)
        max_depth : clip max (0 = auto)
        overlay_frame : if given, blend the depth vis onto this BGR frame
        alpha : overlay blend weight [0=frame only, 1=depth only]

        Returns
        -------
        (H, W, 3) uint8 BGR image
        """
        return depth_to_rgb(
            depth_map,
            colormap=colormap,
            min_depth=min_depth,
            max_depth=max_depth,
            overlay_frame=overlay_frame,
            alpha=alpha,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VideoDepthProcessor  — full video pipeline with output control
# ─────────────────────────────────────────────────────────────────────────────

class VideoDepthProcessor:
    """
    End-to-end video depth processor.

    Wraps a DepthEstimator and handles all I/O: reading frames, writing the
    annotated output video, saving depth arrays, and reporting progress.

    Prefer using ``DepthEstimator.process_video()`` for the common case.
    Use this class directly when you need finer control (e.g. injecting
    into a larger pipeline, custom writers, or per-frame callbacks).

    Parameters
    ----------
    estimator : DepthEstimator instance to use for inference
    """

    def __init__(self, estimator: "DepthEstimator") -> None:
        self.estimator = estimator

    # ── Video I/O helpers ────────────────────────────────────────────────────

    @staticmethod
    def _open_writer(
        path: Path, fps: float, W: int, H: int
    ) -> cv2.VideoWriter:
        """Try common codecs in order; return first that opens successfully."""
        for codec in ("avc1", "mp4v", "H264"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
            if writer.isOpened():
                print(f"[VideoDepthProcessor] VideoWriter  codec={codec}  → {path}")
                return writer
            writer.release()
        raise RuntimeError("Cannot open VideoWriter with any supported codec.")

    @staticmethod
    def _make_frame(
        frame_bgr:  np.ndarray,
        depth_map:  np.ndarray,
        layout:     str,
        colormap:   int,
    ) -> np.ndarray:
        """
        Compose the output frame according to *layout*.

        "side_by_side" : original (left) | depth colourmap (right)
        "overlay" : depth colourmap alpha-blended over original
        "depth_only" : depth colourmap alone
        """
        depth_vis = depth_to_rgb(depth_map, colormap=colormap)

        if layout == "depth_only":
            return depth_vis

        if layout == "overlay":
            return cv2.addWeighted(frame_bgr, 0.5, depth_vis, 0.5, 0)

        # default: side_by_side
        if depth_vis.shape[:2] != frame_bgr.shape[:2]:
            depth_vis = cv2.resize(
                depth_vis, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return np.hstack([frame_bgr, depth_vis])

    @staticmethod
    def _draw_hud(
        canvas: np.ndarray,
        frame_idx: int,
        proc_fps: float,
        depth_map: np.ndarray,
    ) -> None:
        """
        Overlay a compact HUD in the top-left corner (in-place).
        Shows: frame index, processing FPS, and depth statistics.
        """
        valid = depth_map[depth_map > 0]
        d_min = float(valid.min()) if valid.size else 0.0
        d_max = float(valid.max()) if valid.size else 0.0
        d_mean = float(valid.mean()) if valid.size else 0.0

        lines = [
            f"Frame  : {frame_idx}",
            f"FPS    : {proc_fps:5.1f}",
            f"D min  : {d_min:5.1f} m",
            f"D max  : {d_max:5.1f} m",
            f"D mean : {d_mean:5.1f} m",
        ]
        x0, y0, lh = 8, 22, 20
        panel_w = 185
        panel_h = lh * len(lines) + 8
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0 - 4, y0 - 18),
                      (x0 + panel_w, y0 + panel_h), (15, 15, 15), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x0, y0 + i * lh),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (210, 230, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_colorbar(canvas: np.ndarray, depth_map: np.ndarray) -> None:
        """
        Draw a vertical depth colour bar on the right edge (in-place).
        Labels show approximate metric values at evenly spaced positions.
        """
        H, W = canvas.shape[:2]
        bar_w, bar_h = 18, H - 20
        bar_x = W - bar_w - 6
        bar_y = 10

        # Build a vertical gradient from the colourmap
        gradient = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(-1, 1)
        bar_img = cv2.applyColorMap(
            (gradient * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        canvas[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bar_img

        # Depth labels (near at bottom, far at top)
        valid = depth_map[depth_map > 0]
        if valid.size == 0:
            return
        lo = float(np.percentile(valid, 2))
        hi = float(np.percentile(valid, 98))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            depth_val = hi - frac * (hi - lo)   # frac=0 → top of bar → far
            label_y   = bar_y + int(frac * bar_h)
            cv2.putText(
                canvas, f"{depth_val:.0f}m",
                (bar_x - 38, label_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
            )

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        video_path: str,
        out_video: Optional[str] = "renders/depth_output.mp4",
        out_npz: Optional[str] = "renders/depth_maps.npz",
        frame_skip: int = 1,
        max_frames: Optional[int] = None,
        layout: str = "side_by_side",
        colormap: int = cv2.COLORMAP_MAGMA,
        frame_hook: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a video end-to-end.

        Parameters
        ----------
        video_path : path to input video (any OpenCV-readable format)
        out_video : annotated output video path; None = skip writing
        out_npz : .npz archive path for all depth maps; None = skip saving
        frame_skip : process every Nth source frame
        max_frames : stop after N processed frames (None = all)
        layout : "side_by_side" | "overlay" | "depth_only"
        colormap : cv2.COLORMAP_* constant
        frame_hook : callable(frame_bgr, depth_map) called after estimation.
                      Mutate the frame in-place to add extra overlays before
                      the frame is written to the output video.

        Returns
        -------
        Dict[int, np.ndarray]
            Maps processed frame index → depth map (H, W) float32, metres.
        """
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {src}")

        # ── Open source ───────────────────────────────────────────────────────
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = src_fps / max(frame_skip, 1)

        # Output canvas width depends on layout
        out_W = src_W * 2 if layout == "side_by_side" else src_W
        out_H = src_H

        # ── Open writer ───────────────────────────────────────────────────────
        writer = None
        if out_video is not None:
            out_video_path = Path(out_video)
            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = self._open_writer(out_video_path, out_fps, out_W, out_H)

        if out_npz is not None:
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*66}")
        print(f"  VideoDepthProcessor.run()")
        print(f"  Input    : {src}  ({src_W}×{src_H} @ {src_fps:.1f} fps, "
              f"~{src_total} frames)")
        print(f"  Backend  : {self.estimator.backend}  "
              f"device={self.estimator.device}")
        print(f"  Layout   : {layout}  →  {out_W}×{out_H}")
        if out_video:
            print(f"  Video    : {out_video}")
        if out_npz:
            print(f"  NPZ      : {out_npz}")
        print(f"  Skip     : every {frame_skip} frame(s)  →  ~{out_fps:.1f} fps")
        print(f"{'='*66}\n")

        # ── Processing loop ───────────────────────────────────────────────────
        depth_store: Dict[int, np.ndarray] = {}
        fps_window: List[float] = []
        processed = 0
        src_idx = 0
        t_start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Frame-skip
                if src_idx % frame_skip != 0:
                    src_idx += 1
                    continue

                t0 = time.time()

                # ── Depth estimation ─────────────────────────────────────────
                depth = self.estimator.estimate(frame)
                depth_store[processed] = depth

                # ── Optional hook ────────
                if frame_hook is not None:
                    frame_hook(frame, depth)

                # ── Compose output frame ─────────────────────────────────────
                if writer is not None:
                    canvas = self._make_frame(frame, depth, layout, colormap)
                    self._draw_colorbar(canvas, depth)

                    fps_window.append(time.time() - t0)
                    if len(fps_window) > 20:
                        fps_window.pop(0)
                    proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

                    self._draw_hud(canvas, processed, proc_fps, depth)
                    writer.write(canvas)
                else:
                    # Still track timing for the progress line
                    fps_window.append(time.time() - t0)
                    if len(fps_window) > 20:
                        fps_window.pop(0)
                    proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)

                # ── Progress ─────────────────────────────────────────────────
                processed += 1
                src_idx += 1
                if processed % 10 == 0 or processed == 1:
                    pct = src_idx / src_total * 100 if src_total > 0 else 0
                    d_mean = float(depth[depth > 0].mean()) if (depth > 0).any() else 0
                    print(
                        f"  src={src_idx:5d}/~{src_total}  "
                        f"({pct:5.1f}%)  "
                        f"{proc_fps:5.1f} fps  "
                        f"depth_mean={d_mean:.1f}m",
                        end="\r", flush=True,
                    )

                if max_frames is not None and processed >= max_frames:
                    print(f"\n  Reached max_frames={max_frames} — stopping.")
                    break

        except KeyboardInterrupt:
            print("\n  Interrupted by user.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        # ── Save depth maps ───────────────────────────────────────────────────
        if out_npz is not None and depth_store:
            np.savez_compressed(
                out_npz,
                **{str(k): v for k, v in depth_store.items()},
            )
            print(f"\n  Depth maps saved → {out_npz}  "
                  f"({len(depth_store)} frames, "
                  f"{Path(out_npz).stat().st_size // 1024} KB)")

        elapsed = time.time() - t_start
        print(f"\n{'='*66}")
        print(f"  Done.")
        print(f"  Frames processed : {processed}")
        print(f"  Wall time        : {elapsed:.1f}s  "
              f"({processed / max(elapsed, 1e-6):.1f} fps avg)")
        if out_video:
            print(f"  Output video     : {out_video}")
        if out_npz and depth_store:
            print(f"  Depth NPZ        : {out_npz}")
        print(f"{'='*66}")

        return depth_store

    # ── Frame-directory variant ───────────────────────────────────────────────

    def run_on_frames(
        self,
        frame_dir: str,
        out_video: Optional[str] = "renders/depth_output.mp4",
        out_npz: Optional[str] = "renders/depth_maps.npz",
        fps: float = 15.0,
        max_frames: Optional[int] = None,
        layout: str = "side_by_side",
        colormap: int = cv2.COLORMAP_MAGMA,
        frame_hook: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Process a directory of JPEG frames instead of a video file.
        Accepts the same output parameters as ``run()``.

        Parameters
        ----------
        frame_dir : directory containing frame_XXXXX.jpg (or any *.jpg) files
        fps : frame-rate for the output video
        """
        frame_paths = sorted(Path(frame_dir).glob("frame_*.jpg"))
        if not frame_paths:
            frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No JPEG frames found in {frame_dir}")
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
        H, W   = first.shape[:2]
        out_W  = W * 2 if layout == "side_by_side" else W

        writer = None
        if out_video is not None:
            out_video_path = Path(out_video)
            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = self._open_writer(out_video_path, fps, out_W, H)

        if out_npz is not None:
            Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[VideoDepthProcessor] Processing {len(frame_paths)} frames "
              f"from {frame_dir}")

        depth_store: Dict[int, np.ndarray] = {}
        fps_window: List[float] = []
        t_start = time.time()

        for idx, fp in enumerate(frame_paths):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            t0 = time.time()
            depth = self.estimator.estimate(frame)
            depth_store[idx] = depth

            if frame_hook is not None:
                frame_hook(frame, depth)

            if writer is not None:
                canvas = self._make_frame(frame, depth, layout, colormap)
                self._draw_colorbar(canvas, depth)
                fps_window.append(time.time() - t0)
                if len(fps_window) > 20:
                    fps_window.pop(0)
                proc_fps = 1.0 / (sum(fps_window) / len(fps_window) + 1e-9)
                self._draw_hud(canvas, idx, proc_fps, depth)
                writer.write(canvas)

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1:4d}/{len(frame_paths)}]", end="\r", flush=True)

        if writer is not None:
            writer.release()

        if out_npz is not None and depth_store:
            np.savez_compressed(out_npz,
                                **{str(k): v for k, v in depth_store.items()})

        elapsed = time.time() - t_start
        print(f"\n[VideoDepthProcessor] Finished  {len(depth_store)} frames  "
              f"{elapsed:.1f}s  →  {out_video or '(no video)'}")
        return depth_store




def depth_to_rgb(
    depth_map: np.ndarray,
    colormap: int = cv2.COLORMAP_MAGMA,
    min_depth: float = 0.0,
    max_depth: float = 0.0,
    overlay_frame: Optional[np.ndarray] = None,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Normalise and colourise a metric depth map.

    Parameters
    ----------
    depth_map : (H, W) float32 metres
    colormap : cv2.COLORMAP_* constant
    min_depth : lower clip (auto if 0)
    max_depth : upper clip (auto if 0)
    overlay_frame : optional BGR frame to blend depth onto
    alpha : depth blend weight when overlay_frame is provided

    Returns
    -------
    (H, W, 3) uint8 BGR
    """
    d = depth_map.astype(np.float32).copy()

    valid = d[d > 0]
    if valid.size == 0:
        # Completely empty depth map — return a black frame
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    lo = min_depth if min_depth > 0 else float(np.percentile(valid, 2))
    hi = max_depth if max_depth > 0 else float(np.percentile(valid, 98))

    # Guard: if the entire depth map is a single value (e.g. model returned
    # all-80m because of a failed conversion), fall back to the true range
    # so we still produce a visually meaningful (if flat) output.
    if (hi - lo) < 0.5:
        lo = float(valid.min())
        hi = float(valid.max())
    if (hi - lo) < 0.5:
        # Genuinely flat map — show a solid mid-colour rather than crashing
        mid = int(np.clip((lo - 0.3) / 79.7, 0.0, 1.0) * 255)
        flat = np.full((*d.shape, 3), mid, dtype=np.uint8)
        return cv2.applyColorMap(flat, colormap)

    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    # Invert so near = bright (more intuitive for driving)
    norm = 1.0 - norm
    vis  = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)

    if overlay_frame is not None:
        if overlay_frame.shape[:2] != vis.shape[:2]:
            overlay_frame = cv2.resize(
                overlay_frame, (vis.shape[1], vis.shape[0]))
        vis = cv2.addWeighted(overlay_frame, 1.0 - alpha, vis, alpha, 0)

    return vis


def get_object_depth(
    bbox: Union[List[int], Tuple[int, ...]],
    depth_map: np.ndarray,
    inner_fraction: float = 0.5,
    aggregation: str = "median",
) -> Optional[float]:
    """
    Return the representative metric depth (metres) for a detected object.

    Parameters
    ----------
    bbox : [x1, y1, x2, y2] pixel bounding box
    depth_map : (H, W) float32 metric depth map
    inner_fraction : fraction of the box interior to sample
                     (0.5 = central 50 % along each axis, avoids edge bleed)
    aggregation : "median" (robust) | "mean" | "min"

    Returns
    -------
    float depth in metres, or None if no valid depth values exist in the box.
    """
    H, W = depth_map.shape[:2]
    x1 = max(0, min(int(bbox[0]), W - 1))
    y1 = max(0, min(int(bbox[1]), H - 1))
    x2 = max(0, min(int(bbox[2]), W - 1))
    y2 = max(0, min(int(bbox[3]), H - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    # Shrink to the inner fraction of the box
    margin_x = int((x2 - x1) * (1.0 - inner_fraction) / 2)
    margin_y = int((y2 - y1) * (1.0 - inner_fraction) / 2)
    ix1 = x1 + margin_x
    iy1 = y1 + margin_y
    ix2 = x2 - margin_x
    iy2 = y2 - margin_y

    # Fallback: if margins consumed the entire box, use the full box
    if ix2 <= ix1:
        ix1, ix2 = x1, x2
    if iy2 <= iy1:
        iy1, iy2 = y1, y2

    patch = depth_map[iy1:iy2, ix1:ix2].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    if aggregation == "median":
        return float(np.median(valid))
    elif aggregation == "mean":
        return float(np.mean(valid))
    elif aggregation == "min":
        return float(np.min(valid))
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'. Use median/mean/min.")


def align_depth_to_detections(
    detections: list,
    depth_map: np.ndarray,
    calib=None,            # CalibData — if provided, fills position_3d too
    depth_scale: float = 1.0,
    inner_fraction: float = 0.5,
) -> list:
    """
    Enrich each detection in-place with depth and (optionally) 3-D position.

    Supports both DetectionResult dataclasses and plain dicts.

    Parameters
    ----------
    detections : list of DetectionResult or dicts with "bbox" key
    depth_map : (H, W) float32 metres
    calib : CalibData — when provided, back-projects to 3-D and fills
            det.position_3d / det["position_3d"]
    depth_scale : multiply raw depth values to get metres (use 1.0 for ZoeDepth)
    inner_fraction : see get_object_depth

    Returns
    -------
    The same list, modified in-place.
    """
    for det in detections:
        bbox = (
            getattr(det, "bbox", None)
            or (det.get("bbox") if isinstance(det, dict) else None)
        )
        if bbox is None:
            continue

        depth_val = get_object_depth(bbox, depth_map,
                                     inner_fraction=inner_fraction)
        if depth_val is None:
            continue
        depth_val *= depth_scale

        # Write depth
        if hasattr(det, "depth_m"):
            det.depth_m = depth_val
        elif isinstance(det, dict):
            det["depth_m"] = round(depth_val, 3)

        # Optionally write 3-D position
        if calib is not None:
            from calibration import pixel_to_world, camera_to_blender
            x1, y1, x2, y2 = bbox
            uc = (int(x1) + int(x2)) / 2.0
            vc = (int(y1) + int(y2)) / 2.0
            X, Y, Z = pixel_to_world(uc, vc, depth_val, calib)
            bx, by, bz = camera_to_blender(X, Y, Z)

            if hasattr(det, "position_3d"):
                det.position_3d = [bx, by, bz]
            elif isinstance(det, dict):
                det["position_3d"] = [round(bx, 3), round(by, 3), round(bz, 3)]

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Convenience singleton
# ─────────────────────────────────────────────────────────────────────────────

_estimator_singleton: Optional[DepthEstimator] = None


def estimate_depth(
    frame_bgr: np.ndarray,
    device: str = "auto",
    model_type: str = "auto",
) -> np.ndarray:
    """
    Module-level singleton wrapper.
    Initialises DepthEstimator on first call; reuses it thereafter.

    Returns
    -------
    (H, W) float32 metric depth map in metres.
    """
    global _estimator_singleton
    if _estimator_singleton is None:
        _estimator_singleton = DepthEstimator(device=device, model_type=model_type)
    return _estimator_singleton.estimate(frame_bgr)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _numpy_to_pil(rgb: np.ndarray):
    """Convert (H,W,3) uint8 RGB numpy array to PIL Image."""
    from PIL import Image
    return Image.fromarray(rgb.astype(np.uint8))


def _midas_to_metric(
    inverse_depth: np.ndarray,
    max_range_m: float = _MIDAS_MAX_RANGE_M,
    min_range_m: float = _MIDAS_MIN_RANGE_M,
    sky_percentile: float = 2.0,
) -> np.ndarray:
    """
    Convert MiDaS inverse-relative depth to approximate metric depth.

    MiDaS predicts disparity-like values: larger = closer, smaller = farther.
    The correct mapping is a linear inversion:

        inv_norm = (inv - lo) / (hi - lo)   # 0 = far, 1 = near
        depth_m  = max_range - inv_norm * (max_range - min_range)

    So near pixels (high inv_norm) map to min_range_m (close),
    and far pixels (low inv_norm) map to max_range_m (distant).

    sky_percentile trims both ends before normalising so glare / hood
    reflections do not collapse the whole scale.
    """
    inv = inverse_depth.astype(np.float32)

    lo = float(np.percentile(inv, sky_percentile))
    hi = float(np.percentile(inv, 100.0 - sky_percentile))

    # If percentile trim consumed all variance, fall back to true min/max
    if (hi - lo) < 1e-3:
        lo = float(inv.min())
        hi = float(inv.max())
    # Genuinely flat output — return mid-range rather than a wall of max
    if (hi - lo) < 1e-3:
        return np.full_like(inv, (max_range_m + min_range_m) / 2.0)

    # Normalise: 0 = farthest pixel, 1 = nearest pixel
    inv_norm = np.clip((inv - lo) / (hi - lo), 0.0, 1.0)

    # Linear inversion: near -> min_range_m, far -> max_range_m
    depth_m = max_range_m - inv_norm * (max_range_m - min_range_m)

    return np.clip(depth_m, min_range_m, max_range_m).astype(np.float32)



def process_video_pipeline(video_path, out_npz, out_vis=None):
    estimator = DepthEstimator()
    cap = cv2.VideoCapture(str(video_path))
    depth_registry = {}
    frame_idx = 0

    print(f"[depth] Processing {video_path}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        depth_map = estimator.estimate(frame)
        # Store as float16 to save ~50% disk space while keeping cm-level precision
        depth_registry[str(frame_idx)] = depth_map.astype(np.float16)
        
        if out_vis:
            # Optional: save colorized frames for debugging
            vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=3), cv2.COLORMAP_MAGMA)
            cv2.imwrite(str(Path(out_vis) / f"depth_{frame_idx:05d}.jpg"), vis)

        frame_idx += 1
        if frame_idx % 50 == 0: print(f"  Processed {frame_idx} frames")

    np.savez_compressed(out_npz, **depth_registry)
    print(f"[depth] Successfully saved metric depth to {out_npz}")
    cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# __main__  — quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Depth estimation — image or video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input — mutually exclusive
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--image",  type=str, help="Path to a single image")
    src_group.add_argument("--video",  type=str, help="Path to a video file (.mp4 etc.)")
    src_group.add_argument("--frames", type=str, help="Directory of JPEG frames")

    # Shared options
    parser.add_argument("--scene",      default=None,
                        help="Optional explicit scene id (e.g. scene1); otherwise inferred from the input path")
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--model-type", default="auto",
                        choices=["auto", "zoedepth", "midas"])
    parser.add_argument("--out",        default=None,
                        help="Output path (image: .jpg/.png; video: .mp4). Video mode defaults to output/<scene>/depth/depth_output.mp4")
    parser.add_argument("--out-npz",    default=None,
                        help="(video/frames) Save depth maps as .npz archive. Defaults to output/<scene>/depth/depth_maps.npz")

    # Video-specific options
    parser.add_argument("--layout",
                        default="side_by_side",
                        choices=["side_by_side", "overlay", "depth_only"],
                        help="Output video composition")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth source frame (video mode)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N processed frames (video/frames mode)")
    parser.add_argument("--fps",        type=float, default=15.0,
                        help="Output FPS when reading from a frames directory")

    args = parser.parse_args()

    print("\n" + "═" * 64)
    print("  RBE549 / CS549 P3 — Einstein Vision — Depth Estimation Module")
    print("═" * 64 + "\n")

    estimator = DepthEstimator(device=args.device, model_type=args.model_type)
    print(f"\n  Active backend : {estimator.backend}")
    print(f"  Device         : {estimator.device}\n")
    scene_name = infer_scene_name(args.scene, args.image, args.video, args.frames, args.out, args.out_npz)
    output_layout = scene_output_layout(scene_name, create=True)

    # ── Single image ──────────────────────────────────────────────────────────
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"  ERROR: cannot read {args.image}")
            sys.exit(1)

        H, W = frame.shape[:2]
        print(f"  Image size : {W}×{H}")

        depth = estimator.estimate(frame)
        print(f"  Depth range: {depth.min():.2f} – {depth.max():.2f} m")
        print(f"  Depth mean : {depth.mean():.2f} m  std: {depth.std():.2f} m")

        vis = estimator.visualise(depth, overlay_frame=frame, alpha=0.5)

        out_path = args.out or "/tmp/depth_image.jpg"
        cv2.imwrite(out_path, vis)
        print(f"\n  Saved depth visualisation → {out_path}")

        cx, cy = W // 2, H // 2
        bbox = [cx - 50, cy - 50, cx + 50, cy + 50]
        d = get_object_depth(bbox, depth)
        print(f"  get_object_depth (centre 100×100): "
              f"{d:.2f} m" if d else "  no valid depth in crop")

    # ── Video file ────────────────────────────────────────────────────────────
    elif args.video:
        out_video = args.out or str((output_layout.depth / "depth_output.mp4").resolve())
        out_npz = args.out_npz or str((output_layout.depth / "depth_maps.npz").resolve())

        estimator.process_video(
            args.video,
            out_video=out_video,
            out_npz=out_npz,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            layout=args.layout,
        )

    # ── Frame directory ───────────────────────────────────────────────────────
    elif args.frames:
        out_video = args.out or str((output_layout.depth / "depth_output.mp4").resolve())
        out_npz = args.out_npz or str((output_layout.depth / "depth_maps.npz").resolve())

        VideoDepthProcessor(estimator).run_on_frames(
            args.frames,
            out_video=out_video,
            out_npz=out_npz,
            fps=args.fps,
            max_frames=args.max_frames,
            layout=args.layout,
        )

    if args.video or args.frames:
        if not args.out:
            mirror_stage_output(out_video, scene_name, "depth", Path(out_video).name)
        if not args.out_npz:
            mirror_stage_output(out_npz, scene_name, "depth", Path(out_npz).name)