"""Render overlay timelines to video.

Outputs (any combination):

* ``composite``: H.264 MP4, overlays burned into the source footage,
* ``alpha``: transparent overlay stream as WebM (VP9, yuva420p) for a vision mixer,
* ``png_dir``: RGBA PNG sequence (every Nth frame) for inspection or keying.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from pitchlens.schema import CameraTrack, MatchMeta, OverlayTimeline, ScreenTracks

from .compositor import OverlayRenderer, composite, unpremultiply


def _ffmpeg_writer(path: Path, w: int, h: int, fps: float, pix_in: str, codec_args: list[str]) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", *codec_args, str(path)],
        stdin=subprocess.PIPE,
    )


def render_video(footage: Path, timeline: OverlayTimeline, camera: CameraTrack, meta: MatchMeta,
                 tracks: ScreenTracks | None = None, composite_out: Path | None = None, alpha_out: Path | None = None,
                 png_dir: Path | None = None, png_every: int = 30, lang: str | None = None,
                 frame_range: tuple[int, int] | None = None) -> dict:
    cap = cv2.VideoCapture(str(footage))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or timeline.media.fps
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0, f1 = frame_range or (0, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    rend = OverlayRenderer(meta, W, H)
    comp = _ffmpeg_writer(composite_out, W, H, fps, "bgr24", ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart"]) if composite_out else None
    alpha = _ffmpeg_writer(alpha_out, W, H, fps, "rgba", ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0",
                           "-crf", "32", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1"]) if alpha_out else None
    t0 = time.perf_counter()
    for i in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        layer = rend.render_layer(timeline, camera, i, tracks, lang)
        if comp is not None:
            assert comp.stdin is not None
            comp.stdin.write(composite(fr, layer).tobytes())
        if alpha is not None or (png_dir is not None and (i - f0) % png_every == 0):
            straight = unpremultiply(layer)
            if alpha is not None:
                assert alpha.stdin is not None
                alpha.stdin.write(straight.tobytes())
            if png_dir is not None and (i - f0) % png_every == 0:
                png_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(png_dir / f"overlay_{i:05d}.png"), cv2.cvtColor(straight, cv2.COLOR_RGBA2BGRA))
    cap.release()
    for p in (comp, alpha):
        if p is not None:
            assert p.stdin is not None
            p.stdin.close()
            p.wait()
    el = time.perf_counter() - t0
    return {"frames": f1 - f0, "seconds": round(el, 1), "render_fps": round((f1 - f0) / max(el, 1e-6), 1)}


def grab(video: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    _, fr = cap.read()
    cap.release()
    return fr
