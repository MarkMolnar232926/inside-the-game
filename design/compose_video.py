"""Composite the animated overlay onto the real clip (local development only; output is git-ignored).

- removes the broadcaster's burned-in score bug per frame (crowd rows from just above it, feathered)
- alpha-blends design/renders/video_overlay/<frame>.png, or the score-bug-only layer outside the window
- encodes H.264 to outputs/real_deliverables/insight_animated.mp4

python design/compose_video.py [--video data/test_assets/vizuals_test.mp4]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OV = ROOT / "design/renders/video_overlay"


def patch_bug(frame: np.ndarray) -> np.ndarray:
    patch = np.vstack([frame[2:48, 84:672], frame[20:48, 84:672][::-1]]).astype(np.float32)
    a = np.ones(patch.shape[:2], np.float32)
    a = cv2.GaussianBlur(cv2.copyMakeBorder(a[8:-8, 8:-8], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0), (0, 0), 4)[..., None]
    dst = frame[46:120, 84:672].astype(np.float32)
    frame[46:120, 84:672] = (patch * a + dst * (1 - a)).astype(np.uint8)
    return frame


def over(frame: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (frame.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "data/test_assets/vizuals_test.mp4")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/real_deliverables/insight_animated.mp4")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bug = cv2.imread(str(OV / "bug.png"), cv2.IMREAD_UNCHANGED)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)
    f = n_ov = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = OV / f"{f:04d}.png"
        layer = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.exists() else bug
        n_ov += p.exists()
        enc.stdin.write(over(patch_bug(frame), layer).tobytes())
        f += 1
    enc.stdin.close()
    enc.wait()
    print(f"wrote {a.out} ({f} frames, overlay on {n_ov})")


if __name__ == "__main__":
    main()
