"""Manual pitch calibration tool.

Real clips: export a frame, click >=4 pitch keypoints (any image viewer that
shows pixel coordinates, or the web UI's calibration page), and save::

    {"keyframes": [{"frame": 0, "points": [{"name": "right_box_tr", "px": [1012, 388]}, ...]}]}

Names come from ``pitch_model.KEYPOINTS``; a custom point may give
``"pitch": [x, y]`` instead of a name.

Synthetic clips: ``--from-gt`` simulates a human operator by projecting the
visible keypoints of one frame through the ground-truth camera and adding
pixel noise (default 1.5 px). Registration only ever sees these "clicks";
the rest of the ground truth is used for evaluation only.

Usage::

    python -m pitchlens.vision.calibrate export --video clip.mp4 --frame 0 --out frame0.png
    python -m pitchlens.vision.calibrate from-gt --camera gt/camera_gt.json --frame 0 --out calib.json
    python -m pitchlens.vision.calibrate check --video clip.mp4 --calib calib.json --out check.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pitchlens.schema import CameraTrack, apply_h

from . import pitch_model as pm
from .registration import Calibration


def read_frame(video: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {video}")
    return fr


def from_ground_truth(camera: CameraTrack, frames: list[int], noise_px: float = 1.5, seed: int = 0,
                      max_points: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    kfs = []
    for f in frames:
        H = camera.H(f)
        pts = []
        for name, xy in pm.KEYPOINTS.items():
            px = apply_h(H, np.array([xy]))[0]
            if 40 <= px[0] < camera.width - 40 and camera.height * 0.25 <= px[1] < camera.height - 30:
                pts.append({"name": name, "px": (px + rng.normal(0, noise_px, 2)).round(1).tolist()})
        if len(pts) < 4:
            raise RuntimeError(f"frame {f}: only {len(pts)} keypoints visible; pick another keyframe")
        kfs.append({"frame": f, "points": pts[:max_points]})
    return {"keyframes": kfs, "note": "Simulated operator clicks (ground truth + noise)."}


def draw_check(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    out = frame.copy()
    for pl in pm.polylines():
        dense = np.vstack([np.linspace(a, b, 12) for a, b in zip(pl[:-1], pl[1:], strict=True)])
        p = apply_h(H, dense)
        cv2.polylines(out, [np.round(p).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--video", type=Path, required=True)
    e.add_argument("--frame", type=int, default=0)
    e.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("from-gt")
    g.add_argument("--camera", type=Path, required=True)
    g.add_argument("--frame", type=int, nargs="+", default=[0])
    g.add_argument("--noise", type=float, default=1.5)
    g.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--video", type=Path, required=True)
    c.add_argument("--calib", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        cv2.imwrite(str(a.out), read_frame(a.video, a.frame))
        print("keypoint names:", ", ".join(pm.KEYPOINTS))
    elif a.cmd == "from-gt":
        cam = CameraTrack.model_validate_json(a.camera.read_text())
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(from_ground_truth(cam, a.frame, a.noise), indent=2))
        print(a.out)
    else:
        kf = Calibration.load(a.calib)[0]
        cv2.imwrite(str(a.out), draw_check(read_frame(a.video, kf.frame), kf.homography()))
        print(a.out)


if __name__ == "__main__":
    main()
