"""Per-frame pitch geometry for the animated insight on a real clip (local development only).

For every frame in the insight's window, projects the attacking third, its edge line, the hex marks
(+ their landing ripples) through that frame's homography, and takes the featured player's head
point from the anchored screen track. Writes design/out/video_scene.js (window.VSCENE).

python design/video_scene.py [--demo outputs/real_demo] [--t0 14.6] [--dur 9.4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pitchlens.schema import CameraTrack, ScreenTracks, apply_h

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_scene import OUT, circle, path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, default=Path("outputs/real_demo"))
    ap.add_argument("--player", default="KBY-07")
    ap.add_argument("--t0", type=float, default=14.6, help="media time the animation starts (s)")
    ap.add_argument("--dur", type=float, default=9.4)
    a = ap.parse_args()
    cam = CameraTrack.model_validate_json((a.demo / "camera.json").read_text())
    tr = ScreenTracks.model_validate_json((a.demo / "screen_tracks.json").read_text()).tracks[a.player]
    tl = json.loads((a.demo / "timelines/casual.json").read_text())
    mk = next(i for i in tl["items"] if i["type"] == "pitch_markers")
    pts = [(p["x"], p["y"]) for p in mk["points"]]
    f0, f1 = int(round(a.t0 * cam.fps)), min(int(round((a.t0 + a.dur) * cam.fps)), len(cam.homographies) - 1)
    frames = {}
    tf = np.asarray(tr.frames)
    for f in range(f0, f1 + 1):
        H = cam.H(f)
        zone = path(H, np.vstack([np.linspace((70, 0), (105, 0), 30), np.linspace((105, 0), (105, 68), 30),
                                  np.linspace((105, 68), (70, 68), 30), np.linspace((70, 68), (70, 0), 30)]))
        k = int(np.argmin(np.abs(tf - f)))
        head = None
        if abs(int(tf[k]) - f) <= 3:  # short gaps are bridged by the nearest tracked frame
            x0, y0, x1, y1 = tr.boxes[k]
            head = [(x0 + x1) / 2, y0]
        frames[f] = {
            "zone": zone, "edge": path(H, np.linspace((70, -1), (70, 69), 60), close=False),
            "marks": [{"uv": apply_h(H, np.array([[x, y]]))[0].round(1).tolist(),
                       "hex": path(H, circle(x, y, 1.5, 7, np.pi / 6, np.pi / 6 + 2 * np.pi)),
                       "disc": path(H, circle(x, y, 0.55)), "ripple": path(H, circle(x, y, 3.4))} for x, y in pts],
            "head": head,
        }
    out = {"fps": cam.fps, "t0": a.t0, "f0": f0, "f1": f1, "frames": frames}
    (OUT / "video_scene.js").write_text("window.VSCENE = " + json.dumps(out) + ";\n")
    print(f"frames {f0}..{f1} ({(f1 - f0 + 1) / cam.fps:.1f} s), head tracked in "
          f"{sum(v['head'] is not None for v in frames.values())}/{len(frames)}")


if __name__ == "__main__":
    main()
