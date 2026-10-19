"""Prepare one key frame + all projected geometry for the overlay design explorations.

Writes design/out/{frame.jpg, heat.png, turf_number.png, zone.png, scene.js}.
Everything pitch-anchored is projected through the frame's real homography, so the
HTML/SVG comps sit on the grass exactly like the production renderer would.

python design/prepare_scene.py [--demo outputs/real_demo] [--frame 540]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pitchlens.schema import CameraTrack, ScreenTracks, apply_h
from pitchlens.vision import pitch_model as pm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "design" / "out"


def circle(cx, cy, r, n=72, a0=0.0, a1=2 * np.pi):
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def path(H, pts, close=True):
    p = apply_h(H, np.asarray(pts, float))
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in p)
    return d + (" Z" if close else "")


def local_scale(H, xy):
    a = apply_h(H, np.array([xy, [xy[0] + 1, xy[1]], [xy[0], xy[1] + 1]]))
    return float(np.linalg.norm(a[1] - a[0])), float(np.linalg.norm(a[2] - a[0]))


def warp_layer(H, rgba: np.ndarray, ppm: float, W=1920, Hh=1080, x0=-5.0, y0=-5.0):
    """Warp a pitch-space RGBA texture (ppm px per metre, origin at (x0, y0)) into the frame."""
    S = np.array([[1 / ppm, 0, x0], [0, 1 / ppm, y0], [0, 0, 1]])
    return cv2.warpPerspective(rgba, H @ S, (W, Hh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, default=ROOT / "outputs/real_demo")
    ap.add_argument("--video", type=Path, default=ROOT / "data/test_assets/vizuals_test.mp4")
    ap.add_argument("--frame", type=int, default=540)
    ap.add_argument("--player", default="KBY-07")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cam = CameraTrack.model_validate_json((a.demo / "camera.json").read_text())
    tracks = ScreenTracks.model_validate_json((a.demo / "screen_tracks.json").read_text())
    tl = json.loads((a.demo / "timelines/casual.json").read_text())
    H = cam.H(a.frame)

    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    _, frame = cap.read()
    # Remove the broadcaster's burned-in score bug (every design brings its own).
    # Patch it with the crowd texture directly above, feathered, so no inpainting smear remains.
    patch = np.vstack([frame[2:48, 84:672], frame[20:48, 84:672][::-1]]).astype(np.float32)  # crowd rows only
    alpha = np.ones(patch.shape[:2], np.float32)
    alpha = cv2.GaussianBlur(cv2.copyMakeBorder(alpha[8:-8, 8:-8], 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0), (0, 0), 4)[..., None]
    dst = frame[46:120, 84:672].astype(np.float32)
    frame[46:120, 84:672] = (patch * alpha + dst * (1 - alpha)).astype(np.uint8)
    cv2.imwrite(str(OUT / "frame.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])

    # ---- the featured player (anchored track)
    tr = tracks.tracks[a.player]
    k = int(np.argmin(np.abs(np.asarray(tr.frames) - a.frame)))
    box = tr.boxes[k]
    pxy = tr.pitch_xy[k]
    trail_idx = [i for i in range(len(tr.frames)) if a.frame - 75 <= tr.frames[i] <= a.frame]
    trail = [tr.pitch_xy[i] for i in trail_idx]
    sx, _ = local_scale(H, pxy)

    # ---- evidence (from the agent timeline)
    mk = next(i for i in tl["items"] if i["type"] == "pitch_markers")
    pts = [(p["x"], p["y"]) for p in mk["points"]]
    minutes = ["34'", "60'", "62'"]  # minutes of the three interceptions (from the evidence events)
    markers = []
    for (x, y), mnt in zip(pts, minutes, strict=False):
        s, _ = local_scale(H, (x, y))
        c = apply_h(H, np.array([[x, y]]))[0]
        markers.append({
            "x": x, "y": y, "u": float(c[0]), "v": float(c[1]), "scale": s, "minute": mnt,
            "ring1": path(H, circle(x, y, 1.2)), "ring2": path(H, circle(x, y, 2.2)), "ring3": path(H, circle(x, y, 3.4)),
            "disc": path(H, circle(x, y, 0.55)),
            "hex": path(H, circle(x, y, 1.4, 7)),
            "star": path(H, [(x + (1.5 if i % 2 == 0 else 0.62) * np.cos(-np.pi / 2 + i * np.pi / 5),
                              y + (1.5 if i % 2 == 0 else 0.62) * np.sin(-np.pi / 2 + i * np.pi / 5)) for i in range(10)]),
        })
    # connecting ground path through the evidence (chronological)
    chain = path(H, np.vstack([np.linspace(pts[i], pts[i + 1], 40) for i in range(len(pts) - 1)]), close=False)

    # ---- zones and pitch lines
    zone = path(H, np.vstack([np.linspace((70, 0), (105, 0), 30), np.linspace((105, 0), (105, 68), 30),
                              np.linspace((105, 68), (70, 68), 30), np.linspace((70, 68), (70, 0), 30)]))
    zone_edge = path(H, np.linspace((70, -1), (70, 69), 60), close=False)
    lines = [path(H, np.vstack([np.linspace(p0, p1, 16) for p0, p1 in zip(pl[:-1], pl[1:], strict=True)]), close=False)
             for pl in pm.polylines()]
    box18 = path(H, [(88.5, 13.84), (105, 13.84), (105, 54.16), (88.5, 54.16)])

    # ---- player graphics on the ground
    ring = {"r1": path(H, circle(pxy[0], pxy[1], 1.0)), "r2": path(H, circle(pxy[0], pxy[1], 1.6)),
            "r3": path(H, circle(pxy[0], pxy[1], 2.6)),
            "arc": path(H, circle(pxy[0], pxy[1], 1.25, 60, 0.3, 1.65 * np.pi), close=False),
            "trail": path(H, trail, close=False) if len(trail) > 1 else "",
            "spot": path(H, circle(pxy[0], pxy[1], 4.0))}
    foot = apply_h(H, np.array([pxy]))[0]

    # ---- raster layers projected onto the turf
    ppm = 12.0
    w, h = int(115 * ppm), int(78 * ppm)
    yy, xx = np.mgrid[0:h, 0:w]
    X, Y = xx / ppm - 5, yy / ppm - 5
    heat = np.zeros((h, w), np.float32)
    # all KBY ball-wins in the opponent half this match (oriented to attack right) for a richer heat map
    ev = [json.loads(line) for line in (ROOT / "data/synthetic/demo/events.jsonl").read_text().splitlines()]
    for e in ev:
        if e["team_id"] == "KBY" and e["type"] in ("interception", "tackle", "ball_recovery") and e["outcome"] != "fail" \
                and e["x"] is not None and e["t"] <= 3731.5:
            x, y = (e["x"], e["y"]) if e["period"] == 1 else (105 - e["x"], 68 - e["y"])
            if x > 52.5:
                heat += np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * 4.5**2))
    heat /= heat.max() + 1e-6
    cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (np.clip(heat * 1.25, 0, 1) ** 1.3 * 200).astype(np.uint8)
    inside = (X >= 0) & (X <= 105) & (Y >= 0) & (Y <= 68)
    alpha[~inside] = 0
    heat_rgba = np.dstack([cmap, alpha])
    cv2.imwrite(str(OUT / "heat.png"), warp_layer(H, heat_rgba, ppm))

    # painted turf graphic (AR "on the grass"): the number + caption, in empty grass of the attacking third.
    # Texture rows grow with pitch y, which reads mirrored from this camera: draw at y' = 68 - y, then flip.
    big = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    f_num = ImageFont.truetype(str(ROOT / "assets/fonts/Inter-Black.otf"), int(15 * ppm))
    f_cap = ImageFont.truetype(str(ROOT / "assets/fonts/Inter-Black.otf"), int(1.9 * ppm))
    P = lambda x, y: ((x + 5) * ppm, (68 - y + 5) * ppm)  # noqa: E731
    d.text(P(77.0, 13.6), "3", font=f_num, fill=(255, 255, 255, 255), anchor="mm")
    d.text(P(77.0, 5.6), "INTERCEPTIONS", font=f_cap, fill=(255, 255, 255, 255), anchor="ms")
    d.text(P(77.0, 3.3), "IN THE FINAL THIRD", font=f_cap, fill=(255, 255, 255, 255), anchor="ms")
    d.rectangle([*P(73.5, 8.1), *P(80.5, 7.75)], fill=(224, 35, 63, 255))
    num = np.asarray(big.transpose(Image.FLIP_TOP_BOTTOM))
    rgba = cv2.cvtColor(num, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(OUT / "turf_number.png"), warp_layer(H, rgba, ppm))
    # number-only variant (when a side panel already carries the caption)
    big = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.text(P(77.0, 13.6), "3", font=f_num, fill=(255, 255, 255, 255), anchor="mm")
    d.rectangle([*P(73.5, 8.1), *P(80.5, 7.75)], fill=(227, 23, 62, 255))
    rgba = cv2.cvtColor(np.asarray(big.transpose(Image.FLIP_TOP_BOTTOM)), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(OUT / "turf_3.png"), warp_layer(H, rgba, ppm))

    # occlusion matte: non-grass pixels inside every detection box, so on-turf graphics pass under players
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (32, 60, 40), (85, 255, 255)) > 0
    matte = np.zeros(frame.shape[:2], np.uint8)
    for t in tracks.tracks.values():
        if a.frame in t.frames:
            x0, y0, x1, y1 = (int(round(v)) for v in t.boxes[t.frames.index(a.frame)])
            x0, y0, x1, y1 = max(x0 - 6, 0), max(y0 - 4, 0), x1 + 6, y1 + max(4, int(0.35 * (y1 - y0)))
            matte[y0:y1, x0:x1] = (~grass[y0:y1, x0:x1]).astype(np.uint8) * 255
    # plus any coloured (non-grass, non-paint) blob on the pitch, which catches box misses and lagging boxes
    paint = (hsv[..., 1] < 70) & (hsv[..., 2] > 150)
    blob = (~grass & ~paint).astype(np.uint8)
    near = np.zeros_like(blob)  # only around detections (catches box misses / lag), never the open grass
    for t in tracks.tracks.values():
        if a.frame in t.frames:
            x0, y0, x1, y1 = (int(round(v)) for v in t.boxes[t.frames.index(a.frame)])
            near[max(y0 - 12, 0):y1 + 24, max(x0 - 18, 0):x1 + 18] = 1
    blob &= near
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    matte = np.maximum(matte, cv2.dilate(blob, np.ones((3, 3), np.uint8)) * 255)
    matte = cv2.GaussianBlur(cv2.morphologyEx(matte, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)), (0, 0), 0.8)
    cv2.imwrite(str(OUT / "players.png"), np.dstack([frame, matte]))

    # ---- the pass each interception broke (the opponent pass right before it, same orientation as the marker)
    by_id = {e["id"]: e for e in ev}
    order = [e["id"] for e in ev]
    passes = []
    for (mx, my), eid in zip(pts, ["SYN-0006-LATE-e01253", "SYN-0006-LATE-e02119", "SYN-0006-LATE-e02165"], strict=True):
        e = by_id[eid]
        pv = by_id[order[order.index(eid) - 1]]
        flip = -1 if e["period"] == 2 else 1  # KBY attacks right after orientation: 2nd half is mirrored
        dx, dy = flip * (e["x"] - pv["x"]), flip * (e["y"] - pv["y"])
        o = (mx - dx, my - dy)
        u = np.array([dx, dy]) / (np.hypot(dx, dy) + 1e-9)
        ghost = (mx + 5 * u[0], my + 5 * u[1])
        n = np.array([-u[1], u[0]])
        cut = [(mx + 1.6 * n[0], my + 1.6 * n[1]), (mx - 1.6 * n[0], my - 1.6 * n[1])]
        passes.append({"event": eid, "pass": pv["id"], "passer": pv["player_id"],
                       "from": apply_h(H, np.array([o]))[0].tolist(), "to": apply_h(H, np.array([[mx, my]]))[0].tolist(),
                       "ghost": apply_h(H, np.array([ghost]))[0].tolist(), "cut": apply_h(H, np.array(cut)).tolist(),
                       "length_m": round(float(np.hypot(dx, dy)), 1)})
    hexes = [path(H, circle(x, y, 1.5, 7, np.pi / 6, np.pi / 6 + 2 * np.pi)) for x, y in pts]

    # ---- other people for the minimap
    tr_df = pd.read_parquet(a.demo / "clip/vision/tracks.parquet")
    f_df = tr_df[tr_df.frame == a.frame]
    dots = [{"x": float(r.px_s), "y": float(r.py_s), "team": int(r.team)} for r in f_df.itertuples()
            if -2 < r.px_s < 107 and -2 < r.py_s < 70]
    corners = apply_h(np.linalg.inv(H), np.array([[0, 330], [1920, 330], [1920, 1080], [0, 1080]], float))

    scene = {
        "frame": a.frame, "W": 1920, "H": 1080,
        "player": {"id": a.player, "number": 7, "first": "Jory", "last": "Kestane", "team": "Kestrel Bay FC",
                   "box": box, "foot": foot.tolist(), "scale": sx, "pitch": pxy, "speed": 27.4, "distance_km": 8.1},
        "markers": markers, "chain": chain, "zone": zone, "zone_edge": zone_edge, "lines": lines, "box18": box18,
        "ring": ring, "dots": dots, "frustum": corners.tolist(), "passes": passes, "hexes": hexes,
        "zone_edge_ends": apply_h(H, np.array([[70, 0], [70, 68]], float)).tolist(),
        "stat": {"value": 3, "headline": "Interceptions in the attacking third", "share": 100, "dist": 20, "rank": 1,
                 "minute": "62'", "clock": "62:13", "score": [1, 1], "home": "KBY", "away": "IRV",
                 "why": "Kestane keeps snatching the ball high up the pitch: 3 interceptions in the final third, more than anyone tonight.",
                 "why_analyst": "3 final-third interceptions (100% of KBY's), avg 20 m from goal. KBY's press is creating turnovers where they hurt most.",
                 "ppda": 6.8, "ppda_opp": 10.5, "xg": [1.12, 0.38]},
    }
    (OUT / "scene.js").write_text("window.SCENE = " + json.dumps(scene) + ";\n")
    print(json.dumps({"player_box": box, "markers_px": [(round(m["u"]), round(m["v"])) for m in markers],
                      "scale_px_per_m": round(sx, 1), "dots": len(dots)}))


if __name__ == "__main__":
    main()
