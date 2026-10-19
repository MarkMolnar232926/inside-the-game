"""Auto-eventing from video (bonus): estimated touches, passes and ball speed.

From per-frame ball detections and player tracks (both in pitch metres via the
homography) we infer:

* **possession**: the nearest tracked player within 1.6 m of the ball for at
  least 3 consecutive frames,
* **pass_estimated**: possession moves between two players of the same team,
* **turnover_estimated**: possession moves to the other team,
* **ball speed**: median ground speed of the ball between release and receipt.

All outputs are ``source = "vision_estimated"`` with a confidence score, and
can be reconciled with the synthetic event stream (``reconcile``).
Limitation: lofted balls are mislocated (the ground-plane assumption breaks
in the air); such segments get low confidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchlens.schema import CameraTrack, Event, apply_h

from .detect import ColorBlobDetector


def ball_track(video: Path, cam: CameraTrack, out: Path) -> pd.DataFrame:
    p = out / "ball.parquet"
    if p.exists():
        return pd.read_parquet(p)
    dets = pd.read_parquet(out / "detections.parquet") if (out / "detections.parquet").exists() else None
    rows = []
    if dets is None:
        det = ColorBlobDetector()
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            for d in det.detect(fr, cam.H(i)):
                if d.cls == "ball":
                    rows.append((i, (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2))
            i += 1
    else:
        b = dets[dets.cls == "ball"]
        rows = list(zip(b.frame, (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, strict=True))
    cand = pd.DataFrame(rows, columns=["frame", "u", "v"])
    # Link: per frame keep the candidate closest to the previous accepted ball position.
    track, last = [], None
    for f, g in cand.groupby("frame"):
        uv = g[["u", "v"]].to_numpy()
        if last is None:
            k = 0
        else:
            d = np.linalg.norm(uv - last[1], axis=1)
            gap = f - last[0]
            k = int(np.argmin(d))
            if d[k] > 40 + 25 * gap:
                continue
        last = (f, uv[k])
        xy = apply_h(np.linalg.inv(cam.H(int(f))), uv[k : k + 1])[0]
        track.append((int(f), float(uv[k, 0]), float(uv[k, 1]), float(xy[0]), float(xy[1])))
    df = pd.DataFrame(track, columns=["frame", "u", "v", "x", "y"])
    df.to_parquet(p)
    return df


def detect_ball_events(video: Path, cam: CameraTrack, tracks: pd.DataFrame, identity: dict[int, str], out: Path) -> list[dict]:
    ball = ball_track(video, cam, out)
    fps = cam.fps
    by_f = {f: g for f, g in tracks.groupby("frame")}
    owner_seq = []
    for r in ball.itertuples():
        g = by_f.get(r.frame)
        if g is None:
            continue
        d = np.hypot(g.px_s - r.x, g.py_s - r.y)
        k = int(np.argmin(d.to_numpy()))
        if d.iloc[k] < 1.6:
            owner_seq.append((r.frame, int(g.track_id.iloc[k]), int(g.team.iloc[k]), r.x, r.y))
    # Debounce: possession = run of >= 3 frames with the same owner
    runs = []
    for f, tid, team, x, y in owner_seq:
        if runs and runs[-1]["tid"] == tid and f - runs[-1]["f1"] <= 4:
            runs[-1].update(f1=f, x1=x, y1=y, n=runs[-1]["n"] + 1)
        else:
            runs.append(dict(tid=tid, team=team, f0=f, f1=f, x0=x, y0=y, x1=x, y1=y, n=1))
    runs = [r for r in runs if r["n"] >= 3]
    events = []
    team_of_player = {pid: pid.split("-")[0] for pid in identity.values()}
    for a, b in zip(runs[:-1], runs[1:], strict=True):
        pa, pb = identity.get(a["tid"]), identity.get(b["tid"])
        if a["tid"] == b["tid"] or (pa is not None and pa == pb):
            continue  # same player (track fragment), not a pass
        dt = (b["f0"] - a["f1"]) / fps
        if dt <= 0 or dt > 4:
            continue
        seg = ball[(ball.frame >= a["f1"]) & (ball.frame <= b["f0"])]
        dist = float(np.hypot(b["x0"] - a["x1"], b["y0"] - a["y1"]))
        if len(seg) >= 3:
            sp = np.hypot(np.diff(seg.x), np.diff(seg.y)) / (np.diff(seg.frame) / fps)
            speed = float(np.median(sp)) * 3.6
        else:
            speed = dist / dt * 3.6
        coverage = len(seg) / max(b["f0"] - a["f1"] + 1, 1)
        if pa is not None and pb is not None:  # prefer reconciled identities over colour clusters
            same = team_of_player[pa] == team_of_player[pb]
        else:
            same = a["team"] == b["team"] and a["team"] >= 0
        events.append({
            "type": "pass_estimated" if same else "turnover_estimated",
            "source": "vision_estimated",
            "frame_start": int(a["f1"]), "frame_end": int(b["f0"]),
            "t_start": round(a["f1"] / fps, 2), "t_end": round(b["f0"] / fps, 2),
            "from_track": a["tid"], "to_track": b["tid"],
            "from_player": identity.get(a["tid"]), "to_player": identity.get(b["tid"]),
            "x": round(a["x1"], 2), "y": round(a["y1"], 2), "end_x": round(b["x0"], 2), "end_y": round(b["y0"], 2),
            "length_m": round(dist, 1), "ball_speed_kmh": round(speed, 1),
            "confidence": round(float(np.clip(0.4 + 0.5 * coverage - (0.2 if speed > 110 else 0), 0.05, 0.95)), 2),
        })
    (out / "auto_events.json").write_text(json.dumps(events, indent=1))
    return events


def reconcile(estimated: list[dict], synthetic: list[Event], media_t0: float, tol_s: float = 1.2,
              tol_m: float = 6.0) -> dict:
    """Match vision-estimated passes to synthetic pass events; report precision/recall + matches."""
    passes = [e for e in synthetic if e.type.value == "pass" and media_t0 <= e.t]
    used, matches = set(), []
    for est in estimated:
        if est["type"] != "pass_estimated":
            continue
        t = media_t0 + est["t_start"]
        best, bd = None, 1e9
        for e in passes:
            if e.id in used or abs(e.t - t) > tol_s:
                continue
            d = np.hypot((e.x or 0) - est["x"], (e.y or 0) - est["y"])
            if d < tol_m and d < bd:
                best, bd = e, d
        if best is not None:
            used.add(best.id)
            matches.append({"estimated_t": est["t_start"], "event_id": best.id, "dist_m": round(float(bd), 2),
                            "length_true": best.pass_info.length_m if best.pass_info else None,
                            "length_est": est["length_m"]})
    n_est = sum(1 for e in estimated if e["type"] == "pass_estimated")
    t_end = media_t0 + max((e["t_end"] for e in estimated), default=0)
    n_true = sum(1 for e in passes if e.t <= t_end)
    return {"estimated_passes": n_est, "synthetic_passes_in_window": n_true, "matched": len(matches),
            "precision": round(len(matches) / max(n_est, 1), 2), "recall": round(len(matches) / max(n_true, 1), 2),
            "matches": matches}
