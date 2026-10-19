"""End-to-end vision pipeline for one clip (cached, resumable).

Stages and cached artifacts under ``<out>/``:

1. ``homography.json``      pitch registration (``registration.PitchRegistrar``)
2. ``detections.parquet``   per-frame person/ball detections
3. ``tracks.parquet``       ByteTrack-lite tracks + team label + pitch xy + speed
4. ``identity.json``        track id -> player id (reconciled with the dataset)
5. ``screen_tracks.json``   renderer-facing tracks for identified players
6. ``auto_events.json``     vision-estimated touches/passes (source=vision_estimated)
7. ``debug.mp4``            boxes, ids, teams and projected pitch lines

Usage::

    python -m pitchlens.vision.pipeline --video outputs/footage/demo30/footage.mp4 \
        --calib data/calibration/demo30_calib.json --out outputs/footage/demo30/vision \
        --match data/synthetic/demo --match-t0 5052
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchlens.schema import CameraTrack, apply_h

from . import pitch_model as pm
from .auto_events import detect_ball_events
from .detect import ColorBlobDetector
from .registration import Calibration, PitchRegistrar, RegistrationConfig
from .screen_tracks import from_table
from .track import ByteTrackLite, classify_teams, reconcile_identities, shirt_color, smooth_pitch


def run_registration(video: Path, calib: Path, out: Path, force: bool = False,
                     exclude: tuple[tuple[int, int, int, int], ...] = ()) -> CameraTrack:
    p = out / "homography.json"
    if p.exists() and not force:
        return CameraTrack.model_validate_json(p.read_text())
    cam = PitchRegistrar(Calibration.load(calib), RegistrationConfig(exclude_boxes=exclude)).run(video, progress=True)
    p.write_text(cam.model_dump_json())
    return cam


def run_detection_tracking(video: Path, cam: CameraTrack, out: Path, force: bool = False,
                           exclude: tuple[tuple[int, int, int, int], ...] = ()) -> pd.DataFrame:
    p = out / "tracks_raw.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    det = ColorBlobDetector(exclude_boxes=exclude)
    trk = ByteTrackLite()
    cap = cv2.VideoCapture(str(video))
    rows, det_rows, colors = [], [], {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        H = cam.H(i)
        dets = det.detect(fr, H)
        for d in dets:
            det_rows.append((i, d.x1, d.y1, d.x2, d.y2, d.conf, d.cls))
        for tid, box, _ in trk.update(dets, H):
            rows.append((i, tid, *box))
            if i % 5 == 0:
                c = shirt_color(fr, box)
                if c is not None:
                    colors.setdefault(tid, []).append(c)
        i += 1
    cap.release()
    pd.DataFrame(det_rows, columns=["frame", "x1", "y1", "x2", "y2", "conf", "cls"]).to_parquet(out / "detections.parquet")
    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
    med = {tid: np.median(np.array(c), axis=0) for tid, c in colors.items() if len(c) >= 2}
    teams, cent = classify_teams(med)
    df["team"] = df.track_id.map(teams).fillna(-1).astype(int)
    # Foot point -> pitch metres
    foot = np.c_[(df.x1 + df.x2) / 2, df.y2 - 0.06 * (df.y2 - df.y1)]
    pitch = np.zeros_like(foot)
    for f, idx in df.groupby("frame").indices.items():
        pitch[idx] = apply_h(np.linalg.inv(cam.H(int(f))), foot[idx])
    df["pitch_x"], df["pitch_y"] = pitch[:, 0], pitch[:, 1]
    df = smooth_pitch(df, cam.fps)
    df.to_parquet(p)
    (out / "team_centroids_lab.json").write_text(json.dumps(np.asarray(cent).tolist()))
    return df


def reference_positions(match_dir: Path, t0: float, n_frames: int, fps: float) -> pd.DataFrame:
    from pitchlens.footage.broadcast import interpolate_clip
    from pitchlens.synth.generate import SyntheticMatch

    m = SyntheticMatch.load(match_dir)
    clip = interpolate_clip(m.tracking, t0, n_frames / fps, fps)
    rows = []
    for j, e in enumerate(clip.ids):
        ok = ~np.isnan(clip.xy[:, j, 0])
        for f in np.flatnonzero(ok):
            rows.append((int(f), e, float(clip.xy[f, j, 0]), float(clip.xy[f, j, 1])))
    return pd.DataFrame(rows, columns=["frame", "entity", "x", "y"])


def write_debug_video(video: Path, cam: CameraTrack, tracks: pd.DataFrame, identity: dict[int, str], out: Path,
                      scale: float = 0.5) -> None:
    cap = cv2.VideoCapture(str(video))
    W, H = int(cam.width * scale), int(cam.height * scale)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(cam.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    by_f = {f: g for f, g in tracks.groupby("frame")}
    col = {0: (60, 60, 230), 1: (240, 240, 240), -1: (0, 220, 255)}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        Hf = cam.H(i)
        for pl in pm.polylines():
            dense = np.vstack([np.linspace(a, b, 10) for a, b in zip(pl[:-1], pl[1:], strict=True)])
            cv2.polylines(fr, [np.round(apply_h(Hf, dense)).astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
        g = by_f.get(i)
        if g is not None:
            for r in g.itertuples():
                c = col.get(int(r.team), (0, 255, 255))
                cv2.rectangle(fr, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
                label = identity.get(int(r.track_id), f"t{r.track_id}")
                cv2.putText(fr, f"{label} {r.speed_kmh:.0f}", (int(r.x1), int(r.y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            c, 2, cv2.LINE_AA)
        small = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        assert ff.stdin is not None
        ff.stdin.write(small.tobytes())
        i += 1
    assert ff.stdin is not None
    ff.stdin.close()
    ff.wait()


def run(video: Path, calib: Path, out: Path, match_dir: Path | None = None, match_t0: float | None = None,
        anchors: dict[int, str] | None = None, force: bool = False, debug: bool = True,
        exclude: tuple[tuple[int, int, int, int], ...] = ()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    t = time.time()
    cam = run_registration(video, calib, out, force, exclude)
    timings["registration_s"] = round(time.time() - t, 1)
    t = time.time()
    tracks = run_detection_tracking(video, cam, out, force, exclude)
    timings["detect_track_s"] = round(time.time() - t, 1)
    identity: dict[int, str] = dict(anchors or {})
    if match_dir is not None and match_t0 is not None:
        ref = reference_positions(match_dir, match_t0, len(cam.homographies), cam.fps)
        identity.update(reconcile_identities(tracks, ref))
    (out / "identity.json").write_text(json.dumps({str(k): v for k, v in identity.items()}, indent=1))
    tracks["player_id"] = tracks.track_id.map(identity)
    tracks.to_parquet(out / "tracks.parquet")
    ident = tracks.dropna(subset=["player_id"])
    # One row per (player, frame): if two tracks claim the same player in a frame keep the longest track.
    lengths = ident.groupby("track_id").size()
    ident = ident.assign(_len=ident.track_id.map(lengths)).sort_values("_len", ascending=False)
    ident = ident.drop_duplicates(["player_id", "frame"]).sort_values("frame")
    st = from_table(ident, cam.fps, id_col="player_id", track_col="track_id", px="px_s", py="py_s")
    (out / "screen_tracks.json").write_text(st.model_dump_json())
    t = time.time()
    events = detect_ball_events(video, cam, tracks, identity, out)
    timings["auto_events_s"] = round(time.time() - t, 1)
    if debug:
        t = time.time()
        write_debug_video(video, cam, tracks, identity, out / "debug.mp4")
        timings["debug_video_s"] = round(time.time() - t, 1)
    summary = {
        "frames": len(cam.homographies),
        "tracks": int(tracks.track_id.nunique()),
        "identified_tracks": len(identity),
        "identified_players": int(ident.player_id.nunique()),
        "auto_events": len(events),
        "timings": timings,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", type=Path, help="Synthetic match dir for identity reconciliation")
    ap.add_argument("--match-t0", type=float, help="Match clock at video frame 0")
    ap.add_argument("--anchors", type=Path, help='JSON {"track_id": "player_id"} for real clips')
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    anchors = {int(k): v for k, v in json.loads(a.anchors.read_text()).items()} if a.anchors else None
    print(json.dumps(run(a.video, a.calib, a.out, a.match, a.match_t0, anchors, a.force), indent=2))


if __name__ == "__main__":
    main()
