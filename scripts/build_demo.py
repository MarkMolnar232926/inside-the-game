"""Build the complete PitchLens demo bundle (resumable; every stage is cached).

    python scripts/build_demo.py                      # default demo clip (rights-clear synthetic footage)
    python scripts/build_demo.py --burn casual,analyst --alpha casual
    python scripts/build_demo.py --video my_clip.mp4 --calib my_calib.json --anchors anchors.json --match-t0 5199

Output: outputs/demo/{manifest.json, timelines/*.json, camera.json, screen_tracks.json, traces.json,
candidates.json, metrics.json, recap.json, media/*}
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from pitchlens.agents.director import ScreenContext
from pitchlens.agents.orchestrator import PitchLensEngine
from pitchlens.footage.broadcast import BroadcastRenderer, interpolate_clip
from pitchlens.render.assets import build_all
from pitchlens.render.compositor import OverlayRenderer, composite
from pitchlens.render.video import render_video
from pitchlens.schema import CameraTrack, MediaClock, OverlayTimeline, RecapCard, ScreenTracks, TextBlock
from pitchlens.synth.generate import SyntheticMatch, generate
from pitchlens.vision import pipeline as vision
from pitchlens.vision.alignment import retarget_timeline
from pitchlens.vision.calibrate import from_ground_truth

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[build_demo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", type=Path, default=ROOT / "data/synthetic/demo")
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--narrative", default="late_comeback")
    ap.add_argument("--start", type=float, default=3721.0, help="match clock at the first frame")
    ap.add_argument("--duration", type=float, default=30.5)
    ap.add_argument("--video", type=Path, help="use an existing clip instead of rendering synthetic footage")
    ap.add_argument("--calib", type=Path, help="manual calibration JSON (required with --video)")
    ap.add_argument("--anchors", type=Path, help='{"track_id": "player_id"} for real clips')
    ap.add_argument("--orient-team", help="real clips: show this team attacking right (towards the camera's end)")
    ap.add_argument("--no-score-bug", action="store_true", help="the clip already has a burned-in score bug")
    ap.add_argument("--exclude", default="", help='burned-in graphics to ignore, "x0,y0,x1,y1;..." (full-res px)')
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/demo")
    ap.add_argument("--burn", default="casual,analyst", help="profiles to burn into MP4 (comma list, '' for none)")
    ap.add_argument("--alpha", default="casual", help="profile to export as WebM-alpha overlay ('' for none)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out, media = a.out, a.out / "media"
    media.mkdir(parents=True, exist_ok=True)
    (out / "timelines").mkdir(exist_ok=True)

    # 1. data -------------------------------------------------------------
    if not (a.match / "tracking.parquet").exists():
        log(f"generating synthetic match seed={a.seed} narrative={a.narrative}")
        generate(a.seed, a.narrative).save(a.match)
    m = SyntheticMatch.load(a.match)
    build_all(m.meta)

    # 2. footage ----------------------------------------------------------
    clip_dir = out / "clip"
    footage = a.video or (media / "footage.mp4")
    if a.video is None and (a.force or not footage.exists()):
        log("rendering rights-clear synthetic broadcast footage")
        clip = interpolate_clip(m.tracking, a.start, a.duration)
        r = BroadcastRenderer(m.meta)
        r.render(clip, r.cameras(clip, 30.0), footage, gt_dir=clip_dir / "gt")
    # 3. calibration + vision ----------------------------------------------
    calib = a.calib
    if calib is None:
        calib = clip_dir / "calib.json"
        if not calib.exists():
            cam_gt = CameraTrack.model_validate_json((clip_dir / "gt/camera_gt.json").read_text())
            calib.write_text(json.dumps(from_ground_truth(cam_gt, [0]), indent=2))
    anchors = {int(k): v for k, v in json.loads(a.anchors.read_text()).items()} if a.anchors else None
    log("vision: registration, detection, tracking, identity, auto-events (cached)")
    summary = vision.run(footage, calib, clip_dir / "vision", None if a.video else a.match,
                         None if a.video else a.start, anchors, force=a.force,
                         exclude=tuple(tuple(int(v) for v in b.split(",")) for b in a.exclude.split(";") if b))
    log(f"vision summary {summary}")
    cam = CameraTrack.model_validate_json((clip_dir / "vision/homography.json").read_text())
    tracks = ScreenTracks.model_validate_json((clip_dir / "vision/screen_tracks.json").read_text())
    shutil.copyfile(clip_dir / "vision/homography.json", out / "camera.json")
    shutil.copyfile(clip_dir / "vision/screen_tracks.json", out / "screen_tracks.json")

    # 4. agents on the clip window ----------------------------------------
    n = len(cam.homographies)
    clip = interpolate_clip(m.tracking, a.start, n / cam.fps)
    # Real clips: the dataset ball is not in the picture, so the Director gets no ball position.
    screen = ScreenContext(cam, a.start, None if a.video else clip.ball[:, :2])
    log("agents: clip window (screen-aware Director)")
    eng = PitchLensEngine(m.meta, m.events, m.tracking, screen=screen, orient_team=a.orient_team)
    res = eng.run(t_from=a.start, t_to=a.start + n / cam.fps, recap=False)
    by_id = {e.id: e for e in m.events}
    for k, tl in res.timelines.items():
        if a.video:
            moved = retarget_timeline(tl, tracks, by_id, a.start, cam.fps)
            log(f"  {k}: retargeted {moved} evidence point(s) onto anchored tracks")
        if a.no_score_bug:
            tl.items = [i for i in tl.items if i.type.value != "score_bug"]
        (out / "timelines" / f"{k}.json").write_text(tl.model_dump_json())
    win = [t.model_dump() for t in res.trace.traces if a.start - 240 <= t.t_match <= a.start + a.duration + 5]
    (out / "traces.json").write_text(json.dumps(win))
    keep = {c.id for v in res.shown.values() for c in v}
    cands = [c.model_dump() for c in res.candidates if c.id in keep or c.t >= a.start - 240]
    log(f"clip: {sum(len(v) for v in res.shown.values())} insights shown across {len(res.shown)} profiles")

    # 5. full match: recap + metrics ---------------------------------------
    log("agents: full match (recap, metrics)")
    full = PitchLensEngine(m.meta, m.events, m.tracking, keep_tool_traces=False).run()
    cands += [c.model_dump() for c in full.candidates if c.id not in {x["id"] for x in cands}]
    (out / "candidates.json").write_text(json.dumps(cands, default=str))
    metrics = {**full.metrics, "clip_latency_ms_p99": res.metrics["latency_ms_p99"],
               "vision": json.loads((clip_dir / "vision/summary.json").read_text())}
    for name in ("registration_eval.json", "vision_eval.json"):
        p = clip_dir / "vision" / name
        if p.exists():
            metrics[name.split(".")[0]] = json.loads(p.read_text())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    assert full.recap is not None
    (out / "recap.json").write_text(full.recap.model_dump_json(indent=2))
    render_recap_cards(m, full.recap, media)

    # 6. burned-in renders + alpha stream ----------------------------------
    burned = {}
    for prof in [p for p in a.burn.split(",") if p]:
        dst = media / f"burned_{prof}.mp4"
        if a.force or not dst.exists():
            log(f"burn-in render: {prof}")
            tl = OverlayTimeline.model_validate_json((out / "timelines" / f"{prof}.json").read_text())
            info = render_video(footage, tl, cam, m.meta, tracks, composite_out=dst)
            log(f"  {info}")
        burned[prof] = f"/media/{dst.name}"
    alpha = None
    if a.alpha:
        dst = media / f"overlay_alpha_{a.alpha}.webm"
        if a.force or not dst.exists():
            log(f"alpha overlay stream (WebM VP9 yuva420p): {a.alpha}")
            tl = OverlayTimeline.model_validate_json((out / "timelines" / f"{a.alpha}.json").read_text())
            render_video(footage, tl, cam, m.meta, tracks, alpha_out=dst, png_dir=media / "overlay_png", png_every=60)
        alpha = f"/media/{dst.name}"
    if a.video is None:
        vid = "/media/footage.mp4"
    else:
        shutil.copyfile(a.video, media / "footage.mp4")
        vid = "/media/footage.mp4"
    webm = media / "footage.webm"
    if a.force or not webm.exists():  # VP9 copy for browsers without H.264 (e.g. open-source Chromium)
        import subprocess

        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(media / "footage.mp4"), "-c:v", "libvpx-vp9",
                        "-b:v", "0", "-crf", "34", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                        str(webm)], check=True)
    manifest = {
        "match_id": m.meta.match_id, "video": vid, "video_webm": "/media/footage.webm", "fps": cam.fps, "frames": n, "duration_s": n / cam.fps,
        "match_t0": a.start, "profiles": sorted(res.timelines), "languages": eng.languages, "burned": burned,
        "alpha_overlay": alpha, "camera_source": cam.source, "footage": "synthetic" if a.video is None else "user clip",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"done -> {out}")


def render_recap_cards(m: SyntheticMatch, recap, media: Path) -> None:
    """Shareable 1920x1080 recap card per language, rendered by the broadcast compositor."""
    r = OverlayRenderer(m.meta)
    bg = np.zeros((1080, 1920, 3), np.uint8)
    bg[:] = (40, 14, 10)
    for lang in recap.headline:
        item = RecapCard(id=f"recap-{lang}", t_start=0, t_end=10,
                         text={lang: TextBlock(headline=recap.headline[lang], body=recap.summary[lang])},
                         moments=[TextBlock(label=f"{mo.minute}'", headline=mo.text[lang]) for mo in recap.moments[:5]])
        tl = OverlayTimeline(match_id=m.meta.match_id, profile_id="recap", languages=[lang], default_language=lang,
                             media=MediaClock(duration_s=10, match_t_at_media_zero=0), items=[item])
        dummy = CameraTrack(fps=30, width=1920, height=1080, homographies=[np.eye(3).ravel().tolist()])
        img = composite(bg, r.render_layer(tl, dummy, 60, None, lang))
        cv2.imwrite(str(media / f"recap_{lang}.png"), img)


if __name__ == "__main__":
    main()
