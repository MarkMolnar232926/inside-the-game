"""Deliverable #3: 8 s clip, spatial evidence markers glued to the pitch via the *estimated* homography."""
import json
from pathlib import Path

import numpy as np

from pitchlens.insights.orientation import orient_points
from pitchlens.render.video import render_video
from pitchlens.schema import CameraTrack, MediaClock, OverlayTimeline, PitchMarkers, PitchPoint, apply_h
from pitchlens.state.match_state import MatchState
from pitchlens.synth.generate import SyntheticMatch

clip = Path("outputs/footage/demo30")
m = SyntheticMatch.load(Path("data/synthetic/demo"))
est = CameraTrack.model_validate_json((clip / "vision/homography.json").read_text())
gt = CameraTrack.model_validate_json((clip / "gt/camera_gt.json").read_text())
st = MatchState(m.meta)
for e in m.events:
    if e.t > 5052 + 10:
        break
    st.ingest(e)
att = st.evidence[("KBY-07", "interceptions_attacking_third")]
pts = [PitchPoint(x=e.x, y=e.y, label=str(i + 1), event_id=e.id, appear_at=0.45 * i) for i, e in enumerate(att)]
pts, _ = orient_points(pts, {e.id: e for e in m.events}, 2)
tl = OverlayTimeline(match_id=m.meta.match_id, profile_id="deliverable3", languages=["en"],
                     media=MediaClock(duration_s=8, match_t_at_media_zero=5052),
                     items=[PitchMarkers(id="mk", t_start=0.4, t_end=8.0, points=pts, connect=True)])
out = Path("outputs/deliverables")
info = render_video(clip / "footage.mp4", tl, est, m.meta, composite_out=out / "03_markers_glued.mp4",
                    png_dir=out / "03_frames", png_every=60, frame_range=(0, 240))
# Glue error: estimated vs true image position of each marker centre over the clip.
P = np.array([[p.x, p.y] for p in pts])
err = [np.linalg.norm(apply_h(est.H(i), P) - apply_h(gt.H(i), P), axis=1) for i in range(240)]
err = np.array(err)
rep = {**info, "marker_glue_error_px": {"mean": round(float(err.mean()), 2), "max": round(float(err.max()), 2)}}
(out / "03_markers_glued.json").write_text(json.dumps(rep, indent=2))
print(rep)
