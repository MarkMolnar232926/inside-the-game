"""Deliverable #2: static reproduction of the reference overlay style on one synthetic frame."""
from pathlib import Path

import cv2
import pandas as pd

from pitchlens.insights.orientation import orient_points
from pitchlens.render.compositor import OverlayRenderer
from pitchlens.schema import (
    CameraTrack,
    Chip,
    MediaClock,
    Minimap,
    NarrativeLowerThird,
    OverlayTimeline,
    PitchMarkers,
    PitchPoint,
    PlayerHighlight,
    ScoreBug,
    StatCard,
    TextBlock,
)
from pitchlens.synth.generate import SyntheticMatch
from pitchlens.vision.screen_tracks import from_ground_truth

clip = Path("outputs/demo/clip")
m = SyntheticMatch.load(Path("data/synthetic/demo"))
cam = CameraTrack.model_validate_json((clip / "gt/camera_gt.json").read_text())
gt = pd.read_parquet(clip / "gt/gt_boxes.parquet")
tracks = from_ground_truth(gt, 30)
t0 = 3721.0
FRAME = 400
ev = {e.id: e for e in m.events}
ints = [e for e in m.events if e.type.value == "interception" and e.player_id == "KBY-07" and e.t <= t0 + FRAME / 30]
from pitchlens.state.match_state import MatchState

st = MatchState(m.meta)
for e in m.events:
    if e.t > t0 + FRAME / 30:
        break
    st.ingest(e)
att = st.evidence[("KBY-07", "interceptions_attacking_third")]
pts = [PitchPoint(x=e.x, y=e.y, label=str(i + 1), event_id=e.id, appear_at=0.4 * i) for i, e in enumerate(att)]
pts, flipped = orient_points(pts, ev, 2)
p = m.meta.player("KBY-07")
T = FRAME / 30 - 3
items = [
    ScoreBug(id="bug", t_start=0, t_end=999, home_short="KBY", away_short="IRV", home_color=m.meta.home.kit.shirt,
             away_color=m.meta.away.primary, score_track=[(0, *st.score)], clock_offset=t0),
    StatCard(id="card", t_start=T, t_end=T + 20, value=str(len(att)), player_id=p.id, team_id=p.team_id,
             player_number=p.number, player_name=p.name, team_name=m.meta.home.name,
             text={"en": TextBlock(headline="Interceptions in attacking third")}),
    PitchMarkers(id="mk", t_start=T, t_end=T + 20, points=pts, connect=False),
    PlayerHighlight(id="hl", t_start=T, t_end=T + 20, player_id=p.id, player_number=p.number, display_name=p.short_name,
                    chips=[Chip(label="Speed", value="27.4", unit="km/h", appear_at=1.0)]),
    Minimap(id="mm", t_start=T, t_end=T + 20, points=pts, focus_player_id=p.id),
    NarrativeLowerThird(id="lt", t_start=T, t_end=T + 20, tone="pressure",
                        text={"en": TextBlock(label="Why it matters", body="Kestane keeps winning the ball high up the pitch: "
                              "3 interceptions in the attacking third, more than anyone tonight.")}),
]
tl = OverlayTimeline(match_id=m.meta.match_id, profile_id="mock", languages=["en"],
                     media=MediaClock(duration_s=30, match_t_at_media_zero=t0), items=items)
cap = cv2.VideoCapture(str(Path("outputs/demo/media/footage.mp4"))); cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME); _, fr = cap.read()
r = OverlayRenderer(m.meta)
out = r.render_frame(fr, tl, cam, FRAME, tracks)
cv2.imwrite("outputs/deliverables/02_style_mock.png", out)
print("flipped", flipped, [(round(q.x,1), round(q.y,1)) for q in pts])
ref = Path("data/test_assets/reference_frame.png")
if ref.exists():  # local-only side-by-side with the (copyrighted) reference; never committed
    a = cv2.resize(cv2.imread(str(ref)), (960, 540)); b = cv2.resize(out, (960, 540))
    cv2.imwrite("outputs/deliverables/02_style_mock_vs_reference.jpg", cv2.hconcat([a, b]))
