"""Golden-image test for the compositor (deterministic synthetic frame + fixed timeline).

Regenerate goldens after an intentional visual change with:  PITCHLENS_UPDATE_GOLDEN=1 pytest tests/test_renderer_golden.py
"""

import os
from pathlib import Path

import cv2
import numpy as np

from pitchlens.render.compositor import OverlayRenderer, composite, unpremultiply
from pitchlens.schema import (
    CameraTrack,
    Chip,
    MediaClock,
    NarrativeLowerThird,
    OverlayTimeline,
    PitchMarkers,
    PitchPoint,
    ScoreBug,
    StatCard,
    TextBlock,
)
from pitchlens.synth.teams import build_teams
from pitchlens.vision.camera import PinholeCamera

GOLDEN = Path(__file__).with_name("golden")


def _scene():
    from pitchlens.schema import MatchMeta

    h, a = build_teams(4)
    meta = MatchMeta(match_id="G", venue="v", kickoff_utc="", seed=4, narrative="x", duration_s=60, home=h, away=a)
    cam = PinholeCamera.look_at([52.5, -30, 21], [30, 34, 0], 1050, 960, 540)
    camtrack = CameraTrack(fps=30, width=960, height=540, homographies=[cam.ground_homography().ravel().tolist()] * 300)
    frame = np.zeros((540, 960, 3), np.uint8)
    frame[:] = (52, 120, 50)
    p = h.player("KBY-07")
    pts = [PitchPoint(x=x, y=y, appear_at=0.2 * i) for i, (x, y) in enumerate([(20, 20), (28, 30), (12, 14), (30, 12)])]
    tl = OverlayTimeline(match_id="G", profile_id="t", languages=["en"], media=MediaClock(duration_s=10, match_t_at_media_zero=5208),
                         items=[
                             ScoreBug(id="b", t_start=0, t_end=10, home_short="KBY", away_short="IRV", home_color=h.kit.shirt,
                                      away_color=a.primary, score_track=[(0, 2, 1)], clock_offset=5208),
                             StatCard(id="c", t_start=0, t_end=10, value="4", player_id=p.id, team_id="KBY", player_number=7,
                                      player_name=p.name, team_name=h.name, density="dense",
                                      extra_metrics=[Chip(label="share", value="36", unit="%")],
                                      text={"en": TextBlock(headline="Interceptions in attacking third")}),
                             PitchMarkers(id="m", t_start=0, t_end=10, points=pts),
                             NarrativeLowerThird(id="l", t_start=0, t_end=10, tone="pressure",
                                                 text={"en": TextBlock(label="Why it matters", body="Grounded line.")}),
                         ])
    return meta, camtrack, frame, tl


def test_golden_frame():
    meta, cam, frame, tl = _scene()
    layer = OverlayRenderer(meta, 960, 540).render_layer(tl, cam, 90)  # t = 3 s: everything fully animated in
    out = composite(frame, layer)
    path = GOLDEN / "overlay_frame.png"
    if os.getenv("PITCHLENS_UPDATE_GOLDEN") or not path.exists():
        GOLDEN.mkdir(exist_ok=True)
        cv2.imwrite(str(path), out)
    golden = cv2.imread(str(path))
    diff = np.abs(out.astype(int) - golden.astype(int))
    assert diff.mean() < 1.0 and (diff > 40).mean() < 0.005, f"render drifted: mean {diff.mean():.2f}"


def test_alpha_layer_is_transparent_where_empty():
    meta, cam, frame, tl = _scene()
    layer = OverlayRenderer(meta, 960, 540).render_layer(tl, cam, 90)
    rgba = unpremultiply(layer)
    assert rgba[200, 560, 3] == 0  # open grass between graphics: fully transparent
    assert rgba[..., 3].max() == 255


def test_markers_follow_the_camera():
    """Pitch-anchored markers move with the camera; screen graphics do not."""
    meta, cam, frame, tl = _scene()
    r = OverlayRenderer(meta, 960, 540)
    tl.items = [i for i in tl.items if i.type.value == "pitch_markers"]
    cam2 = PinholeCamera.look_at([52.5, -30, 21], [40, 34, 0], 1050, 960, 540)
    moved = CameraTrack(fps=30, width=960, height=540, homographies=[cam2.ground_homography().ravel().tolist()] * 300)
    a = r.render_layer(tl, cam, 90)[..., 3] > 0
    b = r.render_layer(tl, moved, 90)[..., 3] > 0
    ca, cb = np.argwhere(a).mean(axis=0), np.argwhere(b).mean(axis=0)
    assert cb[1] < ca[1] - 40  # camera panned right, so ground-anchored markers slide left in the image
