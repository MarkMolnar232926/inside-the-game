"""Align a synthetic event timeline with a real clip.

The real footage is only a visual substrate; the story comes from the synthetic match. To make the two
line up on screen:

1. **Clock**: ``match_t0`` maps media time to match time (chosen so the key synthetic moment falls where
   the real camera shows the right area of the pitch).
2. **Orientation**: evidence points are shown with the featured team attacking towards the end the real
   camera is looking at (``Personalizer(orient_team=...)``).
3. **Retargeting** (this module): events by an *anchored* player (a real track bound to a fictional
   player id) inside the clip window are drawn where that tracked person actually is at that moment.

Retargeted points are flagged in their label (``retargeted``) so nothing pretends to be measured truth.
"""

from __future__ import annotations

import numpy as np

from pitchlens.schema import Event, OverlayTimeline, ScreenTracks


def retarget_timeline(tl: OverlayTimeline, tracks: ScreenTracks, events: dict[str, Event], match_t0: float,
                      fps: float, max_gap_frames: int = 15) -> int:
    """Move evidence points of anchored players' in-clip events onto their tracked pitch positions. Returns count."""
    n_frames = int(tl.media.duration_s * fps)
    moved = 0
    for it in tl.items:
        pts = getattr(it, "points", None)
        if not pts:
            continue
        for p in pts:
            e = events.get(p.event_id or "")
            if e is None or e.player_id not in tracks.tracks:
                continue
            f = int(round((e.t - match_t0) * fps))
            if not 0 <= f < n_frames:
                continue
            tr = tracks.tracks[e.player_id]
            k = int(np.argmin(np.abs(np.asarray(tr.frames) - f)))
            if abs(tr.frames[k] - f) > max_gap_frames:
                continue
            p.x, p.y = float(tr.pitch_xy[k][0]), float(tr.pitch_xy[k][1])
            p.label = (p.label + " retargeted").strip()
            moved += 1
    return moved
