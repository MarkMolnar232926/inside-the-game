"""Display orientation of evidence points.

Teams swap ends at half-time. When an insight aggregates events from both
halves (e.g. "4 interceptions in the attacking third"), we show every point in
the *current* attacking direction (rotated 180 degrees for the other half),
which is the usual broadcast convention. Captions say so in analyst mode.
"""

from __future__ import annotations

from pitchlens.schema import PITCH_LENGTH, PITCH_WIDTH, Event, PitchPoint


def orient_point(p: PitchPoint, event_period: int, display_period: int) -> PitchPoint:
    if event_period == display_period:
        return p
    return p.model_copy(update={"x": PITCH_LENGTH - p.x, "y": PITCH_WIDTH - p.y})


def orient_points(points: list[PitchPoint], events: dict[str, Event], display_period: int) -> tuple[list[PitchPoint], bool]:
    out, flipped = [], False
    for p in points:
        e = events.get(p.event_id or "")
        if e is not None and e.period != display_period:
            flipped = True
            out.append(orient_point(p, e.period, display_period))
        else:
            out.append(p)
    return out, flipped
