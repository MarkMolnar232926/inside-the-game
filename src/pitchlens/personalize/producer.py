"""Personalizer agent: turn a Director decision into overlay instructions for one profile.

Same insight, different experience: the profile decides which overlay types
are used, how dense the card is, which register the text is written in, which
extra metrics an analyst sees, and when player chips (speed) pop up.
Every item carries text for *all* supported languages, so the viewer can
switch language instantly without re-running the agents.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pitchlens.insights.orientation import orient_points
from pitchlens.narrative import templates as T
from pitchlens.schema import (
    Chip,
    Decision,
    Event,
    Grounding,
    InsightCandidate,
    Minimap,
    NarrativeLowerThird,
    OverlayItem,
    PassLine,
    PitchMarkers,
    PitchPoint,
    PlayerHighlight,
    SpeedTag,
    StatCard,
    TextBlock,
)
from pitchlens.state.match_state import MatchState

from ..agents.base import TraceLog
from ..agents.explainer import Explainer
from .profiles import ViewerProfile

# Analyst chips per metric: (label key, fact key, unit)
DENSE_CHIPS: dict[str, list[tuple[str, str, str]]] = {
    "interceptions_attacking_third": [("share", "share_pct", "%"), ("avg_dist_goal", "avg_dist_goal_m", " m")],
    "tackles_won_attacking_third": [("share", "share_pct", "%"), ("avg_dist_goal", "avg_dist_goal_m", " m")],
    "recoveries_attacking_third": [("share", "share_pct", "%"), ("avg_dist_goal", "avg_dist_goal_m", " m")],
    "interceptions_defensive_third": [("share", "share_pct", "%"), ("avg_dist_goal", "avg_dist_goal_m", " m")],
    "shot_xg": [("team_xg", "team_xg", ""), ("distance", "distance_m", " m")],
    "pass_quality": [("difficulty", "difficulty_10", "/10"), ("completion", "completion_pct", "%")],
    "pass_sequence": [("duration", "duration_s", " s"), ("chaos_index", "chaos_index", "")],
    "turnovers_window": [("chaos_index", "chaos_index", ""), ("passes", "avg_passes_per_possession", "")],
    "ppda": [("ppda", "ppda_match", ""), ("passes", "actions", "")],
    "xg_window": [("xg", "xg_against", ""), ("field_tilt", "field_tilt_pct", "%")],
    "sprints": [("distance", "distance_km", " km")],
    "top_speed": [("distance", "distance_km", " km")],
    "passes_completed": [("completion", "completion_pct", "%"), ("progression", "progressive", "")],
}
MARKER_SHAPE = {"pass_sequence": "dot", "turnovers_window": "cross", "ppda": "ring", "xg_window": "ring"}

SpeedLookup = Callable[[str, float, float], list[tuple[float, float]]]  # (player, t0, t1) -> [(t, kmh)]


class Personalizer:
    name = "personalizer"

    def __init__(self, profile: ViewerProfile, explainer: Explainer, trace: TraceLog, languages: list[str],
                 match_t0: float = 0.0, speed_lookup: SpeedLookup | None = None, events: dict[str, Event] | None = None):
        self.p = profile
        self.ex = explainer
        self.trace = trace
        self.languages = languages
        self.t0 = match_t0
        self.speed_lookup = speed_lookup
        self.events = events or {}
        self.n = 0

    def _id(self, kind: str) -> str:
        self.n += 1
        return f"{self.p.id}-{kind}-{self.n:03d}"

    def texts(self, state: MatchState, c: InsightCandidate) -> tuple[dict[str, tuple[str, str]], Grounding]:
        out, grounding = {}, None
        for lang in self.languages:
            ex, g = self.ex.explain(state, c, self.p.style, lang, with_context=self.p.show_context_line)
            out[lang] = (ex.headline, ex.why)
            if grounding is None or not g.verified:
                grounding = g
        assert grounding is not None
        return out, grounding

    def build(self, state: MatchState, d: Decision, c: InsightCandidate, t_match: float) -> list[OverlayItem]:
        t = t_match - self.t0
        dur = d.duration_s
        vis = set(c.visual) & set(self.p.visuals)
        texts, grounding = self.texts(state, c)
        items: list[OverlayItem] = []
        f = c.evidence.facts
        common = dict(insight_id=c.id, evidence_event_ids=c.evidence.event_ids[-12:], grounding=grounding)
        points, flipped = orient_points(c.evidence.points[-8:], self.events, state.period)
        note = {lang: T.label("orientation_note", lang) for lang in self.languages} if flipped else {}
        if "stat_card" in vis:
            player = state.meta.player(c.player_id) if c.player_id else None
            team = state.meta.team(c.team_id) if c.team_id else None
            chips = []
            if self.p.density == "dense":
                for lab, key, unit in DENSE_CHIPS.get(c.metric, []):
                    if f.get(key) is not None:
                        chips.append(Chip(label=lab, value=T.fmt_number(f[key], "en"), unit=unit))
            items.append(StatCard(
                id=self._id("card"), t_start=t, t_end=t + dur, value=c.value_text, player_id=c.player_id,
                team_id=c.team_id, player_number=player.number if player else None,
                player_name=player.name if player else None, team_name=team.name if team else None,
                placement=d.placement, density=self.p.density if player or self.p.density == "dense" else "minimal",
                extra_metrics=chips, priority=8,
                text={lang: TextBlock(headline=h) for lang, (h, _) in texts.items()}, **common,
            ))
        if "pitch_markers" in vis and points:
            stagger = min(0.45, (dur * 0.45) / max(len(points), 1))
            items.append(PitchMarkers(
                id=self._id("markers"), t_start=t + 0.3, t_end=t + dur,
                points=[p.model_copy(update={"appear_at": round(i * stagger, 2)}) for i, p in enumerate(points)],
                shape=MARKER_SHAPE.get(c.metric, "star"), numbered=c.metric not in ("pass_sequence",),
                connect=c.metric == "pass_sequence", **common,
            ))
        if "player_highlight" in vis and c.player_id:
            items.append(self.highlight(state, c.player_id, t, t + dur, common))
        if "pass_line" in vis and c.metric == "pass_quality" and len(c.evidence.points) >= 2:
            a, b = c.evidence.points[0], c.evidence.points[1]
            acc = "perfect" if f.get("completion_pct", 100) < 55 else "accurate"
            items.append(PassLine(
                id=self._id("pass"), t_start=t, t_end=t + dur, start=a, end=b, distance_m=float(f["distance_m"]),
                difficulty=min(float(f["difficulty_10"]) / 10, 1.0), accuracy=acc,  # type: ignore[arg-type]
                ball_speed_kmh=f.get("ball_speed_kmh"), player_id=c.player_id,
                text={lang: TextBlock(label=f"{f['distance_m']} m · {T.fmt_number(f['ball_speed_kmh'], lang)} km/h")
                      for lang in self.languages}, **common,
            ))
        if "speed_tag" in vis and c.metric == "shot_xg" and c.evidence.points:
            sp = c.evidence.points[0]
            items.append(SpeedTag(
                id=self._id("shotspeed"), t_start=t, t_end=t + min(dur, 5), subject="ball",
                speed_kmh=float(f["shot_speed_kmh"]), style={"x": str(sp.x), "y": str(sp.y)},
                text={lang: TextBlock(label=T.label("shot", lang)) for lang in self.languages}, **common,
            ))
        if "speed_tag" in vis and c.metric == "top_speed" and c.player_id:
            items.append(SpeedTag(
                id=self._id("topspeed"), t_start=t, t_end=t + dur, subject="player", player_id=c.player_id,
                speed_kmh=float(f["speed_kmh"]),
                text={lang: TextBlock(label=T.label("top_speed", lang)) for lang in self.languages}, **common,
            ))
        if "narrative_lower_third" in vis and any(w for _, w in texts.values()):
            items.append(NarrativeLowerThird(
                id=self._id("lower"), t_start=t + 0.6, t_end=t + dur, tone=c.theme, priority=6,
                text={lang: TextBlock(label=T.label("why_it_matters", lang),
                                      body=w + (f" ({note[lang]})" if note and self.p.show_orientation_note else ""))
                      for lang, (_, w) in texts.items()}, **common,
            ))
        if "minimap" in vis and self.p.density == "dense":
            items.append(Minimap(id=self._id("minimap"), t_start=t, t_end=t + dur, points=points,
                                 focus_player_id=c.player_id, **common))
        self.trace.add(self.name, t_match, "build",
                       f"{self.p.id}: {c.metric} -> {', '.join(sorted(i.type.value for i in items))}",
                       output={"items": [i.id for i in items], "style": self.p.style, "density": self.p.density},
                       input_ref=c.id)
        return items

    def highlight(self, state: MatchState, player_id: str, t_start: float, t_end: float, common: dict | None = None,
                  ) -> PlayerHighlight:
        p = state.meta.player(player_id)
        chips = []
        if self.speed_lookup is not None:
            series = self.speed_lookup(player_id, t_start + self.t0, t_end + self.t0)
            over = [(tt, v) for tt, v in series if v >= self.p.speed_threshold_kmh]
            if over:
                peak = max(v for _, v in series)
                chips.append(Chip(label="speed", value=f"{peak:.1f}", unit="km/h",
                                  appear_at=round(max(over[0][0] - self.t0 - t_start, 0.0), 2)))
        return PlayerHighlight(id=self._id("highlight"), t_start=t_start, t_end=t_end, player_id=player_id,
                               player_number=p.number, display_name=p.short_name, chips=chips, **(common or {}))


def tracking_speed_lookup(tracking) -> SpeedLookup:
    """Speed series per player from the synthetic 10 Hz tracking (or vision tracks in pitch metres)."""
    by_entity = {e: g[["t", "speed_kmh"]].to_numpy() for e, g in tracking[tracking.entity != "ball"].groupby("entity")}

    def lookup(player_id: str, t0: float, t1: float) -> list[tuple[float, float]]:
        arr = by_entity.get(player_id)
        if arr is None:
            return []
        m = (arr[:, 0] >= t0) & (arr[:, 0] <= t1)
        return [(float(a), float(b)) for a, b in arr[m]]

    return lookup


def peak(series: list[tuple[float, float]]) -> float:
    return float(np.max([v for _, v in series])) if series else 0.0


_ = PitchPoint  # re-exported for type hints in callers
