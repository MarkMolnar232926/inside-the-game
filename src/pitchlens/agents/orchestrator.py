"""PitchLens orchestrator: the five-stage pipeline with eight cooperating agents.

    ingest ─► Stream Watcher ─► Insight Scout ─► Director (per profile) ─► Explainer ─► Verifier
                 (state)          (detectors)       (what/when/where)        (why)       (grounding)
                                                         │
                                                         ▼
                                     Personalizer (+ Translator/Voice) ─► overlay timeline (JSON)
                                                                                 │
                                                       Recap agent (full time) ◄─┘

One shared interpretation (watcher + scout + explainer cache) feeds any
number of viewer profiles, each with its own Director and Personalizer, so
the same intelligence yields different experiences at almost no extra cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pitchlens.ingest.sources import ReplaySource, StreamItem, Tick
from pitchlens.narrative.templates import SUPPORTED
from pitchlens.schema import (
    Event,
    EventType,
    InsightCandidate,
    MatchMeta,
    MediaClock,
    OverlayItem,
    OverlayTimeline,
    ScoreBug,
)

from ..azure.openai_client import LLMClient, default_llm
from ..personalize.producer import Personalizer, tracking_speed_lookup
from ..personalize.profiles import ViewerProfile, builtin_profiles
from .base import TraceLog
from .director import Director, ScreenContext
from .explainer import Explainer
from .recap import Recap, RecapAgent
from .verifier import Verifier
from .watcher import InsightScout, StreamWatcher


@dataclass
class EngineResult:
    timelines: dict[str, OverlayTimeline]
    trace: TraceLog
    candidates: list[InsightCandidate]
    shown: dict[str, list[InsightCandidate]]
    latency_ms: list[float]
    recap: Recap | None = None
    metrics: dict = field(default_factory=dict)


class PitchLensEngine:
    def __init__(self, meta: MatchMeta, events: list[Event], tracking: pd.DataFrame | None = None,
                 profiles: dict[str, ViewerProfile] | None = None, languages: list[str] | None = None,
                 llm: LLMClient | None = None, screen: ScreenContext | None = None, keep_tool_traces: bool = True):
        self.meta = meta
        self.events = events
        self.tracking = tracking
        self.profiles = profiles or builtin_profiles()
        self.languages = languages or SUPPORTED
        self.trace = TraceLog(keep_tool_calls=keep_tool_traces)
        self.verifier = Verifier(meta)
        self.explainer = Explainer(self.verifier, self.trace, llm or default_llm())
        self.screen = screen
        self.watcher = StreamWatcher(meta)
        self.scout = InsightScout()
        self.by_id = {e.id: e for e in events}
        self.speed_lookup = tracking_speed_lookup(tracking) if tracking is not None else None
        self._reset_profiles(0.0)

    def _reset_profiles(self, match_t0: float) -> None:
        self.directors = {k: Director(p, self.trace, self.screen) for k, p in self.profiles.items()}
        self.personalizers = {
            k: Personalizer(p, self.explainer, self.trace, self.languages, match_t0, self.speed_lookup, self.by_id)
            for k, p in self.profiles.items()
        }
        self.items: dict[str, list[OverlayItem]] = {k: [] for k in self.profiles}
        self.shown: dict[str, list[InsightCandidate]] = {k: [] for k in self.profiles}
        self.match_t0 = match_t0

    # ------------------------------------------------------------- stepping
    def step(self, item: StreamItem, active: bool = True) -> dict[str, list[OverlayItem]]:
        """Process one stream item; returns new overlay items per profile (live mode)."""
        st = self.watcher.state
        is_event = self.watcher.ingest(item)
        cands: list[InsightCandidate] = []
        if is_event:
            assert isinstance(item, Event)
            cands += self.scout.on_event(st, item)
        cands += self.scout.on_time(st)
        for c in cands:
            self.trace.metrics["insights_proposed"] += 1
            self.trace.add(self.scout.name, c.t, "propose",
                           f"{c.detector}: {c.value_text} ({c.player_id or c.team_id or 'match'}) base {c.base_score:.2f}",
                           output={"metric": c.metric, "evidence": len(c.evidence.event_ids)}, input_ref=c.id)
            self.candidates.append(c)
        out: dict[str, list[OverlayItem]] = {}
        if not active:
            return out
        for k, d in self.directors.items():
            if cands:
                d.offer(cands)
            for dec, c in d.decide(st, st.t):
                if dec.preempts:  # cut the card on air: end its graphics now (they animate out)
                    t_media = st.t - self.match_t0
                    for it in self.items[k]:
                        if it.insight_id == dec.preempts and it.t_end > t_media + 0.45:
                            it.t_end = t_media + 0.45
                items = self.personalizers[k].build(st, dec, c, st.t)
                self.items[k].extend(items)
                self.shown[k].append(c)
                out[k] = out.get(k, []) + items
        return out

    def run(self, t_from: float = 0.0, t_to: float | None = None, recap: bool = True, tick_every: int = 5,
            media_offset: float | None = None) -> EngineResult:
        """Replay the match from kick-off (state needs history) but only *show* graphics in [t_from, t_to]."""
        t_end = t_to if t_to is not None else self.meta.duration_s + 600
        self.candidates: list[InsightCandidate] = []
        self._reset_profiles(t_from if media_offset is None else media_offset)
        latency: list[float] = []
        src = ReplaySource(self.events, self.tracking, tick_every=tick_every, t_to=t_end)
        warm = t_from - 2.0
        for item in src:
            active = item.t >= warm
            t0 = time.perf_counter()
            if isinstance(item, Tick) and item.t < warm - 30:
                self.watcher.state.t = item.t  # physical stats only matter close to the window; skip early ticks
                for i, ent in enumerate(item.entities):
                    self.watcher.state.ingest_tracking(item.t, str(ent), float(item.x[i]), float(item.y[i]),
                                                       float(item.speed_kmh[i]), float(item.step_m[i]), item.dt)
                continue
            self.step(item, active=active and item.t >= t_from)
            if isinstance(item, Event) and active:
                latency.append((time.perf_counter() - t0) * 1000)
        timelines = {k: self.timeline(k, t_from, t_end) for k in self.profiles}
        rc = None
        if recap and t_to is None:
            rc = RecapAgent(self.verifier, self.trace, self.languages).run(self.watcher.state,
                                                                           self.shown.get("analyst", []))
        lat = np.array(latency) if latency else np.zeros(1)
        metrics = {
            **dict(self.trace.metrics),
            "events": self.watcher.n_events,
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
            "latency_ms_p99": round(float(np.percentile(lat, 99)), 3),
            "latency_ms_max": round(float(lat.max()), 3),
        }
        return EngineResult(timelines, self.trace, self.candidates, self.shown, latency, rc, metrics)

    # ------------------------------------------------------------- timeline
    def score_bug(self, t_from: float, t_to: float) -> ScoreBug:
        track = [(0.0, 0, 0)]
        for e in self.events:
            if e.t > t_to:
                break
            if e.type == EventType.GOAL or e.type == EventType.KICK_OFF:
                mt = max(e.t - self.match_t0, 0.0)
                track.append((round(mt, 2), *e.score))
        h, a = self.meta.home, self.meta.away
        return ScoreBug(id="score-bug", t_start=0, t_end=t_to - self.match_t0 + 1, home_short=h.short,
                        away_short=a.short, home_color=h.kit.shirt, away_color=a.primary, score_track=track,
                        clock_offset=self.match_t0)

    def timeline(self, profile_id: str, t_from: float, t_to: float) -> OverlayTimeline:
        p = self.profiles[profile_id]
        items: list[OverlayItem] = [self.score_bug(t_from, t_to), *self.items[profile_id]]
        if p.focus_player:
            # Player cam: an always-on highlight with threshold-triggered speed chips.
            pers = self.personalizers[profile_id]
            st = self.watcher.state
            dur = t_to - t_from
            for k in range(int(np.ceil(dur / 10))):
                a, b = t_from - self.match_t0 + 10 * k, min(t_from - self.match_t0 + 10 * (k + 1), t_to - self.match_t0)
                items.append(pers.highlight(st, p.focus_player, a, b))
        return OverlayTimeline(
            match_id=self.meta.match_id, profile_id=profile_id, languages=self.languages, default_language=p.language,
            media=MediaClock(duration_s=t_to - t_from, match_t_at_media_zero=self.match_t0), items=items,
        )
