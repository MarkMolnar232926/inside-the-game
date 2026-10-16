"""Stream Watcher and Insight Scout agents.

The watcher owns :class:`MatchState`; the scout runs detectors and turns
their output into candidates. They are deterministic by design (stats must
be exact); the "agentic" judgement happens downstream in the Director.
"""

from __future__ import annotations

from pitchlens.ingest.sources import StreamItem, Tick
from pitchlens.insights.detectors import Detector, default_detectors
from pitchlens.schema import Event, InsightCandidate, MatchMeta
from pitchlens.state.match_state import MatchState


class StreamWatcher:
    name = "stream_watcher"

    def __init__(self, meta: MatchMeta):
        self.state = MatchState(meta)
        self.n_events = 0
        self.n_ticks = 0

    def ingest(self, item: StreamItem) -> bool:
        """Returns True when the item was an event (scout should run event detectors)."""
        if isinstance(item, Tick):
            self.n_ticks += 1
            st = self.state
            st.t = max(st.t, item.t)
            for i, ent in enumerate(item.entities):
                st.ingest_tracking(item.t, str(ent), float(item.x[i]), float(item.y[i]),
                                   float(item.speed_kmh[i]), float(item.step_m[i]), item.dt)
            return False
        self.n_events += 1
        self.state.ingest(item)
        return True


class InsightScout:
    name = "insight_scout"

    def __init__(self, detectors: list[Detector] | None = None, tick_s: float = 5.0):
        self.detectors = detectors or default_detectors()
        self.tick_s = tick_s
        self._last_tick = -1e9

    def on_event(self, state: MatchState, e: Event) -> list[InsightCandidate]:
        out: list[InsightCandidate] = []
        for d in self.detectors:
            out.extend(d.on_event(state, e))
        return out

    def on_time(self, state: MatchState) -> list[InsightCandidate]:
        if state.t - self._last_tick < self.tick_s:
            return []
        self._last_tick = state.t
        out: list[InsightCandidate] = []
        for d in self.detectors:
            out.extend(d.on_tick(state))
        return out
