"""Rolling match state (stage 2: interpret).

``MatchState`` is updated incrementally, one event (and optionally one
tracking frame) at a time, so it works identically for file replay and for a
live stream. It keeps:

* score, clock, possession sequences,
* per-player counters and the evidence (event ids + locations) behind them,
* rolling team windows for rhythm metrics: PPDA, turnover rate, possession
  sequence length, field tilt, xG momentum,
* physical metrics from tracking: distance, sprints, top speed.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from pitchlens.schema import PITCH_LENGTH, Event, EventType, MatchMeta, Outcome, attack_x, third

SPRINT_KMH = 25.0
SPRINT_MIN_S = 1.0
# Standard PPDA denominator: tackles, interceptions and fouls (loose-ball recoveries excluded).
DEFENSIVE_ACTIONS = {EventType.TACKLE, EventType.INTERCEPTION, EventType.FOUL}


@dataclass
class Possession:
    id: int
    team_id: str
    t_start: float
    t_end: float
    passes: int = 0
    completed: int = 0
    event_ids: list[str] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)
    end_reason: str = ""


@dataclass
class PlayerPhysical:
    distance_m: float = 0.0
    sprints: int = 0
    top_speed_kmh: float = 0.0
    top_speed_t: float = 0.0
    _sprint_run_s: float = 0.0
    _in_sprint: bool = False
    last_speed_kmh: float = 0.0
    last_xy: tuple[float, float] = (0.0, 0.0)


class MatchState:
    def __init__(self, meta: MatchMeta, window_s: float = 600.0):
        self.meta = meta
        self.window_s = window_s
        self.t = 0.0
        self.period = 1
        self.score = (0, 0)
        self.events: list[Event] = []
        self.by_id: dict[str, Event] = {}
        self.counters: dict[str, Counter] = defaultdict(Counter)
        self.evidence: dict[tuple[str, str], list[Event]] = defaultdict(list)
        self.team_counters: dict[str, Counter] = defaultdict(Counter)
        self.possessions: list[Possession] = []
        self.recent: deque[Event] = deque()
        self.physical: dict[str, PlayerPhysical] = defaultdict(PlayerPhysical)
        self.active_players: set[str] = {p.id for p in meta.all_players() if p.starter}
        self.xg_timeline: list[tuple[float, str, float]] = []
        self._team_of = {p.id: p.team_id for p in meta.all_players()}
        self._names = {p.id: p.name for p in meta.all_players()}

    # --------------------------------------------------------------- helpers
    def attacks_right(self, team_id: str, period: int | None = None) -> bool:
        period = self.period if period is None else period
        return (team_id == self.meta.home.id) == (period == 1)

    def norm_x(self, e: Event, x: float | None = None) -> float:
        x = e.x if x is None else x
        assert x is not None and e.team_id is not None
        return attack_x(x, self.attacks_right(e.team_id, e.period))

    def zone(self, e: Event) -> str:
        return third(self.norm_x(e))

    def opponent(self, team_id: str) -> str:
        return self.meta.away.id if team_id == self.meta.home.id else self.meta.home.id

    def team_of(self, player_id: str) -> str:
        return self._team_of[player_id]

    def name(self, player_id: str) -> str:
        return self._names[player_id]

    def count(self, player_id: str, metric: str) -> int:
        return self.counters[player_id][metric]

    def _add(self, e: Event, metric: str) -> None:
        assert e.player_id is not None
        self.counters[e.player_id][metric] += 1
        self.evidence[(e.player_id, metric)].append(e)
        if e.team_id:
            self.team_counters[e.team_id][metric] += 1

    # ---------------------------------------------------------------- ingest
    def ingest(self, e: Event) -> None:
        self.t = e.t
        self.period = e.period
        self.score = e.score
        self.events.append(e)
        self.by_id[e.id] = e
        self.recent.append(e)
        while self.recent and self.recent[0].t < e.t - self.window_s:
            self.recent.popleft()
        self._possession(e)
        if e.player_id is None or e.x is None:
            if e.type == EventType.GOAL:
                self.score = e.score
            return
        if e.type == EventType.SUBSTITUTION:
            self.active_players.discard(e.player_id)
            if e.recipient_id:
                self.active_players.add(e.recipient_id)
            return
        z = self.zone(e)
        t = e.type
        if t == EventType.PASS:
            self._add(e, "passes")
            ok = e.outcome == Outcome.SUCCESS
            if ok:
                self._add(e, "passes_completed")
            assert e.pass_info is not None
            end_norm = self.norm_x(e, e.end_x)
            if third(end_norm) == "attacking":
                self._add(e, "final_third_passes")
                if ok:
                    self._add(e, "final_third_passes_completed")
            if ok and e.pass_info.progressive_m >= 10 and end_norm > PITCH_LENGTH / 2:
                self._add(e, "progressive_passes")
            if e.pass_info.key_pass:
                self._add(e, "key_passes")
        elif t == EventType.INTERCEPTION:
            self._add(e, "interceptions")
            self._add(e, f"interceptions_{z}_third")
        elif t == EventType.TACKLE:
            self._add(e, "tackles")
            if e.outcome == Outcome.SUCCESS:
                self._add(e, "tackles_won")
                self._add(e, f"tackles_won_{z}_third")
        elif t == EventType.PRESSURE:
            self._add(e, "pressures")
            self._add(e, f"pressures_{z}_third")
        elif t == EventType.BALL_RECOVERY:
            self._add(e, "recoveries")
            self._add(e, f"recoveries_{z}_third")
        elif t == EventType.SHOT:
            self._add(e, "shots")
            assert e.shot_info is not None
            self.counters[e.player_id]["xg_milli"] += int(round(e.shot_info.xg * 1000))
            self.xg_timeline.append((e.t, e.team_id or "", e.shot_info.xg))
        elif t == EventType.GOAL:
            self._add(e, "goals")
        elif t == EventType.CARRY:
            self._add(e, "carries")
        elif t == EventType.DUEL and e.outcome == Outcome.SUCCESS:
            self._add(e, "duels_won")
        elif t == EventType.FOUL:
            self._add(e, "fouls")

    def _possession(self, e: Event) -> None:
        if e.type in (EventType.PERIOD_START, EventType.PERIOD_END, EventType.SUBSTITUTION):
            return
        cur = self.possessions[-1] if self.possessions else None
        if e.team_id is None:
            return
        if cur is None or (e.possession_id != cur.id):
            if cur is not None:
                cur.end_reason = e.type.value
            team = e.team_id
            # A possession is owned by the team in possession, not necessarily the event's team
            if e.type in (EventType.PRESSURE, EventType.FOUL):
                team = self.opponent(e.team_id)
            cur = Possession(e.possession_id, team, e.t, e.t)
            self.possessions.append(cur)
        cur.t_end = e.t
        if e.type == EventType.PASS and e.team_id == cur.team_id:
            cur.passes += 1
            cur.event_ids.append(e.id)
            if e.x is not None and e.y is not None:
                cur.points.append((e.x, e.y))
            if e.outcome == Outcome.SUCCESS:
                cur.completed += 1

    def ingest_tracking(self, t: float, entity: str, x: float, y: float, speed_kmh: float, step_m: float, dt: float) -> None:
        """Feed one tracking sample for one player (10 Hz)."""
        if entity == "ball":
            return
        ph = self.physical[entity]
        ph.distance_m += step_m
        ph.last_speed_kmh = speed_kmh
        ph.last_xy = (x, y)
        if speed_kmh > ph.top_speed_kmh:
            ph.top_speed_kmh, ph.top_speed_t = speed_kmh, t
        if speed_kmh >= SPRINT_KMH:
            ph._sprint_run_s += dt
            if ph._sprint_run_s >= SPRINT_MIN_S and not ph._in_sprint:
                ph._in_sprint = True
                ph.sprints += 1
        else:
            ph._sprint_run_s = 0.0
            ph._in_sprint = False

    # -------------------------------------------------------- rhythm metrics
    def window(self, seconds: float | None = None) -> list[Event]:
        seconds = self.window_s if seconds is None else seconds
        pool = self.recent if seconds <= self.window_s else self.events  # long windows scan the full history
        return [e for e in pool if e.t >= self.t - seconds]

    def ppda(self, team_id: str, seconds: float | None = None) -> float | None:
        """Opponent passes allowed per defensive action in the opponent's first 60% (pressing proxy)."""
        opp = self.opponent(team_id)
        passes = actions = 0
        for e in self.window(seconds):
            if e.x is None or e.team_id is None:
                continue
            if e.team_id == opp and e.type == EventType.PASS and self.norm_x(e) < 0.6 * PITCH_LENGTH:
                passes += 1
            elif e.team_id == team_id and e.type in DEFENSIVE_ACTIONS and self.norm_x(e) > 0.4 * PITCH_LENGTH:
                actions += 1
        if actions == 0:
            return None
        return passes / actions

    def turnovers(self, seconds: float) -> list[Event]:
        return [
            e
            for e in self.window(seconds)
            if e.type in (EventType.INTERCEPTION, EventType.TACKLE) and e.outcome == Outcome.SUCCESS
            or e.type == EventType.BALL_RECOVERY
        ]

    def possession_stats(self, seconds: float) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        cutoff = self.t - seconds
        for tid in (self.meta.home.id, self.meta.away.id):
            ps = [p for p in self.possessions if p.team_id == tid and p.t_end >= cutoff]
            dur = sum(max(p.t_end - p.t_start, 1.0) for p in ps)
            out[tid] = {
                "n": len(ps),
                "avg_passes": float(np.mean([p.passes for p in ps])) if ps else 0.0,
                "time_s": dur,
            }
        total = sum(v["time_s"] for v in out.values()) or 1.0
        for v in out.values():
            v["share"] = v["time_s"] / total
        return out

    def field_tilt(self, team_id: str, seconds: float) -> float | None:
        """Share of all attacking-third passes in the window made by ``team_id``."""
        own = opp = 0
        for e in self.window(seconds):
            if e.type != EventType.PASS or e.x is None:
                continue
            if self.zone(e) == "attacking":
                if e.team_id == team_id:
                    own += 1
                else:
                    opp += 1
        if own + opp < 6:
            return None
        return own / (own + opp)

    def xg_window(self, team_id: str, seconds: float) -> float:
        return sum(x for (t, tid, x) in self.xg_timeline if tid == team_id and t >= self.t - seconds)

    def rhythm(self, seconds: float = 300.0) -> dict[str, float]:
        """Control-vs-chaos summary for the last ``seconds``."""
        ev = self.window(seconds)
        passes = [e for e in ev if e.type == EventType.PASS]
        completion = np.mean([e.outcome == Outcome.SUCCESS for e in passes]) if passes else 0.0
        turnovers = len(self.turnovers(seconds))
        ps = self.possession_stats(seconds)
        avg_seq = np.mean([v["avg_passes"] for v in ps.values()]) if ps else 0.0
        minutes = max(seconds / 60, 1e-6)
        tpm = turnovers / minutes
        # chaos index in [0, 1]: many turnovers, short sequences, low completion.
        chaos = float(np.clip(0.45 * (tpm / 3.0) + 0.35 * (1 - min(avg_seq / 6, 1)) + 0.2 * (1 - completion), 0, 1))
        return {
            "turnovers": turnovers,
            "turnovers_per_min": round(tpm, 2),
            "pass_completion": round(float(completion), 3),
            "avg_passes_per_possession": round(float(avg_seq), 2),
            "passes_per_min": round(len(passes) / minutes, 2),
            "chaos_index": round(chaos, 2),
        }
