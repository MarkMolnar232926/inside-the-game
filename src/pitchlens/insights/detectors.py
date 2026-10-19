"""Deterministic insight detectors (Insight Scout's toolbox).

Each detector inspects :class:`MatchState` after an event (``on_event``) or on
a periodic tick (``on_tick``) and proposes :class:`InsightCandidate` objects
with *evidence*: event ids, pitch coordinates, and a ``facts`` dict that
holds every number and name the narrative layer is allowed to mention.
"""

from __future__ import annotations

from collections.abc import Iterable

from pitchlens.schema import Event, EventType, Evidence, InsightCandidate, Outcome, PitchPoint
from pitchlens.state.match_state import MatchState


def _pt(e: Event, label: str = "", use_end: bool = False) -> PitchPoint:
    x, y = (e.end_x, e.end_y) if use_end else (e.x, e.y)
    return PitchPoint(x=float(x or 0), y=float(y or 0), label=label, event_id=e.id)


def player_facts(s: MatchState, player_id: str) -> dict:
    p = s.meta.player(player_id)
    team = s.meta.team(p.team_id)
    return {
        "player_name": p.name,
        "player_short": p.short_name.title(),
        "player_number": p.number,
        "player_role": p.role,
        "team_name": team.name,
        "team_short": team.short,
    }


def team_facts(s: MatchState, team_id: str) -> dict:
    team = s.meta.team(team_id)
    opp = s.meta.team(s.opponent(team_id))
    return {"team_name": team.name, "team_short": team.short, "opponent_name": opp.name, "opponent_short": opp.short}


class Detector:
    name = "base"

    def on_event(self, s: MatchState, e: Event) -> Iterable[InsightCandidate]:
        return ()

    def on_tick(self, s: MatchState) -> Iterable[InsightCandidate]:
        return ()

    def _id(self, s: MatchState, key: str) -> str:
        return f"{self.name}:{key}:{int(s.t)}"


class ZoneActionCount(Detector):
    """'N <actions> in <zone> third' for a single player (e.g. interceptions in the attacking third)."""

    def __init__(self, metric: str, event_type: EventType, label: str, threshold: int, theme: str = "pressure"):
        self.metric = metric
        self.event_type = event_type
        self.label = label
        self.threshold = threshold
        self.theme = theme
        self.name = metric
        # High turnovers (won near the opponent's goal) are worth more than deep regains.
        self.zone_value = 1.15 if "attacking" in metric else 0.85 if "defensive" in metric else 1.0

    def on_event(self, s: MatchState, e: Event) -> Iterable[InsightCandidate]:
        if e.type != self.event_type or e.player_id is None or e.outcome == Outcome.FAIL:
            return
        n = s.count(e.player_id, self.metric)
        if n < self.threshold:
            return
        evs = s.evidence[(e.player_id, self.metric)]
        if not evs or evs[-1].id != e.id:
            return
        team_total = s.team_counters[e.team_id or ""][self.metric]
        best_other = max((c[self.metric] for pid, c in s.counters.items() if pid != e.player_id), default=0)
        goal_x = [105.0 if s.attacks_right(x.team_id or "", x.period) else 0.0 for x in evs]
        dists = [((gx - (x.x or 0)) ** 2 + (34 - (x.y or 0)) ** 2) ** 0.5 for gx, x in zip(goal_x, evs, strict=True)]
        facts = player_facts(s, e.player_id) | {
            "value": n,
            "metric_label": self.label,
            "team_total": team_total,
            "share_pct": round(100 * n / max(team_total, 1)),
            "rank": 1 if n > best_other else 2,
            "avg_dist_goal_m": round(sum(dists) / len(dists)),
            "minute": e.minute,
            "minutes": [x.minute for x in evs],
        }
        yield InsightCandidate(
            id=self._id(s, e.player_id),
            detector=self.name,
            metric=self.metric,
            t=e.t,
            team_id=e.team_id,
            player_id=e.player_id,
            value=n,
            value_text=str(n),
            template_key=self.metric,
            theme=self.theme,  # type: ignore[arg-type]
            base_score=min((0.5 + 0.08 * (n - self.threshold) + 0.1 * (n / max(team_total, 1))) * self.zone_value, 0.95),
            visual=["stat_card", "pitch_markers", "player_highlight", "narrative_lower_third", "minimap"],
            evidence=Evidence(
                event_ids=[x.id for x in evs],
                points=[_pt(x, str(i + 1)) for i, x in enumerate(evs)],
                facts=facts,
            ),
        )


class PassOfTheMatch(Detector):
    """A completed, difficult, progressive pass (pass-quality visual)."""

    name = "quality_pass"

    def __init__(self) -> None:
        self.best = 0.0

    def on_event(self, s: MatchState, e: Event) -> Iterable[InsightCandidate]:
        if e.type != EventType.PASS or e.outcome != Outcome.SUCCESS or e.pass_info is None:
            return
        pi = e.pass_info
        if pi.length_m < 20 or pi.progressive_m < 12 or pi.difficulty < 0.25:
            return
        quality = pi.difficulty * 0.6 + min(pi.length_m / 50, 1) * 0.25 + min(pi.progressive_m / 40, 1) * 0.15
        if quality < self.best * 0.92:
            return
        self.best = max(self.best, quality)
        assert e.player_id and e.recipient_id
        diff10 = round(pi.difficulty * 10, 1)
        facts = player_facts(s, e.player_id) | {
            "value": round(pi.length_m),
            "distance_m": round(pi.length_m),
            "difficulty_10": diff10,
            "completion_pct": round(pi.expected_completion * 100),
            "ball_speed_kmh": round(pi.ball_speed_kmh),
            "recipient_name": s.name(e.recipient_id),
            "recipient_short": s.meta.player(e.recipient_id).short_name.title(),
            "progressive_m": round(pi.progressive_m),
            "minute": e.minute,
        }
        yield InsightCandidate(
            id=self._id(s, e.id),
            detector=self.name,
            metric="pass_quality",
            t=e.t,
            team_id=e.team_id,
            player_id=e.player_id,
            value=round(pi.length_m),
            value_text=f"{round(pi.length_m)}m",
            template_key="quality_pass",
            theme="quality",
            base_score=min(0.45 + quality * 0.5, 0.9),
            visual=["pass_line", "narrative_lower_third"],
            evidence=Evidence(event_ids=[e.id], points=[_pt(e, "start"), _pt(e, "end", use_end=True)], facts=facts),
            expires_after_s=25,
        )


class BigChance(Detector):
    name = "big_chance"

    def on_event(self, s: MatchState, e: Event) -> Iterable[InsightCandidate]:
        if e.type != EventType.SHOT or e.shot_info is None or e.player_id is None:
            return
        si = e.shot_info
        if si.xg < 0.12 and si.speed_kmh < 105:
            return
        goal = e.outcome == Outcome.SUCCESS
        facts = player_facts(s, e.player_id) | {
            "value": round(si.xg, 2),
            "xg": round(si.xg, 2),
            "shot_speed_kmh": round(si.speed_kmh),
            "distance_m": round(si.distance_m),
            "goal": goal,
            "xg_pct": round(si.xg * 100),
            "minute": e.minute,
            "team_xg": round(sum(x for (_, tid, x) in s.xg_timeline if tid == e.team_id), 2),
        }
        yield InsightCandidate(
            id=self._id(s, e.id),
            detector=self.name,
            metric="shot_xg",
            t=e.t,
            team_id=e.team_id,
            player_id=e.player_id,
            value=round(si.xg, 2),
            value_text=f"{si.xg:.2f}",
            template_key="goal" if goal else "big_chance",
            theme="momentum",
            base_score=0.95 if goal else min(0.5 + si.xg, 0.85),
            visual=["speed_tag", "stat_card", "narrative_lower_third"],
            evidence=Evidence(event_ids=[e.id], points=[_pt(e, "shot")], facts=facts),
            expires_after_s=30,
        )


class PossessionSpell(Detector):
    """Long passing sequences = control."""

    name = "control_spell"

    def on_event(self, s: MatchState, e: Event) -> Iterable[InsightCandidate]:
        if e.type != EventType.PASS or not s.possessions:
            return
        p = s.possessions[-1]
        if p.passes not in (10, 15, 20) or e.team_id != p.team_id:
            return
        rh = s.rhythm(300)
        facts = team_facts(s, p.team_id) | {
            "value": p.passes,
            "duration_s": round(p.t_end - p.t_start),
            "minute": e.minute,
            "chaos_index": rh["chaos_index"],
            "avg_passes_per_possession": rh["avg_passes_per_possession"],
        }
        yield InsightCandidate(
            id=self._id(s, f"{p.id}-{p.passes}"),
            detector=self.name,
            metric="pass_sequence",
            t=e.t,
            team_id=p.team_id,
            value=p.passes,
            value_text=str(p.passes),
            template_key="control_spell",
            theme="control",
            base_score=0.45 + 0.02 * p.passes,
            visual=["pitch_markers", "stat_card", "narrative_lower_third"],
            evidence=Evidence(
                event_ids=list(p.event_ids),
                points=[PitchPoint(x=x, y=y, label=str(i + 1), event_id=eid)
                        for i, ((x, y), eid) in enumerate(zip(p.points, p.event_ids, strict=False))],
                facts=facts,
            ),
            expires_after_s=30,
        )


class ChaosSpike(Detector):
    """Burst of turnovers: the game has lost its shape."""

    name = "chaos_spike"
    window = 180.0

    def __init__(self) -> None:
        self.last_t = -1e9

    def on_tick(self, s: MatchState) -> Iterable[InsightCandidate]:
        if s.t - self.last_t < 300:
            return
        tos = s.turnovers(self.window)
        if len(tos) < 9:
            return
        self.last_t = s.t
        rh = s.rhythm(self.window)
        facts = {
            "value": len(tos),
            "window_min": int(self.window // 60),
            "chaos_index": rh["chaos_index"],
            "avg_passes_per_possession": rh["avg_passes_per_possession"],
            "minute": int(s.t // 60) + 1,
            "team_name": s.meta.home.name,
            "opponent_name": s.meta.away.name,
        }
        yield InsightCandidate(
            id=self._id(s, "match"),
            detector=self.name,
            metric="turnovers_window",
            t=s.t,
            value=len(tos),
            value_text=str(len(tos)),
            template_key="chaos_spike",
            theme="chaos",
            base_score=min(0.45 + 0.03 * (len(tos) - 9), 0.8),
            visual=["pitch_markers", "stat_card", "narrative_lower_third"],
            evidence=Evidence(event_ids=[e.id for e in tos], points=[_pt(e) for e in tos], facts=facts),
            expires_after_s=60,
        )


class PressingSurge(Detector):
    """Rolling PPDA falls well below the team's match average."""

    name = "pressing_surge"

    def __init__(self) -> None:
        self.last_t: dict[str, float] = {}

    def on_tick(self, s: MatchState) -> Iterable[InsightCandidate]:
        if s.t < 900:
            return
        for tid in (s.meta.home.id, s.meta.away.id):
            if s.t - self.last_t.get(tid, -1e9) < 900:
                continue
            recent = s.ppda(tid, 600)
            full = s.ppda(tid, 1e9)
            if recent is None or full is None or recent > 0.65 * full or recent > 8:
                continue
            self.last_t[tid] = s.t
            acts = [
                e
                for e in s.window(600)
                if e.team_id == tid
                and e.type in (EventType.TACKLE, EventType.INTERCEPTION, EventType.BALL_RECOVERY)
                and e.outcome != Outcome.FAIL
                and e.x is not None
                and s.norm_x(e) > 0.4 * 105
            ]
            facts = team_facts(s, tid) | {
                "value": round(recent, 1),
                "ppda_recent": round(recent, 1),
                "ppda_match": round(full, 1),
                "window_min": 10,
                "minute": int(s.t // 60) + 1,
                "actions": len(acts),
            }
            yield InsightCandidate(
                id=self._id(s, tid),
                detector=self.name,
                metric="ppda",
                t=s.t,
                team_id=tid,
                value=round(recent, 1),
                value_text=f"{recent:.1f}",
                template_key="pressing_surge",
                theme="pressure",
                base_score=min(0.5 + (full - recent) / max(full, 1) * 0.5, 0.85),
                visual=["pitch_markers", "stat_card", "narrative_lower_third"],
                evidence=Evidence(event_ids=[e.id for e in acts], points=[_pt(e) for e in acts], facts=facts),
                expires_after_s=120,
            )


class MomentumSwing(Detector):
    """xG over the last 10 minutes flips decisively to one side."""

    name = "momentum_swing"

    def __init__(self) -> None:
        self.leader: str | None = None
        self.last_t = -1e9

    def on_tick(self, s: MatchState) -> Iterable[InsightCandidate]:
        h, a = s.meta.home.id, s.meta.away.id
        xh, xa = s.xg_window(h, 600), s.xg_window(a, 600)
        if abs(xh - xa) < 0.35:
            return
        leader = h if xh > xa else a
        if leader == self.leader or s.t - self.last_t < 600:
            return
        self.leader, self.last_t = leader, s.t
        tilt = s.field_tilt(leader, 600)
        facts = team_facts(s, leader) | {
            "value": round(max(xh, xa), 2),
            "xg_for": round(max(xh, xa), 2),
            "xg_against": round(min(xh, xa), 2),
            "window_min": 10,
            "field_tilt_pct": None if tilt is None else round(tilt * 100),
            "minute": int(s.t // 60) + 1,
        }
        shots = [e for e in s.window(600) if e.type == EventType.SHOT and e.team_id == leader]
        yield InsightCandidate(
            id=self._id(s, leader),
            detector=self.name,
            metric="xg_window",
            t=s.t,
            team_id=leader,
            value=round(max(xh, xa), 2),
            value_text=f"{max(xh, xa):.2f}",
            template_key="momentum_swing",
            theme="momentum",
            base_score=min(0.5 + abs(xh - xa), 0.9),
            visual=["stat_card", "minimap", "narrative_lower_third"],
            evidence=Evidence(event_ids=[e.id for e in shots], points=[_pt(e, "shot") for e in shots], facts=facts),
            expires_after_s=120,
        )


class PhysicalMilestones(Detector):
    """Sprint counts and top speeds from tracking."""

    name = "physical"

    def __init__(self) -> None:
        self.sprint_marks: dict[str, int] = {}
        self.speed_record = 29.0

    def on_tick(self, s: MatchState) -> Iterable[InsightCandidate]:
        for pid in list(s.active_players):
            ph = s.physical.get(pid)
            if ph is None:
                continue
            if ph.top_speed_kmh > self.speed_record + 0.3 and abs(ph.top_speed_t - s.t) < 5:
                self.speed_record = ph.top_speed_kmh
                facts = player_facts(s, pid) | {
                    "value": round(ph.top_speed_kmh, 1),
                    "speed_kmh": round(ph.top_speed_kmh, 1),
                    "distance_km": round(ph.distance_m / 1000, 1),
                    "minute": int(s.t // 60) + 1,
                }
                yield InsightCandidate(
                    id=self._id(s, f"speed-{pid}"),
                    detector=self.name,
                    metric="top_speed",
                    t=s.t,
                    team_id=s.team_of(pid),
                    player_id=pid,
                    value=round(ph.top_speed_kmh, 1),
                    value_text=f"{ph.top_speed_kmh:.1f}",
                    template_key="top_speed",
                    theme="physical",
                    base_score=0.55,
                    visual=["speed_tag", "player_highlight"],
                    evidence=Evidence(points=[PitchPoint(x=ph.last_xy[0], y=ph.last_xy[1])], facts=facts),
                    expires_after_s=8,
                )
            mark = (ph.sprints // 10) * 10
            if mark >= 10 and mark > self.sprint_marks.get(pid, 0):
                self.sprint_marks[pid] = mark
                facts = player_facts(s, pid) | {
                    "value": ph.sprints,
                    "sprints": ph.sprints,
                    "distance_km": round(ph.distance_m / 1000, 1),
                    "minute": int(s.t // 60) + 1,
                }
                yield InsightCandidate(
                    id=self._id(s, f"sprints-{pid}"),
                    detector=self.name,
                    metric="sprints",
                    t=s.t,
                    team_id=s.team_of(pid),
                    player_id=pid,
                    value=ph.sprints,
                    value_text=str(ph.sprints),
                    template_key="sprints",
                    theme="physical",
                    base_score=min(0.35 + 0.004 * ph.sprints, 0.7),
                    visual=["player_highlight", "stat_card"],
                    evidence=Evidence(points=[PitchPoint(x=ph.last_xy[0], y=ph.last_xy[1])], facts=facts),
                    expires_after_s=60,
                )


class PassingMilestone(Detector):
    """Completion milestones (every 25 completed passes, shown with completion %)."""

    name = "passing_milestone"

    def on_event(self, s: MatchState, e: Event) -> Iterable[InsightCandidate]:
        if e.type != EventType.PASS or e.outcome != Outcome.SUCCESS or e.player_id is None:
            return
        n = s.count(e.player_id, "passes_completed")
        if n == 0 or n % 25:
            return
        att = s.count(e.player_id, "passes")
        pct = round(100 * n / att)
        facts = player_facts(s, e.player_id) | {
            "value": n, "attempted": att, "completion_pct": pct, "minute": e.minute,
            "progressive": s.count(e.player_id, "progressive_passes"),
        }
        evs = s.evidence[(e.player_id, "progressive_passes")]
        yield InsightCandidate(
            id=self._id(s, e.player_id),
            detector=self.name,
            metric="passes_completed",
            t=e.t,
            team_id=e.team_id,
            player_id=e.player_id,
            value=n,
            value_text=str(n),
            template_key="passing_milestone",
            theme="milestone",
            base_score=0.3 + 0.004 * n + (0.1 if pct >= 90 else 0),
            visual=["stat_card", "player_highlight"],
            evidence=Evidence(event_ids=[x.id for x in evs[-8:]], points=[_pt(x, use_end=True) for x in evs[-8:]],
                              facts=facts),
            expires_after_s=45,
        )


def default_detectors() -> list[Detector]:
    return [
        ZoneActionCount("interceptions_attacking_third", EventType.INTERCEPTION, "interceptions in attacking third", 3),
        ZoneActionCount("tackles_won_attacking_third", EventType.TACKLE, "tackles won in attacking third", 2),
        ZoneActionCount("recoveries_attacking_third", EventType.BALL_RECOVERY, "ball recoveries in attacking third", 3),
        # Own-third interceptions are inflated by the simulator (every cut-out pass counts), so this
        # detector is available but off by default:
        # ZoneActionCount("interceptions_defensive_third", EventType.INTERCEPTION, "interceptions in own third", 4,
        #                 theme="control"),
        PassOfTheMatch(),
        BigChance(),
        PossessionSpell(),
        ChaosSpike(),
        PressingSurge(),
        MomentumSwing(),
        PhysicalMilestones(),
        PassingMilestone(),
    ]
