"""Analyst agent: from raw event-data insights to a short list of player moments worth a producer's time.

The Insight Scout fires on every threshold a stat crosses (7, 8, 9, 10 recoveries...). A producer
cannot review that. The analyst keeps only *player* insights and only the outstanding ones:

* **gate**     - the player leads the match in that stat (rank 1) or owns most of the team's share,
                 or a chance/pass is genuinely big (per-metric rules below);
* **novelty**  - one proposal per player+metric until the value has grown by a meaningful step;
* **rhythm**   - a global gap between proposals so the queue never floods.

Each survivor becomes a :class:`Proposal`: the on-air card (big number, label, player, team), a
grounded explanation (Explainer + Verifier, same as the automatic pipeline), the evidence events,
and the analyst's own reasons, so the producer can judge it in a glance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from pitchlens.agents.base import TraceLog
from pitchlens.agents.explainer import Explainer
from pitchlens.agents.verifier import Verifier
from pitchlens.schema import Event, InsightCandidate, MatchMeta
from pitchlens.state.match_state import MatchState

Status = Literal["pending", "approved", "rejected", "expired", "superseded", "aired"]


@dataclass(frozen=True)
class MetricRule:
    label: str          # big line on the card, e.g. "Interceptions"
    sub: str            # second line; may use {fact} placeholders
    min_value: float    # first proposal only from this value
    step: float         # later proposals only when the value grew by this much
    gate: str = "rank"  # "rank": rank 1 or >= 60% of team; "value": min_value alone
    max_per_player: int = 2  # a player's same stat reaches air at most this often
    extra: Callable[[dict], bool] | None = None  # metric-specific test on the facts
    extra_why: str = ""


RULES: dict[str, MetricRule] = {
    "interceptions_attacking_third": MetricRule("Interceptions", "in the attacking third", 3, 1),
    "tackles_won_attacking_third": MetricRule("Tackles won", "in the attacking third", 3, 1),
    "recoveries_attacking_third": MetricRule("Ball recoveries", "in the attacking third", 8, 5),
    "passes_completed": MetricRule("Completed passes", "{completion_pct}% pass accuracy", 75, 25, gate="value",
                                   max_per_player=1, extra=lambda f: f.get("completion_pct", 0) >= 80,
                                   extra_why="80%+ accuracy at volume"),
    "pass_quality": MetricRule("Metre pass", "{difficulty_10}/10 difficulty", 40, 1e9, gate="value",
                               extra=lambda f: f.get("completion_pct", 100) <= 40,
                               extra_why="a pass completed less than 40% of the time"),
    "shot_xg": MetricRule("xG big chance", "{xg_pct}% chance to score", 0.3, 1e9, gate="value",
                          extra_why="a chance worth 0.3 xG or more"),
}


class PlayerCard(BaseModel):
    id: str
    number: int
    first: str
    last: str
    team_id: str
    team_name: str
    team_short: str
    primary: str
    secondary: str


class EvidenceEvent(BaseModel):
    id: str
    type: str
    minute: int
    x: float | None = None
    y: float | None = None


class Proposal(BaseModel):
    id: str
    insight_id: str
    t: float = Field(description="Match clock (s) when the analyst proposed it")
    minute: int
    metric: str
    detector: str
    value: float
    value_text: str
    label: str
    sublabel: str
    headline: str
    why: str
    verified: bool
    checks: list[str] = Field(default_factory=list)
    generator: str = "template"
    player: PlayerCard
    evidence: list[EvidenceEvent] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    status: Status = "pending"
    expires_t: float = Field(description="Match clock after which the stat is stale")
    decided_wall: float | None = None
    edited: bool = False
    supersedes: str | None = None


class AnalystAgent:
    name = "analyst"

    def __init__(self, meta: MatchMeta, events: list[Event], trace: TraceLog | None = None,
                 explainer: Explainer | None = None, min_gap_s: float = 75.0, ttl_s: float = 150.0,
                 register: str = "casual", lang: str = "en"):
        self.meta = meta
        self.by_id = {e.id: e for e in events}
        self.trace = trace or TraceLog(keep_tool_calls=False)
        self.verifier = Verifier(meta)
        self.explainer = explainer or Explainer(self.verifier, self.trace)
        self.min_gap_s, self.ttl_s = min_gap_s, ttl_s
        self.register, self.lang = register, lang
        self.last_value: dict[tuple[str, str], float] = {}
        self.count: dict[tuple[str, str], int] = {}
        self.last_t = -1e9
        self.n = 0

    # ------------------------------------------------------------------ judgement
    def assess(self, c: InsightCandidate, now: float) -> tuple[bool, float, list[str]]:
        """Is this candidate outstanding enough to interrupt a producer? Returns (keep, score, reasons)."""
        rule = RULES.get(c.metric)
        if rule is None or not c.player_id:
            return False, 0.0, ["not a player metric the analyst covers"]
        f = c.evidence.facts
        reasons: list[str] = []
        if c.value < rule.min_value:
            return False, 0.0, [f"{c.value_text} below the {rule.min_value:g} bar"]
        if rule.gate == "rank":
            rank, share = f.get("rank"), f.get("share_pct", 0)
            if rank == 1:
                reasons.append("leads the match in this stat")
            elif share >= 60:
                reasons.append(f"{share}% of the team's total")
            else:
                return False, 0.0, ["not a stand-out within the match"]
            if share >= 60 and rank == 1:
                reasons.append(f"{share}% of the team's total")
        if rule.extra is not None and not rule.extra(f):
            return False, 0.0, ["below the metric's quality bar"]
        if rule.extra_why:
            reasons.append(rule.extra_why)
        key = (c.player_id, c.metric)
        if self.count.get(key, 0) >= rule.max_per_player:
            return False, 0.0, [f"already proposed {self.count[key]}x for this player"]
        if key in self.last_value and c.value < self.last_value[key] + rule.step:
            return False, 0.0, [f"already proposed at {self.last_value[key]:g}"]
        if now - self.last_t < self.min_gap_s:
            return False, 0.0, ["too soon after the previous proposal"]
        novelty = 1.0 if key not in self.last_value else 0.85
        score = min(1.0, c.base_score * novelty * (1.05 if f.get("rank") == 1 else 1.0))
        if key not in self.last_value:
            reasons.append("first time this stat is flagged for this player")
        if "minutes" in f and len(f["minutes"]) >= 3 and f["minutes"][-1] - f["minutes"][-3] <= 10:
            reasons.append("three in the last 10 minutes")
        return True, round(score, 2), reasons

    # ------------------------------------------------------------------ proposal
    def propose(self, state: MatchState, c: InsightCandidate) -> Proposal | None:
        keep, score, reasons = self.assess(c, state.t)
        self.trace.add(self.name, c.t, "propose" if keep else "skip",
                       f"{c.detector} {c.player_id} {c.value_text}: {'; '.join(reasons)}", input_ref=c.id,
                       output={"score": score})
        if not keep:
            return None
        rule = RULES[c.metric]
        f = c.evidence.facts
        exp, g = self.explainer.explain(state, c, self.register, self.lang)
        p = self.meta.player(c.player_id)
        team = self.meta.team(p.team_id)
        first, _, last = p.name.partition(" ")
        evs = []
        for eid in c.evidence.event_ids:
            e = self.by_id.get(eid)
            if e is not None:
                evs.append(EvidenceEvent(id=e.id, type=e.type.value, minute=int(e.t // 60) + 1, x=e.x, y=e.y))
        self.n += 1
        self.last_value[(c.player_id, c.metric)] = c.value
        self.count[(c.player_id, c.metric)] = self.count.get((c.player_id, c.metric), 0) + 1
        self.last_t = state.t
        try:
            sub = rule.sub.format(**f)
        except (KeyError, ValueError):
            sub = ""
        return Proposal(
            id=f"P{self.n:03d}", insight_id=c.id, t=round(state.t, 1), minute=int(state.t // 60) + 1,
            metric=c.metric, detector=c.detector, value=c.value, value_text=c.value_text, label=rule.label, sublabel=sub,
            headline=exp.headline, why=exp.why, verified=g.verified, checks=g.checks + g.failures,
            generator=exp.generator,
            player=PlayerCard(id=p.id, number=p.number, first=first, last=last or p.name, team_id=team.id,
                              team_name=team.name, team_short=team.short, primary=team.primary,
                              secondary=team.secondary),
            evidence=evs, facts=f, score=score, reasons=reasons, expires_t=round(state.t + self.ttl_s, 1),
        )

    def reverify(self, prop: Proposal, headline: str, why: str) -> tuple[bool, list[str]]:
        """An operator's edit must be as grounded as the agent's text: same Verifier, same evidence."""
        g = self.verifier.check([headline, why], prop.facts, prop.metric, "operator")
        return g.verified, g.failures


def wall() -> float:
    return time.time()
