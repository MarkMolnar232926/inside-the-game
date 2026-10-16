"""Editor / Director agent: decides what goes on screen, when, for how long, and where.

Score of a pending candidate::

    score = base_score x profile_weight x freshness x novelty x visibility

* freshness  - exponential decay with candidate age; expired candidates drop.
* novelty    - cooldowns per metric (150 s) and per subject (40 s). A higher
               value of the same metric (e.g. the 4th interception after
               the 3rd) is only mildly penalised.
* visibility - when the camera is known, the share of evidence points on screen.
* guards     - screen budget (one card at a time, ``min_gap_s`` between cards),
               and *never cover key action*: cards are held for 3 s after
               shots/goals unless the card is about that shot.

Placement flips the panel to the left when the ball is on the right of the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pitchlens.schema import CameraTrack, Decision, EventType, InsightCandidate, apply_h
from pitchlens.state.match_state import MatchState

from ..personalize.profiles import ViewerProfile
from .base import TraceLog

METRIC_COOLDOWN_S = 150.0
SUBJECT_COOLDOWN_S = 40.0
KEY_ACTION_HOLD_S = 3.0
PREEMPT_MIN_ON_S = 4.0  # a card must have had at least this long on screen before it can be cut
PREEMPT_RATIO = 1.35  # ...and only for a clearly bigger story


@dataclass
class ScreenContext:
    """What the Director can know about the picture (only when rendering against video)."""

    camera: CameraTrack
    match_t0: float  # match clock at media t = 0
    ball_xy: np.ndarray | None = None  # (frames, 2) pitch metres, optional

    def frame(self, t_match: float) -> int:
        return int(round((t_match - self.match_t0) * self.camera.fps))

    def in_clip(self, t_match: float) -> bool:
        return 0 <= self.frame(t_match) < len(self.camera.homographies)

    def visible_fraction(self, pts: list[tuple[float, float]], t_match: float, horizon: float = 3.0) -> float:
        if not pts:
            return 1.0
        f0 = self.frame(t_match)
        fr = [min(max(f0 + int(k * self.camera.fps), 0), len(self.camera.homographies) - 1) for k in (0, horizon)]
        best = 0.0
        for f in fr:
            P = apply_h(self.camera.H(f), np.asarray(pts))
            W, H = self.camera.width, self.camera.height
            ok = (P[:, 0] > 0.04 * W) & (P[:, 0] < 0.70 * W) & (P[:, 1] > 0.25 * H) & (P[:, 1] < 0.92 * H)
            best = max(best, float(ok.mean()))
        return best

    def ball_screen_x(self, t_match: float) -> float | None:
        if self.ball_xy is None:
            return None
        f = min(max(self.frame(t_match), 0), len(self.ball_xy) - 1)
        p = apply_h(self.camera.H(f), self.ball_xy[f : f + 1])[0]
        return float(p[0] / self.camera.width)


@dataclass
class Director:
    profile: ViewerProfile
    trace: TraceLog
    screen: ScreenContext | None = None
    pending: dict[str, InsightCandidate] = field(default_factory=dict)
    last_metric: dict[str, tuple[float, float]] = field(default_factory=dict)  # metric -> (t, value)
    last_subject: dict[str, float] = field(default_factory=dict)
    busy_until: float = -1e9
    last_card_end: float = -1e9
    current: tuple[str, float, float] | None = None  # (insight id, score, t shown)
    name: str = "director"

    def offer(self, cands: list[InsightCandidate]) -> None:
        for c in cands:
            # A newer candidate from the same detector about the same subject supersedes the old one.
            key = f"{c.detector}:{c.player_id or c.team_id or 'match'}"
            old = self.pending.get(key)
            if old is not None:
                self.trace.add(self.name, c.t, "supersede", f"{self.profile.id}: {old.id} -> {c.id}", input_ref=old.id)
            self.pending[key] = c

    def _key_action(self, state: MatchState, t: float, cand: InsightCandidate) -> bool:
        for e in reversed(state.events[-12:]):
            if t - e.t > KEY_ACTION_HOLD_S:
                break
            if e.type in (EventType.SHOT, EventType.GOAL) and e.id not in cand.evidence.event_ids:
                return True
        return False

    def score(self, c: InsightCandidate, t: float) -> tuple[float, dict]:
        age = max(t - c.t, 0.0)
        fresh = math.exp(-age / max(0.5 * c.expires_after_s, 1.0))
        w = self.profile.weight(c.metric, c.theme, c.team_id, c.player_id)
        nov = 1.0
        lm = self.last_metric.get(c.metric)
        if lm is not None and t - lm[0] < METRIC_COOLDOWN_S:
            nov *= 0.75 if c.value > lm[1] else 0.25
        subj = c.player_id or c.team_id
        if subj and t - self.last_subject.get(subj, -1e9) < SUBJECT_COOLDOWN_S:
            nov *= 0.6
        vis = 1.0
        if self.screen is not None and "pitch_markers" in c.visual and c.evidence.points:
            vf = self.screen.visible_fraction([(p.x, p.y) for p in c.evidence.points], t)
            vis = 0.7 + 0.45 * vf
        s = c.base_score * w * fresh * nov * vis
        return s, {"base": round(c.base_score, 2), "profile": round(w, 2), "fresh": round(fresh, 2),
                   "novelty": round(nov, 2), "visibility": round(vis, 2)}

    def decide(self, state: MatchState, t: float) -> list[tuple[Decision, InsightCandidate]]:
        # Expire
        for k, c in list(self.pending.items()):
            if t - c.t > c.expires_after_s:
                self.trace.add(self.name, t, "expire", f"{self.profile.id}: {c.id} expired unseen", input_ref=c.id)
                self.trace.metrics["insights_expired"] += 1
                del self.pending[k]
        if not self.pending:
            return []
        if self.screen is not None and not self.screen.in_clip(t):
            return []
        busy = t < self.busy_until or t - self.last_card_end < self.profile.min_gap_s
        on_air = self.current is not None and t < self.busy_until
        if busy and not (on_air and t - self.current[2] >= PREEMPT_MIN_ON_S):  # type: ignore[index]
            return []
        scored = []
        for k, c in self.pending.items():
            s, parts = self.score(c, t)
            scored.append((s, k, c, parts))
        scored.sort(key=lambda x: -x[0])
        s, k, c, parts = scored[0]
        if s < self.profile.threshold:
            return []
        preempts = None
        if busy:
            # Only a clearly bigger story may cut the card that is on air (a human director's "cut to").
            if not on_air or s < PREEMPT_RATIO * self.current[1]:  # type: ignore[index]
                return []
            preempts = self.current[0]  # type: ignore[index]
        if self._key_action(state, t, c):
            self.trace.add(self.name, t, "hold", f"{self.profile.id}: hold {c.id}: key action on the pitch", input_ref=c.id)
            return []
        duration = self.profile.card_s
        if c.metric == "pass_quality":
            duration = 6.5
        elif c.metric == "top_speed":
            duration = 5.0
        placement = "right"
        bx = self.screen.ball_screen_x(t) if self.screen else None
        if bx is not None and bx > 0.62:
            placement = "left"
        others = len(scored) - 1
        formula = " x ".join(f"{n} {v}" for n, v in parts.items())
        reason = f"score {s:.2f} = {formula}; beat {others} other candidate(s)"
        if preempts:
            reason += f"; PREEMPTS {preempts} (score {self.current[1]:.2f}, >= {PREEMPT_RATIO}x)"  # type: ignore[index]
        if bx is not None:
            reason += f"; panel {placement} (ball at {bx:.0%} of frame width)"
        d = Decision(insight_id=c.id, action="show", reason=reason, score=round(s, 3), duration_s=duration,
                     placement=placement, preempts=preempts)  # type: ignore[arg-type]
        self.trace.add(self.name, t, "show", f"{self.profile.id}: {reason}", output=d.model_dump(), input_ref=c.id)
        self.trace.metrics["insights_shown"] += 1
        del self.pending[k]
        self.last_metric[c.metric] = (t, c.value)
        if c.player_id or c.team_id:
            self.last_subject[c.player_id or c.team_id or ""] = t
        self.busy_until = t + duration
        self.last_card_end = t + duration
        self.current = (c.id, s, t)
        if preempts:
            self.trace.metrics["insights_preempted"] += 1
        return [(d, c)]
