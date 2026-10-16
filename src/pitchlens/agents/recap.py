"""Recap agent: automated post-match recap in every supported language.

Produces a headline (detects comebacks), 3-5 key moments with timestamps,
player of the match with an evidence-backed reason, and a tactical summary
(pressing intensity, regains, xG). All strings come from grounded facts and
pass the Verifier.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from pitchlens.narrative import templates as T
from pitchlens.schema import EventType, InsightCandidate, Outcome
from pitchlens.state.match_state import MatchState

from .base import TraceLog
from .verifier import Verifier


class RecapMoment(BaseModel):
    minute: int
    t: float
    kind: str
    text: dict[str, str]
    event_ids: list[str] = Field(default_factory=list)


class Recap(BaseModel):
    match_id: str
    final_score: tuple[int, int]
    headline: dict[str, str]
    summary: dict[str, str]
    moments: list[RecapMoment]
    player_of_the_match: dict
    potm_text: dict[str, str]
    stats: dict
    verified: bool
    failures: list[str] = Field(default_factory=list)


class RecapAgent:
    name = "recap"

    def __init__(self, verifier: Verifier, trace: TraceLog, languages: list[str]):
        self.verifier = verifier
        self.trace = trace
        self.languages = languages

    def potm(self, s: MatchState) -> tuple[str, dict]:
        scores = {}
        for pid, c in s.counters.items():
            xg = c["xg_milli"] / 1000
            score = (3.0 * c["goals"] + 2.0 * xg + 0.9 * c["key_passes"] + 0.6 * c["interceptions_attacking_third"]
                     + 0.25 * c["interceptions"] + 0.35 * c["tackles_won"] + 0.12 * c["progressive_passes"]
                     + 0.01 * c["passes_completed"] + 0.15 * c["recoveries_attacking_third"])
            scores[pid] = round(score, 2)
        best = max(scores, key=scores.get)
        c = s.counters[best]
        return best, {
            "goals": c["goals"], "interceptions": c["interceptions"], "passes_completed": c["passes_completed"],
            "interceptions_attacking_third": c["interceptions_attacking_third"], "key_passes": c["key_passes"],
            "tackles_won": c["tackles_won"], "xg": round(c["xg_milli"] / 1000, 2), "potm_score": scores[best],
        }

    def run(self, s: MatchState, shown: list[InsightCandidate]) -> Recap:
        meta = s.meta
        h, a = meta.home, meta.away
        sh, sa = s.score
        # Comeback = winner trailed at some point.
        trailed = {h.id: False, a.id: False}
        for e in s.events:
            if e.type == EventType.GOAL:
                hh, aa = e.score
                if hh > aa:
                    trailed[a.id] = True
                if aa > hh:
                    trailed[h.id] = True
        xg = Counter()
        for (_, tid, x) in s.xg_timeline:
            xg[tid] += x
        winner = h if sh > sa else a if sa > sh else None
        loser = (a if winner is h else h) if winner else None
        score_txt = f"{max(sh, sa)}-{min(sh, sa)}"
        ppda = {tid: s.ppda(tid, 1e9) for tid in (h.id, a.id)}
        press_team = min(ppda, key=lambda k: ppda[k] if ppda[k] is not None else 99)
        other = a.id if press_team == h.id else h.id
        regains = sum(1 for e in s.events if e.team_id == press_team and e.x is not None and e.outcome != Outcome.FAIL
                      and e.type in (EventType.INTERCEPTION, EventType.TACKLE, EventType.BALL_RECOVERY)
                      and s.norm_x(e) > 52.5)
        potm_id, potm_facts = self.potm(s)
        potm = meta.player(potm_id)
        facts = {
            "winner": winner.name if winner else None, "loser": loser.name if loser else None,
            "winner_short": winner.short if winner else None, "home": h.name, "away": a.name,
            "score": score_txt if winner else f"{sh}-{sa}",
            "xg_w": round(xg[winner.id], 2) if winner else None, "xg_l": round(xg[loser.id], 2) if loser else None,
            "xg_h": round(xg[h.id], 2), "xg_a": round(xg[a.id], 2),
            "press_team": meta.team(press_team).name, "ppda_press": round(ppda[press_team] or 0, 1),
            "ppda_other": round(ppda[other] or 0, 1), "press_regains": regains,
            "player_name": potm.name, "player_short": potm.short_name.title(), "team_short": meta.team(potm.team_id).short,
            **potm_facts,
        }
        # numbers inside the score string ("4-2") are facts too
        facts["score_parts"] = [sh, sa]
        headline, summary, potm_text = {}, {}, {}
        for lang in self.languages:
            if winner is None:
                hl = T.recap_string("headline_draw", facts, lang)
                tac = T.recap_string("tactical_press", facts, lang)
                sm = T.recap_string("summary_draw", facts | {"tactical": tac}, lang)
            else:
                key = "headline_comeback" if trailed[winner.id] else "headline_win"
                hl = T.recap_string(key, facts, lang)
                tac = T.recap_string("tactical_press", facts, lang)
                sm = T.recap_string("summary", facts | {"tactical": tac}, lang)
            reason = T.recap_string("potm_reason", facts, lang)
            headline[lang], summary[lang] = hl, sm
            potm_text[lang] = T.recap_string("potm", facts | {"reason": reason}, lang)
        moments: list[RecapMoment] = []
        for e in s.events:
            if e.type == EventType.GOAL and e.player_id:
                shot = next((x for x in reversed(s.events) if x.type == EventType.SHOT and x.player_id == e.player_id
                             and x.t <= e.t and e.t - x.t < 5), None)
                p = meta.player(e.player_id)
                f = {"player_short": p.short_name.title(), "team_short": meta.team(p.team_id).short,
                     "xg": round(shot.shot_info.xg, 2) if shot and shot.shot_info else 0.0}
                moments.append(RecapMoment(minute=e.minute, t=e.t, kind="goal", event_ids=[e.id],
                                           text={L: T.recap_string("moment_goal", f, L) for L in self.languages}))
        # Add the most newsworthy shown insights (one per metric), then trim.
        seen = set()
        for c in sorted(shown, key=lambda c: -c.base_score):
            if c.metric in seen or c.metric in ("shot_xg",) or len(moments) >= 7 or not (c.player_id or c.team_id):
                continue
            seen.add(c.metric)
            ff = c.evidence.facts
            texts = {}
            for L in self.languages:
                hl_c, _ = T.render(c.template_key, "casual", ff, L)
                key = "moment_insight" if c.player_id else "moment_team_insight"
                texts[L] = T.recap_string(key, ff | {"headline": hl_c, "value": c.value_text}, L)
            moments.append(RecapMoment(minute=int(c.t // 60) + 1, t=c.t, kind=c.metric, text=texts,
                                       event_ids=c.evidence.event_ids[-6:]))
        moments.sort(key=lambda m: m.t)
        allt = [*headline.values(), *summary.values(), *potm_text.values()]
        g = self.verifier.check(allt, facts | {"potm": potm_facts})
        rec = Recap(match_id=meta.match_id, final_score=(sh, sa), headline=headline, summary=summary,
                    moments=moments[:7], player_of_the_match={"player_id": potm_id, **potm_facts}, potm_text=potm_text,
                    stats={"xg": {h.id: round(xg[h.id], 2), a.id: round(xg[a.id], 2)},
                           "ppda": {k: (round(v, 1) if v else None) for k, v in ppda.items()},
                           "regains_opp_half": {press_team: regains}},
                    verified=g.verified, failures=g.failures)
        self.trace.add(self.name, s.t, "recap", headline.get("en", ""), output={"verified": g.verified,
                       "moments": len(rec.moments), "potm": potm_id})
        return rec
