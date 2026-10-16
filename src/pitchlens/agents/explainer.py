"""Explainer agent: the "why it matters" line, grounded only in evidence.

1. Enrich the insight's facts with match-rhythm context from tools
   (control vs chaos, tempo) so explanations can talk about game state.
2. If an LLM is configured, ask it for {headline, why} in the requested
   register, giving it *only* the facts JSON; otherwise (or if the Verifier
   rejects the LLM output) use the deterministic locale templates.

The Verifier always has the last word.
"""

from __future__ import annotations

import json
import time

from pitchlens.narrative import templates as T
from pitchlens.schema import Explanation, Grounding, InsightCandidate
from pitchlens.state.match_state import MatchState

from ..azure.openai_client import LLMClient, MockLLM
from .base import TraceLog
from .verifier import Verifier

SYSTEM = (
    "You are the on-air data explainer for a football broadcast. Write ONE short headline (max 6 words) and ONE "
    "'why it matters' sentence (max 25 words) for the insight. Use ONLY numbers and names present in the FACTS "
    "JSON; never invent statistics, never compare to other matches. Register: {register}. Language: {lang}. "
    'Reply as JSON {{"headline": str, "why": str}}.'
)


def rhythm_context(state: MatchState, c: InsightCandidate) -> tuple[str | None, dict]:
    rh = state.rhythm(300)
    ps = state.possession_stats(300)
    dom = max(ps, key=lambda k: ps[k]["share"]) if ps else None
    facts = {
        "ctx_turnovers_per_min": rh["turnovers_per_min"],
        "ctx_passes_per_min": rh["passes_per_min"],
        "ctx_avg_passes": round(ps[dom]["avg_passes"], 1) if dom else None,
        "ctx_team_short": state.meta.team(dom).short if dom else None,
        "ctx_chaos_index": rh["chaos_index"],
    }
    kind = None
    if rh["chaos_index"] >= 0.6 and c.theme != "control":
        kind = "chaos"
    elif rh["chaos_index"] <= 0.4 and dom:
        kind = "control"
    elif rh["passes_per_min"] >= 16:
        kind = "tempo"
    return kind, facts


class Explainer:
    name = "explainer"

    def __init__(self, verifier: Verifier, trace: TraceLog, llm: LLMClient | None = None):
        self.verifier = verifier
        self.trace = trace
        self.llm = llm or MockLLM()
        self.cache: dict[tuple[str, str, str], tuple[Explanation, Grounding]] = {}

    def enrich(self, state: MatchState, c: InsightCandidate) -> str | None:
        kind, ctx = rhythm_context(state, c)
        c.evidence.facts.update({k: v for k, v in ctx.items() if v is not None})
        c.evidence.facts["ctx_kind"] = kind or "none"
        self.trace.add(self.name, c.t, "tool:get_rhythm", f"rhythm for {c.id}: {kind}", output=ctx, input_ref=c.id)
        return kind

    def explain(self, state: MatchState | None, c: InsightCandidate, register: str, lang: str,
                with_context: bool = False) -> tuple[Explanation, Grounding]:
        key = (c.id, register, lang)
        if key in self.cache:
            return self.cache[key]
        t0 = time.perf_counter()
        if state is not None and "ctx_kind" not in c.evidence.facts:
            self.enrich(state, c)
        facts = c.evidence.facts
        result: tuple[Explanation, Grounding] | None = None
        if not isinstance(self.llm, MockLLM):
            out = self.llm.complete_json(SYSTEM.format(register=register, lang=lang),
                                         "FACTS: " + json.dumps(facts, default=str))
            if out and out.get("why"):
                g = self.verifier.check([out.get("headline", ""), out["why"]], facts, c.metric, "llm")
                if g.verified:
                    result = (Explanation(insight_id=c.id, language=lang, headline=out.get("headline", ""),
                                          why=out["why"], generator="llm"), g)
                else:
                    self.trace.add(self.verifier.name, c.t, "reject", f"LLM text rejected: {g.failures}",
                                   input_ref=c.id, ok=False)
                    self.trace.metrics["llm_rejected"] += 1
        if result is None:
            headline, why = T.render(c.template_key, register, facts, lang)
            kind = facts.get("ctx_kind")
            if with_context and kind in ("control", "chaos", "tempo"):
                try:
                    extra = T.context_line(kind, facts, lang)
                    if len(why) + len(extra) < 210:
                        why = f"{why} {extra}"
                except KeyError:
                    pass
            g = self.verifier.check([headline, why], facts, c.metric, "template")
            if not g.verified:  # should never happen for vetted templates; fail safe to the bare stat
                self.trace.add(self.verifier.name, c.t, "reject", f"template rejected: {g.failures}", input_ref=c.id,
                               ok=False)
                self.trace.metrics["template_rejected"] += 1
                why = ""
            result = (Explanation(insight_id=c.id, language=lang, headline=headline, why=why), g)
        self.trace.add(self.name, c.t, "explain", f"[{lang}/{register}] {result[0].why[:90]}",
                       output={"generator": result[0].generator, "verified": result[1].verified},
                       latency_ms=(time.perf_counter() - t0) * 1000, input_ref=c.id)
        self.cache[key] = result
        return result
