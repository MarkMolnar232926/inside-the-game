"""Verifier / Critic agent: nothing renders unless every claim is grounded.

Checks on any generated line (template or LLM):

1. **Numbers** - every number in the text must equal a number in the
   evidence facts (locale decimal commas normalised), or be one of a small
   set of methodology constants (window lengths, the sprint threshold).
2. **Names** - any roster name (player surname, club name or code) mentioned
   must appear among the evidence facts' strings.
3. **Superlatives** - "most / more than anyone / fastest / record" require
   ``rank == 1`` (or a record metric) in the facts.
4. **Length** - the line must fit the lower third (<= 220 chars).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pitchlens.schema import Grounding, MatchMeta

NUM_RE = re.compile(r"(?<![\w])[+-]?\d+(?:[.,]\d+)?")
METHOD_CONSTANTS = {"5", "10", "25"}  # rolling windows (5/10 min) and the 25 km/h sprint threshold
SUPERLATIVES = [
    "most", "more than anyone", "fastest", "record", "highest", "best", "nobody", "no one",
    "legtöbb", "senki", "meer dan wie ook", "de meeste", "plus que quiconque", "plus que tout le monde",
    "más que nadie", "mehr als jeder", "mais do que", "máximo", "record", "rekord", "csúcs",
]
RECORD_METRICS = {"top_speed"}
SUPER_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in SUPERLATIVES) + r")(?!\w)")


def _numbers_in(v, out: set[str]) -> None:
    if isinstance(v, bool):
        return
    if isinstance(v, int | float):
        f = float(v)
        out.add(f"{f:g}")
        if f.is_integer():
            out.add(str(int(f)))
    elif isinstance(v, dict):
        for x in v.values():
            _numbers_in(x, out)
    elif isinstance(v, list | tuple):
        for x in v:
            _numbers_in(x, out)


def _strings_in(v, out: set[str]) -> None:
    if isinstance(v, str):
        out.add(v.lower())
    elif isinstance(v, dict):
        for x in v.values():
            _strings_in(x, out)
    elif isinstance(v, list | tuple):
        for x in v:
            _strings_in(x, out)


class Verifier:
    name = "verifier"

    def __init__(self, meta: MatchMeta):
        names = set()
        for p in meta.all_players():
            names.add(p.short_name.lower())
            names.add(p.name.lower())
        for t in (meta.home, meta.away):
            names.update({t.name.lower(), t.short.lower(), t.city.lower()})
        self.roster_names = sorted(names, key=len, reverse=True)

    def check(self, texts: Iterable[str], facts: dict, metric: str = "", generator: str = "template") -> Grounding:
        allowed: set[str] = set(METHOD_CONSTANTS)
        _numbers_in(facts, allowed)
        fact_strings: set[str] = set()
        _strings_in(facts, fact_strings)
        joined_facts = " | ".join(fact_strings)
        checks, failures = [], []
        for text in texts:
            if not text:
                continue
            nums = [n.replace(",", ".").lstrip("+") for n in NUM_RE.findall(text)]
            bad = [n for n in nums if n not in allowed and f"{float(n):g}" not in allowed]
            if bad:
                failures.append(f"ungrounded numbers {bad}")
            elif nums:
                checks.append(f"{len(nums)} number(s) match evidence")
            low = text.lower()
            for nm in self.roster_names:
                if re.search(rf"(?<!\w){re.escape(nm)}(?!\w)", low) and nm not in joined_facts:
                    failures.append(f"name '{nm}' not in evidence")
            if SUPER_RE.search(low) and facts.get("rank") != 1 and metric not in RECORD_METRICS:
                failures.append("superlative without rank==1 evidence")
            if len(text) > 220:
                failures.append(f"too long ({len(text)} chars)")
        if not failures:
            checks.append("names and superlatives grounded")
        return Grounding(verified=not failures, checks=checks, failures=failures, generator=generator)  # type: ignore[arg-type]
