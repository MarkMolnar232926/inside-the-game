"""LLM-grounding evaluation: every number and name in generated text must be in the evidence payload.

This test is the build gate described in the brief: it renders the narrative for every insight the
Scout proposes on a real synthetic stream, in every language and register, and fails if any line
contains an ungrounded number, name or superlative. It also checks the Verifier catches typical
LLM hallucinations.
"""

import pytest

from pitchlens.agents.verifier import Verifier
from pitchlens.agents.watcher import InsightScout, StreamWatcher
from pitchlens.ingest.sources import ReplaySource
from pitchlens.narrative import templates as T


@pytest.fixture(scope="module")
def candidates(short_match):
    w, sc = StreamWatcher(short_match.meta), InsightScout()
    out = []
    for item in ReplaySource(short_match.events, short_match.tracking, tick_every=5):
        if w.ingest(item):
            out += sc.on_event(w.state, item)
        out += sc.on_time(w.state)
    return out


def test_scout_finds_varied_insights(candidates):
    assert len({c.metric for c in candidates}) >= 5


@pytest.mark.parametrize("lang", T.SUPPORTED)
@pytest.mark.parametrize("register", T.REGISTERS)
def test_every_template_line_is_grounded(short_match, candidates, lang, register):
    v = Verifier(short_match.meta)
    failures = []
    for c in candidates:
        headline, why = T.render(c.template_key, register, c.evidence.facts, lang)
        g = v.check([headline, why], c.evidence.facts, c.metric)
        if not g.verified:
            failures.append((c.metric, why, g.failures))
    assert not failures, failures[:3]


def test_verifier_rejects_hallucinations(short_match, candidates):
    v = Verifier(short_match.meta)
    c = next(c for c in candidates if c.player_id)
    f = c.evidence.facts
    other = next(p for p in short_match.meta.all_players() if p.short_name.title() != f["player_short"])
    assert not v.check([f"{f['player_short']} has 99 interceptions."], f, c.metric).verified
    assert not v.check([f"{other.short_name.title()} made the pass."], f, c.metric).verified
    if f.get("rank") != 1:
        assert not v.check([f"{f['player_short']} has the most interceptions."], f, c.metric).verified
    assert v.check([f"{f['player_short']}: {f['value']}"], f, c.metric).verified


def test_locale_decimal_commas_are_grounded(short_match):
    v = Verifier(short_match.meta)
    facts = {"xg": 0.37, "player_short": "Kestane", "shot_speed_kmh": 104, "distance_m": 12, "team_short": "KBY",
             "team_xg": 1.42}
    _, why = T.render("big_chance", "analyst", facts, "hu")
    assert "0,37" in why and v.check([why], facts).verified
