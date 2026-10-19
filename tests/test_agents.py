"""Agent orchestration: profiles diverge, decisions are traced, latency budget holds."""

import time

import pytest

from pitchlens.agents.orchestrator import PitchLensEngine
from pitchlens.personalize.profiles import builtin_profiles
from pitchlens.schema import OverlayTimeline


@pytest.fixture(scope="module")
def result(short_match):
    eng = PitchLensEngine(short_match.meta, short_match.events, short_match.tracking, keep_tool_traces=False)
    t0 = time.perf_counter()
    res = eng.run(t_to=short_match.meta.duration_s, recap=False)
    res.metrics["wall_s"] = time.perf_counter() - t0
    return res


def test_profiles_produce_different_experiences(result):
    shown = {k: [c.metric for c in v] for k, v in result.shown.items()}
    assert shown["casual"] != shown["analyst"]
    assert len(shown["analyst"]) > len(shown["casual"])  # denser for analysts
    assert set(shown["single_metric"]) <= {"interceptions_attacking_third", "tackles_won_attacking_third",
                                           "recoveries_attacking_third", "ppda"}
    focus = builtin_profiles()["player_focus"].focus_player
    assert sum(c.player_id == focus for c in result.shown["player_focus"]) >= len(result.shown["player_focus"]) / 2


def test_timelines_validate_and_are_multilingual(result):
    for tl in result.timelines.values():
        tl = OverlayTimeline.model_validate_json(tl.model_dump_json())
        for it in tl.items:
            if it.type.value in ("stat_card", "narrative_lower_third"):
                assert set(it.text) == set(tl.languages)
                assert it.grounding is not None and it.grounding.verified


def test_analyst_cards_carry_extra_metrics(result):
    cards = [i for i in result.timelines["analyst"].items if i.type.value == "stat_card"]
    assert any(c.extra_metrics for c in cards) and all(c.density == "dense" or not c.player_id for c in cards)


def test_every_shown_insight_has_a_traced_reason(result):
    shows = [t for t in result.trace.traces if t.agent == "director" and t.action == "show"]
    assert shows and all("score" in t.summary for t in shows)
    assert result.metrics["insights_proposed"] > result.metrics["insights_shown"] / len(result.shown)


def test_latency_budget(result):
    # Brief: insight decision under ~1 s from event arrival in mock mode.
    assert result.metrics["latency_ms_max"] < 1000
    assert result.metrics["latency_ms_p99"] < 100


def test_recap_full_match():
    from pitchlens.synth.generate import generate

    m = generate(seed=4, narrative="late_comeback", duration_s=900)
    res = PitchLensEngine(m.meta, m.events, m.tracking, profiles={"analyst": builtin_profiles()["analyst"]},
                          keep_tool_traces=False).run()
    rc = res.recap
    assert rc is not None and rc.verified
    assert set(rc.headline) == {"en", "hu", "nl", "fr", "es", "de", "pt"}
    assert rc.player_of_the_match["player_id"]
