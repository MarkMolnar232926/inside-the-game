"""Synthetic generator: determinism, realism bounds, tracking consistency, no real data."""

import numpy as np

from pitchlens.schema import EventType, Outcome
from pitchlens.synth.generate import generate, summarise
from pitchlens.synth.teams import build_teams


def test_deterministic():
    a = generate(seed=11, narrative="balanced", duration_s=300)
    b = generate(seed=11, narrative="balanced", duration_s=300)
    assert [e.model_dump() for e in a.events] == [e.model_dump() for e in b.events]


def test_realistic_rates(short_match):
    s = summarise(short_match)
    for team in s["teams"].values():
        # 20 minutes of a 90-minute match: ~450-700 passes per team per 90 -> ~100-160 per 20'
        assert 80 <= team["passes"] <= 200
        assert 0.65 <= team["pass_completion"] <= 0.93
    n = {t: sum(1 for e in short_match.events if e.type == t) for t in EventType}
    assert n[EventType.PRESSURE] > 0 and n[EventType.INTERCEPTION] > 0 and n[EventType.TACKLE] > 0
    assert n[EventType.PERIOD_START] == 1


def test_coordinates_and_ids(short_match):
    ids = {p.id for p in short_match.meta.all_players()}
    for e in short_match.events:
        if e.player_id:
            assert e.player_id in ids
        if e.x is not None:
            assert -2 <= e.x <= 107 and -2 <= e.y <= 70
        if e.type == EventType.PASS:
            assert e.pass_info is not None and 0 <= e.pass_info.difficulty <= 1
    seqs = [e.seq for e in short_match.events]
    assert seqs == sorted(seqs)


def test_tracking_consistent_with_events(short_match):
    tr = short_match.tracking
    assert set(tr.columns) >= {"frame", "t", "entity", "x", "y", "speed_kmh"}
    players = tr[tr.entity != "ball"]
    assert players.speed_kmh.max() < 40  # no superhuman speeds
    assert not tr[["x", "y"]].isna().any().any()
    # The ball sits at the passer's feet when a completed pass starts
    ball = tr[tr.entity == "ball"].set_index("frame")
    passes = [e for e in short_match.events if e.type == EventType.PASS and e.outcome == Outcome.SUCCESS][:50]
    d = [np.hypot(ball.loc[round(e.t * 10)].x - e.x, ball.loc[round(e.t * 10)].y - e.y) for e in passes]
    assert np.median(d) < 2.0


def test_all_names_fictional():
    home, away = build_teams(4)
    assert home.name == "Kestrel Bay FC" and away.name == "Ironvale Athletic"
    assert len({p.name for p in home.players + away.players}) == 32
