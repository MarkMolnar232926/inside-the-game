"""Detectors: zone logic, evidence, attack-direction normalisation."""

from pitchlens.insights.detectors import ZoneActionCount
from pitchlens.insights.orientation import orient_points
from pitchlens.schema import Event, EventType, Outcome, PitchPoint
from pitchlens.state.match_state import MatchState
from pitchlens.synth.teams import build_teams


def _meta():
    from pitchlens.schema import MatchMeta

    h, a = build_teams(4)
    return MatchMeta(match_id="T", venue="v", kickoff_utc="", seed=4, narrative="balanced", duration_s=5400, home=h, away=a)


def _int(seq, t, x, y, period=1, pid="KBY-07"):
    return Event(id=f"T-e{seq}", match_id="T", seq=seq, t=t, period=period, type=EventType.INTERCEPTION, team_id="KBY",
                 player_id=pid, x=x, y=y, outcome=Outcome.SUCCESS)


def test_interceptions_attacking_third_fires_with_evidence():
    st, det = MatchState(_meta()), ZoneActionCount("interceptions_attacking_third", EventType.INTERCEPTION, "x", 3)
    # KBY attacks right in period 1: x > 70 is the attacking third; x = 40 is not.
    evs = [_int(1, 10, 80, 30), _int(2, 20, 40, 30), _int(3, 30, 90, 20), _int(4, 40, 75, 50)]
    out = []
    for e in evs:
        st.ingest(e)
        out += list(det.on_event(st, e))
    assert len(out) == 1
    c = out[0]
    assert c.value == 3 and c.player_id == "KBY-07"
    assert c.evidence.event_ids == ["T-e1", "T-e3", "T-e4"]
    assert [(p.x, p.y) for p in c.evidence.points] == [(80, 30), (90, 20), (75, 50)]
    assert c.evidence.facts["rank"] == 1 and c.evidence.facts["share_pct"] == 100


def test_second_half_direction_flip():
    st = MatchState(_meta())
    # In period 2 KBY attacks left: x = 20 is now the attacking third.
    e = _int(1, 3000, 20, 30, period=2)
    st.ingest(e)
    assert st.zone(e) == "attacking"
    assert st.count("KBY-07", "interceptions_attacking_third") == 1


def test_orientation_rotates_other_half():
    e1, e2 = _int(1, 10, 80, 30, period=1), _int(2, 3000, 20, 30, period=2)
    pts = [PitchPoint(x=80, y=30, event_id=e1.id), PitchPoint(x=20, y=30, event_id=e2.id)]
    out, flipped = orient_points(pts, {e1.id: e1, e2.id: e2}, display_period=2)
    assert flipped and (out[0].x, out[0].y) == (25, 38) and (out[1].x, out[1].y) == (20, 30)


def test_rhythm_metrics_on_real_stream(short_match):
    st = MatchState(short_match.meta)
    for e in short_match.events:
        st.ingest(e)
    rh = st.rhythm(300)
    assert 0 <= rh["chaos_index"] <= 1
    assert st.ppda("KBY", 1e9) is not None and 0.5 < st.ppda("KBY", 1e9) < 40
