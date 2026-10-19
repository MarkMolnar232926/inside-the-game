"""Human-in-the-loop producer flow: analyst judgement, approval queue, live session, API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pitchlens.producer import ApprovalQueue, LiveSession
from pitchlens.synth.generate import SyntheticMatch

DEMO = Path(__file__).resolve().parents[1] / "data/synthetic/demo"


@pytest.fixture(scope="module")
def match():
    if not (DEMO / "events.jsonl").exists():
        pytest.skip("demo match not generated")
    return SyntheticMatch.load(DEMO)


def run_all(m) -> tuple[LiveSession, list]:
    s = LiveSession(m.meta, m.events)
    props = []
    for e in m.events:
        s.clock = e.t
        props += s.ingest(e)
        s.queue.expire(e.t)
    return s, props


def test_analyst_is_selective_and_grounded(match):
    s, props = run_all(match)
    # a producer can review this many in 90 minutes; the scout alone fires ~100 player insights
    assert 3 <= len(props) <= 12
    for p in props:
        assert p.verified, p.checks
        assert p.player.id and p.label and p.reasons
        assert p.evidence, "every proposal carries the events behind it"
    times = [p.t for p in props]
    assert all(b - a >= s.analyst.min_gap_s for a, b in zip(times, times[1:], strict=False))
    # the hero moment of the demo match is among them
    assert any(p.player.id == "KBY-07" and p.metric == "interceptions_attacking_third" for p in props)


def test_nothing_airs_without_approval(match):
    s, props = run_all(match)
    s.queue = ApprovalQueue()
    q = s.subscribe("onair")
    p = props[0].model_copy(update={"status": "pending"})
    s.queue.add(p)
    assert q.empty()
    s.decide(p.id, approve=False)
    assert q.empty() and s.queue.get(p.id).status == "rejected"
    p2 = props[1].model_copy(update={"status": "pending"})
    s.queue.add(p2)
    s.decide(p2.id, approve=True)
    msg = q.get_nowait()
    assert msg["type"] == "play" and msg["proposal"]["id"] == p2.id
    with pytest.raises(ValueError):
        s.decide(p2.id, approve=True)  # already decided


def test_operator_edits_are_reverified(match):
    s, props = run_all(match)
    s.queue = ApprovalQueue()
    p = props[0].model_copy(update={"status": "pending"})
    s.queue.add(p)
    bad = p.why + " That is 99 in total."
    with pytest.raises(ValueError, match="ungrounded"):
        s.decide(p.id, approve=True, why=bad)
    assert s.queue.get(p.id).status == "pending"
    ok = s.decide(p.id, approve=True, why=p.why + " Huge effort.")
    assert ok.status == "approved" and ok.edited


def test_queue_supersedes_and_expires(match):
    _, props = run_all(match)
    q = ApprovalQueue(max_pending=2)
    a = props[0].model_copy(update={"status": "pending", "id": "X1"})
    b = a.model_copy(update={"id": "X2", "value": a.value + 1})
    q.add(a)
    changed = q.add(b)
    assert [c.id for c in changed] == ["X1"] and q.get("X1").status == "superseded" and b.supersedes == "X1"
    assert [c.id for c in q.expire(b.expires_t + 1)] == ["X2"]


def test_live_session_feed_and_pause(match):
    async def go():
        s = LiveSession(match.meta, match.events, speed=600, start=2000)
        q = s.subscribe("producer")
        s.launch()
        await asyncio.sleep(0.6)
        s.set_running(False)
        t_paused = s.clock
        await asyncio.sleep(0.3)
        assert s.clock == t_paused, "a paused feed does not advance"
        assert s.clock > 2000
        kinds = set()
        while not q.empty():
            kinds.add(q.get_nowait()["type"])
        await s.stop()
        return kinds
    kinds = asyncio.run(go())
    assert {"clock", "event"} <= kinds


def test_api_producer_routes(match):
    from fastapi.testclient import TestClient

    from pitchlens.api import app as api

    with TestClient(api.app) as c:
        assert c.post("/api/session/start?speed=600&start=2000").status_code == 200
        snap = c.get("/api/session").json()
        assert snap["type"] == "snapshot" and len(snap["teams"]) == 2 and snap["players"]
        assert c.post("/api/proposals/NOPE/approve").status_code == 404
        assert c.post("/api/session/pause").json()["running"] is False
        assert c.get("/producer").status_code == 200 and c.get("/onair").status_code == 200
        with c.websocket_connect("/ws/onair") as ws:
            assert ws.receive_json()["type"] == "clock"
