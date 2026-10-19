"""API smoke tests (no demo bundle required)."""

from fastapi.testclient import TestClient

from pitchlens.api.app import app

client = TestClient(app)


def test_health_and_static_contracts():
    assert client.get("/api/health").json()["ok"]
    tok = client.get("/api/tokens").json()
    assert "colors" in tok and "motion" in tok
    profs = client.get("/api/profiles").json()
    assert {p["id"] for p in profs} >= {"casual", "analyst", "player_focus", "single_metric", "favourite_club"}
    assert client.get("/api/locales/hu").json()["lang"] == "hu"
    assert client.get("/api/locales/xx").status_code == 404


def test_live_websocket_pushes_overlays():
    with client.websocket_connect("/ws/live?profile=analyst&speed=2000&start=5100") as ws:
        got_overlay = got_event = False
        for _ in range(4000):
            msg = ws.receive_json()
            got_event |= "event" in msg
            if "overlays" in msg:
                got_overlay = True
                assert msg["latency_ms"] < 1000
                break
        assert got_event and got_overlay
