"""PitchLens API (FastAPI).

REST
  GET  /api/health                      liveness + mode (mock | azure)
  GET  /api/demo                        manifest of the pre-rendered demo bundle
  GET  /api/meta | /api/tokens | /api/profiles | /api/locales/{lang}
  GET  /api/timeline/{profile}          overlay timeline (schema-validated JSON)
  GET  /api/camera | /api/tracks        camera track (homographies) and player screen tracks
  GET  /api/trace?insight_id=&agent=    agent decisions ("why am I seeing this?")
  GET  /api/evidence/{insight_id}       events behind an overlay (for the evidence drawer)
  GET  /api/recap                       post-match recap (all languages)
  GET  /api/metrics                     events/s, insights proposed/shown/rejected, latency
WebSocket
  /ws/live?profile=casual&speed=20&start=4800
      Simulated live feed: synthetic events are paced in accelerated real time,
      run through the full agent pipeline, and overlay instructions are pushed
      as they are decided (Azure Web PubSub in cloud mode).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pitchlens.agents.orchestrator import PitchLensEngine
from pitchlens.ingest.sources import LiveSource, ReplaySource, Tick
from pitchlens.narrative.templates import SUPPORTED, locale
from pitchlens.personalize.profiles import builtin_profiles
from pitchlens.render.assets import TOKENS_PATH
from pitchlens.schema import Event

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = Path(os.getenv("PITCHLENS_DEMO_DIR", ROOT / "outputs" / "demo"))
MATCH_DIR = Path(os.getenv("PITCHLENS_MATCH_DIR", ROOT / "data" / "synthetic" / "demo"))
WEB_DIR = ROOT / "web"

app = FastAPI(title="PitchLens", version="0.1.0", description="Agentic insight overlays for football broadcasts.")


def _json(path: Path):
    if not path.exists():
        raise HTTPException(404, f"{path.name} not built yet - run `python scripts/build_demo.py`")
    return JSONResponse(json.loads(path.read_text()))


@lru_cache(maxsize=1)
def match():
    from pitchlens.synth.generate import SyntheticMatch

    return SyntheticMatch.load(MATCH_DIR)


@lru_cache(maxsize=1)
def full_run():
    """Full-match agent run (no video) for traces, recap and metrics when no demo bundle exists."""
    m = match()
    eng = PitchLensEngine(m.meta, m.events, m.tracking, keep_tool_traces=False)
    return eng.run()


@app.get("/api/health")
def health():
    return {"ok": True, "mode": os.getenv("PITCHLENS_MODE", "mock"), "demo_bundle": DEMO_DIR.exists()}


@app.get("/api/demo")
def demo():
    return _json(DEMO_DIR / "manifest.json")


@app.get("/api/meta")
def meta():
    return JSONResponse(json.loads(match().meta.model_dump_json()))


@app.get("/api/tokens")
def tokens():
    return JSONResponse(json.loads(TOKENS_PATH.read_text()))


@app.get("/api/profiles")
def profiles():
    return [p.model_dump() for p in builtin_profiles().values()]


@app.get("/api/locales/{lang}")
def get_locale(lang: str):
    if lang not in SUPPORTED:
        raise HTTPException(404, "unsupported language")
    return locale(lang)


@app.get("/api/timeline/{profile}")
def timeline(profile: str):
    return _json(DEMO_DIR / "timelines" / f"{profile}.json")


@app.get("/api/camera")
def camera():
    return _json(DEMO_DIR / "camera.json")


@app.get("/api/tracks")
def tracks():
    return _json(DEMO_DIR / "screen_tracks.json")


@app.get("/api/trace")
def trace(insight_id: str | None = None, agent: str | None = None, limit: int = 400):
    p = DEMO_DIR / "traces.json"
    rows = json.loads(p.read_text()) if p.exists() else [t.model_dump() for t in full_run().trace.traces]
    if insight_id:
        rows = [r for r in rows if r.get("input_ref") == insight_id]
    if agent:
        rows = [r for r in rows if r["agent"] == agent]
    return rows[-limit:]


@app.get("/api/evidence/{insight_id:path}")
def evidence(insight_id: str):
    p = DEMO_DIR / "candidates.json"
    cands = json.loads(p.read_text()) if p.exists() else [c.model_dump() for c in full_run().candidates]
    c = next((c for c in cands if c["id"] == insight_id), None)
    if c is None:
        raise HTTPException(404, "unknown insight")
    by_id = {e.id: e for e in match().events}
    evs = [json.loads(by_id[i].model_dump_json()) for i in c["evidence"]["event_ids"] if i in by_id]
    return {"insight": c, "events": evs}


@app.get("/api/recap")
def recap():
    p = DEMO_DIR / "recap.json"
    if p.exists():
        return _json(p)
    rc = full_run().recap
    return JSONResponse(json.loads(rc.model_dump_json())) if rc else {}


@app.get("/api/metrics")
def metrics():
    p = DEMO_DIR / "metrics.json"
    return _json(p) if p.exists() else full_run().metrics


@app.websocket("/ws/live")
async def live(ws: WebSocket, profile: str = Query("casual"), speed: float = Query(20.0), start: float = Query(0.0),
               lang: str = Query("en")):
    await ws.accept()
    m = match()
    profs = builtin_profiles()
    if profile not in profs:
        await ws.close(code=4404)
        return
    eng = PitchLensEngine(m.meta, m.events, m.tracking, profiles={profile: profs[profile]}, keep_tool_traces=False)
    eng.candidates = []
    eng._reset_profiles(0.0)
    # Fast-forward state to the chosen start (history matters for stats), then go live.
    for item in ReplaySource(m.events, None, t_to=start):
        eng.step(item, active=False)
    live_src = LiveSource(ReplaySource(m.events, m.tracking, tick_every=5, t_from=start), speed=speed)
    n_traces = len(eng.trace.traces)
    t_wall0, n_ev = time.perf_counter(), 0
    try:
        async for arrived, item in live_src.stream():
            new = eng.step(item, active=True)
            msg: dict = {"t": round(item.t, 1)}
            if isinstance(item, Tick):
                msg["players"] = [[str(e), round(float(x), 1), round(float(y), 1)]
                                  for e, x, y in zip(item.entities, item.x, item.y, strict=True)]
            if isinstance(item, Event):
                n_ev += 1
                msg["event"] = {"id": item.id, "type": item.type.value, "team_id": item.team_id,
                                "player_id": item.player_id, "x": item.x, "y": item.y, "score": item.score,
                                "outcome": item.outcome.value}
            if new.get(profile):
                msg["overlays"] = [json.loads(i.model_dump_json()) for i in new[profile]]
                msg["latency_ms"] = round((time.perf_counter() - arrived) * 1000, 2)
            if len(eng.trace.traces) > n_traces:
                msg["traces"] = [t.model_dump() for t in eng.trace.traces[n_traces:]][-20:]
                n_traces = len(eng.trace.traces)
            if n_ev and n_ev % 25 == 0:
                el = time.perf_counter() - t_wall0
                msg["metrics"] = {**dict(eng.trace.metrics), "events_per_s": round(n_ev / max(el, 1e-6), 1)}
            if len(msg) > 1:
                await ws.send_json(msg)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return


if (DEMO_DIR / "media").exists():
    app.mount("/media", StaticFiles(directory=DEMO_DIR / "media"), name="media")
if (ROOT / "assets").exists():
    app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.get("/")
def index():
    if (WEB_DIR / "index.html").exists():
        return FileResponse(WEB_DIR / "index.html")
    return RedirectResponse("/docs")
