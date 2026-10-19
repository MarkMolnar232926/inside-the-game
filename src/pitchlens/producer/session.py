"""Live production session: event feed -> analyst -> approval queue -> on-air channel.

One session per match. The feed is paced on a virtual match clock (``speed`` x real time) so the
producer can pause or change speed without the feed rushing to catch up. Two channels fan out:

* ``producer`` - every proposal and status change, the event ticker, clock and score;
* ``onair``    - only approved graphics (plus clock/score for the score bug).

In production the feed is ``pitchlens.azure.eventhubs.EventHubsSource`` and the fan-out is Azure
Web PubSub; locally it is a replay of the synthetic match and in-process queues.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from pitchlens.agents.base import TraceLog
from pitchlens.agents.watcher import InsightScout, StreamWatcher
from pitchlens.ingest.sources import ReplaySource
from pitchlens.schema import Event, MatchMeta

from .analyst import AnalystAgent, Proposal
from .queue import ApprovalQueue

TICK_S = 0.05  # wall-clock granularity of the virtual clock


class LiveSession:
    def __init__(self, meta: MatchMeta, events: list[Event], speed: float = 20.0, start: float = 0.0, **analyst_kw):
        self.meta, self.events = meta, events
        self.speed, self.start = speed, start
        self.trace = TraceLog(keep_tool_calls=False)
        self.watcher = StreamWatcher(meta)
        self.scout = InsightScout()
        self.analyst = AnalystAgent(meta, events, self.trace, **analyst_kw)
        self.queue = ApprovalQueue()
        self.clock = start
        self.running = False
        self.finished = False
        self.ticker: list[dict] = []
        self.subs: dict[str, set[asyncio.Queue]] = {"producer": set(), "onair": set()}
        self.task: asyncio.Task | None = None

    # ------------------------------------------------------------------ fan-out
    def subscribe(self, channel: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.subs[channel].add(q)
        return q

    def unsubscribe(self, channel: str, q: asyncio.Queue) -> None:
        self.subs[channel].discard(q)

    def publish(self, channel: str, msg: dict) -> None:
        data = json.loads(json.dumps(msg, default=str))
        for q in list(self.subs[channel]):
            with contextlib.suppress(asyncio.QueueFull):  # a stalled client must never block the feed
                q.put_nowait(data)

    def clock_msg(self) -> dict:
        st = self.watcher.state
        return {"type": "clock", "t": round(self.clock, 1), "score": list(st.score), "running": self.running,
                "speed": self.speed, "finished": self.finished}

    def snapshot(self) -> dict[str, Any]:
        h, a = self.meta.home, self.meta.away
        return {**self.clock_msg(), "type": "snapshot",
                "teams": [{"id": t.id, "name": t.name, "short": t.short, "primary": t.primary, "secondary": t.secondary}
                          for t in (h, a)],
                "players": {p.id: p.name.split(" ", 1)[-1] for p in self.meta.all_players()},
                "pending": [p.model_dump() for p in self.queue.pending()],
                "history": [p.model_dump() for p in self.queue.history()[:30]],
                "ticker": self.ticker[-30:]}

    def _status(self, ps: list[Proposal]) -> None:
        for p in ps:
            self.publish("producer", {"type": "status", "id": p.id, "status": p.status})

    # ------------------------------------------------------------------ feed
    async def run(self) -> None:
        for e in self.events:  # history before the start matters for the stats; propose nothing from it
            if e.t >= self.start:
                break
            self.watcher.ingest(e)
        self.running = True
        self.publish("producer", self.clock_msg())
        try:
            for item in ReplaySource(self.events, None, t_from=self.start):
                while self.clock < item.t:
                    await asyncio.sleep(TICK_S)
                    if self.running:
                        self.clock = min(item.t, self.clock + TICK_S * self.speed)
                    self._status(self.queue.expire(self.clock))
                self.ingest(item)
            self.finished, self.running = True, False
            self.publish("producer", self.clock_msg())
        except asyncio.CancelledError:
            pass

    def ingest(self, e: Event) -> list[Proposal]:
        st = self.watcher.state
        before = tuple(st.score)
        self.watcher.ingest(e)
        cands = self.scout.on_event(st, e) + self.scout.on_time(st)
        row = {"t": round(e.t, 1), "minute": int(e.t // 60) + 1, "kind": e.type.value, "team": e.team_id,
               "player": e.player_id, "outcome": e.outcome.value}
        self.ticker.append(row)
        self.publish("producer", {"type": "event", **row})
        if tuple(st.score) != before:
            self.publish("onair", self.clock_msg())
        self.publish("producer", self.clock_msg())
        new = []
        for c in cands:
            p = self.analyst.propose(st, c)
            if p is None:
                continue
            self._status(self.queue.add(p))
            self.publish("producer", {"type": "proposal", "proposal": p.model_dump()})
            new.append(p)
        return new

    # ------------------------------------------------------------------ control
    def launch(self) -> None:
        self.task = asyncio.get_running_loop().create_task(self.run())

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    def set_running(self, on: bool) -> None:
        self.running = on and not self.finished
        self.publish("producer", self.clock_msg())

    def set_speed(self, x: float) -> None:
        self.speed = max(0.25, min(120.0, x))
        self.publish("producer", self.clock_msg())

    def decide(self, pid: str, approve: bool, headline: str | None = None, why: str | None = None) -> Proposal:
        p = self.queue.get(pid)
        if approve and (headline is not None or why is not None):
            ok, failures = self.analyst.reverify(p, headline if headline is not None else p.headline,
                                                 why if why is not None else p.why)
            if not ok:
                raise ValueError("; ".join(failures))
        p = self.queue.decide(pid, approve, headline, why)
        self.trace.add("producer", self.clock, "approve" if approve else "reject",
                       f"{p.id} {p.player.last} {p.value_text} {p.label}" + (" (edited)" if p.edited else ""),
                       input_ref=p.insight_id)
        self._status([p])
        if approve:
            self.publish("onair", {**self.clock_msg(), "type": "play", "proposal": p.model_dump()})
        return p
