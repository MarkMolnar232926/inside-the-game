"""Event sources (stage 1: ingest).

All sources yield ``StreamItem`` objects in match-clock order: either an
``Event`` or a tracking ``Tick`` (one 10 Hz frame of player positions).

* :class:`ReplaySource`  - deterministic file replay (as fast as possible).
* :class:`LiveSource`    - the same data paced in (accelerated) real time, async.
* ``pitchlens.azure.eventhubs.EventHubsSource`` - Azure Event Hubs consumer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitchlens.schema import Event


@dataclass(slots=True)
class Tick:
    t: float
    period: int
    entities: np.ndarray  # (N,) str
    x: np.ndarray
    y: np.ndarray
    speed_kmh: np.ndarray
    step_m: np.ndarray
    dt: float = 0.1


StreamItem = Event | Tick


def iter_ticks(tracking: pd.DataFrame, every: int = 1, t_from: float = 0.0, t_to: float = 1e9) -> Iterator[Tick]:
    """Group long-format tracking into per-frame ticks (``every`` = frame stride)."""
    tr = tracking[(tracking.t >= t_from) & (tracking.t <= t_to)]
    tr = tr[tr.entity != "ball"]
    frames = np.sort(tr.frame.unique())[::every]
    tr = tr[tr.frame.isin(frames)].sort_values(["frame", "entity"])
    ent = tr.entity.to_numpy()
    cols = {c: tr[c].to_numpy() for c in ("t", "period", "x", "y", "speed_kmh", "step_m", "frame")}
    bounds = np.flatnonzero(np.diff(cols["frame"])) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(ent)]
    for s, e in zip(starts, ends, strict=True):
        yield Tick(
            t=float(cols["t"][s]),
            period=int(cols["period"][s]),
            entities=ent[s:e],
            x=cols["x"][s:e],
            y=cols["y"][s:e],
            speed_kmh=cols["speed_kmh"][s:e],
            step_m=cols["step_m"][s:e] * every,
            dt=0.1 * every,
        )


class ReplaySource:
    """Merge events and tracking ticks into one time-ordered iterator."""

    def __init__(self, events: list[Event], tracking: pd.DataFrame | None = None, tick_every: int = 5,
                 t_from: float = 0.0, t_to: float = 1e9):
        self.events = [e for e in events if t_from <= e.t <= t_to]
        self.tracking = tracking
        self.tick_every = tick_every
        self.t_from, self.t_to = t_from, t_to

    def __iter__(self) -> Iterator[StreamItem]:
        ticks = iter_ticks(self.tracking, self.tick_every, self.t_from, self.t_to) if self.tracking is not None else iter(())
        nxt_tick = next(ticks, None)
        for e in self.events:
            while nxt_tick is not None and nxt_tick.t <= e.t:
                yield nxt_tick
                nxt_tick = next(ticks, None)
            yield e
        while nxt_tick is not None:
            yield nxt_tick
            nxt_tick = next(ticks, None)


class LiveSource:
    """Pace a replay in real time (``speed`` x faster) to simulate a live feed."""

    def __init__(self, replay: ReplaySource, speed: float = 1.0):
        self.replay = replay
        self.speed = speed

    async def stream(self) -> AsyncIterator[tuple[float, StreamItem]]:
        """Yield ``(arrival_wallclock, item)``; arrival time lets us measure event->overlay latency."""
        t0_wall = time.perf_counter()
        t0_match: float | None = None
        for item in self.replay:
            if t0_match is None:
                t0_match = item.t
            due = (item.t - t0_match) / self.speed
            delay = due - (time.perf_counter() - t0_wall)
            if delay > 0:
                await asyncio.sleep(delay)
            yield time.perf_counter(), item
