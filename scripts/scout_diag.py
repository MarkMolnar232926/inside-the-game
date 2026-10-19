import time
from collections import Counter
from pathlib import Path

from pitchlens.agents.watcher import InsightScout, StreamWatcher
from pitchlens.ingest.sources import ReplaySource
from pitchlens.synth.generate import SyntheticMatch

m = SyntheticMatch.load(Path("data/synthetic/demo"))
w, sc = StreamWatcher(m.meta), InsightScout()
c = Counter(); ex = {}
t0 = time.perf_counter()
for item in ReplaySource(m.events, m.tracking, tick_every=5):
    cands = sc.on_event(w.state, item) if w.ingest(item) else []
    cands += sc.on_time(w.state)
    for cd in cands:
        c[cd.detector] += 1
        ex.setdefault(cd.detector, []).append((round(cd.t / 60, 1), cd.value_text, cd.player_id or cd.team_id))
print("elapsed", time.perf_counter() - t0, "events", w.n_events, "ticks", w.n_ticks)
for k, v in c.items():
    print(k, v, ex[k][:6])
