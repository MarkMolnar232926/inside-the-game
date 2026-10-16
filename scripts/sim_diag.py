"""Quick diagnostics for tuning the simulator rates."""
import sys
from collections import Counter

from pitchlens.synth.generate import generate, summarise

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
nar = sys.argv[2] if len(sys.argv) > 2 else "late_comeback"
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 5400
m = generate(seed, nar, dur)
s = summarise(m)
print(s["final_score"], {k: {kk: v[kk] for kk in ("passes", "pass_completion", "shots", "goals", "xg", "tackles", "interceptions", "pressures")} for k, v in s["teams"].items()})
c = Counter(e.type.value for e in m.events)
print(dict(c))
fails = Counter()
evs = m.events
for i, e in enumerate(evs):
    if e.type.value == "pass" and e.outcome.value == "fail":
        nxt = next((x for x in evs[i + 1:] if x.type.value not in ("pressure",)), None)
        fails[nxt.type.value if nxt else "none"] += 1
print("after failed pass:", dict(fails))
import numpy as np
L = [e.pass_info.length_m for e in evs if e.type.value == "pass"]
print("pass len mean/median", np.mean(L), np.median(L))
inplay = sum(1 for e in evs if e.type.value == "pass")
tr = m.tracking
pl = tr[tr.entity != "ball"]
dist = pl.groupby("entity")["step_m"].sum().sort_values()
print("dist km top/bottom", (dist.tail(3) / 1000).round(2).to_dict(), (dist.head(3) / 1000).round(2).to_dict())
print("max speed", pl.speed_kmh.max())
# attacking third interceptions per player
from pitchlens.schema import attack_x
ati = Counter()
for e in evs:
    if e.type.value == "interception":
        home = e.team_id == m.meta.home.id
        right = home == (e.period == 1)
        if attack_x(e.x, right) > 70:
            ati[e.player_id] += 1
print("att-third interceptions", ati.most_common(5))
