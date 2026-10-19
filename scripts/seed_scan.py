"""Scan seeds for a narrative and report storyline-relevant stats."""
import sys
from collections import Counter

import numpy as np

from pitchlens.schema import attack_x
from pitchlens.synth.generate import generate, summarise

nar = sys.argv[1] if len(sys.argv) > 1 else "late_comeback"
for seed in [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(1, 13):
    m = generate(seed, nar, 5400)
    g = [(round(e.t / 60), e.team_id) for e in m.events if e.type.value == "goal"]
    ati = Counter()
    for e in m.events:
        if e.type.value == "interception":
            right = (e.team_id == m.meta.home.id) == (e.period == 1)
            if attack_x(e.x, right) > 70:
                ati[e.player_id] += 1
    s = summarise(m)
    tr = m.tracking[m.tracking.entity == "KBY-07"].sort_values("frame")
    f = (tr.speed_kmh.to_numpy() >= 25).astype(int)
    d = np.diff(np.r_[0, f, 0]); spr = ((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) >= 10).sum()
    print(seed, m.events[-1].score, g, "K7 att3 int", ati.get("KBY-07"), "sprints", spr, "km", round(tr.step_m.sum() / 1000, 1),
          {k: (v["passes"], v["pass_completion"], v["shots"], v["xg"]) for k, v in s["teams"].items()}, flush=True)
