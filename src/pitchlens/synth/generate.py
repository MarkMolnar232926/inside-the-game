"""Synthetic match generation entry point and exporters.

Example::

    python -m pitchlens.synth.generate --seed 7 --narrative late_comeback --out data/synthetic/demo
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pitchlens.schema import Event, EventType, MatchMeta, Outcome

from .narratives import NARRATIVES
from .simulator import HZ, MatchSimulator, SimConfig
from .teams import build_teams


@dataclass
class SyntheticMatch:
    meta: MatchMeta
    events: list[Event]
    tracking: pd.DataFrame  # long format, one row per entity per 10 Hz frame

    # ------------------------------------------------------------------ io
    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(self.meta.model_dump_json(indent=2))
        with (out / "events.jsonl").open("w") as fh:
            for e in self.events:
                fh.write(e.model_dump_json() + "\n")
        pd.DataFrame([flatten_event(e) for e in self.events]).to_parquet(out / "events.parquet", index=False)
        self.tracking.to_parquet(out / "tracking.parquet", index=False)
        (out / "summary.json").write_text(json.dumps(summarise(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> SyntheticMatch:
        meta = MatchMeta.model_validate_json((path / "meta.json").read_text())
        events = [Event.model_validate_json(line) for line in (path / "events.jsonl").read_text().splitlines()]
        tracking = pd.read_parquet(path / "tracking.parquet")
        return cls(meta, events, tracking)


def flatten_event(e: Event) -> dict:
    d = e.model_dump(exclude={"pass_info", "shot_info", "score"})
    d["home_score"], d["away_score"] = e.score
    for prefix, sub in (("pass_", e.pass_info), ("shot_", e.shot_info)):
        if sub is not None:
            for k, v in sub.model_dump().items():
                d[prefix + k] = v
    return d


def tracking_frame(sim_out, meta: MatchMeta) -> pd.DataFrame:
    tr = sim_out.tracking
    n = len(tr["t"])
    pid = np.array(sim_out.player_index)
    team_lookup = {p.id: p.team_id for p in meta.all_players()}
    frames = np.arange(n)
    rows = []
    for slot in range(22):
        ids = pid[sim_out.slot_ids[:, slot]]
        rows.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "t": tr["t"],
                    "period": tr["period"],
                    "entity": ids,
                    "team_id": [team_lookup[i] for i in ids],
                    "x": tr["x"][:, slot],
                    "y": tr["y"][:, slot],
                    "z": np.zeros(n, np.float32),
                }
            )
        )
    rows.append(
        pd.DataFrame(
            {
                "frame": frames,
                "t": tr["t"],
                "period": tr["period"],
                "entity": "ball",
                "team_id": "",
                "x": tr["x"][:, 22],
                "y": tr["y"][:, 22],
                "z": tr["z"],
            }
        )
    )
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["entity", "frame"]).reset_index(drop=True)
    # Speeds from 10 Hz positions, lightly smoothed (centred 0.5 s window).
    g = df.groupby("entity", sort=False)
    dx = g["x"].diff().fillna(0)
    dy = g["y"].diff().fillna(0)
    step = np.hypot(dx, dy)
    step[step > 5] = 0  # teleports at substitutions / half time
    df["speed_kmh"] = (
        pd.Series(step * HZ * 3.6).groupby(df["entity"]).transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
    ).astype(np.float32)
    df["step_m"] = step.astype(np.float32)
    return df.sort_values(["frame", "entity"]).reset_index(drop=True)


def generate(seed: int = 7, narrative: str = "late_comeback", duration_s: float = 5400.0) -> SyntheticMatch:
    home, away = build_teams(seed)
    meta = MatchMeta(
        match_id=f"SYN-{seed:04d}-{narrative[:4].upper()}",
        venue="Harbourlight Arena (fictional)",
        kickoff_utc="2026-10-17T15:00:00Z",
        seed=seed,
        narrative=narrative,
        duration_s=duration_s,
        home=home,
        away=away,
    )
    sim = MatchSimulator(meta, NARRATIVES[narrative], SimConfig(duration_s=duration_s, seed=seed))
    out = sim.run()
    return SyntheticMatch(meta, out.events, tracking_frame(out, meta))


def summarise(m: SyntheticMatch) -> dict:
    ev = m.events
    by_team: dict[str, Counter] = {m.meta.home.id: Counter(), m.meta.away.id: Counter()}
    for e in ev:
        if e.team_id in by_team:
            by_team[e.team_id][e.type.value] += 1
            if e.type == EventType.PASS and e.outcome == Outcome.SUCCESS:
                by_team[e.team_id]["pass_completed"] += 1
    final = ev[-1].score
    out = {"match_id": m.meta.match_id, "narrative": m.meta.narrative, "final_score": list(final), "teams": {}}
    xg = Counter()
    for e in ev:
        if e.type == EventType.SHOT and e.shot_info:
            xg[e.team_id] += e.shot_info.xg
    for tid, c in by_team.items():
        passes = c["pass"]
        out["teams"][tid] = {
            "passes": passes,
            "pass_completion": round(c["pass_completed"] / max(passes, 1), 3),
            "shots": c["shot"],
            "goals": c["goal"],
            "xg": round(xg[tid], 2),
            "tackles": c["tackle"],
            "interceptions": c["interception"],
            "pressures": c["pressure"],
            "fouls": c["foul"],
            "corners": c["corner"],
        }
    total_poss_passes = sum(t["passes"] for t in out["teams"].values()) or 1
    for t in out["teams"].values():
        t["possession_share_by_passes"] = round(t["passes"] / total_poss_passes, 3)
    out["n_events"] = len(ev)
    out["tracking_rows"] = len(m.tracking)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--narrative", default="late_comeback", choices=sorted(NARRATIVES))
    ap.add_argument("--duration", type=float, default=5400.0, help="Seconds of match clock to simulate")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/demo"))
    a = ap.parse_args()
    m = generate(a.seed, a.narrative, a.duration)
    m.save(a.out)
    print(json.dumps(summarise(m), indent=2))


if __name__ == "__main__":
    main()
