# Data card: PitchLens synthetic match dataset

## Summary

| | |
|---|---|
| **Name** | PitchLens Synthetic Match (`SYN-<seed>-<narrative>`) |
| **Generator** | `pitchlens.synth` v1.0, agent-based 10 Hz simulation |
| **Contains real data?** | **No.** Every club, player, venue, competition, event and coordinate is generated. No real league, broadcast or tracking data was used to build or tune it. |
| **Licence** | MIT (same as the code) |
| **Reproducible** | Yes. Seeded RNG: the same `(seed, narrative, duration)` gives byte-identical events (`tests/test_generator.py::test_deterministic`). |
| **Demo instance** | `seed=6, narrative=late_comeback` (see `docs/DECISIONS.md` D6) |

## What is generated

1. **Teams and squads** (`synth/teams.py`). Two fictional clubs, **Kestrel Bay FC (KBY)** and **Ironvale Athletic (IRV)**. Each has 11 starters and 5 substitutes, with positions, tactical roles, shirt numbers, kits and fictional attributes (`passing, vision, pressing, tackling, pace, stamina, finishing, dribbling`) that drive behaviour. Names are combined from hand-written syllable pools.
2. **Events** (`events.jsonl`, `events.parquet`). One `Event` per action. Types: `pass, carry, shot, goal, save, tackle, interception, duel, pressure, ball_recovery, clearance, foul, free_kick, corner, throw_in, goal_kick, out_of_play, possession_change, substitution, kick_off, period_start/end`. Every event has the match clock, period, team, player, pitch `x, y` (m), optional end location, outcome, possession id and score.
   * `pass_info`: length, progressive metres, **difficulty** (1 - expected completion), height, under-pressure flag, ball speed, key-pass flag.
   * `shot_info`: **xG** (distance/angle model), **shot speed** (km/h), body part, on-target flag, distance.
   * `source`: `synthetic`. Vision-derived events use `vision_estimated` and carry a `confidence`.
3. **Tracking** (`tracking.parquet`). 10 Hz positions for 22 players and the ball (with height `z`), plus derived `speed_kmh` (0.5 s smoothing) and `step_m`. Substitutes appear from the moment they come on.
4. **Metadata** (`meta.json`). Squads, kits, attributes, seed, narrative and duration.
5. **Summary** (`summary.json`). Per-team rates for quick sanity checks.

## How it is generated

Players move towards tactical targets: 4-3-3 slots that shift with the ball, compact out of possession and wide in possession. Speed and acceleration limits come from `pace` and `stamina`, with fatigue over time. The two to three defenders closest to the ball **press** it, more intensely in their attacking third. A presser within 2.6 m creates a `pressure` event and may attempt a tackle, which can become a foul.

The ball carrier holds the ball for a tempo-dependent time and then chooses:

* **Shoot**, with probability rising with xG.
* **Clear**, when deep and under pressure.
* **Carry**, when there is space ahead.
* **Pass**, via softmax over progression, receiver openness, lane safety and length.

Pass success depends on length, lane blocking, pressure and skill. Failed passes become **interceptions** (the opponent nearest the lane, whose reach grows with *anticipation* = pressing x vision), loose-ball **recoveries**, or **out of play**, which leads to throw-in, goal-kick or corner restarts with realistic dead-ball time.

### Narratives

Narratives (`synth/narratives.py`) modulate both teams over the match clock:

* `balanced`
* `dominant_home`
* `late_comeback`: the visitors lead, then the home press surges after 60'.
* `chaotic_end_to_end`

An optional "script" raises conversion of decent chances (xG >= 0.08) inside time windows, so the storyline usually happens without inventing goals from nothing.

## Typical rates (full 90', `late_comeback`, seeds 1-6)

| metric | per team | real elite-league reference (approx.) |
|---|---|---|
| passes | 560-700 | 400-650 |
| pass completion | 77-82% | 78-88% |
| shots | 7-17 | 8-18 |
| xG | 0.5-2.4 | 0.6-2.5 |
| goals per match | 2-6 | ~2.8 |
| tackles | 18-35 | 15-25 |
| sprints > 25 km/h per player | 15-35 | 20-60 |
| distance per outfield starter | 8.5-13.5 km | 9-13 km |

## Known limitations (honest list)

* **Interceptions** (~35-50 per team) run higher than typical provider counts (~10-20), because every cut-out pass is an interception. There is no separate "block" or "loose duel" type.
* There are no aerial duels, offsides, cards or set-piece routines (a corner is a restart, not a delivery). Goalkeeper distribution is simple.
* xG comes from a distance/angle model, not a trained model. Defenders do not block shots geometrically (a blocked shot is a random outcome).
* Player motion is target-following plus noise, with no explicit body orientation or first-touch physics.
* The referee is rendered only in the synthetic footage. It is not part of the dataset.

## Video alignment

* **Synthetic footage**: `pitchlens.footage.broadcast` renders the tracking itself, so events are perfectly aligned. The renderer also exports ground-truth camera and boxes, used *only* for evaluation.
* **Real clips**: `scripts/build_demo.py --video ... --match-t0 <clock>` aligns the synthetic event clock to the media clock. A track anchor map (`{"track_id": "KBY-07"}`) binds our fictional player to a tracked person, so events for that player can be re-anchored to visible positions. See `docs/REAL_CLIPS.md`.

## Intended use

Developing and demonstrating match-intelligence pipelines (stats, explainability, overlays, personalisation) without licensed data. **Not** for betting, scouting or any claim about real teams or players.
