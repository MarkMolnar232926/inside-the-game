"""Agent-based synthetic match simulator.

Produces a football-realistic event stream *and* 10 Hz player/ball tracking
from the same simulated physics, so events, positions and speeds are
mutually consistent (useful for video alignment and for vision tests).

Model sketch (deliberately simple and explainable):

* 22 players move towards tactical targets (formation slot shifted with the
  ball, team compactness in/out of possession) under speed + acceleration
  limits derived from fictional ``pace`` and ``stamina``.
* Defenders closest to the ball press it; intensity depends on the team's
  narrative ``press`` modifier and the zone (higher press in their attacking
  third). Pressers within 3 m create ``pressure`` events and may tackle.
* The ball carrier holds the ball for a tempo-dependent time, then chooses to
  shoot, pass (softmax over progression / openness / lane safety) or carry.
  Pass success depends on length, lane blocking, pressure and skill; failed
  passes become interceptions, loose-ball recoveries or out-of-play.
* Restarts (throw-ins, goal kicks, corners, free kicks, kick-offs) insert
  dead-ball time so ball-in-play time is realistic (~55-60 min per 90).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pitchlens.schema import (
    PITCH_LENGTH as L,
)
from pitchlens.schema import (
    PITCH_WIDTH as W,
)
from pitchlens.schema import (
    Event,
    EventType,
    MatchMeta,
    Outcome,
    PassInfo,
    Player,
    Position,
    ShotInfo,
)

from .narratives import Narrative, TeamMods

DT = 0.1
HZ = 10

# Attack-normalised base slots (team attacks towards +x) for a 4-3-3.
BASE_SLOTS = np.array(
    [
        [6, 34],  # GK
        [30, 9],  # RB
        [26, 25],  # CB
        [26, 43],  # CB
        [30, 59],  # LB
        [38, 34],  # DM
        [46, 22],  # CM
        [52, 44],  # AM
        [62, 10],  # RW
        [62, 58],  # LW
        [66, 34],  # ST
    ],
    dtype=float,
)
# How strongly each slot follows the ball (x, y)
FOLLOW = np.array(
    [[0.08, 0.15]] + [[0.55, 0.35]] * 4 + [[0.6, 0.45]] * 3 + [[0.6, 0.3]] * 3,
    dtype=float,
)


@dataclass
class Flight:
    kind: str  # pass | shot | clearance
    t0: float
    t1: float
    start: np.ndarray
    end: np.ndarray
    apex: float
    passer: int
    receiver: int | None  # slot that will get the ball (teammate or opponent)
    result: str  # complete | intercept | loose | out | goal | saved | blocked | wide
    pass_event_id: str | None = None
    shot_xg: float = 0.0


@dataclass
class DeadBall:
    restart: EventType
    team: int
    spot: np.ndarray
    resume_t: float
    taker: int | None = None


@dataclass
class SimConfig:
    duration_s: float = 5400.0
    seed: int = 7
    stoppage_s: float = 180.0


@dataclass
class SimOutput:
    events: list[Event]
    tracking: dict[str, np.ndarray]  # t, period, x[n, 23], y, z(ball), ids per frame
    slot_ids: np.ndarray  # (frames, 22) player id index per slot
    player_index: list[str] = field(default_factory=list)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance of points p (N,2) to segment a-b, and the projection parameter u."""
    ab = b - a
    denom = float(ab @ ab) or 1e-6
    u = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + u[:, None] * ab
    return np.linalg.norm(p - proj, axis=1), u


class MatchSimulator:
    def __init__(self, meta: MatchMeta, narrative: Narrative, cfg: SimConfig):
        self.meta = meta
        self.nar = narrative
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        teams = [meta.home, meta.away]
        self.team_ids = [t.id for t in teams]
        starters = [[p for p in t.players if p.starter] for t in teams]
        self.bench = [[p for p in t.players if not p.starter] for t in teams]
        self.slot_player: list[Player] = starters[0] + starters[1]
        self.player_index = [p.id for p in meta.all_players()]
        self._pidx = {pid: i for i, pid in enumerate(self.player_index)}
        self.team_of = np.array([0] * 11 + [1] * 11)
        self.slot_in_team = np.array(list(range(11)) * 2)
        self._load_attrs()
        self.pos = np.zeros((22, 2))
        self.vel = np.zeros((22, 2))
        self.noise = np.zeros((22, 2))
        self.dist_run = np.zeros(22)
        self.events: list[Event] = []
        self.seq = 0
        self.score = [0, 0]
        self.possession_team = 0
        self.possession_id = 0
        self.owner: int | None = None
        self.flight: Flight | None = None
        self.dead: DeadBall | None = None
        self.ball = np.array([L / 2, W / 2])
        self.ball_z = 0.0
        self.hold_until = 0.0
        self.received_at = np.array([L / 2, W / 2])
        self.pressured_by: set[int] = set()
        self.period = 1
        self.t = 0.0
        self.subs_done = [0, 0]
        self.sub_times = [sorted(self.rng.uniform(3500, 4900, 3)) for _ in range(2)]
        self.goals_scripted = list(narrative.script)

    # ------------------------------------------------------------------ setup
    def _load_attrs(self) -> None:
        def arr(name: str) -> np.ndarray:
            return np.array([getattr(p.attributes, name) for p in self.slot_player])

        self.a_pace, self.a_pass, self.a_press = arr("pace"), arr("passing"), arr("pressing")
        self.a_tackle, self.a_vision, self.a_fin = arr("tackling"), arr("vision"), arr("finishing")
        self.a_drib, self.a_stam = arr("dribbling"), arr("stamina")

    def attacks_right(self, team: int) -> bool:
        return (team == 0) == (self.period == 1)

    def norm(self, xy: np.ndarray, team: int) -> np.ndarray:
        """Pitch -> attack-normalised coordinates for ``team`` (180 deg rotation)."""
        if self.attacks_right(team):
            return xy
        return np.array([L, W]) - xy

    def denorm(self, xy: np.ndarray, team: int) -> np.ndarray:
        return self.norm(xy, team)  # rotation is its own inverse

    def mods(self, team: int) -> TeamMods:
        return self.nar.mods(team, self.t)

    # ----------------------------------------------------------------- events
    def emit(self, etype: EventType, slot: int | None = None, xy=None, **kw) -> Event:
        self.seq += 1
        team_id = player_id = None
        if slot is not None:
            team_id = self.team_ids[self.team_of[slot]]
            player_id = self.slot_player[slot].id
        elif "team" in kw:
            team_id = self.team_ids[kw.pop("team")]
        kw.pop("team", None)
        x = y = None
        if xy is not None:
            x, y = float(np.clip(xy[0], -1.5, L + 1.5)), float(np.clip(xy[1], -1.5, W + 1.5))
        ev = Event(
            id=f"{self.meta.match_id}-e{self.seq:05d}",
            match_id=self.meta.match_id,
            seq=self.seq,
            t=round(self.t, 2),
            period=self.period,
            type=etype,
            team_id=team_id,
            player_id=player_id,
            x=None if x is None else round(x, 2),
            y=None if y is None else round(y, 2),
            possession_id=self.possession_id,
            score=(self.score[0], self.score[1]),
            **kw,
        )
        self.events.append(ev)
        return ev

    def set_possession(self, team: int, slot: int | None, xy: np.ndarray) -> None:
        if team != self.possession_team:
            self.possession_team = team
            self.possession_id += 1
            self.emit(EventType.POSSESSION_CHANGE, slot, xy, team=team)

    def give_ball(self, slot: int, hold_scale: float = 1.0) -> None:
        self.owner = slot
        self.flight = None
        self.received_at = self.pos[slot].copy()
        self.pressured_by = set()
        tempo = self.mods(self.team_of[slot]).tempo
        self.hold_until = self.t + hold_scale * float(self.rng.gamma(2.4, 0.85)) / tempo + 0.4

    # ---------------------------------------------------------------- targets
    def targets(self) -> np.ndarray:
        tgt = np.zeros((22, 2))
        ball = self.ball
        for team in (0, 1):
            idx = np.where(self.team_of == team)[0]
            bn = self.norm(ball, team)
            m = self.mods(team)
            in_pos = self.possession_team == team
            base = BASE_SLOTS.copy()
            shift_x = (bn[0] - L / 2) * FOLLOW[:, 0]
            shift_y = (bn[1] - W / 2) * FOLLOW[:, 1]
            if in_pos:
                base[1:, 0] += 7 + m.line_height * 0.5
                base[1:, 1] = W / 2 + (base[1:, 1] - W / 2) * 1.12
            else:
                base[1:, 0] += -4 + m.line_height
                base[1:, 1] = W / 2 + (base[1:, 1] - W / 2) * 0.78
            t_norm = base + np.c_[shift_x, shift_y]
            # Forwards in possession drift towards the space behind the line.
            if in_pos:
                t_norm[8:, 0] += 4 * np.sin(self.t / 9 + np.arange(3))
            t_norm[0, 0] = np.clip(4 + 0.07 * bn[0], 3, 14)
            t_norm[0, 1] = W / 2 + (bn[1] - W / 2) * 0.18
            t_norm[:, 0] = np.clip(t_norm[:, 0], 2, L - 3)
            t_norm[:, 1] = np.clip(t_norm[:, 1], 2, W - 2)
            for k, slot in enumerate(idx):
                tgt[slot] = self.denorm(t_norm[k], team)
        tgt += self.noise
        # Pressing: nearest defenders hunt the ball.
        if self.dead is None:
            d_team = 1 - self.possession_team
            m = self.mods(d_team)
            didx = np.where(self.team_of == d_team)[0][1:]
            d = np.linalg.norm(self.pos[didx] - ball, axis=1)
            order = didx[np.argsort(d)]
            bn = self.norm(ball, d_team)
            zone_boost = 0.25 if bn[0] > 2 * L / 3 else (0.0 if bn[0] > L / 3 else -0.1)
            n_press = 1 + int(m.press + zone_boost > 0.6) + int(m.press + zone_boost > 0.85)
            for j, slot in enumerate(order[:n_press]):
                intensity = np.clip(m.press + zone_boost + 0.3 * self.a_press[slot], 0, 1.4)
                if j == 0 or intensity > 0.75:
                    goal_side = self.denorm(np.array([0.0, W / 2]), d_team) - ball
                    goal_side /= np.linalg.norm(goal_side) + 1e-6
                    tgt[slot] = ball + goal_side * (1.6 if j == 0 else 4.0)
        if self.owner is not None:
            o = self.owner
            team = self.team_of[o]
            on = self.norm(self.pos[o], team)
            goal = np.array([L, W / 2])
            direction = goal - on
            direction /= np.linalg.norm(direction) + 1e-6
            lateral = np.array([0.0, np.sin(self.t * 0.7 + o)]) * 0.4
            tgt[o] = self.denorm(on + (direction + lateral) * 2.5, team)
        if self.flight is not None and self.flight.receiver is not None:
            tgt[self.flight.receiver] = self.flight.end
        if self.dead is not None:
            if self.dead.restart == EventType.KICK_OFF:
                for team in (0, 1):
                    idx = np.where(self.team_of == team)[0]
                    kn = BASE_SLOTS.copy()
                    kn[:, 0] = np.minimum(kn[:, 0] * 0.75, L / 2 - 2)
                    for k, slot in enumerate(idx):
                        tgt[slot] = self.denorm(kn[k], team)
            if self.dead.taker is not None:
                tgt[self.dead.taker] = self.dead.spot
        return tgt

    # ----------------------------------------------------------------- motion
    def move(self, tgt: np.ndarray) -> None:
        fatigue = 1 - 0.12 * (self.t / 5400) * (1 - self.a_stam)
        vmax = (6.2 + 2.6 * self.a_pace) * fatigue
        if self.owner is not None:
            vmax[self.owner] *= 0.72  # dribbling is slower than running
        delta = tgt - self.pos
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        # Jog when roughly in position, sprint only when far from the target.
        cruise = np.minimum(vmax[:, None], 0.8 + 0.32 * dist)
        desired = delta / (dist + 1e-6) * cruise
        acc = np.clip(desired - self.vel, -3.5 * DT, 3.5 * DT)
        self.vel += acc
        spd = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = np.where(spd > vmax[:, None], self.vel / (spd + 1e-9) * vmax[:, None], self.vel)
        step = self.vel * DT
        self.pos += step
        self.dist_run += np.linalg.norm(step, axis=1)
        self.pos[:, 0] = np.clip(self.pos[:, 0], -2, L + 2)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -2, W + 2)
        # Ornstein-Uhlenbeck jitter on targets for natural off-ball movement.
        self.noise += -0.04 * self.noise + self.rng.normal(0, 0.18, self.noise.shape)

    # ---------------------------------------------------------------- actions
    def decide(self) -> None:
        o = self.owner
        assert o is not None
        team = self.team_of[o]
        m = self.mods(team)
        on = self.norm(self.pos[o], team)
        opp = np.where(self.team_of != team)[0]
        mates = np.where((self.team_of == team) & (np.arange(22) != o))[0]
        pressure = float(np.min(np.linalg.norm(self.pos[opp] - self.pos[o], axis=1)))
        under_pressure = pressure < 3.0

        # Carry event if the player has moved the ball meaningfully.
        carried = float(np.linalg.norm(self.pos[o] - self.received_at))
        if carried > 5:
            self.emit(
                EventType.CARRY, o, self.received_at, end_x=float(self.pos[o][0]), end_y=float(self.pos[o][1]),
                outcome=Outcome.SUCCESS,
            )

        # Shot?
        dist_goal = float(np.linalg.norm(on - np.array([L, W / 2])))
        if on[0] > 78 and abs(on[1] - W / 2) < 22 and dist_goal < 30:
            xg = self.xg(on, under_pressure)
            p_shoot = np.clip(m.shot_appetite * (xg * 1.1 + 0.008) + 0.025 * self.a_fin[o], 0, 0.6)
            if self.rng.random() < p_shoot:
                self.shoot(o, team, on, xg, under_pressure)
                return

        # Clearance under pressure deep in own half.
        if on[0] < 22 and under_pressure and self.rng.random() < 0.35:
            target_n = np.array([self.rng.uniform(50, 75), self.rng.uniform(8, W - 8)])
            self.launch(o, team, None, self.denorm(target_n, team), kind="clearance", p_complete=0.0)
            return

        # Pass selection.
        mp = self.pos[mates]
        mn = np.array([self.norm(p, team) for p in mp])
        prog = mn[:, 0] - on[0]
        dist = np.linalg.norm(mp - self.pos[o], axis=1)
        open_ = np.array([np.min(np.linalg.norm(self.pos[opp] - p, axis=1)) for p in mp])
        lane = np.array([self._lane_block(self.pos[o], p, opp)[0] for p in mp])
        score = (
            0.07 * prog * m.risk * (0.6 + self.a_vision[o])
            + 0.28 * np.minimum(open_, 8)
            + 0.35 * np.minimum(lane, 6)
            - 0.11 * np.maximum(dist - 12, 0)
        )
        score[dist > 55] = -9
        if self.slot_in_team[o] != 0:
            score[self.slot_in_team[mates] == 0] -= 2.5  # rarely back to the keeper
        # Carry instead of passing when there is space ahead.
        space_ahead = pressure > 6 and on[0] < 88
        if space_ahead and self.rng.random() < 0.18 + 0.3 * self.a_drib[o] - 0.1 * m.tempo:
            self.hold_until = self.t + float(self.rng.uniform(0.8, 2.0))
            return
        prob = np.exp((score - score.max()) / 0.7)
        prob /= prob.sum()
        k = int(self.rng.choice(len(mates), p=prob))
        recv = int(mates[k])
        flight_t = dist[k] / self._pass_speed(dist[k])
        lead = self.pos[recv] + self.vel[recv] * flight_t * 0.6
        lane_d, _ = self._lane_block(self.pos[o], lead, opp)
        p = self.pass_probability(o, float(np.linalg.norm(lead - self.pos[o])), lane_d, under_pressure, m)
        self.launch(o, team, recv, lead, kind="pass", p_complete=p, under_pressure=under_pressure)

    def _lane_block(self, a: np.ndarray, b: np.ndarray, opp: np.ndarray) -> tuple[float, int]:
        d, u = _seg_dist(self.pos[opp], a, b)
        d = np.where((u > 0.05) & (u < 0.98), d, d + 3)
        # Anticipation: players who press and read the game well cut out passes from further away.
        d = d - 1.6 * self.a_press[opp] * self.a_vision[opp]
        k = int(np.argmin(d))
        return float(d[k]), int(opp[k])

    def _pass_speed(self, dist: float) -> float:
        return float(np.clip(9 + 0.32 * dist, 9, 26))

    def pass_probability(self, o: int, dist: float, lane: float, pressured: bool, m: TeamMods) -> float:
        base = 0.93 + 0.06 * self.a_pass[o]
        p = base * math.exp(-max(dist - 12, 0) / 130) * (1 - 0.35 * math.exp(-max(lane, 0.0) / 1.0))
        if pressured:
            p *= 0.9
        return float(np.clip(p * m.pass_acc, 0.05, 0.995))

    def xg(self, on: np.ndarray, pressured: bool) -> float:
        dx = L - on[0]
        dy = abs(on[1] - W / 2)
        dist = math.hypot(dx, dy)
        angle = abs(math.atan2(7.32 * dx, dx * dx + dy * dy - (7.32 / 2) ** 2))
        xg = 0.6 * math.exp(-dist / 6.0) * (0.25 + 0.75 * min(angle / 0.7, 1))
        return float(np.clip(xg * (0.75 if pressured else 1.0), 0.01, 0.85))

    def launch(
        self,
        o: int,
        team: int,
        recv: int | None,
        end: np.ndarray,
        kind: str,
        p_complete: float,
        under_pressure: bool = False,
    ) -> None:
        start = self.pos[o].copy()
        end = end.copy()
        dist = float(np.linalg.norm(end - start))
        opp = np.where(self.team_of != team)[0]
        high = kind == "clearance" or dist > 32
        lane_d, blocker = self._lane_block(start, end, opp)
        if high:
            lane_d += 3.0  # lofted balls clear most lanes
        speed = 21.0 if high else self._pass_speed(dist)
        roll = self.rng.random()
        result, receiver = "complete", recv
        if roll > p_complete:
            if lane_d < 1.2:
                result, receiver = "intercept", blocker
                _, u = _seg_dist(self.pos[blocker][None], start, end)
                end = start + (end - start) * float(np.clip(u[0], 0.15, 0.95))
            elif end[0] < 0 or end[0] > L or end[1] < 0 or end[1] > W or self.rng.random() < 0.25:
                result, receiver = "out", None
                if not (0 <= end[0] <= L and 0 <= end[1] <= W):
                    pass
                else:
                    side = 0.0 if end[1] < W / 2 else W
                    end = np.array([end[0], side + (-0.5 if side == 0 else 0.5)])
            else:
                result = "loose"
                receiver = None
        if kind == "clearance":
            result, receiver = "loose", None
        dist = float(np.linalg.norm(end - start))
        duration = dist / speed + (0.4 if high else 0.0)
        pn_s, pn_e = self.norm(start, team), self.norm(end, team)
        ev = self.emit(
            EventType.CLEARANCE if kind == "clearance" else EventType.PASS,
            o,
            start,
            end_x=float(np.clip(end[0], -1.5, L + 1.5)),
            end_y=float(np.clip(end[1], -1.5, W + 1.5)),
            outcome=Outcome.SUCCESS if result == "complete" else Outcome.FAIL,
            recipient_id=self.slot_player[recv].id if (recv is not None and result == "complete") else None,
            pass_info=PassInfo(
                length_m=round(dist, 1),
                progressive_m=round(float(pn_e[0] - pn_s[0]), 1),
                difficulty=round(float(np.clip(1 - p_complete, 0, 1)), 3),
                height="high" if high else ("low" if dist > 22 else "ground"),
                under_pressure=under_pressure,
                ball_speed_kmh=round(speed * 3.6 * (1.15 if high else 1.0), 1),
                key_pass=False,
                expected_completion=round(float(np.clip(p_complete, 0, 1)), 3),
            ),
        )
        self.owner = None
        self.flight = Flight(
            kind, self.t, self.t + duration, start, end, 0.18 * dist if high else 0.0, o, receiver, result, ev.id
        )

    def shoot(self, o: int, team: int, on: np.ndarray, xg: float, pressured: bool) -> None:
        m = self.mods(team)
        p_goal = xg * m.finishing * (0.5 + 0.35 * self.a_fin[o])
        for g in self.goals_scripted:
            if g.team == team and g.after_s <= self.t <= g.before_s and xg >= 0.08:
                p_goal = max(p_goal, 0.5)
        r = self.rng.random()
        if r < p_goal:
            result = "goal"
        else:
            r2 = self.rng.random()
            result = "saved" if r2 < 0.38 else ("blocked" if r2 < 0.62 else "wide")
        goal_n = np.array([L + 0.5, W / 2 + self.rng.uniform(-3.2, 3.2)])
        if result == "wide":
            goal_n[1] = W / 2 + self.rng.choice([-1, 1]) * self.rng.uniform(4.5, 9)
        if result == "blocked":
            goal_n = on + (np.array([L, W / 2]) - on) * 0.25
        end = self.denorm(goal_n, team)
        speed_kmh = float(np.clip(self.rng.normal(88, 14) + 18 * self.a_fin[o], 45, 128))
        dist = float(np.linalg.norm(end - self.pos[o]))
        ev = self.emit(
            EventType.SHOT,
            o,
            self.pos[o],
            end_x=float(np.clip(end[0], -1.5, L + 1.5)),
            end_y=float(np.clip(end[1], -1.5, W + 1.5)),
            outcome=Outcome.SUCCESS if result == "goal" else Outcome.FAIL,
            shot_info=ShotInfo(
                xg=round(xg, 3),
                speed_kmh=round(speed_kmh, 1),
                body_part="head" if self.rng.random() < 0.12 else "right_foot",
                on_target=result in ("goal", "saved"),
                distance_m=round(float(np.linalg.norm(on - np.array([L, W / 2]))), 1),
            ),
        )
        # Mark the preceding completed pass as a key pass (if same possession).
        for prev in reversed(self.events[:-1]):
            if prev.possession_id != self.possession_id:
                break
            if prev.type == EventType.PASS and prev.outcome == Outcome.SUCCESS and prev.recipient_id == ev.player_id:
                assert prev.pass_info is not None
                prev.pass_info.key_pass = True
                break
        self.owner = None
        self.flight = Flight(
            "shot", self.t, self.t + dist / (speed_kmh / 3.6), self.pos[o].copy(), end, 0.4, o, None, result,
            ev.id, xg,
        )

    # ------------------------------------------------------------- resolution
    def resolve_flight(self) -> None:
        f = self.flight
        assert f is not None
        team = self.team_of[f.passer]
        if f.kind == "shot":
            self.resolve_shot(f, team)
            return
        if f.result == "complete" and f.receiver is not None:
            self.give_ball(f.receiver)
        elif f.result == "intercept" and f.receiver is not None:
            r = f.receiver
            self.emit(EventType.INTERCEPTION, r, f.end, outcome=Outcome.SUCCESS,
                      opponent_id=self.slot_player[f.passer].id)
            self.set_possession(int(self.team_of[r]), r, f.end)
            self.give_ball(r, 0.8)
        elif f.result == "out":
            self.out_of_play(f.end, last_team=team)
        else:  # loose ball: nearest player wins it
            d = np.linalg.norm(self.pos - f.end, axis=1)
            d += self.rng.uniform(0, 3, 22)
            r = int(np.argmin(d))
            if not (0 <= f.end[0] <= L and 0 <= f.end[1] <= W):
                self.out_of_play(f.end, last_team=team)
                return
            self.emit(EventType.BALL_RECOVERY, r, f.end, outcome=Outcome.SUCCESS)
            self.set_possession(int(self.team_of[r]), r, f.end)
            self.ball = f.end.copy()
            self.give_ball(r)

    def resolve_shot(self, f: Flight, team: int) -> None:
        opp = 1 - team
        gk = int(np.where((self.team_of == opp) & (self.slot_in_team == 0))[0][0])
        if f.result == "goal":
            self.score[team] += 1
            self.emit(EventType.GOAL, f.passer, f.end, outcome=Outcome.SUCCESS, note=f"xg={f.shot_xg:.2f}")
            self.goals_scripted = [
                g for g in self.goals_scripted if not (g.team == team and g.after_s <= self.t <= g.before_s)
            ]
            self.start_dead(EventType.KICK_OFF, opp, np.array([L / 2, W / 2]), 40)
        elif f.result == "saved":
            self.emit(EventType.SAVE, gk, f.end, outcome=Outcome.SUCCESS, opponent_id=self.slot_player[f.passer].id)
            if self.rng.random() < 0.35:
                self.corner(team, f.end)
            else:
                self.set_possession(opp, gk, self.pos[gk])
                self.give_ball(gk, 2.5)
        elif f.result == "blocked":
            d = np.linalg.norm(self.pos - f.end, axis=1)
            d[self.team_of == team] += 2
            r = int(np.argmin(d))
            self.emit(EventType.BALL_RECOVERY, r, f.end, outcome=Outcome.SUCCESS, note="blocked shot")
            self.set_possession(int(self.team_of[r]), r, f.end)
            self.give_ball(r)
        else:
            self.out_of_play(f.end, last_team=team)

    def corner(self, team: int, near: np.ndarray) -> None:
        cx = L if near[0] > L / 2 else 0.0
        cy = 0.0 if near[1] < W / 2 else W
        self.start_dead(EventType.CORNER, team, np.array([cx, cy]), 18)

    def out_of_play(self, at: np.ndarray, last_team: int) -> None:
        self.emit(EventType.OUT_OF_PLAY, None, at, team=last_team)
        other = 1 - last_team
        if at[1] <= 0 or at[1] >= W:
            spot = np.array([np.clip(at[0], 1, L - 1), 0.0 if at[1] <= 0 else W])
            self.start_dead(EventType.THROW_IN, other, spot, 7)
        else:
            right_end = at[0] >= L
            attack_right_team = 0 if self.attacks_right(0) else 1
            # Ball out over the goal line of the team defending that end.
            defending = 1 - attack_right_team if right_end else attack_right_team
            if last_team == defending:
                self.corner(other, at)
            else:
                gx = L - 6 if right_end else 6.0
                self.start_dead(EventType.GOAL_KICK, defending, np.array([gx, W / 2 + 6]), 12)

    def start_dead(self, restart: EventType, team: int, spot: np.ndarray, base_wait: float) -> None:
        self.owner = None
        self.flight = None
        idx = np.where(self.team_of == team)[0]
        if restart == EventType.GOAL_KICK:
            taker = int(idx[0])
        elif restart == EventType.KICK_OFF:
            taker = int(idx[10])
        else:
            d = np.linalg.norm(self.pos[idx[1:]] - spot, axis=1)
            taker = int(idx[1:][np.argmin(d)])
        self.ball = spot.astype(float).copy()
        self.dead = DeadBall(restart, team, spot.astype(float), self.t + base_wait * self.rng.uniform(0.7, 1.4), taker)

    def maybe_restart(self) -> None:
        d = self.dead
        assert d is not None
        if self.t < d.resume_t or d.taker is None:
            return
        if np.linalg.norm(self.pos[d.taker] - d.spot) > 1.5 and self.t < d.resume_t + 15:
            return
        self.pos[d.taker] = d.spot.copy()
        self.dead = None
        self.set_possession(d.team, d.taker, d.spot)
        self.emit(d.restart, d.taker, d.spot, team=d.team)
        self.give_ball(d.taker, 0.2)
        self.hold_until = self.t + 0.3

    # ------------------------------------------------------------ duels/press
    def contest(self) -> None:
        o = self.owner
        assert o is not None
        team = self.team_of[o]
        opp = np.where(self.team_of != team)[0]
        d = np.linalg.norm(self.pos[opp] - self.pos[o], axis=1)
        for k in np.where(d < 2.6)[0]:
            s = int(opp[k])
            if s not in self.pressured_by and (not self.pressured_by or self.rng.random() < 0.3):
                self.pressured_by.add(s)
                self.emit(EventType.PRESSURE, s, self.pos[o], opponent_id=self.slot_player[o].id)
        close = opp[d < 2.0]
        for s in close:
            m = self.mods(self.team_of[s])
            if self.rng.random() < 0.002 + 0.0045 * m.press + 0.003 * self.a_tackle[s]:
                if self.rng.random() < 0.2:
                    self.emit(EventType.FOUL, int(s), self.pos[o], opponent_id=self.slot_player[o].id)
                    self.start_dead(EventType.FREE_KICK, int(team), self.pos[o].copy(), 12)
                    return
                p_win = np.clip(0.5 + 0.35 * (self.a_tackle[s] - self.a_drib[o]), 0.2, 0.8)
                if self.rng.random() < p_win:
                    self.emit(EventType.TACKLE, int(s), self.pos[o], outcome=Outcome.SUCCESS,
                              opponent_id=self.slot_player[o].id)
                    self.set_possession(int(self.team_of[s]), int(s), self.pos[o])
                    self.give_ball(int(s), 0.7)
                else:
                    self.emit(EventType.TACKLE, int(s), self.pos[o], outcome=Outcome.FAIL,
                              opponent_id=self.slot_player[o].id)
                    self.emit(EventType.DUEL, o, self.pos[o], outcome=Outcome.SUCCESS,
                              opponent_id=self.slot_player[int(s)].id)
                    self.vel[s] *= 0.2
                return

    # ------------------------------------------------------------------ subs
    def maybe_sub(self) -> None:
        for team in (0, 1):
            k = self.subs_done[team]
            if k >= 3 or self.t < self.sub_times[team][k] or self.owner is not None and self.team_of[self.owner] == team:
                continue
            idx = np.where(self.team_of == team)[0]
            candidates = [s for s in idx[1:]]
            out = max(candidates, key=lambda s: self.dist_run[s] * (1.2 - self.a_stam[s]))
            bench = [p for p in self.bench[team] if p.position != Position.GK]
            if not bench:
                continue
            incoming = bench.pop(0)
            self.bench[team].remove(incoming)
            old = self.slot_player[out]
            self.emit(EventType.SUBSTITUTION, int(out), np.array([L / 2, -1.0]), note=f"on:{incoming.id}")
            self.slot_player[out] = incoming
            self.events[-1].recipient_id = incoming.id
            self._load_attrs()
            self.pos[out] = np.array([L / 2 + self.rng.uniform(-3, 3), -1.0])
            self.vel[out] = 0
            self.dist_run[out] = 0
            self.subs_done[team] += 1
            _ = old

    # ------------------------------------------------------------------- run
    def kickoff_positions(self) -> None:
        for team in (0, 1):
            idx = np.where(self.team_of == team)[0]
            kn = BASE_SLOTS.copy()
            kn[:, 0] = np.minimum(kn[:, 0] * 0.75, L / 2 - 2)
            for k, slot in enumerate(idx):
                self.pos[slot] = self.denorm(kn[k], team)
        self.vel[:] = 0

    def run(self) -> SimOutput:
        total = self.cfg.duration_s
        half = min(2700.0, total)
        end_t = total + (self.cfg.stoppage_s if total >= 5400 else 0)
        n = int(round(end_t / DT)) + 1
        xs = np.zeros((n, 23), np.float32)
        ys = np.zeros((n, 23), np.float32)
        zs = np.zeros(n, np.float32)
        ts = np.zeros(n, np.float32)
        per = np.zeros(n, np.int8)
        slot_ids = np.zeros((n, 22), np.int16)

        self.kickoff_positions()
        self.emit(EventType.PERIOD_START, None, None, team=0)
        self.dead = DeadBall(EventType.KICK_OFF, 0, np.array([L / 2, W / 2]), 0.5, 10)
        self.possession_team = 0
        for i in range(n):
            self.t = i * DT
            if self.period == 1 and self.t >= half and total > half:
                self.emit(EventType.PERIOD_END, None, None, team=0)
                self.period = 2
                self.kickoff_positions()
                self.emit(EventType.PERIOD_START, None, None, team=1)
                self.owner = None
                self.flight = None
                self.dead = DeadBall(EventType.KICK_OFF, 1, np.array([L / 2, W / 2]), self.t + 0.5, 21)
                self.possession_team = 1
            self.step()
            xs[i, :22], ys[i, :22] = self.pos[:, 0], self.pos[:, 1]
            xs[i, 22], ys[i, 22], zs[i] = self.ball[0], self.ball[1], self.ball_z
            ts[i], per[i] = self.t, self.period
            slot_ids[i] = [self._pidx[p.id] for p in self.slot_player]
        self.emit(EventType.PERIOD_END, None, None, team=0)
        return SimOutput(
            events=self.events,
            tracking={"t": ts, "period": per, "x": xs, "y": ys, "z": zs},
            slot_ids=slot_ids,
            player_index=self.player_index,
        )

    def step(self) -> None:
        if self.dead is None and self.flight is None and self.owner is None:
            # Safety net: give the ball to whoever is nearest.
            r = int(np.argmin(np.linalg.norm(self.pos - self.ball, axis=1)))
            self.set_possession(int(self.team_of[r]), r, self.ball)
            self.give_ball(r)
        if self.dead is None and self.owner is not None:
            self.maybe_sub()
        tgt = self.targets()
        self.move(tgt)
        if self.dead is not None:
            self.ball = self.dead.spot.copy()
            self.ball_z = 0.0
            self.maybe_restart()
        elif self.flight is not None:
            f = self.flight
            u = np.clip((self.t - f.t0) / max(f.t1 - f.t0, 1e-3), 0, 1)
            ease = 1 - (1 - u) ** 1.6 if f.kind != "shot" else u  # ground passes decelerate
            self.ball = f.start + (f.end - f.start) * ease
            self.ball_z = 4 * f.apex * u * (1 - u)
            if self.t >= f.t1:
                self.ball_z = 0.0
                self.resolve_flight()
        elif self.owner is not None:
            o = self.owner
            heading = self.vel[o] / (np.linalg.norm(self.vel[o]) + 1e-6)
            self.ball = self.pos[o] + heading * 0.6
            self.ball_z = 0.0
            self.contest()
            if self.owner is not None and self.dead is None and self.t >= self.hold_until:
                self.decide()
