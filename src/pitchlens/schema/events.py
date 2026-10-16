"""Match event schema (stage 1: ingest).

One flat, typed ``Event`` model keeps the stream easy to serialise (JSON
lines, Parquet, Event Hubs) while the optional ``pass_info`` / ``shot_info``
sub-models carry type-specific detail.

Coordinate convention: metres on a 105 x 68 pitch, fixed to the pitch (not
to the attacking direction). Use :func:`attack_x` to normalise so that the
acting team always attacks towards x = 105.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from .match import PITCH_LENGTH, PITCH_WIDTH


class EventType(StrEnum):
    KICK_OFF = "kick_off"
    PASS = "pass"
    CARRY = "carry"
    SHOT = "shot"
    GOAL = "goal"
    SAVE = "save"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    DUEL = "duel"
    PRESSURE = "pressure"
    BALL_RECOVERY = "ball_recovery"
    CLEARANCE = "clearance"
    FOUL = "foul"
    FREE_KICK = "free_kick"
    CORNER = "corner"
    THROW_IN = "throw_in"
    GOAL_KICK = "goal_kick"
    OUT_OF_PLAY = "out_of_play"
    POSSESSION_CHANGE = "possession_change"
    SUBSTITUTION = "substitution"
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"


class Outcome(StrEnum):
    SUCCESS = "success"
    FAIL = "fail"
    NA = "na"


Source = Literal["synthetic", "vision_estimated"]


class PassInfo(BaseModel):
    length_m: float
    progressive_m: float = Field(description="Metres gained towards the opponent goal")
    difficulty: float = Field(ge=0, le=1, description="0 = trivial, 1 = extremely hard")
    height: Literal["ground", "low", "high"] = "ground"
    under_pressure: bool = False
    ball_speed_kmh: float = 0.0
    key_pass: bool = False
    expected_completion: float = Field(0.8, ge=0, le=1)


class ShotInfo(BaseModel):
    xg: float = Field(ge=0, le=1)
    speed_kmh: float
    body_part: Literal["right_foot", "left_foot", "head"] = "right_foot"
    on_target: bool = False
    distance_m: float = 0.0


class Event(BaseModel):
    """A single on-ball (or match-state) event."""

    id: str
    match_id: str
    seq: int = Field(description="Monotonic sequence number within the match")
    t: float = Field(ge=0, description="Match clock in seconds since kick-off (continuous across halves)")
    period: int = Field(1, ge=1, le=2)
    type: EventType
    team_id: str | None = None
    player_id: str | None = None
    x: float | None = Field(None, ge=-2, le=PITCH_LENGTH + 2)
    y: float | None = Field(None, ge=-2, le=PITCH_WIDTH + 2)
    end_x: float | None = None
    end_y: float | None = None
    outcome: Outcome = Outcome.NA
    recipient_id: str | None = None
    opponent_id: str | None = Field(None, description="Opposing player involved (tackled/intercepted)")
    possession_id: int = 0
    source: Source = "synthetic"
    confidence: float = Field(1.0, ge=0, le=1, description="1.0 for synthetic ground truth")
    pass_info: PassInfo | None = None
    shot_info: ShotInfo | None = None
    score: tuple[int, int] = (0, 0)
    note: str | None = None

    @property
    def clock(self) -> str:
        m, s = divmod(int(self.t), 60)
        return f"{m:02d}:{s:02d}"

    @property
    def minute(self) -> int:
        return int(self.t // 60) + 1


def attack_x(x: float, attacks_right: bool) -> float:
    """Normalise an x coordinate so the acting team attacks towards x = 105."""
    return x if attacks_right else PITCH_LENGTH - x


def third(x_norm: float) -> Literal["defensive", "middle", "attacking"]:
    """Zone of an attack-normalised x coordinate."""
    if x_norm < PITCH_LENGTH / 3:
        return "defensive"
    if x_norm < 2 * PITCH_LENGTH / 3:
        return "middle"
    return "attacking"
