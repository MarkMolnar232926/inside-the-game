"""Overlay instruction schema (stage 4: render contract).

This is the main hand-off between the engine and any renderer. Each item is
timed on the media clock (seconds from the start of the video), carries
pitch-space geometry in metres, and has text per language. A renderer needs
three things to draw a frame:

1. this timeline,
2. a camera track (per-frame pitch->image homography), and
3. optionally player screen tracks (for highlights that follow a player).

The bundled Python compositor (``pitchlens.render``) and the browser canvas
renderer (``web/overlay.js``) both read the same JSON.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class OverlayType(StrEnum):
    SCORE_BUG = "score_bug"
    STAT_CARD = "stat_card"
    PLAYER_HIGHLIGHT = "player_highlight"
    PITCH_MARKERS = "pitch_markers"
    SPEED_TAG = "speed_tag"
    PASS_LINE = "pass_line"
    NARRATIVE_LOWER_THIRD = "narrative_lower_third"
    MINIMAP = "minimap"
    RECAP_CARD = "recap_card"


class TextBlock(BaseModel):
    headline: str = ""
    body: str = ""
    label: str = ""


class Grounding(BaseModel):
    verified: bool = False
    checks: list[str] = Field(default_factory=list, description="Human-readable checks that passed")
    failures: list[str] = Field(default_factory=list)
    generator: Literal["template", "llm"] = "template"


class PitchPoint(BaseModel):
    x: float = Field(description="Metres along the pitch length (0..105)")
    y: float = Field(description="Metres across the pitch (0..68)")
    label: str = ""
    event_id: str | None = None
    appear_at: float = Field(0.0, description="Seconds after item t_start when this point appears")


class Chip(BaseModel):
    label: str
    value: str
    unit: str = ""
    appear_at: float = 0.0


class _Base(BaseModel):
    id: str
    t_start: float = Field(ge=0, description="Media clock, seconds")
    t_end: float = Field(ge=0)
    priority: int = Field(5, ge=0, le=10)
    insight_id: str | None = None
    evidence_event_ids: list[str] = Field(default_factory=list)
    text: dict[str, TextBlock] = Field(default_factory=dict, description="Keyed by BCP-47 language code")
    style: dict[str, str] = Field(default_factory=dict, description="Design-token overrides")
    grounding: Grounding | None = None


class ScoreBug(_Base):
    type: Literal[OverlayType.SCORE_BUG] = OverlayType.SCORE_BUG
    home_short: str
    away_short: str
    home_color: str
    away_color: str
    score_track: list[tuple[float, int, int]] = Field(description="(media_t, home, away) change points")
    clock_offset: float = Field(description="match_clock = media_t + clock_offset")


class StatCard(_Base):
    type: Literal[OverlayType.STAT_CARD] = OverlayType.STAT_CARD
    value: str = Field(description="The big number, e.g. '4' or '87%'")
    player_id: str | None = None
    team_id: str | None = None
    player_number: int | None = None
    player_name: str | None = None
    team_name: str | None = None
    portrait: str | None = Field(None, description="Asset key for the portrait illustration")
    crest: str | None = None
    placement: Literal["right", "left"] = "right"
    density: Literal["minimal", "standard", "dense"] = "standard"
    extra_metrics: list[Chip] = Field(default_factory=list, description="Shown in dense (analyst) mode")


class PlayerHighlight(_Base):
    type: Literal[OverlayType.PLAYER_HIGHLIGHT] = OverlayType.PLAYER_HIGHLIGHT
    player_id: str
    player_number: int | None = None
    show_name: bool = True
    display_name: str = ""
    color: str = "accent"
    chips: list[Chip] = Field(default_factory=list)


class PitchMarkers(_Base):
    type: Literal[OverlayType.PITCH_MARKERS] = OverlayType.PITCH_MARKERS
    points: list[PitchPoint]
    shape: Literal["star", "ring", "cross", "dot"] = "star"
    numbered: bool = True
    connect: bool = False
    color: str = "marker"


class SpeedTag(_Base):
    type: Literal[OverlayType.SPEED_TAG] = OverlayType.SPEED_TAG
    subject: Literal["player", "ball"] = "player"
    player_id: str | None = None
    speed_kmh: float
    distance_m: float | None = None
    estimated: bool = False


class PassLine(_Base):
    type: Literal[OverlayType.PASS_LINE] = OverlayType.PASS_LINE
    start: PitchPoint
    end: PitchPoint
    distance_m: float
    difficulty: float = Field(ge=0, le=1)
    accuracy: Literal["perfect", "accurate", "loose", "failed"]
    ball_speed_kmh: float | None = None
    player_id: str | None = None
    recipient_id: str | None = None


class NarrativeLowerThird(_Base):
    type: Literal[OverlayType.NARRATIVE_LOWER_THIRD] = OverlayType.NARRATIVE_LOWER_THIRD
    tone: Literal["neutral", "control", "chaos", "tempo", "pressure", "milestone", "physical", "quality", "momentum"] = "neutral"
    audio: dict[str, str] = Field(default_factory=dict, description="Language -> TTS asset key")


class Minimap(_Base):
    type: Literal[OverlayType.MINIMAP] = OverlayType.MINIMAP
    points: list[PitchPoint] = Field(default_factory=list)
    focus_player_id: str | None = None
    show_team_shape: bool = True


class RecapCard(_Base):
    type: Literal[OverlayType.RECAP_CARD] = OverlayType.RECAP_CARD
    moments: list[TextBlock] = Field(default_factory=list)


OverlayItem = Annotated[
    ScoreBug
    | StatCard
    | PlayerHighlight
    | PitchMarkers
    | SpeedTag
    | PassLine
    | NarrativeLowerThird
    | Minimap
    | RecapCard,
    Field(discriminator="type"),
]


class MediaClock(BaseModel):
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    duration_s: float
    match_t_at_media_zero: float = Field(description="Match clock (s) shown at media t = 0")


class OverlayTimeline(BaseModel):
    schema_version: str = SCHEMA_VERSION
    match_id: str
    profile_id: str
    languages: list[str]
    default_language: str = "en"
    media: MediaClock
    items: list[OverlayItem]

    def active(self, media_t: float) -> list[OverlayItem]:
        return [it for it in self.items if it.t_start <= media_t < it.t_end]
