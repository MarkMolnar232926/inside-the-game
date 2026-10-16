"""Versioned pydantic schemas: the contracts between pipeline stages."""

from .camera import CameraTrack, ScreenTrack, ScreenTracks, apply_h
from .events import Event, EventType, Outcome, PassInfo, ShotInfo, attack_x, third
from .insight import AgentTrace, Decision, Evidence, Explanation, InsightCandidate
from .match import PITCH_LENGTH, PITCH_WIDTH, Kit, MatchMeta, Player, PlayerAttributes, Position, Team
from .overlay import (
    SCHEMA_VERSION,
    Chip,
    Grounding,
    MediaClock,
    Minimap,
    NarrativeLowerThird,
    OverlayItem,
    OverlayTimeline,
    OverlayType,
    PassLine,
    PitchMarkers,
    PitchPoint,
    PlayerHighlight,
    RecapCard,
    ScoreBug,
    SpeedTag,
    StatCard,
    TextBlock,
)

__all__ = [n for n in dir() if not n.startswith("_")]
