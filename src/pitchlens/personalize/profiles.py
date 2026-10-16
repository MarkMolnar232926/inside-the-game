"""Viewer profiles: the same intelligence, different experiences.

A profile changes *what* the Director selects (theme/detector weights,
thresholds, pacing), *how* it is drawn (density, which overlay types) and
*how* it is told (register, language, perspective).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Register = Literal["casual", "analyst", "kids"]

METRIC_FAMILIES: dict[str, set[str]] = {
    "interceptions": {"interceptions_attacking_third", "interceptions_defensive_third"},
    "pressing": {"interceptions_attacking_third", "tackles_won_attacking_third", "recoveries_attacking_third", "ppda"},
    "physical": {"top_speed", "sprints"},
    "passing": {"pass_quality", "passes_completed", "pass_sequence"},
    "chances": {"shot_xg", "xg_window"},
    "rhythm": {"pass_sequence", "turnovers_window", "ppda", "xg_window"},
}


class ViewerProfile(BaseModel):
    id: str
    name: str
    description: str
    style: Register = Field("casual", description="Narrative register")
    language: str = "en"
    density: Literal["minimal", "standard", "dense"] = "standard"
    min_gap_s: float = 18.0  # minimum time between two insight cards
    card_s: float = 9.0  # how long a card stays on screen
    threshold: float = 0.45  # minimum Director score to show
    max_concurrent: int = 1
    theme_weights: dict[str, float] = Field(default_factory=dict)
    detector_weights: dict[str, float] = Field(default_factory=dict)
    visuals: list[str] = Field(
        default_factory=lambda: ["stat_card", "pitch_markers", "player_highlight", "narrative_lower_third",
                                 "pass_line", "speed_tag", "minimap"]
    )
    favourite_team: str | None = None
    focus_player: str | None = None
    single_metric: str | None = None  # key of METRIC_FAMILIES
    show_context_line: bool = False
    show_orientation_note: bool = False
    speed_threshold_kmh: float = 27.0

    def weight(self, metric: str, theme: str, team_id: str | None, player_id: str | None) -> float:
        if self.single_metric and metric not in METRIC_FAMILIES.get(self.single_metric, set()):
            return 0.0
        w = self.theme_weights.get(theme, 1.0) * self.detector_weights.get(metric, 1.0)
        if self.focus_player:
            w *= 3.0 if player_id == self.focus_player else 0.25
        if self.favourite_team and team_id:
            w *= 1.5 if team_id == self.favourite_team else 0.85  # biased, still fair
        return w


def builtin_profiles(featured_player: str = "KBY-07", home_team: str = "KBY") -> dict[str, ViewerProfile]:
    return {
        "casual": ViewerProfile(
            id="casual", name="Casual fan",
            description="Plain language, big visuals, one story at a time.",
            style="casual", density="standard", min_gap_s=22, card_s=9, threshold=0.5,
            theme_weights={"momentum": 1.3, "milestone": 1.1, "quality": 1.2, "physical": 1.1, "chaos": 1.0,
                           "control": 0.8, "pressure": 1.0},
            visuals=["stat_card", "pitch_markers", "player_highlight", "narrative_lower_third", "pass_line", "speed_tag"],
        ),
        "analyst": ViewerProfile(
            id="analyst", name="Data analyst",
            description="Dense metrics, xG/PPDA, evidence and grounding badges.",
            style="analyst", density="dense", min_gap_s=10, card_s=10, threshold=0.38,
            theme_weights={"pressure": 1.3, "control": 1.3, "chaos": 1.3, "momentum": 1.2, "physical": 0.8,
                           "milestone": 0.8},
            show_context_line=True, show_orientation_note=True,
        ),
        "kids": ViewerProfile(
            id="kids", name="Young fan",
            description="Simple, positive words; speed and skill moments; no betting-style numbers.",
            style="kids", density="standard", min_gap_s=25, card_s=8, threshold=0.45,
            theme_weights={"physical": 1.6, "quality": 1.4, "milestone": 1.2, "pressure": 1.1, "control": 0.6,
                           "chaos": 0.7, "momentum": 0.6},
            detector_weights={"xg_window": 0.2, "ppda": 0.2},
            visuals=["stat_card", "pitch_markers", "player_highlight", "narrative_lower_third", "speed_tag", "pass_line"],
        ),
        "player_focus": ViewerProfile(
            id="player_focus", name="Player cam",
            description="Follow one player all match with milestone alerts.",
            style="casual", density="standard", min_gap_s=12, card_s=9, threshold=0.4,
            focus_player=featured_player, speed_threshold_kmh=24.0,
        ),
        "single_metric": ViewerProfile(
            id="single_metric", name="Single metric: pressing",
            description="Only pressing and ball-winning insights.",
            style="analyst", density="standard", min_gap_s=12, card_s=9, threshold=0.3,
            single_metric="pressing",
        ),
        "favourite_club": ViewerProfile(
            id="favourite_club", name=f"{home_team} supporter",
            description="Biased towards your club, still fair to the opponent.",
            style="casual", density="standard", min_gap_s=18, card_s=9, threshold=0.45,
            favourite_team=home_team,
        ),
    }
