"""Insight candidates, decisions and agent traces (stages 2-3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .overlay import PitchPoint

Theme = Literal["control", "chaos", "tempo", "pressure", "milestone", "physical", "quality", "momentum"]


class Evidence(BaseModel):
    event_ids: list[str] = Field(default_factory=list)
    points: list[PitchPoint] = Field(default_factory=list)
    facts: dict[str, Any] = Field(
        default_factory=dict,
        description="Every number/name the text layer may use. The Verifier only accepts claims found here.",
    )


class InsightCandidate(BaseModel):
    id: str
    detector: str
    metric: str = Field(description="Machine key, e.g. 'interceptions_attacking_third'")
    t: float = Field(description="Match clock (s) when the insight became true")
    team_id: str | None = None
    player_id: str | None = None
    value: float
    value_text: str
    template_key: str = Field(description="Narrative template family")
    theme: Theme
    base_score: float = Field(ge=0, le=1, description="Detector's own newsworthiness estimate")
    visual: list[str] = Field(default_factory=list, description="Suggested overlay types")
    evidence: Evidence
    expires_after_s: float = 90.0


class Explanation(BaseModel):
    insight_id: str
    language: str
    headline: str
    why: str
    label: str = ""
    generator: Literal["template", "llm"] = "template"


class Decision(BaseModel):
    insight_id: str
    action: Literal["show", "defer", "drop"]
    reason: str
    score: float = 0.0
    duration_s: float = 0.0
    placement: Literal["right", "left"] = "right"


class AgentTrace(BaseModel):
    seq: int
    agent: str
    t_match: float
    action: str
    summary: str
    input_ref: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    ok: bool = True
