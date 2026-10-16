"""Configurable match storylines.

A narrative modulates each team's behaviour over the match clock: passing
accuracy, pressing intensity, tempo, finishing and risk appetite. Optional
``script`` targets nudge shot conversion so the intended scoreline shape
(e.g. a late comeback) actually happens for most seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TeamMods:
    pass_acc: float = 1.0  # multiplier on pass completion probability
    press: float = 0.5  # 0..1 pressing intensity
    tempo: float = 1.0  # >1 = quicker ball circulation (shorter hold times)
    finishing: float = 1.0  # multiplier on conversion
    shot_appetite: float = 1.0
    risk: float = 1.0  # preference for progressive passes
    line_height: float = 0.0  # metres the defensive block is pushed up


@dataclass(frozen=True)
class ScriptedGoal:
    team: int  # 0 home, 1 away
    after_s: float
    before_s: float


@dataclass(frozen=True)
class Narrative:
    name: str
    description: str
    phases: list[tuple[float, TeamMods, TeamMods]] = field(default_factory=list)
    script: list[ScriptedGoal] = field(default_factory=list)

    def mods(self, team: int, t: float) -> TeamMods:
        current = self.phases[0]
        for phase in self.phases:
            if t >= phase[0]:
                current = phase
        return current[1 + team]


NARRATIVES: dict[str, Narrative] = {
    "balanced": Narrative(
        "balanced",
        "Even contest; both sides trade spells of control.",
        [(0, TeamMods(press=0.55), TeamMods(press=0.5))],
    ),
    "dominant_home": Narrative(
        "dominant_home",
        "Home side controls the ball and suffocates the visitors with a high press.",
        [
            (
                0,
                TeamMods(pass_acc=1.03, press=0.8, tempo=1.1, risk=1.1, line_height=8, finishing=1.1),
                TeamMods(pass_acc=0.95, press=0.35, tempo=0.95, risk=0.9, line_height=-6),
            )
        ],
        [ScriptedGoal(0, 1200, 2400), ScriptedGoal(0, 3600, 5000)],
    ),
    "late_comeback": Narrative(
        "late_comeback",
        "Visitors strike first and sit deep; the home side's pressing surge after the hour flips the match.",
        [
            (0, TeamMods(press=0.5, pass_acc=0.98), TeamMods(press=0.6, pass_acc=1.02, risk=1.05)),
            (
                3600,
                TeamMods(press=0.9, tempo=1.25, risk=1.3, line_height=10, finishing=1.25, shot_appetite=1.4),
                TeamMods(press=0.3, pass_acc=0.95, tempo=0.9, line_height=-8),
            ),
        ],
        [ScriptedGoal(1, 1300, 2600), ScriptedGoal(0, 4300, 4900), ScriptedGoal(0, 4950, 5450)],
    ),
    "chaotic_end_to_end": Narrative(
        "chaotic_end_to_end",
        "Open, frantic game: quick turnovers, stretched lines and chances at both ends.",
        [
            (
                0,
                TeamMods(pass_acc=0.93, press=0.75, tempo=1.35, risk=1.4, shot_appetite=1.5, line_height=4),
                TeamMods(pass_acc=0.93, press=0.75, tempo=1.35, risk=1.4, shot_appetite=1.5, line_height=4),
            )
        ],
    ),
}
