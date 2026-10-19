"""Fictional clubs and squads.

Every name below is invented for PitchLens. Names are composed from
hand-written syllable pools so they read naturally across languages without
referring to real people. Attributes are seeded and drive the simulator.
"""

from __future__ import annotations

import numpy as np

from pitchlens.schema import Kit, Player, PlayerAttributes, Position, Team

# (position, role, attribute bias)
FORMATION_433: list[tuple[Position, str, dict[str, float]]] = [
    (Position.GK, "sweeper keeper", {"passing": 0.0}),
    (Position.RB, "overlapping full-back", {"pace": 0.15, "stamina": 0.15}),
    (Position.CB, "ball-playing centre-back", {"passing": 0.1, "tackling": 0.2}),
    (Position.CB, "stopper", {"tackling": 0.25}),
    (Position.LB, "inverted full-back", {"passing": 0.1, "stamina": 0.1}),
    (Position.DM, "anchor", {"tackling": 0.2, "vision": 0.1}),
    (Position.CM, "box-to-box", {"stamina": 0.25, "pressing": 0.1}),
    (Position.AM, "pressing playmaker", {"vision": 0.25, "pressing": 0.3, "passing": 0.15}),
    (Position.RW, "inside forward", {"pace": 0.25, "dribbling": 0.25}),
    (Position.LW, "wide creator", {"pace": 0.2, "dribbling": 0.2, "vision": 0.1}),
    (Position.ST, "pressing forward", {"finishing": 0.35, "pressing": 0.2}),
]
BENCH: list[tuple[Position, str, dict[str, float]]] = [
    (Position.GK, "reserve keeper", {}),
    (Position.CB, "defensive cover", {"tackling": 0.2}),
    (Position.CM, "tempo setter", {"passing": 0.15}),
    (Position.RW, "impact winger", {"pace": 0.3}),
    (Position.ST, "target forward", {"finishing": 0.25}),
]
NUMBERS_433 = [1, 2, 4, 5, 3, 6, 8, 7, 11, 17, 9]
NUMBERS_BENCH = [13, 15, 14, 19, 20]

FIRST = [
    "Jory", "Tamsin", "Ilo", "Brenn", "Caius", "Oren", "Lazlo", "Femi", "Ruan", "Teodor", "Kasim", "Aurel",
    "Nilo", "Dace", "Mavi", "Sorin", "Elko", "Pell", "Joaquin", "Arvo", "Ciro", "Hale", "Ismo", "Rafe",
    "Tobin", "Vasco", "Wim", "Yannick", "Zeno", "Bastien", "Corin", "Davor", "Emeka", "Gael", "Hollis",
    "Ivo", "Kenji", "Lior", "Marek", "Noor",
]
LAST = [
    "Kestane", "Morrow-Hale", "Vantrell", "Okafor-Lind", "Brisquet", "Talmadge", "Quennell", "Ostrava",
    "Delacrow", "Fennimore", "Haskari", "Ilunga-Reyes", "Javorek", "Kelderman", "Lisowane", "Marquand",
    "Nyborg", "Oyelaran", "Pellacini", "Ravensby", "Sulejmani-Hart", "Thornquist", "Uwazurike", "Valcourt",
    "Wexley", "Yardane", "Zubrovin", "Abernathe", "Brightwell", "Castellane", "Drumond", "Esterhal",
    "Fairbourne", "Gravino", "Holtzmeyer", "Ingersol",
]

TEAM_SPECS = {
    "home": dict(
        id="KBY",
        name="Kestrel Bay FC",
        short="KBY",
        city="Kestrel Bay",
        kit=Kit(shirt="#c8233a", shorts="#14183d", socks="#c8233a", number="#ffffff", gk_shirt="#f2c230"),
        primary="#c8233a",
        secondary="#14183d",
        style="high press",
    ),
    "away": dict(
        id="IRV",
        name="Ironvale Athletic",
        short="IRV",
        city="Ironvale",
        kit=Kit(shirt="#f2f4f7", shorts="#17a398", socks="#f2f4f7", number="#14183d", gk_shirt="#7a3cf0"),
        primary="#17a398",
        secondary="#f2f4f7",
        style="possession",
    ),
}

# The demo storyline revolves around this player (index into FORMATION_433).
FEATURED_INDEX = 7


def build_team(side: str, rng: np.random.Generator, used_names: set[str]) -> Team:
    spec = TEAM_SPECS[side]
    players: list[Player] = []
    roster = [(p, r, b, n, True) for (p, r, b), n in zip(FORMATION_433, NUMBERS_433, strict=True)]
    roster += [(p, r, b, n, False) for (p, r, b), n in zip(BENCH, NUMBERS_BENCH, strict=True)]
    for i, (pos, role, bias, number, starter) in enumerate(roster):
        while True:
            first, last = rng.choice(FIRST), rng.choice(LAST)
            if last not in used_names:
                used_names.add(last)
                break
        if side == "home" and i == FEATURED_INDEX:
            first, last = "Jory", "Kestane"  # the demo protagonist, KBY #7
        base = {k: float(np.clip(rng.normal(0.55, 0.1), 0.2, 0.9)) for k in PlayerAttributes.model_fields}
        for k, v in bias.items():
            base[k] = float(np.clip(base[k] + v, 0, 0.98))
        if pos == Position.GK:
            base["pace"] = 0.3
        players.append(
            Player(
                id=f"{spec['id']}-{number:02d}",
                team_id=spec["id"],
                name=f"{first} {last}",
                short_name=last.upper(),
                number=number,
                position=pos,
                role=role,
                starter=starter,
                attributes=PlayerAttributes(**base),
                skin_tone=int(rng.integers(0, 6)),
            )
        )
    return Team(players=players, formation="4-3-3", **spec)


def build_teams(seed: int) -> tuple[Team, Team]:
    rng = np.random.default_rng(seed)
    used: set[str] = {"Kestane"}
    return build_team("home", rng, used), build_team("away", rng, used)
