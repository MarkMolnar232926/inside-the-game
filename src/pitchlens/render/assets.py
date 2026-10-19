"""Procedurally generated, rights-clear graphics assets.

* Club crests: heraldic shield with the club initials and a simple emblem.
* Player portraits: a stylised head-and-shoulders illustration in the club
  kit with the shirt number. Deliberately non-photoreal, so no real
  likeness is ever implied.

Assets are deterministic (seeded by id) and cached under
``assets/generated/``.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from pitchlens.schema import MatchMeta, Player, Team

ROOT = Path(__file__).resolve().parents[3]
FONTS = ROOT / "assets" / "fonts"
GENERATED = ROOT / "assets" / "generated"
TOKENS_PATH = Path(__file__).with_name("tokens.json")

SKIN = ["#f1c9a5", "#e0ac83", "#c68a5e", "#a86b43", "#7d4a2b", "#5a3420"]
HAIR = ["#1c1714", "#3a2618", "#6b4423", "#b0793f", "#d9b56b", "#2b2b2b"]


@lru_cache(maxsize=1)
def tokens() -> dict:
    return json.loads(TOKENS_PATH.read_text())


@lru_cache(maxsize=64)
def font(kind: str, px: int) -> ImageFont.FreeTypeFont:
    name = tokens()["fonts"].get(kind, kind)
    return ImageFont.truetype(str(FONTS / name), px)


def rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _seed(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def vertical_gradient(w: int, h: int, top: str, bottom: str) -> Image.Image:
    a, b = rgb(top), rgb(bottom)
    grad = Image.new("RGB", (1, h))
    for y in range(h):
        u = y / max(h - 1, 1)
        grad.putpixel((0, y), tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3)))
    return grad.resize((w, h)).convert("RGBA")


# --------------------------------------------------------------------- crest
def crest(team: Team, size: int = 256) -> Image.Image:
    path = GENERATED / f"crest_{team.id}.png"
    if path.exists():
        return Image.open(path).convert("RGBA")
    S = size * 4
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    shield = [(S * 0.12, S * 0.08), (S * 0.88, S * 0.08), (S * 0.88, S * 0.5), (S * 0.5, S * 0.95), (S * 0.12, S * 0.5)]
    prim, sec = rgb(team.primary), rgb(team.secondary)
    d.polygon(shield, fill=sec)
    inner = [(S * 0.18, S * 0.13), (S * 0.82, S * 0.13), (S * 0.82, S * 0.49), (S * 0.5, S * 0.87), (S * 0.18, S * 0.49)]
    d.polygon(inner, fill=prim)
    # Emblem: a kestrel-like chevron for KBY, a mountain for IRV, else a star.
    if team.id == "KBY":
        d.polygon([(S * 0.26, S * 0.42), (S * 0.5, S * 0.28), (S * 0.74, S * 0.42), (S * 0.5, S * 0.36)], fill=sec)
        d.polygon([(S * 0.44, S * 0.36), (S * 0.5, S * 0.22), (S * 0.56, S * 0.36)], fill=sec)
    else:
        d.polygon([(S * 0.24, S * 0.46), (S * 0.42, S * 0.24), (S * 0.52, S * 0.36), (S * 0.6, S * 0.28),
                   (S * 0.76, S * 0.46)], fill=sec)
    f = font("display", int(S * 0.2))
    txt = team.short
    tw = d.textlength(txt, font=f)
    d.text(((S - tw) / 2, S * 0.5), txt, font=f, fill=sec)
    out = im.resize((size, size), Image.LANCZOS)
    GENERATED.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return out


# ------------------------------------------------------------------ portrait
def portrait(player: Player, team: Team, w: int = 400, h: int = 470) -> Image.Image:
    path = GENERATED / f"portrait_{player.id}.png"
    if path.exists():
        return Image.open(path).convert("RGBA")
    S = 3
    W, H = w * S, h * S
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    rnd = _seed(player.id)
    skin = rgb(SKIN[player.skin_tone])
    skin_dark = tuple(int(c * 0.82) for c in skin)
    hair = rgb(HAIR[rnd % len(HAIR)] if player.skin_tone < 3 else HAIR[rnd % 2])
    shirt = rgb(team.kit.gk_shirt if player.position.value == "GK" else team.kit.shirt)
    trim = rgb(team.kit.shorts)
    cx = W / 2
    # Shoulders / torso
    d.rounded_rectangle([W * 0.06, H * 0.66, W * 0.94, H * 1.2], radius=int(W * 0.22), fill=shirt)
    # Neck
    d.rectangle([cx - W * 0.085, H * 0.5, cx + W * 0.085, H * 0.7], fill=skin_dark)
    # Collar (V-neck)
    d.polygon([(cx - W * 0.16, H * 0.665), (cx, H * 0.79), (cx + W * 0.16, H * 0.665), (cx + W * 0.1, H * 0.665),
               (cx, H * 0.74), (cx - W * 0.1, H * 0.665)], fill=trim)
    d.polygon([(cx - W * 0.1, H * 0.665), (cx, H * 0.74), (cx + W * 0.1, H * 0.665)], fill=skin_dark)
    # Shirt number on chest
    f = font("display", int(H * 0.15))
    num = str(player.number)
    tw = d.textlength(num, font=f)
    d.text((W * 0.66 - tw / 2, H * 0.8), num, font=f, fill=rgb(team.kit.number))
    # Crest patch
    cr = crest(team, int(W * 0.12)).resize((int(W * 0.12), int(W * 0.12)))
    im.alpha_composite(cr, (int(W * 0.24), int(H * 0.82)))
    # Ears
    for sx in (-1, 1):
        d.ellipse([cx + sx * W * 0.2 - W * 0.04, H * 0.33, cx + sx * W * 0.2 + W * 0.04, H * 0.45], fill=skin_dark)
    # Head
    d.ellipse([cx - W * 0.2, H * 0.14, cx + W * 0.2, H * 0.58], fill=skin)
    # Hair styles
    style = (rnd >> 4) % 4
    if style == 0:  # short crop
        d.chord([cx - W * 0.205, H * 0.11, cx + W * 0.205, H * 0.42], 180, 360, fill=hair)
    elif style == 1:  # quiff
        d.chord([cx - W * 0.205, H * 0.12, cx + W * 0.205, H * 0.4], 180, 360, fill=hair)
        d.ellipse([cx - W * 0.12, H * 0.07, cx + W * 0.16, H * 0.2], fill=hair)
    elif style == 2:  # buzz
        d.chord([cx - W * 0.2, H * 0.135, cx + W * 0.2, H * 0.36], 180, 360, fill=hair)
    else:  # curls
        for k in range(9):
            ox = (k - 4) * W * 0.045
            d.ellipse([cx + ox - W * 0.06, H * 0.1 + abs(k - 4) * H * 0.012, cx + ox + W * 0.06,
                       H * 0.23 + abs(k - 4) * H * 0.012], fill=hair)
    # Minimal, friendly features
    eye_y = H * 0.37
    for sx in (-1, 1):
        d.ellipse([cx + sx * W * 0.075 - W * 0.018, eye_y - H * 0.012, cx + sx * W * 0.075 + W * 0.018,
                   eye_y + H * 0.012], fill=(30, 26, 30))
        d.line([cx + sx * W * 0.045, eye_y - H * 0.035, cx + sx * W * 0.115, eye_y - H * 0.04], fill=hair,
               width=int(W * 0.014))
    d.arc([cx - W * 0.06, H * 0.43, cx + W * 0.06, H * 0.49], 20, 160, fill=skin_dark, width=int(W * 0.012))
    # Soft rim light
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).ellipse([cx + W * 0.05, H * 0.15, cx + W * 0.26, H * 0.6], fill=(255, 255, 255, 34))
    im.alpha_composite(shade.filter(ImageFilter.GaussianBlur(W * 0.03)))
    out = im.resize((w, h), Image.LANCZOS)
    GENERATED.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return out


def build_all(meta: MatchMeta) -> None:
    for team in (meta.home, meta.away):
        crest(team)
        for p in team.players:
            portrait(p, team)
