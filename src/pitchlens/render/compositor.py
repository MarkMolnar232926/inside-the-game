"""Broadcast overlay compositor (stage 4: render).

Turns ``OverlayTimeline`` + ``CameraTrack`` (+ optional ``ScreenTracks``)
into pixels. All drawing happens on a *premultiplied* RGBA layer, so the same
code produces (a) burned-in video and (b) a clean alpha overlay stream (PNG
sequence / WebM-alpha / ProRes 4444) for a broadcaster's vision mixer.

Screen-anchored graphics (score bug, insight panel, lower third, minimap) are
rendered once per item+language with Pillow (high-quality text) and cached;
pitch-anchored graphics (markers, rings, pass lines) are projected through the
per-frame homography every frame, so they stay glued to the grass while the
camera moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pitchlens.narrative import templates as T
from pitchlens.schema import (
    CameraTrack,
    MatchMeta,
    NarrativeLowerThird,
    OverlayTimeline,
    PassLine,
    PitchMarkers,
    PlayerHighlight,
    RecapCard,
    ScoreBug,
    ScreenTracks,
    SpeedTag,
    StatCard,
    apply_h,
)
from pitchlens.schema import Minimap as MinimapItem
from pitchlens.vision import pitch_model as pm

from . import assets
from .assets import font, rgb, tokens


# ------------------------------------------------------------------ easing
def ease_out_cubic(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 1 - (1 - u) ** 3


def ease_in_cubic(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u**3


def ease_out_back(u: float, s: float = 1.9) -> float:
    u = min(max(u, 0.0), 1.0) - 1
    return 1 + u * u * ((s + 1) * u + s)


def envelope(t: float, t0: float, t1: float, tin: float, tout: float) -> tuple[float, float]:
    """(in_progress, out_progress) each eased in [0, 1]."""
    return ease_out_cubic((t - t0) / max(tin, 1e-6)), ease_in_cubic((t - (t1 - tout)) / max(tout, 1e-6))


# ----------------------------------------------------------- layer helpers
def pil_to_premul(im: Image.Image) -> np.ndarray:
    """PIL RGBA (straight) -> numpy BGRA premultiplied uint8."""
    a = np.asarray(im.convert("RGBA"), dtype=np.uint16)
    out = np.empty_like(a)
    alpha = a[..., 3:4]
    out[..., :3] = (a[..., [2, 1, 0]] * alpha + 127) // 255
    out[..., 3:4] = alpha
    return out.astype(np.uint8)


def blit(dst: np.ndarray, src: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Premultiplied 'over' of ``src`` onto ``dst`` at (x, y), clipped to bounds."""
    if opacity <= 0.003:
        return
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y : y1 - y, x0 - x : x1 - x]
    if opacity < 0.999:
        s = cv2.multiply(s, np.full(s.shape, 255, np.uint8), scale=opacity / 255)
    d = dst[y0:y1, x0:x1]
    inv = cv2.merge([255 - s[..., 3]] * 4)
    dst[y0:y1, x0:x1] = cv2.add(s, cv2.multiply(d, inv, scale=1 / 255))


def draw_poly_layer(layer: np.ndarray, polys: list[np.ndarray], color: str, alpha: float, fill: bool = True,
                    thickness: int = 2) -> None:
    """Draw anti-aliased polygons with a given opacity onto a premultiplied layer (ROI-local)."""
    if alpha <= 0.003 or not polys:
        return
    allp = np.vstack(polys)
    if not np.isfinite(allp).all():
        return
    H, W = layer.shape[:2]
    pad = thickness + 3
    x0, y0 = np.floor(allp.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(allp.max(axis=0)).astype(int) + pad
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return
    tmp = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    b, g, r = rgb(color)[::-1]
    col = (b, g, r, 255)
    pts = [np.round((p - [x0, y0]) * 16).astype(np.int32) for p in polys]
    if fill:
        cv2.fillPoly(tmp, pts, col, cv2.LINE_AA, shift=4)
    else:
        cv2.polylines(tmp, pts, True, col, thickness, cv2.LINE_AA, shift=4)
    blit(layer, tmp, int(x0), int(y0), alpha)


def composite(frame_bgr: np.ndarray, layer: np.ndarray) -> np.ndarray:
    a = layer[..., 3]
    if not a.any():
        return frame_bgr.copy()
    inv = cv2.merge([255 - a] * 3)
    return cv2.add(cv2.multiply(frame_bgr, inv, scale=1 / 255), layer[..., :3])


def unpremultiply(layer: np.ndarray) -> np.ndarray:
    """Premultiplied BGRA -> straight RGBA (for PNG / alpha video export)."""
    a = layer[..., 3:4].astype(np.float32)
    rgbf = np.where(a > 0, layer[..., :3].astype(np.float32) * 255.0 / np.maximum(a, 1), 0)
    out = np.concatenate([rgbf[..., ::-1], a], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def star_points(cx: float, cy: float, r: float, n: int = 5, inner: float = 0.45, rot: float = -math.pi / 2) -> np.ndarray:
    pts = []
    for k in range(2 * n):
        rr = r if k % 2 == 0 else r * inner
        a = rot + k * math.pi / n
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return np.array(pts)


def circle_points(cx: float, cy: float, r: float, n: int = 48, a0: float = 0, a1: float = 2 * math.pi) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


# ------------------------------------------------------------- renderer
@dataclass
class RenderContext:
    t: float  # media seconds
    frame: int
    H: np.ndarray  # pitch -> image
    scale: float  # output height / 1080
    lang: str
    tracks: ScreenTracks | None
    ball_px: tuple[float, float] | None = None


@dataclass
class OverlayRenderer:
    meta: MatchMeta
    width: int = 1920
    height: int = 1080
    sprite_cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tok = tokens()
        self.c = self.tok["colors"]
        self.lay = self.tok["layout"]
        self.mo = self.tok["motion"]
        self.s = self.height / 1080.0

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    # ============================================================ frame
    def render_layer(self, timeline: OverlayTimeline, camera: CameraTrack, frame_idx: int,
                     tracks: ScreenTracks | None = None, lang: str | None = None) -> np.ndarray:
        t = frame_idx / timeline.media.fps
        ctx = RenderContext(t, frame_idx, camera.H(frame_idx), self.s, lang or timeline.default_language, tracks)
        layer = np.zeros((self.height, self.width, 4), np.uint8)
        active = timeline.active(t)
        # Ground-anchored first (under the people-tracking graphics), then screen graphics.
        order = {
            "pitch_markers": 0, "pass_line": 1, "player_highlight": 2, "speed_tag": 3, "score_bug": 4,
            "minimap": 5, "stat_card": 6, "narrative_lower_third": 7, "recap_card": 8,
        }
        for it in sorted(active, key=lambda i: order.get(i.type.value, 9)):
            fn = getattr(self, f"draw_{it.type.value}", None)
            if fn is not None:
                fn(layer, it, ctx, timeline)
        return layer

    def render_frame(self, frame_bgr: np.ndarray, timeline: OverlayTimeline, camera: CameraTrack, frame_idx: int,
                     tracks: ScreenTracks | None = None, lang: str | None = None) -> np.ndarray:
        return composite(frame_bgr, self.render_layer(timeline, camera, frame_idx, tracks, lang))

    def _text(self, it, lang: str):
        return it.text.get(lang) or it.text.get("en") or next(iter(it.text.values()), None)

    # ======================================================= score bug
    def draw_score_bug(self, layer: np.ndarray, it: ScoreBug, ctx: RenderContext, tl: OverlayTimeline) -> None:
        home, away = 0, 0
        for (mt, h, a) in it.score_track:
            if ctx.t >= mt:
                home, away = h, a
        clock = ctx.t + it.clock_offset
        mm, ss = divmod(int(clock), 60)
        key = ("bug", home, away, mm, ss, it.home_short, it.away_short)
        sprite = self.sprite_cache.get(key)
        if sprite is None:
            sprite = self._bug_sprite(it, home, away, f"{mm:02d}:{ss:02d}")
            if len(self.sprite_cache) > 400:
                self.sprite_cache.clear()
            self.sprite_cache[key] = sprite
        x = int(self.width * self.lay["safe_x"])
        y = int(self.height * self.lay["safe_y"])
        fin, _ = envelope(ctx.t, it.t_start, it.t_end, 0.4, 0.3)
        blit(layer, sprite, x, y - int((1 - fin) * self.px(20)), fin)

    def _bug_sprite(self, it: ScoreBug, home: int, away: int, clock: str) -> np.ndarray:
        S = 2
        h = self.px(self.lay["bug_h"]) * S
        f = font("bold", self.px(self.lay["bug_font_px"]) * S)
        fc = font("semibold", self.px(self.lay["bug_font_px"] - 2) * S)
        pad = int(h * 0.32)
        tw = lambda s, ff=f: int(ff.getlength(s))  # noqa: E731
        w_team = max(tw(it.home_short), tw(it.away_short)) + 2 * pad
        w_score = int(h * 0.95)
        w_clock = tw("00:00", fc) + 2 * pad
        stripe = int(h * 0.12)
        W = stripe + w_team + 2 * w_score + w_team + w_clock
        im = Image.new("RGBA", (W, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, W - 1, h - 1], radius=int(h * 0.14), fill=rgb(self.c["bug_bg"]) + (240,))
        d.rectangle([0, 0, stripe, h], fill=rgb(it.home_color))
        x = stripe
        ty = (h - f.size) / 2 - h * 0.05

        def center(text: str, x0: int, w: int, ff, fill=(255, 255, 255)):
            d.text((x0 + (w - ff.getlength(text)) / 2, ty), text, font=ff, fill=fill)

        center(it.home_short, x, w_team, f)
        x += w_team
        d.rectangle([x, 0, x + w_score, h], fill=rgb(it.home_color))
        center(str(home), x, w_score, f)
        x += w_score
        d.rectangle([x, 0, x + w_score, h], fill=rgb(it.away_color))
        center(str(away), x, w_score, f)
        x += w_score
        center(it.away_short, x, w_team, f)
        x += w_team
        d.rounded_rectangle([x, 0, W - 1, h - 1], radius=int(h * 0.14), fill=rgb(self.c["bug_clock_bg"]) + (250,))
        d.rectangle([x, 0, x + int(h * 0.2), h], fill=rgb(self.c["bug_clock_bg"]) + (250,))
        center(clock, x, w_clock, fc)
        im = im.resize((W // S, h // S), Image.LANCZOS)
        return pil_to_premul(im)

    # ======================================================= stat card
    def _panel_sprites(self, it: StatCard, lang: str) -> list[np.ndarray]:
        key = ("card", it.id, lang)
        if key in self.sprite_cache:
            return self.sprite_cache[key]
        S = 2
        L = self.lay
        Wp = self.px(L["panel_w"]) * S
        tb = self._text(it, lang)
        headline = (tb.headline if tb else "").upper()
        # --- header: big number + headline
        hh = self.px(L["panel_header_h"]) * S
        head = vertical_gradient_h(Wp, hh, self.c["panel_header_from"], self.c["panel_header_to"])
        d = ImageDraw.Draw(head)
        fbig = font("display", self.px(L["big_number_px"]) * S)
        value = it.value
        while fbig.getlength(value) > Wp * 0.42 and fbig.size > 40:
            fbig = font("display", int(fbig.size * 0.9))
        vw = fbig.getlength(value)
        pad = int(Wp * 0.06)
        bbox = fbig.getbbox(value)
        d.text((pad, (hh - (bbox[3] - bbox[1])) / 2 - bbox[1]), value, font=fbig, fill=(255, 255, 255))
        fh = font("bold", self.px(L["headline_px"]) * S)
        max_w = Wp - pad * 3 - vw
        lines = wrap(headline, fh, max_w, 3)
        while len(lines) > 3 or any(fh.getlength(li) > max_w for li in lines):
            fh = font("bold", int(fh.size * 0.9))
            lines = wrap(headline, fh, max_w, 3)
        lh = fh.size * 1.08
        y0 = (hh - lh * len(lines)) / 2
        for i, line in enumerate(lines):
            d.text((pad * 2 + vw, y0 + i * lh - fh.size * 0.06), line, font=fh, fill=(255, 255, 255))
        sprites = [pil_to_premul(head.resize((Wp // S, hh // S), Image.LANCZOS))]
        # --- portrait
        if it.player_id and it.density != "minimal":
            ph = self.px(L["panel_portrait_h"]) * S
            if it.density == "dense":
                ph = int(ph * 0.72)
            box = vertical_gradient_h(Wp, ph, self.c["portrait_from"], self.c["portrait_to"])
            player = self.meta.player(it.player_id)
            team = self.meta.team(player.team_id)
            por = assets.portrait(player, team)
            scale = min(Wp * 0.95 / por.width, ph * 0.96 / por.height)
            por = por.resize((int(por.width * scale), int(por.height * scale)), Image.LANCZOS)
            box.alpha_composite(por, ((Wp - por.width) // 2, ph - por.height))
            sprites.append(pil_to_premul(box.resize((Wp // S, ph // S), Image.LANCZOS)))
        # --- name plate
        if it.player_name or it.team_name:
            plh = self.px(L["panel_plate_h"]) * S
            plate = Image.new("RGBA", (Wp, plh), rgb(self.c["plate"]) + (255,))
            d = ImageDraw.Draw(plate)
            d.rectangle([0, 0, Wp, int(plh * 0.04)], fill=rgb(self.c["plate_edge"]))
            crest_sz = int(plh * 0.66)
            right_pad = pad
            if it.team_id:
                team = self.meta.team(it.team_id)
                cbox = int(plh * 0.78)
                cx0 = Wp - right_pad - cbox
                d.rounded_rectangle([cx0, (plh - cbox) // 2, cx0 + cbox, (plh + cbox) // 2], radius=int(cbox * 0.08),
                                    fill=(255, 255, 255))
                cr = assets.crest(team).resize((crest_sz, crest_sz), Image.LANCZOS)
                plate.alpha_composite(cr, (cx0 + (cbox - crest_sz) // 2, (plh - crest_sz) // 2))
                text_w = cx0 - pad * 1.6
            else:
                text_w = Wp - 2 * pad
            if it.player_name:
                num = f"#{it.player_number} " if it.player_number else ""
                first, _, last = it.player_name.partition(" ")
                fnum = font("display", self.px(L["plate_number_px"]) * S)
                fname = font("medium", self.px(L["plate_name_px"]) * S)
                flast = font("display", self.px(L["plate_name_px"]) * S)
                total = fnum.getlength(num) + fname.getlength(first.upper() + " ") + flast.getlength(last.upper())
                k = min(1.0, text_w / max(total, 1))
                if k < 1:
                    fnum, fname, flast = (font(n, int(f_.size * k)) for n, f_ in
                                          (("display", fnum), ("medium", fname), ("display", flast)))
                ty = plh * 0.17
                x = pad
                d.text((x, ty), num, font=fnum, fill=(255, 255, 255))
                x += fnum.getlength(num)
                d.text((x, ty), first.upper() + " ", font=fname, fill=(255, 255, 255))
                x += fname.getlength(first.upper() + " ")
                d.text((x, ty), last.upper(), font=flast, fill=(255, 255, 255))
            if it.team_name:
                ft = font("bold", self.px(L["plate_team_px"]) * S)
                d.text((pad, plh * 0.6), it.team_name.upper(), font=ft, fill=rgb(self.c["text_dim"]))
            sprites.append(pil_to_premul(plate.resize((Wp // S, plh // S), Image.LANCZOS)))
        # --- analyst chips
        if it.extra_metrics and it.density == "dense":
            sprites.append(self._chips_sprite(it.extra_metrics, Wp // S, lang))
        self.sprite_cache[key] = sprites
        return sprites

    def _chips_sprite(self, chips, width: int, lang: str = "en") -> np.ndarray:
        S = 2
        Wp = width * S
        f_l = font("semibold", self.px(17) * S)
        f_v = font("display", self.px(28) * S)
        cols = 2
        rows = math.ceil(len(chips) / cols)
        cell_h = int(self.px(64) * S)
        im = Image.new("RGBA", (Wp, rows * cell_h), rgb(self.c["chip_bg"]) + (235,))
        d = ImageDraw.Draw(im)
        cw = Wp // cols
        for i, ch in enumerate(chips):
            cx, cy = (i % cols) * cw, (i // cols) * cell_h
            d.line([cx, cy, cx + cw, cy], fill=rgb(self.c["plate_edge"]), width=S)
            d.text((cx + self.px(16) * S, cy + self.px(7) * S), T.label(ch.label, lang).upper(), font=f_l,
                   fill=rgb(self.c["text_dim"]))
            d.text((cx + self.px(16) * S, cy + self.px(27) * S), f"{ch.value}{ch.unit}", font=f_v, fill=(255, 255, 255))
        return pil_to_premul(im.resize((Wp // S, rows * cell_h // S), Image.LANCZOS))

    def draw_stat_card(self, layer: np.ndarray, it: StatCard, ctx: RenderContext, tl: OverlayTimeline) -> None:
        sprites = self._panel_sprites(it, ctx.lang)
        L = self.lay
        Wp = sprites[0].shape[1]
        margin = int(self.width * L["safe_x"])
        x_final = self.width - margin - Wp if it.placement == "right" else margin
        y = self.px(L["panel_top"])
        gap = self.px(L["panel_gap"])
        for k, sp in enumerate(sprites):
            t0 = it.t_start + k * self.mo["stagger_s"]
            t1 = it.t_end - (len(sprites) - 1 - k) * self.mo["stagger_s"] * 0.6
            fin, fout = envelope(ctx.t, t0, t1, self.mo["slide_in_s"], self.mo["slide_out_s"])
            off = (1 - fin) * (Wp + margin) + fout * (Wp + margin)
            x = int(x_final + off) if it.placement == "right" else int(x_final - off)
            blit(layer, sp, x, y, min(1.0, fin * 1.4) * (1 - fout * 0.6))
            y += sp.shape[0] + (0 if k == 0 else gap // 2) + (gap if k == 0 else 0)

    # ===================================================== pitch markers
    def draw_pitch_markers(self, layer: np.ndarray, it: PitchMarkers, ctx: RenderContext, tl: OverlayTimeline) -> None:
        mo = self.mo
        fade_out = ease_in_cubic((ctx.t - (it.t_end - mo["fade_s"])) / mo["fade_s"])
        r_m = self.lay["marker_radius_m"]
        col = self.c.get(it.color, it.color if it.color.startswith("#") else self.c["marker"])
        shown = []
        for k, p in enumerate(it.points):
            ta = it.t_start + p.appear_at
            if ctx.t < ta:
                continue
            u = (ctx.t - ta) / mo["marker_pop_s"]
            pop = ease_out_back(u) if u < 1 else 1.0
            pulse = 1 + 0.07 * math.sin(2 * math.pi * mo["marker_pulse_hz"] * (ctx.t - ta)) * min(u, 1)
            scale = max(pop, 0.0) * pulse * (1 - 0.3 * fade_out)
            alpha = min(1.0, u * 2) * (1 - fade_out)
            shown.append((k, p, scale, alpha, ta))
        # Connecting trail on the grass (under the markers)
        if it.connect and len(shown) >= 2:
            pts = np.array([[p.x, p.y] for _, p, *_ in shown])
            dense = []
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                n = max(int(np.linalg.norm(b - a) / 0.8), 2)
                for j in range(0, n, 2):  # dashed
                    u0, u1 = j / n, min((j + 1) / n, 1)
                    seg = np.array([a + (b - a) * u0, a + (b - a) * u1])
                    dense.append(ground_ribbon(seg, 0.16))
            polys = [apply_h(ctx.H, q) for q in dense]
            draw_poly_layer(layer, polys, "#ffffff", 0.55 * min(s[3] for s in shown))
        for k, p, scale, alpha, ta in shown:
            if scale <= 0.01:
                continue
            # Glow ring that expands once on appearance
            u = (ctx.t - ta) / 0.9
            if u < 1:
                ring = ground_ribbon(circle_points(p.x, p.y, r_m * (1 + 1.6 * u), 40), 0.12 + 0.2 * (1 - u), closed=True)
                draw_poly_layer(layer, [apply_h(ctx.H, ring)], self.c["marker_glow"], alpha * (1 - u) * 0.9)
            if it.shape == "star":
                shape = star_points(p.x, p.y, r_m * scale, rot=-math.pi / 2)
            elif it.shape == "ring":
                shape = ground_ribbon(circle_points(p.x, p.y, r_m * scale, 40), 0.25, closed=True)
            else:
                shape = circle_points(p.x, p.y, r_m * 0.6 * scale, 28)
            img_pts = apply_h(ctx.H, shape)
            # A soft dark under-shadow improves contrast on bright grass.
            draw_poly_layer(layer, [img_pts + np.array([0, 2 * self.s])], "#000000", 0.25 * alpha)
            draw_poly_layer(layer, [img_pts], col, alpha)
            draw_poly_layer(layer, [img_pts], self.c["marker_edge"], alpha * 0.9, fill=False, thickness=max(1, self.px(1.6)))
            if it.numbered:
                c_img = apply_h(ctx.H, np.array([[p.x, p.y]]))[0]
                self._number_badge(layer, str(k + 1), c_img, alpha, scale)

    def _number_badge(self, layer: np.ndarray, label: str, at: np.ndarray, alpha: float, scale: float) -> None:
        key = ("badge", label)
        sp = self.sprite_cache.get(key)
        if sp is None:
            S = 3
            d_ = self.px(30) * S
            im = Image.new("RGBA", (d_, d_), (0, 0, 0, 0))
            dr = ImageDraw.Draw(im)
            dr.ellipse([0, 0, d_ - 1, d_ - 1], fill=(255, 255, 255, 255))
            dr.ellipse([S * 2, S * 2, d_ - 1 - S * 2, d_ - 1 - S * 2], fill=rgb(self.c["plate"]))
            f = font("display", int(d_ * 0.52))
            bb = f.getbbox(label)
            dr.text(((d_ - (bb[2] - bb[0])) / 2 - bb[0], (d_ - (bb[3] - bb[1])) / 2 - bb[1]), label, font=f,
                    fill=(255, 255, 255))
            sp = pil_to_premul(im.resize((d_ // S, d_ // S), Image.LANCZOS))
            self.sprite_cache[key] = sp
        h, w = sp.shape[:2]
        lift = self.px(34) * min(scale, 1.0)
        blit(layer, sp, int(at[0] - w / 2), int(at[1] - h / 2 - lift), alpha)

    # =================================================== player highlight
    def _player_screen(self, ctx: RenderContext, player_id: str):
        if ctx.tracks is None or player_id not in ctx.tracks.tracks:
            return None
        tr = ctx.tracks.tracks[player_id]
        if not tr.frames:
            return None
        i = int(np.searchsorted(tr.frames, ctx.frame))
        if i >= len(tr.frames) or tr.frames[i] != ctx.frame:
            # tolerate short gaps (missed detections): nearest frame within 6
            j = int(np.clip(i, 0, len(tr.frames) - 1))
            cand = [k for k in (j - 1, j) if 0 <= k < len(tr.frames)]
            k = min(cand, key=lambda k: abs(tr.frames[k] - ctx.frame))
            if abs(tr.frames[k] - ctx.frame) > 6:
                return None
            i = k
        box = tr.boxes[i]
        xy = tr.pitch_xy[i] if tr.pitch_xy else None
        spd = tr.speed_kmh[i] if tr.speed_kmh else None
        return box, xy, spd

    def draw_player_highlight(self, layer: np.ndarray, it: PlayerHighlight, ctx: RenderContext, tl: OverlayTimeline) -> None:
        got = self._player_screen(ctx, it.player_id)
        if got is None:
            return
        box, xy, spd = got
        fin, fout = envelope(ctx.t, it.t_start, it.t_end, 0.35, 0.35)
        alpha = fin * (1 - fout)
        col = self.c.get(it.color, self.c["accent"])
        if xy is not None:
            r = self.lay["ring_radius_m"] * (0.6 + 0.4 * ease_out_back(fin))
            spin = ctx.t * 1.4
            arc = ground_ribbon(circle_points(xy[0], xy[1], r, 40, spin, spin + 1.6 * math.pi), 0.2)
            draw_poly_layer(layer, [apply_h(ctx.H, arc) + np.array([0, 1.5 * self.s])], "#000000", alpha * 0.3)
            draw_poly_layer(layer, [apply_h(ctx.H, arc)], col, alpha)
        # Marker triangle above the head
        cx = (box[0] + box[2]) / 2
        top = box[1] - self.px(14) - self.px(4) * math.sin(ctx.t * 5)
        tri = np.array([[cx - self.px(9), top - self.px(14)], [cx + self.px(9), top - self.px(14)], [cx, top]])
        draw_poly_layer(layer, [tri], col, alpha)
        # Name tag + chips
        y = top - self.px(22)
        if it.show_name and it.display_name:
            tag = self._tag_sprite(f"{it.display_name}", self.c["plate"], "#ffffff", f"#{it.player_number}" if it.player_number else None)
            blit(layer, tag, int(cx - tag.shape[1] / 2), int(y - tag.shape[0]), alpha)
            y -= tag.shape[0] + self.px(6)
        for ch in it.chips:
            if ctx.t < it.t_start + ch.appear_at:
                continue
            u = ease_out_back((ctx.t - it.t_start - ch.appear_at) / self.mo["chip_pop_s"])
            sp = self._tag_sprite(f"{ch.value} {ch.unit}".strip(), self.c["speed"], self.c["speed_text"],
                                  T.label(ch.label, ctx.lang).upper())
            blit(layer, sp, int(cx - sp.shape[1] / 2), int(y - sp.shape[0] * u), alpha * min(u, 1))
            y -= sp.shape[0] + self.px(6)

    def _tag_sprite(self, text: str, bg: str, fg: str, prefix: str | None = None) -> np.ndarray:
        key = ("tag", text, bg, fg, prefix)
        sp = self.sprite_cache.get(key)
        if sp is not None:
            return sp
        S = 2
        f = font("bold", self.px(self.lay["chip_font_px"]) * S)
        fp = font("display", self.px(self.lay["chip_font_px"]) * S)
        pad = int(f.size * 0.45)
        pw = int(fp.getlength(prefix) + pad) if prefix else 0
        w = int(f.getlength(text)) + pw + 2 * pad
        h = int(f.size * 1.5)
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(h * 0.22), fill=rgb(bg) + (245,))
        if prefix:
            d.rounded_rectangle([0, 0, pw + pad, h - 1], radius=int(h * 0.22), fill=rgb(self.c["accent"]) + (255,))
            d.text((pad * 0.6, h * 0.16), prefix, font=fp, fill=(255, 255, 255))
        d.text((pw + pad * 1.2, h * 0.16), text, font=f, fill=rgb(fg))
        sp = pil_to_premul(im.resize((w // S, h // S), Image.LANCZOS))
        self.sprite_cache[key] = sp
        return sp

    # ========================================================= speed tag
    def draw_speed_tag(self, layer: np.ndarray, it: SpeedTag, ctx: RenderContext, tl: OverlayTimeline) -> None:
        fin, fout = envelope(ctx.t, it.t_start, it.t_end, 0.25, 0.3)
        alpha = fin * (1 - fout)
        label = "BALL" if it.subject == "ball" else "SPEED"
        tb = self._text(it, ctx.lang)
        if tb and tb.label:
            label = tb.label.upper()
        text = f"{it.speed_kmh:.0f} km/h" + (" (est.)" if it.estimated else "")
        sp = self._tag_sprite(text, self.c["speed"], self.c["speed_text"], label)
        anchor = None
        if it.player_id:
            got = self._player_screen(ctx, it.player_id)
            if got is not None:
                b = got[0]
                anchor = ((b[0] + b[2]) / 2, b[1] - self.px(70))
        if anchor is None and it.style.get("x") and it.style.get("y"):
            anchor = tuple(apply_h(ctx.H, np.array([[float(it.style["x"]), float(it.style["y"])]]))[0] - [0, self.px(40)])
        if anchor is None:
            return
        u = ease_out_back(fin)
        blit(layer, sp, int(anchor[0] - sp.shape[1] / 2), int(anchor[1] - sp.shape[0] * u), alpha)

    # ========================================================== pass line
    def draw_pass_line(self, layer: np.ndarray, it: PassLine, ctx: RenderContext, tl: OverlayTimeline) -> None:
        dur = it.t_end - it.t_start
        grow = ease_out_cubic((ctx.t - it.t_start) / min(0.9, dur / 3))
        fout = ease_in_cubic((ctx.t - (it.t_end - 0.4)) / 0.4)
        alpha = 1 - fout
        a = np.array([it.start.x, it.start.y])
        b = np.array([it.end.x, it.end.y])
        end = a + (b - a) * grow
        col = {"perfect": self.c["pass_good"], "accurate": self.c["pass_good"], "loose": self.c["pass_loose"],
               "failed": self.c["pass_failed"]}[it.accuracy]
        n = 30
        pts = np.array([a + (end - a) * u for u in np.linspace(0, 1, n)])
        rib = ground_ribbon(pts, 0.35)
        draw_poly_layer(layer, [apply_h(ctx.H, rib) + np.array([0, 2 * self.s])], "#000000", 0.3 * alpha)
        draw_poly_layer(layer, [apply_h(ctx.H, rib)], col, alpha)
        # Arrow head on the ground
        d = b - a
        dn = d / (np.linalg.norm(d) + 1e-9)
        nrm = np.array([-dn[1], dn[0]])
        head = np.array([end + dn * 1.6, end + nrm * 1.0, end - nrm * 1.0])
        draw_poly_layer(layer, [apply_h(ctx.H, head)], col, alpha)
        for p in (a,):
            draw_poly_layer(layer, [apply_h(ctx.H, circle_points(p[0], p[1], 0.7, 24))], "#ffffff", alpha)
        if grow > 0.98:
            tb = self._text(it, ctx.lang)
            label = tb.label if tb and tb.label else f"{it.distance_m:.0f} m"
            mid = apply_h(ctx.H, np.array([(a + b) / 2]))[0]
            sp = self._tag_sprite(label, self.c["plate"], "#ffffff", f"{it.difficulty * 10:.1f}")
            blit(layer, sp, int(mid[0] - sp.shape[1] / 2), int(mid[1] - sp.shape[0] - self.px(18)), alpha)

    # ====================================================== lower third
    def draw_narrative_lower_third(self, layer: np.ndarray, it: NarrativeLowerThird, ctx: RenderContext,
                                   tl: OverlayTimeline) -> None:
        tb = self._text(it, ctx.lang)
        if tb is None:
            return
        key = ("lt", it.id, ctx.lang)
        sp = self.sprite_cache.get(key)
        if sp is None:
            sp = self._lower_sprite(tb.label or "WHY IT MATTERS", tb.body or tb.headline, it.tone, ctx.lang)
            self.sprite_cache[key] = sp
        fin, fout = envelope(ctx.t, it.t_start, it.t_end, 0.45, 0.35)
        x = int(self.width * self.lay["safe_x"])
        y = int(self.height * (1 - self.lay["safe_y"])) - sp.shape[0]
        # Wipe reveal from the left
        reveal = int(sp.shape[1] * fin)
        if reveal <= 2:
            return
        blit(layer, sp[:, :reveal], x, y + int(fout * self.px(30)), 1 - fout)

    def _lower_sprite(self, label: str, body: str, tone: str, lang: str) -> np.ndarray:
        S = 2
        Lw = self.lay
        maxw = int(self.width * 0.6) * S
        fl = font("display", self.px(Lw["lower_label_px"]) * S)
        fb = font("semibold", self.px(Lw["lower_font_px"]) * S)
        pad = int(fb.size * 0.6)
        lines = wrap(body, fb, maxw - 2 * pad, 2)
        while len(lines) > 2:
            fb = font("semibold", int(fb.size * 0.92))
            lines = wrap(body, fb, maxw - 2 * pad, 2)
        lab_h = int(fl.size * 1.7)
        body_h = int(fb.size * 1.32 * len(lines) + pad * 1.1)
        w = int(max(fb.getlength(li) for li in lines) + 2 * pad) if lines else 400
        w = max(w, int(fl.getlength(label.upper()) + 6 * pad))
        h = lab_h + body_h
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        tone_col = self.c.get(tone, self.c["lower_label_bg"]) if tone != "neutral" else self.c["lower_label_bg"]
        lw = int(fl.getlength(label.upper()) + 2 * pad)
        d.rectangle([0, 0, lw, lab_h], fill=rgb(tone_col) + (255,))
        d.text((pad, lab_h * 0.2), label.upper(), font=fl, fill=(255, 255, 255))
        fbadge = font("bold", int(fl.size * 0.9))
        d.text((lw + pad * 0.6, lab_h * 0.24), lang.upper(), font=fbadge, fill=rgb(self.c["text_dim"]))
        d.rectangle([0, lab_h, w, h], fill=rgb(self.c["lower_bg"]) + (236,))
        d.rectangle([0, lab_h, int(S * 5), h], fill=rgb(tone_col) + (255,))
        for i, line in enumerate(lines):
            d.text((pad, lab_h + pad * 0.45 + i * fb.size * 1.32), line, font=fb, fill=(255, 255, 255))
        return pil_to_premul(im.resize((w // S, h // S), Image.LANCZOS))

    # ============================================================ minimap
    def draw_minimap(self, layer: np.ndarray, it: MinimapItem, ctx: RenderContext, tl: OverlayTimeline) -> None:
        fin, fout = envelope(ctx.t, it.t_start, it.t_end, 0.4, 0.4)
        alpha = fin * (1 - fout)
        if alpha <= 0.01:
            return
        mw, mh = self.px(self.lay["minimap_w"]), self.px(self.lay["minimap_h"])
        x0 = self.width - int(self.width * self.lay["safe_x"]) - mw
        y0 = int(self.height * (1 - self.lay["safe_y"])) - mh
        S = 2
        im = Image.new("RGBA", (mw * S, mh * S), rgb(self.c["minimap_bg"]) + (210,))
        d = ImageDraw.Draw(im)
        pad = 8 * S

        def m2p(x: float, y: float) -> tuple[float, float]:
            return pad + x / pm.L * (mw * S - 2 * pad), mh * S - pad - y / pm.W * (mh * S - 2 * pad)

        line_col = rgb(self.c["minimap_line"]) + (200,)
        for pl in pm.polylines():
            d.line([m2p(*p) for p in pl], fill=line_col, width=S)
        # Camera footprint (image corners back-projected onto the pitch)
        try:
            Hinv = np.linalg.inv(ctx.H)
            ys = self.height * np.array([0.32, 0.32, 1.0, 1.0])
            corners = apply_h(Hinv, np.c_[[0, self.width, self.width, 0], ys])
            corners[:, 0] = np.clip(corners[:, 0], -3, pm.L + 3)
            corners[:, 1] = np.clip(corners[:, 1], -3, pm.W + 3)
            d.polygon([m2p(*c) for c in corners], fill=(255, 255, 255, 34), outline=(255, 255, 255, 110))
        except np.linalg.LinAlgError:
            pass
        if ctx.tracks is not None:
            for pid in ctx.tracks.tracks:
                got = self._player_screen(ctx, pid)
                if got is None or got[1] is None:
                    continue
                x, y = got[1]
                team = self.meta.player(pid).team_id if pid != "ball" else None
                c = self.meta.team(team).kit.shirt if team else "#ffffff"
                r = 5 * S if pid == it.focus_player_id else 3.4 * S
                px_, py_ = m2p(x, y)
                d.ellipse([px_ - r, py_ - r, px_ + r, py_ + r], fill=rgb(c) + (255,),
                          outline=rgb(self.c["accent"]) if pid == it.focus_player_id else (20, 20, 20))
        for p in it.points:
            if ctx.t < it.t_start + p.appear_at:
                continue
            px_, py_ = m2p(p.x, p.y)
            d.polygon([tuple(q) for q in star_points(px_, py_, 6.5 * S)], fill=rgb(self.c["marker"]) + (255,),
                      outline=(255, 255, 255))
        sp = pil_to_premul(im.resize((mw, mh), Image.LANCZOS))
        blit(layer, sp, x0, y0 + int((1 - fin) * self.px(20)), alpha)

    # ========================================================= recap card
    def draw_recap_card(self, layer: np.ndarray, it: RecapCard, ctx: RenderContext, tl: OverlayTimeline) -> None:
        key = ("recap", it.id, ctx.lang)
        sp = self.sprite_cache.get(key)
        if sp is None:
            S = 2
            w, h = int(self.width * 0.56) * S, int(self.height * 0.62) * S
            im = vertical_gradient_h(w, h, self.c["panel_header_from"], self.c["plate"])
            d = ImageDraw.Draw(im)
            tb = self._text(it, ctx.lang)
            fh = font("display", self.px(46) * S)
            fb = font("semibold", self.px(24) * S)
            fm = font("bold", self.px(26) * S)
            pad = self.px(36) * S
            y = pad
            for line in wrap((tb.headline if tb else "").upper(), fh, w - 2 * pad, 2):
                d.text((pad, y), line, font=fh, fill=(255, 255, 255))
                y += fh.size * 1.1
            y += pad * 0.3
            if tb and tb.body:
                for line in wrap(tb.body, fb, w - 2 * pad, 3):
                    d.text((pad, y), line, font=fb, fill=rgb(self.c["text_dim"]))
                    y += fb.size * 1.3
            y += pad * 0.4
            for m in it.moments[:5]:
                d.rectangle([pad, y + fm.size * 0.15, pad + S * 5, y + fm.size * 1.05], fill=rgb(self.c["accent"]))
                d.text((pad + S * 18, y), f"{m.label}  {m.headline}", font=fm, fill=(255, 255, 255))
                y += fm.size * 1.55
            sp = pil_to_premul(im.resize((w // S, h // S), Image.LANCZOS))
            self.sprite_cache[key] = sp
        fin, fout = envelope(ctx.t, it.t_start, it.t_end, 0.5, 0.4)
        x = (self.width - sp.shape[1]) // 2
        y = (self.height - sp.shape[0]) // 2 + int((1 - fin) * self.px(40))
        blit(layer, sp, x, y, fin * (1 - fout))


# ---------------------------------------------------------------- helpers
def ground_ribbon(pts: np.ndarray, width_m: float, closed: bool = False) -> np.ndarray:
    """Turn a pitch-space polyline into a polygon of real width (metres), for perspective-correct strokes."""
    pts = np.asarray(pts, float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    d = np.gradient(pts, axis=0)
    n = np.c_[-d[:, 1], d[:, 0]]
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    left = pts + n * width_m / 2
    right = pts - n * width_m / 2
    return np.vstack([left, right[::-1]])


def vertical_gradient_h(w: int, h: int, top: str, bottom: str) -> Image.Image:
    a, b = np.array(rgb(top), float), np.array(rgb(bottom), float)
    u = np.linspace(0, 1, h)[:, None]
    # Diagonal sheen like broadcast panels: blend vertical + slight horizontal
    v = np.linspace(0, 1, w)[None, :]
    k = np.clip(0.75 * u + 0.25 * v, 0, 1)[..., None]
    arr = (a + (b - a) * k).astype(np.uint8)
    alpha = np.full((h, w, 1), 255, np.uint8)
    return Image.fromarray(np.concatenate([arr, alpha], axis=-1), "RGBA")


def wrap(text: str, f, max_w: float, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if f.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
