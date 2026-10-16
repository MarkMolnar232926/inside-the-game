"""Rights-clear synthetic broadcast footage.

Renders a tactical-camera view of a synthetic match directly from the
generated 10 Hz tracking: striped grass, painted lines, goals, ad boards with
our own fictional branding, a crowd, 22 animated players in fictional kits,
a referee and the ball. The camera is a virtual operator panning/zooming with
play. Because we know the camera exactly, every frame also yields a
ground-truth homography and ground-truth player boxes, which we use to
*evaluate* the vision stack (the vision stack itself never reads them).

Usage::

    python -m pitchlens.footage.broadcast --match data/synthetic/demo --start 4130 --duration 30 \
        --out outputs/footage/demo_clip
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pitchlens.schema import CameraTrack, MatchMeta
from pitchlens.vision import pitch_model as pm
from pitchlens.vision.camera import CameraOperator, OperatorConfig, PinholeCamera

SKIN = ["#f1c9a5", "#e0ac83", "#c68a5e", "#a86b43", "#7d4a2b", "#5a3420"]
REF_KIT = {"shirt": "#1a1a1a", "shorts": "#1a1a1a", "socks": "#1a1a1a"}


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ----------------------------------------------------------------- textures
@dataclass
class Texture:
    img: np.ndarray  # BGRA
    origin: np.ndarray  # world point of pixel (0, 0)
    e1: np.ndarray  # world vector per texture pixel along columns
    e2: np.ndarray  # world vector per texture pixel along rows


def grass_texture(ppm: float = 14.0, rng: np.random.Generator | None = None) -> Texture:
    rng = rng or np.random.default_rng(0)
    x0, x1, y0, y1 = -14.0, 119.0, -9.0, 76.0
    w, h = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    xs = x0 + (np.arange(w) + 0.5) / ppm
    ys = y0 + (np.arange(h) + 0.5) / ppm
    X, Y = np.meshgrid(xs, ys)
    band = np.floor(X / (pm.L / 18)).astype(int) % 2
    base = np.where(band[..., None] == 0, np.array([58, 128, 52]), np.array([50, 113, 45])).astype(np.float32)
    inside = (X >= -2) & (X <= pm.L + 2) & (Y >= -2) & (Y <= pm.W + 2)
    base[~inside] = np.array([44, 98, 40])
    low = cv2.resize(rng.normal(0, 1, (h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hi = rng.normal(0, 1, (h, w)).astype(np.float32)
    shade = 1 + 0.05 * low + 0.025 * hi
    # Soft floodlight falloff towards the far side.
    shade *= (1.04 - 0.08 * (Y / pm.W)).astype(np.float32)
    img = np.clip(base * shade[..., None], 0, 255).astype(np.uint8)
    bgra = cv2.cvtColor(np.ascontiguousarray(img[..., ::-1]), cv2.COLOR_BGR2BGRA)  # RGB -> BGRA
    return Texture(bgra, np.array([x0, y0, 0.0]), np.array([1 / ppm, 0, 0]), np.array([0, 1 / ppm, 0]))


def stand_texture(rng: np.random.Generator, ppm: float = 7.0) -> Texture:
    x0, x1 = -45.0, 150.0
    slope = math.radians(30)
    depth = 46.0
    w, h = int((x1 - x0) * ppm), int(depth * ppm)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 30, 28)
    palette = np.array([hex_bgr(c) for c in ("#c8233a", "#e8e8e8", "#1b2a6b", "#17a398", "#2e2e38", "#d9a441",
                                              "#8a1f2c", "#f2f4f7", "#3b3b46")])
    weights = np.array([0.22, 0.14, 0.08, 0.14, 0.18, 0.04, 0.08, 0.06, 0.06])
    n = int(w * h / 9)
    pts = np.c_[rng.integers(0, w, n), rng.integers(0, h, n)]
    cols = palette[rng.choice(len(palette), n, p=weights / weights.sum())]
    for (px, py), c in zip(pts, cols, strict=True):
        cv2.circle(img, (int(px), int(py)), 1, tuple(int(v) for v in c), -1)
    for r in range(0, h, int(ppm * 0.9)):  # seat rows
        img[r : r + 1] = (img[r : r + 1] * 0.6).astype(np.uint8)
    for tier in (int(h * 0.42), int(h * 0.44)):  # tier balcony
        img[tier : tier + 3] = (70, 64, 60)
    for vx in range(0, w, int(ppm * 22)):  # stairways
        img[:, vx : vx + 4] = (82, 76, 72)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Row index 0 = top of the stand (far from pitch); flip so origin is at the pitch edge.
    bgra = bgra[::-1].copy()
    e2 = np.array([0, math.cos(slope), math.sin(slope)]) / ppm
    return Texture(bgra, np.array([x0, 77.0, 0.6]), np.array([1 / ppm, 0, 0]), e2)


def board_texture(length: float, ppm: float = 36.0, height: float = 0.9) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    w, h = int(length * ppm), int(height * ppm)
    im = Image.new("RGB", (w, h), (20, 26, 72))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(str(Path(__file__).resolve().parents[3] / "assets/fonts/Inter-ExtraBold.otf"), int(h * 0.62))
    msgs = ["PITCHLENS", "SYNTHETIC LEAGUE", "INSIDE THE GAME", "KESTREL BAY FC", "IRONVALE ATHLETIC"]
    x, i = 10, 0
    seg = int(ppm * 15)
    while x < w:
        bg = [(20, 26, 72), (255, 255, 255), (23, 163, 152)][i % 3]
        fg = [(255, 255, 255), (20, 26, 72), (255, 255, 255)][i % 3]
        d.rectangle([x - 10, 0, x - 10 + seg, h], fill=bg)
        d.text((x + 8, h * 0.17), msgs[i % len(msgs)], font=font, fill=fg)
        x += seg
        i += 1
    arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGRA)
    return arr[::-1].copy()  # row 0 at ground level


# ------------------------------------------------------------------ helpers
def warp_texture(frame: np.ndarray, cam: PinholeCamera, tex: Texture) -> None:
    """Composite a planar world texture into ``frame`` (in place), warping only its screen ROI."""
    H = cam.plane_homography(tex.origin, tex.e1, tex.e2)
    h, w = tex.img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ H.T
    if (corners[:, 2] <= 1e-6).any():
        x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]  # plane crosses the camera: warp full frame
    else:
        c = corners[:, :2] / corners[:, 2:]
        x0, y0 = np.floor(np.clip(c.min(axis=0), 0, [frame.shape[1], frame.shape[0]])).astype(int)
        x1, y1 = np.ceil(np.clip(c.max(axis=0), 0, [frame.shape[1], frame.shape[0]])).astype(int)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    warped = cv2.warpPerspective(tex.img, T @ H, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    roi = frame[y0:y1, x0:x1]
    np.copyto(roi, warped[..., :3], where=warped[..., 3:4] > 127)


def draw_lines(frame: np.ndarray, cam: PinholeCamera) -> None:
    quads = pm.line_quads()
    pts = np.c_[quads.reshape(-1, 2), np.zeros(len(quads) * 4)]
    uv, z = cam.project(pts)
    uv = uv.reshape(-1, 4, 2)
    ok = (z.reshape(-1, 4) > 0).all(axis=1)
    polys = [np.round(q * 16).astype(np.int32) for q in uv[ok]]
    overlay = frame.copy()
    cv2.fillPoly(overlay, polys, (236, 240, 238), lineType=cv2.LINE_AA, shift=4)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    for s in pm.spots():
        p, _ = cam.project(np.array([[s[0], s[1], 0.0]]))
        r = max(cam.f / max(float(np.linalg.norm(cam.C - [s[0], s[1], 0])), 1) * 0.15, 1.0)
        cv2.ellipse(frame, tuple(np.round(p[0]).astype(int)), (int(r * 1.3), max(int(r * 0.5), 1)), 0, 0, 360,
                    (236, 240, 238), -1, cv2.LINE_AA)


def draw_goals(frame: np.ndarray, cam: PinholeCamera) -> None:
    for gx, s in ((0.0, -1), (pm.L, 1)):
        y0, y1 = pm.W / 2 - pm.GOAL_W / 2, pm.W / 2 + pm.GOAL_W / 2
        d = 2.0
        front = [(gx, y0, 0), (gx, y0, pm.GOAL_H), (gx, y1, pm.GOAL_H), (gx, y1, 0)]
        back = [(gx + s * d, y0, 0), (gx + s * d * 0.6, y0, pm.GOAL_H), (gx + s * d * 0.6, y1, pm.GOAL_H),
                (gx + s * d, y1, 0)]
        P, z = cam.project(np.array(front + back, float))
        if (z <= 0).any():
            continue
        P = np.round(P * 16).astype(np.int32)
        net = frame.copy()
        for k in range(3):
            cv2.line(net, tuple(P[k + 4]), tuple(P[k + 5]), (215, 220, 220), 1, cv2.LINE_AA, shift=4)
        for k in range(4):
            cv2.line(net, tuple(P[k]), tuple(P[k + 4]), (215, 220, 220), 1, cv2.LINE_AA, shift=4)
        # Net mesh
        for u in np.linspace(0, 1, 12):
            a = P[4] + (P[7] - P[4]) * u
            b = P[5] + (P[6] - P[5]) * u
            cv2.line(net, tuple(a.astype(np.int32)), tuple(b.astype(np.int32)), (200, 205, 205), 1, cv2.LINE_AA, shift=4)
        cv2.addWeighted(net, 0.45, frame, 0.55, 0, frame)
        thick = max(int(cam.f / float(np.linalg.norm(cam.C - [gx, pm.W / 2, 0])) * 0.12), 1)
        for k in range(3):
            cv2.line(frame, tuple(P[k]), tuple(P[k + 1]), (250, 250, 250), thick, cv2.LINE_AA, shift=4)


def draw_person(frame: np.ndarray, cam: PinholeCamera, x: float, y: float, heading: np.ndarray, phase: float,
                stride: float, kit: dict[str, str], skin: str, height: float = 1.82) -> tuple[float, float, float, float] | None:
    """Draw a stylised player figure; returns its image bounding box."""
    hx, hy = heading
    side = np.array([-hy, hx])
    sw = math.sin(phase) * 0.32 * stride

    def P(dx_fwd: float, dx_side: float, z: float) -> np.ndarray:
        wx = x + hx * dx_fwd + side[0] * dx_side
        wy = y + hy * dx_fwd + side[1] * dx_side
        return np.array([wx, wy, z])

    pts = np.array(
        [
            P(0, 0, 0),  # 0 ground centre
            P(sw, 0.11, 0.0),  # 1 foot L
            P(-sw, -0.11, 0.0),  # 2 foot R
            P(0, 0.12, 0.92),  # 3 hip L
            P(0, -0.12, 0.92),  # 4 hip R
            P(0, 0.2, 1.48),  # 5 shoulder L
            P(0, -0.2, 1.48),  # 6 shoulder R
            P(0, 0, height - 0.14),  # 7 head centre
            P(-sw * 0.8, 0.27, 0.95),  # 8 hand L
            P(sw * 0.8, -0.27, 0.95),  # 9 hand R
            P(0, 0, 0.66),  # 10 shorts bottom centre
            P(sw * 0.45, 0.11, 0.42),  # 11 knee L (sock top)
            P(-sw * 0.45, -0.11, 0.42),  # 12 knee R
            P(0, 0, height),  # 13 top of head
        ]
    )
    uv, z = cam.project(pts)
    if (z <= 0).any():
        return None
    scale = cam.f / float(z[0])  # pixels per metre at the player's depth
    if not (-80 < uv[0, 0] < cam.width + 80 and -150 < uv[0, 1] < cam.height + 150):
        return None
    sh = 16
    U = np.round(uv * sh).astype(np.int32)

    def t(m: float) -> int:
        return max(int(round(m * scale)), 1)

    # Shadow
    cv2.ellipse(frame, tuple(U[0]), (int(0.42 * scale * sh), int(0.16 * scale * sh)), 0, 0, 360, (24, 52, 22), -1,
                cv2.LINE_AA, shift=4)
    shirt, shorts, socks = hex_bgr(kit["shirt"]), hex_bgr(kit["shorts"]), hex_bgr(kit["socks"])
    # Legs: socks below knee, skin above
    skin_c = hex_bgr(skin)
    for foot, knee, hip in ((1, 11, 3), (2, 12, 4)):
        cv2.line(frame, tuple(U[knee]), tuple(U[hip]), skin_c, t(0.13), cv2.LINE_AA, shift=4)
        cv2.line(frame, tuple(U[foot]), tuple(U[knee]), socks, t(0.13), cv2.LINE_AA, shift=4)
        cv2.circle(frame, tuple(U[foot]), int(t(0.07) * sh), (25, 25, 25), -1, cv2.LINE_AA, shift=4)
    # Arms
    for sh_i, hand in ((5, 8), (6, 9)):
        cv2.line(frame, tuple(U[sh_i]), tuple(U[hand]), skin_c, t(0.09), cv2.LINE_AA, shift=4)
    # Shorts
    sh_poly = np.array([U[3], U[4], U[4] + (U[10] - (U[3] + U[4]) // 2), U[3] + (U[10] - (U[3] + U[4]) // 2)])
    cv2.fillPoly(frame, [sh_poly], shorts, cv2.LINE_AA, shift=4)
    # Torso (shirt) with short sleeves
    torso = np.array([U[5], U[6], U[4], U[3]])
    cv2.fillPoly(frame, [torso], shirt, cv2.LINE_AA, shift=4)
    for sh_i, hand in ((5, 8), (6, 9)):
        sleeve_end = U[sh_i] + (U[hand] - U[sh_i]) * 0.35
        cv2.line(frame, tuple(U[sh_i]), tuple(sleeve_end.astype(np.int32)), shirt, t(0.13), cv2.LINE_AA, shift=4)
    # Head
    r = int(0.115 * scale * sh)
    cv2.circle(frame, tuple(U[7]), r, skin_c, -1, cv2.LINE_AA, shift=4)
    cv2.circle(frame, tuple(U[7] + np.array([0, -r // 3])), int(r * 0.85), (30, 30, 34), -1, cv2.LINE_AA, shift=4)
    cv2.circle(frame, tuple(U[7] + np.array([0, r // 5])), int(r * 0.82), skin_c, -1, cv2.LINE_AA, shift=4)
    xs, ys = uv[:, 0], uv[:, 1]
    return float(xs.min() - 2), float(ys[13] - 2), float(xs.max() + 2), float(uv[0, 1] + 0.16 * scale + 2)


def draw_ball(frame: np.ndarray, cam: PinholeCamera, x: float, y: float, z: float) -> tuple[float, float] | None:
    uv, d = cam.project(np.array([[x, y, z + 0.11], [x, y, 0.0]]))
    if d[0] <= 0:
        return None
    r = max(cam.f / d[0] * 0.13, 2.0)
    sh = 16
    cv2.ellipse(frame, tuple(np.round(uv[1] * sh).astype(int)), (int(r * 1.3 * sh), int(r * 0.5 * sh)), 0, 0, 360,
                (25, 50, 22), -1, cv2.LINE_AA, shift=4)
    c = tuple(np.round(uv[0] * sh).astype(int))
    cv2.circle(frame, c, int(r * sh), (250, 250, 250), -1, cv2.LINE_AA, shift=4)
    cv2.circle(frame, c, int(r * sh), (90, 90, 90), 1, cv2.LINE_AA, shift=4)
    return float(uv[0, 0]), float(uv[0, 1])


# ------------------------------------------------------------------- render
@dataclass
class ClipData:
    t: np.ndarray  # (F,) match clock per video frame
    ids: list[str]
    xy: np.ndarray  # (F, N, 2), NaN when the entity is not on the pitch
    ball: np.ndarray  # (F, 3)


def interpolate_clip(tracking: pd.DataFrame, t0: float, duration: float, fps: float = 30.0) -> ClipData:
    tr = tracking[(tracking.t >= t0 - 1) & (tracking.t <= t0 + duration + 1)]
    times = t0 + np.arange(int(round(duration * fps))) / fps
    ids = sorted(e for e in tr.entity.unique() if e != "ball")
    xy = np.full((len(times), len(ids), 2), np.nan)
    for j, e in enumerate(ids):
        s = tr[tr.entity == e]
        ok = (times >= s.t.min()) & (times <= s.t.max())
        xy[ok, j, 0] = np.interp(times[ok], s.t, s.x)
        xy[ok, j, 1] = np.interp(times[ok], s.t, s.y)
    b = tr[tr.entity == "ball"]
    ball = np.c_[np.interp(times, b.t, b.x), np.interp(times, b.t, b.y), np.interp(times, b.t, b.z)]
    return ClipData(times, ids, xy, ball)


class BroadcastRenderer:
    def __init__(self, meta: MatchMeta, width: int = 1920, height: int = 1080, seed: int = 3):
        self.meta = meta
        self.width, self.height = width, height
        rng = np.random.default_rng(seed)
        self.grass = grass_texture(rng=rng)
        self.stand = stand_texture(rng)
        board = board_texture(150.0)
        ppm = board.shape[0] / 0.9
        self.board_far = Texture(board, np.array([-22.0, 70.5, 0.0]), np.array([1 / ppm, 0, 0]), np.array([0, 0, 1 / ppm]))
        side = board[:, : int(ppm * 60)].copy()
        self.board_right = Texture(side, np.array([110.0, 64.0, 0.0]), np.array([0, -1 / ppm, 0]), np.array([0, 0, 1 / ppm]))
        self.board_left = Texture(side, np.array([-5.0, 4.0, 0.0]), np.array([0, 1 / ppm, 0]), np.array([0, 0, 1 / ppm]))
        # End stands behind each goal (same crowd texture, rotated planes).
        st = self.stand
        sl = math.radians(30)
        ppm_s = 1 / np.linalg.norm(st.e1)
        self.stand_left = Texture(st.img, np.array([-9.0, -45.0, 0.6]), np.array([0, 1 / ppm_s, 0]),
                                  np.array([-math.cos(sl), 0, math.sin(sl)]) / ppm_s)
        self.stand_right = Texture(st.img, np.array([114.0, 150.0, 0.6]), np.array([0, -1 / ppm_s, 0]),
                                   np.array([math.cos(sl), 0, math.sin(sl)]) / ppm_s)
        self.kits = {}
        self.skin = {}
        for team in (meta.home, meta.away):
            for p in team.players:
                k = team.kit
                shirt = k.gk_shirt if p.position.value == "GK" else k.shirt
                self.kits[p.id] = {"shirt": shirt, "shorts": k.shorts, "socks": k.socks if p.position.value != "GK" else shirt}
                self.skin[p.id] = SKIN[p.skin_tone]
        self.noise_rng = np.random.default_rng(seed + 1)

    def cameras(self, clip: ClipData, fps: float, cfg: OperatorConfig | None = None) -> list[PinholeCamera]:
        return CameraOperator(cfg or OperatorConfig(width=self.width, height=self.height)).path(clip.ball[:, :2], fps)

    def referee_path(self, clip: ClipData, fps: float) -> np.ndarray:
        """Referee shadows play on a diagonal, ~12 m from the ball (rendered only; not in the dataset)."""
        b = clip.ball[:, :2]
        k = max(int(fps * 2), 1)
        kernel = np.ones(k) / k
        sm = np.c_[np.convolve(np.pad(b[:, 0], (k, k), mode="edge"), kernel, "same")[k:-k],
                   np.convolve(np.pad(b[:, 1], (k, k), mode="edge"), kernel, "same")[k:-k]]
        return sm + np.array([-9.0, 7.0])

    def render(self, clip: ClipData, cams: list[PinholeCamera], out_path: Path, fps: float = 30.0,
               gt_dir: Path | None = None, frames_dir: Path | None = None, every_png: int = 0) -> dict:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
             f"{self.width}x{self.height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
            stdin=subprocess.PIPE,
        )
        ref = self.referee_path(clip, fps)
        F = len(clip.t)
        prev = np.nan_to_num(clip.xy[0])
        dist = np.zeros(len(clip.ids) + 1)
        gt_rows = []
        sky = np.zeros((self.height, self.width, 3), np.uint8)
        sky[:] = (34, 26, 22)
        yy = np.linspace(0, 1, self.height)[:, None]
        xx = np.linspace(-1, 1, self.width)[None, :]
        vig = (255 * (1 - 0.18 * (xx**2 + (yy - 0.5) ** 2 * 1.2))).astype(np.uint8)
        vignette = cv2.merge([vig, vig, vig])
        noise_bank = [self.noise_rng.normal(6, 2.2, (self.height, self.width, 3)).clip(0, 12).astype(np.uint8)
                      for _ in range(6)]
        for i in range(F):
            cam = cams[i]
            frame = sky.copy()
            warp_texture(frame, cam, self.stand_left)
            warp_texture(frame, cam, self.stand_right)
            warp_texture(frame, cam, self.stand)
            warp_texture(frame, cam, self.grass)
            draw_lines(frame, cam)
            for tex in (self.board_far, self.board_right, self.board_left):
                warp_texture(frame, cam, tex)
            draw_goals(frame, cam)
            # Draw people far-to-near (painter's algorithm).
            people = []
            cur = clip.xy[i]
            for j, pid in enumerate(clip.ids):
                if np.isnan(cur[j, 0]):
                    continue
                people.append((cur[j, 1], j, pid))
            people.append((ref[i, 1], -1, "referee"))
            people.sort(key=lambda p: -p[0])
            for _, j, pid in people:
                if j >= 0:
                    x, y = cur[j]
                    v = (cur[j] - prev[j]) * fps if i > 0 else np.zeros(2)
                else:
                    x, y = ref[i]
                    v = (ref[i] - ref[i - 1]) * fps if i > 0 else np.zeros(2)
                spd = float(np.linalg.norm(v))
                dist[j] += spd / fps
                heading = v / spd if spd > 0.3 else np.array([1.0 if (j < 0 or pid.startswith(self.meta.home.id)) else -1.0, 0.0])
                kit = REF_KIT if j < 0 else self.kits[pid]
                skin = SKIN[2] if j < 0 else self.skin[pid]
                bbox = draw_person(frame, cam, float(x), float(y), heading, dist[j] * 2 * math.pi / 2.2,
                                   min(spd / 6.0, 1.0), kit, skin)
                if bbox is not None and gt_dir is not None:
                    gt_rows.append((i, pid, *bbox, float(x), float(y), spd * 3.6))
            bpos = draw_ball(frame, cam, *clip.ball[i])
            if bpos is not None and gt_dir is not None:
                gt_rows.append((i, "ball", bpos[0] - 3, bpos[1] - 3, bpos[0] + 3, bpos[1] + 3, *clip.ball[i, :2], 0.0))
            prev = np.where(np.isnan(cur), prev, cur)
            # Broadcast "look": slight softness, sensor noise, vignette.
            frame = cv2.GaussianBlur(frame, (3, 3), 0.55)
            frame = cv2.multiply(frame, vignette, scale=1 / 255)
            frame = cv2.subtract(cv2.add(frame, noise_bank[i % len(noise_bank)]), (6, 6, 6, 0))
            assert ff.stdin is not None
            ff.stdin.write(frame.tobytes())
            if frames_dir is not None and every_png and i % every_png == 0:
                frames_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frames_dir / f"frame_{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        assert ff.stdin is not None
        ff.stdin.close()
        ff.wait()
        info = {"frames": F, "fps": fps, "width": self.width, "height": self.height,
                "match_t_start": float(clip.t[0]), "video": str(out_path)}
        if gt_dir is not None:
            gt_dir.mkdir(parents=True, exist_ok=True)
            CameraTrack(
                fps=fps, width=self.width, height=self.height, source="synthetic_ground_truth",
                homographies=[c.ground_homography().ravel().tolist() for c in cams],
                quality=[1.0] * F,
            ).model_dump_json()
            (gt_dir / "camera_gt.json").write_text(
                CameraTrack(fps=fps, width=self.width, height=self.height, source="synthetic_ground_truth",
                            homographies=[c.ground_homography().ravel().tolist() for c in cams],
                            quality=[1.0] * F).model_dump_json()
            )
            (gt_dir / "camera_params.json").write_text(json.dumps(
                [{"C": c.C.tolist(), "R": c.R.tolist(), "f": c.f} for c in cams]))
            pd.DataFrame(gt_rows, columns=["frame", "entity", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y", "speed_kmh"]
                         ).to_parquet(gt_dir / "gt_boxes.parquet", index=False)
            (gt_dir / "clip.json").write_text(json.dumps(info, indent=2))
        return info


def main() -> None:
    from pitchlens.synth.generate import SyntheticMatch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--match", type=Path, default=Path("data/synthetic/demo"))
    ap.add_argument("--start", type=float, required=True, help="Match clock (s) of the first frame")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", type=Path, default=Path("outputs/footage/clip"))
    ap.add_argument("--png-every", type=int, default=0)
    a = ap.parse_args()
    m = SyntheticMatch.load(a.match)
    clip = interpolate_clip(m.tracking, a.start, a.duration, a.fps)
    r = BroadcastRenderer(m.meta, a.width, a.height)
    cams = r.cameras(clip, a.fps)
    info = r.render(clip, cams, a.out / "footage.mp4", a.fps, gt_dir=a.out / "gt", frames_dir=a.out / "frames",
                    every_png=a.png_every)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
