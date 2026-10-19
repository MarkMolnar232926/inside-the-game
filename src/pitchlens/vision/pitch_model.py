"""Metric model of a 105 x 68 m pitch: lines, keypoints and sample points.

Used by the synthetic footage renderer (to draw lines), by the calibration
tool (named keypoints) and by registration refinement (line samples).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

L, W = 105.0, 68.0
LINE_W = 0.12
BOX_D, BOX_W = 16.5, 40.32
GA_D, GA_W = 5.5, 18.32
CIRCLE_R = 9.15
SPOT_D = 11.0
GOAL_W, GOAL_H = 7.32, 2.44


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


@lru_cache(maxsize=1)
def polylines() -> list[np.ndarray]:
    """All pitch markings as polylines in metres (centre-line of each painted line)."""
    cy = W / 2
    lines: list[np.ndarray] = [
        np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),
        np.array([[L / 2, 0], [L / 2, W]], float),
        _arc(L / 2, cy, CIRCLE_R, 0, 2 * math.pi, 96),
    ]
    for x0, s in ((0.0, 1), (L, -1)):
        lines.append(np.array([[x0, cy - BOX_W / 2], [x0 + s * BOX_D, cy - BOX_W / 2],
                               [x0 + s * BOX_D, cy + BOX_W / 2], [x0, cy + BOX_W / 2]], float))
        lines.append(np.array([[x0, cy - GA_W / 2], [x0 + s * GA_D, cy - GA_W / 2],
                               [x0 + s * GA_D, cy + GA_W / 2], [x0, cy + GA_W / 2]], float))
        # Penalty arc: the part of the spot-centred circle outside the box.
        spot = x0 + s * SPOT_D
        half = math.acos((BOX_D - SPOT_D) / CIRCLE_R)
        base = 0.0 if s == 1 else math.pi
        lines.append(_arc(spot, cy, CIRCLE_R, base - half, base + half, 40))
    for cx_, cy_, a0 in ((0, 0, 0), (L, 0, math.pi / 2), (L, W, math.pi), (0, W, -math.pi / 2)):
        lines.append(_arc(cx_, cy_, 1.0, a0, a0 + math.pi / 2, 10))
    return lines


@lru_cache(maxsize=1)
def spots() -> np.ndarray:
    return np.array([[L / 2, W / 2], [SPOT_D, W / 2], [L - SPOT_D, W / 2]], float)


@lru_cache(maxsize=1)
def line_quads() -> np.ndarray:
    """(N, 4, 2) ground-plane quads that tile every painted line with its real width."""
    quads = []
    h = LINE_W / 2
    for pl in polylines():
        for a, b in zip(pl[:-1], pl[1:], strict=True):
            d = b - a
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * h
            quads.append([a + n, b + n, b - n, a - n])
    return np.asarray(quads, float)


def sample_points(step: float = 0.5) -> np.ndarray:
    """Points along every line (for registration refinement)."""
    pts = []
    for pl in polylines():
        for a, b in zip(pl[:-1], pl[1:], strict=True):
            n = max(int(np.linalg.norm(b - a) / step), 1)
            u = np.linspace(0, 1, n, endpoint=False)
            pts.append(a + (b - a) * u[:, None])
    return np.vstack(pts)


KEYPOINTS: dict[str, tuple[float, float]] = {
    "corner_bl": (0, 0),
    "corner_tl": (0, W),
    "corner_br": (L, 0),
    "corner_tr": (L, W),
    "halfway_bottom": (L / 2, 0),
    "halfway_top": (L / 2, W),
    "centre_spot": (L / 2, W / 2),
    "circle_bottom": (L / 2, W / 2 - CIRCLE_R),
    "circle_top": (L / 2, W / 2 + CIRCLE_R),
    "left_box_bl": (0, W / 2 - BOX_W / 2),
    "left_box_br": (BOX_D, W / 2 - BOX_W / 2),
    "left_box_tr": (BOX_D, W / 2 + BOX_W / 2),
    "left_box_tl": (0, W / 2 + BOX_W / 2),
    "left_ga_br": (GA_D, W / 2 - GA_W / 2),
    "left_ga_tr": (GA_D, W / 2 + GA_W / 2),
    "left_spot": (SPOT_D, W / 2),
    "right_box_bl": (L - BOX_D, W / 2 - BOX_W / 2),
    "right_box_br": (L, W / 2 - BOX_W / 2),
    "right_box_tr": (L, W / 2 + BOX_W / 2),
    "right_box_tl": (L - BOX_D, W / 2 + BOX_W / 2),
    "right_ga_bl": (L - GA_D, W / 2 - GA_W / 2),
    "right_ga_tl": (L - GA_D, W / 2 + GA_W / 2),
    "right_spot": (L - SPOT_D, W / 2),
    "right_arc_bottom": (L - BOX_D, W / 2 - math.sqrt(CIRCLE_R**2 - (BOX_D - SPOT_D) ** 2)),
    "right_arc_top": (L - BOX_D, W / 2 + math.sqrt(CIRCLE_R**2 - (BOX_D - SPOT_D) ** 2)),
}
