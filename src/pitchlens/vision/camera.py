"""Pinhole broadcast camera and a virtual camera operator.

World frame: x along the pitch (0..105), y across (0..68), z up (metres).
The main camera sits high in the near-side stand, roughly level with the
halfway line, and pans/tilts/zooms to follow play, like a real tactical feed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    C: np.ndarray  # camera centre (3,)
    R: np.ndarray  # world->camera rotation (rows: right, down, forward)
    f: float
    width: int = 1920
    height: int = 1080

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1.0]])

    @classmethod
    def look_at(cls, C, target, f: float, width: int = 1920, height: int = 1080) -> PinholeCamera:
        C = np.asarray(C, float)
        fwd = np.asarray(target, float) - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        return cls(C, np.vstack([right, down, fwd]), f, width, height)

    def ground_homography(self) -> np.ndarray:
        """3x3 homography mapping pitch (x, y, z=0) metres -> image pixels."""
        t = -self.R @ self.C
        H = self.K @ np.c_[self.R[:, 0], self.R[:, 1], t]
        return H / H[2, 2]

    def plane_homography(self, origin, e1, e2) -> np.ndarray:
        """Homography for an arbitrary world plane O + u*e1 + v*e2 -> image."""
        M = np.c_[self.R @ np.asarray(e1, float), self.R @ np.asarray(e2, float),
                  self.R @ (np.asarray(origin, float) - self.C)]
        return self.K @ M

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N, 3) world points -> (N, 2) pixels and camera depth (N,)."""
        X = np.asarray(X, float).reshape(-1, 3)
        Xc = (X - self.C) @ self.R.T
        z = Xc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        uv = np.c_[self.f * Xc[:, 0] / zs + self.width / 2, self.f * Xc[:, 1] / zs + self.height / 2]
        return uv, z


@dataclass
class OperatorConfig:
    position: tuple[float, float, float] = (52.5, -30.0, 21.0)
    f_base: float = 2100.0  # ~62 m visible across the centre at 1080p
    f_tight: float = 2500.0
    stiffness: float = 1.6  # spring natural frequency (rad/s)
    lead_s: float = 0.5
    width: int = 1920
    height: int = 1080


class CameraOperator:
    """Critically-damped spring that follows the ball like a human operator."""

    def __init__(self, cfg: OperatorConfig | None = None):
        self.cfg = cfg or OperatorConfig()

    def path(self, ball_xy: np.ndarray, fps: float, ball_v: np.ndarray | None = None) -> list[PinholeCamera]:
        cfg = self.cfg
        n = len(ball_xy)
        if ball_v is None:
            ball_v = np.gradient(ball_xy, axis=0) * fps
        aim = ball_xy + ball_v * cfg.lead_s
        aim[:, 0] = np.clip(aim[:, 0], 22, 83)
        aim[:, 1] = np.clip(34 + (aim[:, 1] - 34) * 0.45, 24, 44)
        dt = 1 / fps
        w = cfg.stiffness
        pos = aim[0].copy()
        vel = np.zeros(2)
        zoom = 0.0
        cams = []
        for i in range(n):
            acc = w * w * (aim[i] - pos) - 2 * w * vel
            vel += acc * dt
            pos += vel * dt
            spd = float(np.linalg.norm(ball_v[i]))
            near_goal = max(0.0, (abs(pos[0] - 52.5) - 20) / 12)
            zoom_target = float(np.clip(0.55 * near_goal - 0.04 * max(spd - 6, 0), 0, 1))
            zoom += (zoom_target - zoom) * dt * 0.6
            f = cfg.f_base + (cfg.f_tight - cfg.f_base) * zoom
            cams.append(PinholeCamera.look_at(cfg.position, [pos[0], pos[1], 0.0], f, cfg.width, cfg.height))
        return cams
