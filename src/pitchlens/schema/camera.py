"""Camera and screen-track schemas consumed by renderers.

``CameraTrack`` stores one pitch->image homography per video frame. Pitch
points are (x, y) metres on the ground plane; image points are pixels.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


class CameraTrack(BaseModel):
    fps: float
    width: int
    height: int
    source: Literal["synthetic_ground_truth", "estimated", "manual"] = "estimated"
    homographies: list[list[float]] = Field(description="Per frame, row-major 3x3 pitch->image (9 floats)")
    quality: list[float] = Field(default_factory=list, description="Per-frame registration confidence 0..1")

    def H(self, frame: int) -> np.ndarray:
        frame = int(np.clip(frame, 0, len(self.homographies) - 1))
        return np.asarray(self.homographies[frame], dtype=np.float64).reshape(3, 3)

    def project(self, frame: int, pts: np.ndarray) -> np.ndarray:
        """Project (N, 2) pitch metres -> (N, 2) pixels for a frame."""
        return apply_h(self.H(frame), pts)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.c_[pts, np.ones(len(pts))] @ H.T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    return hom[:, :2] / w


class ScreenTrack(BaseModel):
    """Image-space track of one identified player (from vision + identity reconciliation)."""

    player_id: str
    track_id: int
    frames: list[int]
    boxes: list[tuple[float, float, float, float]] = Field(description="x1, y1, x2, y2 pixels")
    pitch_xy: list[tuple[float, float]] = Field(default_factory=list)
    speed_kmh: list[float] = Field(default_factory=list)
    source: Literal["synthetic", "vision_estimated"] = "vision_estimated"


class ScreenTracks(BaseModel):
    fps: float
    tracks: dict[str, ScreenTrack]
