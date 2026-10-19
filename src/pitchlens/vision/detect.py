"""Player and ball detection.

``ColorBlobDetector`` is a dependency-free CPU detector: on a registered
frame it segments non-grass blobs on the ground plane, rejects painted lines
with a morphological opening, and splits merged blobs by the expected player
size at that depth (from the homography). It runs at well above real time on
CPU and is good enough for tactical-camera footage.

``UltralyticsDetector`` (optional; ``pip install pitchlens[vision]``) wraps a
YOLO/RT-DETR model behind the same interface for harder real-world footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from pitchlens.schema import apply_h

from .registration import ground_region


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: str  # person | ball

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, H: np.ndarray) -> list[Detection]: ...


def local_scale(H: np.ndarray, img_xy: tuple[float, float]) -> float:
    """Pixels per metre on the ground around an image point."""
    Hi = np.linalg.inv(H)
    p = apply_h(Hi, np.array([img_xy]))[0]
    a = apply_h(H, np.array([p, p + [1.0, 0.0]]))
    return float(np.linalg.norm(a[1] - a[0]))


class ColorBlobDetector:
    def __init__(self, work_scale: float = 0.5, exclude_boxes: tuple[tuple[int, int, int, int], ...] = ()):
        self.s = work_scale
        self.exclude_boxes = exclude_boxes

    def detect(self, frame: np.ndarray, H: np.ndarray) -> list[Detection]:
        s = self.s
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        S = np.diag([s, s, 1.0])
        Hs = S @ H
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (30, 50, 35), (90, 255, 255))
        ground = ground_region(Hs, small.shape[:2], margin=1.5)
        for x0, y0, x1, y1 in self.exclude_boxes:
            ground[int(y0 * s) : int(y1 * s), int(x0 * s) : int(x1 * s)] = 0
        non = cv2.bitwise_and(cv2.bitwise_not(grass), ground)
        # Players are drawn partly above the ground polygon edge: extend the mask upwards a little.
        non_up = cv2.bitwise_and(cv2.bitwise_not(grass), cv2.dilate(ground, np.ones((25, 3), np.uint8), anchor=(1, 24)))
        non = cv2.bitwise_or(non, non_up)
        # Remove thin painted lines and noise, keep body-sized blobs.
        body = cv2.morphologyEx(non, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)))
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(body, connectivity=8)
        dets: list[Detection] = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if area < 12:
                continue
            foot = (x + w / 2, y + h)
            sc = local_scale(Hs, foot)
            exp_h, exp_w = 1.85 * sc, 0.62 * sc
            if h < 0.45 * exp_h or w > 6 * exp_w or h > 3.2 * exp_h:
                continue
            ncols = max(1, int(round(w / (exp_w * 1.25)))) if w > 1.7 * exp_w else 1
            nrows = max(1, int(round(h / (exp_h * 1.05)))) if h > 1.6 * exp_h else 1
            for r in range(nrows):
                for c in range(ncols):
                    bx0 = x + c * w / ncols
                    by0 = y + r * h / nrows
                    sub = lab[int(by0) : int(by0 + h / nrows), int(bx0) : int(bx0 + w / ncols)] == k
                    if sub.sum() < 8:
                        continue
                    ys, xs = np.nonzero(sub)
                    bb = (bx0 + xs.min(), by0 + ys.min(), bx0 + xs.max() + 1, by0 + ys.max() + 1)
                    fill = sub.sum() / max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1)
                    ratio = (bb[3] - bb[1]) / max(exp_h, 1)
                    conf = float(np.clip(0.35 + 0.4 * min(ratio, 1.2) / 1.2 + 0.25 * fill, 0, 1))
                    if ncols * nrows > 1:
                        conf *= 0.8
                    dets.append(Detection(bb[0] / s, bb[1] / s, bb[2] / s, bb[3] / s, conf, "person"))
        dets += self._ball(frame, H, dets)
        return dets

    def _ball(self, frame: np.ndarray, H: np.ndarray, people: list[Detection]) -> list[Detection]:
        """Small, round, bright blob on grass that is not a painted line (model lines masked via H)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 185), (180, 70, 255))
        ground = ground_region(H, frame.shape[:2], margin=1.0)
        white = cv2.bitwise_and(white, ground)
        white = cv2.bitwise_and(white, cv2.bitwise_not(model_line_mask(H, frame.shape[:2])))
        white = cv2.bitwise_and(white, cv2.bitwise_not(goal_mask(H, frame.shape[:2])))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(white, connectivity=8)
        grass = cv2.inRange(hsv, (30, 50, 35), (90, 255, 255))
        out = []
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if not (5 <= area <= 160) or w > 16 or h > 16 or max(w, h) > 2.2 * min(w, h):
                continue
            cx, cy = cent[k]
            if any(p.x1 - 4 <= cx <= p.x2 + 4 and p.y1 - 4 <= cy <= p.y2 + 4 for p in people):
                continue
            r = int(max(w, h)) + 4
            ring = grass[max(int(cy) - r, 0) : int(cy) + r + 1, max(int(cx) - r, 0) : int(cx) + r + 1]
            if ring.size == 0 or ring.mean() < 255 * 0.5:
                continue
            fill = area / float(w * h)
            out.append(Detection(x - 1, y - 1, x + w + 1, y + h + 1, float(np.clip(0.3 + 0.6 * fill, 0, 1)), "ball"))
        return out


def goal_mask(H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Image region covered by both goals (footprint lifted by the goal height at the local scale)."""
    from . import pitch_model as pm

    m = np.zeros(shape, np.uint8)
    for gx, s in ((0.0, -1), (pm.L, 1)):
        foot = np.array([[gx - s * 0.8, pm.W / 2 - 5], [gx + s * 3.0, pm.W / 2 - 5], [gx + s * 3.0, pm.W / 2 + 5],
                         [gx - s * 0.8, pm.W / 2 + 5]])
        img = apply_h(H, foot)
        if not np.isfinite(img).all() or np.abs(img).max() > 1e5:
            continue
        lift = 2.9 * local_scale(H, tuple(img.mean(axis=0)))
        hull = cv2.convexHull(np.round(np.vstack([img, img - [0, lift]])).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 255)
    return m


def model_line_mask(H: np.ndarray, shape: tuple[int, int], width_px: int = 7) -> np.ndarray:
    """Pixels covered by the projected pitch markings (and penalty spots)."""
    from . import pitch_model as pm

    m = np.zeros(shape, np.uint8)
    for pl in pm.polylines():
        dense = np.vstack([np.linspace(a, b, 16) for a, b in zip(pl[:-1], pl[1:], strict=True)])
        cv2.polylines(m, [np.round(apply_h(H, dense)).astype(np.int32)], False, 255, width_px)
    for sp in pm.spots():
        c = apply_h(H, sp[None])[0]
        cv2.circle(m, (int(c[0]), int(c[1])), width_px, 255, -1)
    return m


class UltralyticsDetector:
    """Optional deep detector. Requires ``ultralytics`` and a local weights file."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25):
        from ultralytics import YOLO  # type: ignore[import-not-found]

        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame: np.ndarray, H: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, conf=self.conf, verbose=False, classes=[0, 32])[0]
        out = []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                           strict=True):
            out.append(Detection(*map(float, b), float(c), "ball" if int(k) == 32 else "person"))
        return out
