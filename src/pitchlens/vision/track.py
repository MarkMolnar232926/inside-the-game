"""Multi-object tracking, team classification and identity reconciliation.

* :class:`ByteTrackLite` - ByteTrack-style two-stage association
  (high-confidence detections first, then low-confidence ones for lost
  tracks) with a constant-velocity predictor and Hungarian matching on a
  mixed IoU / foot-distance cost.
* :func:`classify_teams` - k-means on shirt colours (Lab), with outliers
  (goalkeepers, referee) labelled ``other``.
* :func:`reconcile_identities` - maps anonymous track ids to dataset player
  ids by Hungarian matching of pitch positions against the synthetic
  tracking stream (synthetic footage) or a user-provided anchor map (real
  clips).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

from pitchlens.schema import apply_h

from .detect import Detection


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    id: int
    box: np.ndarray
    pos: np.ndarray  # pitch metres (foot point)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # m/frame
    hits: int = 1
    age: int = 0
    lost: int = 0

    def predict(self) -> np.ndarray:
        return self.pos + self.vel


class ByteTrackLite:
    """Two-stage association in *pitch space* (camera motion removed via the homography)."""

    def __init__(self, high: float = 0.55, low: float = 0.2, max_lost: int = 45, min_hits: int = 3,
                 gate_m: float = 2.2):
        self.high, self.low = high, low
        self.max_lost, self.min_hits = max_lost, min_hits
        self.gate_m = gate_m
        self.tracks: list[Track] = []
        self.next_id = 1

    def _match(self, tracks: list[Track], pos: np.ndarray, gate: float):
        if not tracks or len(pos) == 0:
            return [], list(range(len(tracks))), list(range(len(pos)))
        pred = np.array([t.predict() for t in tracks])
        C = np.linalg.norm(pred[:, None] - pos[None], axis=2)
        # Lost tracks get a wider gate that grows with time since last seen.
        g = np.array([gate + 0.15 * t.lost for t in tracks])[:, None]
        Cg = np.where(g >= C, C, 1e6)
        r, c = linear_sum_assignment(Cg)
        matches, ut, ub = [], set(range(len(tracks))), set(range(len(pos)))
        for i, j in zip(r, c, strict=True):
            if Cg[i, j] < 1e5:
                matches.append((i, j))
                ut.discard(i)
                ub.discard(j)
        return matches, sorted(ut), sorted(ub)

    def update(self, dets: list[Detection], H: np.ndarray | None = None) -> list[tuple[int, np.ndarray, float]]:
        people = [d for d in dets if d.cls == "person"]
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in people]).reshape(-1, 4)
        conf = np.array([d.conf for d in people])
        foot = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]] if len(boxes) else np.zeros((0, 2))
        pos = apply_h(np.linalg.inv(H), foot) if H is not None and len(foot) else foot / 20.0
        hi, lo = np.where(conf >= self.high)[0], np.where((conf >= self.low) & (conf < self.high))[0]
        m1, ut, ub = self._match(self.tracks, pos[hi], self.gate_m)
        for i, j in m1:
            self._apply(self.tracks[i], boxes[hi[j]], pos[hi[j]])
        rest = [self.tracks[i] for i in ut]
        m2, ut2, _ = self._match(rest, pos[lo], self.gate_m * 0.8)
        for i, j in m2:
            self._apply(rest[i], boxes[lo[j]], pos[lo[j]])
        for i in ut2:
            t = rest[i]
            t.lost += 1
            t.pos = t.predict()
            t.vel *= 0.9
        for j in ub:
            self.tracks.append(Track(self.next_id, boxes[hi[j]].copy(), pos[hi[j]].copy()))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        out = []
        for t in self.tracks:
            t.age += 1
            if t.lost == 0 and t.hits >= self.min_hits:
                out.append((t.id, t.box.copy(), 1.0))
        return out

    @staticmethod
    def _apply(t: Track, box: np.ndarray, pos: np.ndarray) -> None:
        if t.lost == 0:
            t.vel = 0.7 * t.vel + 0.3 * (pos - t.pos)
        t.box = box.copy()
        t.pos = pos.copy()
        t.hits += 1
        t.lost = 0


# ------------------------------------------------------------------ teams
def shirt_color(frame: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    x1, y1, x2, y2 = box.astype(int)
    h = y2 - y1
    w = x2 - x1
    if h < 8 or w < 3:
        return None
    crop = frame[max(y1 + int(0.22 * h), 0) : max(y1 + int(0.5 * h), 1), max(x1 + int(0.2 * w), 0) : max(x2 - int(0.2 * w), 1)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    pix = crop.reshape(-1, 3)
    keep = ~((hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 50))  # drop grass
    if keep.sum() < 3:
        return None
    lab = cv2.cvtColor(pix[keep][None].astype(np.uint8), cv2.COLOR_BGR2LAB)[0].astype(float)
    return np.median(lab, axis=0)


def classify_teams(colors: dict[int, np.ndarray], outlier_factor: float = 2.6) -> tuple[dict[int, int], np.ndarray]:
    """Return {track_id: 0|1|-1} and the two team centroids (Lab). -1 = other (GK/referee)."""
    ids = list(colors)
    X = np.array([colors[i] for i in ids], np.float32)
    if len(X) < 2:
        return {i: 0 for i in ids}, X
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    _, labels, cent = cv2.kmeans(X, 2, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    d = np.linalg.norm(X - cent[labels], axis=1)
    spread = np.median(d) + 1e-6
    out = {}
    for i, lab, dd in zip(ids, labels, d, strict=True):
        out[i] = int(lab) if dd < outlier_factor * max(spread, 6.0) else -1
    return out, cent


# --------------------------------------------------------------- identity
def smooth_pitch(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Savitzky-Golay smoothing of per-track pitch positions + speed (km/h)."""
    parts = []
    for _, g in df.groupby("track_id"):
        g = g.sort_values("frame").copy()
        if len(g) >= 11:
            g["px_s"] = savgol_filter(g.pitch_x, 11, 2)
            g["py_s"] = savgol_filter(g.pitch_y, 11, 2)
            vx = savgol_filter(g.pitch_x, 11, 2, deriv=1) * fps
            vy = savgol_filter(g.pitch_y, 11, 2, deriv=1) * fps
            g["speed_kmh"] = np.hypot(vx, vy) * 3.6
        else:
            g["px_s"], g["py_s"], g["speed_kmh"] = g.pitch_x, g.pitch_y, 0.0
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df


def reconcile_identities(tracks: pd.DataFrame, reference: pd.DataFrame, max_dist_m: float = 3.0) -> dict[int, str]:
    """Majority-vote mapping track_id -> player id.

    ``reference`` has columns frame, entity, x, y (dataset positions resampled to video frames).
    """
    votes: dict[int, dict[str, int]] = {}
    ref_by_frame = {f: g for f, g in reference.groupby("frame")}
    for f, g in tracks.groupby("frame"):
        r = ref_by_frame.get(f)
        if r is None or len(r) == 0:
            continue
        A = g[["px_s", "py_s"]].to_numpy()
        B = r[["x", "y"]].to_numpy()
        C = np.linalg.norm(A[:, None] - B[None], axis=2)
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci, strict=True):
            if C[i, j] <= max_dist_m:
                tid = int(g.track_id.iloc[i])
                votes.setdefault(tid, {})
                ent = str(r.entity.iloc[j])
                votes[tid][ent] = votes[tid].get(ent, 0) + 1
    mapping: dict[int, str] = {}
    for tid, v in votes.items():
        ent, n = max(v.items(), key=lambda kv: kv[1])
        if n >= 5:
            mapping[tid] = ent
    return mapping
