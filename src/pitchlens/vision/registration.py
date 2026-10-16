"""Pitch registration: a per-frame pitch->image homography for any broadcast clip.

Pipeline (robust and pragmatic rather than research-grade):

1. **Manual keyframe calibration** - >=4 clicked image points matched to
   named pitch keypoints (``pitch_model.KEYPOINTS``), saved as JSON. One
   keyframe is enough; more keyframes reduce drift on long clips.
2. **Camera-motion propagation** - Lucas-Kanade optical flow on features that
   lie on the *ground plane* (inside the projected pitch area, away from
   players, stands and graphics); inter-frame homography by RANSAC.
3. **Line refinement (drift killer)** - white line pixels are detected; model
   line samples are projected with the current estimate, snapped to their
   nearest detected line pixel (distance-transform lookup), and the
   homography is re-solved (a few ICP iterations, Huber-like trimming).
4. **Temporal smoothing** - each homography is represented by where four
   fixed image anchors land on the pitch; those 8 numbers are smoothed with a
   zero-phase Gaussian, then turned back into homographies.

Every frame gets a quality score (fraction of projected line samples that
found a nearby detected line pixel).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.optimize import least_squares

from pitchlens.schema import CameraTrack, apply_h

from . import pitch_model as pm


@dataclass
class Calibration:
    frame: int
    pitch: np.ndarray  # (N, 2) metres
    image: np.ndarray  # (N, 2) pixels

    @classmethod
    def load(cls, path: Path) -> list[Calibration]:
        data = json.loads(Path(path).read_text())
        out = []
        for kf in data["keyframes"]:
            names = [p["name"] for p in kf["points"]]
            pitch = np.array([p.get("pitch") or pm.KEYPOINTS[n] for p, n in zip(kf["points"], names, strict=True)], float)
            image = np.array([p["px"] for p in kf["points"]], float)
            out.append(cls(int(kf["frame"]), pitch, image))
        return out

    def homography(self) -> np.ndarray:
        H, _ = cv2.findHomography(self.pitch, self.image, 0)
        return H / H[2, 2]


@dataclass
class RegistrationConfig:
    work_scale: float = 0.5  # process at half resolution, report at full
    max_features: int = 600
    refine_every: int = 1
    refine_iters: int = 3
    snap_radius_px: float = 9.0  # at work scale
    smooth_sigma_frames: float = 2.0
    exclude_top_px: int = 0  # e.g. burned-in score bug rows to ignore
    exclude_boxes: tuple[tuple[int, int, int, int], ...] = ()  # (x0, y0, x1, y1) full-res graphics to ignore


def green_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (30, 50, 35), (90, 255, 255))


def line_mask(bgr: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """White-ish painted line pixels on the grass (thin, bright, low saturation)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    # Keep thin structures: top-hat on the value channel suppresses big bright blobs (white kits).
    v = hsv[..., 2]
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    thin = cv2.threshold(tophat, 28, 255, cv2.THRESH_BINARY)[1]
    m = cv2.bitwise_and(bright, thin)
    # Must be near grass (lines are painted on it)
    near_grass = cv2.dilate(green_mask(bgr), np.ones((5, 5), np.uint8))
    return cv2.bitwise_and(cv2.bitwise_and(m, near_grass), ground)


def ground_region(H: np.ndarray, shape: tuple[int, int], margin: float = 2.5) -> np.ndarray:
    poly = np.array([[-margin, -margin], [pm.L + margin, -margin], [pm.L + margin, pm.W + margin], [-margin, pm.W + margin]])
    # Densify edges so the polygon follows perspective correctly when partly off-screen.
    dense = np.vstack([np.linspace(poly[i], poly[(i + 1) % 4], 40) for i in range(4)])
    img = apply_h(H, dense)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(img).astype(np.int32)], 255)
    return mask


def player_mask(bgr: np.ndarray, ground: np.ndarray) -> np.ndarray:
    """Non-grass, non-line blobs on the ground region (players, ball, referee) - excluded from flow."""
    g = green_mask(bgr)
    non = cv2.bitwise_and(cv2.bitwise_not(g), ground)
    non = cv2.morphologyEx(non, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(non, np.ones((7, 7), np.uint8))


class PitchRegistrar:
    def __init__(self, keyframes: list[Calibration], cfg: RegistrationConfig | None = None):
        assert keyframes, "at least one calibrated keyframe is required"
        self.keyframes = sorted(keyframes, key=lambda k: k.frame)
        self.cfg = cfg or RegistrationConfig()
        s = self.cfg.work_scale
        self.S = np.diag([s, s, 1.0])  # full-res -> work-res
        self.Si = np.linalg.inv(self.S)
        self.samples = pm.sample_points(0.6)

    # -------------------------------------------------------------- core
    def _prep(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.cfg.work_scale
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return small, gray

    def _static_mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.full(shape, 255, np.uint8)
        s = self.cfg.work_scale
        if self.cfg.exclude_top_px:
            m[: int(self.cfg.exclude_top_px * s)] = 0
        for x0, y0, x1, y1 in self.cfg.exclude_boxes:
            m[int(y0 * s) : int(y1 * s), int(x0 * s) : int(x1 * s)] = 0
        return m

    def refine(self, small: np.ndarray, Hw: np.ndarray, static: np.ndarray) -> tuple[np.ndarray, float]:
        """Chamfer alignment: move 4 image anchors so projected model lines sit on detected line pixels.

        Residuals are the (truncated) distance-transform values at the projected line samples,
        minimised with a robust least-squares solver over the 8 anchor coordinates.
        """
        shape = small.shape[:2]
        h, w = shape
        ground = cv2.bitwise_and(ground_region(Hw, shape), static)
        lines = line_mask(small, ground)
        lines = cv2.bitwise_and(lines, cv2.bitwise_not(player_mask(small, ground)))
        if cv2.countNonZero(lines) < 60:
            return Hw, 0.0
        r = self.cfg.snap_radius_px
        dist = cv2.distanceTransform(cv2.bitwise_not(lines), cv2.DIST_L2, 5)
        dist = np.minimum(dist, r).astype(np.float32)
        proj = apply_h(Hw, self.samples)
        ok = (proj[:, 0] >= 2) & (proj[:, 0] < w - 2) & (proj[:, 1] >= 2) & (proj[:, 1] < h - 2)
        if ok.sum() < 40:
            return Hw, 0.0
        ix, iy = proj[ok, 0].astype(int), proj[ok, 1].astype(int)
        vis = ground[iy, ix] > 0
        M = self.samples[ok][vis]
        if len(M) < 40:
            return Hw, 0.0
        anchors_img = np.array([[0.2 * w, 0.5 * h], [0.8 * w, 0.5 * h], [0.8 * w, 0.95 * h], [0.2 * w, 0.95 * h]])
        anchors_pitch = apply_h(np.linalg.inv(Hw), anchors_img).astype(np.float32)

        def H_of(delta: np.ndarray) -> np.ndarray:
            dst = (anchors_img + delta.reshape(4, 2)).astype(np.float32)
            return cv2.getPerspectiveTransform(anchors_pitch, dst)

        def residuals(delta: np.ndarray) -> np.ndarray:
            P = apply_h(H_of(delta), M)
            return map_coordinates(dist, [P[:, 1], P[:, 0]], order=1, mode="nearest")

        r0 = residuals(np.zeros(8))
        sol = least_squares(residuals, np.zeros(8), loss="soft_l1", f_scale=1.5, diff_step=1e-3,
                            x_scale=2.0, max_nfev=60)
        Hn = H_of(sol.x)
        r1 = residuals(sol.x)
        before, after = float(np.mean(np.exp(-(r0**2) / 2))), float(np.mean(np.exp(-(r1**2) / 2)))
        quality = float(np.mean(r1 < 1.5))
        if after <= before or np.abs(sol.x).max() > 4 * r:
            return Hw, float(np.mean(r0 < 1.5))
        return Hn / Hn[2, 2], quality

    def _fit_score(self, Hw: np.ndarray, dist: np.ndarray, ground: np.ndarray) -> float:
        h, w = dist.shape
        proj = apply_h(Hw, self.samples)
        ok = (proj[:, 0] >= 1) & (proj[:, 0] < w - 1) & (proj[:, 1] >= 1) & (proj[:, 1] < h - 1)
        if ok.sum() < 30:
            return 0.0
        ix, iy = proj[ok, 0].astype(int), proj[ok, 1].astype(int)
        vis = ground[iy, ix] > 0
        d = dist[iy, ix][vis]
        return float(np.mean(np.exp(-(d**2) / 2.0))) if len(d) else 0.0

    def run(self, video_path: Path, n_frames: int | None = None, progress: bool = False) -> CameraTrack:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames:
            total = min(total, n_frames)
        frames_small, grays = [], []
        for _ in range(total):
            ok, fr = cap.read()
            if not ok:
                break
            sm, g = self._prep(fr)
            frames_small.append(sm)
            grays.append(g)
        cap.release()
        total = len(grays)
        static = self._static_mask(grays[0].shape)
        Hs: list[np.ndarray | None] = [None] * total
        Q = np.zeros(total)
        # Propagate forwards and backwards from every keyframe; keep the estimate closest to a keyframe.
        dist_to_kf = np.full(total, np.inf)
        for kf in self.keyframes:
            if kf.frame >= total:
                continue
            H0 = self.S @ kf.homography()
            for direction in (1, -1):
                Hw = H0.copy()
                rng = range(kf.frame, total) if direction == 1 else range(kf.frame, -1, -1)
                prev = None
                for i in rng:
                    if prev is not None:
                        G = self.flow_h(grays[prev], grays[i], Hw, frames_small[prev], static)
                        if G is not None:
                            Hw = G @ Hw
                    if (i - kf.frame) % self.cfg.refine_every == 0:
                        Hw, q = self.refine(frames_small[i], Hw, static)
                    else:
                        q = Q[i]
                    d = abs(i - kf.frame)
                    if d < dist_to_kf[i]:
                        dist_to_kf[i] = d
                        Hs[i] = Hw.copy()
                        Q[i] = q
                    prev = i
                    if progress and i % 100 == 0:
                        print(f"  registration frame {i}/{total} q={q:.2f}", flush=True)
        H_full = [self.Si @ h for h in Hs]  # type: ignore[operator]
        H_full = smooth_homographies(H_full, W, Hh, self.cfg.smooth_sigma_frames)
        return CameraTrack(fps=fps, width=W, height=Hh, source="estimated",
                           homographies=[(h / h[2, 2]).ravel().tolist() for h in H_full], quality=Q.round(3).tolist())

    def flow_h(self, g0: np.ndarray, g1: np.ndarray, Hw: np.ndarray, small0: np.ndarray, static: np.ndarray):
        ground = cv2.bitwise_and(ground_region(Hw, g0.shape, margin=1.0), static)
        mask = cv2.bitwise_and(ground, cv2.bitwise_not(player_mask(small0, ground)))
        p0 = cv2.goodFeaturesToTrack(g0, self.cfg.max_features, 0.003, 7, mask=mask, blockSize=7)
        if p0 is None or len(p0) < 20:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
        st = st.ravel() == 1
        if st.sum() < 15:
            return None
        G, inl = cv2.findHomography(p0[st], p1[st], cv2.RANSAC, 1.0)
        if G is None or inl is None or inl.sum() < 12:
            return None
        return G


def blend_h(Ha: np.ndarray, Hb: np.ndarray, w: float) -> np.ndarray:
    """Blend two homographies by interpolating where they send a fixed set of pitch points."""
    pts = np.array([[20, 10], [85, 10], [85, 58], [20, 58]], float)
    a, b = apply_h(Ha, pts), apply_h(Hb, pts)
    H, _ = cv2.findHomography(pts, a * (1 - w) + b * w, 0)
    return H / H[2, 2]


def smooth_homographies(Hs: list[np.ndarray], W: int, H: int, sigma: float) -> list[np.ndarray]:
    """Zero-phase temporal smoothing via the pitch locations of 4 fixed image anchors."""
    if sigma <= 0 or len(Hs) < 3:
        return Hs
    anchors = np.array([[0.2 * W, 0.45 * H], [0.8 * W, 0.45 * H], [0.85 * W, 0.95 * H], [0.15 * W, 0.95 * H]])
    P = np.stack([apply_h(np.linalg.inv(h), anchors) for h in Hs])  # (F, 4, 2) pitch metres
    Ps = gaussian_filter1d(P, sigma, axis=0, mode="nearest")
    out = []
    for k in range(len(Hs)):
        Hk, _ = cv2.findHomography(Ps[k], anchors, 0)
        out.append(Hk / Hk[2, 2])
    return out


def reprojection_error(H_est: np.ndarray, H_gt: np.ndarray, W: int, Himg: int) -> float:
    """Mean pixel distance between estimated and true projections of on-screen pitch line samples."""
    pts = pm.sample_points(1.0)
    gt = apply_h(H_gt, pts)
    ok = (gt[:, 0] >= 0) & (gt[:, 0] < W) & (gt[:, 1] >= 0) & (gt[:, 1] < Himg)
    if ok.sum() == 0:
        return float("nan")
    est = apply_h(H_est, pts[ok])
    return float(np.mean(np.linalg.norm(est - gt[ok], axis=1)))
