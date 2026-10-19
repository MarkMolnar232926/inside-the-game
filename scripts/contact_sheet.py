"""Contact sheet of sampled frames + camera-motion / pitch-line visibility notes for any clip.

python scripts/contact_sheet.py --video outputs/footage/demo30/footage.mp4 --out outputs/deliverables/01_contact_sheet.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pitchlens.vision.registration import green_mask, line_mask

ap = argparse.ArgumentParser()
ap.add_argument("--video", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--cols", type=int, default=4)
a = ap.parse_args()
cap = cv2.VideoCapture(str(a.video))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))
idx = np.linspace(0, N - 1, a.n).astype(int)
tiles, notes = [], []
prev_gray = None
# global camera motion: median optical-flow translation between consecutive frames (sampled every 5 frames)
pan = []
for i in range(0, N, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(cv2.resize(fr, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        p0 = cv2.goodFeaturesToTrack(prev_gray, 300, 0.01, 6)
        if p0 is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, p0, None)
            d = (p1 - p0)[st.ravel() == 1].reshape(-1, 2)
            pan.append((i / fps, float(np.median(d[:, 0])) * 4 * fps / 5, float(np.median(d[:, 1])) * 4 * fps / 5))
    prev_gray = g
for k, i in enumerate(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, fr = cap.read()
    if not ok:
        continue
    grass = green_mask(fr).mean() / 255
    full = np.full(fr.shape[:2], 255, np.uint8)
    lines = line_mask(fr, full).mean() / 255 * 100
    t = i / fps
    notes.append({"frame": int(i), "t": round(t, 2), "grass_pct": round(grass * 100, 1), "line_px_pct": round(lines, 2)})
    tile = cv2.resize(fr, (480, 270))
    cv2.rectangle(tile, (0, 0), (480, 30), (20, 16, 40), -1)
    cv2.putText(tile, f"#{i}  t={t:5.2f}s  grass {grass*100:4.1f}%  lines {lines:4.2f}%", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
rows = [np.hstack(tiles[r * a.cols:(r + 1) * a.cols]) for r in range(len(tiles) // a.cols)]
sheet = np.vstack(rows)
a.out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
pan = np.array(pan)
summary = {
    "video": str(a.video), "frames": N, "fps": fps, "size": [W, H],
    "pan_px_per_s": {"mean_abs_x": round(float(np.abs(pan[:, 1]).mean()), 1), "max_abs_x": round(float(np.abs(pan[:, 1]).max()), 1),
                     "mean_abs_y": round(float(np.abs(pan[:, 2]).mean()), 1)},
    "pan_track": [[round(t, 2), round(x, 1), round(y, 1)] for t, x, y in pan],
    "samples": notes,
}
a.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "pan_track"}, indent=1))
