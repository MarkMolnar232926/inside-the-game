"""Export a frame with a labelled pixel grid (helps reading calibration clicks by eye).

python scripts/grid_frame.py VIDEO FRAME OUT [x0 y0 x1 y1]   # optional crop, grid labels stay in full-frame pixels
"""
import sys

import cv2

video, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
_, fr = cap.read()
x0, y0, x1, y1 = (map(int, sys.argv[4:8]) if len(sys.argv) > 7 else (0, 0, fr.shape[1], fr.shape[0]))
step = 100 if (x1 - x0) > 900 else 50
for x in range((x0 // step) * step, x1, step):
    cv2.line(fr, (x, y0), (x, y1), (0, 255, 255) if x % 500 == 0 else (0, 200, 255), 1)
    cv2.putText(fr, str(x), (x + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
for y in range((y0 // step) * step, y1, step):
    cv2.line(fr, (x0, y), (x1, y), (255, 0, 255) if y % 500 == 0 else (255, 120, 255), 1)
    cv2.putText(fr, str(y), (x0 + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
cv2.imwrite(out, fr[y0:y1, x0:x1])
