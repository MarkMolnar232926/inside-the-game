# Running PitchLens on your own clip (local only)

The engine, overlay JSON, homography code and renderers are **source-agnostic**. The public demo uses rights-clear synthetic footage. The same commands work on any tactical-camera clip you have the rights to use locally.

> **Compliance:** broadcast footage contains third-party trademarks, sponsor boards, player likenesses and broadcaster graphics. Keep such clips in `data/test_assets/` (git-ignored). Never commit them or use them in the public demo video.

```bash
# 1. Put the clip somewhere ignored by git
cp ~/Downloads/vizuals_test.mp4 data/test_assets/

# 2. Export a frame where >= 4 pitch keypoints are visible (penalty box corners, box arc ends, etc.)
python -m pitchlens.vision.calibrate export --video data/test_assets/vizuals_test.mp4 --frame 0 --out data/cache/frame0.png

# 3. Click the keypoints in any image viewer and write data/calibration/vizuals_test.json:
#    {"keyframes": [{"frame": 0, "points": [
#        {"name": "right_box_tl", "px": [1012, 388]}, {"name": "right_box_bl", "px": [1490, 812]},
#        {"name": "right_ga_tl", "px": [1385, 412]}, {"name": "right_arc_top", "px": [905, 520]}]}]}
#    Names: see pitch_model.KEYPOINTS. Any point can also be given as {"pitch": [x, y], "px": [u, v]}.
#    Add more keyframes (frame N) if the camera swings far; propagation runs from the nearest one.
python -m pitchlens.vision.calibrate check --video data/test_assets/vizuals_test.mp4 \
    --calib data/calibration/vizuals_test.json --out data/cache/calib_check.png   # red lines should sit on the paint

# 4. Mask burned-in broadcaster graphics so they never feed the camera solver or detector
#    (--exclude "x0,y0,x1,y1;...", e.g. the score bug at top-left and any side panel)

# 5. Run the vision pipeline + agents + renderers. --anchors binds a tracked person to our fictional player:
echo '{"7": "KBY-07"}' > data/cache/anchors.json     # track ids are listed in <out>/clip/vision/debug.mp4
python scripts/build_demo.py --video data/test_assets/vizuals_test.mp4 --calib data/calibration/vizuals_test.json \
    --anchors data/cache/anchors.json --exclude "100,40,620,130" --start 3721 --out outputs/local_real
```

What changes on a real clip:

* **Identity**: there is no dataset tracking to reconcile against, so identities come from the anchor map. Unanchored players are still detected, tracked and team-classified (jersey-colour k-means), but they are not named.
* **Events**: the synthetic events stay synthetic (`source=synthetic`). The video is only a visual substrate. Auto-eventing adds `vision_estimated` passes and turnovers with confidence scores.
* **Detector**: the built-in `ColorBlobDetector` assumes a green pitch and kits that contrast with it. For harder footage, install the optional extra (`pip install pitchlens[vision]`) and swap in `UltralyticsDetector` (YOLO/RT-DETR + the same ByteTrack-lite tracker).
