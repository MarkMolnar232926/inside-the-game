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
    --anchors data/cache/anchors.json --exclude "95,50,655,112" --start 3719.8 --orient-team KBY --no-score-bug \
    --out outputs/real_demo
```

What changes on a real clip:

* **Identity**: there is no dataset tracking to reconcile against, so identities come from the anchor map. Unanchored players are still detected, tracked and team-classified (jersey-colour k-means), but they are not named.
* **Events**: the synthetic events stay synthetic (`source=synthetic`). The video is only a visual substrate. Auto-eventing adds `vision_estimated` passes and turnovers with confidence scores.
* **Detector**: the built-in `ColorBlobDetector` assumes a green pitch and kits that contrast with it. For harder footage, install the optional extra (`pip install pitchlens[vision]`) and swap in `UltralyticsDetector` (YOLO/RT-DETR + the same ByteTrack-lite tracker).

## Alignment flags (synthetic story on real footage)

| Flag | Effect |
|---|---|
| `--start <match clock>` | media t = 0 maps to this match time. Pick it so the key synthetic moment falls where the camera shows the right area (we used 3719.8 so #7's 3rd high interception lands at 11.5 s, when the right penalty area is on screen) |
| `--orient-team KBY` | evidence is drawn with this team attacking towards x = 105, i.e. the end the real camera looks at |
| `--anchors` | binds a real track to a fictional player: the highlight follows that person, and their in-clip events are *retargeted* to the person's tracked position (label `retargeted`) |
| `--no-score-bug` | the clip already has a broadcaster score bug; ours is dropped |
| `--exclude` | masks burned-in graphics for the camera solver and detector |

## Results on the provided development clip (local only, never committed)

The clip: 1920x1080, 30 fps, 914 frames. It is a tactical camera with strong zoom changes (tight on the centre circle, then the right penalty area, then a wide zoom-out and a pan back to midfield) and pans up to ~700 px/s.

* **Calibration**: 5 manual keyframes (frames 0, 415, 664, 765, 900; 5-8 points each, read off `scripts/grid_frame.py` overlays). Click residuals 1-6 px.
* **Registration, leave-one-out** (register without a keyframe, compare with its manual calibration, `scripts/eval_real_registration.py`): frame 415 **5.3 px**, frame 664 **3.0 px**, frame 765 **67 px**. The last one needs its own keyframe: in the wide zoom-out the paint is thin and the flow underestimates the zoom. With all 5 keyframes, the projected lines sit on the paint in every sampled frame (`05_vision_debug_real.mp4`).
* **Detection/tracking**: ~20 detections per frame. Team colours separate cleanly (Lab L 183 vs 86). Some white-kit players are missed against bright lines. Tracks fragment more than on synthetic footage (157 tracks), but the anchored player's track spans 839/914 frames.
* **What changed in the code to get there**: the line mask no longer erases thin paint with the player mask; frame-to-frame motion uses the whole image (a pan/tilt/zoom camera gives one homography for all static scene points); and each frame keeps the propagation candidate whose lines best match the paint, not merely the nearest keyframe's. The same changes improved synthetic registration from 2.0 to 0.9 px mean.
* **Not yet good**: vision speeds on the real clip are noisy (bbox jitter at far distances), so speed chips use the synthetic dataset speeds of the fictional player. Auto-eventing has no ground truth to score against here.
