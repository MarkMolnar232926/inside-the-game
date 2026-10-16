"""Leave-one-out check of registration on a real clip: register from some keyframes, compare at held-out ones.

python scripts/eval_real_registration.py VIDEO CALIB OUT_JSON --use 0,3 --exclude "95,50,655,112"
"""
import argparse
import json
import time
from pathlib import Path

from pitchlens.vision.registration import Calibration, PitchRegistrar, RegistrationConfig, reprojection_error

ap = argparse.ArgumentParser()
ap.add_argument("video", type=Path)
ap.add_argument("calib", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--use", default="", help="indices of keyframes to register from (rest are held out)")
ap.add_argument("--exclude", default="")
a = ap.parse_args()
kfs = Calibration.load(a.calib)
use = [int(i) for i in a.use.split(",")] if a.use else list(range(len(kfs)))
excl = tuple(tuple(int(v) for v in b.split(",")) for b in a.exclude.split(";") if b)
t = time.time()
cam = PitchRegistrar([kfs[i] for i in use], RegistrationConfig(exclude_boxes=excl)).run(a.video, progress=False)
held = {}
for i, k in enumerate(kfs):
    if i not in use:
        held[k.frame] = round(reprojection_error(cam.H(k.frame), k.homography(), cam.width, cam.height), 2)
rep = {"seconds": round(time.time() - t, 1), "used_keyframes": [kfs[i].frame for i in use],
       "held_out_error_px_vs_manual": held, "mean_quality": round(sum(cam.quality) / len(cam.quality), 3)}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.with_suffix(".camera.json").write_text(cam.model_dump_json())
a.out.write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
