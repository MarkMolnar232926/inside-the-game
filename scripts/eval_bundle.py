"""Evaluate a synthetic-footage demo bundle against its ground truth and merge results into metrics.json.

python scripts/eval_bundle.py [outputs/demo]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from pitchlens.schema import CameraTrack
from pitchlens.synth.generate import SyntheticMatch
from pitchlens.vision.auto_events import reconcile
from pitchlens.vision.registration import reprojection_error

demo = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/demo")
clip, vis = demo / "clip", demo / "clip" / "vision"
man = json.loads((demo / "manifest.json").read_text())
est = CameraTrack.model_validate_json((vis / "homography.json").read_text())
gt = CameraTrack.model_validate_json((clip / "gt/camera_gt.json").read_text())
err = np.array([reprojection_error(est.H(i), gt.H(i), gt.width, gt.height) for i in range(len(est.homographies))])
reg = {"frames": len(err), "mean_px": round(float(np.nanmean(err)), 2), "p95_px": round(float(np.nanpercentile(err, 95)), 2),
       "max_px": round(float(np.nanmax(err)), 2), "mean_quality": round(float(np.mean(est.quality)), 3)}
(vis / "registration_eval.json").write_text(json.dumps(reg, indent=2))
out = subprocess.run([sys.executable, "scripts/eval_vision.py", str(clip)], capture_output=True, text=True, check=True).stdout
visev = json.loads(out)
m = SyntheticMatch.load(Path("data/synthetic/demo"))
rec = reconcile(json.loads((vis / "auto_events.json").read_text()), m.events, man["match_t0"])
(vis / "auto_events_reconciliation.json").write_text(json.dumps(rec, indent=1))
metrics = json.loads((demo / "metrics.json").read_text())
metrics["registration_eval"] = reg
metrics["vision_eval"] = visev
metrics["auto_events_eval"] = {k: v for k, v in rec.items() if k != "matches"}
(demo / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps({"registration": reg, "vision": visev, "auto_events": metrics["auto_events_eval"]}, indent=1))
