"""Run pitch registration on a clip and compare with ground truth (synthetic footage only)."""
import json, sys, time
from pathlib import Path
import numpy as np
from pitchlens.schema import CameraTrack
from pitchlens.vision.registration import Calibration, PitchRegistrar, RegistrationConfig, reprojection_error

clip = Path(sys.argv[1]); calib = Path(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else None
t = time.time()
reg = PitchRegistrar(Calibration.load(calib), RegistrationConfig())
est = reg.run(clip / "footage.mp4", n_frames=n, progress=True)
el = time.time() - t
gt = CameraTrack.model_validate_json((clip / "gt/camera_gt.json").read_text())
errs = np.array([reprojection_error(est.H(i), gt.H(i), gt.width, gt.height) for i in range(len(est.homographies))])
out = clip / "vision"; out.mkdir(exist_ok=True)
(out / "homography.json").write_text(est.model_dump_json())
rep = {"frames": len(errs), "seconds": round(el, 1), "fps": round(len(errs) / el, 1), "mean_px": round(float(np.nanmean(errs)), 2),
       "p95_px": round(float(np.nanpercentile(errs, 95)), 2), "max_px": round(float(np.nanmax(errs)), 2),
       "mean_quality": round(float(np.mean(est.quality)), 3), "err_every_60": [round(float(e), 2) for e in errs[::60]]}
(out / "registration_eval.json").write_text(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
