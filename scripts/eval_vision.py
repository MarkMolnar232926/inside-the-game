"""Evaluate detection/tracking/identity against synthetic ground truth boxes."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from pitchlens.vision.track import iou

clip = Path(sys.argv[1]); vis = clip / "vision"
gt = pd.read_parquet(clip / "gt/gt_boxes.parquet")
gt = gt[(gt.entity != "ball") & (gt.x2 > 0) & (gt.x1 < 1920) & (gt.y2 > 0) & (gt.y1 < 1080)]
det = pd.read_parquet(vis / "detections.parquet"); det = det[det.cls == "person"]
tr = pd.read_parquet(vis / "tracks.parquet")
ident = json.loads((vis / "identity.json").read_text())
tp = fp = fn = 0; id_ok = id_n = 0; frag = {}
gtf = {f: g for f, g in gt.groupby("frame")}; trf = {f: g for f, g in tr.groupby("frame")}
for f, g in det.groupby("frame"):
    G = gtf.get(f)
    if G is None: fp += len(g); continue
    I = iou(g[["x1","y1","x2","y2"]].to_numpy(), G[["x1","y1","x2","y2"]].to_numpy())
    r, c = linear_sum_assignment(-I); m = I[r, c] > 0.3
    tp += m.sum(); fp += len(g) - m.sum(); fn += len(G) - m.sum()
for f, g in trf.items():
    G = gtf.get(f)
    if G is None: continue
    I = iou(g[["x1","y1","x2","y2"]].to_numpy(), G[["x1","y1","x2","y2"]].to_numpy())
    r, c = linear_sum_assignment(-I)
    for i, j in zip(r, c):
        if I[i, j] > 0.3:
            ent = G.entity.iloc[j]; tid = int(g.track_id.iloc[i])
            frag.setdefault(ent, set()).add(tid)
            if ent != "referee":
                id_n += 1; id_ok += ident.get(str(tid)) == ent
fr = [len(v) for k, v in frag.items() if k != "referee"]
rep = {"det_precision": round(tp / max(tp + fp, 1), 3), "det_recall": round(tp / max(tp + fn, 1), 3),
       "tracks_per_gt_player_mean": round(float(np.mean(fr)), 2), "tracks_per_gt_player_max": int(max(fr)),
       "identity_accuracy_on_matched_boxes": round(id_ok / max(id_n, 1), 3)}
(vis / "vision_eval.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))
