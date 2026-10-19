"""Build renderer-facing ``ScreenTracks`` from tracking tables."""

from __future__ import annotations

import pandas as pd

from pitchlens.schema import ScreenTrack, ScreenTracks


def from_table(df: pd.DataFrame, fps: float, id_col: str = "entity", source: str = "vision_estimated",
               track_col: str | None = None, px: str = "pitch_x", py: str = "pitch_y") -> ScreenTracks:
    """``df`` needs frame, x1, y1, x2, y2, pitch_x/pitch_y, speed_kmh and an id column."""
    tracks: dict[str, ScreenTrack] = {}
    for pid, g in df.groupby(id_col):
        pid = str(pid)
        if pid in ("referee", "nan", "None"):
            continue
        g = g.sort_values("frame").drop_duplicates("frame")
        tracks[pid] = ScreenTrack(
            player_id=pid,
            track_id=int(g[track_col].iloc[0]) if track_col else 0,
            frames=g.frame.astype(int).tolist(),
            boxes=list(map(tuple, g[["x1", "y1", "x2", "y2"]].round(1).to_numpy().tolist())),
            pitch_xy=list(map(tuple, g[[px, py]].round(2).to_numpy().tolist())),
            speed_kmh=g.speed_kmh.round(1).tolist(),
            source=source,  # type: ignore[arg-type]
        )
    return ScreenTracks(fps=fps, tracks=tracks)


def from_ground_truth(gt_boxes: pd.DataFrame, fps: float) -> ScreenTracks:
    return from_table(gt_boxes, fps, source="synthetic")
