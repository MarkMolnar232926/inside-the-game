"""Export JSON Schema documents for every public contract.

Run ``python -m pitchlens.schema.export`` to regenerate ``docs/schema/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from . import CameraTrack, Event, InsightCandidate, MatchMeta, OverlayTimeline, ScreenTracks
from .overlay import SCHEMA_VERSION

MODELS = {
    "match_meta": MatchMeta,
    "event": Event,
    "insight_candidate": InsightCandidate,
    "overlay_timeline": OverlayTimeline,
    "camera_track": CameraTrack,
    "screen_tracks": ScreenTracks,
}


def export(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in MODELS.items():
        schema = TypeAdapter(model).json_schema()
        schema["$id"] = f"https://pitchlens.dev/schema/{SCHEMA_VERSION}/{name}.json"
        schema["x-schema-version"] = SCHEMA_VERSION
        p = out_dir / f"{name}.schema.json"
        p.write_text(json.dumps(schema, indent=2))
        written.append(p)
    return written


if __name__ == "__main__":
    for p in export(Path("docs/schema")):
        print(p)
