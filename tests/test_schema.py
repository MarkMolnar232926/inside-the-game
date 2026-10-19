"""Schema contracts and JSON Schema export."""

import json

import jsonschema
from pydantic import TypeAdapter

from pitchlens.schema import (
    SCHEMA_VERSION,
    MediaClock,
    OverlayTimeline,
    PitchMarkers,
    PitchPoint,
    StatCard,
    TextBlock,
)
from pitchlens.schema.export import MODELS, export


def _timeline():
    return OverlayTimeline(
        match_id="M", profile_id="casual", languages=["en", "hu"], media=MediaClock(duration_s=10, match_t_at_media_zero=0),
        items=[StatCard(id="c", t_start=1, t_end=5, value="4", text={"en": TextBlock(headline="X"), "hu": TextBlock(headline="Y")}),
               PitchMarkers(id="m", t_start=1, t_end=5, points=[PitchPoint(x=10, y=20)])],
    )


def test_roundtrip_discriminated_union():
    tl = _timeline()
    back = OverlayTimeline.model_validate_json(tl.model_dump_json())
    assert back == tl and back.items[1].type.value == "pitch_markers"
    assert [i.id for i in back.active(2.0)] == ["c", "m"] and back.active(6.0) == []


def test_json_schema_validates_instances(tmp_path):
    paths = export(tmp_path)
    assert len(paths) == len(MODELS)
    schema = json.loads((tmp_path / "overlay_timeline.schema.json").read_text())
    assert schema["x-schema-version"] == SCHEMA_VERSION
    jsonschema.validate(json.loads(_timeline().model_dump_json()), schema)
    bad = json.loads(_timeline().model_dump_json())
    bad["items"][0]["type"] = "hologram"
    try:
        jsonschema.validate(bad, schema)
        raise AssertionError("invalid overlay type accepted")
    except jsonschema.ValidationError:
        pass


def test_committed_schema_docs_up_to_date():
    from pathlib import Path

    for name, model in MODELS.items():
        p = Path(__file__).resolve().parents[1] / "docs/schema" / f"{name}.schema.json"
        committed = json.loads(p.read_text())
        current = TypeAdapter(model).json_schema()
        assert committed["properties"].keys() == current["properties"].keys(), f"re-run schema export for {name}"
