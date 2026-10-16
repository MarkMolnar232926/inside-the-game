# Schema reference (v1.0.0)

The machine-readable JSON Schemas live in [`docs/schema/`](schema/). Regenerate them with `pitchlens schema`; CI fails if they drift from the pydantic models in `src/pitchlens/schema/`.

| File | Model | Purpose |
|---|---|---|
| `event.schema.json` | `Event` | One match event (ingest stage). Pitch metres on 105 x 68, match clock, `pass_info` / `shot_info`, `source` + `confidence` |
| `match_meta.schema.json` | `MatchMeta` | Fictional teams, squads, kits, attributes, seed, narrative |
| `insight_candidate.schema.json` | `InsightCandidate` | Detector output with `evidence` (event ids, pitch points, facts) |
| `overlay_timeline.schema.json` | `OverlayTimeline` | **The renderer contract** |
| `camera_track.schema.json` | `CameraTrack` | Per-frame pitch->image homographies + quality |
| `screen_tracks.schema.json` | `ScreenTracks` | Identified player boxes / pitch xy / speed per frame |

## Overlay timeline in one example

```json
{
  "schema_version": "1.0.0", "match_id": "SYN-0006-LATE", "profile_id": "casual",
  "languages": ["en", "hu", "nl", "fr", "es", "de", "pt"], "default_language": "en",
  "media": {"fps": 30, "width": 1920, "height": 1080, "duration_s": 30.5, "match_t_at_media_zero": 3721},
  "items": [
    {"type": "stat_card", "id": "casual-card-001", "t_start": 10.3, "t_end": 19.3, "value": "3",
     "player_id": "KBY-07", "player_number": 7, "player_name": "Jory Kestane", "team_name": "Kestrel Bay FC",
     "placement": "right", "density": "standard",
     "text": {"en": {"headline": "Interceptions in attacking third"}, "hu": {"headline": "Labdaszerzés a támadóharmadban"}},
     "grounding": {"verified": true, "checks": ["1 number(s) match evidence"], "generator": "template"},
     "insight_id": "interceptions_attacking_third:KBY-07:3731", "evidence_event_ids": ["SYN-0006-LATE-e01532", "..."]},
    {"type": "pitch_markers", "id": "casual-markers-002", "t_start": 10.6, "t_end": 19.3, "shape": "star", "numbered": true,
     "points": [{"x": 19.5, "y": 9.5, "event_id": "...", "appear_at": 0.0}, {"x": 11.6, "y": 21.0, "appear_at": 0.45}]},
    {"type": "player_highlight", "id": "casual-highlight-003", "t_start": 10.3, "t_end": 19.3, "player_id": "KBY-07",
     "display_name": "KESTANE", "chips": [{"label": "speed", "value": "27.9", "unit": "km/h", "appear_at": 2.1}]},
    {"type": "narrative_lower_third", "id": "casual-lower-004", "t_start": 10.9, "t_end": 19.3, "tone": "pressure",
     "text": {"en": {"label": "Why it matters", "body": "Kestane keeps snatching the ball high up the pitch: ..."}}}
  ]
}
```

Rules for renderers:

* Times are on the **media clock** (seconds from the first video frame). `match_clock = media_t + match_t_at_media_zero`.
* Pitch geometry is in metres. Project it with the frame's homography from `CameraTrack` (`frame = round(media_t * fps)`).
* Player-anchored items (`player_highlight`, `speed_tag` with `player_id`) use `ScreenTracks` (nearest frame within 6).
* Pick `text[lang]`, falling back to `en`. Chip labels are locale label keys (`/api/locales/{lang}` -> `labels`).
* Style tokens (colours, fonts, sizes, motion) are in `src/pitchlens/render/tokens.json`, served at `/api/tokens`.
