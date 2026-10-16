# 90-second demo script

The video is assembled automatically by `scripts/make_demo_video.py` from rights-clear material only: synthetic footage, generated graphics, a UI screen recording of our own app, and no music. The voice-over below can be read live, or synthesised with Azure AI Speech (`pitchlens.agents.voice.VoiceAgent`).

| t | Picture | Voice-over |
|---|---|---|
| 0:00-0:04 | Title card | "This is PitchLens. It turns raw match events into explainable graphics that sit *on* the pitch." |
| 0:04-0:08 | Clean synthetic broadcast feed | "Everything you see is synthetic: fictional clubs, a simulated match, and broadcast footage rendered from simulated tracking. Our vision stack registers the pitch from a single manual calibration." |
| 0:08-0:20 | Casual-fan render of the hero moment: #7 Kestane intercepts, the Director cuts to *3 interceptions in attacking third*, stars pop onto the grass, ring and name tag on the player, why-it-matters line | "An agent team watches the event stream. The Scout finds the stat. The Director decides it's worth airtime *now*. The Explainer says why it matters, and the Verifier checks every number against the data. The evidence is drawn where it happened, and it stays glued to the grass while the camera pans." |
| 0:20-0:27 | Same moment, analyst profile | "Same intelligence, different viewer. Analysts get team share, distance to goal and the rhythm context: control versus chaos." |
| 0:27-0:34 | Same moment, young-fan profile in Hungarian | "Kids get simple words. Seven languages, switchable instantly, because every overlay carries text for all of them." |
| 0:34-0:40 | Player cam | "Or follow one player all match, with speed chips when they cross a threshold." |
| 0:40-1:14 | UI recording: viewer → *Why am I seeing this?* drawer (pitch map, events, facts, verifier checks, agent chain) → analyst + Spanish → live feed with overlays pushed over WebSocket → recap tab | "Every graphic is explainable. One click shows the events behind it, the grounded facts, the Director's scoring formula and the Verifier's checks. In live mode, events stream in and overlays are decided in about ten milliseconds. Overlay instructions are plain JSON, so any broadcast graphics engine can render them." |
| 1:14-1:20 | Recap cards | "At full time the Recap agent writes the story, with player of the match and evidence, in every language." |
| 1:20-1:27 | Architecture slide | "Five stages, eight agents, on Azure: OpenAI in AI Foundry, Translator and Speech, Event Hubs and Web PubSub, Container Apps. It also runs fully offline." |
| 1:27-1:30 | End card | "PitchLens. See the stat. Know why it matters." |

**Recording tips:** run `python scripts/build_demo.py`, start `uvicorn pitchlens.api.app:app`, run `NODE_PATH=$(npm root -g) node scripts/ui/record_ui.cjs http://localhost:8000 outputs/demo/media/ui_capture.webm`, then run `python scripts/make_demo_video.py`.
