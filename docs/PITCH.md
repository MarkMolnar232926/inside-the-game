# PitchLens: the elevator pitch

**PitchLens is an agentic insight-overlay engine for football broadcasts.** It watches the live event stream, decides which stat is worth showing *right now*, explains *why it matters*, and draws the evidence directly onto the pitch in the video. A viewer doesn't just read "4 interceptions in attacking third": they see the four stars appear on the grass where they happened, glued to the turf while the camera pans.

## The problem

Broadcasters and streamers sit on rich event data, but on screen it mostly becomes a number in a box. Choosing the right stat at the right moment, saying why it matters, and drawing it spatially takes a producer, a data journalist and a graphics operator. That doesn't scale to every match, language and viewer. Fans also want different things: a casual fan wants one story, an analyst wants PPDA and xG, a kid wants speed, a supporter wants their club.

## What we built

1. **Ingest**: a synthetic, football-realistic match (events + 10 Hz tracking, fictional clubs) replayed or streamed live (Azure Event Hubs ready).
2. **Interpret**: rolling match state (possession sequences, zones, PPDA, field tilt, xG momentum, sprints) and 12 deterministic detectors that produce insight candidates *with evidence*.
3. **Explain**: an agent team. A **Director** acts as the broadcast director (newsworthiness, freshness, cooldowns, screen budget, never covering key action, cutting to a bigger story). An **Explainer** writes the why-it-matters line (control vs chaos, tempo, pressure). A **Verifier** rejects any number, name or superlative not in the evidence.
4. **Render**: a versioned, machine-readable **overlay timeline** (JSON). It is drawn by a Python broadcast compositor (burn-in or transparent alpha stream for a vision mixer) *and* by a browser canvas renderer, using per-frame homographies from our pitch-registration stack.
5. **Personalize**: the same intelligence becomes six experiences (casual, analyst, young fan, player cam, single-metric, favourite club) in seven languages, switchable instantly. A **Recap** agent writes the post-match story, again verified.

## Microsoft / Azure technology

* **Azure OpenAI in Azure AI Foundry**: Explainer and Recap agents (JSON mode, facts-only prompts, tool schemas from our typed `ToolRegistry`).
* **Azure AI Translator** and **Azure AI Speech** (neural TTS, SSML): multilingual storytelling and commentary audio.
* **Azure Event Hubs**: live match-event ingest. **Azure Web PubSub**: fan-out of overlay instructions to viewers.
* **Azure Blob Storage**: rendered assets and recap cards. **Azure Container Apps**: hosting, via `azd up` with Bicep. **GitHub Actions**: lint, tests (including the grounding eval), Docker build.
* Everything has a local mock behind the same interfaces, so the whole pipeline runs offline, with no keys, in CI.

## Why it is credible

* **Explainable by construction.** Every overlay links to its evidence events, its facts, the Director's scoring formula and the Verifier's checks ("Why am I seeing this?"). CI runs a grounding eval over every template in every language and register. The build fails on any ungrounded number or name.
* **Measured, not eyeballed.** On our synthetic footage we know the true camera. Registration from a single manual calibration holds about 1 px mean error over 30 s of panning. Markers stay glued to the grass.
* **Real-time.** A full 90-minute match runs through all agents for 6 profiles x 7 languages in seconds. Event-to-decision is ~10 ms at p99, against a 1 s budget.
* **Rights-clear.** Synthetic data, synthetic footage, generated crests and portraits, open fonts. No real clubs, players, leagues or broadcast marks.

## Who it is for

Rights holders and broadcasters (automated, consistent graphics in every language), streaming platforms (personalised feeds), studios (the evidence and recap tooling), and the graphics partners who already render overlays: the JSON contract drops into their engines.
