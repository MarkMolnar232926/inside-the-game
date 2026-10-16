# Decision log

Short, dated records of choices made while building PitchLens, newest last.

| # | Decision | Why | Scoring axis |
|---|----------|-----|--------------|
| D1 | **The test clip `vizuals_test.mp4` was not available in the build environment** (only the reference image arrived, as a chat attachment). The rights-clear *synthetic broadcast footage generator* (planned for Phase 7) was moved to Phase 3 and became the primary development substrate. All vision code stays source-agnostic and accepts any clip via `--video`; see `docs/REAL_CLIPS.md`. | Keeps the build unblocked, and synthetic footage has *known ground-truth cameras*, which turns homography accuracy into a measurable test instead of an eyeball check. | Tech, UX, compliance |
| D2 | Name: **PitchLens**. Fictional clubs **Kestrel Bay FC (KBY)**, crimson/navy, and **Ironvale Athletic (IRV)**, white/teal. Demo protagonist: **#7 Jory Kestane** (KBY, "pressing playmaker"). | All marks are invented; colours mirror the red-vs-white contrast of the reference footage. | Compliance |
| D3 | Python 3.11+ (developed on 3.13), pydantic v2 contracts, FastAPI, OpenCV/NumPy. Plain HTML/JS web UI instead of React. | No build step, fastest iteration; the canvas renderer is ~1 file. | UX, maintainability |
| D4 | The simulator is **agent-based at 10 Hz** (players chase tactical targets; possession state machine), not an event-only Markov chain. | One simulation yields consistent events *and* tracking, so speeds, sprints, pressure and video alignment all agree. | Dataset innovation |
| D5 | Narrative "scripts" only nudge conversion of decent chances (xG >= 0.08) inside time windows. They never inject goals out of nowhere. | Storylines (late comeback, etc.) happen reliably, while the data still looks like football. | Dataset realism |
| D6 | Demo match: `seed=19, narrative=late_comeback`. IRV lead 2-1 at half-time, KBY win 4-2 (69', 80', 85'). #7's 4th attacking-third interception at 84:21 sets up the 85' winner. | Best story among 26 scanned seeds (`scripts/seed_scan.py`). | Real-world impact, UX |
| D7 | Pitch lines in the synthetic footage are drawn as projected ground-plane quads (real 12 cm width), not warped texture. | Perspective-correct, anti-aliased lines at every depth; no texture minification shimmer. | UX |
| D8 | The overlay timeline is renderer-agnostic JSON on the *media clock*, with pitch-space geometry. Camera (homography) and player screen tracks are separate streams. | The engine, the camera solver and the renderer can each be swapped (e.g. a broadcaster's own graphics engine). | Tech, impact |
