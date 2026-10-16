"""Assemble the < 2 minute submission video from rights-clear material only.

Inputs: the demo bundle (outputs/demo, from scripts/build_demo.py) and an optional UI screen recording
(outputs/demo/media/ui_capture.webm, from scripts/ui/record_ui.cjs). No music, no third-party footage.

    python scripts/make_demo_video.py --out outputs/submission/pitchlens_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pitchlens.render.assets import font, rgb, tokens
from pitchlens.render.video import render_video
from pitchlens.schema import CameraTrack, OverlayTimeline, ScreenTracks
from pitchlens.synth.generate import SyntheticMatch

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 30
C = tokens()["colors"]


def slide(lines: list[tuple[str, str, int]], sub: str = "", boxes: list | None = None) -> np.ndarray:
    """Brand slide. lines = [(text, font kind, px)]."""
    im = Image.new("RGB", (W, H), rgb("#070a1f"))
    d = ImageDraw.Draw(im)
    for y in range(H):  # subtle vertical gradient
        k = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * k) for a, b in zip(rgb("#0d1450"), rgb("#070a1f"), strict=True)))
    d.ellipse([140, 120, 200, 180], outline=rgb(C["accent"]), width=12)
    y = 260 if boxes is None else 110
    for text, kind, px in lines:
        d.text((140, y), text, font=font(kind, px), fill=(255, 255, 255))
        y += int(px * 1.25)
    if sub:
        d.text((140, y + 10), sub, font=font("semibold", 32), fill=rgb(C["text_dim"]))
    for (x0, y0, x1, y1, title, body, col) in boxes or []:
        d.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=rgb("#151b4a"), outline=rgb(col), width=4)
        d.text((x0 + 22, y0 + 18), title, font=font("bold", 30), fill=(255, 255, 255))
        yy = y0 + 64
        for ln in body.split("\n"):
            d.text((x0 + 22, yy), ln, font=font("semibold", 22), fill=rgb(C["text_dim"]))
            yy += 30
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f = font("bold", 30)
    tw = d.textlength(text, font=f)
    x0 = (W - tw) / 2 - 26
    d.rounded_rectangle([x0, 22, x0 + tw + 52, 80], radius=29, fill=(7, 10, 31, 225), outline=rgb(C["accent"]) + (255,), width=3)
    d.text((x0 + 26, 34), text, font=f, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(Image.alpha_composite(im, ov).convert("RGB")), cv2.COLOR_RGB2BGR)


def frames_of(video: Path, f0: int, f1: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for _ in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[1] != W:
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
        yield fr
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, default=ROOT / "outputs/demo")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/submission/pitchlens_demo.mp4")
    a = ap.parse_args()
    media, seg = a.demo / "media", a.demo / "segments"
    seg.mkdir(exist_ok=True)
    man = json.loads((a.demo / "manifest.json").read_text())
    m = SyntheticMatch.load(ROOT / "data/synthetic/demo")
    cam = CameraTrack.model_validate_json((a.demo / "camera.json").read_text())
    tracks = ScreenTracks.model_validate_json((a.demo / "screen_tracks.json").read_text())
    footage = media / "footage.mp4"
    # Hero moment = first stat card of the casual profile
    tl_c = json.loads((a.demo / "timelines/casual.json").read_text())
    hero = min(i["t_start"] for i in tl_c["items"] if i["type"] == "stat_card")
    h0 = max(int((hero - 1.0) * FPS), 0)
    variants = [("analyst", "en", "Data analyst: dense metrics, team share, context"),
                ("kids", "hu", "Young fan, in Hungarian: simple words, big moments"),
                ("player_focus", "en", "Player cam: follow #7 all match, speed chips on thresholds")]
    for prof, lang, _ in variants:
        dst = seg / f"{prof}_{lang}.mp4"
        if not dst.exists():
            tl = OverlayTimeline.model_validate_json((a.demo / f"timelines/{prof}.json").read_text())
            render_video(footage, tl, cam, m.meta, tracks, composite_out=dst, lang=lang,
                         frame_range=(h0, min(h0 + 9 * FPS, man["frames"])))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    def put(fr: np.ndarray, k: int = 1) -> None:
        nonlocal n
        for _ in range(k):
            ff.stdin.write(fr.tobytes())  # type: ignore[union-attr]
            n += 1

    put(slide([("PitchLens", "display", 150), ("Insight overlays that see the game", "bold", 56)],
              "An agent team turns match events into explainable, on-pitch graphics. Synthetic data, fictional clubs."), 4 * FPS)
    for k, fr in enumerate(frames_of(footage, 0, 4 * FPS)):
        put(caption(fr, "Synthetic broadcast footage rendered from synthetic tracking. No real data." if k < 2 * FPS
                    else "Vision: pitch registration, tracking, identity - from one manual calibration"))
    casual = media / "burned_casual.mp4"
    for k, fr in enumerate(frames_of(casual, h0, min(h0 + 12 * FPS, man["frames"]))):
        txt = ("Director picks the moment, Verifier checks every number" if k < 4 * FPS else
               "Evidence on the grass: each interception, glued to the pitch as the camera pans")
        put(caption(fr, txt))
    for prof, lang, txt in variants:
        for fr in frames_of(seg / f"{prof}_{lang}.mp4", 0, 7 * FPS if prof != "player_focus" else 6 * FPS):
            put(caption(fr, txt))
    ui = media / "ui_capture.webm"
    if ui.exists():
        for fr in frames_of(ui, int(4.5 * FPS), int(52 * FPS)):  # skip page load
            put(fr)
    rc = media / "recap_en.png"
    if rc.exists():
        put(caption(cv2.resize(cv2.imread(str(rc)), (W, H)), "Recap agent: automatic post-match story, all languages"), 4 * FPS)
        put(caption(cv2.resize(cv2.imread(str(media / "recap_fr.png")), (W, H)), "... in French, Spanish, German, Dutch, Portuguese, Hungarian"), 2 * FPS)
    boxes = [
        (140, 330, 600, 560, "1 Ingest", "Synthetic events + 10 Hz tracking\nAzure Event Hubs", "#7fb2ff"),
        (640, 330, 1100, 560, "2 Interpret", "Match state, 12 detectors\nPPDA, xG, sprints, evidence", "#7fb2ff"),
        (1140, 330, 1780, 560, "3 Explain", "Director, Explainer, Verifier\nAzure OpenAI (AI Foundry) + grounding", C["milestone"]),
        (140, 600, 900, 830, "4 Render", "Overlay timeline JSON + homography\nPython compositor, alpha WebM, browser canvas", C["accent"]),
        (940, 600, 1780, 830, "5 Personalize", "6 profiles x 7 languages, recap\nAzure AI Translator + AI Speech", "#ff9f6b"),
    ]
    put(slide([("One engine, five stages, eight agents", "display", 64)],
              "Hosted on Azure Container Apps (azd + Bicep), Web PubSub fan-out, Blob Storage, GitHub Actions CI", boxes), 7 * FPS)
    put(slide([("PitchLens", "display", 130), ("See the stat. Know why it matters.", "bold", 56)],
              "Open source, runs offline in mock mode, Azure-ready. All data, clubs and footage are synthetic."), 3 * FPS)
    ff.stdin.close()  # type: ignore[union-attr]
    ff.wait()
    print(json.dumps({"out": str(a.out), "seconds": round(n / FPS, 1)}))


if __name__ == "__main__":
    main()
