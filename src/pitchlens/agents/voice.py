"""Translator / Voice agent: multilingual text and optional text-to-speech.

* Template mode: every language is rendered natively from its own locale
  template (no machine translation, so grammar and number formats are right).
* LLM mode: free-form English from the Explainer is machine-translated with
  Azure AI Translator, then re-verified (numbers must survive translation).
* Voice: SSML for Azure AI Speech neural voices; with keys configured the
  audio is synthesised to ``<out>/<id>.<lang>.mp3``, otherwise only the SSML
  is written (mock mode).
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import httpx

VOICES = {
    "en": "en-GB-RyanNeural", "hu": "hu-HU-TamasNeural", "nl": "nl-NL-MaartenNeural", "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural", "de": "de-DE-ConradNeural", "pt": "pt-PT-DuarteNeural",
}
LOCALES = {"en": "en-GB", "hu": "hu-HU", "nl": "nl-NL", "fr": "fr-FR", "es": "es-ES", "de": "de-DE", "pt": "pt-PT"}


def ssml(text: str, lang: str, style: str = "excited") -> str:
    v = VOICES.get(lang, VOICES["en"])
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{LOCALES.get(lang, "en-GB")}">'
            f'<voice name="{v}"><mstts:express-as style="{style}"><prosody rate="+6%">{html.escape(text)}'
            f"</prosody></mstts:express-as></voice></speak>")


class AzureTranslator:
    def __init__(self, key: str, region: str, endpoint: str = "https://api.cognitive.microsofttranslator.com"):
        self.key, self.region, self.endpoint = key, region, endpoint.rstrip("/")

    @classmethod
    def from_env(cls) -> AzureTranslator | None:
        key = os.getenv("AZURE_TRANSLATOR_KEY")
        if not key:
            return None
        return cls(key, os.getenv("AZURE_TRANSLATOR_REGION", "westeurope"),
                   os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com"))

    def translate(self, text: str, to: list[str], src: str = "en") -> dict[str, str]:
        r = httpx.post(f"{self.endpoint}/translate", params={"api-version": "3.0", "from": src, "to": to},
                       headers={"Ocp-Apim-Subscription-Key": self.key, "Ocp-Apim-Subscription-Region": self.region},
                       json=[{"text": text}], timeout=8)
        r.raise_for_status()
        return {t["to"]: t["text"] for t in r.json()[0]["translations"]}


class VoiceAgent:
    name = "voice"

    def __init__(self, out_dir: Path | None = None):
        self.key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION", "westeurope")
        self.out_dir = out_dir

    @property
    def live(self) -> bool:
        return bool(self.key) and os.getenv("PITCHLENS_MODE", "mock") == "azure"

    def speak(self, item_id: str, text: str, lang: str) -> str | None:
        """Return an asset key (path relative to out_dir) or None."""
        if self.out_dir is None or not text:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        doc = ssml(text, lang)
        (self.out_dir / f"{item_id}.{lang}.ssml").write_text(doc)
        if not self.live:
            return f"{item_id}.{lang}.ssml"
        r = httpx.post(f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1",
                       headers={"Ocp-Apim-Subscription-Key": self.key or "", "Content-Type": "application/ssml+xml",
                                "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3"},
                       content=doc.encode(), timeout=15)
        r.raise_for_status()
        (self.out_dir / f"{item_id}.{lang}.mp3").write_bytes(r.content)
        return f"{item_id}.{lang}.mp3"
