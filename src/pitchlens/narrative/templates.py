"""Locale templates and grounded slot filling (deterministic narrative fallback).

Templates only reference slots that exist in an insight's ``evidence.facts``
(plus ``ctx_*`` rhythm facts the Explainer adds). Missing slots raise, so a
template can never invent a number. Numbers are formatted per locale
(decimal comma for hu/nl/fr/es/de/pt).
"""

from __future__ import annotations

import json
import string
from functools import cache
from pathlib import Path
from typing import Any

LOCALE_DIR = Path(__file__).with_name("locales")
SUPPORTED = ["en", "hu", "nl", "fr", "es", "de", "pt"]
REGISTERS = ["casual", "analyst", "kids"]


@cache
def locale(lang: str) -> dict:
    return json.loads((LOCALE_DIR / f"{lang}.json").read_text())


def fmt_number(v: Any, lang: str) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Facts are already rounded by the detectors; print them exactly (no re-rounding = no drift).
        return f"{v:g}".replace(".", locale(lang)["decimal"])
    return str(v)


class _Formatter(string.Formatter):
    def __init__(self, lang: str):
        self.lang = lang

    def format_field(self, value: Any, format_spec: str) -> str:
        return fmt_number(value, self.lang)


def slots(template: str) -> list[str]:
    return [f for _, f, _, _ in string.Formatter().parse(template) if f]


def fill(template: str, facts: dict, lang: str) -> str:
    missing = [s for s in slots(template) if s not in facts or facts[s] is None]
    if missing:
        raise KeyError(f"template needs ungrounded slots {missing}")
    return _Formatter(lang).format(template, **facts)


def label(key: str, lang: str) -> str:
    return locale(lang)["labels"].get(key) or locale("en")["labels"].get(key, key)


def render(template_key: str, register: str, facts: dict, lang: str) -> tuple[str, str]:
    """Return (headline, why) for an insight in a register and language."""
    t = locale(lang)["templates"].get(template_key) or locale("en")["templates"][template_key]
    headline = fill(t["headline"], facts, lang)
    reg = register if register in t else "casual"
    key = f"{reg}_top" if facts.get("rank") == 1 and f"{reg}_top" in t else reg
    return headline, fill(t[key], facts, lang)


def context_line(kind: str, facts: dict, lang: str) -> str:
    return fill(locale(lang)["context"][kind], facts, lang)


def recap_string(key: str, facts: dict, lang: str) -> str:
    return fill(locale(lang)["recap"][key], facts, lang)
