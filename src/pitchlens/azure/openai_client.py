"""LLM clients behind one interface.

* :class:`AzureOpenAIClient` - Azure OpenAI / Azure AI Foundry chat completions
  (REST via httpx, JSON mode, optional function tools). Configured from
  ``AZURE_OPENAI_*`` environment variables.
* :class:`MockLLM` - deterministic, offline stand-in that returns ``None`` so
  agents fall back to grounded templates. Used in CI and by default.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import httpx


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, tools: list[dict] | None = None,
                      max_tokens: int = 300) -> dict | None: ...


class MockLLM:
    name = "mock"

    def complete_json(self, system: str, user: str, tools: list[dict] | None = None,
                      max_tokens: int = 300) -> dict | None:
        return None


class AzureOpenAIClient:
    name = "azure-openai"

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str = "2024-10-21",
                 timeout: float = 8.0):
        self.url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        self.headers = {"api-key": api_key, "Content-Type": "application/json"}
        self.client = httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> AzureOpenAIClient | None:
        ep, key = os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")
        if not ep or not key:
            return None
        return cls(ep, key, os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                   os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))

    def complete_json(self, system: str, user: str, tools: list[dict] | None = None,
                      max_tokens: int = 300) -> dict | None:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.4,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if tools:
            body["tools"] = tools
        try:
            r = self.client.post(self.url, headers=self.headers, json=body)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return json.loads(msg.get("content") or "{}")
        except (httpx.HTTPError, KeyError, json.JSONDecodeError):
            return None


def default_llm() -> LLMClient:
    if os.getenv("PITCHLENS_MODE", "mock") == "azure":
        c = AzureOpenAIClient.from_env()
        if c is not None:
            return c
    return MockLLM()
