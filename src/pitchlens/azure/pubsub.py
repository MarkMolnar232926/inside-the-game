"""Fan-out of overlay instructions to viewers: Azure Web PubSub or an in-process hub."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class LocalHub:
    """In-process broadcast hub used by the FastAPI WebSocket endpoint (mock mode)."""

    def __init__(self) -> None:
        self.queues: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.queues.discard(q)

    async def publish(self, group: str, message: dict[str, Any]) -> None:
        for q in list(self.queues):
            if q.full():
                continue
            q.put_nowait({"group": group, **message})


class WebPubSubHub:
    """Publishes to Azure Web PubSub groups (one group per profile); clients connect directly to Azure."""

    def __init__(self) -> None:
        from azure.messaging.webpubsubservice import WebPubSubServiceClient  # type: ignore[import-not-found]

        self.client = WebPubSubServiceClient.from_connection_string(
            os.environ["AZURE_WEBPUBSUB_CONNECTION_STRING"], hub=os.getenv("AZURE_WEBPUBSUB_HUB", "pitchlens"))

    async def publish(self, group: str, message: dict[str, Any]) -> None:
        await asyncio.to_thread(self.client.send_to_group, group, json.dumps(message), content_type="application/json")

    def client_url(self, group: str) -> str:
        tok = self.client.get_client_access_token(roles=[f"webpubsub.joinLeaveGroup.{group}"])
        return tok["url"]


def default_hub() -> LocalHub | WebPubSubHub:
    if os.getenv("PITCHLENS_MODE", "mock") == "azure" and os.getenv("AZURE_WEBPUBSUB_CONNECTION_STRING"):
        return WebPubSubHub()
    return LocalHub()
