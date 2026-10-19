"""Azure Event Hubs adapter: publish and consume the synthetic event stream.

Install extras: ``pip install pitchlens[azure]``. Without a connection string
the producer is a no-op and the consumer raises, so callers fall back to
:class:`pitchlens.ingest.sources.LiveSource`.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable

from pitchlens.schema import Event


class EventHubsPublisher:
    def __init__(self, conn: str | None = None, hub: str | None = None):
        self.conn = conn or os.getenv("AZURE_EVENTHUB_CONNECTION_STRING")
        self.hub = hub or os.getenv("AZURE_EVENTHUB_NAME", "match-events")
        self._producer = None
        if self.conn:
            from azure.eventhub import EventHubProducerClient  # type: ignore[import-not-found]

            self._producer = EventHubProducerClient.from_connection_string(self.conn, eventhub_name=self.hub)

    @property
    def enabled(self) -> bool:
        return self._producer is not None

    def publish(self, events: Iterable[Event], partition_key: str | None = None) -> int:
        if self._producer is None:
            return 0
        from azure.eventhub import EventData  # type: ignore[import-not-found]

        batch = self._producer.create_batch(partition_key=partition_key)
        n = 0
        for e in events:
            batch.add(EventData(e.model_dump_json()))
            n += 1
        self._producer.send_batch(batch)
        return n


class EventHubsSource:
    """Async consumer yielding validated ``Event`` objects (match id = partition key)."""

    def __init__(self, conn: str | None = None, hub: str | None = None, consumer_group: str = "$Default"):
        self.conn = conn or os.getenv("AZURE_EVENTHUB_CONNECTION_STRING")
        self.hub = hub or os.getenv("AZURE_EVENTHUB_NAME", "match-events")
        self.group = consumer_group
        if not self.conn:
            raise RuntimeError("AZURE_EVENTHUB_CONNECTION_STRING not set; use LiveSource for local mode")

    async def stream(self) -> AsyncIterator[Event]:
        import asyncio

        from azure.eventhub.aio import EventHubConsumerClient  # type: ignore[import-not-found]

        q: asyncio.Queue[Event] = asyncio.Queue()

        async def on_event(partition_context, event):  # noqa: ANN001
            q.put_nowait(Event.model_validate(json.loads(event.body_as_str())))
            await partition_context.update_checkpoint(event)

        client = EventHubConsumerClient.from_connection_string(self.conn, consumer_group=self.group,
                                                               eventhub_name=self.hub)
        task = asyncio.create_task(client.receive(on_event=on_event, starting_position="-1"))
        try:
            while True:
                yield await q.get()
        finally:
            task.cancel()
            await client.close()
