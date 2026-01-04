"""Pub/sub compatibility layer for Python 3.8.

This provides a simple pub/sub implementation that matches the interface
used by pyview-web, but works with Python 3.8.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from .types import HubDict, PubSubCallback  # noqa: TC001

logger = logging.getLogger(__name__)


# Global pub/sub hub (singleton pattern like pyview)
pub_sub_hub: HubDict = defaultdict(list)


class PubSub:
    """Simple pub/sub implementation compatible with pyview-web interface."""

    def __init__(self, hub: HubDict, topic: str) -> None:
        """Initialize PubSub.

        Args:
            hub: The pub/sub hub (shared dictionary).
            topic: The topic to subscribe/publish to.
        """
        self.hub = hub
        self.topic = topic

    def _call_callback(
        self, callback: PubSubCallback, topic: str, payload: Any
    ) -> None:
        """Call a single callback synchronously.

        Args:
            callback: The callback function to call.
            topic: The topic name.
            payload: The message payload.
        """
        try:
            callback(topic, payload)
        except Exception as e:
            logger.exception("Error in pub/sub callback for topic '%s': %s", topic, e)

    async def _call_async_callback(
        self, callback: PubSubCallback, topic: str, payload: Any
    ) -> None:
        """Call a single async callback.

        Args:
            callback: The async callback function to call.
            topic: The topic name.
            payload: The message payload.
        """
        try:
            await callback(topic, payload)
        except Exception as e:
            logger.exception("Error in pub/sub callback for topic '%s': %s", topic, e)

    async def send_all_on_topic_async(self, topic: str, payload: Any) -> None:
        """Send a message to all subscribers on a topic.

        Args:
            topic: The topic to publish to.
            payload: The message payload.
        """
        if topic not in self.hub:
            return

        # Call all subscribers
        for callback in self.hub[topic]:
            if asyncio.iscoroutinefunction(callback):
                await self._call_async_callback(callback, topic, payload)
            else:
                self._call_callback(callback, topic, payload)

    def subscribe(self, callback: PubSubCallback) -> None:
        """Subscribe to the topic.

        Args:
            callback: Function to call when messages are published.
        """
        if callback not in self.hub[self.topic]:
            self.hub[self.topic].append(callback)

    def unsubscribe(self, callback: PubSubCallback) -> None:
        """Unsubscribe from the topic.

        Args:
            callback: Function to remove from subscribers.
        """
        if callback in self.hub[self.topic]:
            self.hub[self.topic].remove(callback)
