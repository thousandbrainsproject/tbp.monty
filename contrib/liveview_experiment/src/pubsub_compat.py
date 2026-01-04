"""Pub/sub compatibility layer for Python 3.8.

This provides a simple pub/sub implementation that matches the interface
used by pyview-web, but works with Python 3.8.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


# Global pub/sub hub (singleton pattern like pyview)
pub_sub_hub: Dict[str, List[Callable[[str, Any], None]]] = defaultdict(list)


class PubSub:
    """Simple pub/sub implementation compatible with pyview-web interface."""

    def __init__(self, hub: Dict[str, List[Callable[[str, Any], None]]], topic: str) -> None:
        """Initialize PubSub.

        Args:
            hub: The pub/sub hub (shared dictionary).
            topic: The topic to subscribe/publish to.
        """
        self.hub = hub
        self.topic = topic

    def _call_callback(self, callback: Callable[[str, Any], None], topic: str, payload: Any) -> None:
        """Call a single callback synchronously.
        
        Args:
            callback: The callback function to call.
            topic: The topic name.
            payload: The message payload.
        """
        try:
            callback(topic, payload)
        except Exception as e:
            logger.error(f"Error in pub/sub callback for topic '{topic}': {e}", exc_info=True)

    async def _call_async_callback(
        self, callback: Callable[[str, Any], None], topic: str, payload: Any
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
            logger.error(f"Error in pub/sub callback for topic '{topic}': {e}", exc_info=True)

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

    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to the topic.

        Args:
            callback: Function to call when messages are published.
        """
        if callback not in self.hub[self.topic]:
            self.hub[self.topic].append(callback)

    def unsubscribe(self, callback: Callable[[str, Any], None]) -> None:
        """Unsubscribe from the topic.

        Args:
            callback: Function to remove from subscribers.
        """
        if callback in self.hub[self.topic]:
            self.hub[self.topic].remove(callback)

