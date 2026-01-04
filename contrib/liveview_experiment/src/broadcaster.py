"""Broadcaster for experiment data streams.

This module provides a simple interface for parallel processes to publish
data to the LiveView dashboard via pub/sub.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from .pubsub_compat import PubSub, pub_sub_hub
from .pubsub_http import get_http_client

logger = logging.getLogger(__name__)


class ExperimentBroadcaster:
    """Broadcasts experiment data updates via PubSub.
    
    This class can be used by parallel processes (threads, async tasks, etc.)
    to publish data to the LiveView dashboard. The LiveView subscribes to
    these topics and updates the display in real-time.
    
    Example:
        ```python
        # In a parallel process
        broadcaster = ExperimentBroadcaster()
        
        # Publish a metric update
        await broadcaster.publish_metric("loss", 0.5, epoch=1)
        
        # Publish custom data
        await broadcaster.publish_data("sensor_data", {"value": 123})
        ```
    """

    def __init__(self, base_topic: str = "experiment:updates:root") -> None:
        """Initialize the broadcaster.
        
        Args:
            base_topic: The base topic for broadcasts. Should match the
                       state manager's broadcast_topic.
        """
        self.base_topic = base_topic
        # Create sub-topics for different data types
        self.metrics_topic = f"{base_topic}:metrics"
        self.data_topic = f"{base_topic}:data"
        self.logs_topic = f"{base_topic}:logs"

    async def publish_metric(
        self, 
        name: str, 
        value: float, 
        **metadata: Any
    ) -> None:
        """Publish a metric update.
        
        Args:
            name: Metric name (e.g., "loss", "accuracy")
            value: Metric value
            **metadata: Additional metadata (e.g., epoch, step, episode)
        """
        payload = {"type": "metric", "name": name, "value": value, **metadata}
        
        # Try in-process pub/sub first
        try:
            pubsub = PubSub(pub_sub_hub, self.metrics_topic)
            await pubsub.send_all_on_topic_async(self.metrics_topic, payload)
            logger.debug(f"Published metric '{name}' via in-process pub/sub")
        except Exception as e:
            logger.debug(f"In-process pub/sub failed: {e}")
        
        # Also try HTTP (for cross-process)
        http_client = get_http_client()
        if http_client:
            http_client.send_message(self.metrics_topic, payload)

    async def publish_data(
        self, 
        stream_name: str, 
        data: Dict[str, Any]
    ) -> None:
        """Publish custom data to a named stream.
        
        Args:
            stream_name: Name of the data stream (e.g., "sensor_data", "activations")
            data: Dictionary of data to publish
        """
        payload = {"type": "data", "stream": stream_name, "data": data}
        
        # Try in-process pub/sub first
        try:
            pubsub = PubSub(pub_sub_hub, self.data_topic)
            await pubsub.send_all_on_topic_async(self.data_topic, payload)
            logger.debug(f"Published data stream '{stream_name}' via in-process pub/sub")
        except Exception as e:
            logger.debug(f"In-process pub/sub failed: {e}")
        
        # Also try HTTP (for cross-process)
        http_client = get_http_client()
        if http_client:
            http_client.send_message(self.data_topic, payload)

    async def publish_log(
        self, 
        level: str, 
        message: str, 
        **metadata: Any
    ) -> None:
        """Publish a log message.
        
        Args:
            level: Log level (e.g., "info", "warning", "error")
            message: Log message
            **metadata: Additional metadata
        """
        payload = {"type": "log", "level": level, "message": message, **metadata}
        
        # Try in-process pub/sub first
        try:
            pubsub = PubSub(pub_sub_hub, self.logs_topic)
            await pubsub.send_all_on_topic_async(self.logs_topic, payload)
            logger.debug(f"Published log [{level}] via in-process pub/sub")
        except Exception as e:
            logger.debug(f"In-process pub/sub failed: {e}")
        
        # Also try HTTP (for cross-process)
        http_client = get_http_client()
        if http_client:
            http_client.send_message(self.logs_topic, payload)

    def publish_metric_sync(
        self, 
        name: str, 
        value: float, 
        **metadata: Any
    ) -> None:
        """Synchronous version of publish_metric for use in threads.
        
        Args:
            name: Metric name
            value: Metric value
            **metadata: Additional metadata
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.publish_metric(name, value, **metadata),
                    loop
                )
            else:
                loop.run_until_complete(
                    self.publish_metric(name, value, **metadata)
                )
        except RuntimeError:
            # No event loop, create a new one
            asyncio.run(self.publish_metric(name, value, **metadata))
        except Exception as e:
            logger.error(f"Failed to publish metric (sync): {e}", exc_info=True)

    def publish_data_sync(
        self, 
        stream_name: str, 
        data: Dict[str, Any]
    ) -> None:
        """Synchronous version of publish_data for use in threads.
        
        Args:
            stream_name: Name of the data stream
            data: Dictionary of data to publish
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.publish_data(stream_name, data),
                    loop
                )
            else:
                loop.run_until_complete(
                    self.publish_data(stream_name, data)
                )
        except RuntimeError:
            # No event loop, create a new one
            asyncio.run(self.publish_data(stream_name, data))
        except Exception as e:
            logger.error(f"Failed to publish data (sync): {e}", exc_info=True)

