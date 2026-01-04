"""HTTP-based pub/sub for cross-process communication.

When LiveView runs in a separate process, we use HTTP to send updates
from the main process to the LiveView server.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

try:
    import urllib.request
    import urllib.parse
except ImportError:
    urllib = None  # type: ignore

logger = logging.getLogger(__name__)


class HttpPubSubClient:
    """HTTP client for sending pub/sub messages to LiveView server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        """Initialize HTTP pub/sub client.
        
        Args:
            base_url: Base URL of the LiveView server.
        """
        self.base_url = base_url.rstrip("/")
        self._endpoint = f"{self.base_url}/api/pubsub"

    def send_message(self, topic: str, payload: Any) -> bool:
        """Send a message to the LiveView server via HTTP.
        
        Args:
            topic: The topic name.
            payload: The message payload.
            
        Returns:
            True if successful, False otherwise.
        """
        if urllib is None:
            logger.warning("urllib not available, cannot send HTTP messages")
            return False

        try:
            data = json.dumps({"topic": topic, "payload": payload}).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.5) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Failed to send HTTP pub/sub message: {e}")
            return False


# Global HTTP client instance
_http_client: Optional[HttpPubSubClient] = None


def set_http_client(client: HttpPubSubClient) -> None:
    """Set the global HTTP client for cross-process pub/sub."""
    global _http_client
    _http_client = client


def get_http_client() -> Optional[HttpPubSubClient]:
    """Get the global HTTP client."""
    return _http_client

