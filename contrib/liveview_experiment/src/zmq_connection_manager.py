"""ZMQ connection management with retry logic."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment, unused-ignore]


class ZmqConnectionManager:
    """Manages ZMQ socket connection with retry logic."""

    @staticmethod
    def create_and_configure_socket(context: Any) -> Any | None:
        """Create and configure ZMQ PUB socket.

        Args:
            context: ZMQ context (zmq.Context)

        Returns:
            Configured socket (zmq.Socket) or None if creation fails
        """
        if zmq is None or context is None:
            return None

        try:
            socket = context.socket(zmq.PUB)
            if socket is None:
                return None

            ZmqConnectionManager._configure_socket_options(socket)
            return socket
        except Exception as e:
            logger.exception("Failed to create ZMQ socket: %s", e)
            return None

    @staticmethod
    def _configure_socket_options(socket: Any) -> None:  # zmq.Socket
        """Configure socket options for better performance.

        Args:
            socket: ZMQ socket to configure
        """
        # LINGER: Wait up to 1 second for pending messages to be sent before closing
        # This ensures final messages (like "completed" status) are delivered
        socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger time
        socket.setsockopt(zmq.SNDHWM, 1000)  # High water mark for send queue

    @staticmethod
    def connect_with_retry(
        socket: Any,
        zmq_host: str,
        zmq_port: int,
        max_retries: int = 10,
        retry_delay: float = 0.5,  # zmq.Socket
    ) -> bool:
        """Connect socket with retry logic to handle slow joiner problem.

        Args:
            socket: ZMQ socket to connect
            zmq_host: ZMQ host
            zmq_port: ZMQ port
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds

        Returns:
            True if connected, False otherwise
        """
        if zmq is None or socket is None:
            return False

        for attempt in range(max_retries):
            try:
                socket.connect(f"tcp://{zmq_host}:{zmq_port}")
                time.sleep(0.1)  # Small delay to ensure connection is established
                logger.info(
                    "ZMQ broadcaster connected to tcp://%s:%d (attempt %d)",
                    zmq_host,
                    zmq_port,
                    attempt + 1,
                )
                return True
            except zmq.ZMQError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        "ZMQ connection attempt %d failed, retrying in %gs: %s",
                        attempt + 1,
                        retry_delay,
                        e,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.warning(
                        "Failed to connect to ZMQ subscriber after %d attempts. "
                        "Subscriber may not be ready yet. Continuing anyway.",
                        max_retries,
                    )
                    # Continue anyway - ZMQ will auto-reconnect when subscriber is ready
                    return True

        return False
