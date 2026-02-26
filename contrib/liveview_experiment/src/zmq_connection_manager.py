"""ZMQ connection management with retry logic."""

from __future__ import annotations

import logging
import time
from typing import Any

from .experiment_config import ConnectionRetryParams  # noqa: TC001

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
        params: ConnectionRetryParams,
    ) -> bool:
        """Connect socket with retry logic to handle slow joiner problem.

        Args:
            socket: ZMQ socket to connect
            params: Connection parameters with retry settings

        Returns:
            True if connected, False otherwise
        """
        if zmq is None or socket is None:
            return False

        for attempt in range(params.max_retries):
            if ZmqConnectionManager._try_connect(
                socket, params.zmq_host, params.zmq_port, attempt + 1
            ):
                return True

            if attempt < params.max_retries - 1:
                ZmqConnectionManager._log_retry(attempt + 1, params.retry_delay)
                time.sleep(params.retry_delay)
            else:
                ZmqConnectionManager._log_final_failure(params.max_retries)
                return True  # Continue anyway - ZMQ will auto-reconnect

        return False

    @staticmethod
    def _try_connect(socket: Any, zmq_host: str, zmq_port: int, attempt: int) -> bool:
        """Attempt to connect socket.

        Args:
            socket: ZMQ socket to connect
            zmq_host: ZMQ host
            zmq_port: ZMQ port
            attempt: Attempt number

        Returns:
            True if connected, False otherwise
        """
        if zmq is None:
            return False

        try:
            socket.connect(f"tcp://{zmq_host}:{zmq_port}")
            time.sleep(0.1)  # Small delay to ensure connection is established
            logger.info(
                "ZMQ broadcaster connected to tcp://%s:%d (attempt %d)",
                zmq_host,
                zmq_port,
                attempt,
            )
            return True
        except zmq.ZMQError:
            return False

    @staticmethod
    def _log_retry(attempt: int, retry_delay: float) -> None:
        """Log retry attempt.

        Args:
            attempt: Attempt number
            retry_delay: Delay before retry
        """
        logger.debug(
            "ZMQ connection attempt %d failed, retrying in %gs",
            attempt,
            retry_delay,
        )

    @staticmethod
    def _log_final_failure(max_retries: int) -> None:
        """Log final connection failure.

        Args:
            max_retries: Maximum number of retries attempted
        """
        logger.warning(
            "Failed to connect to ZMQ subscriber after %d attempts. "
            "Subscriber may not be ready yet. Continuing anyway.",
            max_retries,
        )
