"""ZMQ command publisher for sending commands to experiments.

The LiveView server uses this to send commands (like abort) back to
the running experiment process.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class Command:
    """A command to send to the experiment.

    Attributes:
        type: Command type (e.g., "abort", "pause", "resume").
        payload: Optional additional data for the command.
        timestamp: Unix timestamp when command was created.
    """

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize command to JSON string."""
        return json.dumps(
            {
                "type": self.type,
                "payload": self.payload,
                "timestamp": self.timestamp,
            }
        )


class CommandPublisher:
    """ZMQ PUB socket for publishing commands to experiments.

    The publisher binds to a port, and experiments connect to receive commands.
    This follows the same bind/connect pattern as the telemetry stream, but
    in the opposite direction.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5556) -> None:
        """Initialize the command publisher.

        Args:
            host: Host to bind to.
            port: Port to bind to.
        """
        self.host = host
        self.port = port
        self._socket: Any = None
        self._context: Any = None
        self._initialized = False

    def initialize(self, context: Any = None) -> bool:
        """Initialize the ZMQ socket.

        Args:
            context: Optional existing ZMQ context. Creates new one if None.

        Returns:
            True if initialization succeeded.
        """
        if zmq is None:
            logger.error("pyzmq not available. Command publisher disabled.")
            return False

        try:
            if context is None:
                self._context = zmq.Context()
            else:
                self._context = context

            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger
            self._socket.bind(f"tcp://{self.host}:{self.port}")

            self._initialized = True
            logger.info("Command publisher bound to tcp://%s:%d", self.host, self.port)
            return True

        except (zmq.ZMQError, OSError) as e:
            logger.exception("Failed to initialize command publisher: %s", e)
            return False

    def publish(self, command: Command) -> bool:
        """Publish a command to subscribers.

        Args:
            command: The command to publish.

        Returns:
            True if command was sent successfully.
        """
        if not self._initialized or self._socket is None:
            logger.warning("Command publisher not initialized, cannot publish")
            return False

        try:
            # Send with topic prefix for filtering
            topic = f"cmd:{command.type}"
            message = f"{topic} {command.to_json()}"
            self._socket.send_string(message, zmq.NOBLOCK)
            logger.info("Published command: %s", command.type)
            return True
        except zmq.ZMQError as e:
            logger.exception("Failed to publish command: %s", e)
            return False

    def abort_experiment(self, reason: str = "User requested abort") -> bool:
        """Send an abort command to the experiment.

        Args:
            reason: Reason for the abort.

        Returns:
            True if command was sent successfully.
        """
        command = Command(type="abort", payload={"reason": reason})
        return self.publish(command)

    def close(self) -> None:
        """Close the socket and clean up resources."""
        if self._socket is not None:
            try:
                self._socket.close()
                logger.info("Command publisher socket closed")
            except zmq.ZMQError as e:
                logger.debug("Error closing command socket: %s", e)
            self._socket = None

        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if publisher is initialized and ready."""
        return self._initialized
