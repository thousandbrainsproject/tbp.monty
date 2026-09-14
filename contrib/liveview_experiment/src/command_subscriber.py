"""ZMQ command subscriber for receiving commands from LiveView server.

The experiment uses this to listen for commands (like abort) from the
LiveView server running in a separate process.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class ReceivedCommand:
    """A command received from the LiveView server.

    Attributes:
        type: Command type (e.g., "abort", "pause", "resume").
        payload: Additional data for the command.
        timestamp: Unix timestamp when command was sent.
    """

    type: str
    payload: dict[str, Any]
    timestamp: float

    @classmethod
    def from_json(cls, json_str: str) -> ReceivedCommand | None:
        """Parse a command from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            Parsed command or None if parsing fails.
        """
        try:
            data = json.loads(json_str)
            return cls(
                type=data.get("type", "unknown"),
                payload=data.get("payload", {}),
                timestamp=data.get("timestamp", 0.0),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug("Failed to parse command: %s", e)
            return None


CommandHandler = Callable[[ReceivedCommand], None]


class CommandSubscriber:
    """ZMQ SUB socket for receiving commands from the LiveView server.

    The subscriber connects to the server's PUB socket to receive commands.
    Commands are processed in a background thread to avoid blocking the
    experiment's main loop.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5556) -> None:
        """Initialize the command subscriber.

        Args:
            host: Host to connect to.
            port: Port to connect to.
        """
        self.host = host
        self.port = port
        self._socket: Any = None
        self._context: Any = None
        self._owns_context = False
        self._initialized = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._handlers: dict[str, list[CommandHandler]] = {}
        self._abort_requested = threading.Event()

    def initialize(self, context: Any = None) -> bool:
        """Initialize the ZMQ socket.

        Args:
            context: Optional existing ZMQ context. Creates new one if None.

        Returns:
            True if initialization succeeded.
        """
        if zmq is None:
            logger.error("pyzmq not available. Command subscriber disabled.")
            return False

        try:
            if context is None:
                self._context = zmq.Context()
                self._owns_context = True
            else:
                self._context = context
                self._owns_context = False

            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.LINGER, 100)  # Quick cleanup
            self._socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout for polling
            self._socket.setsockopt_string(
                zmq.SUBSCRIBE, "cmd:"
            )  # Subscribe to commands
            self._socket.connect(f"tcp://{self.host}:{self.port}")

            self._initialized = True
            logger.info(
                "Command subscriber connected to tcp://%s:%d", self.host, self.port
            )
            return True

        except (zmq.ZMQError, OSError) as e:
            logger.exception("Failed to initialize command subscriber: %s", e)
            return False

    def register_handler(self, command_type: str, handler: CommandHandler) -> None:
        """Register a handler for a specific command type.

        Args:
            command_type: Type of command to handle (e.g., "abort").
            handler: Callback function to invoke when command is received.
        """
        if command_type not in self._handlers:
            self._handlers[command_type] = []
        self._handlers[command_type].append(handler)
        logger.debug("Registered handler for command type: %s", command_type)

    def start(self) -> bool:
        """Start the background thread to listen for commands.

        Returns:
            True if thread started successfully.
        """
        if not self._initialized:
            logger.warning("Cannot start: subscriber not initialized")
            return False

        if self._running:
            logger.debug("Subscriber already running")
            return True

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Command subscriber started")
        return True

    def stop(self) -> None:
        """Stop the background listening thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Command subscriber stopped")

    def _listen_loop(self) -> None:
        """Background loop to receive and process commands."""
        while self._running:
            self._receive_and_process_once()

    def _receive_and_process_once(self) -> None:
        """Receive a single command message (non-blocking) and process it."""
        try:
            # Non-blocking receive with timeout
            message = self._socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            # No message available, nothing to do
            return
        except zmq.ZMQError as e:
            if self._running:  # Only log if not shutting down
                logger.debug("ZMQ error in command listener: %s", e)
            return
        except (AttributeError, ValueError, TypeError) as e:
            logger.exception("Error in command listener: %s", e)
            return

        self._process_message(message)

    def _process_message(self, message: str) -> None:
        """Process a received message.

        Args:
            message: Raw message string in format "topic json_payload".
        """
        try:
            command = self._parse_command(message)
            if command is None:
                return

            self._handle_parsed_command(command)
        except (ValueError, TypeError, KeyError) as e:
            logger.exception("Error processing command message: %s", e)

    def _parse_command(self, message: str) -> ReceivedCommand | None:
        """Parse a raw message string into a ReceivedCommand."""
        # Split topic from payload
        parts = message.split(" ", 1)
        if len(parts) != 2:
            logger.debug("Invalid message format: %s", message[:50])
            return None

        _topic, json_payload = parts
        return ReceivedCommand.from_json(json_payload)

    def _handle_parsed_command(self, command: ReceivedCommand) -> None:
        """Handle a parsed command (set flags and call handlers)."""
        logger.info("Received command: %s", command.type)

        # Set abort flag for quick checking
        if command.type == "abort":
            self._abort_requested.set()

        # Invoke registered handlers
        handlers = self._handlers.get(command.type, [])
        for handler in handlers:
            try:
                handler(command)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.exception("Handler error for %s: %s", command.type, e)

    def is_abort_requested(self) -> bool:
        """Check if an abort command has been received.

        Returns:
            True if abort was requested.
        """
        return self._abort_requested.is_set()

    def clear_abort(self) -> None:
        """Clear the abort request flag."""
        self._abort_requested.clear()

    def close(self) -> None:
        """Close the socket and clean up resources."""
        self.stop()

        if self._socket is not None:
            try:
                self._socket.close()
                logger.info("Command subscriber socket closed")
            except zmq.ZMQError as e:
                logger.debug("Error closing command socket: %s", e)
            self._socket = None

        if self._owns_context and self._context is not None:
            try:
                self._context.term()
            except zmq.ZMQError as e:
                logger.debug("Error terminating context: %s", e)
            self._context = None

        self._initialized = False

    @property
    def is_running(self) -> bool:
        """Check if the subscriber is running."""
        return self._running
