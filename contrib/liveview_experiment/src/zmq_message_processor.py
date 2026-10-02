"""Processes ZMQ messages in a loop."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from .zmq_message_handler import ZmqMessageHandler

if TYPE_CHECKING:
    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment, unused-ignore]


class ZmqMessageProcessor:
    """Processes ZMQ messages in a loop."""

    @staticmethod
    async def process_messages(
        socket: Any,  # zmq.Socket
        state_manager: ExperimentStateManager,
        experiment_completed: asyncio.Event | None,
    ) -> None:
        """Process ZMQ messages in a loop.

        Args:
            socket: ZMQ socket to receive messages from
            state_manager: Experiment state manager
            experiment_completed: Optional event to signal completion
        """
        if zmq is None:
            return

        handler = ZmqMessageHandler(state_manager, experiment_completed)

        while True:
            try:
                message_bytes = socket.recv(zmq.NOBLOCK)
                message_str = message_bytes.decode("utf-8")
                payload = json.loads(message_str)
                await handler.process_message(payload)
            except zmq.Again:
                await asyncio.sleep(0.01)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse ZMQ message as JSON: %s", e)
            except Exception as e:
                logger.exception("Error processing ZMQ message: %s", e)

    @staticmethod
    async def close_socket(socket: Any) -> None:  # zmq.Socket
        """Close ZMQ socket gracefully.

        Args:
            socket: ZMQ socket to close
        """
        if not socket or zmq is None:
            return

        try:
            socket.close()
            await asyncio.sleep(0.1)
        except (zmq.ZMQError, AttributeError) as e:
            logger.debug("Error closing ZMQ socket: %s", e)
