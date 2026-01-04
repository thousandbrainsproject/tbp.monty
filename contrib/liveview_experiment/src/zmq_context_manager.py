"""ZMQ context and lifecycle management."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment, unused-ignore]


class ZmqContextManager:
    """Manages ZMQ context lifecycle and cleanup."""

    @staticmethod
    def create_context() -> Any | None:  # zmq.Context
        """Create ZMQ context.

        Returns:
            ZMQ context or None if zmq is not available
        """
        if zmq is None:
            logger.warning("pyzmq not available. ZMQ subscriber will not start.")
            return None

        try:
            zmq_context = zmq.Context()
            logger.info("ZMQ context created for server")
            return zmq_context
        except (zmq.ZMQError, RuntimeError) as e:
            logger.exception("Failed to create ZMQ context: %s", e)
            return None

    @staticmethod
    async def cleanup_context(zmq_context: Any | None) -> None:  # zmq.Context
        """Clean up ZMQ context.

        Args:
            zmq_context: ZMQ context to terminate
        """
        if not zmq_context:
            return

        try:
            await asyncio.sleep(0.1)  # Brief pause to ensure all operations complete
            zmq_context.term()
            logger.info("ZMQ context terminated")
        except (AttributeError, RuntimeError) as e:
            logger.debug("Error terminating ZMQ context: %s", e)

    @staticmethod
    async def cleanup_tasks(*tasks: asyncio.Task[Any] | None) -> None:
        """Cancel and await cleanup of async tasks.

        Args:
            *tasks: Tasks to cancel and await
        """
        for task in tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
