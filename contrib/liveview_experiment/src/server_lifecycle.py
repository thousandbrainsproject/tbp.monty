"""Server lifecycle management including startup and shutdown."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)


class ServerLifecycleManager:
    """Manages server lifecycle including startup, monitoring, and shutdown."""

    @staticmethod
    async def _wait_indefinitely_for_aborted() -> None:
        """Wait indefinitely for aborted experiments.

        Allows users to inspect the final state before manually stopping.
        """
        logger.info(
            "Experiment was aborted. LiveView server will continue running "
            "indefinitely. Press Ctrl+C to stop manually."
        )
        try:
            while True:
                await asyncio.sleep(3600)  # Sleep in 1-hour chunks
        except asyncio.CancelledError:
            logger.info("Shutdown monitor cancelled")

    @staticmethod
    async def _shutdown_after_linger(
        status: str, server: Any, linger_seconds: int
    ) -> None:
        """Shut down server after linger period.

        Args:
            status: Experiment status
            server: Uvicorn server instance
            linger_seconds: Seconds to wait before shutdown
        """
        status_msg = "errored" if status == "error" else "completed"
        logger.info(
            "Experiment %s. LiveView will linger for %d seconds before shutdown...",
            status_msg,
            linger_seconds,
        )
        await asyncio.sleep(linger_seconds)
        logger.info(
            "%d second linger period complete. Shutting down LiveView server...",
            linger_seconds,
        )
        server.should_exit = True

    @staticmethod
    async def monitor_completion_and_shutdown(
        experiment_completed: asyncio.Event,
        state_manager: ExperimentStateManager,
        server: Any,  # uvicorn.Server
        linger_seconds: int = 60,
    ) -> None:
        """Wait for experiment completion, then shut down after linger period.

        The server will only shut down automatically for "completed" or "error"
        statuses. If the experiment is "aborted", the server will continue running
        indefinitely so users can inspect the final state.

        Args:
            experiment_completed: Event that signals when experiment completes
            state_manager: Experiment state manager
            server: Uvicorn server instance
            linger_seconds: Seconds to wait before shutdown after completion
        """
        await experiment_completed.wait()
        status = state_manager.experiment_state.status

        # Never auto-shutdown for aborted experiments - let users inspect the state
        if status in ("aborted", "aborting"):
            await ServerLifecycleManager._wait_indefinitely_for_aborted()
            return

        await ServerLifecycleManager._shutdown_after_linger(
            status, server, linger_seconds
        )

    @staticmethod
    async def _close_all_connections(state_manager: ExperimentStateManager) -> None:
        """Close all WebSocket connections gracefully.

        Args:
            state_manager: Experiment state manager with registered sockets
        """
        if not state_manager.connected_sockets:
            return

        socket_count = len(state_manager.connected_sockets)
        logger.info("Closing %d WebSocket connection(s)...", socket_count)

        # Create a copy of the set to avoid modification during iteration
        sockets_to_close = list(state_manager.connected_sockets)
        for socket in sockets_to_close:
            try:
                # Unregister and let pyview handle the disconnection
                state_manager.unregister_socket(socket)
            except (AttributeError, RuntimeError, TypeError) as e:
                logger.debug("Error closing socket: %s", e)

        # Small delay to allow WebSocket close frames to be sent
        await asyncio.sleep(0.1)

    @staticmethod
    async def run_server(
        server: Any,  # uvicorn.Server
        host: str,
        port: int,
        state_manager: ExperimentStateManager,
    ) -> None:
        """Run the web server with logging.

        Args:
            server: Uvicorn server instance
            host: Server host
            port: Server port
            state_manager: Experiment state manager
        """
        logger.info("Starting LiveView server on http://%s:%d", host, port)
        logger.info("Listening to pub/sub topic: %s", state_manager.broadcast_topic)

        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            # Gracefully close all WebSocket connections before shutdown
            await ServerLifecycleManager._close_all_connections(state_manager)
