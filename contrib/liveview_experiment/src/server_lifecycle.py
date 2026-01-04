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
    async def monitor_completion_and_shutdown(
        experiment_completed: asyncio.Event,
        state_manager: ExperimentStateManager,
        server: Any,  # uvicorn.Server
        linger_seconds: int = 60,
    ) -> None:
        """Wait for experiment completion, then shut down after linger period.

        Args:
            experiment_completed: Event that signals when experiment completes
            state_manager: Experiment state manager
            server: Uvicorn server instance
            linger_seconds: Seconds to wait before shutdown after completion
        """
        await experiment_completed.wait()
        status = state_manager.experiment_state.status

        if status == "error":
            logger.info(
                "Experiment errored. LiveView will linger for %d seconds "
                "before shutdown...",
                linger_seconds,
            )
        else:
            logger.info(
                "Experiment completed. LiveView will linger for %d seconds "
                "before shutdown...",
                linger_seconds,
            )

        await asyncio.sleep(linger_seconds)
        logger.info(
            "%d second linger period complete. Shutting down LiveView server...",
            linger_seconds,
        )
        server.should_exit = True

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
