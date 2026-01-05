"""Orchestrates server startup and lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from contrib.liveview_experiment.src.server_lifecycle import ServerLifecycleManager
from contrib.liveview_experiment.src.server_setup import LiveViewServerSetup
from contrib.liveview_experiment.src.zmq_context_manager import ZmqContextManager

if TYPE_CHECKING:
    import uvicorn

    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)

try:
    import uvicorn
except ImportError:
    uvicorn = None


class ServerOrchestrator:
    """Orchestrates server startup and lifecycle."""

    @staticmethod
    async def run_with_zmq(
        host: str,
        port: int,
        zmq_port: int,
        zmq_host: str,
        state_manager: ExperimentStateManager,
    ) -> None:
        """Run server with ZMQ subscriber.

        Args:
            host: Server host
            port: Server port
            zmq_port: ZMQ subscriber port
            zmq_host: ZMQ subscriber host
            state_manager: Experiment state manager
        """
        zmq_context = ZmqContextManager.create_context()
        experiment_completed = asyncio.Event()

        app = LiveViewServerSetup.create_app(state_manager)
        server = ServerOrchestrator._create_server(app, host, port)

        zmq_task = ServerOrchestrator._start_zmq_subscriber(
            zmq_context, state_manager, zmq_port, zmq_host, experiment_completed
        )

        shutdown_task = asyncio.create_task(
            ServerLifecycleManager.monitor_completion_and_shutdown(
                experiment_completed, state_manager, server
            )
        )

        try:
            await ServerLifecycleManager.run_server(server, host, port, state_manager)
        finally:
            await ZmqContextManager.cleanup_tasks(shutdown_task, zmq_task)
            await ZmqContextManager.cleanup_context(zmq_context)

    @staticmethod
    def _create_server(app: Any, host: str, port: int) -> Any:  # PyView, uvicorn.Server
        """Create Uvicorn server configuration.

        Args:
            app: PyView application
            host: Server host
            port: Server port

        Returns:
            Uvicorn server instance
        """
        if uvicorn is None:
            raise ImportError("uvicorn not available")

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            workers=1,
            access_log=False,
        )
        return uvicorn.Server(config)

    @staticmethod
    def _start_zmq_subscriber(
        zmq_context: Any | None,  # zmq.Context
        state_manager: ExperimentStateManager,
        zmq_port: int,
        zmq_host: str,
        experiment_completed: asyncio.Event,
    ) -> asyncio.Task[Any] | None:
        """Start ZMQ subscriber task.

        Args:
            zmq_context: ZMQ context
            state_manager: Experiment state manager
            zmq_port: ZMQ subscriber port
            zmq_host: ZMQ subscriber host
            experiment_completed: Event to signal completion

        Returns:
            ZMQ subscriber task or None
        """
        if not zmq_context:
            return None

        from contrib.liveview_experiment.src.liveview_server_standalone import (  # noqa: PLC0415
            run_zmq_subscriber,
        )

        return asyncio.create_task(
            run_zmq_subscriber(
                state_manager, zmq_context, zmq_port, zmq_host, experiment_completed
            )
        )
