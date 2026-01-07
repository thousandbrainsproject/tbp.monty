"""Orchestrates server startup and lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from contrib.liveview_experiment.src.command_publisher import CommandPublisher
from contrib.liveview_experiment.src.server_lifecycle import ServerLifecycleManager
from contrib.liveview_experiment.src.server_setup import LiveViewServerSetup
from contrib.liveview_experiment.src.zmq_context_manager import ZmqContextManager

from .experiment_config import (
    ServerConfig,
    ZmqSubscriberParams,
    ZmqSubscriberRunParams,
)

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
        config: ServerConfig,
        state_manager: ExperimentStateManager,
    ) -> None:
        """Run server with ZMQ subscriber and command publisher.

        Args:
            config: Server configuration
            state_manager: Experiment state manager
        """
        zmq_context = ZmqContextManager.create_context()
        experiment_completed = asyncio.Event()

        # Initialize command publisher for sending commands back to experiment
        command_publisher = ServerOrchestrator._create_command_publisher(
            config, zmq_context
        )
        state_manager.command_publisher = command_publisher

        app = LiveViewServerSetup.create_app(state_manager)
        server = ServerOrchestrator._create_server(app, config.host, config.port)

        zmq_params = ZmqSubscriberParams(
            zmq_port=config.zmq_port, zmq_host=config.zmq_host
        )
        zmq_task = ServerOrchestrator._start_zmq_subscriber(
            zmq_context, state_manager, zmq_params, experiment_completed
        )

        shutdown_task = asyncio.create_task(
            ServerLifecycleManager.monitor_completion_and_shutdown(
                experiment_completed, state_manager, server
            )
        )

        try:
            await ServerLifecycleManager.run_server(
                server, config.host, config.port, state_manager
            )
        finally:
            if command_publisher:
                command_publisher.close()
            await ZmqContextManager.cleanup_tasks(shutdown_task, zmq_task)
            await ZmqContextManager.cleanup_context(zmq_context)

    @staticmethod
    def _create_command_publisher(
        config: ServerConfig,
        zmq_context: Any,
    ) -> CommandPublisher | None:
        """Create and initialize command publisher.

        Args:
            config: Server configuration
            zmq_context: ZMQ context

        Returns:
            Initialized command publisher, or None if initialization failed.
        """
        # Command port is subscriber port + 1 by convention
        command_port = config.zmq_port + 1
        publisher = CommandPublisher(host=config.zmq_host, port=command_port)

        if publisher.initialize(zmq_context):
            logger.info(
                "Command publisher ready on port %d (subscriber on %d)",
                command_port,
                config.zmq_port,
            )
            return publisher

        logger.warning("Command publisher failed to initialize")
        return None

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
        params: ZmqSubscriberParams,
        experiment_completed: asyncio.Event,
    ) -> asyncio.Task[Any] | None:
        """Start ZMQ subscriber task.

        Args:
            zmq_context: ZMQ context
            state_manager: Experiment state manager
            params: ZMQ subscriber parameters
            experiment_completed: Event to signal completion

        Returns:
            ZMQ subscriber task or None
        """
        if not zmq_context:
            return None

        from contrib.liveview_experiment.src.liveview_server_standalone import (  # noqa: PLC0415
            run_zmq_subscriber,
        )

        run_params = ZmqSubscriberRunParams(
            zmq_context=zmq_context,
            zmq_port=params.zmq_port,
            zmq_host=params.zmq_host,
            experiment_completed=experiment_completed,
        )
        return asyncio.create_task(run_zmq_subscriber(state_manager, run_params))
