"""Orchestrates server startup and lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
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
    uvicorn = None  # type: ignore[assignment]


class ServerOrchestrator:
    """Orchestrates server startup and lifecycle."""

    @staticmethod
    def _create_command_publisher_and_register(
        config: ServerConfig,
        state_manager: ExperimentStateManager,
        zmq_context: Any,
    ) -> Any | None:
        """Create and register command publisher with state manager.

        Args:
            config: Server configuration
            state_manager: Experiment state manager
            zmq_context: ZMQ context

        Returns:
            Command publisher instance or None if initialization failed
        """
        command_publisher = ServerOrchestrator._create_command_publisher(
            config, zmq_context
        )
        state_manager.command_publisher = command_publisher
        return command_publisher

    @staticmethod
    def _create_liveview_server(
        state_manager: ExperimentStateManager, host: str, port: int
    ) -> Any:
        """Create LiveView server application and uvicorn server.

        Args:
            state_manager: Experiment state manager
            host: Server host
            port: Server port

        Returns:
            Uvicorn server instance
        """
        app = LiveViewServerSetup.create_app(state_manager)
        return ServerOrchestrator._create_server(app, host, port)

    @staticmethod
    def _start_background_tasks(
        zmq_context: Any,
        state_manager: ExperimentStateManager,
        config: ServerConfig,
        experiment_completed: asyncio.Event,
        server: Any,
    ) -> tuple[asyncio.Task[Any] | None, asyncio.Task[Any]]:
        """Start ZMQ subscriber and shutdown monitor background tasks.

        Args:
            zmq_context: ZMQ context
            state_manager: Experiment state manager
            config: Server configuration
            experiment_completed: Event to signal completion
            server: Uvicorn server instance

        Returns:
            Tuple of (zmq_task, shutdown_task)
        """
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

        return zmq_task, shutdown_task

    @staticmethod
    async def _handle_shutdown_interrupt(
        shutdown_task: asyncio.Task[Any] | None,
    ) -> None:
        """Handle KeyboardInterrupt by cancelling shutdown monitor task.

        Args:
            shutdown_task: Shutdown monitor task to cancel
        """
        logger.info("Server shutdown requested")
        if shutdown_task and not shutdown_task.done():
            shutdown_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await shutdown_task

    @staticmethod
    async def _cleanup_resources(
        command_publisher: Any | None,
        shutdown_task: asyncio.Task[Any] | None,
        zmq_task: asyncio.Task[Any] | None,
        zmq_context: Any,
    ) -> None:
        """Clean up all server resources.

        Args:
            command_publisher: Command publisher to close
            shutdown_task: Shutdown monitor task
            zmq_task: ZMQ subscriber task
            zmq_context: ZMQ context to cleanup
        """
        if command_publisher:
            command_publisher.close()
        await ZmqContextManager.cleanup_tasks(shutdown_task, zmq_task)
        await ZmqContextManager.cleanup_context(zmq_context)

    @staticmethod
    async def run_with_zmq(
        config: ServerConfig,
        state_manager: ExperimentStateManager,
    ) -> None:
        """Run server with ZMQ subscriber and command publisher.

        Orchestrates server lifecycle as a cohesive script:
        1. Initialize ZMQ context and events
        2. Set up command publisher
        3. Create LiveView server
        4. Start background tasks (ZMQ subscriber, shutdown monitor)
        5. Run server (handle interrupts)
        6. Clean up resources

        Args:
            config: Server configuration
            state_manager: Experiment state manager
        """
        # Phase 1: Initialize
        zmq_context = ZmqContextManager.create_context()
        experiment_completed = asyncio.Event()

        # Phase 2: Set up components
        command_publisher = ServerOrchestrator._create_command_publisher_and_register(
            config, state_manager, zmq_context
        )
        server = ServerOrchestrator._create_liveview_server(
            state_manager, config.host, config.port
        )
        zmq_task, shutdown_task = ServerOrchestrator._start_background_tasks(
            zmq_context, state_manager, config, experiment_completed, server
        )

        # Phase 3: Run server
        try:
            await ServerLifecycleManager.run_server(
                server, config.host, config.port, state_manager
            )
        except KeyboardInterrupt:
            await ServerOrchestrator._handle_shutdown_interrupt(shutdown_task)
        finally:
            # Phase 4: Cleanup
            await ServerOrchestrator._cleanup_resources(
                command_publisher, shutdown_task, zmq_task, zmq_context
            )

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
