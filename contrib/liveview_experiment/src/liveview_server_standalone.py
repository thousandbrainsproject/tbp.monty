#!/usr/bin/env python3
"""Standalone LiveView server that can run in a separate Python 3.11+ process.

This allows the main experiment to run in Python 3.8 while the LiveView
server runs in Python 3.11+ with pyview-web.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore[assignment, unused-ignore]

from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

# Add the contrib directory to the path so we can import
CONTRIB_DIR = Path(__file__).parent.parent.parent
TBP_MONTY_ROOT = CONTRIB_DIR.parent
# Add both the project root and contrib to path
sys.path.insert(0, str(TBP_MONTY_ROOT))
sys.path.insert(0, str(CONTRIB_DIR))
# Also add src if it exists (for tbp.monty imports)
if (TBP_MONTY_ROOT / "src").exists():
    sys.path.insert(0, str(TBP_MONTY_ROOT / "src"))

try:
    import uvicorn
    from pyview import PyView
except ImportError as e:
    print(f"Error: pyview-web not available: {e}", file=sys.stderr)
    print(
        "This LiveView server requires Python >= 3.11 and pyview-web", file=sys.stderr
    )
    print(
        "Please run setup.sh to install dependencies in the LiveView "
        "virtual environment",
        file=sys.stderr,
    )
    sys.exit(1)

# Import must happen after pyview is available (Python 3.11+)
# The liveview_experiment module will import pyview, which is fine here
# These are conditional imports (inside try block), so PLC0415 doesn't apply
try:
    from contrib.liveview_experiment.src.liveview_experiment import (
        ExperimentLiveView,
    )
    from contrib.liveview_experiment.src.state_manager import (
        ExperimentStateManager,
    )
    from contrib.liveview_experiment.src.static_file_server import (
        StaticFileServer,
    )
    from contrib.liveview_experiment.src.zmq_message_handler import (
        ZmqMessageHandler,
    )
except ImportError as e:
    print(f"ERROR: Failed to import LiveView modules: {e}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print(f"CONTRIB_DIR: {CONTRIB_DIR}", file=sys.stderr)
    print(f"TBP_MONTY_ROOT: {TBP_MONTY_ROOT}", file=sys.stderr)
    import traceback

    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Configure logging: DEBUG for our modules, INFO for everything else (including pyview)
logging.basicConfig(level=logging.INFO)
# Allow DEBUG for our modules only
logging.getLogger("contrib.liveview_experiment").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


async def handle_pubsub_request(request: Request) -> dict[str, str]:
    """Handle HTTP pub/sub requests from main process."""
    try:
        body = await request.json()
        topic = body.get("topic")
        payload = body.get("payload")

        if topic and payload:
            # Route to appropriate handler
            state_manager = request.app.state.state_manager
            if topic == state_manager.metrics_topic:
                state_manager._handle_metric_message(topic, payload)
            elif topic == state_manager.data_topic:
                state_manager._handle_data_message(topic, payload)
            elif topic == state_manager.logs_topic:
                state_manager._handle_log_message(topic, payload)

            return {"status": "ok"}
        return {"status": "error", "message": "Missing topic or payload"}
    except Exception as e:
        logger.exception("Error handling pub/sub request: %s", e)
        return {"status": "error", "message": str(e)}


async def run_server(host: str, port: int, state_manager: Any = None) -> None:
    """Run the LiveView server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        state_manager: Optional shared state manager instance.
            If None, creates a new one.
    """
    # Use provided state manager or create a new one
    if state_manager is None:
        route_path = "/"
        state_manager = ExperimentStateManager(route_path=route_path)

    app = PyView()

    # Configure static file serving (following mvg_departures approach)
    static_file_server = StaticFileServer()
    static_file_server.register_routes(app)

    # Create a factory function that returns the LiveView instance
    # Store the instance so we can use it for broadcasting
    live_view_instance = None

    def create_live_view() -> ExperimentLiveView:
        nonlocal live_view_instance
        live_view_instance = ExperimentLiveView(state_manager)
        # Ensure state_manager has reference to this instance
        state_manager.liveview_instance = live_view_instance
        return live_view_instance

    app.add_live_view("/", create_live_view)

    # Store state_manager for HTTP pub/sub endpoint
    app.state.state_manager = state_manager

    # Add HTTP endpoint for cross-process pub/sub
    try:

        async def pubsub_endpoint(request: Any) -> Any:
            """Handle HTTP pub/sub requests."""
            try:
                body = await request.json()
                topic = body.get("topic")
                payload = body.get("payload")

                if topic and payload:
                    # Route to appropriate handler
                    if topic == state_manager.metrics_topic:
                        state_manager._handle_metric_message(topic, payload)
                    elif topic == state_manager.data_topic:
                        state_manager._handle_data_message(topic, payload)
                    elif topic == state_manager.logs_topic:
                        state_manager._handle_log_message(topic, payload)

                    return JSONResponse({"status": "ok"})
                return JSONResponse(
                    {"status": "error", "message": "Missing topic or payload"},
                    status_code=400,
                )
            except Exception as e:
                logger.exception("Error handling pub/sub request: %s", e)
                return JSONResponse(
                    {"status": "error", "message": str(e)}, status_code=500
                )

        # Add route to app (PyView is based on Starlette)
        app.routes.append(Route("/api/pubsub", pubsub_endpoint, methods=["POST"]))
        logger.info("HTTP pub/sub endpoint registered at /api/pubsub")
    except (ImportError, AttributeError) as e:
        logger.warning("Could not register HTTP pub/sub endpoint: %s", e)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        workers=1,
        access_log=False,
    )
    server = uvicorn.Server(config)

    logger.info("Starting LiveView server on http://%s:%d", host, port)
    logger.info("Listening to pub/sub topic: %s", state_manager.broadcast_topic)
    await server.serve()


def _create_zmq_socket(zmq_context: Any, zmq_host: str, zmq_port: int) -> Any:
    """Create and configure ZMQ subscriber socket.

    Args:
        zmq_context: ZMQ context
        zmq_host: ZMQ subscriber host
        zmq_port: ZMQ subscriber port

    Returns:
        Configured ZMQ socket
    """
    socket = zmq_context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger time
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
    socket.bind(f"tcp://{zmq_host}:{zmq_port}")
    return socket


async def run_zmq_subscriber(
    state_manager: ExperimentStateManager,
    zmq_context: Any,  # zmq.Context type not available if zmq is None
    zmq_port: int,
    zmq_host: str,
    experiment_completed: asyncio.Event | None = None,
) -> None:
    """Run ZMQ subscriber to receive messages from experiment process.

    Subscribes to all ZMQ messages, parses JSON payloads, aggregates state,
    and broadcasts updates to connected LiveView clients via PyView's pubsub.

    Args:
        state_manager: ExperimentStateManager instance
        zmq_context: ZMQ context (must live as long as the server)
        zmq_port: ZMQ subscriber port
        zmq_host: ZMQ subscriber host
        experiment_completed: Optional asyncio.Event to signal when experiment completes
    """
    if zmq is None:
        logger.error("pyzmq not available. ZMQ subscriber will not start.")
        return

    if zmq_context is None:
        logger.error("ZMQ context not provided. ZMQ subscriber will not start.")
        return

    socket = _create_zmq_socket(zmq_context, zmq_host, zmq_port)
    await asyncio.sleep(0.2)  # Ensure binding is complete

    logger.info(
        "ZMQ subscriber bound to tcp://%s:%d, subscribed to all messages, "
        "waiting for publishers",
        zmq_host,
        zmq_port,
    )

    try:
        while True:
            try:
                message_bytes = socket.recv(zmq.NOBLOCK)
                message_str = message_bytes.decode("utf-8")
                payload = json.loads(message_str)
                handler = ZmqMessageHandler(state_manager, experiment_completed)
                await handler.process_message(payload)
            except zmq.Again:
                # No message available, yield to event loop
                await asyncio.sleep(0.01)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse ZMQ message as JSON: %s", e)
            except Exception as e:
                logger.exception("Error processing ZMQ message: %s", e)
    finally:
        if socket:
            try:
                socket.close()
                await asyncio.sleep(0.1)
            except (zmq.ZMQError, AttributeError) as e:
                logger.debug("Error closing ZMQ socket: %s", e)
        logger.info("ZMQ subscriber closed")


def main() -> None:
    """Main entry point for standalone server."""
    parser = argparse.ArgumentParser(description="Standalone LiveView server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--zmq-port", type=int, default=5555, help="ZMQ subscriber port"
    )
    parser.add_argument("--zmq-host", default="127.0.0.1", help="ZMQ subscriber host")

    args = parser.parse_args()

    async def run_with_zmq() -> None:
        """Run server with ZMQ subscriber.

        The ZMQ context is created at server startup and lives for the entire
        server lifetime, ensuring it persists longer than the subscriber task.
        """
        # Create ZMQ context at server level (lives as long as server)
        zmq_context = None
        if zmq is not None:
            try:
                zmq_context = zmq.Context()
                logger.info("ZMQ context created for server")
            except Exception as e:
                logger.exception("Failed to create ZMQ context: %s", e)
        else:
            logger.warning("pyzmq not available. ZMQ subscriber will not start.")

        # Create state manager
        route_path = "/"
        state_manager = ExperimentStateManager(route_path=route_path)

        # Create event to track experiment completion
        experiment_completed = asyncio.Event()

        # Create uvicorn server config and instance
        app = PyView()

        # Configure static file serving
        static_file_server = StaticFileServer()
        static_file_server.register_routes(app)

        # Create LiveView instance
        live_view_instance = None

        def create_live_view() -> Any:
            nonlocal live_view_instance
            live_view_instance = ExperimentLiveView(state_manager)
            state_manager.liveview_instance = live_view_instance
            return live_view_instance

        app.add_live_view("/", create_live_view)
        app.state.state_manager = state_manager

        # Add HTTP endpoint for cross-process pub/sub
        try:

            async def pubsub_endpoint(request: Request) -> JSONResponse:
                """Handle HTTP pub/sub requests."""
                try:
                    body = await request.json()
                    topic = body.get("topic")
                    payload = body.get("payload")

                    if topic and payload:
                        if topic == state_manager.metrics_topic:
                            state_manager._handle_metric_message(topic, payload)
                        elif topic == state_manager.data_topic:
                            state_manager._handle_data_message(topic, payload)
                        elif topic == state_manager.logs_topic:
                            state_manager._handle_log_message(topic, payload)

                        return JSONResponse({"status": "ok"})
                    return JSONResponse(
                        {"status": "error", "message": "Missing topic or payload"},
                        status_code=400,
                    )
                except Exception as e:
                    logger.exception("Error handling pub/sub request: %s", e)
                    return JSONResponse(
                        {"status": "error", "message": str(e)}, status_code=500
                    )

            app.routes.append(Route("/api/pubsub", pubsub_endpoint, methods=["POST"]))
            logger.info("HTTP pub/sub endpoint registered at /api/pubsub")
        except (ImportError, AttributeError) as e:
            logger.warning("Could not register HTTP pub/sub endpoint: %s", e)

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            workers=1,
            access_log=False,
        )
        server = uvicorn.Server(config)

        # Start ZMQ subscriber in background (uses server-level context)
        zmq_task = None
        if zmq_context:
            zmq_task = asyncio.create_task(
                run_zmq_subscriber(
                    state_manager,
                    zmq_context,
                    args.zmq_port,
                    args.zmq_host,
                    experiment_completed,
                )
            )

        # Create a task to monitor for experiment completion
        # and shut down after 1 minute
        shutdown_task = None

        async def monitor_completion_and_shutdown() -> None:
            """Wait for experiment completion or error, then wait 1 minute.

            Shuts down the server after the experiment completes or errors.
            """
            await experiment_completed.wait()
            status = state_manager.experiment_state.status
            if status == "error":
                logger.info(
                    "Experiment errored. LiveView will linger for 1 minute "
                    "before shutdown..."
                )
            else:
                logger.info(
                    "Experiment completed. LiveView will linger for 1 minute "
                    "before shutdown..."
                )
            await asyncio.sleep(60)  # Wait 1 minute
            logger.info(
                "1 minute linger period complete. Shutting down LiveView server..."
            )
            # Stop the uvicorn server
            server.should_exit = True

        shutdown_task = asyncio.create_task(monitor_completion_and_shutdown())

        try:
            # Run the web server
            logger.info(
                "Starting LiveView server on http://%s:%d", args.host, args.port
            )
            logger.info("Listening to pub/sub topic: %s", state_manager.broadcast_topic)
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        finally:
            # Cancel shutdown task if still running
            if shutdown_task and not shutdown_task.done():
                shutdown_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await shutdown_task

            # Cancel ZMQ task when server stops
            if zmq_task:
                zmq_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await zmq_task

            # Terminate ZMQ context only after server and subscriber have stopped
            if zmq_context:
                try:
                    await asyncio.sleep(
                        0.1
                    )  # Brief pause to ensure all operations complete
                    zmq_context.term()
                    logger.info("ZMQ context terminated")
                except (AttributeError, RuntimeError) as e:
                    logger.debug("Error terminating ZMQ context: %s", e)

    try:
        asyncio.run(run_with_zmq())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.exception("Server error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
