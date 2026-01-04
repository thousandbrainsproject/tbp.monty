#!/usr/bin/env python3
"""Standalone LiveView server that can run in a separate Python 3.11+ process.

This allows the main experiment to run in Python 3.8 while the LiveView
server runs in Python 3.11+ with pyview-web.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

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
    from pyview import PyView
    import uvicorn
except ImportError as e:
    print(f"Error: pyview-web not available: {e}", file=sys.stderr)
    print("This LiveView server requires Python >= 3.11 and pyview-web", file=sys.stderr)
    print("Please run setup.sh to install dependencies in the LiveView virtual environment", file=sys.stderr)
    sys.exit(1)

# Import must happen after pyview is available (Python 3.11+)
# The liveview_experiment module will import pyview, which is fine here
try:
    from contrib.liveview_experiment.src.liveview_experiment import ExperimentLiveView
    from contrib.liveview_experiment.src.state_manager import ExperimentStateManager
    from contrib.liveview_experiment.src.static_file_server import StaticFileServer
except ImportError as e:
    print(f"ERROR: Failed to import LiveView modules: {e}", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print(f"CONTRIB_DIR: {CONTRIB_DIR}", file=sys.stderr)
    print(f"TBP_MONTY_ROOT: {TBP_MONTY_ROOT}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def handle_pubsub_request(request: Any) -> Any:
    """Handle HTTP pub/sub requests from main process."""
    try:
        import json
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
    except Exception as e:
        logger.error(f"Error handling pub/sub request: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


async def run_server(host: str, port: int, state_manager: Any = None) -> None:
    """Run the LiveView server.
    
    Args:
        host: Host to bind to.
        port: Port to bind to.
        state_manager: Optional shared state manager instance. If None, creates a new one.
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
        from starlette.routing import Route
        from starlette.responses import JSONResponse
        
        async def pubsub_endpoint(request: Any) -> Any:
            """Handle HTTP pub/sub requests."""
            try:
                import json
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
                return JSONResponse({"status": "error", "message": "Missing topic or payload"}, status_code=400)
            except Exception as e:
                logger.error(f"Error handling pub/sub request: {e}", exc_info=True)
                return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
        
        # Add route to app (PyView is based on Starlette)
        app.routes.append(Route("/api/pubsub", pubsub_endpoint, methods=["POST"]))
        logger.info("HTTP pub/sub endpoint registered at /api/pubsub")
    except Exception as e:
        logger.warning(f"Could not register HTTP pub/sub endpoint: {e}")
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        workers=1,
        access_log=False,
    )
    server = uvicorn.Server(config)
    
    logger.info(f"Starting LiveView server on http://{host}:{port}")
    logger.info(f"Listening to pub/sub topic: {state_manager.broadcast_topic}")
    await server.serve()


async def run_zmq_subscriber(
    state_manager: Any, 
    zmq_context: Any, 
    zmq_port: int, 
    zmq_host: str
) -> None:
    """Run ZMQ subscriber to receive messages from experiment process.
    
    Subscribes to all ZMQ messages, parses JSON payloads, aggregates state,
    and broadcasts updates to connected LiveView clients via PyView's pubsub.
    
    Args:
        state_manager: ExperimentStateManager instance
        zmq_context: ZMQ context (must live as long as the server)
        zmq_port: ZMQ subscriber port
        zmq_host: ZMQ subscriber host
    """
    try:
        import zmq
        import json
    except ImportError:
        logger.error("pyzmq not available. ZMQ subscriber will not start.")
        return
    
    if zmq_context is None:
        logger.error("ZMQ context not provided. ZMQ subscriber will not start.")
        return
    
    # Create socket from the provided context (context lives as long as server)
    socket = zmq_context.socket(zmq.SUB)
    # Set LINGER to ensure messages are received before socket closes
    socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger time
    
    # Subscribe to ALL messages (empty string = all messages)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    # SUB binds (server side) - publishers will connect to us
    socket.bind(f"tcp://{zmq_host}:{zmq_port}")
    
    # Small delay to ensure binding is complete before publishers connect
    await asyncio.sleep(0.2)
    
    logger.info(f"ZMQ subscriber bound to tcp://{zmq_host}:{zmq_port}, subscribed to all messages, waiting for publishers")
    
    try:
        while True:
            try:
                # Receive message as bytes (no topic prefix - just JSON)
                message_bytes = socket.recv(zmq.NOBLOCK)
                message_str = message_bytes.decode('utf-8')
                
                try:
                    payload = json.loads(message_str)
                    
                    # Extract message type from payload (not from topic)
                    msg_type = payload.get("type", "unknown")
                    
                    # Route to appropriate handler based on message type
                    if msg_type == "metric":
                        # Metric payload: {"type": "metric", "name": ..., "value": ..., ...metadata}
                        state_manager._handle_metric_message(state_manager.metrics_topic, payload)
                        await state_manager.broadcast_update()
                    elif msg_type == "data":
                        # Data payload: {"type": "data", "stream": ..., ...data}
                        data_payload = {
                            "type": "data",
                            "stream": payload.get("stream", "unknown"),
                            "data": {k: v for k, v in payload.items() if k not in ("type", "stream")}
                        }
                        state_manager._handle_data_message(state_manager.data_topic, data_payload)
                        await state_manager.broadcast_update()
                    elif msg_type == "log":
                        # Log payload: {"type": "log", "level": ..., "message": ..., ...metadata}
                        state_manager._handle_log_message(state_manager.logs_topic, payload)
                        await state_manager.broadcast_update()
                    elif msg_type == "state":
                        # State payload: direct state updates
                        # Update state directly
                        if isinstance(payload, dict):
                            from datetime import datetime, timezone
                            updated_keys = []
                            for key, value in payload.items():
                                if key == "type":
                                    continue  # Skip the type field
                                if hasattr(state_manager.experiment_state, key):
                                    # Handle datetime string conversion
                                    if key in ("experiment_start_time", "last_update") and isinstance(value, str):
                                        try:
                                            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                                        except (ValueError, AttributeError):
                                            logger.warning(f"Failed to parse datetime for {key}: {value}")
                                            continue
                                    setattr(state_manager.experiment_state, key, value)
                                    updated_keys.append(key)
                            # Update last_update timestamp
                            state_manager.experiment_state.last_update = datetime.now(timezone.utc)
                            # Broadcast update to web UI
                            await state_manager.broadcast_update()
                    else:
                        logger.warning(f"Unknown message type: {msg_type}, ignoring")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse ZMQ message as JSON: {e}")
            except zmq.Again:
                # No message available, yield to event loop
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing ZMQ message: {e}", exc_info=True)
    finally:
        # Close socket only - context is managed at server level
        if socket:
            try:
                socket.close()
                # Small delay to ensure any pending operations complete
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.debug(f"Error closing ZMQ socket: {e}")
        logger.info("ZMQ subscriber closed")


def main() -> None:
    """Main entry point for standalone server."""
    parser = argparse.ArgumentParser(description="Standalone LiveView server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ subscriber port")
    parser.add_argument("--zmq-host", default="127.0.0.1", help="ZMQ subscriber host")
    
    args = parser.parse_args()
    
    async def run_with_zmq() -> None:
        """Run server with ZMQ subscriber.
        
        The ZMQ context is created at server startup and lives for the entire
        server lifetime, ensuring it persists longer than the subscriber task.
        """
        # Create ZMQ context at server level (lives as long as server)
        zmq_context = None
        try:
            import zmq
            zmq_context = zmq.Context()
            logger.info("ZMQ context created for server")
        except ImportError:
            logger.warning("pyzmq not available. ZMQ subscriber will not start.")
        except Exception as e:
            logger.error(f"Failed to create ZMQ context: {e}")
        
        # Create state manager
        route_path = "/"
        state_manager = ExperimentStateManager(route_path=route_path)
        
        # Start ZMQ subscriber in background (uses server-level context)
        zmq_task = None
        if zmq_context:
            zmq_task = asyncio.create_task(
                run_zmq_subscriber(state_manager, zmq_context, args.zmq_port, args.zmq_host)
            )
        
        try:
            # Run the web server with the shared state manager
            await run_server(args.host, args.port, state_manager)
        finally:
            # Cancel ZMQ task when server stops
            if zmq_task:
                zmq_task.cancel()
                try:
                    await zmq_task
                except asyncio.CancelledError:
                    pass
            
            # Terminate ZMQ context only after server and subscriber have stopped
            if zmq_context:
                try:
                    await asyncio.sleep(0.1)  # Brief pause to ensure all operations complete
                    zmq_context.term()
                    logger.info("ZMQ context terminated")
                except Exception as e:
                    logger.debug(f"Error terminating ZMQ context: {e}")
    
    try:
        asyncio.run(run_with_zmq())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

