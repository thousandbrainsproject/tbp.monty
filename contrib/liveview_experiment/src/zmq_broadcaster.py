"""ZMQ-based broadcaster for cross-process communication.

This module provides a ZMQ publisher that can be used from the main experiment
process to send updates to the LiveView server running in a separate process.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    zmq = None  # type: ignore

logger = logging.getLogger(__name__)


class ZmqBroadcaster:
    """ZMQ publisher for sending experiment updates to LiveView server.
    
    This broadcaster connects to a ZMQ publisher socket and sends JSON-encoded
    messages that the LiveView server can subscribe to and display.
    """

    def __init__(
        self,
        zmq_port: int = 5555,
        zmq_host: str = "127.0.0.1"
    ) -> None:
        """Initialize ZMQ broadcaster.
        
        Args:
            zmq_port: Port for ZMQ publisher socket. Defaults to 5555.
            zmq_host: Host for ZMQ publisher socket. Defaults to 127.0.0.1.
        """
        if not ZMQ_AVAILABLE:
            logger.warning("pyzmq not available. Install with: pip install pyzmq")
            self._socket = None
            self._context = None
            return
        
        self.zmq_port = zmq_port
        self.zmq_host = zmq_host
        self._context: Optional[Any] = None
        self._socket: Optional[Any] = None
        self._connected = False
        
    def connect(self) -> None:
        """Connect to ZMQ publisher socket."""
        if not ZMQ_AVAILABLE or self._connected:
            return
        
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            # Set socket options for better performance
            # LINGER: Wait up to 1 second for pending messages to be sent before closing
            # This ensures final messages (like "completed" status) are delivered
            self._socket.setsockopt(zmq.LINGER, 1000)  # 1 second linger time
            self._socket.setsockopt(zmq.SNDHWM, 1000)  # High water mark for send queue
            
            # PUB connects (client side) - subscriber binds and waits for us
            # Retry connecting until subscriber is ready (handles slow joiner problem)
            import time
            max_retries = 10
            retry_delay = 0.5  # 500ms between retries
            connected = False
            
            for attempt in range(max_retries):
                try:
                    self._socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
                    # Small delay to ensure connection is established
                    time.sleep(0.1)
                    connected = True
                    logger.info(f"ZMQ broadcaster connected to tcp://{self.zmq_host}:{self.zmq_port} (attempt {attempt + 1})")
                    break
                except zmq.ZMQError as e:
                    if attempt < max_retries - 1:
                        logger.debug(f"ZMQ connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                        time.sleep(retry_delay)
                    else:
                        logger.warning(
                            f"Failed to connect to ZMQ subscriber after {max_retries} attempts. "
                            f"Subscriber may not be ready yet. Continuing anyway."
                        )
                        # Continue anyway - ZMQ will auto-reconnect when subscriber is ready
                        connected = True  # Mark as connected so we can try to send messages
            
            self._connected = connected
        except Exception as e:
            logger.error(f"Failed to connect ZMQ broadcaster: {e}", exc_info=True)
            self._socket = None
            self._context = None
            self._connected = False
    
    def publish(self, message_type: str, data: Dict[str, Any]) -> None:
        """Publish a message to the ZMQ socket.
        
        Args:
            message_type: Type of message (e.g., "metric", "data", "log", "state")
            data: Message data dictionary
        """
        if not self._connected or not self._socket:
            return
        
        try:
            payload = {
                "type": message_type,
                **data
            }
            message = json.dumps(payload).encode('utf-8')
            # Send JSON directly without topic prefix - message type is in the payload
            self._socket.send(message)
            logger.debug(f"ZMQ sent message type '{message_type}' (size: {len(message)} bytes): {json.dumps(payload)[:100]}...")
        except Exception as e:
            logger.error(f"Failed to publish ZMQ message: {e}", exc_info=True)
    
    def publish_metric(self, name: str, value: float, **metadata: Any) -> None:
        """Publish a metric update.
        
        Args:
            name: Metric name (e.g., "loss", "accuracy")
            value: Metric value
            **metadata: Additional metadata (e.g., epoch, step, episode)
        """
        self.publish("metric", {
            "name": name,
            "value": value,
            **metadata
        })
    
    def publish_data(self, stream_name: str, data: Dict[str, Any]) -> None:
        """Publish a data stream update.
        
        Args:
            stream_name: Name of the data stream
            data: Data dictionary
        """
        self.publish("data", {
            "stream": stream_name,
            **data
        })
    
    def publish_log(self, level: str, message: str, **metadata: Any) -> None:
        """Publish a log message.
        
        Args:
            level: Log level (e.g., "info", "warning", "error")
            message: Log message
            **metadata: Additional metadata
        """
        self.publish("log", {
            "level": level,
            "message": message,
            **metadata
        })
    
    def publish_state(self, state: Dict[str, Any]) -> None:
        """Publish a state update.
        
        Args:
            state: State dictionary
        """
        self.publish("state", state)
    
    def close(self) -> None:
        """Close ZMQ connection.
        
        Ensures the context lives longer than the experiment by:
        1. Closing the socket first (with LINGER timeout to send pending messages)
        2. Waiting a small moment for any final messages to be sent
        3. Then terminating the context
        """
        if not self._connected:
            return
        
        # Close socket first - LINGER setting ensures pending messages are sent
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        
        # Small delay to ensure final messages are sent before context termination
        # This is especially important for the "completed" or "error" status messages
        if self._context:
            try:
                import time
                time.sleep(0.1)  # Brief pause to allow final messages to be sent
            except Exception:
                pass
        
        # Terminate context only after socket is closed and messages are sent
        if self._context:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None
        
        self._connected = False
        logger.info("ZMQ broadcaster closed")

