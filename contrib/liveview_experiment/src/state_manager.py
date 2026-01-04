"""State manager for experiment LiveView."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

try:
    # Use PyView's actual pubsub system (Python 3.11+)
    from pyview.live_socket import pub_sub_hub
    from pyview.vendor.flet.pubsub import PubSub
except ImportError:
    # Fallback to our custom pubsub for Python 3.8
    from .pubsub_compat import PubSub, pub_sub_hub

from .experiment_state import ExperimentState
from .types import MessagePayload, MetricData  # noqa: TC001

if TYPE_CHECKING:
    from pyview import LiveViewSocket

logger = logging.getLogger(__name__)


class ExperimentStateManager:
    """Manages shared state for experiment LiveView and updates."""

    def __init__(self, route_path: str = "/") -> None:
        """Initialize the state manager.

        Args:
            route_path: The path for this route, used to create a unique topic.
        """
        self.route_path = route_path
        # Initialize with default values so LiveView isn't empty
        self.experiment_state = ExperimentState(
            run_name="Experiment",
            status="initializing",
            experiment_mode="train",
        )
        self.connected_sockets: set[LiveViewSocket[ExperimentState]] = set()
        self.liveview_instance: Any = (
            None  # Reference to LiveView instance for direct handle_info calls
        )
        # Create route-specific topic based on path
        normalized_path = route_path.strip("/").replace("/", ":") or "root"
        self.broadcast_topic: str = f"experiment:updates:{normalized_path}"
        # Sub-topics for different data streams
        self.metrics_topic: str = f"{self.broadcast_topic}:metrics"
        self.data_topic: str = f"{self.broadcast_topic}:data"
        self.logs_topic: str = f"{self.broadcast_topic}:logs"

        # Throttling for broadcasts (max once per second)
        self._last_broadcast_time: float = 0.0
        self._broadcast_throttle_seconds: float = 1.0
        self._pending_broadcast: bool = False

        # Subscribe to pub/sub topics to auto-update state from broadcaster
        self._subscribe_to_topics()

    def _subscribe_to_topics(self) -> None:
        """Subscribe to pub/sub topics to auto-update state."""
        # Subscribe to metrics topic
        metrics_pubsub = PubSub(pub_sub_hub, self.metrics_topic)
        metrics_pubsub.subscribe(self._handle_metric_message)

        # Subscribe to data topic
        data_pubsub = PubSub(pub_sub_hub, self.data_topic)
        data_pubsub.subscribe(self._handle_data_message)

        # Subscribe to logs topic
        logs_pubsub = PubSub(pub_sub_hub, self.logs_topic)
        logs_pubsub.subscribe(self._handle_log_message)

    def _extract_metric_data(self, payload: MessagePayload) -> MetricData | None:
        """Extract metric data from payload.

        Returns:
            Tuple of (name, value, metadata) if valid, None otherwise.
        """
        if payload.get("type") != "metric":
            return None

        name = payload.get("name")
        value = payload.get("value")
        if name is None or value is None:
            return None

        metadata = {
            k: v for k, v in payload.items() if k not in ("type", "name", "value")
        }
        return (name, value, metadata)

    def _handle_metric_message(self, _topic: str, payload: Any) -> None:
        """Handle metric message from pub/sub."""
        if not isinstance(payload, dict):
            return

        metric_data = self._extract_metric_data(payload)
        if metric_data is None:
            return

        name, value, metadata = metric_data
        self.update_metric(name, value, **metadata)

    def _handle_data_message(self, _topic: str, payload: Any) -> None:
        """Handle data stream message from pub/sub."""
        if isinstance(payload, dict) and payload.get("type") == "data":
            stream_name = payload.get("stream")
            data = payload.get("data")
            if stream_name is not None and data is not None:
                self.update_data_stream(stream_name, data)

    def _handle_log_message(self, _topic: str, payload: Any) -> None:
        """Handle log message from pub/sub."""
        if isinstance(payload, dict) and payload.get("type") == "log":
            level = payload.get("level", "info")
            message = payload.get("message", "")
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in ("type", "level", "message")
            }
            self.add_log(level, message, **metadata)

    def register_socket(self, socket: LiveViewSocket[ExperimentState]) -> bool:
        """Register a socket for updates.

        Args:
            socket: The LiveView socket to register.

        Returns:
            True if the socket was registered successfully.
        """
        self.connected_sockets.add(socket)
        logger.info(
            "Registered socket for route '%s', total connected: %d",
            self.route_path,
            len(self.connected_sockets),
        )
        return True

    def unregister_socket(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Unregister a socket.

        Args:
            socket: The LiveView socket to unregister.
        """
        self.connected_sockets.discard(socket)
        logger.info(
            "Unregistered socket for route '%s', total connected: %d",
            self.route_path,
            len(self.connected_sockets),
        )

    async def broadcast_update(self) -> None:
        """Broadcast an update signal to all registered sockets via PyView pubsub.

        Throttled to max once per second. Uses PyView's pubsub system to trigger
        handle_info on all subscribed sockets, which then update their context.
        """
        current_time = time.time()
        time_since_last_broadcast = current_time - self._last_broadcast_time

        # If less than throttle period has passed, mark as pending and return
        if time_since_last_broadcast < self._broadcast_throttle_seconds:
            self._pending_broadcast = True
            return

        # Reset throttle timer and pending flag
        self._last_broadcast_time = current_time
        self._pending_broadcast = False

        # Use PyView's pubsub system to broadcast update (like mvg_departures)
        # This triggers handle_info on all subscribed sockets
        try:
            pubsub = PubSub(pub_sub_hub, self.broadcast_topic)
            # Send to the topic - PyView will route to all subscribed sockets
            await pubsub.send_all_on_topic_async(self.broadcast_topic, "update")

            # Also manually trigger handle_info on all connected sockets as fallback
            # This ensures updates are delivered even if pubsub routing fails
            if self.liveview_instance:
                for socket in list(self.connected_sockets):
                    try:
                        # Manually call handle_info to ensure update is processed
                        await self.liveview_instance.handle_info("update", socket)
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(
                            "Failed to manually trigger handle_info on socket: %s", e
                        )
        except Exception as e:
            logger.exception("Failed to broadcast update via pubsub: %s", e)

    def update_metric(self, name: str, value: float, **metadata: Any) -> None:
        """Update a metric in the state.

        Args:
            name: Metric name
            value: Metric value
            **metadata: Additional metadata (e.g., epoch, step)
        """
        if "metrics" not in self.experiment_state.metrics:
            self.experiment_state.metrics["metrics"] = {}
        self.experiment_state.metrics["metrics"][name] = {"value": value, **metadata}
        self.experiment_state.last_update = datetime.now(timezone.utc)

    def update_data_stream(self, stream_name: str, data: MessagePayload) -> None:
        """Update a data stream in the state.

        Args:
            stream_name: Name of the data stream
            data: Data to store
        """
        self.experiment_state.data_streams[stream_name] = data
        self.experiment_state.last_update = datetime.now(timezone.utc)

    def add_log(self, level: str, message: str, **metadata: Any) -> None:
        """Add a log message to the state.

        Args:
            level: Log level (e.g., "info", "warning", "error")
            message: Log message
            **metadata: Additional metadata
        """
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        self.experiment_state.recent_logs.append(log_entry)
        # Keep only the most recent logs
        if (
            len(self.experiment_state.recent_logs)
            > self.experiment_state.max_log_history
        ):
            self.experiment_state.recent_logs = self.experiment_state.recent_logs[
                -self.experiment_state.max_log_history :
            ]
        self.experiment_state.last_update = datetime.now(timezone.utc)
