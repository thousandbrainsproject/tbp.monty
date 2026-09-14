"""State manager for experiment LiveView."""

from __future__ import annotations

import logging
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
from .visualization_state import ChartBufferConfig, VisualizationStateManager

if TYPE_CHECKING:
    from .liveview_types import MessagePayload, MetricData, MetricMetadata

if TYPE_CHECKING:
    from pyview import LiveViewSocket

from .broadcast_manager import BroadcastManager

if TYPE_CHECKING:
    from .liveview_experiment import ExperimentLiveView

logger = logging.getLogger(__name__)

# Well-known visualization stream names
EVIDENCE_CHART_STREAM = "evidence_chart"
MESH_VIEWER_STREAM = "mesh_viewer"
SENSOR_IMAGES_STREAM = "sensor_images"
VISUALIZATION_STREAMS = {
    EVIDENCE_CHART_STREAM,
    MESH_VIEWER_STREAM,
    SENSOR_IMAGES_STREAM,
}


class ExperimentStateManager:
    """Manages shared state for experiment LiveView and updates."""

    def __init__(
        self,
        route_path: str = "/",
        chart_buffer_config: ChartBufferConfig | None = None,
    ) -> None:
        """Initialize the state manager.

        Args:
            route_path: The path for this route, used to create a unique topic.
            chart_buffer_config: Configuration for chart data buffering.
                Defaults to keeping all points with 1-second throttling.
        """
        self.route_path = route_path
        # Initialize with default values so LiveView isn't empty
        self.experiment_state = ExperimentState(
            run_name="Experiment",
            status="initializing",
            experiment_mode="train",
        )
        self.connected_sockets: set[LiveViewSocket[ExperimentState]] = set()
        if TYPE_CHECKING:
            self.liveview_instance: ExperimentLiveView | None = None
        else:
            self.liveview_instance: Any = None  # Reference to LiveView instance
        # Command publisher is set by ServerOrchestrator
        self.command_publisher: Any = None
        # Create route-specific topic based on path
        normalized_path = route_path.strip("/").replace("/", ":") or "root"
        self.broadcast_topic: str = f"experiment:updates:{normalized_path}"
        # Sub-topics for different data streams
        self.metrics_topic: str = f"{self.broadcast_topic}:metrics"
        self.data_topic: str = f"{self.broadcast_topic}:data"
        self.logs_topic: str = f"{self.broadcast_topic}:logs"

        # Throttling for broadcasts (max ~10 times per second)
        self._last_broadcast_time: float = 0.0
        self._broadcast_throttle_seconds: float = 0.1
        self._pending_broadcast: bool = False

        # Visualization state management with configurable buffering
        self.visualization = VisualizationStateManager(chart_buffer_config)

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

    def _handle_metric_message(self, _topic: str, payload: MessagePayload) -> None:
        """Handle metric message from pub/sub."""
        if not isinstance(payload, dict):
            return

        metric_data = self._extract_metric_data(payload)
        if metric_data is None:
            return

        name, value, metadata = metric_data
        self.update_metric(name, value, **metadata)

    def _handle_data_message(self, _topic: str, payload: MessagePayload) -> None:
        """Handle data stream message from pub/sub."""
        if isinstance(payload, dict) and payload.get("type") == "data":
            stream_name = payload.get("stream")
            data = payload.get("data")
            if stream_name is not None and data is not None:
                self.update_data_stream(stream_name, data)

    def _handle_log_message(self, _topic: str, payload: MessagePayload) -> None:
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

    async def broadcast_update(self, force: bool = False) -> None:
        """Broadcast an update signal to all registered sockets via PyView pubsub.

        Throttled to max once per second. Uses PyView's pubsub system to trigger
        handle_info on all subscribed sockets, which then update their context.

        Args:
            force: If True, bypass throttling and broadcast immediately
                   (useful for terminal states like aborted/completed/error)
        """
        if not hasattr(self, "_broadcast_manager"):
            self._broadcast_manager = BroadcastManager(
                self, throttle_seconds=self._broadcast_throttle_seconds
            )

        await self._broadcast_manager.broadcast_if_needed(force=force)

    def update_metric(
        self, name: str, value: float, **metadata: MetricMetadata  # noqa: ARG002
    ) -> None:
        """Update a metric in the state.

        Args:
            name: Metric name
            value: Metric value
            **metadata: Additional metadata (e.g., epoch, step) - currently unused
        """
        self.experiment_state.metrics[name] = value
        self.experiment_state.last_update = datetime.now(timezone.utc)

    def update_data_stream(self, stream_name: str, data: MessagePayload) -> None:
        """Update a data stream in the state.

        Routes visualization-specific streams to the visualization manager
        for buffered history, and stores other streams directly.

        Args:
            stream_name: Name of the data stream
            data: Data to store
        """
        # Route visualization streams to specialized handlers
        if stream_name in VISUALIZATION_STREAMS:
            self._update_visualization_stream(stream_name, data)
        else:
            self._update_generic_data_stream(stream_name, data)

        self.experiment_state.last_update = datetime.now(timezone.utc)

    def _update_visualization_stream(
        self, stream_name: str, data: MessagePayload
    ) -> None:
        """Update one of the known visualization streams."""
        handlers = {
            EVIDENCE_CHART_STREAM: self.visualization.process_evidence_data,
            MESH_VIEWER_STREAM: self.visualization.process_mesh_data,
            SENSOR_IMAGES_STREAM: self.visualization.process_sensor_images,
        }
        handler = handlers.get(stream_name)
        if handler is not None:
            handler(data)

    def _update_generic_data_stream(
        self, stream_name: str, data: MessagePayload
    ) -> None:
        """Update a non-visualization data stream."""
        self.experiment_state.data_streams[stream_name] = data

    def add_log(self, level: str, message: str, **metadata: MetricMetadata) -> None:
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
