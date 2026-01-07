"""LiveView for displaying experiment progress."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from markupsafe import Markup
from pyview import LiveView, LiveViewSocket
from pyview.events import InfoEvent
from pyview.template.live_template import LiveRender, LiveTemplate
from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader

from .experiment_state import ExperimentState
from .state_normalizer import StateNormalizer

if TYPE_CHECKING:
    from .types import MessagePayload, TemplateAssigns

if TYPE_CHECKING:
    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)


class ExperimentLiveView(LiveView[ExperimentState]):
    """LiveView for displaying experiment progress."""

    def __init__(self, state_manager: ExperimentStateManager) -> None:
        """Initialize the LiveView.

        Args:
            state_manager: The state manager for this experiment.
        """
        super().__init__()
        self.state_manager = state_manager
        # Register this LiveView instance with the state manager
        # so it can call handle_info directly
        state_manager.liveview_instance = self

    def _create_context_from_state(self, state: ExperimentState) -> ExperimentState:
        """Create a new ExperimentState from the shared state.

        Creates a copy with normalized values for use as socket context.
        Preserves 0 values and provides safe defaults for None values.

        Args:
            state: Source experiment state

        Returns:
            New ExperimentState instance for socket context
        """
        normalized_status = self._normalize_status(state.status)
        return StateNormalizer.normalize(state, normalized_status)

    def _extract_model_name(self, model_path: str | None) -> str:
        """Extract a readable model name from a model path.

        Extracts the meaningful model folder name from a full path like:
        /path/to/pretrained_ycb_v11/surf_agent_1lm_10distinctobj/pretrained/
        -> surf_agent_1lm_10distinctobj

        Args:
            model_path: Full path to model or None

        Returns:
            Short readable model name
        """
        if not model_path:
            return ""
        path_str = str(model_path).rstrip("/")
        parts = path_str.split("/")
        # Filter out common non-informative parts
        skip_parts = {"pretrained", "pretrained_models", ""}
        meaningful_parts = [p for p in parts[-3:] if p not in skip_parts]
        if meaningful_parts:
            return meaningful_parts[-1]  # Return the most specific part
        return path_str.split("/")[-1] if parts else ""

    def _update_context_from_state(
        self, socket: LiveViewSocket[ExperimentState]
    ) -> None:
        """Update socket context from shared state manager.

        Mutates the existing context object (like mvg_departures)
        so PyView detects changes. If context doesn't exist yet,
        initializes it first (shouldn't happen in normal flow).

        Args:
            socket: The socket connection.
        """
        state = self.state_manager.experiment_state

        # Ensure context is initialized
        # (should already be set in mount, but safety check)
        if socket.context is None:
            logger.warning("socket.context is None, initializing it now")
            self._set_socket_context(socket)
            return

        # Replace the entire context object
        # (PyView detects object identity changes)
        # This ensures PyView detects the change and triggers a re-render
        socket.context = self._create_context_from_state(state)

    def _normalize_status(self, status: str | None) :  # type: ignore[override]
        """Normalize status to a value understood by the UI.

        We treat a small, explicit set of states as first‑class:

        - ``initializing``: LiveView waiting for first data
        - ``running``: experiment is in progress (train or eval)
        - ``completed``: experiment finished successfully
        - ``error``: experiment crashed
        - ``aborting``: abort requested, experiment still shutting down
        - ``aborted``: experiment stopped early by the user

        Any other non‑empty status string is passed through unchanged so that
        future extensions still show *something* sensible instead of silently
        mapping to ``initializing``.
        """
        normalized = (status or "initializing").lower()
        valid_statuses = (
            "initializing",
            "running",
            "completed",
            "error",
            "aborting",
            "aborted",
        )
        if normalized in valid_statuses:
            return normalized
        return normalized

    def _normalize_mode_display(self, mode: str | None) -> str:
        """Normalize experiment mode for display.

        Args:
            mode: Raw mode string

        Returns:
            Display-friendly mode string
        """
        raw_mode = mode or "train"
        if raw_mode == "train":
            return "training"
        if raw_mode == "eval":
            return "evaluating"
        if not raw_mode or str(raw_mode).strip() == "":
            return "training"
        return str(raw_mode)

    def _format_elapsed_time(self, start_time: datetime | None) -> str:
        """Format elapsed time since experiment start.

        Args:
            start_time: Experiment start time

        Returns:
            Formatted elapsed time string or "N/A"
        """
        if not start_time:
            return "N/A"
        elapsed = datetime.now(timezone.utc) - start_time
        return str(elapsed).split(".")[0]  # Remove microseconds

    def _format_last_update(self, last_update: datetime | None) -> str:
        """Format last update timestamp.

        Args:
            last_update: Last update datetime

        Returns:
            Formatted timestamp string or "Never"
        """
        if not last_update:
            return "Never"
        return last_update.strftime("%Y-%m-%d %H:%M:%S UTC")

    def _build_string_assigns(self, state: ExperimentState) -> dict[str, str]:
        """Build string template assigns.

        Args:
            state: Experiment state

        Returns:
            Dictionary of string template variables
        """
        return {
            "run_name": str(state.run_name) if state.run_name else "Experiment",
            "experiment_name": (
                str(state.experiment_name) if state.experiment_name else ""
            ),
            "environment_name": (
                str(state.environment_name) if state.environment_name else ""
            ),
            "config_path": str(state.config_path) if state.config_path else "",
            "model_path": str(state.model_path) if state.model_path else "",
            "model_name": self._extract_model_name(state.model_name_or_path),
            "error_message": (str(state.error_message) if state.error_message else ""),
            "setup_message": (str(state.setup_message) if state.setup_message else ""),
        }

    def _build_numeric_assigns(self, state: ExperimentState) -> dict[str, int]:
        """Build required numeric template assigns (always integers).

        Args:
            state: Experiment state

        Returns:
            Dictionary of numeric template variables
        """

        def safe_int(value: int | None) -> int:
            return int(value) if value is not None else 0

        return {
            "current_epoch": safe_int(state.current_epoch),
            "current_episode": safe_int(state.current_episode),
            "current_step": safe_int(state.current_step),
            "total_train_steps": safe_int(state.total_train_steps),
            "train_episodes": safe_int(state.train_episodes),
            "train_epochs": safe_int(state.train_epochs),
            "total_eval_steps": safe_int(state.total_eval_steps),
            "eval_episodes": safe_int(state.eval_episodes),
            "eval_epochs": safe_int(state.eval_epochs),
            "learning_module_count": safe_int(state.learning_module_count),
            "sensor_module_count": safe_int(state.sensor_module_count),
        }

    def _build_optional_numeric_assigns(
        self, state: ExperimentState
    ) -> dict[str, int | None]:
        """Build optional numeric template assigns (can be None).

        Args:
            state: Experiment state

        Returns:
            Dictionary of optional numeric template variables
        """

        def safe_int_or_none(value: int | None) -> int | None:
            return int(value) if value is not None else None

        return {
            "max_train_steps": safe_int_or_none(state.max_train_steps),
            "max_eval_steps": safe_int_or_none(state.max_eval_steps),
            "max_total_steps": safe_int_or_none(state.max_total_steps),
            "n_train_epochs": safe_int_or_none(state.n_train_epochs),
            "n_eval_epochs": safe_int_or_none(state.n_eval_epochs),
            "seed": safe_int_or_none(state.seed),
            "min_lms_match": safe_int_or_none(state.min_lms_match),
        }

    def _build_boolean_assigns(self, state: ExperimentState) -> dict[str, bool]:
        """Build boolean template assigns.

        Args:
            state: Experiment state

        Returns:
            Dictionary of boolean template variables
        """

        def safe_bool(value: bool | None) -> bool:
            return bool(value) if value is not None else False

        return {
            "do_train": safe_bool(state.do_train),
            "do_eval": safe_bool(state.do_eval),
            "show_sensor_output": safe_bool(state.show_sensor_output),
            "has_error": bool(state.error_message),
        }

    def _build_template_assigns(self, state: ExperimentState) -> TemplateAssigns:
        """Build template assigns dictionary from state.

        Args:
            state: The current experiment state.

        Returns:
            Dictionary of template variables for rendering.
        """
        normalized_status = self._normalize_status(state.status)
        raw_mode = state.experiment_mode or "train"
        mode_display = self._normalize_mode_display(raw_mode)

        assigns: dict[str, Any] = {}
        assigns.update(self._build_string_assigns(state))
        assigns.update(self._build_numeric_assigns(state))
        assigns.update(self._build_optional_numeric_assigns(state))
        assigns.update(self._build_boolean_assigns(state))

        # Status and mode
        assigns.update(
            {
                "status": normalized_status,
                "status_display": normalized_status.upper(),
                "experiment_mode": str(raw_mode),
                "experiment_mode_display": mode_display,
                "experiment_mode_display_upper": mode_display.upper(),
            }
        )

        # Time values
        assigns.update(
            {
                "elapsed_time": self._format_elapsed_time(state.experiment_start_time),
                "last_update": self._format_last_update(state.last_update),
            }
        )

        # Complex values
        assigns.update(
            {
                "data_streams": (
                    state.data_streams if state.data_streams is not None else {}
                ),
                "recent_logs": state.recent_logs[-20:] if state.recent_logs else [],
            }
        )

        # Visualization data
        assigns.update(self._build_visualization_assigns())

        return assigns

    def _build_visualization_assigns(self) -> dict[str, Any]:
        """Build visualization-related template assigns.

        Chart data is embedded as JSON for DOM transport.
        push_event is attempted but falls back to JSON tag.

        Returns:
            Dictionary with chart data and metadata for template.
        """
        viz_state = self.state_manager.visualization.state
        sensor_images = viz_state.sensor_images

        # Wrap JSON in Markup to prevent HTML escaping in template
        return {
            "chart_data_json": Markup(viz_state.get_chart_data_json()),
            "chart_point_count": viz_state.point_count,
            # Sensor images (base64-encoded PNG)
            "camera_image_b64": sensor_images.camera_image,
            "depth_image_b64": sensor_images.depth_image,
            "sensor_image_step": sensor_images.step,
            "has_sensor_images": bool(
                sensor_images.camera_image or sensor_images.depth_image
            ),
        }

    def _set_socket_context(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Set socket context with current state (like mvg_departures).

        Uses pre-cached config values (like setup_message) or defaults if config
        hasn't been received and cached yet.

        Args:
            socket: LiveView socket.
        """
        state = self.state_manager.experiment_state
        logger.debug(
            "Setting socket context: run_name=%s, experiment_name=%s, "
            "current_step=%d, current_epoch=%d",
            state.run_name,
            state.experiment_name,
            state.current_step,
            state.current_epoch,
        )
        socket.context = self._create_context_from_state(state)

    async def mount(
        self, socket: LiveViewSocket[ExperimentState], _session: dict
    ) -> None:
        """Mount the LiveView and register socket for updates."""
        if not self.state_manager.register_socket(socket):
            return

        # Set socket context with current state
        self._set_socket_context(socket)

        # Subscribe to pubsub when socket is connected (has subscribe method)
        # mount is called with UnconnectedSocket during HTTP, and
        # ConnectedLiveViewSocket when WebSocket connects
        socket_id = getattr(socket, "id", "unknown")
        if hasattr(socket, "subscribe"):
            await self._subscribe_to_broadcast_topic(socket, socket_id)
            # Reset sent tracking so first push sends all accumulated data
            self.state_manager.visualization.state.reset_sent_tracking()
            # Push any existing chart data immediately
            await self._push_chart_updates(socket)

    async def _subscribe_to_broadcast_topic(
        self, socket: LiveViewSocket[ExperimentState], socket_id: str
    ) -> None:
        """Subscribe socket to broadcast topic."""
        try:
            await socket.subscribe(self.state_manager.broadcast_topic)
            logger.info(
                "Successfully subscribed socket %s to broadcast topic: %s",
                socket_id,
                self.state_manager.broadcast_topic,
            )
        except Exception as e:
            logger.exception(
                "Failed to subscribe socket %s to topic %s: %s",
                socket_id,
                self.state_manager.broadcast_topic,
                e,
            )

    async def unmount(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Unmount the LiveView and unregister socket."""
        self.state_manager.unregister_socket(socket)

    async def disconnect(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Handle socket disconnection."""
        self.state_manager.unregister_socket(socket)

    def _handle_metric_update(
        self, payload: MessagePayload, socket: LiveViewSocket[ExperimentState]
    ) -> bool:
        """Handle metric update from pub/sub.

        Returns:
            True if handled, False otherwise.
        """
        if payload.get("type") != "metric":
            return False

        name = payload.get("name")
        value = payload.get("value")
        if name is None or value is None:
            return False

        metadata = {
            k: v for k, v in payload.items() if k not in ("type", "name", "value")
        }
        self.state_manager.update_metric(name, value, **metadata)
        self._update_context_from_state(socket)
        return True

    def _handle_data_update(
        self, payload: MessagePayload, socket: LiveViewSocket[ExperimentState]
    ) -> bool:
        """Handle data stream update from pub/sub.

        Returns:
            True if handled, False otherwise.
        """
        if payload.get("type") != "data":
            return False

        stream_name = payload.get("stream")
        data = payload.get("data")
        if stream_name is None or data is None:
            return False

        self.state_manager.update_data_stream(stream_name, data)
        self._update_context_from_state(socket)
        return True

    def _handle_log_update(
        self, payload: MessagePayload, socket: LiveViewSocket[ExperimentState]
    ) -> bool:
        """Handle log update from pub/sub.

        Returns:
            True if handled, False otherwise.
        """
        if payload.get("type") != "log":
            return False

        level = payload.get("level", "info")
        message = payload.get("message", "")
        metadata = {
            k: v for k, v in payload.items() if k not in ("type", "level", "message")
        }
        self.state_manager.add_log(level, message, **metadata)
        self._update_context_from_state(socket)
        return True

    def _route_topic_to_handler(
        self,
        topic: str,
        payload: MessagePayload,
        socket: LiveViewSocket[ExperimentState],
    ) -> bool:
        """Route topic to appropriate handler.

        Returns:
            True if handled, False otherwise.
        """
        if topic == self.state_manager.metrics_topic:
            return self._handle_metric_update(payload, socket)
        if topic == self.state_manager.data_topic:
            return self._handle_data_update(payload, socket)
        if topic == self.state_manager.logs_topic:
            return self._handle_log_update(payload, socket)
        return False

    def _handle_info_event(
        self, event: InfoEvent, socket: LiveViewSocket[ExperimentState]
    ) -> None:
        """Handle InfoEvent from pub/sub."""
        topic = event.name
        payload = event.payload

        # Handle general state update
        if payload == "update":
            self._update_context_from_state(socket)
            return

        # Handle typed payloads
        if not isinstance(payload, dict):
            return

        # Route to appropriate handler
        if self._route_topic_to_handler(topic, payload, socket):
            return

    async def handle_info(
        self, event: str | InfoEvent, socket: LiveViewSocket[ExperimentState]
    ) -> None:
        """Handle update messages from pubsub.

        Handles multiple types of events:
        - "update": General state update
        - Metrics: Updates to metrics from parallel processes
        - Data: Updates to data streams
        - Logs: Log messages from parallel processes
        """
        if isinstance(event, InfoEvent):
            self._handle_info_event(event, socket)
        elif event == "update":
            self._update_context_from_state(socket)
        else:
            logger.debug("Received direct payload: %s", event)

        # Push incremental chart data via dedicated channel (not DOM diff)
        await self._push_chart_updates(socket)

    async def handle_event(
        self,
        event: str,
        _payload: Any = None,
        socket: LiveViewSocket[ExperimentState] | None = None,
    ) -> None:
        """Handle UI events from the client.

        Args:
            event: Event name (e.g., "abort_experiment").
            _payload: Event payload data (optional, may be None for phx-click events).
            socket: The socket that triggered the event (required).
        """
        if socket is None:
            logger.warning("handle_event called without socket for event: %s", event)
            return

        # Normalize payload to empty dict if None
        # (pyview may pass None for events without payload)
        if _payload is None:
            _payload = {}

        if event == "abort_experiment":
            await self._handle_abort_event(socket)
        else:
            logger.debug("Unknown event: %s", event)

    async def _handle_abort_event(
        self, socket: LiveViewSocket[ExperimentState]
    ) -> None:
        """Handle abort experiment request from UI.

        Args:
            socket: The socket that triggered the abort.
        """
        logger.info("Abort experiment requested from LiveView UI")

        # Send abort command via ZMQ
        if self.state_manager.command_publisher:
            success = self.state_manager.command_publisher.abort_experiment(
                reason="User requested abort via LiveView UI"
            )
            if success:
                logger.info("Abort command sent successfully")
                # Update local state to reflect pending abort
                self.state_manager.experiment_state.status = "aborting"
                self._update_context_from_state(socket)
            else:
                logger.error("Failed to send abort command")
        else:
            logger.warning("No command publisher available, cannot send abort")

    async def _push_chart_updates(
        self, socket: LiveViewSocket[ExperimentState]
    ) -> None:
        """Push incremental chart data via push_event.

        Uses pyview's push_event to send only new data points to the client,
        avoiding large DOM diffs for chart data. This follows the pattern
        described in https://elixirschool.com/blog/live-view-with-channels

        Args:
            socket: LiveView socket (must be connected to push events)
        """
        # Only push if socket supports push_event (connected socket)
        if not hasattr(socket, "push_event"):
            return

        viz_state = self.state_manager.visualization.state
        unsent_data = viz_state.get_unsent_data()

        if unsent_data is not None:
            await socket.push_event("chart_data", unsent_data)
            viz_state.mark_as_sent()

    def _load_template(self) -> LiveTemplate:
        """Load and prepare template for rendering.

        Returns:
            LiveTemplate object.
        """
        current_file_path = Path(__file__).resolve()
        templates_dir = current_file_path.parent.parent / "templates"

        if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
            ibis.loader = FileReloader(str(templates_dir))

        template_path = "experiment.html"
        template_file = templates_dir / template_path
        template_content = template_file.read_text(encoding="utf-8")

        template = ibis.Template(template_content)
        return LiveTemplate(template)

    def _create_error_template(self, error: Exception, meta: Any) -> LiveRender:
        """Create error template for rendering failures.

        Args:
            error: Exception that occurred.
            meta: Template metadata.

        Returns:
            LiveRender object with error template.
        """
        try:
            error_template = ibis.Template(
                "<div>Error rendering template: {{ error }}</div>"
            )
            error_live_template = LiveTemplate(error_template)
            error_assigns = {"error": str(error)}
            return LiveRender(error_live_template, error_assigns, meta)
        except (ValueError, AttributeError):
            minimal_template = ibis.Template("<div>Error: Failed to render</div>")
            minimal_live_template = LiveTemplate(minimal_template)
            return LiveRender(minimal_live_template, {}, meta)

    def _extract_state_from_assigns(
        self, assigns: ExperimentState | dict
    ) -> ExperimentState:
        """Extract ExperimentState from assigns parameter.

        Args:
            assigns: Either ExperimentState or dict containing state.

        Returns:
            ExperimentState instance.
        """
        if isinstance(assigns, ExperimentState):
            return assigns

        if isinstance(assigns, dict):
            state = assigns.get("context", self.state_manager.experiment_state)
            if isinstance(state, ExperimentState):
                return state

        return self.state_manager.experiment_state

    async def render(self, assigns: ExperimentState | dict, meta: Any) -> str:
        """Render the HTML template."""

        try:
            # Extract state from assigns (socket.context)
            state = self._extract_state_from_assigns(assigns)
            # Ensure state is valid
            if state is None:
                logger.error("State is None, using default state")
                state = ExperimentState()
            # Build template assigns dict - this is what the template uses
            template_assigns = self._build_template_assigns(state)
            live_template = self._load_template()
            return LiveRender(live_template, template_assigns, meta)  # type: ignore[no-any-return]
        except Exception as e:
            logger.exception("Error rendering template: %s", e)
            traceback.print_exc()
            return self._create_error_template(e, meta)  # type: ignore[no-any-return]
