"""LiveView for displaying experiment progress."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Union

from pyview import LiveView, LiveViewSocket, is_connected
from pyview.events import InfoEvent
from pyview.template.live_template import LiveRender, LiveTemplate
from pyview.vendor import ibis
from pyview.vendor.ibis.loaders import FileReloader

from .experiment_state import ExperimentState
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
        # Register this LiveView instance with the state manager so it can call handle_info directly
        state_manager.liveview_instance = self

    def _update_context_from_state(
        self, socket: LiveViewSocket[ExperimentState]
    ) -> None:
        """Update socket context from shared state manager.
        
        Mutates the existing context object (like mvg_departures) so PyView detects changes.
        If context doesn't exist yet, initializes it first (shouldn't happen in normal flow).

        Args:
            socket: The socket connection.
        """
        state = self.state_manager.experiment_state
        
        # Ensure context is initialized (should already be set in mount, but safety check)
        if socket.context is None:
            logger.warning("socket.context is None, initializing it now")
            self._set_socket_context(socket)
            return
        
        # Normalize status
        raw_status = state.status or "initializing"
        normalized_status = raw_status.lower()
        if normalized_status not in ("initializing", "running", "completed", "error"):
            normalized_status = "initializing"
        
        # Replace the entire context object (PyView detects object identity changes)
        # This ensures PyView detects the change and triggers a re-render
        socket.context = ExperimentState(
            total_train_steps=state.total_train_steps or 0,
            train_episodes=state.train_episodes or 0,
            train_epochs=state.train_epochs or 0,
            total_eval_steps=state.total_eval_steps or 0,
            eval_episodes=state.eval_episodes or 0,
            eval_epochs=state.eval_epochs or 0,
            experiment_mode=state.experiment_mode or "train",
            current_epoch=state.current_epoch or 0,
            current_episode=state.current_episode or 0,
            current_step=state.current_step or 0,
            run_name=state.run_name or "Experiment",
            experiment_name=state.experiment_name or "",
            environment_name=state.environment_name or "",
            config_path=state.config_path or "",
            experiment_start_time=state.experiment_start_time,
            last_update=state.last_update,
            max_train_steps=state.max_train_steps,
            max_eval_steps=state.max_eval_steps,
            max_total_steps=state.max_total_steps,
            n_train_epochs=state.n_train_epochs,
            n_eval_epochs=state.n_eval_epochs,
            do_train=state.do_train if state.do_train is not None else False,
            do_eval=state.do_eval if state.do_eval is not None else False,
            show_sensor_output=state.show_sensor_output if state.show_sensor_output is not None else False,
            seed=state.seed,
            model_path=state.model_path,
            model_name_or_path=state.model_name_or_path or "",
            min_lms_match=state.min_lms_match,
            learning_module_count=state.learning_module_count or 0,
            sensor_module_count=state.sensor_module_count or 0,
            metrics=state.metrics.copy() if state.metrics else {},
            data_streams=state.data_streams.copy() if state.data_streams else {},
            recent_logs=state.recent_logs.copy() if state.recent_logs else [],
            max_log_history=state.max_log_history or 100,
            status=normalized_status,
            error_message=state.error_message,
            setup_message=state.setup_message or "",
        )
        
        logger.debug(
            f"Updated context: mode={state.experiment_mode}, epoch={state.current_epoch}, step={state.current_step}, status={normalized_status}"
        )

    def _build_template_assigns(
        self, state: ExperimentState
    ) -> Dict[str, Any]:
        """Build template assigns dictionary from state.

        Args:
            state: The current experiment state.

        Returns:
            Dictionary of template variables for rendering.
        """
        now = datetime.now(timezone.utc)
        elapsed_time = None
        if state.experiment_start_time:
            elapsed = now - state.experiment_start_time
            elapsed_time = str(elapsed).split(".")[0]  # Remove microseconds

        last_update_str = "Never"
        if state.last_update:
            last_update_str = state.last_update.strftime("%Y-%m-%d %H:%M:%S")

        # Normalize status - ensure it's always a valid value
        normalized_status = (state.status or "initializing").lower()
        if normalized_status not in ("initializing", "running", "completed", "error"):
            normalized_status = "initializing"
        
        # Normalize mode display: "train" -> "training", "eval" -> "evaluating"
        # Ensure we always have a valid mode
        raw_mode = state.experiment_mode or "train"
        if raw_mode == "train":
            mode_display = "training"
        elif raw_mode == "eval":
            mode_display = "evaluating"
        else:
            mode_display = str(raw_mode)
        
        # Ensure all template variables are always strings, never None or undefined
        # This prevents template engine from rendering "undefined" for missing variables
        logger.debug(f"Building template assigns: run_name={state.run_name}, experiment_name={state.experiment_name}, current_step={state.current_step}, current_epoch={state.current_epoch}")
        return {
            "run_name": str(state.run_name) if state.run_name else "Experiment",
            "experiment_name": str(state.experiment_name) if state.experiment_name else "",
            "environment_name": str(state.environment_name) if state.environment_name else "",
            "config_path": str(state.config_path) if state.config_path else "",
            "status": normalized_status,
            "status_display": normalized_status.upper(),  # Pre-uppercase for template
            "experiment_mode": str(raw_mode),
            "experiment_mode_display": mode_display,
            "experiment_mode_display_upper": mode_display.upper(),
            "current_epoch": state.current_epoch,
            "current_episode": state.current_episode,
            "current_step": state.current_step,
            "total_train_steps": state.total_train_steps,
            "train_episodes": state.train_episodes,
            "train_epochs": state.train_epochs,
            "total_eval_steps": state.total_eval_steps,
            "eval_episodes": state.eval_episodes,
            "eval_epochs": state.eval_epochs,
            "max_train_steps": state.max_train_steps,
            "max_eval_steps": state.max_eval_steps,
            "max_total_steps": state.max_total_steps,
            "n_train_epochs": state.n_train_epochs,
            "n_eval_epochs": state.n_eval_epochs,
            "do_train": state.do_train,
            "do_eval": state.do_eval,
            "show_sensor_output": state.show_sensor_output,
            "seed": state.seed,
            "model_path": state.model_path,
            "model_name_or_path": str(state.model_name_or_path) if state.model_name_or_path else "",
            "min_lms_match": state.min_lms_match,
            "learning_module_count": state.learning_module_count,
            "sensor_module_count": state.sensor_module_count,
            "elapsed_time": elapsed_time or "N/A",
            "last_update": last_update_str,
            "error_message": state.error_message or "",
            "has_error": bool(state.error_message),
            "data_streams": state.data_streams,
            "recent_logs": state.recent_logs[-20:],  # Show last 20 logs
            "setup_message": state.setup_message or "",  # Cached setup message for display
        }

    def _set_socket_context(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Set socket context with current state (like mvg_departures).
        
        Uses pre-cached config values (like setup_message) or defaults if config
        hasn't been received and cached yet.
        
        Args:
            socket: LiveView socket.
        """
        state = self.state_manager.experiment_state
        logger.debug(f"Setting socket context: run_name={state.run_name}, experiment_name={state.experiment_name}, current_step={state.current_step}, current_epoch={state.current_epoch}")
        # Normalize status
        raw_status = state.status or "initializing"
        normalized_status = raw_status.lower()
        if normalized_status not in ("initializing", "running", "completed", "error"):
            normalized_status = "initializing"
        
        # Use cached setup_message if available, otherwise use default/empty
        setup_message = state.setup_message or ""
        
        # Use defaults for fields that haven't been set yet
        socket.context = ExperimentState(
            total_train_steps=state.total_train_steps or 0,
            train_episodes=state.train_episodes or 0,
            train_epochs=state.train_epochs or 0,
            total_eval_steps=state.total_eval_steps or 0,
            eval_episodes=state.eval_episodes or 0,
            eval_epochs=state.eval_epochs or 0,
            experiment_mode=state.experiment_mode or "train",
            current_epoch=state.current_epoch or 0,
            current_episode=state.current_episode or 0,
            current_step=state.current_step or 0,
            run_name=state.run_name or "Experiment",
            experiment_name=state.experiment_name or "",
            environment_name=state.environment_name or "",
            config_path=state.config_path or "",
            experiment_start_time=state.experiment_start_time,
            last_update=state.last_update,
            max_train_steps=state.max_train_steps,
            max_eval_steps=state.max_eval_steps,
            max_total_steps=state.max_total_steps,
            n_train_epochs=state.n_train_epochs,
            n_eval_epochs=state.n_eval_epochs,
            do_train=state.do_train if state.do_train is not None else False,
            do_eval=state.do_eval if state.do_eval is not None else False,
            show_sensor_output=state.show_sensor_output if state.show_sensor_output is not None else False,
            seed=state.seed,
            model_path=state.model_path,
            model_name_or_path=state.model_name_or_path or "",
            min_lms_match=state.min_lms_match,
            learning_module_count=state.learning_module_count or 0,
            sensor_module_count=state.sensor_module_count or 0,
            metrics=state.metrics.copy() if state.metrics else {},
            data_streams=state.data_streams.copy() if state.data_streams else {},
            recent_logs=state.recent_logs.copy() if state.recent_logs else [],
            max_log_history=state.max_log_history or 100,
            status=normalized_status,
            error_message=state.error_message,
            setup_message=setup_message,  # Use cached setup_message or empty string
        )

    async def mount(
        self, socket: LiveViewSocket[ExperimentState], _session: dict
    ) -> None:
        """Mount the LiveView and register socket for updates."""
        if not self.state_manager.register_socket(socket):
            return

        # Set socket context with current state (like mvg_departures)
        self._set_socket_context(socket)

        # Subscribe to the broadcast topic when socket is connected
        # This is critical for receiving pubsub "update" messages
        socket_id = getattr(socket, "id", "unknown")
        if is_connected(socket):
            await self._subscribe_to_broadcast_topic(socket, socket_id)
        else:
            logger.debug(
                f"Socket {socket_id} not connected during mount, will subscribe when connected"
            )

    async def _subscribe_to_broadcast_topic(
        self, socket: LiveViewSocket[ExperimentState], socket_id: str
    ) -> None:
        """Subscribe socket to broadcast topic."""
        try:
            await socket.subscribe(self.state_manager.broadcast_topic)
            logger.info(
                f"Successfully subscribed socket {socket_id} to broadcast topic: {self.state_manager.broadcast_topic}"
            )
        except Exception as e:
            logger.error(
                f"Failed to subscribe socket {socket_id} to topic {self.state_manager.broadcast_topic}: {e}",
                exc_info=True,
            )

    async def unmount(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Unmount the LiveView and unregister socket."""
        self.state_manager.unregister_socket(socket)

    async def disconnect(self, socket: LiveViewSocket[ExperimentState]) -> None:
        """Handle socket disconnection."""
        self.state_manager.unregister_socket(socket)

    def _handle_metric_update(self, payload: Dict[str, Any], socket: LiveViewSocket[ExperimentState]) -> bool:
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
        
        metadata = {k: v for k, v in payload.items() 
                   if k not in ("type", "name", "value")}
        self.state_manager.update_metric(name, value, **metadata)
        self._update_context_from_state(socket)
        logger.debug(f"Updated metric '{name}' = {value}")
        return True

    def _handle_data_update(self, payload: Dict[str, Any], socket: LiveViewSocket[ExperimentState]) -> bool:
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
        logger.debug(f"Updated data stream '{stream_name}'")
        return True

    def _handle_log_update(self, payload: Dict[str, Any], socket: LiveViewSocket[ExperimentState]) -> bool:
        """Handle log update from pub/sub.
        
        Returns:
            True if handled, False otherwise.
        """
        if payload.get("type") != "log":
            return False
        
        level = payload.get("level", "info")
        message = payload.get("message", "")
        metadata = {k: v for k, v in payload.items() 
                   if k not in ("type", "level", "message")}
        self.state_manager.add_log(level, message, **metadata)
        self._update_context_from_state(socket)
        logger.debug(f"Added log [{level}]: {message}")
        return True

    def _route_topic_to_handler(
        self, topic: str, payload: Dict[str, Any], socket: LiveViewSocket[ExperimentState]
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

    def _handle_info_event(self, event: InfoEvent, socket: LiveViewSocket[ExperimentState]) -> None:
        """Handle InfoEvent from pub/sub."""
        topic = event.name
        payload = event.payload
        
        # Handle general state update
        if payload == "update":
            logger.debug(f"Received 'update' signal on topic '{topic}', updating context from state")
            self._update_context_from_state(socket)
            logger.debug(f"Context updated, current step: {socket.context.current_step}, status: {socket.context.status}")
            return
        
        # Handle typed payloads
        if not isinstance(payload, dict):
            logger.debug(f"Received InfoEvent from topic '{topic}' with non-dict payload: {payload}")
            return
        
        # Route to appropriate handler
        if self._route_topic_to_handler(topic, payload, socket):
            return
        
        logger.debug(f"Received InfoEvent from topic '{topic}' with payload: {payload}")

    async def handle_info(
        self, event: Union[str, InfoEvent], socket: LiveViewSocket[ExperimentState]
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
        elif isinstance(event, str):
            if event == "update":
                logger.info(f"Received 'update' string signal, updating context from state")
                self._update_context_from_state(socket)
                logger.info(f"Context updated, current step: {socket.context.current_step}, status: {socket.context.status}")
            else:
                logger.debug(f"Received direct payload: {event}")

    def _load_template(self) -> LiveTemplate:
        """Load and prepare template for rendering.

        Returns:
            LiveTemplate object.
        """
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = Path(current_file_dir).parent / "templates"

        if not hasattr(ibis, "loader") or not isinstance(ibis.loader, FileReloader):
            ibis.loader = FileReloader(str(templates_dir))

        template_path = "experiment.html"
        template_file = templates_dir / template_path
        with open(template_file, encoding="utf-8") as f:
            template_content = f.read()

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
            error_template = ibis.Template("<div>Error rendering template: {{ error }}</div>")
            error_live_template = LiveTemplate(error_template)
            error_assigns = {"error": str(error)}
            return LiveRender(error_live_template, error_assigns, meta)
        except Exception:
            minimal_template = ibis.Template("<div>Error: Failed to render</div>")
            minimal_live_template = LiveTemplate(minimal_template)
            return LiveRender(minimal_live_template, {}, meta)

    def _extract_state_from_assigns(self, assigns: Union[ExperimentState, dict]) -> ExperimentState:
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

    async def render(
        self, assigns: Union[ExperimentState, dict], meta: Any
    ) -> str:
        """Render the HTML template."""
        logger.debug(f"Render called at {datetime.now(timezone.utc)}, assigns type: {type(assigns)}")

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
            logger.error(f"Error rendering template: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return self._create_error_template(e, meta)  # type: ignore[no-any-return]

