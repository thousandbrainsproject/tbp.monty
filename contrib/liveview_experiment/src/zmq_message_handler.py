"""ZMQ message handler for processing experiment updates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

    from .liveview_types import MessagePayload
    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """Snapshot of experiment state for comparison.

    Attributes:
        status: Previous status value
        mode: Previous mode value
        error: Previous error message
        run_name: Previous run name
        completion_detected: Whether completion was detected
    """

    status: str
    mode: str
    error: str | None
    run_name: str
    completion_detected: bool


class ZmqMessageHandler:
    """Handles ZMQ messages from the experiment process.

    Processes different message types (metric, data, log, state) and updates
    the experiment state accordingly.
    """

    def __init__(
        self,
        state_manager: ExperimentStateManager,
        experiment_completed: asyncio.Event | None = None,
    ) -> None:
        """Initialize the message handler.

        Args:
            state_manager: The experiment state manager to update
            experiment_completed: Optional event to signal when experiment completes
        """
        self.state_manager = state_manager
        self.experiment_completed = experiment_completed
        self._datetime_keys = ("experiment_start_time", "last_update")

    async def process_message(self, payload: MessagePayload) -> None:
        """Process a ZMQ message based on its type.

        Args:
            payload: Parsed message payload with 'type' field
        """
        msg_type = payload.get("type", "unknown")

        handlers = {
            "metric": self._handle_metric,
            "data": self._handle_data,
            "log": self._handle_log,
            "state": self._handle_state,
        }

        handler = handlers.get(msg_type)
        if handler:
            await handler(payload)
        else:
            logger.warning("Unknown message type: %s, ignoring", msg_type)

    async def _handle_metric(self, payload: MessagePayload) -> None:
        """Handle metric message from ZMQ.

        Args:
            payload: Metric message payload
        """
        self.state_manager._handle_metric_message(
            self.state_manager.metrics_topic, payload
        )
        await self.state_manager.broadcast_update()

    async def _handle_data(self, payload: MessagePayload) -> None:
        """Handle data stream message from ZMQ.

        Args:
            payload: Data message payload
        """
        data_payload: MessagePayload = {
            "type": "data",
            "stream": payload.get("stream", "unknown"),
            "data": {k: v for k, v in payload.items() if k not in ("type", "stream")},
        }
        self.state_manager._handle_data_message(
            self.state_manager.data_topic, data_payload
        )
        await self.state_manager.broadcast_update()

    async def _handle_log(self, payload: MessagePayload) -> None:
        """Handle log message from ZMQ.

        Args:
            payload: Log message payload
        """
        self.state_manager._handle_log_message(self.state_manager.logs_topic, payload)
        await self.state_manager.broadcast_update()

    def _check_new_experiment(self, payload: MessagePayload) -> None:
        """Check if this is a new experiment and reset visualization if needed.

        Args:
            payload: State message payload
        """
        new_run_name = payload.get("run_name")
        if (
            new_run_name
            and new_run_name != self.state_manager.experiment_state.run_name
            and self.state_manager.experiment_state.run_name  # Not initial empty state
        ):
            # In multi-publisher/parallel scenarios, we do NOT clear visualization
            # state when a new run_name is detected. The visualization manager
            # partitions data by run_name and aggregates progress across publishers.
            # Clearing would drop evidence history and sensor snapshots, causing
            # charts/images to disappear and CRDT-like counters to reset.
            logger.info(
                "New experiment detected: %s -> %s; "
                "preserving existing visualization states",
                self.state_manager.experiment_state.run_name,
                new_run_name,
            )

    def _update_state_from_payload(self, payload: MessagePayload) -> bool:
        """Update experiment state from payload and detect completion.

        Args:
            payload: State message payload

        Returns:
            True if completion was detected, False otherwise
        """
        completion_detected = False
        for key, value in payload.items():
            if key == "type":
                continue

            if not hasattr(self.state_manager.experiment_state, key):
                continue

            # Handle datetime string conversion
            value = self._normalize_value(key, value)
            if value is None:
                continue

            setattr(self.state_manager.experiment_state, key, value)

            # Check if experiment has completed or errored
            if self._check_completion(key, value):
                completion_detected = True

        return completion_detected

    def _is_critical_update(self, snapshot: StateSnapshot) -> bool:
        """Check if this update contains critical signals that need immediate broadcast.

        Args:
            snapshot: Previous state snapshot for comparison

        Returns:
            True if this is a critical update requiring immediate broadcast
        """
        if snapshot.completion_detected:
            return True

        new_status = self.state_manager.experiment_state.status
        new_mode = self.state_manager.experiment_state.experiment_mode
        new_error = self.state_manager.experiment_state.error_message
        new_run_name = self.state_manager.experiment_state.run_name

        # Status transitions are critical
        # (especially initializing -> running, running -> aborting)
        if snapshot.status != new_status:
            return True

        # Error messages are critical
        if new_error and (not snapshot.error or snapshot.error != new_error):
            return True

        # Mode transitions are important (train -> eval)
        if snapshot.mode != new_mode:
            return True

        # New experiment detection is critical (run_name changed)
        return bool(
            new_run_name and new_run_name != snapshot.run_name and snapshot.run_name
        )

    async def _handle_state(self, payload: MessagePayload) -> None:
        """Handle state update message from ZMQ.

        Args:
            payload: State message payload
        """
        # Check if this is a new experiment (run_name changed)
        self._check_new_experiment(payload)

        # Capture state snapshot before updating
        snapshot = StateSnapshot(
            status=self.state_manager.experiment_state.status,
            mode=self.state_manager.experiment_state.experiment_mode,
            error=self.state_manager.experiment_state.error_message,
            run_name=self.state_manager.experiment_state.run_name,
            completion_detected=False,
        )

        # Update state from payload and detect completion
        snapshot.completion_detected = self._update_state_from_payload(payload)

        # Check if this is a critical update
        critical_update = self._is_critical_update(snapshot)

        # Update last_update timestamp
        self.state_manager.experiment_state.last_update = datetime.now(timezone.utc)

        # Force immediate broadcast for critical signals to ensure browser gets updates
        # especially important for fast experiments and status transitions
        await self.state_manager.broadcast_update(force=critical_update)

    def _normalize_value(self, key: str, value: Any) -> Any:
        """Normalize value based on its type (e.g., datetime conversion).

        Args:
            key: State key name
            value: Raw value to normalize

        Returns:
            Normalized value or None if normalization failed
        """
        if key in self._datetime_keys and isinstance(value, str):
            parsed_value = self._parse_datetime(value)
            if parsed_value is None:
                logger.warning("Failed to parse datetime for %s: %s", key, value)
                return None
            return parsed_value
        return value

    def _check_completion(self, key: str, value: Any) -> bool:
        """Check if experiment has completed and signal if needed.

        Signals completion for "completed", "error", and "aborted" statuses.
        The shutdown monitor will check the status and decide whether to
        actually shut down (aborted experiments keep the server running).

        Args:
            key: State key name
            value: State value

        Returns:
            True if completion was detected, False otherwise
        """
        if key != "status":
            return False
        if value not in ("completed", "error", "aborted"):
            return False
        if not self.experiment_completed:
            return False

        status_msg_map = {
            "completed": "completed",
            "error": "errored",
            "aborted": "aborted",
        }
        status_msg = status_msg_map[value]
        logger.info(
            "Experiment %s - shutdown monitor will check status",
            status_msg,
        )
        self.experiment_completed.set()
        return True

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        """Parse ISO format datetime string.

        Args:
            value: ISO format datetime string

        Returns:
            Parsed datetime or None if parsing fails
        """
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
