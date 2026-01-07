"""ZMQ message handler for processing experiment updates."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

    from .state_manager import ExperimentStateManager
    from .types import MessagePayload

logger = logging.getLogger(__name__)


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

    async def _handle_state(self, payload: MessagePayload) -> None:
        """Handle state update message from ZMQ.

        Args:
            payload: State message payload
        """
        # Check if this is a new experiment (run_name changed)
        new_run_name = payload.get("run_name")
        if (
            new_run_name
            and new_run_name != self.state_manager.experiment_state.run_name
            and self.state_manager.experiment_state.run_name  # Not initial empty state
        ):
            logger.info(
                "New experiment detected: %s -> %s, resetting visualization state",
                self.state_manager.experiment_state.run_name,
                new_run_name,
            )
            self.state_manager.visualization.clear()

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
            self._check_completion(key, value)

        # Update last_update timestamp
        self.state_manager.experiment_state.last_update = datetime.now(timezone.utc)
        await self.state_manager.broadcast_update()

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

    def _check_completion(self, key: str, value: Any) -> None:
        """Check if experiment has completed and signal if needed.

        Args:
            key: State key name
            value: State value
        """
        if (
            key == "status"
            and value in ("completed", "error")
            and self.experiment_completed
        ):
            status_msg = "completed" if value == "completed" else "errored"
            logger.info(
                "Experiment %s - will linger for 1 minute before shutdown",
                status_msg,
            )
            self.experiment_completed.set()

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
