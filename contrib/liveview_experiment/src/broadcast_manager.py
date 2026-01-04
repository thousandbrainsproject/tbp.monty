"""Manages broadcast updates with throttling."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state_manager import ExperimentStateManager

logger = logging.getLogger(__name__)

try:
    from pyview.live_socket import pub_sub_hub
    from pyview.vendor.flet.pubsub import PubSub
except ImportError:
    from .pubsub_compat import PubSub, pub_sub_hub


class BroadcastManager:
    """Manages broadcast updates with throttling and fallback mechanisms."""

    def __init__(
        self,
        state_manager: ExperimentStateManager,
        throttle_seconds: float = 1.0,
    ) -> None:
        """Initialize broadcast manager.

        Args:
            state_manager: Experiment state manager
            throttle_seconds: Minimum seconds between broadcasts

        """
        self.state_manager = state_manager
        self._throttle_seconds = throttle_seconds
        self._last_broadcast_time = 0.0
        self._pending_broadcast = False

    async def broadcast_if_needed(self) -> None:
        """Broadcast update if throttle period has passed.

        Checks throttle and broadcasts via pubsub and direct socket calls.
        """
        if not self._should_broadcast():
            self._pending_broadcast = True
            return

        self._reset_throttle()
        await self._perform_broadcast()

    def _should_broadcast(self) -> bool:
        """Check if enough time has passed since last broadcast.

        Returns:
            True if broadcast should proceed

        """
        current_time = time.time()
        time_since_last = current_time - self._last_broadcast_time
        return time_since_last >= self._throttle_seconds

    def _reset_throttle(self) -> None:
        """Reset throttle timer and pending flag."""
        self._last_broadcast_time = time.time()
        self._pending_broadcast = False

    async def _perform_broadcast(self) -> None:
        """Perform the actual broadcast via pubsub and direct calls."""
        try:
            await self._broadcast_via_pubsub()
            await self._broadcast_to_sockets()
        except (AttributeError, RuntimeError, TypeError):
            logger.exception("Failed to broadcast update via pubsub")

    async def _broadcast_via_pubsub(self) -> None:
        """Broadcast via PyView's pubsub system."""
        pubsub = PubSub(pub_sub_hub, self.state_manager.broadcast_topic)
        await pubsub.send_all_on_topic_async(
            self.state_manager.broadcast_topic,
            "update",
        )

    async def _broadcast_to_sockets(self) -> None:
        """Manually trigger handle_info on all connected sockets as fallback."""
        if not self.state_manager.liveview_instance:
            return

        for socket in list(self.state_manager.connected_sockets):
            try:
                await self.state_manager.liveview_instance.handle_info("update", socket)
            except (AttributeError, RuntimeError) as e:
                logger.debug("Failed to manually trigger handle_info on socket: %s", e)
