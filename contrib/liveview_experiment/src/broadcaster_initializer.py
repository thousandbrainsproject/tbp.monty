"""Initialize and configure ZMQ broadcaster for experiments."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .zmq_broadcaster import ZmqBroadcaster

logger = logging.getLogger(__name__)


class BroadcasterInitializer:
    """Handles initialization and initial state broadcast for ZMQ broadcaster."""

    @staticmethod
    def initialize_and_broadcast(
        broadcaster: ZmqBroadcaster,
        run_name: str,
        metadata: dict[str, str],
    ) -> None:
        """Initialize broadcaster and broadcast initial state.

        Args:
            broadcaster: ZMQ broadcaster instance
            run_name: Experiment run name
            metadata: Experiment metadata dictionary
        """
        try:
            broadcaster.connect()
            logger.info(
                "ZMQ broadcaster initialized on tcp://%s:%d",
                broadcaster.zmq_host,
                broadcaster.zmq_port,
            )

            BroadcasterInitializer._broadcast_initial_state(
                broadcaster, run_name, metadata
            )
            logger.info("Initial state broadcasted to LiveView")
        except (OSError, RuntimeError) as e:
            logger.warning(
                "Failed to initialize ZMQ broadcaster: %s. "
                "Continuing without LiveView updates.",
                e,
            )
            raise

    @staticmethod
    def _broadcast_initial_state(
        broadcaster: ZmqBroadcaster,
        run_name: str,
        metadata: dict[str, str],
    ) -> None:
        """Broadcast initial experiment state.

        Args:
            broadcaster: ZMQ broadcaster instance
            run_name: Experiment run name
            metadata: Experiment metadata
        """
        experiment_start_time = datetime.now(timezone.utc)
        broadcaster.publish_state(
            {
                "run_name": run_name,
                "experiment_name": metadata["experiment_name"],
                "environment_name": metadata["environment_name"],
                "config_path": metadata["config_path"],
                "experiment_start_time": experiment_start_time.isoformat(),
                "status": "initializing",
                "experiment_mode": "train",  # Default mode
            }
        )
