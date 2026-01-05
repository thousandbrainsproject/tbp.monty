"""Initializes experiment with LiveView configuration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .broadcaster_initializer import BroadcasterInitializer
from .experiment_config_extractor import ExperimentConfigExtractor
from .zmq_broadcaster import ZmqBroadcaster

if TYPE_CHECKING:
    from .types import ConfigDict

logger = logging.getLogger(__name__)


class ExperimentInitializer:
    """Initializes experiment with LiveView configuration."""

    @staticmethod
    def extract_config(
        config: ConfigDict,
        liveview_port: int | None,
        liveview_host: str | None,
        enable_liveview: bool | None,
        zmq_port: int | None,
    ) -> dict[str, Any]:
        """Extract and normalize LiveView configuration.

        Args:
            config: Experiment configuration
            liveview_port: Optional LiveView port override
            liveview_host: Optional LiveView host override
            enable_liveview: Optional enable flag override
            zmq_port: Optional ZMQ port override

        Returns:
            Dictionary with normalized configuration values
        """
        return ExperimentConfigExtractor.extract_liveview_config(
            config, liveview_port, liveview_host, enable_liveview, zmq_port
        )

    @staticmethod
    def setup_broadcaster(
        experiment: Any,
        zmq_port: int,
        zmq_host: str,
        run_name: str,
        metadata: dict[str, str],
    ) -> ZmqBroadcaster | None:
        """Set up ZMQ broadcaster and broadcast initial state.

        Args:
            experiment: Experiment instance
            zmq_port: ZMQ port
            zmq_host: ZMQ host
            run_name: Experiment run name
            metadata: Experiment metadata dictionary

        Returns:
            Broadcaster instance or None if setup failed
        """
        try:
            broadcaster = ZmqBroadcaster(zmq_port=zmq_port, zmq_host=zmq_host)
            experiment._experiment_start_time = datetime.now(timezone.utc)
            experiment._experiment_status = "initializing"

            BroadcasterInitializer.initialize_and_broadcast(
                broadcaster, run_name, metadata
            )

            return broadcaster
        except (OSError, RuntimeError) as e:
            logger.warning(
                "Failed to initialize ZMQ broadcaster: %s. "
                "Continuing without LiveView updates.",
                e,
            )
            return None
