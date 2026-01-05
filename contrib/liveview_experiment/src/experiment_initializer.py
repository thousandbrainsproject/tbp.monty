"""Initializes experiment with LiveView configuration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .broadcaster_initializer import BroadcasterInitializer
from .experiment_config import (  # noqa: TC001
    BroadcasterSetupParams,
    LiveViewConfigOverrides,
)
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
        overrides: LiveViewConfigOverrides | None = None,
    ) -> dict[str, Any]:
        """Extract and normalize LiveView configuration.

        Args:
            config: Experiment configuration
            overrides: Optional configuration overrides

        Returns:
            Dictionary with normalized configuration values
        """
        return ExperimentConfigExtractor.extract_liveview_config(config, overrides)

    @staticmethod
    def setup_broadcaster(
        experiment: Any,
        params: BroadcasterSetupParams,
    ) -> ZmqBroadcaster | None:
        """Set up ZMQ broadcaster and broadcast initial state.

        Args:
            experiment: Experiment instance
            params: Broadcaster setup parameters

        Returns:
            Broadcaster instance or None if setup failed
        """
        try:
            broadcaster = ZmqBroadcaster(
                zmq_port=params.zmq_port, zmq_host=params.zmq_host
            )
            experiment._experiment_start_time = datetime.now(timezone.utc)
            experiment._experiment_status = "initializing"

            BroadcasterInitializer.initialize_and_broadcast(
                broadcaster, params.run_name, params.metadata
            )

            return broadcaster
        except (OSError, RuntimeError) as e:
            logger.warning(
                "Failed to initialize ZMQ broadcaster: %s. "
                "Continuing without LiveView updates.",
                e,
            )
            return None
