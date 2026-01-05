"""Extract and normalize experiment configuration settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ConfigDict

from .experiment_config import LiveViewConfigOverrides


class ExperimentConfigExtractor:
    """Extracts and normalizes configuration for LiveView experiments."""

    @staticmethod
    def extract_liveview_config(
        config: ConfigDict,
        overrides: LiveViewConfigOverrides | None = None,
    ) -> dict[str, Any]:
        """Extract LiveView configuration from config dict or parameters.

        Args:
            config: Experiment configuration dictionary
            overrides: Optional configuration overrides

        Returns:
            Dictionary with normalized configuration values
        """
        if overrides is None:
            overrides = LiveViewConfigOverrides()

        if hasattr(config, "get"):
            return ExperimentConfigExtractor._extract_from_dict(config, overrides)
        return ExperimentConfigExtractor._extract_defaults(overrides)

    @staticmethod
    def _extract_from_dict(
        config: ConfigDict,
        overrides: LiveViewConfigOverrides,
    ) -> dict[str, Any]:
        """Extract configuration from dict-like config object.

        Args:
            config: Config dictionary
            overrides: Configuration overrides

        Returns:
            Normalized configuration dictionary
        """
        result: dict[str, Any] = {
            "liveview_port": (
                overrides.liveview_port
                if overrides.liveview_port is not None
                else config.get("liveview_port", 8000)
            ),
            "liveview_host": (
                overrides.liveview_host
                if overrides.liveview_host is not None
                else config.get("liveview_host", "127.0.0.1")
            ),
            "enable_liveview": (
                overrides.enable_liveview
                if overrides.enable_liveview is not None
                else config.get("enable_liveview", True)
            ),
            "zmq_port": (
                overrides.zmq_port
                if overrides.zmq_port is not None
                else config.get("zmq_port", 5555)
            ),
        }

        # Extract zmq_host with normalization
        zmq_host_from_config = config.get("zmq_host", None)
        if zmq_host_from_config:
            result["zmq_host"] = zmq_host_from_config
        else:
            result["zmq_host"] = result["liveview_host"]

        # Normalize localhost to 127.0.0.1 for ZMQ
        if not result["zmq_host"] or result["zmq_host"] == "localhost":
            result["zmq_host"] = "127.0.0.1"

        return result

    @staticmethod
    def _extract_defaults(
        overrides: LiveViewConfigOverrides,
    ) -> dict[str, Any]:
        """Extract configuration using defaults when config is not dict-like.

        Args:
            overrides: Configuration overrides

        Returns:
            Normalized configuration dictionary
        """
        return {
            "liveview_port": (
                overrides.liveview_port if overrides.liveview_port is not None else 8000
            ),
            "liveview_host": (
                overrides.liveview_host
                if overrides.liveview_host is not None
                else "127.0.0.1"
            ),
            "enable_liveview": (
                overrides.enable_liveview
                if overrides.enable_liveview is not None
                else True
            ),
            "zmq_port": (
                overrides.zmq_port if overrides.zmq_port is not None else 5555
            ),
            "zmq_host": "127.0.0.1",  # Default to localhost for ZMQ
        }
