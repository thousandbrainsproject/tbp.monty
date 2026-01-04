"""Extract and normalize experiment configuration settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import ConfigDict
else:
    from .types import ConfigDict  # noqa: TC001


class ExperimentConfigExtractor:
    """Extracts and normalizes configuration for LiveView experiments."""

    @staticmethod
    def extract_liveview_config(
        config: ConfigDict,
        liveview_port: int | None = None,
        liveview_host: str | None = None,
        enable_liveview: bool | None = None,
        zmq_port: int | None = None,
    ) -> dict[str, Any]:
        """Extract LiveView configuration from config dict or parameters.

        Args:
            config: Experiment configuration dictionary
            liveview_port: Override port for LiveView server
            liveview_host: Override host for LiveView server
            enable_liveview: Override enable flag
            zmq_port: Override ZMQ port

        Returns:
            Dictionary with normalized configuration values
        """
        if hasattr(config, "get"):
            return ExperimentConfigExtractor._extract_from_dict(
                config, liveview_port, liveview_host, enable_liveview, zmq_port
            )
        return ExperimentConfigExtractor._extract_defaults(
            liveview_port, liveview_host, enable_liveview, zmq_port
        )

    @staticmethod
    def _extract_from_dict(
        config: ConfigDict,
        liveview_port: int | None,
        liveview_host: str | None,
        enable_liveview: bool | None,
        zmq_port: int | None,
    ) -> dict[str, Any]:
        """Extract configuration from dict-like config object.

        Args:
            config: Config dictionary
            liveview_port: Override port
            liveview_host: Override host
            enable_liveview: Override enable flag
            zmq_port: Override ZMQ port

        Returns:
            Normalized configuration dictionary
        """
        result: dict[str, Any] = {
            "liveview_port": (
                liveview_port
                if liveview_port is not None
                else config.get("liveview_port", 8000)
            ),
            "liveview_host": (
                liveview_host
                if liveview_host is not None
                else config.get("liveview_host", "127.0.0.1")
            ),
            "enable_liveview": (
                enable_liveview
                if enable_liveview is not None
                else config.get("enable_liveview", True)
            ),
            "zmq_port": (
                zmq_port if zmq_port is not None else config.get("zmq_port", 5555)
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
        liveview_port: int | None,
        liveview_host: str | None,
        enable_liveview: bool | None,
        zmq_port: int | None,
    ) -> dict[str, Any]:
        """Extract configuration using defaults when config is not dict-like.

        Args:
            liveview_port: Override port
            liveview_host: Override host
            enable_liveview: Override enable flag
            zmq_port: Override ZMQ port

        Returns:
            Normalized configuration dictionary
        """
        return {
            "liveview_port": liveview_port if liveview_port is not None else 8000,
            "liveview_host": (
                liveview_host if liveview_host is not None else "127.0.0.1"
            ),
            "enable_liveview": enable_liveview if enable_liveview is not None else True,
            "zmq_port": zmq_port if zmq_port is not None else 5555,
            "zmq_host": "127.0.0.1",  # Default to localhost for ZMQ
        }
