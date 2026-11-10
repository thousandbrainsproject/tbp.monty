"""Extracts configuration values safely."""

from __future__ import annotations

from typing import Any


class ConfigValueExtractor:
    """Extracts configuration values safely."""

    @staticmethod
    def safe_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get value from config dictionary.

        Args:
            config: Configuration dictionary
            key: Key to get
            default: Default value if key not found or config has no get method

        Returns:
            Config value or default
        """
        if hasattr(config, "get"):
            return config.get(key, default)
        return default

    @staticmethod
    def extract_optional_config_values(
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract optional configuration values.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary with optional config values
        """
        return {
            "show_sensor_output": ConfigValueExtractor.safe_get(
                config, "show_sensor_output", False
            ),
            "seed": ConfigValueExtractor.safe_get(config, "seed"),
            "model_name_or_path": ConfigValueExtractor.safe_get(
                config, "model_name_or_path"
            ),
            "min_lms_match": ConfigValueExtractor.safe_get(config, "min_lms_match"),
        }
