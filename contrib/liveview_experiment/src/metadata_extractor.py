"""Extract experiment metadata from various sources."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .liveview_types import ConfigDict

logger = logging.getLogger(__name__)

try:
    from hydra.core.global_hydra import GlobalHydra

    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    GlobalHydra = None  # type: ignore[assignment, unused-ignore, misc]


class ExperimentMetadata:
    """Experiment metadata container."""

    def __init__(
        self,
        environment_name: str = "",
        experiment_name: str = "",
        config_path: str = "",
    ) -> None:
        """Initialize metadata.

        Args:
            environment_name: Conda environment name
            experiment_name: Experiment name
            config_path: Path to experiment config file
        """
        self.environment_name = environment_name
        self.experiment_name = experiment_name
        self.config_path = config_path

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary.

        Returns:
            Dictionary with metadata fields
        """
        return {
            "environment_name": self.environment_name,
            "experiment_name": self.experiment_name,
            "config_path": self.config_path,
        }


class MetadataExtractor:
    """Extracts experiment metadata from Hydra, config, and environment."""

    def __init__(
        self, config: ConfigDict | None = None, run_name: str | None = None
    ) -> None:
        """Initialize the metadata extractor.

        Args:
            config: Experiment configuration object
            run_name: Optional run name as fallback
        """
        self.config = config
        self.run_name = run_name or ""

    def extract(self) -> ExperimentMetadata:
        """Extract all available metadata.

        Returns:
            ExperimentMetadata instance with all available information
        """
        environment_name = self._extract_environment_name()
        experiment_name = self._extract_experiment_name()
        config_path = self._extract_config_path()

        return ExperimentMetadata(
            environment_name=environment_name,
            experiment_name=experiment_name,
            config_path=config_path,
        )

    def _extract_environment_name(self) -> str:
        """Extract environment name from environment variables.

        Returns:
            Environment name or empty string
        """
        return os.environ.get("CONDA_DEFAULT_ENV", "")

    def _extract_experiment_name(self) -> str:
        """Extract experiment name from Hydra or config.

        Returns:
            Experiment name or empty string
        """
        # Try Hydra first
        name = self._extract_from_hydra_choices()
        if name:
            return name

        # Try config
        name = self._extract_from_config()
        if name:
            return name

        # Fallback to run_name
        return self.run_name

    def _extract_from_hydra_choices(self) -> str:
        """Extract experiment name from Hydra runtime choices.

        Returns:
            Experiment name or empty string
        """
        if not HYDRA_AVAILABLE or GlobalHydra is None:
            return ""

        try:
            hydra_instance = GlobalHydra.instance()
            if (
                hydra_instance is not None
                and hydra_instance.hydra is not None
                and hasattr(hydra_instance.hydra, "runtime")
                and hasattr(hydra_instance.hydra.runtime, "choices")
                and "experiment" in hydra_instance.hydra.runtime.choices
            ):
                return str(hydra_instance.hydra.runtime.choices["experiment"])
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not get experiment name from Hydra: %s", e)

        return ""

    def _extract_from_config(self) -> str:
        """Extract experiment name from config object.

        Returns:
            Experiment name or empty string
        """
        if self.config is None or not hasattr(self.config, "get"):
            return ""

        exp_config = self.config.get("experiment", {})
        if isinstance(exp_config, dict) and "_name_" in exp_config:
            return str(exp_config["_name_"])

        return ""

    def _extract_config_path(self) -> str:
        """Extract config path from Hydra config sources.

        Returns:
            Config path or empty string
        """
        if not HYDRA_AVAILABLE or GlobalHydra is None:
            return ""

        try:
            hydra_instance = GlobalHydra.instance()
            config_sources = self._get_config_sources(hydra_instance)
            if not config_sources:
                return ""

            # Look for experiment config file first
            experiment_path = self._find_experiment_config(config_sources)
            if experiment_path:
                return experiment_path

            # Fallback to any config file
            return self._find_any_config_file(config_sources)
        except (AttributeError, RuntimeError) as e:
            logger.debug("Could not get config path from Hydra: %s", e)

        return ""

    @staticmethod
    def _get_config_sources(hydra_instance: Any) -> list[Any] | None:
        """Get config sources from Hydra instance.

        Args:
            hydra_instance: GlobalHydra instance

        Returns:
            List of config sources or None
        """
        if (
            hydra_instance is None
            or hydra_instance.hydra is None
            or not hasattr(hydra_instance.hydra, "runtime")
            or not hasattr(hydra_instance.hydra.runtime, "config_sources")
        ):
            return None

        return list(hydra_instance.hydra.runtime.config_sources)

    @staticmethod
    def _find_experiment_config(config_sources: list[Any]) -> str:
        """Find experiment config file in sources.

        Args:
            config_sources: List of config source objects

        Returns:
            Config path or empty string
        """
        for source in config_sources:
            if not hasattr(source, "path"):
                continue

            path_str = str(source.path)
            if "experiment" in path_str and path_str.endswith((".yaml", ".yml")):
                return path_str

        return ""

    @staticmethod
    def _find_any_config_file(config_sources: list[Any]) -> str:
        """Find any config file in sources.

        Args:
            config_sources: List of config source objects

        Returns:
            Config path or empty string
        """
        for source in config_sources:
            path_str = str(getattr(source, "path", ""))
            if path_str and path_str.endswith((".yaml", ".yml")):
                return path_str

        return ""
