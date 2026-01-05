"""Builds state update dictionaries for experiment broadcasting."""

from __future__ import annotations

from typing import Any

from .experiment_config import (
    CoreStateFields,
    EvaluationConfig,
    SetupMessageParams,
    TrainingConfig,
)
from .setup_message_builder import SetupMessageBuilder
from .state_update_fields import StateUpdateFields


class StateUpdateBuilder:
    """Builds state update dictionaries for experiment broadcasting."""

    @staticmethod
    def build_setup_message(
        metadata: dict[str, str],
        run_name: str,
        training: TrainingConfig,
        evaluation: EvaluationConfig,
        config: dict[str, Any],
        model_path: str | None = None,
    ) -> str:
        """Build formatted setup message string.

        Args:
            metadata: Experiment metadata dictionary
            run_name: Experiment run name
            training: Training configuration
            evaluation: Evaluation configuration
            config: Experiment configuration
            model_path: Optional model path

        Returns:
            Formatted setup message
        """
        params = SetupMessageParams(
            metadata=metadata,
            run_name=run_name,
            training=training,
            evaluation=evaluation,
            config=config,
            model_path=model_path,
        )
        return StateUpdateBuilder.build_setup_message_from_params(params)

    @staticmethod
    def build_setup_message_from_params(params: SetupMessageParams) -> str:
        """Build setup message from parameters dataclass."""
        setup_lines = SetupMessageBuilder.build_core_lines(
            params.metadata,
            params.run_name,
            params.training,
            params.evaluation,
        )
        SetupMessageBuilder.add_optional_lines(
            setup_lines, params.config, params.model_path
        )
        return "\n".join(setup_lines)

    @staticmethod
    def build_initial_state_update(
        core_fields: CoreStateFields,
        config: dict[str, Any],
        model_path: str | None = None,
    ) -> dict[str, Any]:
        """Build initial state update dictionary.

        Args:
            core_fields: Core state fields
            config: Experiment configuration
            model_path: Optional model path

        Returns:
            State update dictionary
        """
        state_update = StateUpdateFields.build_core_fields(core_fields)
        state_update.update(StateUpdateFields.build_config_fields(config))

        if model_path:
            state_update["model_path"] = str(model_path)

        return state_update

    @staticmethod
    def build_model_info_update(
        learning_module_count: int, sensor_module_count: int
    ) -> dict[str, int]:
        """Build model info update dictionary.

        Args:
            learning_module_count: Number of learning modules
            sensor_module_count: Number of sensor modules

        Returns:
            Model info update dictionary
        """
        return {
            "learning_module_count": learning_module_count,
            "sensor_module_count": sensor_module_count,
        }
