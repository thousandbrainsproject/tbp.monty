"""Builds state update dictionaries for experiment broadcasting."""

from __future__ import annotations

from typing import Any

from .setup_message_builder import SetupMessageBuilder
from .state_update_fields import StateUpdateFields


class StateUpdateBuilder:
    """Builds state update dictionaries for experiment broadcasting."""

    @staticmethod
    def build_setup_message(
        metadata: dict[str, str],
        run_name: str,
        do_train: bool,
        max_train_steps: int,
        n_train_epochs: int,
        do_eval: bool,
        max_eval_steps: int,
        n_eval_epochs: int,
        config: dict[str, Any],
        model_path: str | None = None,
    ) -> str:
        """Build formatted setup message string.

        Args:
            metadata: Experiment metadata
            run_name: Experiment run name
            do_train: Whether training is enabled
            max_train_steps: Maximum training steps
            n_train_epochs: Number of training epochs
            do_eval: Whether evaluation is enabled
            max_eval_steps: Maximum evaluation steps
            n_eval_epochs: Number of evaluation epochs
            config: Experiment configuration
            model_path: Optional model path

        Returns:
            Formatted setup message
        """
        setup_lines = SetupMessageBuilder.build_core_lines(
            metadata,
            run_name,
            do_train,
            max_train_steps,
            n_train_epochs,
            do_eval,
            max_eval_steps,
            n_eval_epochs,
        )
        SetupMessageBuilder.add_optional_lines(setup_lines, config, model_path)
        return "\n".join(setup_lines)

    @staticmethod
    def build_initial_state_update(
        run_name: str,
        metadata: dict[str, str],
        experiment_start_time: Any,  # datetime
        status: str,
        max_train_steps: int,
        max_eval_steps: int,
        max_total_steps: int,
        n_train_epochs: int,
        n_eval_epochs: int,
        do_train: bool,
        do_eval: bool,
        config: dict[str, Any],
        setup_message: str,
        model_path: str | None = None,
    ) -> dict[str, Any]:
        """Build initial state update dictionary.

        Args:
            run_name: Experiment run name
            metadata: Experiment metadata
            experiment_start_time: Experiment start time
            status: Experiment status
            max_train_steps: Maximum training steps
            max_eval_steps: Maximum evaluation steps
            max_total_steps: Maximum total steps
            n_train_epochs: Number of training epochs
            n_eval_epochs: Number of evaluation epochs
            do_train: Whether training is enabled
            do_eval: Whether evaluation is enabled
            config: Experiment configuration
            setup_message: Formatted setup message
            model_path: Optional model path

        Returns:
            State update dictionary
        """
        state_update = StateUpdateFields.build_core_fields(
            run_name,
            metadata,
            experiment_start_time,
            status,
            max_train_steps,
            max_eval_steps,
            max_total_steps,
            n_train_epochs,
            n_eval_epochs,
            do_train,
            do_eval,
            setup_message,
        )
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
