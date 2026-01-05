"""Builds setup message lines for experiment configuration."""

from __future__ import annotations

from typing import Any


class SetupMessageBuilder:
    """Builds setup message lines for experiment configuration."""

    @staticmethod
    def build_core_lines(
        metadata: dict[str, str],
        run_name: str,
        do_train: bool,
        max_train_steps: int,
        n_train_epochs: int,
        do_eval: bool,
        max_eval_steps: int,
        n_eval_epochs: int,
    ) -> list[str]:
        """Build core setup message lines.

        Args:
            metadata: Experiment metadata
            run_name: Experiment run name
            do_train: Whether training is enabled
            max_train_steps: Maximum training steps
            n_train_epochs: Number of training epochs
            do_eval: Whether evaluation is enabled
            max_eval_steps: Maximum evaluation steps
            n_eval_epochs: Number of evaluation epochs

        Returns:
            List of core setup message lines
        """
        training_info = (
            f"Training: {'Yes' if do_train else 'No'} "
            f"(max steps: {max_train_steps}, epochs: {n_train_epochs})"
        )
        eval_info = (
            f"Evaluation: {'Yes' if do_eval else 'No'} "
            f"(max steps: {max_eval_steps}, epochs: {n_eval_epochs})"
        )
        return [
            f"Experiment: {metadata['experiment_name']}",
            f"Environment: {metadata['environment_name']}",
            f"Config: {metadata['config_path']}",
            f"Run: {run_name}",
            training_info,
            eval_info,
        ]

    @staticmethod
    def add_optional_lines(
        setup_lines: list[str], config: dict[str, Any], model_path: str | None
    ) -> None:
        """Add optional configuration lines to setup message.

        Args:
            setup_lines: List to append to
            config: Experiment configuration
            model_path: Optional model path
        """
        if config.get("seed") is not None:
            setup_lines.append(f"Seed: {config.get('seed')}")
        if config.get("model_name_or_path"):
            setup_lines.append(f"Model: {config.get('model_name_or_path')}")
        if model_path:
            setup_lines.append(f"Model Path: {model_path}")
        if config.get("min_lms_match") is not None:
            setup_lines.append(f"Min LMs Match: {config.get('min_lms_match')}")
        if config.get("show_sensor_output") is not None:
            setup_lines.append(
                f"Show Sensor Output: {config.get('show_sensor_output')}"
            )
