"""Builds state update field dictionaries."""

from __future__ import annotations

from typing import Any


class StateUpdateFields:
    """Builds state update field dictionaries."""

    @staticmethod
    def build_core_fields(
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
        setup_message: str,
    ) -> dict[str, Any]:
        """Build core state update fields.

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
            setup_message: Formatted setup message

        Returns:
            Dictionary with core fields
        """
        return {
            "run_name": run_name,
            "experiment_name": metadata["experiment_name"],
            "environment_name": metadata["environment_name"],
            "config_path": metadata["config_path"],
            "experiment_start_time": experiment_start_time.isoformat(),
            "status": status,
            "max_train_steps": max_train_steps,
            "max_eval_steps": max_eval_steps,
            "max_total_steps": max_total_steps,
            "n_train_epochs": n_train_epochs,
            "n_eval_epochs": n_eval_epochs,
            "do_train": do_train,
            "do_eval": do_eval,
            "setup_message": setup_message,
        }

    @staticmethod
    def build_config_fields(config: dict[str, Any]) -> dict[str, Any]:
        """Build configuration-related fields.

        Args:
            config: Experiment configuration

        Returns:
            Dictionary with config fields
        """
        has_get = hasattr(config, "get")
        return {
            "show_sensor_output": (
                config.get("show_sensor_output", False) if has_get else False
            ),
            "seed": config.get("seed") if has_get else None,
            "model_name_or_path": (
                config.get("model_name_or_path") if has_get else None
            ),
            "min_lms_match": config.get("min_lms_match") if has_get else None,
        }
