"""Builds state update field dictionaries."""

from __future__ import annotations

from typing import Any

from .experiment_config import CoreStateFields  # noqa: TC001


class StateUpdateFields:
    """Builds state update field dictionaries."""

    @staticmethod
    def build_core_fields(fields: CoreStateFields) -> dict[str, Any]:
        """Build core state update fields.

        Args:
            fields: Core state fields

        Returns:
            Dictionary with core fields
        """
        return {
            "run_name": fields.run_name,
            "experiment_name": fields.metadata["experiment_name"],
            "environment_name": fields.metadata["environment_name"],
            "config_path": fields.metadata["config_path"],
            "experiment_start_time": fields.experiment_start_time.isoformat(),
            "status": fields.status,
            "max_train_steps": fields.limits.max_train_steps,
            "max_eval_steps": fields.limits.max_eval_steps,
            "max_total_steps": fields.limits.max_total_steps,
            "n_train_epochs": fields.limits.n_train_epochs,
            "n_eval_epochs": fields.limits.n_eval_epochs,
            "do_train": fields.training.do_train,
            "do_eval": fields.evaluation.do_eval,
            "setup_message": fields.setup_message,
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
