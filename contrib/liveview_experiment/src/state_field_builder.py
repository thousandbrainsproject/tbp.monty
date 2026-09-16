"""Builds individual field groups for state normalization."""

from __future__ import annotations

from typing import Any

from .experiment_state import ExperimentState  # noqa: TC001


class StateFieldBuilder:
    """Builds individual field groups for state normalization."""

    @staticmethod
    def build_numeric_fields(state: ExperimentState) -> dict[str, int]:
        """Build numeric fields with safe defaults.

        Args:
            state: Source experiment state

        Returns:
            Dictionary of numeric fields
        """
        from .state_normalizer import StateNormalizer  # noqa: PLC0415

        safe_int = StateNormalizer._safe_int
        return {
            "total_train_steps": safe_int(state.total_train_steps),
            "train_episodes": safe_int(state.train_episodes),
            "train_epochs": safe_int(state.train_epochs),
            "total_eval_steps": safe_int(state.total_eval_steps),
            "eval_episodes": safe_int(state.eval_episodes),
            "eval_epochs": safe_int(state.eval_epochs),
            "current_epoch": safe_int(state.current_epoch),
            "current_episode": safe_int(state.current_episode),
            "current_step": safe_int(state.current_step),
            "learning_module_count": safe_int(state.learning_module_count),
            "sensor_module_count": safe_int(state.sensor_module_count),
        }

    @staticmethod
    def build_string_fields(state: ExperimentState) -> dict[str, str]:
        """Build string fields with safe defaults.

        Args:
            state: Source experiment state

        Returns:
            Dictionary of string fields
        """
        from .state_normalizer import StateNormalizer  # noqa: PLC0415

        safe_str = StateNormalizer._safe_str
        return {
            "experiment_mode": safe_str(state.experiment_mode, "train"),
            "run_name": safe_str(state.run_name, "Experiment"),
            "experiment_name": safe_str(state.experiment_name),
            "environment_name": safe_str(state.environment_name),
            "config_path": safe_str(state.config_path),
            "model_name_or_path": safe_str(state.model_name_or_path),
            "error_message": safe_str(state.error_message),
            "setup_message": safe_str(state.setup_message),
        }

    @staticmethod
    def build_optional_fields(state: ExperimentState) -> dict[str, Any]:
        """Build optional fields (preserve None).

        Args:
            state: Source experiment state

        Returns:
            Dictionary of optional fields
        """
        return {
            "experiment_start_time": state.experiment_start_time,
            "last_update": state.last_update,
            "max_train_steps": state.max_train_steps,
            "max_eval_steps": state.max_eval_steps,
            "max_total_steps": state.max_total_steps,
            "n_train_epochs": state.n_train_epochs,
            "n_eval_epochs": state.n_eval_epochs,
            "seed": state.seed,
            "model_path": state.model_path,
            "min_lms_match": state.min_lms_match,
        }

    @staticmethod
    def build_boolean_fields(state: ExperimentState) -> dict[str, bool]:
        """Build boolean fields with safe defaults.

        Args:
            state: Source experiment state

        Returns:
            Dictionary of boolean fields
        """
        from .state_normalizer import StateNormalizer  # noqa: PLC0415

        safe_bool = StateNormalizer._safe_bool
        return {
            "do_train": safe_bool(state.do_train),
            "do_eval": safe_bool(state.do_eval),
            "show_sensor_output": safe_bool(state.show_sensor_output),
        }

    @staticmethod
    def build_complex_fields(state: ExperimentState) -> dict[str, Any]:
        """Build complex fields (dicts, lists) with safe defaults.

        Args:
            state: Source experiment state

        Returns:
            Dictionary of complex fields
        """
        from .state_normalizer import StateNormalizer  # noqa: PLC0415

        safe_int = StateNormalizer._safe_int
        return {
            "metrics": state.metrics.copy() if state.metrics is not None else {},
            "data_streams": (
                state.data_streams.copy() if state.data_streams is not None else {}
            ),
            "recent_logs": (
                state.recent_logs.copy() if state.recent_logs is not None else []
            ),
            "max_log_history": safe_int(state.max_log_history) or 100,
        }
