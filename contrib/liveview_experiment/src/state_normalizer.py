"""Normalizes experiment state values for template rendering."""

from __future__ import annotations

from .experiment_state import ExperimentState


class StateNormalizer:
    """Normalizes experiment state values for template rendering."""

    @staticmethod
    def normalize(state: ExperimentState, normalized_status: str) -> ExperimentState:
        """Create normalized state copy for socket context.

        Args:
            state: Source experiment state
            normalized_status: Normalized status string

        Returns:
            New normalized ExperimentState instance
        """
        return ExperimentState(
            # Numeric values - preserve 0, default to 0 if None
            total_train_steps=StateNormalizer._safe_int(state.total_train_steps),
            train_episodes=StateNormalizer._safe_int(state.train_episodes),
            train_epochs=StateNormalizer._safe_int(state.train_epochs),
            total_eval_steps=StateNormalizer._safe_int(state.total_eval_steps),
            eval_episodes=StateNormalizer._safe_int(state.eval_episodes),
            eval_epochs=StateNormalizer._safe_int(state.eval_epochs),
            current_epoch=StateNormalizer._safe_int(state.current_epoch),
            current_episode=StateNormalizer._safe_int(state.current_episode),
            current_step=StateNormalizer._safe_int(state.current_step),
            learning_module_count=StateNormalizer._safe_int(
                state.learning_module_count
            ),
            sensor_module_count=StateNormalizer._safe_int(state.sensor_module_count),
            # String values - preserve empty strings, provide defaults if None
            experiment_mode=StateNormalizer._safe_str(state.experiment_mode, "train"),
            run_name=StateNormalizer._safe_str(state.run_name, "Experiment"),
            experiment_name=StateNormalizer._safe_str(state.experiment_name),
            environment_name=StateNormalizer._safe_str(state.environment_name),
            config_path=StateNormalizer._safe_str(state.config_path),
            model_name_or_path=StateNormalizer._safe_str(state.model_name_or_path),
            error_message=StateNormalizer._safe_str(state.error_message),
            setup_message=StateNormalizer._safe_str(state.setup_message),
            # Optional values - keep None for template conditionals
            experiment_start_time=state.experiment_start_time,
            last_update=state.last_update,
            max_train_steps=state.max_train_steps,
            max_eval_steps=state.max_eval_steps,
            max_total_steps=state.max_total_steps,
            n_train_epochs=state.n_train_epochs,
            n_eval_epochs=state.n_eval_epochs,
            seed=state.seed,
            model_path=state.model_path,
            min_lms_match=state.min_lms_match,
            # Boolean values - explicit None checks
            do_train=StateNormalizer._safe_bool(state.do_train),
            do_eval=StateNormalizer._safe_bool(state.do_eval),
            show_sensor_output=StateNormalizer._safe_bool(state.show_sensor_output),
            # Complex values - always provide defaults
            metrics=state.metrics.copy() if state.metrics is not None else {},
            data_streams=(
                state.data_streams.copy() if state.data_streams is not None else {}
            ),
            recent_logs=(
                state.recent_logs.copy() if state.recent_logs is not None else []
            ),
            max_log_history=StateNormalizer._safe_int(state.max_log_history) or 100,
            # Status - always defined
            status=normalized_status,
        )

    @staticmethod
    def _safe_int(value: int | None) -> int:
        """Convert value to int, defaulting to 0 if None.

        Args:
            value: Value to convert

        Returns:
            Integer value or 0
        """
        return value if value is not None else 0

    @staticmethod
    def _safe_str(value: str | None, default: str = "") -> str:
        """Convert value to string, defaulting to default if None.

        Args:
            value: Value to convert
            default: Default value if None

        Returns:
            String value or default
        """
        return value if value is not None else default

    @staticmethod
    def _safe_bool(value: bool | None) -> bool:
        """Convert value to bool, defaulting to False if None.

        Args:
            value: Value to convert

        Returns:
            Boolean value or False
        """
        return value if value is not None else False
