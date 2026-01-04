"""Builds state update dictionaries for experiment broadcasting."""

from __future__ import annotations

from typing import Any


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
        training_info = (
            f"Training: {'Yes' if do_train else 'No'} "
            f"(max steps: {max_train_steps}, epochs: {n_train_epochs})"
        )
        eval_info = (
            f"Evaluation: {'Yes' if do_eval else 'No'} "
            f"(max steps: {max_eval_steps}, epochs: {n_eval_epochs})"
        )
        setup_lines = [
            f"Experiment: {metadata['experiment_name']}",
            f"Environment: {metadata['environment_name']}",
            f"Config: {metadata['config_path']}",
            f"Run: {run_name}",
            training_info,
            eval_info,
        ]

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
        state_update = {
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
            "show_sensor_output": (
                config.get("show_sensor_output", False)
                if hasattr(config, "get")
                else False
            ),
            "seed": config.get("seed") if hasattr(config, "get") else None,
            "model_name_or_path": (
                config.get("model_name_or_path") if hasattr(config, "get") else None
            ),
            "min_lms_match": (
                config.get("min_lms_match") if hasattr(config, "get") else None
            ),
            "setup_message": setup_message,
        }

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
