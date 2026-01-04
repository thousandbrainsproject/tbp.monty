"""Experiment state dataclass for LiveView."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class ExperimentState:
    """State for the experiment LiveView."""

    # Experiment progress
    total_train_steps: int = 0
    train_episodes: int = 0
    train_epochs: int = 0
    total_eval_steps: int = 0
    eval_episodes: int = 0
    eval_epochs: int = 0

    # Current mode
    experiment_mode: str = "train"  # "train" or "eval"
    current_epoch: int = 0
    current_episode: int = 0
    current_step: int = 0

    # Experiment info
    run_name: str = ""
    # Name of the experiment config
    # (e.g., "tutorial_surf_agent_2obj_with_liveview")
    experiment_name: str = ""
    environment_name: str = ""  # Conda environment name (e.g., "tbp.monty")
    config_path: str = ""  # Path to the experiment config file
    experiment_start_time: datetime | None = None
    last_update: datetime | None = None

    # Experiment configuration
    max_train_steps: int | None = None
    max_eval_steps: int | None = None
    max_total_steps: int | None = None
    n_train_epochs: int | None = None
    n_eval_epochs: int | None = None
    do_train: bool = False
    do_eval: bool = False
    show_sensor_output: bool = False
    seed: int | None = None
    model_path: str | None = None
    model_name_or_path: str | None = None
    min_lms_match: int | None = None

    # Model info
    learning_module_count: int = 0
    sensor_module_count: int = 0

    # Performance metrics (can be extended)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Streaming data from parallel processes
    # Maps stream names to their latest data
    data_streams: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Recent log messages (for display)
    recent_logs: list[dict[str, Any]] = field(default_factory=list)
    max_log_history: int = 100  # Maximum number of log messages to keep

    # Status
    status: str = "initializing"  # "initializing", "running", "completed", "error"
    error_message: str | None = None

    # Setup message (cached initial configuration for display)
    setup_message: str | None = None
