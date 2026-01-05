"""Configuration dataclasses for experiment setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    do_train: bool
    max_train_steps: int
    n_train_epochs: int


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""

    do_eval: bool
    max_eval_steps: int
    n_eval_epochs: int


@dataclass
class ExperimentLimits:
    """Experiment step and epoch limits."""

    max_train_steps: int
    max_eval_steps: int
    max_total_steps: int
    n_train_epochs: int
    n_eval_epochs: int


@dataclass
class SetupMessageParams:
    """Parameters for building setup message."""

    metadata: dict[str, str]
    run_name: str
    training: TrainingConfig
    evaluation: EvaluationConfig
    config: dict[str, Any]
    model_path: str | None = None


@dataclass
class CoreStateFields:
    """Core fields for state update."""

    run_name: str
    metadata: dict[str, str]
    experiment_start_time: datetime
    status: str
    limits: ExperimentLimits
    training: TrainingConfig
    evaluation: EvaluationConfig
    setup_message: str
