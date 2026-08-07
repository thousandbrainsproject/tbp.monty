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


@dataclass
class LiveViewConfigOverrides:
    """Optional overrides for LiveView configuration."""

    liveview_port: int | None = None
    liveview_host: str | None = None
    enable_liveview: bool | None = None
    zmq_port: int | None = None


@dataclass
class BroadcasterSetupParams:
    """Parameters for setting up ZMQ broadcaster."""

    zmq_port: int
    zmq_host: str
    run_name: str
    metadata: dict[str, str]


@dataclass
class ConnectionRetryParams:
    """Parameters for ZMQ connection with retry logic."""

    zmq_host: str
    zmq_port: int
    max_retries: int = 10
    retry_delay: float = 0.5


@dataclass
class ServerConfig:
    """Configuration for LiveView server."""

    host: str
    port: int
    zmq_port: int
    zmq_host: str


@dataclass
class ZmqSubscriberParams:
    """Parameters for ZMQ subscriber setup."""

    zmq_port: int
    zmq_host: str


@dataclass
class ZmqSubscriberRunParams:
    """Parameters for running ZMQ subscriber."""

    zmq_context: Any  # zmq.Context
    zmq_port: int
    zmq_host: str
    experiment_completed: Any | None = None  # asyncio.Event | None
