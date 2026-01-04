"""Experiment state dataclass for LiveView."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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
    experiment_name: str = ""  # Name of the experiment config (e.g., "tutorial_surf_agent_2obj_with_liveview")
    environment_name: str = ""  # Conda environment name (e.g., "tbp.monty")
    config_path: str = ""  # Path to the experiment config file
    experiment_start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # Experiment configuration
    max_train_steps: Optional[int] = None
    max_eval_steps: Optional[int] = None
    max_total_steps: Optional[int] = None
    n_train_epochs: Optional[int] = None
    n_eval_epochs: Optional[int] = None
    do_train: bool = False
    do_eval: bool = False
    show_sensor_output: bool = False
    seed: Optional[int] = None
    model_path: Optional[str] = None
    model_name_or_path: Optional[str] = None
    min_lms_match: Optional[int] = None
    
    # Model info
    learning_module_count: int = 0
    sensor_module_count: int = 0
    
    # Performance metrics (can be extended)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Streaming data from parallel processes
    # Maps stream names to their latest data
    data_streams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Recent log messages (for display)
    recent_logs: List[Dict[str, Any]] = field(default_factory=list)
    max_log_history: int = 100  # Maximum number of log messages to keep
    
    # Status
    status: str = "initializing"  # "initializing", "running", "completed", "error"
    error_message: Optional[str] = None
    
    # Setup message (cached initial configuration for display)
    setup_message: Optional[str] = None

