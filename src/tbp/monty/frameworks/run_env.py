# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Environment setup utilities for Monty framework."""

import logging
import os
from pathlib import Path
from typing import Dict, Union

# Environment variable names
MONTY_LOGS_VAR = "MONTY_LOGS"
MONTY_MODELS_VAR = "MONTY_MODELS"
MONTY_DATA_VAR = "MONTY_DATA"
WANDB_DIR_VAR = "WANDB_DIR"

# Type alias for path-like objects
PathLike = Union[str, Path]

# Configure logger
logger = logging.getLogger(__name__)


def setup_env(
    monty_logs_dir_default: PathLike = "~/tbp/results/monty/",
) -> Dict[str, str]:
    """Set up environment variables for Monty framework.

    This function initializes required environment variables if they are not
    already set, using sensible defaults for the Monty framework.

    Args:
        monty_logs_dir_default: Default directory path for Monty logs.
            Can be a string or Path object. Defaults to "~/tbp/results/monty/".

    Returns:
        Dict[str, str]: A dictionary containing all configured environment
            variables and their paths.

    Side Effects:
        Sets the following environment variables if not already present:
        - MONTY_LOGS: Directory for Monty logs
        - MONTY_MODELS: Directory for pretrained models (under MONTY_LOGS)
        - MONTY_DATA: Directory for data
        - WANDB_DIR: Directory for Weights & Biases logs (defaults to MONTY_LOGS)

    Raises:
        ValueError: If the provided path is invalid or cannot be processed.

    Example:
        >>> env_paths = setup_env()
        >>> print(env_paths['MONTY_LOGS'])
        /home/user/tbp/results/monty/

        >>> env_paths = setup_env(Path.home() / "custom" / "logs")
        >>> print(env_paths['MONTY_MODELS'])
        /home/user/custom/logs/pretrained_models
    """
    # Validate input
    try:
        monty_logs_dir_default = str(monty_logs_dir_default)
    except Exception as e:
        raise ValueError(f"Invalid path provided: {monty_logs_dir_default}") from e

    # Initialize environment paths dictionary
    env_paths = {}

    # Set up MONTY_LOGS
    monty_logs_dir = _get_or_set_env_var(
        MONTY_LOGS_VAR, default_value=monty_logs_dir_default
    )
    env_paths[MONTY_LOGS_VAR] = monty_logs_dir

    # Set up MONTY_MODELS (depends on MONTY_LOGS)
    monty_models_default = str(Path(monty_logs_dir) / "pretrained_models")
    monty_models_dir = _get_or_set_env_var(
        MONTY_MODELS_VAR, default_value=monty_models_default
    )
    env_paths[MONTY_MODELS_VAR] = monty_models_dir

    # Set up MONTY_DATA
    monty_data_dir = _get_or_set_env_var(MONTY_DATA_VAR, default_value="~/tbp/data/")
    env_paths[MONTY_DATA_VAR] = monty_data_dir

    # Set up WANDB_DIR (uses MONTY_LOGS as default)
    wandb_dir = _get_or_set_env_var(WANDB_DIR_VAR, default_value=monty_logs_dir)
    env_paths[WANDB_DIR_VAR] = wandb_dir

    return env_paths


def _get_or_set_env_var(var_name: str, default_value: Union[str, Path]) -> str:
    """Get environment variable or set it to default value.

    This function always expands user paths (~) and resolves paths
    to absolute paths for consistency.

    Args:
        var_name: Name of the environment variable.
        default_value: Default value to use if variable is not set.
            Can be a string or Path object.

    Returns:
        The value of the environment variable (either existing or newly set),
        with user paths expanded.

    Raises:
        ValueError: If path processing fails.
    """
    value = os.getenv(var_name)

    if value is None:
        # Convert to Path for processing
        try:
            path = Path(str(default_value))
            # Always expand user and resolve to absolute path
            expanded_path = path.expanduser()
            value = str(expanded_path)
        except Exception as e:
            raise ValueError(
                f"Failed to process path for {var_name}: {default_value}"
            ) from e

        os.environ[var_name] = value
        logger.info(f"{var_name} not set. Using default directory: {value}")
    else:
        logger.debug(f"{var_name} already set to: {value}")

    return value
