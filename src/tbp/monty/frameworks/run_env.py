# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os


def setup_env(monty_logs_dir_default: str = "~/tbp/results/monty/"):
    """Setup environment variables for Monty.

    Args:
        monty_logs_dir_default (str): Default directory for Monty logs.
    """
    monty_logs_dir = os.getenv("MONTY_LOGS")

    if monty_logs_dir is None:
        monty_logs_dir = monty_logs_dir_default
        os.environ["MONTY_LOGS"] = monty_logs_dir
        print(f"MONTY_LOGS not set. Using default directory: {monty_logs_dir}")

    monty_models_dir = os.getenv("MONTY_MODELS")

    if monty_models_dir is None:
        monty_models_dir = f"{monty_logs_dir}pretrained_models/"
        os.environ["MONTY_MODELS"] = monty_models_dir
        print(f"MONTY_MODELS not set. Using default directory: {monty_models_dir}")

    wandb_dir = os.getenv("WANDB_DIR")

    if wandb_dir is None:
        wandb_dir = monty_logs_dir
        os.environ["WANDB_DIR"] = wandb_dir
        print(f"WANDB_DIR not set. Using default directory: {wandb_dir}")
