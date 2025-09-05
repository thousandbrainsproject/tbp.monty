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

from pathlib import Path

def setup_env(monty_logs_dir_default: str = "~/tbp/results/monty/"):
    """Setup environment variables for Monty.

    Args:
        monty_logs_dir_default: Default directory for Monty logs.
    """

    dir_path = os.getenv("MONTY_LOGS")
    if dir_path is None:
        dir_path = str(Path(monty_logs_dir_default).expanduser())
        os.environ["MONTY_LOGS"] = dir_path
        print(f"MONTY_LOGS not set. Using default directory: {dir_path}")

    monty_logs_dir = Path(dir_path)
    
    dir_path = os.getenv("MONTY_MODELS")
    if dir_path is None:
        dir_path = str(Path("~/tbp/results/monty").expanduser() / "pretrained_models")
        os.environ["MONTY_MODELS"] = dir_path
        print(f"MONTY_MODELS not set. Using default directory: {dir_path}")

    dir_path = os.getenv("MONTY_DATA")
    if dir_path is None:
        dir_path = str(Path("~/tbp/data").expanduser())
        os.environ["MONTY_DATA"] = dir_path
        print(f"MONTY_DATA not set. Using default directory: {dir_path}")

    dir_path = os.getenv("WANDB_DIR")
    if dir_path is None:
        dir_path = str(monty_logs_dir)
        os.environ["WANDB_DIR"] = dir_path
        print(f"WANDB_DIR not set. Using default directory: {dir_path}")
