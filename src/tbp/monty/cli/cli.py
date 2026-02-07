# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import argparse
import os

from .commands import run
from .config import config

__all__ = ["main"]


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser("run")
    parser_run.set_defaults(func=run)

    # global flags
    parser.add_argument(
        "--monty.logs-dir",
        help="Monty logs directory (env: MONTY_LOGS)",
        default=os.getenv("MONTY_LOGS", ""),
    )
    parser.add_argument(
        "--monty.models-dir",
        help="Monty models directory (env: MONTY_MODELS)",
        default=os.getenv("MONTY_MODELS", ""),
    )
    parser.add_argument(
        "--monty.data-dir",
        help="Monty data directory (env: MONTY_DATA)",
        default=os.getenv("MONTY_DATA", ""),
    )
    parser.add_argument(
        "--wandb.dir",
        help="wandb directory (env: MONTY_DATA_DIR, default: same as --monty.logs-dir)",
        default=os.getenv("WANDB_DIR", ""),
    )

    # `run` flags
    parser_run.add_argument("experiment", help="Name of the experiment to run")
    parser_run.add_argument(
        "-p",
        "--parallel",
        action="store_true",
        help="Execute in parallel (env: MONTY_PARALLEL)",
        default=bool(os.getenv("MONTY_PARALLEL")),
    )

    return parser.parse_args().__dict__


def main() -> None:
    args = parse_args()
    command = args.pop("func")
    configs = config(args)
    command(*configs)
