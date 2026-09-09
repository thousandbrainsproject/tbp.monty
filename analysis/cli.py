# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Shared argparse pieces for command-line entry points."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from analysis.telemetry import resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path


def run_directory(run: str) -> Path:
    """Argparse type: an experiment directory, or a run name under ``RESULTS_DIR``.

    Returns:
        The directory.

    Raises:
        argparse.ArgumentTypeError: If neither exists.
    """
    try:
        return resolve_run_dir(run)
    except FileNotFoundError as error:
        raise argparse.ArgumentTypeError(str(error)) from error

