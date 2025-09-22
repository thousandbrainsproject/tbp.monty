# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_association_metrics(exp_path: str) -> int:
    """Plot association metrics from train/eval CSVs.

    Looks for `<exp_path>/train_stats.csv` and `<exp_path>/eval_stats.csv` and, if
    present, plots the following metrics per episode index:
      - assoc_total
      - assoc_strong
      - assoc_avg_strength

    Args:
        exp_path: Path to the experiment directory containing CSV stats.

    Returns:
        Exit code (0 on success, 1 on error locating directory).
    """
    exp_dir = Path(exp_path)
    if not exp_dir.exists():
        logger.error(f"Experiment path not found: {exp_path}")
        return 1

    plt.style.use("seaborn-darkgrid")

    splits = ["train", "eval"]
    any_plotted = False

    for split in splits:
        csv_path = exp_dir / f"{split}_stats.csv"
        if not csv_path.exists():
            logger.info(f"Skipping missing CSV: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to read %s", csv_path)
            continue

        metrics = [
            c
            for c in ["assoc_total", "assoc_strong", "assoc_avg_strength"]
            if c in df.columns
        ]
        if not metrics:
            available_cols = list(df.columns)
            logger.warning(
                "No assoc_* metrics found in %s. Available columns: %s",
                csv_path,
                available_cols,
            )
            continue

        ax = df[metrics].plot(title=f"{split} association metrics")
        ax.set_xlabel("episode index")
        ax.set_ylabel("value")
        plt.tight_layout()
        any_plotted = True

    if any_plotted:
        plt.show()
    else:
        logger.warning(
            "No plots were generated. Ensure CSVs exist and contain assoc_* columns."
        )

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    """Register the association_metrics subcommand.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "association_metrics",
        help="Plot association metrics from train/eval CSVs.",
        parents=[parent_parser] if parent_parser else [],
    )

    parser.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment output with train_stats.csv and "
            "eval_stats.csv."
        ),
    )

    parser.set_defaults(
        func=lambda args: sys.exit(plot_association_metrics(args.experiment_log_dir))
    )
