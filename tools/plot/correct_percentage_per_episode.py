# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib import rcParams

from tbp.monty.frameworks.utils.logging_utils import load_stats

if TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


def plot_correct_percentage_per_episode(exp_path: str) -> int:
    """Bar chart showing how many steps the correct object had the highest evidence.

    Args:
        exp_path: Path to the experiment directory containing the detailed stats file.

    Returns:
        Exit code.
    """
    if not Path(exp_path).exists():
        logger.error(f"Experiment path not found: {exp_path}")
        return 1

    # Load detailed stats
    _, _, detailed_stats, _ = load_stats(exp_path, False, False, True, False)

    correct_object_hits = []
    episode_labels = []

    for _, episode_data in enumerate(detailed_stats.values()):
        evidences_data = episode_data["LM_0"]["max_evidence"]
        target_obj = episode_data["target"]["primary_target_object"]

        count = sum(1 for ts in evidences_data if max(ts, key=ts.get) == target_obj)
        percentage = (count / len(evidences_data)) * 100

        correct_object_hits.append(percentage)
        episode_labels.append(target_obj)

    rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(
        episode_labels,
        correct_object_hits,
        color="#8ecae6",
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_title("Correct Object MLH Percentage per Episode", fontweight="bold")
    ax.set_xlabel("Episode (target object)")
    ax.set_ylabel("Correct Steps (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(episode_labels)))
    ax.set_xticklabels(episode_labels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.2)

    fig.tight_layout()
    plt.show()

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: argparse.ArgumentParser | None = None,
) -> None:
    """Add the correct_percentage_per_episode subparser to the main parser.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "correct_percentage_per_episode",
        help="Plot a bar chart of how often the correct object was the MLH.",
        parents=[parent_parser] if parent_parser else [],
    )
    parser.add_argument(
        "experiment_log_dir",
        help=(
            "The directory containing the experiment log with the detailed stats file."
        ),
    )
    parser.set_defaults(
        func=lambda args: sys.exit(
            plot_correct_percentage_per_episode(args.experiment_log_dir)
        )
    )
