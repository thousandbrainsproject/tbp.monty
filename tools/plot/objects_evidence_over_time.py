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

from matplotlib import patches, transforms
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.utils.logging_utils import load_stats

logger = logging.getLogger(__name__)


def plot_objects_evidence_over_time(exp_path: str) -> int:
    """Plot evidence scores for each object over time.

    This function visualizes the evidence scores for each object. The plot is produced
    over a sequence of episodes, and overlays colored rectangles highlighting when a
    particular target object is active.

    Args:
        exp_path (str): Path to the experiment directory containing the detailed stats
            file.

    Returns:
        int: Exit code.
    """
    if not Path(exp_path).exists():
        logger.error(f"Experiment path not found: {exp_path}")
        return 1

    # load detailed stats
    _, _, detailed_stats, _ = load_stats(exp_path, False, False, True, False)

    plt.style.use("seaborn-darkgrid")
    # fix colors for distinct objects (tab10 supports up to 10 distinct colors)
    cmap = plt.cm.tab10
    num_colors = len(DISTINCT_OBJECTS)
    ycb_colors = {obj: cmap(i / num_colors) for i, obj in enumerate(DISTINCT_OBJECTS)}

    classes = {
        k: [] for k in list(detailed_stats["0"]["LM_0"]["max_evidence"][0].keys())
    }
    target_objects = []  # Objects in each segment, e.g., ['strawberry', 'banana']
    target_transitions = []  # Transition points on the x-axis, e.g., [49, 99]

    for episode_data in detailed_stats.values():
        evidences_data = episode_data["LM_0"]["max_evidence"]

        # append evidence data to classes
        for ts in evidences_data:
            for k, v in ts.items():
                classes[k].append(v)

        # collect the target object of this episode
        target_objects.append(episode_data["target"]["primary_target_object"])

        # collect target transition point
        target_transitions.append(len(evidences_data))

    # Create the plot
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot the lines
    for obj, scores in classes.items():
        ax.plot(
            scores,
            marker="o",
            linestyle="-",
            label=obj,
            color=ycb_colors.get(obj, "gray"),
        )

    # Add colored rectangles indicating the current target object
    box_height = 0.02
    prev_x = 0
    for obj, x in zip(target_objects, target_transitions):
        rect = patches.Rectangle(
            (prev_x, 1 - box_height),
            (x - 1),
            box_height,
            transform=transforms.blended_transform_factory(ax.transData, ax.transAxes),
            edgecolor="black",
            facecolor=ycb_colors.get(obj, "gray"),
            lw=1,
            alpha=1.0,
            clip_on=True,
        )
        ax.add_patch(rect)
        prev_x += x - 1

    # Formatting
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Evidence Scores", fontsize=14)
    ax.set_title(
        "Evidence Scores Over Time with Resampling",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(title="Objects", fontsize=10, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(10))

    # Show plot
    plt.show()

    return 0


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent_parser: Optional[argparse.ArgumentParser] = None,
) -> None:
    """Add the objects_evidence_over_time subparser to the main parser.

    Args:
        subparsers: The subparsers object from the main parser.
        parent_parser: Optional parent parser for shared arguments.
    """
    parser = subparsers.add_parser(
        "objects_evidence_over_time",
        help="Plot evidence scores for each object over time.",
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
            plot_objects_evidence_over_time(args.experiment_log_dir)
        )
    )
