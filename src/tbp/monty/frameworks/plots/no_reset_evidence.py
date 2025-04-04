# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os

import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.ticker import MultipleLocator

from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.utils.logging_utils import load_stats

plt.style.use("seaborn-darkgrid")

# fix colors for distinct objects (tab10 supports up to 10 distinct colors)
cmap = plt.cm.tab10
num_colors = len(DISTINCT_OBJECTS)
YCB_COLORS = {obj: cmap(i / num_colors) for i, obj in enumerate(DISTINCT_OBJECTS)}


def plot_objects_evidence_over_time(exp_path: str) -> None:
    """Plot evidence scores for each object over time.

    This function visualizes the evidence scores for each object. The plot is produced
    over a sequence of episodes, and overlays colored rectangles highlighting when a
    particular target object is active.

    Args:
        exp_path (str): Path to the experiment directory containing the detailed stats
            file.

    Raises:
        FileNotFoundError: If the stats file cannot be found in the specified path.
    """
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")

    # load detailed stats
    _, _, detailed_stats, _ = load_stats(exp_path, False, False, True, False)

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
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the lines
    for obj, scores in classes.items():
        ax.plot(
            scores,
            marker="o",
            linestyle="-",
            label=obj,
            color=YCB_COLORS.get(obj, "gray"),
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
            facecolor=YCB_COLORS.get(obj, "gray"),
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


def plot_pose_error_over_time(exp_path: str) -> None:
    """Plot MLH pose error and theoretical limits over time.

    This function visualizes the theoretical pose error limit vs. the actual
    pose error over time, along with a correctness indicator of whether the
    predicted object (`mlo`) matches the target object. It generates a two-row
    plot where the top subplot shows binary correctness (Correct/Wrong) of MLO
    and the bottom subplot shows pose error metrics.

    The theoretical limit is calculated by finding the minimum pose error over
    all existing hypotheses in Monty's hypothesis space. This metric conveys the
    best possible performance if Monty selects the best hypothesis as its most
    likely hypothesis (MLH).

    Args:
        exp_path (str): Path to the experiment directory containing detailed stats
            data.

    Raises:
        FileNotFoundError: If the given `exp_path` does not exist.
    """
    if not os.path.exists(exp_path):
        raise FileNotFoundError(f"Experiment path not found: {exp_path}")

    # load detailed stats
    _, _, detailed_stats, _ = load_stats(exp_path, False, False, True, False)

    dfs = []
    for ep_data in detailed_stats.values():
        steps = len(ep_data["LM_0"]["target_object_theoretical_limit"])
        target = [ep_data["LM_0"]["target"]["object"] for _ in range(steps)]
        th_limit = ep_data["LM_0"]["target_object_theoretical_limit"]
        mlo = [ep_data["LM_0"]["current_mlh"][i]["graph_id"] for i in range(steps)]
        obj_error = ep_data["LM_0"]["target_object_pose_error"]

        dfs.append(
            pd.DataFrame(
                {
                    "target": target,
                    "th_limit": th_limit,
                    "mlo": mlo,
                    "obj_error": obj_error,
                }
            )
        )

    # Combine all episodes into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Create stacked subplots: (MLO accuracy, pose error)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [0.5, 4]}
    )

    mlo_correct = [
        1 if mlo == target else 0 for mlo, target in zip(df["mlo"], df["target"])
    ]
    colors = ["green" if c else "red" for c in mlo_correct]

    ax0.scatter(
        np.array(df.index), np.array(mlo_correct), c=colors, marker="o", s=30, alpha=0.8
    )
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["wrong", "correct"])
    ax0.set_ylabel("MLO", fontsize=14)
    ax0.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax0.set_ylim(-0.5, 1.5)
    ax0.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax1.plot(
        np.array(df.index),
        np.array(df["th_limit"]),
        color="black",
        linestyle="-",
        linewidth=2,
        label="Theoretical limit",
    )
    ax1.scatter(
        np.array(df.index),
        np.array(df["obj_error"]),
        c="gray",
        marker="o",
        s=50,
        alpha=0.75,
        label="MLH of target object",
    )

    # Labels and formatting
    ax1.set_xlabel("Steps", fontsize=14)
    ax1.set_ylabel("Pose Error", fontsize=14)
    ax0.set_title(
        "MLH Pose Error vs. Theoretical Limit", fontsize=14, fontweight="bold"
    )

    # Create Legend
    black_line = mlines.Line2D(
        [], [], color="black", linewidth=2, label="Theoretical limit"
    )
    gray_dots = mlines.Line2D(
        [],
        [],
        color="gray",
        marker="o",
        linestyle="None",
        markersize=8,
        label="MLH of target object",
    )
    green_diamond = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=8,
        label="Correct MLO",
    )
    red_diamond = mlines.Line2D(
        [],
        [],
        color="red",
        marker="o",
        linestyle="None",
        markersize=8,
        label="Wrong MLO",
    )

    ax1.legend(
        handles=[black_line, gray_dots, green_diamond, red_diamond],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=True,
        shadow=True,
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()
