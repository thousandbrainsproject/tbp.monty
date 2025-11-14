"""Script to compare evidence values for two objects across episodes.

Reads detailed_run_stats.json from an experiment directory and creates plots
comparing evidence values for 007_disk_tbp_horz and 009_disk_numenta_horz.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.utils.logging_utils import deserialize_json_chunks


def load_evidence_data(detailed_stats, object_names):
    """Extract evidence arrays for specified objects from detailed stats.

    Args:
        detailed_stats: Dictionary of episode data loaded from detailed_run_stats.json
        object_names: List of object names to extract
            (e.g., ["007_disk_tbp_horz", "009_disk_numenta_horz"])

    Returns:
        Dictionary mapping episode_id to a dict containing:
            - steps: List of step indices
            - object_name: Dict with 'mean', 'std', 'max' arrays for each object
    """
    episode_data = {}

    for episode_id, episode_stats in detailed_stats.items():
        if "LM_0" not in episode_stats:
            continue

        lm_data = episode_stats["LM_0"]
        if "evidences" not in lm_data:
            continue

        evidences = lm_data["evidences"]
        num_steps = len(evidences)

        episode_info = {
            "steps": list(range(num_steps)),
        }

        for obj_name in object_names:
            obj_means = []
            obj_stds = []
            obj_maxs = []
            obj_steps = []

            for step_idx in range(num_steps):
                step_evidences = evidences[step_idx]
                if obj_name not in step_evidences:
                    # Object not present at this step, skip
                    continue

                obj_evidence_array = step_evidences[obj_name]
                if isinstance(obj_evidence_array, list):
                    obj_evidence_array = np.array(obj_evidence_array)

                obj_means.append(np.mean(obj_evidence_array))
                obj_stds.append(np.std(obj_evidence_array))
                obj_maxs.append(np.max(obj_evidence_array))
                obj_steps.append(step_idx)

            # Only add if we found data for this object
            if obj_means:
                episode_info[obj_name] = {
                    "mean": np.array(obj_means),
                    "std": np.array(obj_stds),
                    "max": np.array(obj_maxs),
                    "steps": np.array(obj_steps),
                }

        # Only add episode if we have data for at least one object
        if len(episode_info) > 1:  # More than just "steps"
            episode_data[episode_id] = episode_info

    return episode_data


def plot_all_evidences(episode_data, episode_id, object_names, output_dir):
    """Create plot with all evidence values and STD shaded area.

    Args:
        episode_data: Dictionary containing evidence data for the episode
        episode_id: String identifier for the episode
        object_names: List of object names to plot
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#1f77b4", "#ff7f0e"]  # Blue and orange

    # Collect all y-values to set consistent y-axis limits
    all_y_values = []

    for obj_name in object_names:
        if obj_name not in episode_data:
            continue

        obj_data = episode_data[obj_name]
        means = obj_data["mean"]
        stds = obj_data["std"]

        all_y_values.extend([means - stds, means + stds])

    # Set consistent y-axis limits
    if all_y_values:
        y_min = min(np.min(vals) for vals in all_y_values)
        y_max = max(np.max(vals) for vals in all_y_values)
        y_margin = (y_max - y_min) * 0.05  # 5% margin
        y_lim = (y_min - y_margin, y_max + y_margin)
    else:
        y_lim = None

    for idx, obj_name in enumerate(object_names):
        ax = axes[idx]

        if obj_name not in episode_data:
            ax.text(
                0.5,
                0.5,
                f"No data for {obj_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(obj_name, fontsize=12, fontweight="bold")
            continue

        obj_data = episode_data[obj_name]
        steps = obj_data["steps"]
        means = obj_data["mean"]
        stds = obj_data["std"]
        maxs = obj_data["max"]

        color = colors[idx % len(colors)]

        # Plot mean line
        ax.plot(steps, means, label="Mean", color=color, linewidth=2)

        # Plot shaded area for ±1 STD
        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            alpha=0.3,
            color=color,
            label="±1 STD",
        )

        ax.set_xlabel("Step Number", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Evidence Value", fontsize=12)
        ax.set_title(obj_name, fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Set consistent y-axis limits
        if y_lim:
            ax.set_ylim(y_lim)

        # Add text annotation for max evidence at last step
        if len(maxs) > 0:
            last_max = maxs[-1]
            last_step = steps[-1]
            ax.text(
                0.98,
                0.98,
                f"Max at step {last_step}: {last_max:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    fig.suptitle(
        f"Evidence Values Over Steps - Episode {episode_id}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = Path(output_dir) / f"episode_{episode_id}_all_evidences.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_max_evidence(episode_data, episode_id, object_names, output_dir):
    """Create plot with max evidence only.

    Args:
        episode_data: Dictionary containing evidence data for the episode
        episode_id: String identifier for the episode
        object_names: List of object names to plot
        output_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = ["#1f77b4", "#ff7f0e"]  # Blue and orange

    # Collect all y-values to set consistent y-axis limits
    all_maxs = []

    for obj_name in object_names:
        if obj_name not in episode_data:
            continue

        obj_data = episode_data[obj_name]
        maxs = obj_data["max"]
        all_maxs.append(maxs)

    # Set consistent y-axis limits
    if all_maxs:
        y_min = min(np.min(vals) for vals in all_maxs)
        y_max = max(np.max(vals) for vals in all_maxs)
        y_margin = (y_max - y_min) * 0.05  # 5% margin
        y_lim = (y_min - y_margin, y_max + y_margin)
    else:
        y_lim = None

    for idx, obj_name in enumerate(object_names):
        ax = axes[idx]

        if obj_name not in episode_data:
            ax.text(
                0.5,
                0.5,
                f"No data for {obj_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(obj_name, fontsize=12, fontweight="bold")
            continue

        obj_data = episode_data[obj_name]
        steps = obj_data["steps"]
        maxs = obj_data["max"]

        color = colors[idx % len(colors)]

        # Plot max line
        ax.plot(
            steps,
            maxs,
            label="Max Evidence",
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )

        ax.set_xlabel("Step Number", fontsize=12)
        if idx == 0:
            ax.set_ylabel("Maximum Evidence Value", fontsize=12)
        ax.set_title(obj_name, fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Set consistent y-axis limits
        if y_lim:
            ax.set_ylim(y_lim)

        # Add text annotation for max evidence at last step
        if len(maxs) > 0:
            last_max = maxs[-1]
            last_step = steps[-1]
            ax.text(
                0.98,
                0.98,
                f"Max at step {last_step}: {last_max:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    fig.suptitle(
        f"Maximum Evidence Over Steps - Episode {episode_id}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = Path(output_dir) / f"episode_{episode_id}_max_evidence.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def main():
    """Main function to parse arguments, load data, and generate plots.

    Raises:
        FileNotFoundError: If detailed_run_stats.json is not found in the
            specified directory.
    """
    parser = argparse.ArgumentParser(
        description="Compare evidence values for two objects across episodes"
    )
    parser.add_argument(
        "experiment_log_dir",
        type=str,
        help="Directory containing detailed_run_stats.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as experiment_log_dir)",
    )

    args = parser.parse_args()

    experiment_log_dir = Path(args.experiment_log_dir)
    json_file = experiment_log_dir / "detailed_run_stats.json"

    if not json_file.exists():
        raise FileNotFoundError(
            f"Could not find detailed_run_stats.json at {json_file}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else experiment_log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detailed stats
    print(f"Loading detailed stats from {json_file}...")
    detailed_stats = deserialize_json_chunks(json_file)
    print(f"Loaded {len(detailed_stats)} episodes")

    # Object names to compare
    object_names = ["007_disk_tbp_horz", "009_disk_numenta_horz"]

    # Extract evidence data
    print("Extracting evidence data...")
    episode_data = load_evidence_data(detailed_stats, object_names)
    print(f"Found data for {len(episode_data)} episodes")

    # Generate plots for each episode
    for episode_id, episode_info in episode_data.items():
        print(f"\nProcessing episode {episode_id}...")
        plot_all_evidences(episode_info, episode_id, object_names, output_dir)
        plot_max_evidence(episode_info, episode_id, object_names, output_dir)

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

