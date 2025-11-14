"""Script to visualize and compare results between 2 experiments.

Reads eval_stats.csv from two experiment directories and creates comparison plots
for accuracy, monty_steps, and rotation_error.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_experiment_data(csv_path):
    """Load eval_stats.csv from an experiment directory.

    Args:
        csv_path: Path to eval_stats.csv file

    Returns:
        pandas DataFrame with experiment data
    """
    df = pd.read_csv(csv_path)
    return df


def calculate_accuracy(df):
    """Calculate accuracy as percentage of correct or correct_mlh results.

    Args:
        df: DataFrame with primary_performance column

    Returns:
        Accuracy percentage (0-100)
    """
    if "primary_performance" not in df.columns:
        raise ValueError("DataFrame must contain 'primary_performance' column")

    correct_mask = df["primary_performance"].isin(["correct", "correct_mlh"])
    accuracy = (correct_mask.sum() / len(df)) * 100
    return accuracy


def get_monty_steps_data(df):
    """Get raw monty_steps data.

    Args:
        df: DataFrame with monty_steps column

    Returns:
        Array of monty_steps values (with NaN removed)
    """
    if "monty_steps" not in df.columns:
        raise ValueError("DataFrame must contain 'monty_steps' column")

    monty_steps = df["monty_steps"].dropna().values
    return monty_steps


def get_rotation_error_data(df):
    """Get raw rotation_error data in degrees.

    Args:
        df: DataFrame with rotation_error column (in radians)

    Returns:
        Array of rotation_error values in degrees (with NaN removed)
    """
    if "rotation_error" not in df.columns:
        raise ValueError("DataFrame must contain 'rotation_error' column")

    # Drop NaN values and convert from radians to degrees
    rotation_error_rad = df["rotation_error"].dropna()
    rotation_error_deg = (rotation_error_rad * 180 / np.pi).values
    return rotation_error_deg


def create_comparison_plots(df1, df2, name1, name2, output_dir):
    """Create comparison plots for two experiments.

    Args:
        df1: DataFrame for first experiment
        df2: DataFrame for second experiment
        name1: Name/label for first experiment
        name2: Name/label for second experiment
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate statistics for both experiments
    acc1 = calculate_accuracy(df1)
    acc2 = calculate_accuracy(df2)

    # Get raw data for violin plots
    monty_data1 = get_monty_steps_data(df1)
    monty_data2 = get_monty_steps_data(df2)

    rot_data1 = get_rotation_error_data(df1)
    rot_data2 = get_rotation_error_data(df2)

    # Calculate means and medians for text labels
    monty_mean1 = np.mean(monty_data1)
    monty_mean2 = np.mean(monty_data2)
    monty_median1 = np.median(monty_data1)
    monty_median2 = np.median(monty_data2)
    rot_mean1 = np.mean(rot_data1)
    rot_mean2 = np.mean(rot_data2)
    rot_median1 = np.median(rot_data1)
    rot_median2 = np.median(rot_data2)

    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Accuracy
    ax1 = axes[0]
    experiments = [name1, name2]
    accuracies = [acc1, acc2]
    bars1 = ax1.bar(experiments, accuracies, color=["#1f77b4", "#ff7f0e"], alpha=0.7)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Plot 2: Monty Steps (Violin Plot)
    ax2 = axes[1]
    monty_data = [monty_data1, monty_data2]
    parts = ax2.violinplot(
        monty_data,
        positions=range(len(experiments)),
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )
    # Color the violins
    for pc, color in zip(parts["bodies"], ["#1f77b4", "#ff7f0e"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    # Color the other elements
    for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)
    ax2.set_xticks(range(len(experiments)))
    ax2.set_xticklabels(experiments)
    ax2.set_ylabel("Monty Steps", fontsize=12)
    ax2.set_title("Monty Steps Distribution", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    # Add mean and median value labels
    monty_means_combined = [monty_mean1, monty_mean2]
    monty_medians_combined = [monty_median1, monty_median2]
    y_max_monty = max([np.max(d) for d in monty_data])
    # Adjust y-axis limits to accommodate labels
    ax2.set_ylim(top=y_max_monty * 1.3)
    for pos, (mean_val, median_val) in enumerate(
        zip(monty_means_combined, monty_medians_combined)
    ):
        ax2.text(
            pos,
            y_max_monty * 1.20,
            f"Mean: {mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax2.text(
            pos,
            y_max_monty * 1.12,
            f"Median: {median_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 3: Rotation Error (Violin Plot)
    ax3 = axes[2]
    rot_data = [rot_data1, rot_data2]
    parts = ax3.violinplot(
        rot_data,
        positions=range(len(experiments)),
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )
    # Color the violins
    for pc, color in zip(parts["bodies"], ["#1f77b4", "#ff7f0e"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    # Color the other elements
    for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)
    ax3.set_xticks(range(len(experiments)))
    ax3.set_xticklabels(experiments)
    ax3.set_ylabel("Rotation Error (degrees)", fontsize=12)
    ax3.set_title("Rotation Error Distribution", fontsize=14, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    # Add mean and median value labels
    rot_means_combined = [rot_mean1, rot_mean2]
    rot_medians_combined = [rot_median1, rot_median2]
    y_max_rot = max([np.max(d) for d in rot_data])
    # Adjust y-axis limits to accommodate labels
    ax3.set_ylim(top=y_max_rot * 1.3)
    for pos, (mean_val, median_val) in enumerate(
        zip(rot_means_combined, rot_medians_combined)
    ):
        ax3.text(
            pos,
            y_max_rot * 1.20,
            f"Mean: {mean_val:.1f}°",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax3.text(
            pos,
            y_max_rot * 1.12,
            f"Median: {median_val:.1f}°",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout(pad=3.0)

    # Save combined plot
    output_path = output_dir / "experiment_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot: {output_path}")

    # Also save individual plots
    # Accuracy
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    bars1 = ax1.bar(experiments, accuracies, color=["#1f77b4", "#ff7f0e"], alpha=0.7)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    output_path1 = output_dir / "experiment_comparison_accuracy.png"
    plt.savefig(output_path1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy plot: {output_path1}")

    # Monty Steps (Violin Plot)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    monty_data = [monty_data1, monty_data2]
    parts = ax2.violinplot(
        monty_data,
        positions=range(len(experiments)),
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )
    # Color the violins
    for pc, color in zip(parts["bodies"], ["#1f77b4", "#ff7f0e"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    # Color the other elements
    for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)
    ax2.set_xticks(range(len(experiments)))
    ax2.set_xticklabels(experiments)
    ax2.set_ylabel("Monty Steps", fontsize=12)
    ax2.set_title("Monty Steps Distribution", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    # Add mean and median value labels
    monty_means = [monty_mean1, monty_mean2]
    monty_medians = [monty_median1, monty_median2]
    y_max = max([np.max(d) for d in monty_data])
    # Adjust y-axis limits to accommodate labels
    ax2.set_ylim(top=y_max * 1.3)
    for pos, (mean_val, median_val) in enumerate(zip(monty_means, monty_medians)):
        ax2.text(
            pos,
            y_max * 1.20,
            f"Mean: {mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax2.text(
            pos,
            y_max * 1.12,
            f"Median: {median_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout(pad=2.0)
    output_path2 = output_dir / "experiment_comparison_monty_steps.png"
    plt.savefig(output_path2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved monty steps plot: {output_path2}")

    # Rotation Error (Violin Plot)
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    rot_data = [rot_data1, rot_data2]
    parts = ax3.violinplot(
        rot_data,
        positions=range(len(experiments)),
        showmeans=True,
        showmedians=True,
        widths=0.6,
    )
    # Color the violins
    for pc, color in zip(parts["bodies"], ["#1f77b4", "#ff7f0e"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    # Color the other elements
    for key in ["cbars", "cmins", "cmaxes", "cmeans", "cmedians"]:
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.5)
    ax3.set_xticks(range(len(experiments)))
    ax3.set_xticklabels(experiments)
    ax3.set_ylabel("Rotation Error (degrees)", fontsize=12)
    ax3.set_title("Rotation Error Distribution", fontsize=14, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    # Add mean and median value labels
    rot_means = [rot_mean1, rot_mean2]
    rot_medians = [rot_median1, rot_median2]
    y_max = max([np.max(d) for d in rot_data])
    # Adjust y-axis limits to accommodate labels
    ax3.set_ylim(top=y_max * 1.3)
    for pos, (mean_val, median_val) in enumerate(zip(rot_means, rot_medians)):
        ax3.text(
            pos,
            y_max * 1.20,
            f"Mean: {mean_val:.1f}°",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax3.text(
            pos,
            y_max * 1.12,
            f"Median: {median_val:.1f}°",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout(pad=2.0)
    output_path3 = output_dir / "experiment_comparison_rotation_error.png"
    plt.savefig(output_path3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved rotation error plot: {output_path3}")


def main():
    """Main function to compare two experiments."""
    # Define experiment paths
    exp1_path = Path(
        "/Users/hlee/tbp/results/monty/projects/evidence_eval_runs/logs/disk_inference_2d_on_2d/eval_stats.csv"
    )
    exp2_path = Path(
        "/Users/hlee/tbp/results/monty/projects/evidence_eval_runs/logs/disk_inference_control_on_control/eval_stats.csv"
    )

    # Experiment names for labels
    exp1_name = "disk_inference_2d_on_2d"
    exp2_name = "disk_inference_control_on_control"

    # Output directory
    output_dir = Path("/Users/hlee/tbp/feat.2d_sensor/results")

    print(f"Loading experiment 1: {exp1_path}")
    df1 = load_experiment_data(exp1_path)
    print(f"  Loaded {len(df1)} rows")

    print(f"Loading experiment 2: {exp2_path}")
    df2 = load_experiment_data(exp2_path)
    print(f"  Loaded {len(df2)} rows")

    print("\nCreating comparison plots...")
    create_comparison_plots(df1, df2, exp1_name, exp2_name, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
