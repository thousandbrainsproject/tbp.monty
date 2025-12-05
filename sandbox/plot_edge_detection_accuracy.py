"""Plot average angular error comparison between naive and weighted edge detection methods.

Reads edge_detection_results.csv from angled_lines results, extracts ground truth angles
from filenames, and creates a bar plot comparing average absolute angular errors.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_ground_truth_angle(filename: str) -> float:
    """Extract ground truth angle in degrees from filename.

    Args:
        filename: Filename like 'angled_20deg_high.png'

    Returns:
        Ground truth angle in degrees
    """
    match = re.search(r"angled_(\d+)deg", filename)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract angle from filename: {filename}")


def normalize_angle_difference(detected_rad: float, ground_truth_deg: float) -> float:
    """Calculate angular error accounting for line symmetry (180° ambiguity).

    Since edge detection returns tangent direction, angles 0° and 180° represent
    the same line orientation. We compute the minimum angular distance.

    Args:
        detected_rad: Detected angle in radians
        ground_truth_deg: Ground truth angle in degrees

    Returns:
        Absolute angular error in degrees (0 to 90)
    """
    detected_deg = np.degrees(detected_rad)

    # Normalize both angles to [0, 180) range due to line symmetry
    detected_normalized = detected_deg % 180
    gt_normalized = ground_truth_deg % 180

    # Calculate difference
    diff = abs(detected_normalized - gt_normalized)

    # Account for wrap-around (e.g., 179° vs 1° should give 2°, not 178°)
    if diff > 90:
        diff = 180 - diff

    return diff


def main():
    """Main function to load data, compute errors, and create bar plot."""
    # Load CSV
    csv_path = Path(
        "/Users/hlee/tbp/feat.2d_sensor/results/"
        "synthetic_edge_test_images_comparison/angled_lines/edge_detection_results.csv"
    )
    df = pd.read_csv(csv_path)

    # Calculate errors for each row
    naive_errors = []
    weighted_errors = []

    for _, row in df.iterrows():
        gt_angle = extract_ground_truth_angle(row["filename"])
        naive_error = normalize_angle_difference(row["default_angle"], gt_angle)
        weighted_error = normalize_angle_difference(row["center_aware_angle"], gt_angle)
        naive_errors.append(naive_error)
        weighted_errors.append(weighted_error)

    # Calculate mean errors
    mean_naive_error = np.mean(naive_errors)
    mean_weighted_error = np.mean(weighted_errors)
    std_naive_error = np.std(naive_errors)
    std_weighted_error = np.std(weighted_errors)

    print(f"Naive method:    mean error = {mean_naive_error:.2f}° ± {std_naive_error:.2f}°")
    print(f"Weighted method: mean error = {mean_weighted_error:.2f}° ± {std_weighted_error:.2f}°")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ["Naive", "Weighted"]
    means = [mean_naive_error, mean_weighted_error]
    stds = [std_naive_error, std_weighted_error]
    colors = ["#e74c3c", "#27ae60"]

    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 0.5,
            f"{mean:.1f}°",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    ax.set_ylabel("Mean Angular Error (degrees)", fontsize=14)
    ax.set_title("Edge Detection Accuracy: Naive vs Weighted", fontsize=18, fontweight="bold")
    ax.set_ylim(0, max(means) + max(stds) + 5)

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()

    # Save plot
    output_path = Path(
        "/Users/hlee/tbp/feat.2d_sensor/results/"
        "synthetic_edge_test_images_comparison/angled_lines/accuracy_comparison.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()

