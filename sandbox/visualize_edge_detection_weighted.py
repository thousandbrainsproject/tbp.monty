"""Script to visualize weighted edge detection on synthetic RGB images.

Reads RGB PNG images from a user-selected folder in synthetic_edge_test_images directory,
applies the weighted (center-aware) edge detection method, and creates visualizations.
"""

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tbp.monty.frameworks.utils.edge_detection_utils import (
    compute_edge_features_center_weighted,
)

# Edge detection thresholds matching two_d_sensor_module.py defaults
EDGE_THRESHOLD = 0.1
COHERENCE_THRESHOLD = 0.05

# Arrow length scaling parameters
MAX_ARROW_LENGTH = 30  # Maximum arrow length in pixels
MAX_STRENGTH_COHERENCE = 4  # Product value at which max length is reached


def draw_tangent_only(
    patch: np.ndarray,
    edge_direction: Optional[float] = None,
    tangent_color: Tuple[int, int, int] = (255, 255, 0),
    arrow_length: int = 20,
) -> np.ndarray:
    """Draw only the tangent arrow (yellow line) on a patch.

    Args:
        patch: RGB patch of shape (H, W, 3).
        edge_direction: Edge tangent direction in radians, if available.
        tangent_color: RGB color for tangent arrow (default: yellow).
        arrow_length: Length of arrow in pixels.

    Returns:
        Patch with tangent arrow drawn on it.
    """
    patch_with_pose = patch.copy()
    center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2

    # Draw tangent arrow only if we have an edge direction
    if edge_direction is not None:
        tangent_end_x = int(center_x + arrow_length * np.cos(edge_direction))
        tangent_end_y = int(center_y + arrow_length * np.sin(edge_direction))

        cv2.arrowedLine(
            patch_with_pose,
            (center_x, center_y),
            (tangent_end_x, tangent_end_y),
            tangent_color,
            thickness=3,
            tipLength=0.3,
        )

    cv2.circle(patch_with_pose, (center_x, center_y), 3, (255, 0, 0), -1)

    return patch_with_pose


def process_single_image(patch: np.ndarray, output_path: Path) -> dict:
    """Process one image and save visualization.

    Args:
        patch: RGB image patch as numpy array
        output_path: Path where the figure should be saved

    Returns:
        Dictionary with results:
        {
            'strength': edge_strength,
            'coherence': coherence,
            'theta': tangent_theta,
            'ec': edge_strength * coherence
        }
    """
    # Apply weighted edge detection method
    strength, coherence, theta = compute_edge_features_center_weighted(patch)
    ec = strength * coherence

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Filter edges using same thresholds as two_d_sensor_module.py
    has_edge = (strength > EDGE_THRESHOLD) and (coherence > COHERENCE_THRESHOLD)
    edge_direction = theta if has_edge else None

    # Calculate arrow length: linear scaling up to MAX_ARROW_LENGTH
    arrow_length = int(
        min(MAX_ARROW_LENGTH, (ec / MAX_STRENGTH_COHERENCE) * MAX_ARROW_LENGTH)
    )

    annotated_patch = draw_tangent_only(
        patch.copy(), edge_direction=edge_direction, arrow_length=arrow_length
    )

    # Display the annotated patch
    ax.imshow(annotated_patch)
    ax.axis("off")

    # Set title with angle in degrees, E (edge strength), and C (coherence)
    if has_edge:
        angle_deg = np.degrees(theta)
        title = f"E={strength:.2f}, C={coherence:.2f}, θ={angle_deg:.1f}°"
    else:
        title = f"E={strength:.2f}, C={coherence:.2f}, No Edge"
    print(title)
    ax.set_title(title, fontsize=24)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(output_path)
    plt.close()

    # Return results dictionary
    return {
        "strength": strength,
        "coherence": coherence,
        "theta": theta,
        "ec": ec,
    }


def plot_histograms(
    strengths: List[float],
    coherences: List[float],
    output_path: Path,
) -> None:
    """Plot histograms of edge strengths, coherences, and their products.

    Args:
        strengths: List of edge strengths.
        coherences: List of coherences.
        output_path: Path where the histogram figure should be saved.
    """
    # Calculate products
    products = [s * c for s, c in zip(strengths, coherences)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Edge strength histogram
    axes[0].hist(
        strengths, bins=30, alpha=0.7, color="orange", edgecolor="black"
    )
    axes[0].axvline(EDGE_THRESHOLD, color="red", linestyle="--", label="threshold")
    axes[0].set_xlabel("Edge Strength")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Edge Strength Distribution")
    axes[0].legend()

    # Coherence histogram
    axes[1].hist(
        coherences, bins=30, alpha=0.7, color="orange", edgecolor="black"
    )
    axes[1].axvline(COHERENCE_THRESHOLD, color="red", linestyle="--", label="threshold")
    axes[1].set_xlabel("Coherence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Coherence Distribution")
    axes[1].legend()

    # Product histogram (strength * coherence = EC)
    axes[2].hist(
        products, bins=30, alpha=0.7, color="orange", edgecolor="black"
    )
    axes[2].axvline(
        MAX_STRENGTH_COHERENCE,
        color="green",
        linestyle="--",
        label=f"max arrow ({MAX_STRENGTH_COHERENCE})",
    )
    axes[2].set_xlabel("EC (Strength × Coherence)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("EC Distribution")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to: {output_path}")


def get_available_folders(input_dir: Path) -> List[str]:
    """Get list of available subdirectories in input_dir.

    Args:
        input_dir: Base directory to search for subdirectories

    Returns:
        List of subdirectory names (sorted)
    """
    folders = []
    for item in input_dir.iterdir():
        if item.is_dir():
            folders.append(item.name)
    return sorted(folders)


def select_folder_interactive(input_dir: Path) -> str:
    """Interactively select a folder from available subdirectories.

    Args:
        input_dir: Base directory containing subdirectories

    Returns:
        Selected folder name
    """
    folders = get_available_folders(input_dir)

    if not folders:
        raise ValueError(f"No subdirectories found in {input_dir}")

    print("\nAvailable folders:")
    for i, folder in enumerate(folders, 1):
        print(f"  {i}. {folder}")

    while True:
        try:
            choice = input(f"\nSelect folder (1-{len(folders)}) or folder name: ").strip()

            # Try to parse as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(folders):
                    return folders[idx]
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(folders)}.")
            # Try to match folder name
            elif choice in folders:
                return choice
            else:
                print(f"Invalid selection. Please enter a number (1-{len(folders)}) or folder name.")
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            raise SystemExit(1)


def main():
    """Main function to process all images and generate visualization results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Visualize weighted edge detection on synthetic images"
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        help="Folder name in synthetic_edge_test_images to process (if not provided, will prompt interactively)",
    )
    args = parser.parse_args()

    # Set up paths
    input_dir = Path("/Users/hlee/tbp/feat.2d_sensor/results/synthetic_edge_test_images")

    # Select folder
    if args.folder:
        selected_folder = args.folder
        folder_path = input_dir / selected_folder
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"Error: Folder '{selected_folder}' not found in {input_dir}")
            print(f"Available folders: {', '.join(get_available_folders(input_dir))}")
            return
    else:
        selected_folder = select_folder_interactive(input_dir)
        folder_path = input_dir / selected_folder

    print(f"\nProcessing folder: {selected_folder}")

    # Set up output directory
    output_dir = Path("results/synthetic_edge_test_images_comparison") / f"{selected_folder}_weighted"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find PNG files in selected folder (not recursively)
    image_files = sorted(folder_path.glob("*.png"))
    print(f"Found {len(image_files)} PNG images in {selected_folder}")

    if not image_files:
        print(f"No PNG images found in {folder_path}")
        return

    # CSV file path
    csv_path = output_dir / "edge_detection_results.csv"

    # Statistics
    total_processed = 0
    total_failed = 0

    # Data collection for histograms
    all_strengths = []
    all_coherences = []

    # Process each image with CSV file context manager
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "filename",
                "angle",
                "strength",
                "coherence",
                "ec",
            ]
        )

        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                # Load image
                patch = plt.imread(image_file)

                # Ensure RGB format (handle RGBA if present)
                if patch.shape[2] == 4:
                    patch = patch[:, :, :3]

                # Convert from [0, 1] range to [0, 255] range if needed
                # plt.imread() loads PNG images in [0, 1] range, but edge detection
                # functions expect [0, 255] range (they divide by 255.0 internally)
                if patch.dtype == np.float32 or patch.dtype == np.float64:
                    if patch.max() <= 1.0:
                        patch = (patch * 255.0).astype(np.uint8)
                elif patch.dtype != np.uint8:
                    # If it's some other type, ensure it's uint8
                    patch = np.clip(patch, 0, 255).astype(np.uint8)

                # Get relative path from input_dir to preserve subdirectory structure
                relative_path = image_file.relative_to(input_dir)
                output_path = output_dir / relative_path.name

                # Create output subdirectory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Process image
                results = process_single_image(patch, output_path)

                # Collect data for histograms
                all_strengths.append(results["strength"])
                all_coherences.append(results["coherence"])

                # Write to CSV
                csv_writer.writerow(
                    [
                        image_file.name,
                        results["theta"],  # angle in radians
                        results["strength"],
                        results["coherence"],
                        results["ec"],
                    ]
                )

                total_processed += 1

            except Exception as e:
                print(f"\nWarning: Failed to process {image_file.name}: {e}")
                total_failed += 1
                continue

    # Generate histograms
    if all_strengths:
        histogram_path = output_dir / "histograms.png"
        plot_histograms(
            all_strengths,
            all_coherences,
            histogram_path,
        )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total images failed: {total_failed}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"CSV results: {csv_path.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

