"""Script to compare three edge detection methods on RGB images.

Reads RGB PNG images from a directory, applies three edge detection methods,
and creates comparison visualizations with results saved to CSV.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tbp.monty.frameworks.utils.edge_detection_utils import (
    compute_edge_features_at_center,
    compute_edge_features_at_center_histogram,
    compute_edge_features_center_weighted,
    draw_2d_pose_on_patch,
)

# Edge detection thresholds matching two_d_sensor_module.py defaults
EDGE_THRESHOLD = 0.1
COHERENCE_THRESHOLD = 0.05


def process_single_image(patch: np.ndarray, output_path: Path) -> dict:
    """Process one image and save comparison visualization.

    Args:
        patch: RGB image patch as numpy array
        output_path: Path where the comparison figure should be saved

    Returns:
        Dictionary with results for all three methods:
        {
            'default': (edge_strength, coherence, tangent_theta),
            'center_aware': (edge_strength, coherence, tangent_theta),
            'histogram': (edge_strength, coherence, tangent_theta)
        }
    """
    # Apply all three edge detection methods
    default_strength, default_coherence, default_theta = (
        compute_edge_features_at_center(patch)
    )
    center_aware_strength, center_aware_coherence, center_aware_theta = (
        compute_edge_features_center_weighted(patch)
    )
    histogram_strength, histogram_coherence, histogram_theta = (
        compute_edge_features_at_center_histogram(patch)
    )

    # Create 1x3 subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Method configurations
    methods = [
        {
            "name": "default",
            "strength": default_strength,
            "coherence": default_coherence,
            "theta": default_theta,
            "ax": axes[0],
        },
        {
            "name": "center-aware",
            "strength": center_aware_strength,
            "coherence": center_aware_coherence,
            "theta": center_aware_theta,
            "ax": axes[1],
        },
        {
            "name": "histogram",
            "strength": histogram_strength,
            "coherence": histogram_coherence,
            "theta": histogram_theta,
            "ax": axes[2],
        },
    ]

    # Process each method
    for method in methods:
        ax = method["ax"]
        strength = method["strength"]
        coherence = method["coherence"]
        theta = method["theta"]

        # Draw pose on patch
        # Filter edges using same thresholds as two_d_sensor_module.py
        has_edge = (strength > EDGE_THRESHOLD) and (coherence > COHERENCE_THRESHOLD)
        edge_direction = theta if has_edge else None
        annotated_patch = draw_2d_pose_on_patch(patch.copy(), edge_direction=edge_direction)

        # Display the annotated patch
        ax.imshow(annotated_patch)
        ax.axis("off")

        # Set title with metrics
        title = f"{method['name']}, E={strength:.3f}, C={coherence:.3f}"
        print(title)
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(output_path)
    plt.close()

    # Return results dictionary
    return {
        "default": (default_strength, default_coherence, default_theta),
        "center_aware": (center_aware_strength, center_aware_coherence, center_aware_theta),
        "histogram": (histogram_strength, histogram_coherence, histogram_theta),
    }


def main():
    """Main function to process all images and generate comparison results."""
    # Set up paths
    input_dir = Path("/Users/hlee/tbp/data/tbp_numenta_disks_RGB")
    output_dir = Path("tbp_numenta_disks_RGB_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PNG files
    image_files = sorted(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} PNG images")

    # CSV file path
    csv_path = output_dir / "edge_detection_results.csv"

    # Prepare CSV writer
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "filename",
            "default_angle",
            "default_strength",
            "default_coherence",
            "center_aware_angle",
            "center_aware_strength",
            "center_aware_coherence",
            "histogram_angle",
            "histogram_strength",
            "histogram_coherence",
        ]
    )

    # Statistics
    total_processed = 0
    total_failed = 0

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

            # Process image
            output_path = output_dir / image_file.name
            results = process_single_image(patch, output_path)

            # Write to CSV
            csv_writer.writerow(
                [
                    image_file.name,
                    results["default"][2],  # angle
                    results["default"][0],  # strength
                    results["default"][1],  # coherence
                    results["center_aware"][2],  # angle
                    results["center_aware"][0],  # strength
                    results["center_aware"][1],  # coherence
                    results["histogram"][2],  # angle
                    results["histogram"][0],  # strength
                    results["histogram"][1],  # coherence
                ]
            )

            total_processed += 1

        except Exception as e:
            print(f"\nWarning: Failed to process {image_file.name}: {e}")
            total_failed += 1
            continue

    # Close CSV file
    csv_file.close()

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

