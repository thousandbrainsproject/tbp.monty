"""Script to test center-weighted edge detection on diagonal line pattern images.

Applies compute_edge_features_center_weighted from edge_detection_utils to
three diagonal line pattern images at different resolutions.
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tbp.monty.frameworks.utils.edge_detection_utils import (
    compute_edge_features_center_weighted,
    draw_2d_pose_on_patch,
)


def load_image(filepath: Path) -> np.ndarray:
    """Load an image file as RGB array.

    Args:
        filepath: Path to image file

    Returns:
        RGB image array of shape (H, W, 3) with dtype uint8
    """
    img = plt.imread(filepath)
    # Convert to uint8 if needed (plt.imread may return float in [0, 1])
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    # Ensure RGB format (3 channels)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        # Remove alpha channel if present
        img = img[:, :, :3]
    return img


def main():
    """Test center-weighted edge detection on diagonal line pattern images."""
    # Input directory
    input_dir = Path("results/diagonal_line_patterns")
    
    # Output directory for results
    output_dir = Path("results/edge_detection_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Image files to process
    image_files = [
        "diagonal_lines_16x16.png",
        "diagonal_lines_32x32.png",
        "diagonal_lines_64x64.png",
    ]

    # Parameters for center-weighted edge detection
    # Adjust these based on image size
    params_by_size = {
        16: {
            "radius": 6.0,
            "sigma_r": 3.0,
            "win_sigma": 1.0,
            "ksize": 5,
        },
        32: {
            "radius": 12.0,
            "sigma_r": 6.0,
            "win_sigma": 1.0,
            "ksize": 7,
        },
        64: {
            "radius": 14.0,
            "sigma_r": 7.0,
            "win_sigma": 1.0,
            "ksize": 7,
        },
    }

    results = []
    results_win_sigma = []
    results_blurred = []

    # win_sigma values to test (without blur)
    win_sigma_values = [0.5, 1.0, 1.5, 2.0]
    
    # Gaussian blur parameters (sigma values to try)
    blur_sigmas = [0.5, 1.0, 1.5]

    for filename in image_files:
        filepath = input_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue

        print(f"\nProcessing {filename}...")
        
        # Load image
        image = load_image(filepath)
        size = image.shape[0]  # Assume square images
        
        # Get parameters for this size
        params = params_by_size.get(size, params_by_size[64])
        
        # Process original image with different win_sigma values (no blur)
        for win_sigma in win_sigma_values:
            print(f"  Processing with win_sigma={win_sigma}...")
            edge_strength, coherence, tangent_theta = compute_edge_features_center_weighted(
                image,
                radius=params["radius"],
                sigma_r=params["sigma_r"],
                win_sigma=win_sigma,
                ksize=params["ksize"],
                c_min=0.75,
                e_min=0.01,
            )

            # Convert angle to degrees for display
            tangent_theta_deg = np.degrees(tangent_theta) if tangent_theta > 0 else None

            # Print results
            print(f"    Edge strength: {edge_strength:.4f}")
            print(f"    Coherence: {coherence:.4f}")
            if tangent_theta_deg is not None:
                print(f"    Tangent angle: {tangent_theta_deg:.2f}°")
            else:
                print(f"    Tangent angle: No edge detected")

            # Create visualization without text overlay
            annotated_image = draw_2d_pose_on_patch(
                image.copy(),
                edge_direction=tangent_theta if tangent_theta > 0 else None,
                label_text=None,  # No text overlay
            )

            # Save annotated image
            output_filename = f"edge_detected_{filename.replace('.png', '')}_win_sigma_{win_sigma:.1f}.png"
            output_path = output_dir / output_filename
            plt.imsave(output_path, annotated_image)
            print(f"    Saved annotated image: {output_path}")

            # Store results
            result_entry = {
                "filename": filename,
                "size": size,
                "win_sigma": win_sigma,
                "image": image,
                "annotated_image": annotated_image,
                "edge_strength": edge_strength,
                "coherence": coherence,
                "tangent_theta": tangent_theta,
                "tangent_theta_deg": tangent_theta_deg,
            }
            results_win_sigma.append(result_entry)
            
            # Store the default win_sigma (1.0) result in the main results list for backward compatibility
            if win_sigma == 1.0:
                results.append(result_entry)

        # Process with Gaussian blur
        for blur_sigma in blur_sigmas:
            print(f"  Processing with Gaussian blur (σ={blur_sigma})...")
            
            # Apply Gaussian blur
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            blurred_bgr = cv2.GaussianBlur(image_bgr, (0, 0), blur_sigma)
            blurred_image = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply center-weighted edge detection
            edge_strength_blur, coherence_blur, tangent_theta_blur = compute_edge_features_center_weighted(
                blurred_image,
                radius=params["radius"],
                sigma_r=params["sigma_r"],
                win_sigma=params["win_sigma"],
                ksize=params["ksize"],
                c_min=0.75,
                e_min=0.01,
            )

            # Convert angle to degrees for display
            tangent_theta_deg_blur = np.degrees(tangent_theta_blur) if tangent_theta_blur > 0 else None

            # Print results
            print(f"    Edge strength: {edge_strength_blur:.4f}")
            print(f"    Coherence: {coherence_blur:.4f}")
            if tangent_theta_deg_blur is not None:
                print(f"    Tangent angle: {tangent_theta_deg_blur:.2f}°")
            else:
                print(f"    Tangent angle: No edge detected")

            # Create visualization without text overlay
            annotated_image_blur = draw_2d_pose_on_patch(
                blurred_image.copy(),
                edge_direction=tangent_theta_blur if tangent_theta_blur > 0 else None,
                label_text=None,  # No text overlay
            )

            # Save annotated image
            output_filename_blur = f"edge_detected_{filename.replace('.png', '')}_blur_{blur_sigma:.1f}.png"
            output_path_blur = output_dir / output_filename_blur
            plt.imsave(output_path_blur, annotated_image_blur)
            print(f"    Saved annotated image: {output_path_blur}")

            # Store results
            results_blurred.append({
                "filename": filename,
                "size": size,
                "blur_sigma": blur_sigma,
                "image": blurred_image,
                "annotated_image": annotated_image_blur,
                "edge_strength": edge_strength_blur,
                "coherence": coherence_blur,
                "tangent_theta": tangent_theta_blur,
                "tangent_theta_deg": tangent_theta_deg_blur,
            })

    # Create summary visualizations
    if results_win_sigma:
        print("\n" + "=" * 60)
        print("Summary of Results (win_sigma variations):")
        print("=" * 60)
        
        # Create summary visualizations for different win_sigma values
        for win_sigma in win_sigma_values:
            win_sigma_results = [r for r in results_win_sigma if r["win_sigma"] == win_sigma]
            if not win_sigma_results:
                continue
            
            # Print summary for this win_sigma
            print(f"\nwin_sigma={win_sigma}:")
            for r in win_sigma_results:
                print(f"  {r['filename']} ({r['size']}x{r['size']}):")
                print(f"    Edge Strength: {r['edge_strength']:.4f}")
                print(f"    Coherence: {r['coherence']:.4f}")
                if r['tangent_theta_deg'] is not None:
                    print(f"    Tangent Angle: {r['tangent_theta_deg']:.2f}°")
                else:
                    print(f"    Tangent Angle: No edge detected")
            
            # Create a figure with all results side by side for this win_sigma
            fig, axes = plt.subplots(1, len(win_sigma_results), figsize=(5 * len(win_sigma_results), 5))
            if len(win_sigma_results) == 1:
                axes = [axes]
            
            for idx, r in enumerate(win_sigma_results):
                axes[idx].imshow(r["annotated_image"])
                # Create subtitle with metrics
                subtitle = (
                    f"E={r['edge_strength']:.3f}, C={r['coherence']:.3f}"
                    + (f", θ={r['tangent_theta_deg']:.1f}°" if r['tangent_theta_deg'] is not None else ", No edge")
                )
                axes[idx].set_title(
                    f"{r['filename']}\n{r['size']}x{r['size']} (win_σ={win_sigma})\n{subtitle}",
                    fontsize=10
                )
                axes[idx].axis("off")
            
            plt.tight_layout()
            summary_path = output_dir / f"edge_detection_summary_win_sigma_{win_sigma:.1f}.png"
            plt.savefig(summary_path, dpi=150, bbox_inches="tight")
            print(f"Saved summary visualization (win_σ={win_sigma}): {summary_path}")
            plt.close()
        
        # Also create the default summary (win_sigma=1.0) with the original filename for backward compatibility
        if results:
            default_results = [r for r in results_win_sigma if r["win_sigma"] == 1.0]
            if default_results:
                fig, axes = plt.subplots(1, len(default_results), figsize=(5 * len(default_results), 5))
                if len(default_results) == 1:
                    axes = [axes]
                
                for idx, r in enumerate(default_results):
                    axes[idx].imshow(r["annotated_image"])
                    subtitle = (
                        f"E={r['edge_strength']:.3f}, C={r['coherence']:.3f}"
                        + (f", θ={r['tangent_theta_deg']:.1f}°" if r['tangent_theta_deg'] is not None else ", No edge")
                    )
                    axes[idx].set_title(f"{r['filename']}\n{r['size']}x{r['size']}\n{subtitle}", fontsize=10)
                    axes[idx].axis("off")
                
                plt.tight_layout()
                summary_path = output_dir / "edge_detection_summary.png"
                plt.savefig(summary_path, dpi=150, bbox_inches="tight")
                print(f"\nSaved default summary visualization: {summary_path}")
                plt.close()

        # Create summary visualizations for blurred versions
        if results_blurred:
            # Group by blur sigma
            for blur_sigma in blur_sigmas:
                blurred_results = [r for r in results_blurred if r["blur_sigma"] == blur_sigma]
                if not blurred_results:
                    continue
                
                fig, axes = plt.subplots(1, len(blurred_results), figsize=(5 * len(blurred_results), 5))
                if len(blurred_results) == 1:
                    axes = [axes]
                
                for idx, r in enumerate(blurred_results):
                    axes[idx].imshow(r["annotated_image"])
                    # Create subtitle with metrics
                    subtitle = (
                        f"E={r['edge_strength']:.3f}, C={r['coherence']:.3f}"
                        + (f", θ={r['tangent_theta_deg']:.1f}°" if r['tangent_theta_deg'] is not None else ", No edge")
                    )
                    axes[idx].set_title(
                        f"{r['filename']}\n{r['size']}x{r['size']} (blur σ={blur_sigma})\n{subtitle}",
                        fontsize=10
                    )
                    axes[idx].axis("off")
                
                plt.tight_layout()
                summary_path_blur = output_dir / f"edge_detection_summary_blur_{blur_sigma:.1f}.png"
                plt.savefig(summary_path_blur, dpi=150, bbox_inches="tight")
                print(f"Saved blurred summary visualization (σ={blur_sigma}): {summary_path_blur}")
                plt.close()

    print(f"\nProcessing complete! Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
