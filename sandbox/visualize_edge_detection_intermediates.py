"""Visualize intermediate outputs of compute_edge_features_at_center function.

This script loads lotus.png, processes it through the edge detection pipeline,
and saves visualizations of all intermediate steps along with printed values.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tbp.monty.frameworks.utils.edge_detection_utils import (
    DEFAULT_KERNEL_SIZE,
    DEFAULT_WINDOW_SIGMA,
    EPSILON,
    SOBEL_KERNEL_SIZE,
    draw_2d_pose_on_patch,
    get_patch_center,
    gradient_to_tangent_angle,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "edge_detection_intermediates"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_image_with_center_marker(
    image: np.ndarray,
    center_r: int,
    center_c: int,
    filepath: Path,
    colormap: str = "gray",
    title: str | None = None,
) -> None:
    """Save image with center pixel marked.

    Args:
        image: Image array to save
        center_r: Row coordinate of center
        center_c: Column coordinate of center
        filepath: Path to save image
        colormap: Matplotlib colormap name
        title: Optional title for the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap=colormap)
    ax.plot(center_c, center_r, "r+", markersize=15, markeredgewidth=2)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def save_image(
    image: np.ndarray,
    filepath: Path,
    colormap: str = "gray",
    title: str | None = None,
) -> None:
    """Save image without center marker.

    Args:
        image: Image array to save
        filepath: Path to save image
        colormap: Matplotlib colormap name
        title: Optional title for the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap=colormap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main function to visualize edge detection intermediates."""
    # Load input image
    input_path = Path(__file__).parent / "lotus.png"
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    print(f"Loading image from: {input_path}")
    patch_rgb = plt.imread(str(input_path))
    
    # Ensure RGB format (remove alpha channel if present)
    if patch_rgb.shape[2] == 4:
        patch_rgb = patch_rgb[:, :, :3]
    
    print(f"Input image shape: {patch_rgb.shape}")
    print(f"Input image dtype: {patch_rgb.dtype}")
    print(f"Input image value range: [{patch_rgb.min():.3f}, {patch_rgb.max():.3f}]")
    print()

    # Save original RGB
    save_image(
        patch_rgb,
        OUTPUT_DIR / "00_input_rgb.png",
        colormap=None,
        title="Input RGB Patch",
    )
    print("Saved: 00_input_rgb.png")

    # Step 1: Convert to BGR (for OpenCV compatibility)
    img_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

    # Step 2: Convert to grayscale and normalize
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    print(f"Grayscale shape: {gray.shape}")
    print(f"Grayscale value range: [{gray.min():.3f}, {gray.max():.3f}]")
    print()

    # Save grayscale
    save_image(
        gray,
        OUTPUT_DIR / "01_grayscale.png",
        colormap="gray",
        title="Grayscale Conversion",
    )
    print("Saved: 01_grayscale.png")

    # Step 3: Compute gradients using Sobel
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806

    print(f"Gradient Ix shape: {Ix.shape}")
    print(f"Gradient Ix value range: [{Ix.min():.3f}, {Ix.max():.3f}]")
    print(f"Gradient Iy shape: {Iy.shape}")
    print(f"Gradient Iy value range: [{Iy.min():.3f}, {Iy.max():.3f}]")
    print()

    # Save gradients
    save_image(
        Ix,
        OUTPUT_DIR / "02_gradient_x.png",
        colormap="viridis",
        title="Gradient Ix (horizontal)",
    )
    print("Saved: 02_gradient_x.png")

    save_image(
        Iy,
        OUTPUT_DIR / "03_gradient_y.png",
        colormap="viridis",
        title="Gradient Iy (vertical)",
    )
    print("Saved: 03_gradient_y.png")

    # Step 4: Compute structure tensor components (before blur)
    Jxx_before = Ix * Ix  # noqa: N806
    Jyy_before = Iy * Iy  # noqa: N806
    Jxy_before = Ix * Iy  # noqa: N806

    print(f"Structure tensor components (before blur):")
    print(f"  Jxx range: [{Jxx_before.min():.6f}, {Jxx_before.max():.6f}]")
    print(f"  Jyy range: [{Jyy_before.min():.6f}, {Jyy_before.max():.6f}]")
    print(f"  Jxy range: [{Jxy_before.min():.6f}, {Jxy_before.max():.6f}]")
    print()

    # Get center coordinates for marking
    r, c = get_patch_center(*gray.shape)
    print(f"Center pixel coordinates: (r={r}, c={c})")
    print()

    # Save structure tensor components before blur
    save_image_with_center_marker(
        Jxx_before,
        r,
        c,
        OUTPUT_DIR / "04_structure_tensor_xx.png",
        colormap="hot",
        title="Structure Tensor Jxx (before blur)",
    )
    print("Saved: 04_structure_tensor_xx.png")

    save_image_with_center_marker(
        Jyy_before,
        r,
        c,
        OUTPUT_DIR / "05_structure_tensor_yy.png",
        colormap="hot",
        title="Structure Tensor Jyy (before blur)",
    )
    print("Saved: 05_structure_tensor_yy.png")

    save_image_with_center_marker(
        Jxy_before,
        r,
        c,
        OUTPUT_DIR / "06_structure_tensor_xy.png",
        colormap="hot",
        title="Structure Tensor Jxy (before blur)",
    )
    print("Saved: 06_structure_tensor_xy.png")

    # Step 5: Apply Gaussian blur
    win_sigma = DEFAULT_WINDOW_SIGMA
    ksize = DEFAULT_KERNEL_SIZE
    print(f"Applying Gaussian blur: sigma={win_sigma}, ksize={ksize}")
    print()

    Jxx = cv2.GaussianBlur(Jxx_before, (ksize, ksize), win_sigma)  # noqa: N806
    Jyy = cv2.GaussianBlur(Jyy_before, (ksize, ksize), win_sigma)  # noqa: N806
    Jxy = cv2.GaussianBlur(Jxy_before, (ksize, ksize), win_sigma)  # noqa: N806

    print(f"Structure tensor components (after blur):")
    print(f"  Jxx range: [{Jxx.min():.6f}, {Jxx.max():.6f}]")
    print(f"  Jyy range: [{Jyy.min():.6f}, {Jyy.max():.6f}]")
    print(f"  Jxy range: [{Jxy.min():.6f}, {Jxy.max():.6f}]")
    print()

    # Save structure tensor components after blur
    save_image_with_center_marker(
        Jxx,
        r,
        c,
        OUTPUT_DIR / "07_structure_tensor_xx_blurred.png",
        colormap="hot",
        title="Structure Tensor Jxx (after blur)",
    )
    print("Saved: 07_structure_tensor_xx_blurred.png")

    save_image_with_center_marker(
        Jyy,
        r,
        c,
        OUTPUT_DIR / "08_structure_tensor_yy_blurred.png",
        colormap="hot",
        title="Structure Tensor Jyy (after blur)",
    )
    print("Saved: 08_structure_tensor_yy_blurred.png")

    save_image_with_center_marker(
        Jxy,
        r,
        c,
        OUTPUT_DIR / "09_structure_tensor_xy_blurred.png",
        colormap="hot",
        title="Structure Tensor Jxy (after blur)",
    )
    print("Saved: 09_structure_tensor_xy_blurred.png")

    # Step 6: Extract values at center pixel
    jxx = float(Jxx[r, c])
    jyy = float(Jyy[r, c])
    jxy = float(Jxy[r, c])

    print("=" * 60)
    print("VALUES AT CENTER PIXEL:")
    print("=" * 60)
    print(f"Center coordinates: (r={r}, c={c})")
    print(f"Structure tensor at center:")
    print(f"  jxx = {jxx:.8f}")
    print(f"  jyy = {jyy:.8f}")
    print(f"  jxy = {jxy:.8f}")
    print()

    # Step 7: Compute eigenvalues
    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy**2))
    lam1 = 0.5 * (jxx + jyy + disc)
    lam2 = 0.5 * (jxx + jyy - disc)

    print(f"Discriminant: {disc:.8f}")
    print(f"Eigenvalues:")
    print(f"  λ1 = {lam1:.8f}")
    print(f"  λ2 = {lam2:.8f}")
    print()

    # Step 8: Compute final outputs
    edge_strength = np.sqrt(max(lam1, 0.0))
    coherence = (lam1 - lam2) / (lam1 + lam2 + EPSILON)
    gradient_theta = 0.5 * np.arctan2(2.0 * jxy, (jxx - jyy + EPSILON))
    tangent_theta = gradient_to_tangent_angle(gradient_theta)

    print("=" * 60)
    print("FINAL OUTPUTS:")
    print("=" * 60)
    print(f"Edge strength: {edge_strength:.6f}")
    print(f"Coherence: {coherence:.6f}")
    print(f"Gradient theta: {gradient_theta:.6f} radians ({np.degrees(gradient_theta):.2f} degrees)")
    print(f"Tangent theta: {tangent_theta:.6f} radians ({np.degrees(tangent_theta):.2f} degrees)")
    print("=" * 60)
    print()

    # Step 9: Create final visualization with edge direction
    annotated_patch = draw_2d_pose_on_patch(
        patch_rgb.copy(),
        edge_direction=tangent_theta,
        label_text=f"E={edge_strength:.3f}, C={coherence:.3f}",
    )
    
    save_image(
        annotated_patch,
        OUTPUT_DIR / "10_final_result.png",
        colormap=None,
        title="Final Result with Edge Direction",
    )
    print("Saved: 10_final_result.png")

    # Step 10: Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Input, grayscale, gradients
    axes[0, 0].imshow(patch_rgb)
    axes[0, 0].set_title("Input RGB", fontsize=12)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].plot(c, r, "r+", markersize=10, markeredgewidth=2)
    axes[0, 1].set_title("Grayscale (center marked)", fontsize=12)
    axes[0, 1].axis("off")
    
    # Combine gradients for visualization
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    axes[0, 2].imshow(gradient_magnitude, cmap="viridis")
    axes[0, 2].plot(c, r, "r+", markersize=10, markeredgewidth=2)
    axes[0, 2].set_title("Gradient Magnitude", fontsize=12)
    axes[0, 2].axis("off")
    
    # Row 2: Structure tensor components (after blur), final result
    axes[1, 0].imshow(Jxx, cmap="hot")
    axes[1, 0].plot(c, r, "r+", markersize=10, markeredgewidth=2)
    axes[1, 0].set_title("Jxx (blurred)", fontsize=12)
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(Jyy, cmap="hot")
    axes[1, 1].plot(c, r, "r+", markersize=10, markeredgewidth=2)
    axes[1, 1].set_title("Jyy (blurred)", fontsize=12)
    axes[1, 1].axis("off")
    
    axes[1, 2].imshow(annotated_patch)
    axes[1, 2].set_title(
        f"Final Result\nE={edge_strength:.3f}, C={coherence:.3f}\nθ={np.degrees(tangent_theta):.1f}°",
        fontsize=12,
    )
    axes[1, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Saved: summary_visualization.png")
    print()
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

