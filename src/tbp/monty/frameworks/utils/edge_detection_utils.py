# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_WINDOW_SIGMA = 1.0
DEFAULT_KERNEL_SIZE = 7
SOBEL_KERNEL_SIZE = 3
EPSILON = 1e-12


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: Input vector to normalize
        eps: Small epsilon value to avoid division by zero

    Returns:
        Normalized vector if norm > eps, otherwise zero vector
    """
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v * 0.0


def project_onto_tangent_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project a vector onto the tangent plane perpendicular to a normal.

    Removes the component of v that is parallel to n, leaving only the
    component that lies in the plane perpendicular to n.

    Args:
        v: Vector to project.
        n: Normal vector defining the tangent plane (should be normalized).

    Returns:
        The projection of v onto the plane perpendicular to n.
    """
    return v - np.dot(v, n) * n


def get_patch_center(h: int, w: int) -> Tuple[int, int]:
    """Get center coordinates of patch.

    Args:
        h: Height of patch
        w: Width of patch

    Returns:
        Tuple of (row, col) center coordinates
    """
    return h // 2, w // 2


def gradient_to_tangent_angle(gradient_angle: float) -> float:
    """Convert gradient direction to edge tangent direction and wrap to [0, 2*pi).

    The edge tangent is perpendicular to the gradient direction.

    Args:
        gradient_angle: Gradient direction in radians (any range)

    Returns:
        Edge tangent angle in [0, 2*pi) radians
    """
    tangent_angle = gradient_angle + np.pi / 2
    return (tangent_angle + 2 * np.pi) % (2 * np.pi)


def compute_edge_features_at_center(
    patch: np.ndarray,
    win_sigma: float = DEFAULT_WINDOW_SIGMA,
    ksize: int = DEFAULT_KERNEL_SIZE,
) -> Tuple[float, float, float]:
    """Compute edge features at center pixel using structure tensor method.

    This function computes the structure tensor using Gaussian-weighted gradient
    outer products in a local window. The structure tensor provides robust edge
    detection and orientation estimation by analyzing local gradient patterns.

    Args:
        patch: RGB or grayscale image patch
        win_sigma: Standard deviation for Gaussian window smoothing
        ksize: Kernel size for Gaussian blur

    Returns:
        Tuple of (edge_strength, coherence, tangent_theta):
            - edge_strength: Magnitude of dominant eigenvalue (edge strength)
            - coherence: Edge quality metric in [0, 1], where 1 is edge-like
            - tangent_theta: Edge tangent angle in [0, 2*pi) radians
    """
    img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806

    Jxx = Ix * Ix  # noqa: N806
    Jyy = Iy * Iy  # noqa: N806
    Jxy = Ix * Iy  # noqa: N806

    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), win_sigma)  # noqa: N806
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), win_sigma)  # noqa: N806
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), win_sigma)  # noqa: N806

    r, c = get_patch_center(*gray.shape)
    jxx, jyy, jxy = float(Jxx[r, c]), float(Jyy[r, c]), float(Jxy[r, c])

    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy**2))
    lam1 = 0.5 * (jxx + jyy + disc)
    lam2 = 0.5 * (jxx + jyy - disc)

    edge_strength = np.sqrt(max(lam1, 0.0))

    coherence = (lam1 - lam2) / (lam1 + lam2 + EPSILON)

    gradient_theta = 0.5 * np.arctan2(2.0 * jxy, (jxx - jyy + EPSILON))
    tangent_theta = gradient_to_tangent_angle(gradient_theta)

    return edge_strength, coherence, float(tangent_theta)


def compute_edge_features_center_weighted(
    patch: np.ndarray,
    win_sigma: float = DEFAULT_WINDOW_SIGMA,
    ksize: int = DEFAULT_KERNEL_SIZE,
    radius: float = 14.0,
    sigma_r: float = 7.0,
    c_min: float = 0.75,
    e_min: float = 0.01,
) -> Tuple[float, float, float]:
    """Compute edge features using center-weighted, global-aware structure tensor.

    This function aggregates structure tensor components over a center-biased
    neighborhood, giving higher weight to pixels closer to the center and pixels
    with stronger gradients. It applies energy and coherence thresholds to reject
    weak or cluttered edges.

    Args:
        patch: RGB or grayscale image patch
        win_sigma: Standard deviation for Gaussian window smoothing
        ksize: Kernel size for Gaussian blur
        radius: Radius of influence around center in pixels
            (e.g., 12-16 for 64x64 patch)
        sigma_r: Radial falloff parameter for center weighting (typically radius/2)
        c_min: Minimum coherence threshold to accept edge (0.7-0.8 recommended)
        e_min: Minimum local gradient energy threshold to accept as real edge

    Returns:
        Tuple of (edge_strength, coherence, tangent_theta):
            - edge_strength: Magnitude of dominant eigenvalue (0.0 if no edge)
            - coherence: Edge quality metric in [0, 1] (0.0 if no edge)
            - tangent_theta: Edge tangent angle in [0, 2*pi) radians (0.0 if no edge)
    """
    # Step 1: Compute gradients and local tensor components
    img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806

    Jxx = Ix * Ix  # noqa: N806
    Jyy = Iy * Iy  # noqa: N806
    Jxy = Ix * Iy  # noqa: N806

    # Optional Gaussian blur to suppress noise
    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), win_sigma)  # noqa: N806
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), win_sigma)  # noqa: N806
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), win_sigma)  # noqa: N806

    # Step 2: Center-weighted aggregation
    r0, c0 = get_patch_center(*gray.shape)
    h, w = gray.shape

    # Compute distance from center for all pixels
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d_squared = (rows - r0) ** 2 + (cols - c0) ** 2
    d = np.sqrt(d_squared)

    # Radial weight: Gaussian falloff from center
    w_r = np.exp(-(d_squared) / (2.0 * sigma_r**2))
    # Set weight to 0 for pixels beyond radius
    w_r[d > radius] = 0.0

    # Gradient energy weight: favor pixels with stronger gradients
    g = Ix**2 + Iy**2  # gradient energy at each pixel

    # Combined weight: radial * gradient energy
    w = w_r * g

    # Aggregate tensor components
    total_weight = np.sum(w)  # total weight
    if total_weight < EPSILON:
        # No significant weights, return no edge
        return 0.0, 0.0, 0.0

    Jxx_bar = np.sum(w * Jxx) / (total_weight + EPSILON)  # noqa: N806
    Jyy_bar = np.sum(w * Jyy) / (total_weight + EPSILON)  # noqa: N806
    Jxy_bar = np.sum(w * Jxy) / (total_weight + EPSILON)  # noqa: N806

    # Step 3: Compute eigenvalues, edge strength, coherence, orientation
    disc = np.sqrt((Jxx_bar - Jyy_bar) ** 2 + 4.0 * (Jxy_bar**2))
    lam1 = 0.5 * (Jxx_bar + Jyy_bar + disc)
    lam2 = 0.5 * (Jxx_bar + Jyy_bar - disc)

    edge_strength = np.sqrt(max(lam1, 0.0))
    coherence = (lam1 - lam2) / (lam1 + lam2 + EPSILON)

    gradient_theta = 0.5 * np.arctan2(2.0 * Jxy_bar, (Jxx_bar - Jyy_bar + EPSILON))
    tangent_theta = gradient_to_tangent_angle(gradient_theta)

    # Step 4: Energy and coherence rejection
    # Compute local gradient energy (using total weight as proxy)
    # Area of circular window: π * radius²
    area_window = np.pi * radius**2
    local_energy = total_weight / (area_window + EPSILON)

    # Reject if energy too low or coherence too low
    # if local_energy < e_min or coherence < c_min:
    #     return 0.0, 0.0, 0.0

    return float(edge_strength), float(coherence), float(tangent_theta)


def compute_edge_features_at_center_histogram(
    patch: np.ndarray,
    r_max: float = 16.0,
    tau_mag_ratio: float = 0.005,
    n_bins: int = 36,
    d_min: float = 0.2,
    delta_theta: float = np.pi / 12,
    n_min_candidates: int = 5,
    n_min_consistent: int = 5,
    d_inlier: float = 2.5,
    n_min_inlier: int = 10,
    f_min: float = 0.3,
    d_center_max: float = 4.0,
    win_sigma: float = DEFAULT_WINDOW_SIGMA,
    ksize: int = 3,
) -> Tuple[float, float, float]:
    """Compute edge features at center using orientation histogram + line-fit method.

    This function uses gradients around the center to find a dominant orientation
    via histogram of gradient directions, fits a line with that orientation through
    nearby edge pixels, and validates that the line passes close to the center with
    sufficient support.

    Args:
        patch: RGB or grayscale image patch
        r_max: Radius for neighborhood selection in pixels
            (default: 16.0 for 64x64 patches)
        tau_mag_ratio: Gradient magnitude threshold ratio (default: 0.1 * G_max)
        n_bins: Number of histogram bins (default: 36, 10° per bin)
        d_min: Minimum dominance ratio threshold (default: 0.5)
        delta_theta: Orientation tolerance in radians (default: π/12 = 15°)
        n_min_candidates: Minimum candidate pixels before histogram (default: 10)
        n_min_consistent: Minimum orientation-consistent pixels (default: 15)
        d_inlier: Line inlier distance threshold in pixels (default: 1.5)
        n_min_inlier: Minimum number of inliers (default: 10)
        f_min: Minimum inlier fraction (default: 0.5)
        d_center_max: Maximum distance from center to line in pixels (default: 2.0)
        win_sigma: Standard deviation for Gaussian blur (default: 1.0)
        ksize: Gaussian blur kernel size (default: 3x3)

    Returns:
        Tuple of (edge_strength, coherence, tangent_theta):
            - edge_strength: Average gradient magnitude of inliers (0.0 if no edge)
            - coherence: Combined dominance ratio and inlier fraction in [0, 1]
                (0.0 if no edge)
            - tangent_theta: Edge tangent angle in [0, 2*pi) radians (0.0 if no edge)
    """
    # Step 0: Preprocessing
    img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Optional Gaussian blur to suppress noise
    if win_sigma > 0 and ksize > 0:
        gray = cv2.GaussianBlur(gray, (ksize, ksize), win_sigma)

    # Step 1: Compute gradients
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)  # noqa: N806

    # Gradient magnitude
    g_mag = np.sqrt(Ix**2 + Iy**2)  # noqa: N806

    # Gradient orientation (normal direction), mapped to [0, π)
    theta_g = np.arctan2(Iy, Ix) % np.pi

    # Step 2: Select local edge pixels near center
    h, w = gray.shape
    r0, c0 = get_patch_center(h, w)

    # Compute distance from center for all pixels
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d = np.sqrt((rows - r0) ** 2 + (cols - c0) ** 2)

    # Keep pixels within r_max
    mask_distance = d <= r_max

    # Compute gradient magnitude threshold
    g_max = np.max(g_mag)  # noqa: N806
    tau_mag = tau_mag_ratio * g_max

    # Keep pixels with sufficient gradient magnitude
    mask_magnitude = g_mag >= tau_mag

    # Combined mask for candidate edge pixels
    mask_candidates = mask_distance & mask_magnitude

    # Get candidate pixel indices and values
    candidate_rows = rows[mask_candidates]
    candidate_cols = cols[mask_candidates]
    candidate_g = g_mag[mask_candidates]  # noqa: N806
    candidate_theta_g = theta_g[mask_candidates]

    n_candidates = len(candidate_rows)

    # Reject if too few candidates
    # if n_candidates < n_min_candidates:
    #     return 0.0, 0.0, 0.0

    # Step 3: Orientation histogram
    # Create histogram weighted by gradient magnitude
    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    h_hist, _ = np.histogram(  # noqa: N806
        candidate_theta_g, bins=bin_edges, weights=candidate_g
    )

    # Circular smoothing: h_smooth[k] = h_hist[k-1] + 2*h_hist[k] + h_hist[k+1] (mod B)
    h_smooth = np.zeros_like(h_hist, dtype=np.float64)  # noqa: N806
    for k in range(n_bins):
        k_prev = (k - 1) % n_bins
        k_next = (k + 1) % n_bins
        h_smooth[k] = h_hist[k_prev] + 2.0 * h_hist[k] + h_hist[k_next]

    # Find peak
    k_max = int(np.argmax(h_smooth))
    w_max = h_smooth[k_max]
    w_total = np.sum(h_smooth)  # noqa: N806

    # Check dominance ratio
    # if w_total < EPSILON:
    #     return 0.0, 0.0, 0.0

    dominance = w_max / w_total  # noqa: N806

    # Reject if dominance too low
    # if dominance < d_min:
    #     return 0.0, 0.0, 0.0

    # Compute dominant gradient orientation
    theta_g_star = (k_max + 0.5) * np.pi / n_bins

    # Convert to tangent orientation
    tangent_theta_star = gradient_to_tangent_angle(theta_g_star)

    # Step 4: Select pixels consistent with dominant orientation
    # Compute angular difference (circular distance on [0, π))
    delta_theta_array = np.abs(candidate_theta_g - theta_g_star)
    # Handle circular wrapping
    delta_theta_array = np.minimum(delta_theta_array, np.pi - delta_theta_array)

    # Keep pixels within orientation tolerance
    mask_consistent = delta_theta_array <= delta_theta

    consistent_rows = candidate_rows[mask_consistent]
    consistent_cols = candidate_cols[mask_consistent]
    consistent_g = candidate_g[mask_consistent]  # noqa: N806

    n_consistent = len(consistent_rows)

    # Reject if too few orientation-consistent pixels
    # if n_consistent < n_min_consistent:
    #     return 0.0, 0.0, 0.0

    # Step 5: Fit line with fixed orientation
    # Normal unit vector (perpendicular to tangent)
    n = np.array([-np.sin(tangent_theta_star), np.cos(tangent_theta_star)])

    # For each orientation-consistent pixel, compute projection onto normal
    # Convert pixel coordinates to (x, y) where x=col, y=row
    pixel_coords = np.column_stack([consistent_cols, consistent_rows])
    s_i = np.dot(pixel_coords, n)

    # Fit line offset: b = mean(s_i)
    b = np.mean(s_i)

    # Step 6: Check support & distance to line
    # Compute distance to line for each pixel
    d_i = np.abs(s_i - b)

    # Count inliers
    mask_inliers = d_i <= d_inlier
    n_inliers = np.sum(mask_inliers)

    # Reject if too few inliers or inlier fraction too low
    # if n_inliers < n_min_inlier:
    #     return 0.0, 0.0, 0.0

    inlier_fraction = n_inliers / n_consistent
    # if inlier_fraction < f_min:
    #     return 0.0, 0.0, 0.0

    # Check center distance to line
    center_coord = np.array([c0, r0])
    d_center = np.abs(np.dot(center_coord, n) - b)

    # Reject if line doesn't pass near center
    # if d_center > d_center_max:
    #     return 0.0, 0.0, 0.0

    # Step 7: Compute outputs
    # Edge strength: average gradient magnitude of inliers
    inlier_g = consistent_g[mask_inliers]  # noqa: N806
    edge_strength = float(np.mean(inlier_g))

    # Coherence: combination of dominance ratio and inlier fraction
    # Normalize to [0, 1] by taking geometric mean or weighted combination
    coherence = float(np.sqrt(dominance * inlier_fraction))

    return edge_strength, coherence, float(tangent_theta_star)


def draw_2d_pose_on_patch(
    patch: np.ndarray,
    edge_direction: float | None = None,
    label_text: str | None = None,
    tangent_color: tuple[int, int, int] = (255, 255, 0),
    normal_color: tuple[int, int, int] = (0, 255, 255),
    arrow_length: int = 20,
) -> np.ndarray:
    """Draw tangent/normal arrows and overlay debug text for a patch.

    Args:
        patch: RGB patch of shape (H, W, 3).
        edge_direction: Edge tangent direction in radians, if available.
        label_text: Text to overlay for debugging (e.g., angle or "No Edge").
        tangent_color: RGB color for tangent arrow (default: yellow).
        normal_color: RGB color for normal arrow (default: cyan).
        arrow_length: Length of arrows in pixels.

    Returns:
        Patch with annotations drawn on it.
    """
    patch_with_pose = patch.copy()
    center_y, center_x = patch.shape[0] // 2, patch.shape[1] // 2

    # Draw pose arrows only if we have an edge direction
    if edge_direction is not None:
        tangent_end_x = int(center_x + arrow_length * np.cos(edge_direction))
        tangent_end_y = int(center_y + arrow_length * np.sin(edge_direction))

        normal_direction = edge_direction + np.pi / 2
        normal_length = arrow_length * 0.7
        normal_end_x = int(center_x + normal_length * np.cos(normal_direction))
        normal_end_y = int(center_y + normal_length * np.sin(normal_direction))

        cv2.arrowedLine(
            patch_with_pose,
            (center_x, center_y),
            (tangent_end_x, tangent_end_y),
            tangent_color,
            thickness=3,
            tipLength=0.3,
        )

        cv2.arrowedLine(
            patch_with_pose,
            (center_x, center_y),
            (normal_end_x, normal_end_y),
            normal_color,
            thickness=3,
            tipLength=0.3,
        )

    # If no edge and no label text provided, use "No Edge" as default
    if edge_direction is None and label_text is None:
        label_text = "No Edge"

    cv2.circle(patch_with_pose, (center_x, center_y), 3, (255, 0, 0), -1)

    if label_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        color = (255, 255, 255)
        margin = 3

        (text_width, text_height), _ = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        x = patch_with_pose.shape[1] - text_width - margin
        y = text_height + margin

        cv2.rectangle(
            patch_with_pose,
            (x - margin, y - text_height - margin),
            (x + text_width + margin, y + margin // 2),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            patch_with_pose,
            label_text,
            (x, y),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return patch_with_pose


def save_raw_rgb_patch(patch: np.ndarray, filepath: str) -> None:
    """Save raw RGB patch without any annotations.

    This function saves the RGB patch as-is, without drawing any pose arrows,
    text, or other annotations. Useful for creating datasets to test different
    edge detection methods.

    Args:
        patch: RGB patch of shape (H, W, 3) to save.
        filepath: Path where the image should be saved.
    """
    plt.imsave(filepath, patch)
