# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

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


def compute_arc_length_correction(
    chord_length: float,
    curvature: float,
    threshold: float = 0.001,
) -> float:
    """Compute arc length from chord length using surface curvature.

    For a circular arc with curvature k and chord length c, the arc length is:
        arc = (2/k) * arcsin(k*c/2)

    This corrects for the underestimation that occurs when projecting 3D
    movement onto a tangent plane on curved surfaces.

    Args:
        chord_length: Projected displacement magnitude (chord of the arc).
        curvature: Normal curvature in the direction of movement (1/radius).
            Positive for convex surfaces, negative for concave.
        threshold: Skip correction when |k*c| < threshold (already accurate).

    Returns:
        Estimated arc length. Returns chord_length unchanged if curvature
        is negligible or would cause numerical issues.
    """
    kc = abs(curvature * chord_length)

    if kc < threshold:
        # Curvature effect negligible, chord ~ arc
        return chord_length

    if kc >= 2.0:
        # Chord longer than diameter - invalid geometry, skip correction
        return chord_length

    # arc = (2/k) * arcsin(k*c/2)
    arc_length = (2.0 / abs(curvature)) * np.arcsin(kc / 2.0)
    return arc_length


def is_geometric_edge(
    depth_patch: np.ndarray,
    edge_theta: float,
    depth_threshold: float = 0.01,
) -> bool:
    """Check if detected edge is geometric (depth discontinuity) vs texture.

    Geometric edges occur at object boundaries or surface creases where depth
    changes abruptly. Texture edges occur on flat surfaces where depth is
    continuous. This function computes the depth gradient perpendicular to
    the detected edge direction and checks if it exceeds a threshold.

    Args:
        depth_patch: Depth image patch (same size as RGB patch used for edge
            detection). Values should be in consistent units (e.g., meters).
        edge_theta: Edge tangent angle in radians from RGB edge detection.
        depth_threshold: Maximum allowed depth gradient magnitude for texture
            edges. Edges with perpendicular depth gradient above this value
            are classified as geometric.

    Returns:
        True if edge is geometric (should be filtered out), False if texture edge.
    """
    # Compute depth gradients using Sobel
    depth_dx = cv2.Sobel(depth_patch, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
    depth_dy = cv2.Sobel(depth_patch, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE)

    # Direction perpendicular to edge (normal to edge line)
    edge_normal_angle = edge_theta + np.pi / 2
    nx = np.cos(edge_normal_angle)
    ny = np.sin(edge_normal_angle)

    # Depth gradient in direction perpendicular to edge, sampled at patch center
    cy, cx = depth_patch.shape[0] // 2, depth_patch.shape[1] // 2
    depth_gradient_perp = abs(nx * depth_dx[cy, cx] + ny * depth_dy[cy, cx])

    return depth_gradient_perp > depth_threshold


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


def compute_weighted_structure_tensor_edge_features(
    patch: np.ndarray,
    win_sigma: float = DEFAULT_WINDOW_SIGMA,
    ksize: int = DEFAULT_KERNEL_SIZE,
    radius: float = 14.0,
    sigma_r: float = 7.0,
    max_center_offset: float | None = None,
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
        max_center_offset: Maximum allowed distance from patch center to edge in pixels.
            If None, Step 4 (center proximity check) is skipped. If a float value is
            provided, edges that do not pass within this distance of the center are
            rejected.

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

    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), win_sigma)  # noqa: N806
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), win_sigma)  # noqa: N806
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), win_sigma)  # noqa: N806

    # Step 2a: Center-weighted aggregation
    r0, c0 = get_patch_center(*gray.shape)
    h, w = gray.shape

    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d_squared = (rows - r0) ** 2 + (cols - c0) ** 2
    d = np.sqrt(d_squared)

    # Step 2b: Radial weight (Gaussian falloff from center)
    w_r = np.exp(-(d_squared) / (2.0 * sigma_r**2))
    w_r[d > radius] = 0.0
    g = Ix**2 + Iy**2
    w = w_r * g

    total_weight = np.sum(w)
    if total_weight < EPSILON:
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

    # Step 4: Check if edge passes near patch center
    if max_center_offset is not None:
        nx = np.cos(gradient_theta)
        ny = np.sin(gradient_theta)

        dr = rows - r0
        dc = cols - c0

        dist_normal = nx * dc + ny * dr
        d_center = np.sum(w * dist_normal) / (total_weight + EPSILON)

        if abs(d_center) > max_center_offset:
            return 0.0, 0.0, 0.0

    return float(edge_strength), float(coherence), float(tangent_theta)


def draw_2d_pose_on_patch(
    patch: np.ndarray,
    edge_direction: float | None = None,
    label_text: str | None = None,
    tangent_color: tuple[int, int, int] = (255, 255, 0),
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

        cv2.arrowedLine(
            patch_with_pose,
            (center_x, center_y),
            (tangent_end_x, tangent_end_y),
            tangent_color,
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


def save_raw_rgb_if_needed(
    save_raw_rgb: bool,
    is_exploring: bool,
    observed_state,
    rgba_image: np.ndarray,
    raw_rgb_base_dir: Path,
    episode_counter: int,
    step_counter: int,
) -> int:
    """Save raw RGB patch if saving is enabled and conditions are met.

    Args:
        save_raw_rgb: Whether to save raw RGB patches.
        is_exploring: Whether the agent is currently exploring.
        observed_state: Processed state from observation processor.
        rgba_image: RGBA image patch from raw observation data.
        raw_rgb_base_dir: Directory where raw RGB images should be saved.
        episode_counter: Current episode number.
        step_counter: Current step number within the episode.

    Returns:
        Updated step_counter (incremented if image was saved).
    """
    if not (save_raw_rgb and not is_exploring and observed_state.get_on_object()):
        return step_counter

    # Extract RGB patch from RGBA if needed
    if rgba_image.shape[2] == 4:
        patch = rgba_image[:, :, :3]
    else:
        patch = rgba_image

    # Create directory if it doesn't exist
    raw_rgb_base_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename: ep{episode:02d}_step{step:03d}.png
    # Uses same counters as debug_visualize
    filename = f"ep{episode_counter:02d}_step{step_counter:03d}.png"
    filepath = raw_rgb_base_dir / filename
    save_raw_rgb_patch(patch, str(filepath))

    return step_counter + 1
