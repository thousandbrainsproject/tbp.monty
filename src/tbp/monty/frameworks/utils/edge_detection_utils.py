# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Utility functions for 2D edge detection and pose extraction."""

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, scharr_h, scharr_v, sobel_h, sobel_v


def detect_edges_rgb(
    rgba_image,
    method="sobel",
    gaussian_sigma=1.0,
    edge_threshold=0.1,
    non_max_suppression=True,
    return_gradients=True,
):
    """Detect edges in RGB image using various methods.

    Args:
        rgba_image: Input RGBA image of shape (H, W, 4)
        method: Edge detection method ('sobel', 'scharr', 'canny')
        gaussian_sigma: Standard deviation for Gaussian smoothing
        edge_threshold: Minimum edge strength threshold
        non_max_suppression: Whether to apply non-maximum suppression
        return_gradients: Whether to return gradient components

    Returns:
        Dictionary containing:
            - edge_magnitude: Edge strength at each pixel
            - edge_orientation: Edge orientation in radians [0, π]
            - edge_mask: Binary mask of detected edges
            - gradients: (grad_x, grad_y) if return_gradients is True
    """
    # Convert to grayscale
    gray_image = rgb2gray(rgba_image[:, :, :3])

    # Apply Gaussian smoothing if requested
    if gaussian_sigma > 0:
        gray_image = gaussian(gray_image, sigma=gaussian_sigma, preserve_range=True)

    # Compute gradients based on method
    if method == "sobel":
        grad_x = sobel_v(gray_image)
        grad_y = sobel_h(gray_image)
    elif method == "scharr":
        grad_x = scharr_v(gray_image)
        grad_y = scharr_h(gray_image)
    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    # Calculate magnitude and orientation
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_orientation = np.arctan2(grad_y, grad_x)

    # Normalize orientation to [0, π] for undirected edges
    edge_orientation = np.abs(edge_orientation)
    edge_orientation = np.minimum(edge_orientation, np.pi - edge_orientation)

    # Apply non-maximum suppression
    if non_max_suppression:
        edge_magnitude = _non_maximum_suppression(
            edge_magnitude,
            edge_orientation + np.pi / 2,  # Perpendicular to gradient
        )

    # Create binary edge mask
    edge_mask = edge_magnitude > edge_threshold

    result = {
        "edge_magnitude": edge_magnitude,
        "edge_orientation": edge_orientation,
        "edge_mask": edge_mask,
    }

    if return_gradients:
        result["gradients"] = (grad_x, grad_y)

    return result


def _non_maximum_suppression(magnitude, orientation):
    """Apply non-maximum suppression to edge magnitude.

    Args:
        magnitude: Edge magnitude array
        orientation: Edge orientation array in radians

    Returns:
        Suppressed edge magnitude array
    """
    suppressed = np.zeros_like(magnitude)
    h, w = magnitude.shape

    # Convert orientation to degrees for easier handling
    angle = orientation * 180 / np.pi
    angle[angle < 0] += 180  # Ensure positive angles

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            try:
                q = 255
                r = 255

                # Angle 0-22.5 and 157.5-180
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 22.5-67.5
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 67.5-112.5
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 112.5-157.5
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

            except IndexError:
                pass

    return suppressed


def extract_local_edge_features(
    edge_info, center_coord, neighborhood_size=5, compute_curvature=False
):
    """Extract local edge features around a center point.

    Args:
        edge_info: Dictionary from detect_edges_rgb
        center_coord: (row, col) center coordinate
        neighborhood_size: Size of local neighborhood to analyze
        compute_curvature: Whether to compute local edge curvature

    Returns:
        Dictionary with local edge features:
            - center_magnitude: Edge strength at center
            - center_orientation: Edge orientation at center
            - local_coherence: Consistency of edge orientation in neighborhood
            - edge_density: Fraction of neighborhood pixels that are edges
            - edge_curvature: Local curvature estimate (if requested)
    """
    center_row, center_col = center_coord
    edge_magnitude = edge_info["edge_magnitude"]
    edge_orientation = edge_info["edge_orientation"]
    edge_mask = edge_info["edge_mask"]

    # Extract center features
    center_magnitude = edge_magnitude[center_row, center_col]
    center_orientation = edge_orientation[center_row, center_col]

    # Define neighborhood
    half_size = neighborhood_size // 2
    row_start = max(0, center_row - half_size)
    row_end = min(edge_magnitude.shape[0], center_row + half_size + 1)
    col_start = max(0, center_col - half_size)
    col_end = min(edge_magnitude.shape[1], center_col + half_size + 1)

    # Extract neighborhood
    local_magnitude = edge_magnitude[row_start:row_end, col_start:col_end]
    local_orientation = edge_orientation[row_start:row_end, col_start:col_end]
    local_mask = edge_mask[row_start:row_end, col_start:col_end]

    # Calculate local coherence (orientation consistency)
    if np.sum(local_mask) > 1:
        edge_orientations = local_orientation[local_mask]
        # Calculate circular variance for orientation consistency
        mean_cos = np.mean(np.cos(2 * edge_orientations))
        mean_sin = np.mean(np.sin(2 * edge_orientations))
        local_coherence = np.sqrt(mean_cos**2 + mean_sin**2)
    else:
        local_coherence = 0.0

    # Calculate edge density
    edge_density = np.sum(local_mask) / local_mask.size

    result = {
        "center_magnitude": float(center_magnitude),
        "center_orientation": float(center_orientation),
        "local_coherence": float(local_coherence),
        "edge_density": float(edge_density),
    }

    # Compute edge curvature if requested
    if compute_curvature:
        curvature = _estimate_edge_curvature(
            local_orientation, local_mask, center_coord
        )
        result["edge_curvature"] = float(curvature)
    else:
        result["edge_curvature"] = 0.0

    return result


def _estimate_edge_curvature(orientation_map, edge_mask, center_coord):
    """Estimate local edge curvature from orientation changes.

    Args:
        orientation_map: Local orientation map
        edge_mask: Local edge mask
        center_coord: Center coordinate (relative to local patch)

    Returns:
        Estimated curvature value
    """
    if np.sum(edge_mask) < 3:
        return 0.0

    try:
        # Find edge pixels and their orientations
        edge_pixels = np.where(edge_mask)
        orientations = orientation_map[edge_pixels]

        # Calculate distances from center
        center_r, center_c = center_coord
        distances = np.sqrt(
            (edge_pixels[0] - center_r) ** 2 + (edge_pixels[1] - center_c) ** 2
        )

        # Sort by distance to trace along edge
        sorted_indices = np.argsort(distances)
        sorted_orientations = orientations[sorted_indices]

        # Calculate orientation changes
        if len(sorted_orientations) < 3:
            return 0.0

        # Compute discrete curvature as change in orientation per unit length
        orientation_changes = np.diff(sorted_orientations)
        # Handle angle wrapping
        orientation_changes = np.abs(orientation_changes)
        orientation_changes = np.minimum(
            orientation_changes, np.pi - orientation_changes
        )

        # Average curvature
        mean_curvature = np.mean(orientation_changes)
        return mean_curvature

    except Exception:
        return 0.0


def find_dominant_edge_direction(edge_info, center_coord, search_radius=3):
    """Find the dominant edge direction in a local neighborhood.

    This function is useful when multiple edges are present and we need
    to select the most prominent one.

    Args:
        edge_info: Dictionary from detect_edges_rgb
        center_coord: (row, col) center coordinate
        search_radius: Radius around center to search for edges

    Returns:
        Dictionary with dominant edge information:
            - orientation: Dominant edge orientation
            - strength: Strength of dominant edge
            - confidence: Confidence in the direction estimate
    """
    center_row, center_col = center_coord
    edge_magnitude = edge_info["edge_magnitude"]
    edge_orientation = edge_info["edge_orientation"]

    # Define search area
    row_start = max(0, center_row - search_radius)
    row_end = min(edge_magnitude.shape[0], center_row + search_radius + 1)
    col_start = max(0, center_col - search_radius)
    col_end = min(edge_magnitude.shape[1], center_col + search_radius + 1)

    # Extract local data
    local_magnitude = edge_magnitude[row_start:row_end, col_start:col_end]
    local_orientation = edge_orientation[row_start:row_end, col_start:col_end]

    # Weight orientations by magnitude
    if np.max(local_magnitude) > 0:
        weights = local_magnitude / np.max(local_magnitude)

        # Calculate weighted circular mean
        weighted_cos = np.sum(weights * np.cos(2 * local_orientation))
        weighted_sin = np.sum(weights * np.sin(2 * local_orientation))

        dominant_orientation = 0.5 * np.arctan2(weighted_sin, weighted_cos)
        if dominant_orientation < 0:
            dominant_orientation += np.pi

        # Calculate confidence as concentration of orientations
        confidence = np.sqrt(weighted_cos**2 + weighted_sin**2) / np.sum(weights)

        # Average strength in the neighborhood
        strength = np.mean(local_magnitude[local_magnitude > 0])

    else:
        dominant_orientation = 0.0
        strength = 0.0
        confidence = 0.0

    return {
        "orientation": float(dominant_orientation),
        "strength": float(strength),
        "confidence": float(confidence),
    }


def create_pose_vectors_from_edge(edge_orientation, surface_normal=None):
    """Create 3D pose vectors from 2D edge orientation.

    Args:
        edge_orientation: Edge orientation angle in radians [0, π]
        surface_normal: Optional 3D surface normal vector

    Returns:
        Tuple of (pose_vectors, pose_fully_defined):
            - pose_vectors: 3x3 array with pose vectors as rows
            - pose_fully_defined: Boolean (always False for 2D edges)
    """
    # Edge tangent vector in image plane (z=0)
    edge_tangent = np.array(
        [
            np.cos(edge_orientation),
            np.sin(edge_orientation),
            0.0,
        ]
    )

    # Perpendicular to edge in image plane
    edge_perpendicular = np.array(
        [
            -np.sin(edge_orientation),
            np.cos(edge_orientation),
            0.0,
        ]
    )

    # Use provided surface normal or default z-axis
    if surface_normal is not None:
        surface_normal = surface_normal / np.linalg.norm(surface_normal)
    else:
        surface_normal = np.array([0.0, 0.0, 1.0])

    # Construct pose matrix
    pose_vectors = np.vstack(
        [
            edge_tangent,
            edge_perpendicular,
            surface_normal,
        ]
    )

    # 2D edges don't fully define 3D pose
    pose_fully_defined = False

    return pose_vectors, pose_fully_defined


def project_movement_to_2d_surface(movement_3d, surface_normal):
    """Project 3D movement vector onto 2D surface.

    Args:
        movement_3d: 3D movement vector
        surface_normal: Surface normal vector

    Returns:
        2D movement vector projected onto surface plane
    """
    # Normalize surface normal
    normal = surface_normal / np.linalg.norm(surface_normal)

    # Project movement onto surface plane: v_2d = v_3d - (v_3d · n) * n
    normal_component = np.dot(movement_3d, normal)
    movement_2d = movement_3d - normal_component * normal

    return movement_2d


def center_coords(h, w):
    """Get center coordinates of patch.

    Args:
        h: Height of patch
        w: Width of patch

    Returns:
        Tuple of (row, col) center coordinates
    """
    return h // 2, w // 2


def gradient_to_tangent_angle(gradient_angle):
    """Convert gradient direction to edge tangent direction and wrap to [0, 2π).

    The edge tangent is perpendicular to the gradient direction.

    Args:
        gradient_angle: Gradient direction in radians (any range)

    Returns:
        Edge tangent angle in [0, 2π) radians
    """
    # Edge tangent is perpendicular to gradient (rotate by 90°)
    tangent_angle = gradient_angle + np.pi / 2
    # Wrap to [0, 2π)
    return (tangent_angle + 2 * np.pi) % (2 * np.pi)


def structure_tensor_center(patch, win_sigma=1.0, ksize=7):
    """Enhanced structure tensor at center with OpenCV implementation.

    This function computes the structure tensor using Gaussian-weighted gradient
    outer products in a local window. The structure tensor provides robust edge
    detection and orientation estimation by analyzing local gradient patterns.

    Args:
        patch: RGB or grayscale image patch
        win_sigma: Standard deviation for Gaussian window smoothing (default: 1.0)
        ksize: Kernel size for Gaussian blur (default: 7)

    Returns:
        Tuple of (edge_strength, coherence, tangent_theta):
            - edge_strength: Magnitude of dominant eigenvalue (edge strength)
            - coherence: Edge quality metric in [0, 1], where 1 is edge-like
            - tangent_theta: Edge tangent angle in [0, 2π) radians
    """
    # Convert to BGR for OpenCV (if RGB)
    if len(patch.shape) == 3:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # Compute gradients on luminance, normalized to [0, 1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Compute outer products
    Jxx = Ix * Ix
    Jyy = Iy * Iy
    Jxy = Ix * Iy

    # Apply Gaussian smoothing to integrate over local window
    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), win_sigma)
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), win_sigma)
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), win_sigma)

    # Extract center values
    r, c = center_coords(*gray.shape)
    jxx, jyy, jxy = float(Jxx[r, c]), float(Jyy[r, c]), float(Jxy[r, c])

    # Compute eigenvalues
    disc = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy**2))
    lam1 = 0.5 * (jxx + jyy + disc)  # Larger eigenvalue
    lam2 = 0.5 * (jxx + jyy - disc)  # Smaller eigenvalue

    # Edge strength is square root of larger eigenvalue
    edge_strength = np.sqrt(max(lam1, 0.0))

    # Coherence measures edge quality: ~[0,1], where 1=edge-like, 0=flat/noisy
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-12)

    # Compute gradient direction (eigenvector of larger eigenvalue)
    gradient_theta = 0.5 * np.arctan2(2.0 * jxy, (jxx - jyy + 1e-12))

    # Convert to edge tangent direction [0, 2π)
    tangent_theta = gradient_to_tangent_angle(gradient_theta)

    return edge_strength, coherence, float(tangent_theta)
