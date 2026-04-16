# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
    project_onto_tangent_plane,
)

logger = logging.getLogger(__name__)


@dataclass
class EdgeDetectionConfig:
    """Configuration for structure-tensor edge detection.

    Controls the Gaussian smoothing, center-weighted aggregation, thresholds
    for edge acceptance, and depth-based geometric edge filtering.
    """

    gaussian_sigma: float = 1.0
    kernel_size: int = 7
    strength_threshold: float = 0.1
    coherence_threshold: float = 0.5
    radius: float = 14.0
    sigma_r: float = 7.0
    depth_edge_threshold: float = 0.01
    max_center_offset: Optional[int] = None


def is_geometric_edge(
    depth_patch: np.ndarray,
    edge_tangent_theta: float,
    depth_delta_threshold: float = 0.01,
) -> bool:
    """Check if detected edge is a geometric edge (depth discontinuity) vs a 2D texture edge.

    Geometric edges occur at object boundaries or surface creases where depth
    changes abruptly. Texture edges will be detected wherever there is an abrupt
    discontinuity in image intensity. We will use detected geometric edges to identify
    candidate texture edges that do not correspond to a 2D surface (such as where the
    red handle of a mug is seen against the black background of a simulator's void).
    This function computes the depth gradient perpendicular to the detected edge
    direction and checks if it exceeds a threshold.

    Args:
        depth_patch: Depth image patch (same size as RGB patch used for edge
            detection). Values should be in consistent units (e.g., meters).
        edge_tangent_theta: Edge tangent angle in radians from RGB edge detection.
        depth_delta_threshold: Maximum allowed depth gradient magnitude for texture
            edges. Edges with perpendicular depth gradient above this value
            are classified as geometric.

    Returns:
        True if edge is geometric, False if texture edge.
    """
    depth_dx = cv2.Sobel(depth_patch, cv2.CV_32F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(depth_patch, cv2.CV_32F, 0, 1, ksize=3)

    edge_normal_angle = edge_tangent_theta + np.pi / 2
    nx = np.cos(edge_normal_angle)
    ny = np.sin(edge_normal_angle)

    cy, cx = depth_patch.shape[0] // 2, depth_patch.shape[1] // 2
    depth_gradient_perp = abs(nx * depth_dx[cy, cx] + ny * depth_dy[cy, cx])

    return depth_gradient_perp > depth_delta_threshold


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


def edge_angle_to_3d_tangent(
    theta: float,
    surface_normal: np.ndarray,
    world_camera: np.ndarray,
) -> np.ndarray:
    """Project a 2D edge angle from an image to a 3D tangent vector on a surface.

    Builds an orthonormal tangent basis (tx, ty) on the surface, aligned with
    the camera's image axes, then combines them using theta.

    Args:
        theta: Edge angle in radians, measured counterclockwise image x axis.
            In an image, +x is right and +y is down.
        surface_normal: Surface normal vector in world frame.
        world_camera: 4x4 world-to-camera transformation matrix.

    Returns:
        3D unit tangent vector in world frame.
    """
    surface_normal = normalize(surface_normal)

    R = world_camera[:3, :3]  # noqa: N806

    # Camera x-axis ("image right") expressed in world coordinates
    image_x_world = R.T @ np.array([1.0, 0.0, 0.0])

    tx = project_onto_tangent_plane(image_x_world, surface_normal)
    if np.linalg.norm(tx) < 1e-12:
        fallback = R.T @ np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fallback, surface_normal)) > 0.99:
            fallback = R.T @ np.array([0.0, 1.0, 0.0])
        tx = project_onto_tangent_plane(fallback, surface_normal)
    tx = normalize(tx)

    ty = normalize(np.cross(surface_normal, tx))

    t_world = np.cos(theta) * tx + np.sin(theta) * ty
    return normalize(t_world)


def edge_angle_to_2d_pose(
    theta: float,
    world_camera: np.ndarray,
) -> np.ndarray:
    """Build 2D pose vectors from an edge angle.

    Returns the standard basis rotated by the edge angle in the world
    xy-plane. When the camera is tilted, the camera's image x-axis is
    projected into the xy-plane to determine the reference direction.

    Args:
        theta: Edge angle in radians (counterclockwise from image x-axis).
        world_camera: 4x4 world-to-camera transformation matrix.

    Returns:
        3x3 array whose rows are [normal, edge_tangent, edge_perp].
        Normal is always [0, 0, 1]; tangent and perp lie in the z=0 plane.
    """
    R = world_camera[:3, :3]  # noqa: N806
    image_x_world = R.T @ np.array([1.0, 0.0, 0.0])
    ref_angle = np.arctan2(image_x_world[1], image_x_world[0])
    world_theta = ref_angle + theta

    cos_t, sin_t = np.cos(world_theta), np.sin(world_theta)
    return np.array([
        [0.0, 0.0, 1.0],
        [cos_t, sin_t, 0.0],
        [-sin_t, cos_t, 0.0],
    ])


def compute_weighted_structure_tensor_edge_features(
    patch: np.ndarray,
    edge_detection_config: Optional[EdgeDetectionConfig] = None,
) -> Tuple[float, float, Optional[float]]:
    """Compute edge features using center-weighted, global-aware structure tensor.

    This function aggregates structure tensor components over a center-biased
    neighborhood, giving higher weight to pixels closer to the center and pixels
    with stronger gradients. Returns edge strength and coherence (in addition to
    orientation) for caller to threshold (i.e. reject weak or cluttered edges).

    Reference:
        Nazar Khan, "Corner Detection" lecture notes, Section on Structure
        Tensor. http://faculty.pucit.edu.pk/nazarkhan/teaching/Spring2021/CS565/Lectures/lecture6_corner_detection.pdf

    Args:
        patch: RGB image patch.
        edge_detection_config: Edge detection configuration parameters. If None, uses
            default EdgeDetectionConfig.

    Returns:
        Tuple of (edge_strength, coherence, tangent_theta):
            - edge_strength: Magnitude of dominant eigenvalue (0.0 if no edge)
            - coherence: Edge quality metric in [0, 1] (0.0 if no edge)
            - tangent_theta: Edge tangent angle in [0, 2*pi) radians (None if no edge)

    Notes:
        1. The Gaussian blur (Step 1) convolves all pixels in the patch with a
            Gaussian so as to effectively average their values by neighbors.
        2. The radial weight (Step 2b) is a single Gaussian placed at the center
            of the patch, and determines how much gradients associated with
            different pixels will be weighted based on their displacement from
            the center.
    """
    if edge_detection_config is None:
        edge_detection_config = EdgeDetectionConfig()

    # Step 1: Compute gradients and local tensor components
    blur_ksize = edge_detection_config.kernel_size
    win_sigma = edge_detection_config.gaussian_sigma

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # noqa: N806
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # noqa: N806

    Jxx = Ix * Ix  # noqa: N806
    Jyy = Iy * Iy  # noqa: N806
    Jxy = Ix * Iy  # noqa: N806

    Jxx = cv2.GaussianBlur(Jxx, (blur_ksize, blur_ksize), win_sigma)  # noqa: N806
    Jyy = cv2.GaussianBlur(Jyy, (blur_ksize, blur_ksize), win_sigma)  # noqa: N806
    Jxy = cv2.GaussianBlur(Jxy, (blur_ksize, blur_ksize), win_sigma)  # noqa: N806

    # Step 2a: Center-weighted aggregation
    r0, c0 = gray.shape[0] // 2, gray.shape[1] // 2
    h, w = gray.shape

    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d_squared = (rows - r0) ** 2 + (cols - c0) ** 2
    d = np.sqrt(d_squared)

    # Step 2b: Radial and Gradient Strength Weighting
    # Include gradient magnitude in weights so strong edges slightly off-center
    # still contribute meaningfully rather than being suppressed by radial falloff.
    w_r = np.exp(-(d_squared) / (2.0 * edge_detection_config.sigma_r**2))
    w_r[d > edge_detection_config.radius] = 0.0
    g = Ix**2 + Iy**2
    weights = w_r * g

    total_weight = np.sum(weights)
    if total_weight < 1e-12:
        return 0.0, 0.0, None

    Jxx_bar = np.sum(weights * Jxx) / total_weight  # noqa: N806
    Jyy_bar = np.sum(weights * Jyy) / total_weight  # noqa: N806
    Jxy_bar = np.sum(weights * Jxy) / total_weight  # noqa: N806

    # Step 3: Compute eigenvalues, edge strength, coherence, orientation
    disc = np.sqrt((Jxx_bar - Jyy_bar) ** 2 + 4.0 * (Jxy_bar**2))
    lam1 = 0.5 * (Jxx_bar + Jyy_bar + disc)
    lam2 = 0.5 * (Jxx_bar + Jyy_bar - disc)

    edge_strength = np.sqrt(max(lam1, 0.0))
    coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-12)

    gradient_theta = 0.5 * np.arctan2(2.0 * Jxy_bar, Jxx_bar - Jyy_bar)
    tangent_theta = gradient_to_tangent_angle(gradient_theta)

    # Step 4: Check if edge passes near patch center
    max_center_offset = edge_detection_config.max_center_offset
    if max_center_offset is not None:
        nx = np.cos(gradient_theta)
        ny = np.sin(gradient_theta)

        dr = rows - r0
        dc = cols - c0

        dist_normal = nx * dc + ny * dr
        d_center = np.sum(weights * dist_normal) / total_weight

        if abs(d_center) > max_center_offset:
            return 0.0, 0.0, None

    return float(edge_strength), float(coherence), float(tangent_theta)
