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
from typing import NamedTuple

import cv2
import numpy as np

from tbp.monty.math import DIVISION_BY_ZERO_GUARD

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
    max_center_offset: int | None = None


def is_geometric_edge(
    depth_patch: np.ndarray,
    edge_theta: float,
    depth_threshold: float = 0.01,
) -> bool:
    """Check if detected edge is a geometric edge (depth discontinuity).

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
        edge_theta: Edge tangent angle in radians from RGB edge detection.
        depth_threshold: Maximum allowed depth gradient magnitude for texture
            edges. Edges with perpendicular depth gradient above this value
            are classified as geometric.

    Returns:
        True if edge is geometric, False if texture edge.
    """
    depth_dx = cv2.Sobel(depth_patch, cv2.CV_32F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(depth_patch, cv2.CV_32F, 0, 1, ksize=3)

    edge_normal_angle = edge_theta + np.pi / 2
    nx = np.cos(edge_normal_angle)
    ny = np.sin(edge_normal_angle)

    cy, cx = depth_patch.shape[0] // 2, depth_patch.shape[1] // 2
    depth_gradient_perp = abs(nx * depth_dx[cy, cx] + ny * depth_dy[cy, cx])

    return depth_gradient_perp > depth_threshold


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


def edge_angle_to_2d_pose(
    theta: float,
    world_camera: np.ndarray,
) -> np.ndarray:
    """Build 2D pose vectors from an edge angle.

    Returns the standard basis rotated by the edge angle in the world
    xy-plane. When the camera is tilted, the camera's image x-axis is
    projected into the xy-plane to determine the reference direction.

    Args:
        theta: Edge angle in radians in image coordinates (y-down), measured
            counterclockwise from the image x-axis.
        world_camera: 4x4 world-to-camera transformation matrix.

    Returns:
        3x3 array whose rows are [normal, edge_tangent, edge_perp].
        Normal is always [0, 0, 1]; tangent and perp lie in the z=0 plane.
    """
    R = world_camera[:3, :3]  # noqa: N806
    image_x_world = R.T @ np.array([1.0, 0.0, 0.0])
    ref_angle = np.arctan2(image_x_world[1], image_x_world[0])
    world_theta = ref_angle - theta

    cos_t, sin_t = np.cos(world_theta), np.sin(world_theta)
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
        ]
    )


@dataclass
class StructureTensor:
    """Structure tensor at a single point: a 2x2 symmetric matrix [[Jxx, Jxy], [Jxy, Jyy]]."""

    Jxx: float
    Jyy: float
    Jxy: float

    @property
    def eigenvalues(self) -> tuple[float, float]:
        """Returns (lambda_min, lambda_max) of the 2x2 structure tensor."""
        matrix = np.array([[self.Jxx, self.Jxy], [self.Jxy, self.Jyy]])
        lambda_min, lambda_max = np.linalg.eigh(matrix)[0]
        return lambda_min, lambda_max

    @property
    def gradient_theta(self) -> float:
        """Gradient direction in radians (normal to the dominant edge)."""
        return 0.5 * np.arctan2(2.0 * self.Jxy, self.Jxx - self.Jyy)

    @property
    def edge_strength(self) -> float:
        """Magnitude of the dominant eigenvalue."""
        _, lambda_max = self.eigenvalues
        return np.sqrt(max(lambda_max, 0.0))

    @property
    def coherence(self) -> float:
        """Edge quality in [0, 1]: 1 means perfectly oriented, 0 means isotropic."""
        lambda_min, lambda_max = self.eigenvalues
        return (lambda_max - lambda_min) / (lambda_max + lambda_min + DIVISION_BY_ZERO_GUARD)

    @property
    def edge_orientation(self) -> float:
        """Edge orientation angle in [0, pi] radians."""
        return gradient_to_tangent_angle(self.gradient_theta)


class EdgeFeatures(NamedTuple):
    """Edge features extracted from a single image patch."""

    strength: float
    coherence: float
    orientation: float | None


def _compute_sobel_gradients(
    gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical Sobel gradients.

    Args:
        gray: Grayscale image patch as float32 in [0, 1].

    Returns:
        Tuple of (Ix, Iy): horizontal and vertical gradient arrays.
    """
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # noqa: N806
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # noqa: N806
    return Ix, Iy


def _compute_per_pixel_structure_tensors(
    Ix: np.ndarray,  # noqa: N803
    Iy: np.ndarray,  # noqa: N803
    config: EdgeDetectionConfig,
) -> np.ndarray:
    """Build the smoothed per-pixel structure tensor field from Sobel gradients.

    Args:
        Ix: Horizontal Sobel gradients.
        Iy: Vertical Sobel gradients.
        config: Edge detection configuration.

    Returns:
        Array of shape (h, w, 2, 2) where each entry is the Gaussian-smoothed
        2x2 structure tensor [[Jxx, Jxy], [Jxy, Jyy]] at that pixel.
    """
    Jxx = Ix * Ix  # noqa: N806  # (h, w)
    Jyy = Iy * Iy  # noqa: N806  # (h, w)
    Jxy = Ix * Iy  # noqa: N806  # (h, w)

    ksize = config.kernel_size
    sigma = config.gaussian_sigma
    Jxx = cv2.GaussianBlur(Jxx, (ksize, ksize), sigma)  # noqa: N806  # (h, w)
    Jyy = cv2.GaussianBlur(Jyy, (ksize, ksize), sigma)  # noqa: N806  # (h, w)
    Jxy = cv2.GaussianBlur(Jxy, (ksize, ksize), sigma)  # noqa: N806  # (h, w)

    h, w = Jxx.shape
    tensor_per_pixel = np.empty((h, w, 2, 2), dtype=np.float32)
    tensor_per_pixel[..., 0, 0] = Jxx
    tensor_per_pixel[..., 1, 1] = Jyy
    tensor_per_pixel[..., 0, 1] = Jxy
    tensor_per_pixel[..., 1, 0] = Jxy
    return tensor_per_pixel


def _compute_center_weights(
    shape: tuple[int, int],
    Ix: np.ndarray,  # noqa: N803
    Iy: np.ndarray,  # noqa: N803
    config: EdgeDetectionConfig,
) -> tuple[np.ndarray, np.floating]:
    """Build radial + gradient-strength weight map centered on the patch.

    Weights combine a Gaussian radial falloff (suppressing far-from-center pixels)
    with local gradient magnitude (so strong off-center edges still contribute).

    Args:
        shape: (height, width) of the patch.
        Ix: Horizontal Sobel gradients.
        Iy: Vertical Sobel gradients.
        config: Edge detection configuration.

    Returns:
        Tuple of (weights, total_weight).
    """
    h, w = shape
    r0, c0 = h // 2, w // 2

    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d_squared = (rows - r0) ** 2 + (cols - c0) ** 2
    d = np.sqrt(d_squared)

    w_r = np.exp(-d_squared / (2.0 * config.sigma_r**2))
    w_r[d > config.radius] = 0.0
    weights = w_r * (Ix**2 + Iy**2)

    total_weight = np.sum(weights)
    return weights, total_weight


def _aggregate_tensor(
    tensor_per_pixel: np.ndarray,
    weights: np.ndarray,
    total_weight: float,
) -> StructureTensor:
    """Reduce a per-pixel structure tensor field to a single representative tensor.

    Args:
        tensor_per_pixel: Per-pixel structure tensors, shape (h, w, 2, 2).
        weights: Per-pixel weights, shape (h, w).
        total_weight: Sum of weights (must be > 0).

    Returns:
        StructureTensor representing the weighted aggregate over the patch.
    """
    w = weights[..., np.newaxis, np.newaxis]
    aggregated = np.sum(w * tensor_per_pixel, axis=(0, 1)) / total_weight
    return StructureTensor(
        Jxx=float(aggregated[0, 0]),
        Jyy=float(aggregated[1, 1]),
        Jxy=float(aggregated[0, 1]),
    )


def _passes_center_check(
    weights: np.ndarray,
    total_weight: np.floating,
    gradient_theta: float,
    max_center_offset: int | None,
) -> bool:
    """Return True if the detected edge passes close enough to the patch center.

    Args:
        weights: Per-pixel weights.
        total_weight: Sum of weights.
        gradient_theta: Gradient direction in radians (normal to edge).
        max_center_offset: Maximum allowed weighted distance from center, or None
            to skip the check.

    Returns:
        True if edge passes the center check (or check is disabled).
    """
    if max_center_offset is None:
        return True

    h, w = weights.shape
    r0, c0 = h // 2, w // 2
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    nx = np.cos(gradient_theta)
    ny = np.sin(gradient_theta)
    dist_normal = nx * (cols - c0) + ny * (rows - r0)
    d_center = np.sum(weights * dist_normal) / total_weight

    return abs(d_center) <= max_center_offset


def compute_edge_features(
    patch: np.ndarray,
    edge_detection_config: EdgeDetectionConfig | None = None,
) -> EdgeFeatures:
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
        EdgeFeatures with:
            - strength: Magnitude of dominant eigenvalue (0.0 if no edge)
            - coherence: Edge quality metric in [0, 1] (0.0 if no edge)
            - orientation: Edge orientation angle in [0, pi) radians (None if no edge)

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

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    Ix, Iy = _compute_sobel_gradients(gray)  # noqa: N806
    tensor_per_pixel = _compute_per_pixel_structure_tensors(Ix, Iy, edge_detection_config)

    weights, total_weight = _compute_center_weights(
        gray.shape, Ix, Iy, edge_detection_config
    )
    if total_weight < DIVISION_BY_ZERO_GUARD:
        return EdgeFeatures(strength=0.0, coherence=0.0, orientation=None)

    aggregated = _aggregate_tensor(tensor_per_pixel, weights, total_weight)

    if not _passes_center_check(
        weights,
        total_weight,
        aggregated.gradient_theta,
        edge_detection_config.max_center_offset,
    ):
        return EdgeFeatures(strength=0.0, coherence=0.0, orientation=None)

    return EdgeFeatures(
        strength=float(aggregated.edge_strength),
        coherence=float(aggregated.coherence),
        orientation=float(aggregated.edge_orientation),
    )
