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

import cv2
import numpy as np

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.math import DEFAULT_TOLERANCE

logger = logging.getLogger(__name__)

DEFAULT_POSE_2D = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _gradient_to_tangent_angle(gradient_angle: float) -> float:
    """Convert gradient direction to edge tangent direction and wrap to [0, 2*pi).

    The edge tangent is perpendicular to the gradient direction.

    Args:
        gradient_angle: Gradient direction in radians (any range)

    Returns:
        Edge tangent angle in [0, 2*pi) radians
    """
    tangent_angle = gradient_angle + np.pi / 2
    return (tangent_angle + 2 * np.pi) % (2 * np.pi)


@dataclass
class StructureTensor:
    """Structure tensor at a single point.

    A 2x2 symmetric matrix [[Jxx, Jxy], [Jxy, Jyy]].
    """

    xx: float
    yy: float
    xy: float

    @property
    def eigenvalues(self) -> tuple[float, float]:
        """Returns (lambda_min, lambda_max) of the 2x2 structure tensor."""
        matrix = np.array([[self.xx, self.xy], [self.xy, self.yy]])
        lambda_min, lambda_max = np.linalg.eigh(matrix)[0]
        return lambda_min, lambda_max

    @property
    def gradient_theta(self) -> float:
        """Gradient direction in radians (normal to the dominant edge)."""
        return 0.5 * np.arctan2(2.0 * self.xy, self.xx - self.yy)

    @property
    def edge_strength(self) -> float:
        """Magnitude of the dominant eigenvalue."""
        _, lambda_max = self.eigenvalues
        return np.sqrt(max(lambda_max, 0.0))

    @property
    def coherence(self) -> float:
        """Edge quality in [0, 1]: 1 means perfectly oriented, 0 means isotropic."""
        lambda_min, lambda_max = self.eigenvalues
        lambda_diff = lambda_max - lambda_min
        if lambda_diff < DEFAULT_TOLERANCE:
            return 0.0
        return lambda_diff / (lambda_max + lambda_min)

    @property
    def edge_angle(self) -> float:
        """Edge angle in [0, pi] radians."""
        return _gradient_to_tangent_angle(self.gradient_theta)


@dataclass
class EdgeFeatures:
    """Edge features extracted from a single image patch."""

    strength: float
    coherence: float
    angle: float | None
    pose_2d: np.ndarray
    is_geometric_edge: bool
    has_edge: bool


class EdgeDetector:
    def __init__(
        self,
        gaussian_sigma: float = 1.0,
        kernel_size: int = 7,
        strength_threshold: float = 0.1,
        coherence_threshold: float = 0.5,
        radius: float = 14.0,
        sigma_r: float = 7.0,
        depth_edge_threshold: float = 0.01,
        max_center_offset: int | None = None,
    ):
        """Initialize EdgeDetector.

        Args:
            gaussian_sigma: float = 1.0
            kernel_size: int = 7
            strength_threshold: float = 0.1
            coherence_threshold: float = 0.5
            radius: float = 14.0
            sigma_r: float = 7.0
            depth_edge_threshold: float = 0.01
            max_center_offset: int | None = None

        Raises:
            ValueError: Invalid argument for gaussian_sigma.
        """
        self._gaussian_sigma = gaussian_sigma
        self._kernel_size = kernel_size
        self._strength_threshold = strength_threshold
        self._coherence_threshold = coherence_threshold
        self._radius = radius
        self._sigma_r = sigma_r
        self._depth_edge_threshold = depth_edge_threshold
        self._max_center_offset = max_center_offset

        if self._kernel_size % 2 != 1:
            raise ValueError("Kernel size must be odd.")

    def __call__(
        self,
        observation: SensorObservation,
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
            observation: Sensor observation.

        Returns:
            EdgeFeatures with:
                - strength: Magnitude of dominant eigenvalue (0.0 if no edge)
                - coherence: Edge quality metric in [0, 1] (0.0 if no edge)
                - orientation: Edge orientation angle in [0, pi) radians
                  (None if no edge)

        Notes:
            1. The Gaussian blur (Step 1) convolves all pixels in the patch with a
                Gaussian so as to effectively average their values by neighbors.
            2. The radial weight (Step 2b) is a single Gaussian placed at the center
                of the patch, and determines how much gradients associated with
                different pixels will be weighted based on their displacement from
                the center.
        """
        patch = observation["rgba"][:, :, :3]
        depth = observation["depth"]
        grayscale = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        Ix, Iy = self._compute_sobel_gradients(grayscale)  # noqa: N806
        tensor_per_pixel = self._compute_per_pixel_structure_tensors(Ix, Iy)

        weights, total_weight = self._compute_center_weights(grayscale.shape, Ix, Iy)
        if total_weight < DEFAULT_TOLERANCE:
            return EdgeFeatures(
                strength=0.0,
                coherence=0.0,
                angle=None,
                pose_2d=DEFAULT_POSE_2D,
                is_geometric_edge=False,
                has_edge=False,
            )

        aggregated = self._aggregate_tensor(tensor_per_pixel, weights, total_weight)

        if not self._passes_center_check(
            weights,
            total_weight,
            aggregated.gradient_theta,
        ):
            return EdgeFeatures(
                strength=0.0,
                coherence=0.0,
                angle=None,
                pose_2d=DEFAULT_POSE_2D,
                is_geometric_edge=False,
                has_edge=False,
            )

        has_edge = (
            aggregated.edge_strength > self._strength_threshold
            and aggregated.coherence > self._coherence_threshold
        )

        pose_2d = self._angle_to_pose_2d(
            aggregated.edge_angle, observation["world_camera"]
        )

        return EdgeFeatures(
            strength=float(aggregated.edge_strength),
            coherence=float(aggregated.coherence),
            angle=float(aggregated.edge_angle),
            pose_2d=pose_2d,
            is_geometric_edge=self._is_geometric_edge(depth, aggregated.edge_angle),
            has_edge=has_edge,
        )

    @staticmethod
    def _compute_sobel_gradients(
        grayscale: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute horizontal and vertical Sobel gradients.

        Args:
            grayscale: Grayscale image patch as float32 in [0, 1].

        Returns:
            Tuple of (Ix, Iy): horizontal and vertical gradient arrays.
        """
        Ix = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)  # noqa: N806
        Iy = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)  # noqa: N806
        return Ix, Iy

    def _compute_per_pixel_structure_tensors(
        self,
        Ix: np.ndarray,  # noqa: N803
        Iy: np.ndarray,  # noqa: N803
    ) -> np.ndarray:
        """Build the smoothed per-pixel structure tensor field from Sobel gradients.

        Args:
            Ix: Horizontal Sobel gradients.
            Iy: Vertical Sobel gradients.

        Returns:
            Array of shape (h, w, 2, 2) where each entry is the Gaussian-smoothed
            2x2 structure tensor [[Jxx, Jxy], [Jxy, Jyy]] at that pixel.
        """
        Jxx = Ix * Ix  # noqa: N806  # (h, w)
        Jyy = Iy * Iy  # noqa: N806  # (h, w)
        Jxy = Ix * Iy  # noqa: N806  # (h, w)

        ksize = self._kernel_size
        sigma = self._gaussian_sigma
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
        self,
        shape: tuple[int, int],
        Ix: np.ndarray,  # noqa: N803
        Iy: np.ndarray,  # noqa: N803
    ) -> tuple[np.ndarray, np.floating]:
        """Build radial + gradient-strength weight map centered on the patch.

        Weights combine a Gaussian radial falloff (suppressing far-from-center pixels)
        with local gradient magnitude (so strong off-center edges still contribute).

        Args:
            shape: (height, width) of the patch.
            Ix: Horizontal Sobel gradients.
            Iy: Vertical Sobel gradients.

        Returns:
            Tuple of (weights, total_weight).
        """
        h, w = shape
        r0, c0 = h // 2, w // 2

        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d_squared = (rows - r0) ** 2 + (cols - c0) ** 2
        d = np.sqrt(d_squared)

        w_r = np.exp(-d_squared / (2.0 * self._sigma_r**2))
        w_r[d > self._radius] = 0.0
        weights = w_r * (Ix**2 + Iy**2)

        total_weight = np.sum(weights)
        return weights, total_weight

    @staticmethod
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
            xx=float(aggregated[0, 0]),
            yy=float(aggregated[1, 1]),
            xy=float(aggregated[0, 1]),
        )

    def _passes_center_check(
        self,
        weights: np.ndarray,
        total_weight: np.floating,
        gradient_theta: float,
    ) -> bool:
        """Return True if the detected edge passes close enough to the patch center.

        Args:
            weights: Per-pixel weights.
            total_weight: Sum of weights.
            gradient_theta: Gradient direction in radians (normal to edge).

        Returns:
            True if edge passes the center check (or check is disabled).
        """
        if self._max_center_offset is None:
            return True

        h, w = weights.shape
        r0, c0 = h // 2, w // 2
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        nx = np.cos(gradient_theta)
        ny = np.sin(gradient_theta)
        dist_normal = nx * (cols - c0) + ny * (rows - r0)
        d_center = np.sum(weights * dist_normal) / total_weight

        return abs(d_center) <= self._max_center_offset

    def _is_geometric_edge(
        self,
        depth_patch: np.ndarray,
        edge_theta: float,
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

        return depth_gradient_perp > self._depth_edge_threshold

    def _angle_to_pose_2d(
        self,
        angle: float,
        world_camera: np.ndarray,
    ) -> np.ndarray:
        """Build 2D pose vectors from an edge angle.

        Returns the standard basis rotated by the edge angle in the world
        xy-plane. When the camera is tilted, the camera's image x-axis is
        projected into the xy-plane to determine the reference direction.

        Args:
            angle: Edge angle in radians in image coordinates (y-down), measured
                from the image x-axis toward the image y-axis (i.e., toward the
                bottom of the image).
            world_camera: 4x4 world-to-camera transformation matrix.

        Returns:
            3x3 array whose rows are [normal, edge_tangent, edge_perp].
            Normal is always [0, 0, 1]; tangent and perp lie in the z=0 plane.
        """
        R = world_camera[:3, :3]  # noqa: N806
        image_x_world = R.T @ np.array([1.0, 0.0, 0.0])
        ref_angle = np.arctan2(image_x_world[1], image_x_world[0])
        world_theta = ref_angle - angle

        cos_t, sin_t = np.cos(world_theta), np.sin(world_theta)
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [cos_t, sin_t, 0.0],
                [-sin_t, cos_t, 0.0],
            ]
        )
