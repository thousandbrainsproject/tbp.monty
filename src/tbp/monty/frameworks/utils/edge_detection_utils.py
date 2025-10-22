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
import numpy as np

DEFAULT_WINDOW_SIGMA = 1.0
DEFAULT_KERNEL_SIZE = 7
SOBEL_KERNEL_SIZE = 3
EPSILON = 1e-12


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
