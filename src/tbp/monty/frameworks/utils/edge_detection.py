# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import cv2
import numpy as np


def is_geometric_edge(
    depth_patch: np.ndarray,
    edge_tangent_theta: float,
    depth_delta_threshold: float = 0.01,
) -> bool:
    """Check if detected edge is a geometric edge (depth discontinuity) vs a texture edge.

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
