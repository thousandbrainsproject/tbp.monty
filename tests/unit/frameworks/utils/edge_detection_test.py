# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
from hypothesis import example, given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetectionConfig,
    compute_weighted_structure_tensor_edge_features,
    edge_angle_to_2d_pose,
    edge_angle_to_3d_tangent,
    gradient_to_tangent_angle,
    is_geometric_edge,
)


angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)


class GradientToTangentAngleTest(unittest.TestCase):
    """Property-based tests for gradient_to_tangent_angle."""

    @given(gradient_angle=angles)
    def test_result_in_range(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        assert 0.0 <= result < 2 * np.pi

    @given(gradient_angle=angles)
    def test_perpendicularity(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        remainder = (result - gradient_angle) % np.pi
        self.assertAlmostEqual(remainder, np.pi / 2)


class IsGeometricEdgeTest(unittest.TestCase):
    """Unit tests for the is_geometric_edge function."""

    def _make_flat_patch(self, size=32, depth=1.0):
        return np.full((size, size), depth, dtype=np.float32)

    def _make_step_patch(self, size=32):
        """Left half=0.0, right half=1.0 (vertical step edge).

        Returns:
            Grayscale float32 array with a vertical step at the midpoint.
        """
        patch = np.zeros((size, size), dtype=np.float32)
        patch[:, size // 2 :] = 1.0
        return patch

    def test_flat_depth_returns_false(self):
        patch = self._make_flat_patch()
        self.assertFalse(is_geometric_edge(patch, edge_theta=0.0))

    def test_step_edge_perpendicular_returns_true(self):
        # Vertical step creates horizontal gradient (dx).
        # edge_theta=pi/2 (vertical tangent) => edge normal at pi (along -x),
        # which is aligned with the horizontal depth gradient => geometric edge.
        patch = self._make_step_patch()
        self.assertTrue(is_geometric_edge(patch, edge_theta=np.pi / 2))

    def test_step_edge_parallel_returns_false(self):
        # edge_theta=0 (horizontal tangent) => edge normal at pi/2 (along y),
        # perpendicular to horizontal depth gradient => not detected.
        patch = self._make_step_patch()
        self.assertFalse(is_geometric_edge(patch, edge_theta=0.0))

    def test_threshold_boundary(self):
        # Mild gradient: small step, edge_theta=pi/2 so normal aligns with dx.
        patch = np.zeros((32, 32), dtype=np.float32)
        patch[:, 16:] = 0.001
        self.assertFalse(
            is_geometric_edge(patch, edge_theta=np.pi / 2, depth_threshold=1.0)
        )
        self.assertTrue(
            is_geometric_edge(patch, edge_theta=np.pi / 2, depth_threshold=1e-6)
        )


class EdgeAngleTo3dTangentTest(unittest.TestCase):
    """Unit tests for the edge_angle_to_3d_tangent function."""

    def test_identity_camera_z_normal_theta_zero(self):
        result = edge_angle_to_3d_tangent(
            theta=0.0,
            surface_normal=np.array([0.0, 0.0, 1.0]),
            world_camera=np.eye(4),
        )
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_identity_camera_z_normal_theta_pi_half(self):
        result = edge_angle_to_3d_tangent(
            theta=np.pi / 2,
            surface_normal=np.array([0.0, 0.0, 1.0]),
            world_camera=np.eye(4),
        )
        # ty = cross(n, tx) = cross([0,0,1], [1,0,0]) = [0,1,0]
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])

    def test_result_is_unit_vector(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]:
            for normal in [
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
            ]:
                result = edge_angle_to_3d_tangent(theta, normal, np.eye(4))
                self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)

    def test_result_orthogonal_to_normal(self):
        for theta in [0, np.pi / 6, np.pi / 3, np.pi]:
            for normal in [
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
            ]:
                n_unit = normal / np.linalg.norm(normal)
                result = edge_angle_to_3d_tangent(theta, normal, np.eye(4))
                self.assertAlmostEqual(np.dot(result, n_unit), 0.0, places=5)

    def test_4x4_matrix_with_translation(self):
        cam_with_translation = np.eye(4)
        cam_with_translation[:3, 3] = [1.0, 2.0, 3.0]
        result = edge_angle_to_3d_tangent(
            theta=0.0,
            surface_normal=np.array([0.0, 0.0, 1.0]),
            world_camera=cam_with_translation,
        )
        result_identity = edge_angle_to_3d_tangent(
            theta=0.0,
            surface_normal=np.array([0.0, 0.0, 1.0]),
            world_camera=np.eye(4),
        )
        # Translation should not affect the tangent direction
        np.testing.assert_array_almost_equal(result, result_identity)

    def test_zero_normal_raises(self):
        with self.assertRaises(ValueError):
            edge_angle_to_3d_tangent(
                theta=0.0,
                surface_normal=np.array([0.0, 0.0, 0.0]),
                world_camera=np.eye(4),
            )

    def test_oblique_normal(self):
        normal = np.array([1.0, 1.0, 1.0])
        result = edge_angle_to_3d_tangent(0.0, normal, np.eye(4))
        n_unit = normal / np.linalg.norm(normal)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
        self.assertAlmostEqual(np.dot(result, n_unit), 0.0, places=5)

    def test_normal_along_x_axis(self):
        normal = np.array([1.0, 0.0, 0.0])
        result = edge_angle_to_3d_tangent(0.0, normal, np.eye(4))
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
        self.assertAlmostEqual(result[0], 0.0, places=5)


class EdgeAngleTo2dPoseTest(unittest.TestCase):
    """Unit tests for the edge_angle_to_2d_pose function."""

    def test_identity_camera_theta_zero(self):
        pose = edge_angle_to_2d_pose(theta=0.0, world_camera=np.eye(4))
        np.testing.assert_array_almost_equal(pose[0], [0, 0, 1])
        np.testing.assert_array_almost_equal(pose[1], [1, 0, 0])
        np.testing.assert_array_almost_equal(pose[2], [0, 1, 0])

    def test_identity_camera_theta_pi_half(self):
        pose = edge_angle_to_2d_pose(theta=np.pi / 2, world_camera=np.eye(4))
        np.testing.assert_array_almost_equal(pose[0], [0, 0, 1])
        np.testing.assert_array_almost_equal(pose[1], [0, 1, 0])
        np.testing.assert_array_almost_equal(pose[2], [-1, 0, 0])

    def test_identity_camera_theta_pi_quarter(self):
        pose = edge_angle_to_2d_pose(theta=np.pi / 4, world_camera=np.eye(4))
        s2 = np.sqrt(2) / 2
        np.testing.assert_array_almost_equal(pose[0], [0, 0, 1])
        np.testing.assert_array_almost_equal(pose[1], [s2, s2, 0])
        np.testing.assert_array_almost_equal(pose[2], [-s2, s2, 0])

    def test_normal_always_001(self):
        for theta in [0, np.pi / 6, np.pi / 3, np.pi, 5.0]:
            pose = edge_angle_to_2d_pose(theta, np.eye(4))
            np.testing.assert_array_almost_equal(pose[0], [0, 0, 1])

    def test_z_component_always_zero_for_tangents(self):
        for theta in [0, np.pi / 4, np.pi / 2, np.pi]:
            pose = edge_angle_to_2d_pose(theta, np.eye(4))
            self.assertAlmostEqual(pose[1][2], 0.0)
            self.assertAlmostEqual(pose[2][2], 0.0)

    def test_orthonormality(self):
        for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            pose = edge_angle_to_2d_pose(theta, np.eye(4))
            for i in range(3):
                self.assertAlmostEqual(np.linalg.norm(pose[i]), 1.0, places=6)
            for i in range(3):
                for j in range(i + 1, 3):
                    self.assertAlmostEqual(
                        np.dot(pose[i], pose[j]), 0.0, places=6
                    )

    def test_tilted_camera_90_yaw(self):
        # 90-degree CCW rotation around z: world_camera maps world -> camera.
        # R = Rz(pi/2), so R.T @ [1,0,0] = first row of R = [0,1,0].
        R = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        cam = np.eye(4)
        cam[:3, :3] = R
        pose = edge_angle_to_2d_pose(theta=0.0, world_camera=cam)
        # ref_angle = atan2(1,0) = pi/2, world_theta = pi/2
        np.testing.assert_array_almost_equal(pose[1], [0, 1, 0])
        np.testing.assert_array_almost_equal(pose[2], [-1, 0, 0])

    def test_translation_does_not_affect_result(self):
        cam = np.eye(4)
        cam[:3, 3] = [10.0, 20.0, 30.0]
        pose_translated = edge_angle_to_2d_pose(np.pi / 3, cam)
        pose_identity = edge_angle_to_2d_pose(np.pi / 3, np.eye(4))
        np.testing.assert_array_almost_equal(pose_translated, pose_identity)


class ComputeWeightedStructureTensorEdgeFeaturesTest(unittest.TestCase):
    """Unit tests for compute_weighted_structure_tensor_edge_features."""

    @staticmethod
    def _make_rgb_patch(size, pattern) -> np.ndarray:
        """Generate synthetic RGB patches for testing.

        Args:
            size: Patch dimension (square).
            pattern: One of "uniform", "vertical_edge", "horizontal_edge",
                "diagonal_edge".

        Returns:
            uint8 RGB array of shape (size, size, 3).

        Raises:
            ValueError: If pattern is not recognized.
        """
        if pattern == "uniform":
            return np.full((size, size, 3), 128, dtype=np.uint8)
        if pattern == "vertical_edge":
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            patch[:, size // 2 :] = 255
            return patch
        if pattern == "horizontal_edge":
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            patch[size // 2 :, :] = 255
            return patch
        if pattern == "diagonal_edge":
            patch = np.zeros((size, size, 3), dtype=np.uint8)
            for r in range(size):
                patch[r, r:] = 255
            return patch
        raise ValueError(f"Unknown pattern: {pattern}")

    def test_uniform_patch_returns_zero_strength(self):
        patch = self._make_rgb_patch(32, "uniform")
        strength, _coherence, _theta = compute_weighted_structure_tensor_edge_features(
            patch
        )
        self.assertAlmostEqual(strength, 0.0)

    def test_vertical_edge_detected(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        strength, coherence, _ = compute_weighted_structure_tensor_edge_features(patch)
        self.assertGreater(strength, 0.0)
        self.assertGreater(coherence, 0.0)

    def test_vertical_edge_orientation(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        _, _, theta = compute_weighted_structure_tensor_edge_features(patch)
        # Vertical edge tangent should be near pi/2 or 3*pi/2
        angle_to_vertical = min(abs(theta - np.pi / 2), abs(theta - 3 * np.pi / 2))
        self.assertLess(angle_to_vertical, 0.3)

    def test_horizontal_edge_orientation(self):
        patch = self._make_rgb_patch(32, "horizontal_edge")
        _, _, theta = compute_weighted_structure_tensor_edge_features(patch)
        # Horizontal edge tangent should be near 0 or pi
        angle_to_horizontal = min(abs(theta), abs(theta - np.pi))
        self.assertLess(angle_to_horizontal, 0.3)

    def test_default_params_used_when_none(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        result = compute_weighted_structure_tensor_edge_features(
            patch, edge_detection_config=None
        )
        self.assertEqual(len(result), 3)

    def test_returns_three_floats(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        result = compute_weighted_structure_tensor_edge_features(patch)
        self.assertEqual(len(result), 3)
        for val in result:
            self.assertIsInstance(val, float)

    def test_center_offset_rejects_off_center_edge(self):
        # Edge at right boundary, not at center
        patch = np.full((32, 32, 3), 0, dtype=np.uint8)
        patch[:, 28:] = 255
        config = EdgeDetectionConfig(max_center_offset=1)
        strength, coherence, theta = compute_weighted_structure_tensor_edge_features(
            patch, config
        )
        self.assertAlmostEqual(strength, 0.0)
        self.assertAlmostEqual(coherence, 0.0)
        self.assertIsNone(theta)

    def test_coherence_in_zero_one_range(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        _, coherence, _ = compute_weighted_structure_tensor_edge_features(patch)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)

    def test_tangent_theta_in_valid_range(self):
        patch = self._make_rgb_patch(32, "vertical_edge")
        _, _, theta = compute_weighted_structure_tensor_edge_features(patch)
        self.assertGreaterEqual(theta, 0.0)
        self.assertLess(theta, 2 * np.pi)


if __name__ == "__main__":
    unittest.main()
