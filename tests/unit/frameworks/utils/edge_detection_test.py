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
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetectionConfig,
    StructureTensor,
    _compute_center_weights,
    _passes_center_check,
    compute_edge_features,
    edge_angle_to_2d_pose,
    gradient_to_tangent_angle,
    is_geometric_edge,
)
from tbp.monty.math import DEFAULT_TOLERANCE

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)
a_scalar = st.floats(min_value=DEFAULT_TOLERANCE, max_value=100.0)


@st.composite
def structure_tensors(draw, max_value=100.0, allow_zero_matrix=True):
    """Generate valid PSD structure tensors.

    Args:
        max_value: Maximum value for Jxx, Jyy.
        allow_zero_matrix: If True, allows zero/near-zero tensors.

    Returns:
        PSD StructureTensor satisfying Jxy^2 <= Jxx * Jyy.
    """
    min_val = 0.0 if allow_zero_matrix else DEFAULT_TOLERANCE
    Jxx = draw(st.floats(min_value=min_val, max_value=max_value).filter(lambda x: abs(x) > DEFAULT_TOLERANCE))
    Jyy = draw(st.floats(min_value=min_val, max_value=max_value).filter(lambda x: abs(x) > DEFAULT_TOLERANCE))
    # Cauchy-Schwarz bound: |Jxy| <= sqrt(Jxx * Jyy) guarantees det(J) >= 0
    max_Jxy = np.sqrt(Jxx * Jyy)
    Jxy = draw(st.floats(min_value=-max_Jxy, max_value=max_Jxy).filter(lambda x: abs(x) > DEFAULT_TOLERANCE))
    return StructureTensor(Jxx=Jxx, Jyy=Jyy, Jxy=Jxy)


@st.composite
def rotation_3x3(draw):
    """Draw a uniformly random SO(3) rotation matrix.

    Returns:
        rot: 3x3 rotation matrix.
    """
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    return Rotation.random(random_state=rng).as_matrix()


@st.composite
def camera_4x4(draw):
    """Draw a random 4x4 world-to-camera matrix with arbitrary translation.

    Returns:
        cam: 4x4 world-to-camera matrix with arbitrary translation.
    """
    R = draw(rotation_3x3())  # noqa: N806
    tx, ty, tz = (draw(st.floats(min_value=-100.0, max_value=100.0)) for _ in range(3))
    cam = np.eye(4)
    cam[:3, :3] = R
    cam[:3, 3] = [tx, ty, tz]
    return cam


PATCH_SIZE = 64

positive_thresholds = st.floats(min_value=1e-8, max_value=10.0)

STEP_EDGE_IMAGE = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
STEP_EDGE_IMAGE[:, : PATCH_SIZE // 2] = 1.0


def make_rgb_patch(size: int, pattern: str) -> np.ndarray:
    """Generate a synthetic RGB uint8 patch.

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


VERTICAL_EDGE_PATCH = make_rgb_patch(PATCH_SIZE, "vertical_edge")
HORIZONTAL_EDGE_PATCH = make_rgb_patch(PATCH_SIZE, "horizontal_edge")


@st.composite
def edge_patch(draw, patterns=None):
    """Generate a canonical-pattern RGB patch at the fixed PATCH_SIZE."""
    if patterns is None:
        patterns = ["uniform", "vertical_edge", "horizontal_edge", "diagonal_edge"]
    pattern = draw(st.sampled_from(patterns))
    return make_rgb_patch(PATCH_SIZE, pattern)


@st.composite
def flat_depth_image(draw):
    """Generate a constant-depth patch with a random depth value.

    Returns:
        Float32 array of shape (PATCH_SIZE, PATCH_SIZE) filled with a constant depth.
    """
    depth = draw(st.floats(min_value=0.01, max_value=100.0))
    return np.full((PATCH_SIZE, PATCH_SIZE), depth, dtype=np.float32)


@st.composite
def center_weight_inputs(draw):
    """Generate (shape, Ix, Iy, config) for _compute_center_weights.

    Uses a uniform gradient array to keep generation fast while covering
    all structural properties.

    Returns:
        Tuple of ((h, w), Ix, Iy, config).
    """
    h, w = PATCH_SIZE, PATCH_SIZE
    g = draw(a_scalar)
    Ix = np.full((h, w), g, dtype=np.float32)
    Iy = np.full((h, w), g, dtype=np.float32)
    radius = draw(st.floats(min_value=1.0, max_value=20.0))
    sigma_r = draw(st.floats(min_value=0.5, max_value=10.0))
    config = EdgeDetectionConfig(radius=radius, sigma_r=sigma_r)
    return (h, w), Ix, Iy, config


@st.composite
def center_check_inputs(draw):
    """Generate valid inputs for _passes_center_check.

    Uses center_weight_inputs to get realistic (weights, total_weight) pairs
    with total_weight > 0. Filters the rare zero-weight case.

    Returns:
        Tuple of (weights, total_weight, gradient_theta, max_center_offset).
    """
    shape, Ix, Iy, config = draw(center_weight_inputs())
    weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
    assume(total_weight > 0)
    gradient_theta = draw(angles)
    max_center_offset = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=50))
    )
    return weights, total_weight, gradient_theta, max_center_offset


class GradientToTangentAngleTest(unittest.TestCase):
    @given(gradient_angle=angles)
    def test_result_in_range(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        assert 0.0 <= result < 2 * np.pi

    @given(gradient_angle=angles)
    def test_perpendicularity(self, gradient_angle):
        result = gradient_to_tangent_angle(gradient_angle)
        remainder = (result - gradient_angle) % np.pi
        np.testing.assert_allclose(remainder, np.pi / 2, atol=DEFAULT_TOLERANCE)


class IsGeometricEdgeTest(unittest.TestCase):
    @given(patch=flat_depth_image(), theta=angles, threshold=positive_thresholds)
    @example(
        patch=np.full((PATCH_SIZE, PATCH_SIZE), 1.0, dtype=np.float32),
        theta=0.0,
        threshold=0.01,
    )
    def test_flat_depth_returns_false(self, patch, theta, threshold):
        self.assertFalse(is_geometric_edge(patch, theta, threshold))

    @given(theta=angles)
    @example(theta=np.pi / 2)
    @example(theta=0.0)
    def test_theta_has_period_pi(self, theta):
        result_base = is_geometric_edge(STEP_EDGE_IMAGE, theta)
        result_shifted = is_geometric_edge(STEP_EDGE_IMAGE, theta + np.pi)
        self.assertEqual(result_base, result_shifted)

    @given(theta=angles, t_low=positive_thresholds, t_high=positive_thresholds)
    @example(theta=np.pi / 2, t_low=1e-6, t_high=1.0)
    def test_lower_threshold_preserves_detection(self, theta, t_low, t_high):
        assume(t_low < t_high)
        if is_geometric_edge(STEP_EDGE_IMAGE, theta, t_high):
            self.assertTrue(is_geometric_edge(STEP_EDGE_IMAGE, theta, t_low))


class EdgeAngleTo2dPoseTest(unittest.TestCase):
    def test_identity_camera_theta_zero(self):
        """Canonical reference: identity camera, theta=0 aligns with world x-axis."""
        pose = edge_angle_to_2d_pose(theta=0.0, world_camera=np.eye(4))
        np.testing.assert_allclose(pose[0], [0, 0, 1])
        np.testing.assert_allclose(pose[1], [1, 0, 0])
        np.testing.assert_allclose(pose[2], [0, 1, 0])

    def test_tilted_camera_90_yaw(self):
        """Camera yawed 90 degrees CCW shifts world_theta by pi/2."""
        # R = Rz(pi/2) so R.T @ [1,0,0] = [0, 1, 0], ref_angle = pi/2.
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)  # noqa: N806
        cam = np.eye(4)
        cam[:3, :3] = R
        pose = edge_angle_to_2d_pose(theta=0.0, world_camera=cam)
        np.testing.assert_allclose(pose[1], [0, 1, 0], atol=DEFAULT_TOLERANCE)
        np.testing.assert_allclose(pose[2], [-1, 0, 0], atol=DEFAULT_TOLERANCE)

    @given(theta=angles, cam=camera_4x4())
    @example(theta=0.0, cam=np.eye(4))
    def test_normal_is_001(self, theta, cam):
        """Row 0 is always the world z-axis, regardless of theta or camera."""
        pose = edge_angle_to_2d_pose(theta, cam)
        np.testing.assert_allclose(pose[0], [0.0, 0.0, 1.0])

    @given(theta=angles, cam=camera_4x4())
    def test_tangent_and_perp_lie_in_xy_plane(self, theta, cam):
        """Rows 1 and 2 always have zero z-component."""
        pose = edge_angle_to_2d_pose(theta, cam)
        np.testing.assert_allclose(pose[1][2], 0.0, atol=DEFAULT_TOLERANCE)
        np.testing.assert_allclose(pose[2][2], 0.0, atol=DEFAULT_TOLERANCE)

    @given(theta=angles, cam=camera_4x4())
    @example(theta=0.0, cam=np.eye(4))
    def test_orthonormality(self, theta, cam):
        """Result is always an orthonormal frame (unit rows, mutually orthogonal)."""
        pose = edge_angle_to_2d_pose(theta, cam)
        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(pose[i]), 1.0, atol=DEFAULT_TOLERANCE)
        for i in range(3):
            for j in range(i + 1, 3):
                np.testing.assert_allclose(
                    np.dot(pose[i], pose[j]), 0.0, atol=DEFAULT_TOLERANCE
                )

    @given(theta=angles, R=rotation_3x3())
    def test_translation_invariance(self, theta, R):  # noqa: N803
        """Translation component of the camera matrix does not affect the result."""
        cam_no_t = np.eye(4)
        cam_no_t[:3, :3] = R
        cam_with_t = cam_no_t.copy()
        cam_with_t[:3, 3] = [10.0, 20.0, 30.0]
        np.testing.assert_allclose(
            edge_angle_to_2d_pose(theta, cam_with_t),
            edge_angle_to_2d_pose(theta, cam_no_t),
        )

    @given(theta=angles, cam=camera_4x4())
    def test_theta_periodicity_2pi(self, theta, cam):
        """Shifting theta by 2*pi returns the identical pose."""
        tol = max(DEFAULT_TOLERANCE * abs(theta), DEFAULT_TOLERANCE)
        np.testing.assert_allclose(
            edge_angle_to_2d_pose(theta, cam),
            edge_angle_to_2d_pose(theta + 2 * np.pi, cam),
            atol=tol,
        )

    @given(theta=angles, cam=camera_4x4())
    def test_theta_shift_pi_negates_tangent_and_perp(self, theta, cam):
        """Shifting theta by pi negates the tangent and perp rows (normal unchanged)."""
        tol = max(DEFAULT_TOLERANCE * abs(theta), DEFAULT_TOLERANCE)
        pose = edge_angle_to_2d_pose(theta, cam)
        pose_shifted = edge_angle_to_2d_pose(theta + np.pi, cam)
        np.testing.assert_allclose(pose[0], pose_shifted[0], atol=tol)
        np.testing.assert_allclose(pose[1], -pose_shifted[1], atol=tol)
        np.testing.assert_allclose(pose[2], -pose_shifted[2], atol=tol)


class StructureTensorTest(unittest.TestCase):
    def test_eigenvalues_match_analytical(self):
        t = StructureTensor(Jxx=3.0, Jyy=1.0, Jxy=1.0)
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(lambda_min, 2.0 - np.sqrt(2.0), atol=DEFAULT_TOLERANCE)
        np.testing.assert_allclose(lambda_max, 2.0 + np.sqrt(2.0), atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=0.0, Jyy=0.0, Jxy=0.0))
    def test_eigenvalues_ordered(self, t):
        lambda_min, lambda_max = t.eigenvalues
        assert lambda_min <= lambda_max

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=0.0, Jyy=9.0, Jxy=0.0))
    def test_edge_strength_nonnegative(self, t):
        assert t.edge_strength >= 0.0

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0))
    def test_coherence_in_unit_interval(self, t):
        assert 0.0 <= t.coherence <= 1.0

    @given(t=structure_tensors())
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0))
    def test_edge_orientation_range(self, t):
        assert 0.0 <= t.edge_orientation <= np.pi

    @given(t=structure_tensors())
    def test_eigenvalue_trace_equals_jxx_plus_jyy(self, t):
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(lambda_min + lambda_max, t.Jxx + t.Jyy, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors())
    def test_eigenvalue_product_equals_determinant(self, t):
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(lambda_min * lambda_max, t.Jxx * t.Jyy - t.Jxy**2, atol=DEFAULT_TOLERANCE)

    @given(k=a_scalar)
    def test_isotropic_coherence_is_zero(self, k):
        t = StructureTensor(Jxx=k, Jyy=k, Jxy=0.0)
        np.testing.assert_allclose(t.coherence, 0.0, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors(), k=a_scalar)
    def test_scaling_multiplies_edge_strength(self, t, k):
        scaled = StructureTensor(Jxx=k * t.Jxx, Jyy=k * t.Jyy, Jxy=k * t.Jxy)
        np.testing.assert_allclose(scaled.edge_strength, np.sqrt(k) * t.edge_strength, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors(), k=a_scalar)
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0), k=2.0)
    @example(t=StructureTensor(Jxx=0.0, Jyy=9.0, Jxy=0.0), k=3.0)
    def test_scaling_preserves_gradient_theta(self, t, k):
        scaled = StructureTensor(Jxx=k * t.Jxx, Jyy=k * t.Jyy, Jxy=k * t.Jxy)
        np.testing.assert_allclose(scaled.gradient_theta, t.gradient_theta, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors())
    def test_edge_strength_equals_sqrt_lambda_max(self, t):
        _, lambda_max = t.eigenvalues
        np.testing.assert_allclose(t.edge_strength, np.sqrt(max(lambda_max, 0.0)), atol=1e-10)


class TestComputeEdgeFeatures:
    def test_uniform_patch_returns_zero_strength(self):
        patch = make_rgb_patch(PATCH_SIZE, "uniform")
        strength, coherence, orientation = compute_edge_features(patch)
        assert strength == pytest.approx(0.0)
        assert coherence == pytest.approx(0.0)
        assert orientation is None

    def test_vertical_edge_detected(self):
        strength, coherence, _ = compute_edge_features(VERTICAL_EDGE_PATCH)
        assert strength > 0.0
        assert coherence > 0.0

    def test_vertical_edge_orientation(self):
        _, _, theta = compute_edge_features(VERTICAL_EDGE_PATCH)
        # Vertical edge tangent should be near pi/2 or 3*pi/2
        angle_to_vertical = min(abs(theta - np.pi / 2), abs(theta - 3 * np.pi / 2))
        assert angle_to_vertical < 0.3

    def test_horizontal_edge_orientation(self):
        _, _, theta = compute_edge_features(HORIZONTAL_EDGE_PATCH)
        # Horizontal edge tangent should be near 0 or pi
        angle_to_horizontal = min(abs(theta), abs(theta - np.pi))
        assert angle_to_horizontal < 0.3

    def test_diagonal_edge_detected(self):
        patch = make_rgb_patch(PATCH_SIZE, "diagonal_edge")
        strength, coherence, orientation = compute_edge_features(patch)
        assert strength > 0.0
        assert coherence > 0.0
        assert orientation is not None

    def test_default_params_used_when_none(self):
        compute_edge_features(VERTICAL_EDGE_PATCH, edge_detection_config=None)

    def test_center_offset_rejects_off_center_edge(self):
        # Edge at right boundary, not at center
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        patch[:, PATCH_SIZE - 4 :] = 255
        config = EdgeDetectionConfig(max_center_offset=1)
        strength, coherence, theta = compute_edge_features(patch, config)
        assert strength == pytest.approx(0.0)
        assert coherence == pytest.approx(0.0)
        assert theta is None

    @given(patch=edge_patch())
    def test_output_ranges_valid(self, patch):
        strength, coherence, orientation = compute_edge_features(patch)
        assert strength >= 0.0
        assert 0.0 <= coherence <= 1.0
        assert orientation is None or 0.0 <= orientation <= np.pi

    @given(patch=edge_patch())
    def test_zero_strength_implies_no_edge_fields(self, patch):
        strength, coherence, orientation = compute_edge_features(patch)
        if strength == 0.0:
            assert coherence == 0.0
            assert orientation is None


class ComputeCenterWeightsTest(unittest.TestCase):
    def test_zero_gradients_give_zero_weight(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.zeros((h, w), dtype=np.float32)
        Iy = np.zeros((h, w), dtype=np.float32)
        weights, total_weight = _compute_center_weights(
            (h, w), Ix, Iy, EdgeDetectionConfig()
        )
        assert total_weight == 0.0
        assert np.all(weights == 0.0)

    def test_center_pixel_has_maximum_weight(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.ones((h, w), dtype=np.float32)
        Iy = np.ones((h, w), dtype=np.float32)
        config = EdgeDetectionConfig(radius=1000.0)
        weights, _ = _compute_center_weights((h, w), Ix, Iy, config)
        r0, c0 = h // 2, w // 2
        assert weights[r0, c0] == np.max(weights)

    def test_pixel_just_outside_radius_is_zero(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.ones((h, w), dtype=np.float32)
        Iy = np.ones((h, w), dtype=np.float32)
        config = EdgeDetectionConfig(radius=2.0)
        weights, _ = _compute_center_weights((h, w), Ix, Iy, config)
        r0, c0 = h // 2, w // 2
        assert weights[r0 + 3, c0] == 0.0

    @given(inputs=center_weight_inputs())
    def test_weights_nonnegative(self, inputs):
        shape, Ix, Iy, config = inputs
        weights, _ = _compute_center_weights(shape, Ix, Iy, config)
        assert np.all(weights >= 0.0)

    @given(inputs=center_weight_inputs())
    def test_total_weight_equals_sum_of_weights(self, inputs):
        shape, Ix, Iy, config = inputs
        weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
        np.testing.assert_allclose(total_weight, np.sum(weights))

    @given(inputs=center_weight_inputs())
    def test_output_shape_matches_input(self, inputs):
        shape, Ix, Iy, config = inputs
        weights, _ = _compute_center_weights(shape, Ix, Iy, config)
        assert weights.shape == shape

    @given(inputs=center_weight_inputs())
    def test_pixels_beyond_radius_have_zero_weight(self, inputs):
        shape, Ix, Iy, config = inputs
        h, w = shape
        r0, c0 = h // 2, w // 2
        weights, _ = _compute_center_weights(shape, Ix, Iy, config)
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d = np.sqrt((rows - r0) ** 2 + (cols - c0) ** 2)
        assert np.all(weights[d > config.radius] == 0.0)

    @given(inputs=center_weight_inputs(), k=a_scalar)
    def test_gradient_scaling_scales_weights_quadratically(self, inputs, k):
        shape, Ix, Iy, config = inputs
        weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
        weights_scaled, total_weight_scaled = _compute_center_weights(
            shape, k * Ix, k * Iy, config
        )
        tol = max(DEFAULT_TOLERANCE * k**2, DEFAULT_TOLERANCE)
        np.testing.assert_allclose(weights_scaled, k**2 * weights, rtol=DEFAULT_TOLERANCE, atol=tol)
        np.testing.assert_allclose(total_weight_scaled, k**2 * total_weight, rtol=DEFAULT_TOLERANCE, atol=tol)


class PassesCenterCheckTest(unittest.TestCase):
    def test_none_offset_always_passes(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        weights = np.ones((h, w), dtype=np.float32)
        total_weight = float(weights.sum())
        assert _passes_center_check(weights, total_weight, 0.0, None)

    def test_centered_weights_pass_any_offset(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        weights = np.ones((h, w), dtype=np.float32)
        total_weight = float(weights.sum())
        assert _passes_center_check(weights, total_weight, np.pi / 4, 1)

    def test_weight_at_top_fails_tight_offset(self):
        # All weight at (row=0, col=c0). With theta=pi/2, ny=1, nx=0:
        # dist_normal[0, c0] = ny*(0 - r0) = -r0 = -32, so abs(d_center) = 32 > 1.
        h, w = PATCH_SIZE, PATCH_SIZE
        r0, c0 = h // 2, w // 2
        weights = np.zeros((h, w), dtype=np.float32)
        weights[0, c0] = 1.0
        assert not _passes_center_check(weights, 1.0, np.pi / 2, 1)

    def test_weight_at_right_fails_along_x_axis(self):
        # All weight at (row=r0, col=w-1). With theta=0, nx=1, ny=0:
        # dist_normal[r0, w-1] = nx*(w-1-c0) = w//2 - 1 = 31, so abs(d_center) = 31 > 1.
        h, w = PATCH_SIZE, PATCH_SIZE
        r0, c0 = h // 2, w // 2
        weights = np.zeros((h, w), dtype=np.float32)
        weights[r0, w - 1] = 1.0
        assert not _passes_center_check(weights, 1.0, 0.0, 1)

    @given(inputs=center_check_inputs())
    def test_none_offset_always_true(self, inputs):
        weights, total_weight, gradient_theta, _ = inputs
        assert _passes_center_check(weights, total_weight, gradient_theta, None)

    @given(inputs=center_weight_inputs(), theta=angles, offset=st.integers(min_value=0, max_value=100))
    def test_symmetric_weights_pass_any_nonneg_offset(self, inputs, theta, offset):
        # center_weight_inputs produces radially symmetric weights (radial Gaussian * uniform
        # gradient magnitude), so sum(weights*(cols-c0)) = 0 and sum(weights*(rows-r0)) = 0,
        # giving d_center = 0 for any theta.
        shape, Ix, Iy, config = inputs
        weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
        assume(total_weight > 0)
        assert _passes_center_check(weights, total_weight, theta, offset)

    @given(inputs=center_check_inputs(), delta=st.integers(min_value=0, max_value=50))
    def test_offset_monotonicity(self, inputs, delta):
        weights, total_weight, gradient_theta, max_center_offset = inputs
        assume(max_center_offset is not None)
        if _passes_center_check(weights, total_weight, gradient_theta, max_center_offset):
            assert _passes_center_check(
                weights, total_weight, gradient_theta, max_center_offset + delta
            )
