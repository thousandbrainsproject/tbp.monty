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

from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetectionConfig,
    StructureTensor,
    _compute_center_weights,
    _passes_center_check,
    compute_edge_features,
    gradient_to_tangent_angle,
)
from tbp.monty.math import DEFAULT_TOLERANCE

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)
a_scalar = st.floats(min_value=DEFAULT_TOLERANCE, max_value=100.0)


@st.composite
def structure_tensors(draw, max_value=100.0, allow_zero_matrix=True):
    """Generate valid PSD structure tensors.

    Args:
        draw: Hypothesis draw function (injected by @st.composite).
        max_value: Maximum value for Jxx, Jyy.
        allow_zero_matrix: If True, allows zero/near-zero tensors.

    Returns:
        PSD StructureTensor satisfying Jxy^2 <= Jxx * Jyy.
    """
    min_val = 0.0 if allow_zero_matrix else DEFAULT_TOLERANCE
    Jxx = draw(  # noqa: N806
        st.floats(min_value=min_val, max_value=max_value).filter(
            lambda x: abs(x) > DEFAULT_TOLERANCE
        )
    )
    Jyy = draw(  # noqa: N806
        st.floats(min_value=min_val, max_value=max_value).filter(
            lambda x: abs(x) > DEFAULT_TOLERANCE
        )
    )
    # Cauchy-Schwarz bound: |Jxy| <= sqrt(Jxx * Jyy) guarantees det(J) >= 0
    max_Jxy = np.sqrt(Jxx * Jyy)  # noqa: N806
    Jxy = draw(  # noqa: N806
        st.floats(min_value=-max_Jxy, max_value=max_Jxy).filter(
            lambda x: abs(x) > DEFAULT_TOLERANCE
        )
    )
    return StructureTensor(Jxx=Jxx, Jyy=Jyy, Jxy=Jxy)


PATCH_SIZE = 64


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
    """Generate a canonical-pattern RGB patch at the fixed PATCH_SIZE.

    Returns:
        An RGB patch array of shape (PATCH_SIZE, PATCH_SIZE, 3).
    """
    if patterns is None:
        patterns = ["uniform", "vertical_edge", "horizontal_edge", "diagonal_edge"]
    pattern = draw(st.sampled_from(patterns))
    return make_rgb_patch(PATCH_SIZE, pattern)


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
    Ix = np.full((h, w), g, dtype=np.float32)  # noqa: N806
    Iy = np.full((h, w), g, dtype=np.float32)  # noqa: N806
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
    shape, Ix, Iy, config = draw(center_weight_inputs())  # noqa: N806
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


class StructureTensorTest(unittest.TestCase):
    def test_eigenvalues_match_analytical(self):
        t = StructureTensor(Jxx=3.0, Jyy=1.0, Jxy=1.0)
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(
            lambda_min, 2.0 - np.sqrt(2.0), atol=DEFAULT_TOLERANCE
        )
        np.testing.assert_allclose(
            lambda_max, 2.0 + np.sqrt(2.0), atol=DEFAULT_TOLERANCE
        )

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
        np.testing.assert_allclose(
            lambda_min + lambda_max, t.Jxx + t.Jyy, atol=DEFAULT_TOLERANCE
        )

    @given(t=structure_tensors())
    def test_eigenvalue_product_equals_determinant(self, t):
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(
            lambda_min * lambda_max, t.Jxx * t.Jyy - t.Jxy**2, atol=DEFAULT_TOLERANCE
        )

    @given(k=a_scalar)
    def test_isotropic_coherence_is_zero(self, k):
        t = StructureTensor(Jxx=k, Jyy=k, Jxy=0.0)
        np.testing.assert_allclose(t.coherence, 0.0, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors(), k=a_scalar)
    def test_scaling_multiplies_edge_strength(self, t, k):
        scaled = StructureTensor(Jxx=k * t.Jxx, Jyy=k * t.Jyy, Jxy=k * t.Jxy)
        np.testing.assert_allclose(
            scaled.edge_strength, np.sqrt(k) * t.edge_strength, atol=DEFAULT_TOLERANCE
        )

    @given(t=structure_tensors(), k=a_scalar)
    @example(t=StructureTensor(Jxx=4.0, Jyy=0.0, Jxy=0.0), k=2.0)
    @example(t=StructureTensor(Jxx=0.0, Jyy=9.0, Jxy=0.0), k=3.0)
    def test_scaling_preserves_gradient_theta(self, t, k):
        scaled = StructureTensor(Jxx=k * t.Jxx, Jyy=k * t.Jyy, Jxy=k * t.Jxy)
        np.testing.assert_allclose(
            scaled.gradient_theta, t.gradient_theta, atol=DEFAULT_TOLERANCE
        )

    @given(t=structure_tensors())
    def test_edge_strength_equals_sqrt_lambda_max(self, t):
        _, lambda_max = t.eigenvalues
        np.testing.assert_allclose(
            t.edge_strength, np.sqrt(max(lambda_max, 0.0)), atol=1e-10
        )


class ComputeCenterWeightsTest(unittest.TestCase):
    def test_zero_gradients_give_zero_weight(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.zeros((h, w), dtype=np.float32)  # noqa: N806
        Iy = np.zeros((h, w), dtype=np.float32)  # noqa: N806
        weights, total_weight = _compute_center_weights(
            (h, w), Ix, Iy, EdgeDetectionConfig()
        )
        assert total_weight == 0.0
        assert np.all(weights == 0.0)

    def test_center_pixel_has_maximum_weight(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.ones((h, w), dtype=np.float32)  # noqa: N806
        Iy = np.ones((h, w), dtype=np.float32)  # noqa: N806
        config = EdgeDetectionConfig(radius=1000.0)
        weights, _ = _compute_center_weights((h, w), Ix, Iy, config)
        r0, c0 = h // 2, w // 2
        assert weights[r0, c0] == np.max(weights)

    def test_pixel_just_outside_radius_is_zero(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.ones((h, w), dtype=np.float32)  # noqa: N806
        Iy = np.ones((h, w), dtype=np.float32)  # noqa: N806
        config = EdgeDetectionConfig(radius=2.0)
        weights, _ = _compute_center_weights((h, w), Ix, Iy, config)
        r0, c0 = h // 2, w // 2
        assert weights[r0 + 3, c0] == 0.0

    @given(inputs=center_weight_inputs())
    def test_weights_nonnegative(self, inputs):
        shape, Ix, Iy, config = inputs  # noqa: N806
        weights, _ = _compute_center_weights(shape, Ix, Iy, config)
        assert np.all(weights >= 0.0)

    @given(inputs=center_weight_inputs())
    def test_total_weight_equals_sum_of_weights(self, inputs):
        shape, Ix, Iy, config = inputs  # noqa: N806
        weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
        np.testing.assert_allclose(total_weight, np.sum(weights))

    @given(inputs=center_weight_inputs())
    def test_output_shape_matches_input(self, inputs):
        shape, Ix, Iy, config = inputs  # noqa: N806
        weights, _ = _compute_center_weights(shape, Ix, Iy, config)
        assert weights.shape == shape

    @given(inputs=center_weight_inputs())
    def test_pixels_beyond_radius_have_zero_weight(self, inputs):
        shape, Ix, Iy, config = inputs  # noqa: N806
        h, w = shape
        r0, c0 = h // 2, w // 2
        weights, _ = _compute_center_weights(shape, Ix, Iy, config)
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d = np.sqrt((rows - r0) ** 2 + (cols - c0) ** 2)
        assert np.all(weights[d > config.radius] == 0.0)

    @given(inputs=center_weight_inputs(), k=a_scalar)
    def test_gradient_scaling_scales_weights_quadratically(self, inputs, k):
        shape, Ix, Iy, config = inputs  # noqa: N806
        weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
        weights_scaled, total_weight_scaled = _compute_center_weights(
            shape, k * Ix, k * Iy, config
        )
        tol = max(DEFAULT_TOLERANCE * k**2, DEFAULT_TOLERANCE)
        np.testing.assert_allclose(
            weights_scaled, k**2 * weights, rtol=DEFAULT_TOLERANCE, atol=tol
        )
        np.testing.assert_allclose(
            total_weight_scaled, k**2 * total_weight, rtol=DEFAULT_TOLERANCE, atol=tol
        )


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
        c0 = w // 2
        weights = np.zeros((h, w), dtype=np.float32)
        weights[0, c0] = 1.0
        assert not _passes_center_check(weights, 1.0, np.pi / 2, 1)

    def test_weight_at_right_fails_along_x_axis(self):
        # All weight at (row=r0, col=w-1). With theta=0, nx=1, ny=0:
        # dist_normal[r0, w-1] = nx*(w-1-c0) = w//2 - 1 = 31, so abs(d_center) = 31 > 1.
        h, w = PATCH_SIZE, PATCH_SIZE
        r0 = h // 2
        weights = np.zeros((h, w), dtype=np.float32)
        weights[r0, w - 1] = 1.0
        assert not _passes_center_check(weights, 1.0, 0.0, 1)

    @given(inputs=center_check_inputs())
    def test_none_offset_always_true(self, inputs):
        weights, total_weight, gradient_theta, _ = inputs
        assert _passes_center_check(weights, total_weight, gradient_theta, None)

    @given(
        inputs=center_weight_inputs(),
        theta=angles,
        offset=st.integers(min_value=0, max_value=100),
    )
    def test_symmetric_weights_pass_any_nonneg_offset(self, inputs, theta, offset):
        # center_weight_inputs produces radially symmetric weights (radial Gaussian
        # * uniform gradient magnitude), so sum(weights*(cols-c0)) = 0 and
        # sum(weights*(rows-r0)) = 0, giving d_center = 0 for any theta.
        shape, Ix, Iy, config = inputs  # noqa: N806
        weights, total_weight = _compute_center_weights(shape, Ix, Iy, config)
        assume(total_weight > 0)
        assert _passes_center_check(weights, total_weight, theta, offset)

    @given(inputs=center_check_inputs(), delta=st.integers(min_value=0, max_value=50))
    def test_offset_monotonicity(self, inputs, delta):
        weights, total_weight, gradient_theta, max_center_offset = inputs
        assume(max_center_offset is not None)
        if _passes_center_check(
            weights, total_weight, gradient_theta, max_center_offset
        ):
            assert _passes_center_check(
                weights, total_weight, gradient_theta, max_center_offset + delta
            )


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
