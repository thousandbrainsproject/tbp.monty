# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.sensor_processing import (
    FLAT_THRESHOLD,
    arc_from_projection,
    bilinear_sample,
    directional_curvature,
    local_ternary_pattern_and_hist,
    ltp_codes,
    ror_encoding,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
)
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    nonzero_orthogonal_vectors,
)

# Max tangent-plane displacement per step (meters).
MAX_PROJ = 0.05

# Curvature is reciprocal of the radius, thus 1e3 corresponds
# to 1 mm radius (sharp edge)
MIN_K = -1e3
MAX_K = 1e3

projections = st.floats(min_value=-MAX_PROJ, max_value=MAX_PROJ) | st.just(0.0)
curvatures = st.floats(min_value=-MAX_K, max_value=MAX_K) | st.just(0.0)


@st.composite
def regime_params(draw, min_kp, max_kp):
    """Generate (tangent_projection, curvature) targeting a specific |k*p| regime.

    Draws a product kp in [min_kp, max_kp], then factors it into curvature k
    and projection p = kp/k. A random sign is applied to the projection.

    Returns:
        Tuple of (tangent_projection, curvature).
    """
    kp = draw(st.floats(min_value=min_kp, max_value=max_kp))
    min_k = max(kp / MAX_PROJ, 0.01)
    k = draw(st.floats(min_value=min_k, max_value=MAX_K))
    p = kp / k
    sign = draw(st.sampled_from([-1, 1]))
    return sign * p, k


flat_params = regime_params(min_kp=0, max_kp=FLAT_THRESHOLD - DEFAULT_TOLERANCE)
out_of_bound_params = regime_params(min_kp=1.0 + DEFAULT_TOLERANCE, max_kp=2.0)


@st.composite
def orthonormal_vectors(draw):
    v, n = draw(nonzero_orthogonal_vectors())
    return normalize(v), n


@st.composite
def curvature_values(draw):
    k1 = draw(st.floats(min_value=MIN_K, max_value=MAX_K))
    k2 = draw(st.floats(min_value=MIN_K, max_value=MAX_K))
    assume(k1 >= k2)
    return k1, k2


class ComputeArcFromTangentProjectionTest(unittest.TestCase):
    def test_known_correction(self):
        # k=1, p=0.5 => arcsin(0.5)/1 = pi/6 (~0.52)
        result = arc_from_projection(0.5, curvature=1.0)
        npt.assert_allclose(result, np.pi / 6)

    def test_out_of_bounds_params_edge_case(self):
        # kp = 1.0 exactly: guard fires, returns projection unchanged
        assert arc_from_projection(1.0, curvature=1.0) == 1.0

    @given(tangent_projection=projections, curvature=curvatures)
    def test_corrected_length_geq_projection(self, tangent_projection, curvature):
        result = arc_from_projection(tangent_projection, curvature)
        assert abs(result) >= abs(tangent_projection)

    @given(tangent_projection=projections, curvature=curvatures)
    @example(tangent_projection=0.0, curvature=2.0)
    def test_sign_preservation(self, tangent_projection, curvature):
        result = arc_from_projection(tangent_projection, curvature)
        if tangent_projection > 0.0:
            assert result > 0.0
        elif tangent_projection < 0:
            assert result < 0.0
        else:
            assert result == 0.0

    @given(tangent_projection=projections, curvature=curvatures)
    def test_negating_projection_negates_result(self, tangent_projection, curvature):
        pos = arc_from_projection(tangent_projection, curvature)
        neg = arc_from_projection(-tangent_projection, curvature)
        assert neg == -1.0 * pos

    @given(tangent_projection=projections, curvature=curvatures)
    def test_curvature_sign_does_not_affect_result(self, tangent_projection, curvature):
        pos_k = arc_from_projection(tangent_projection, curvature)
        neg_k = arc_from_projection(tangent_projection, -curvature)
        assert pos_k == neg_k

    @given(params=flat_params)
    def test_flat_bypass_returns_projection(self, params):
        tangent_projection, curvature = params
        result = arc_from_projection(tangent_projection, curvature)
        assert result == tangent_projection

    @given(params=out_of_bound_params)
    def test_out_of_bounds_returns_projection(self, params):
        tangent_projection, curvature = params
        result = arc_from_projection(tangent_projection, curvature)
        assert result == tangent_projection


class DirectionalCurvatureTest(unittest.TestCase):
    @given(vectors=orthonormal_vectors(), ks=curvature_values())
    def test_zero_direction_returns_zero(self, vectors, ks):
        pc1, pc2 = vectors
        k1, k2 = ks
        result = directional_curvature(
            np.array([0.0, 0.0, 0.0]),
            k1=k1,
            k2=k2,
            pc1_dir=pc1,
            pc2_dir=pc2,
        )
        npt.assert_allclose(result, 0.0, atol=DEFAULT_TOLERANCE)

    @given(
        angle=st.floats(min_value=0, max_value=2 * np.pi),
        ks=curvature_values(),
        vectors=orthonormal_vectors(),
    )
    def test_euler_formula(self, angle, ks, vectors):
        pc1, pc2 = vectors
        k1, k2 = ks
        # Create a vector in the same plane as pc1 and pc2.
        direction = pc1 * np.cos(angle) + pc2 * np.sin(angle)
        result = directional_curvature(
            direction, k1=k1, k2=k2, pc1_dir=pc1, pc2_dir=pc2
        )
        expected = k1 * np.cos(angle) ** 2 + k2 * np.sin(angle) ** 2
        tol = max(
            DEFAULT_TOLERANCE * abs(k1),
            DEFAULT_TOLERANCE * abs(k2),
            DEFAULT_TOLERANCE,
        )
        npt.assert_allclose(result, expected, atol=tol, rtol=DEFAULT_TOLERANCE)

    @given(vectors=orthonormal_vectors())
    def test_out_of_plane_movement_raises(self, vectors):
        pc1, pc2 = vectors
        movement_direction = np.cross(pc1, pc2)
        with pytest.raises(ValueError, match="must lie in the plane"):
            directional_curvature(
                movement_direction=movement_direction,
                k1=Mock(),
                k2=Mock(),
                pc1_dir=pc1,
                pc2_dir=pc2,
            )

    @given(vectors=orthonormal_vectors())
    def test_pcs_not_unit_vectors_raises(self, vectors):
        pc1, pc2 = vectors
        scaled_pc1 = pc1 * 2.0
        with pytest.raises(ValueError, match="must be unit vectors"):
            directional_curvature(
                movement_direction=Mock(),
                k1=Mock(),
                k2=Mock(),
                pc1_dir=scaled_pc1,
                pc2_dir=pc2,
            )

        scaled_pc2 = pc2 * 2.0
        with pytest.raises(ValueError, match="must be unit vectors"):
            directional_curvature(
                movement_direction=Mock(),
                k1=Mock(),
                k2=Mock(),
                pc1_dir=pc1,
                pc2_dir=scaled_pc2,
            )


def ltp_codes_loop(
    gray_patch: np.ndarray,
    n_neighbors: int = 8,
    radius: float = 1.0,
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit, non-vectorized reference for ``ltp_codes``.

    Loops over every pixel and every neighbor and accumulates the ternary bits
    one element at a time. This exercises the vectorized bit-packing logic of
    ``ltp_codes`` against a plain Python equivalent.

    Note:
        Neighbor values are obtained from the same ``bilinear_sample`` primitive
        used by the implementation (sampling one coordinate at a time), so the
        reference is independent of which interpolation backend ``bilinear_sample``
        happens to use. This matters at ``threshold == 0``: there the ``>=``/``<=``
        comparisons become exact-equality ties whenever a neighbor coincides with
        the center (e.g. border pixels whose offset is clipped back onto the
        center row/column), and sub-epsilon differences between interpolation
        backends would otherwise flip individual bits. The coordinate is built
        exactly as the implementation does -- a ``float32`` grid value plus the
        ``float32``-cast neighbor offset -- so the sampled value matches bit for
        bit.

    Args:
        gray_patch: Grayscale image patch.
        n_neighbors: Number of neighbors in the circular neighborhood.
        radius: Radius of the neighborhood in pixels.
        threshold: Threshold for the local ternary pattern.

    Returns:
        Tuple of ``(codes_pos, codes_neg)`` code maps.
    """
    h, w = gray_patch.shape
    codes_pos = np.zeros((h, w), dtype=np.uint32)
    codes_neg = np.zeros((h, w), dtype=np.uint32)

    for y in range(h):
        for x in range(w):
            center = gray_patch[y, x]
            for i in range(n_neighbors):
                theta = 2.0 * np.pi * i / n_neighbors
                ny = np.float32(y) + np.float32(-radius * np.sin(theta))
                nx = np.float32(x) + np.float32(radius * np.cos(theta))

                neighbor = bilinear_sample(
                    gray_patch,
                    np.array([ny], dtype=np.float32),
                    np.array([nx], dtype=np.float32),
                )[0]

                if neighbor >= center + threshold:
                    codes_pos[y, x] |= np.uint32(1) << i
                if neighbor <= center - threshold:
                    codes_neg[y, x] |= np.uint32(1) << i

    return codes_pos, codes_neg


class BilinearSampleTest(unittest.TestCase):
    def test_integer_coordinates_return_exact_pixels(self):
        image = np.arange(12, dtype=np.float64).reshape(3, 4)
        y, x = np.meshgrid(np.arange(3.0), np.arange(4.0), indexing="ij")
        sampled = bilinear_sample(image, y.ravel(), x.ravel())
        npt.assert_allclose(sampled, image.ravel())

    def test_midpoint_between_two_pixels_is_their_average(self):
        image = np.array([[0.0, 10.0], [20.0, 30.0]])
        # Halfway along the top row -> mean(0, 10) = 5.
        npt.assert_allclose(
            bilinear_sample(image, np.array([0.0]), np.array([0.5])), [5.0]
        )
        # Halfway down the left column -> mean(0, 20) = 10.
        npt.assert_allclose(
            bilinear_sample(image, np.array([0.5]), np.array([0.0])), [10.0]
        )

    def test_center_of_2x2_is_mean_of_four_corners(self):
        image = np.array([[0.0, 10.0], [20.0, 30.0]])
        npt.assert_allclose(
            bilinear_sample(image, np.array([0.5]), np.array([0.5])), [15.0]
        )

    def test_known_weighted_interpolation(self):
        # Sample (y=0.25, x=0.75) on a 2x2 patch.
        #   top    = 0 * 0.25 + 10 * 0.75 = 7.5
        #   bottom = 20 * 0.25 + 30 * 0.75 = 27.5
        #   value  = 7.5 * 0.75 + 27.5 * 0.25 = 12.5
        image = np.array([[0.0, 10.0], [20.0, 30.0]])
        npt.assert_allclose(
            bilinear_sample(image, np.array([0.25]), np.array([0.75])), [12.5]
        )

    def test_out_of_bounds_clamps_to_nearest_edge(self):
        image = np.array([[0.0, 10.0], [20.0, 30.0]])
        # Far past every edge -> the respective corner pixel.
        npt.assert_allclose(
            bilinear_sample(image, np.array([-5.0]), np.array([-5.0])), [0.0]
        )
        npt.assert_allclose(
            bilinear_sample(image, np.array([99.0]), np.array([99.0])), [30.0]
        )
        # Out of bounds on one axis only still interpolates the other axis.
        npt.assert_allclose(
            bilinear_sample(image, np.array([-1.0]), np.array([0.5])), [5.0]
        )

    def test_preserves_input_shape(self):
        image = np.arange(25, dtype=np.float64).reshape(5, 5)
        y = np.full((3, 2), 1.5)
        x = np.full((3, 2), 2.5)
        assert bilinear_sample(image, y, x).shape == (3, 2)


class LocalTernaryPatternTest(unittest.TestCase):
    def test_invalid_n_neighbors_raises(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            ltp_codes(np.zeros((3, 3)), n_neighbors=0)

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError, match="radius"):
            ltp_codes(np.zeros((3, 3)), radius=0.0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            ltp_codes(np.zeros((3, 3)), threshold=-1.0)

    def test_non_2d_patch_raises(self):
        with pytest.raises(ValueError, match="2D"):
            ltp_codes(np.zeros((3, 3, 3)))

    def test_output_shape_and_dtype(self):
        patch = np.zeros((4, 5))
        codes_pos, codes_neg = ltp_codes(patch, n_neighbors=8)
        assert codes_pos.shape == (4, 5)
        assert codes_neg.shape == (4, 5)
        assert codes_pos.dtype == np.uint32
        assert codes_neg.dtype == np.uint32

    def test_known_cross_pattern_center(self):
        # Center (30) is darker than all four axis neighbors (50), so with
        # n_neighbors=4 every positive bit is set: 0b1111 = 15, and no
        # negative bit is set.
        patch = np.array(
            [[10.0, 50.0, 10.0], [50.0, 30.0, 50.0], [10.0, 50.0, 10.0]]
        )
        codes_pos, codes_neg = ltp_codes(
            patch, n_neighbors=4, radius=1.0, threshold=0.0
        )
        assert codes_pos[1, 1] == 0b1111
        assert codes_neg[1, 1] == 0

    def test_known_single_direction_bright(self):
        # Only the right neighbor (bit 0 for n_neighbors=4) is brighter.
        patch = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 100.0], [0.0, 0.0, 0.0]]
        )
        codes_pos, codes_neg = ltp_codes(
            patch, n_neighbors=4, radius=1.0, threshold=5.0
        )
        assert codes_pos[1, 1] == 0b0001
        assert codes_neg[1, 1] == 0

    def test_known_single_direction_dark(self):
        # Center bright, only the right neighbor is darker -> negative bit 0.
        patch = np.array(
            [[100.0, 100.0, 100.0], [100.0, 100.0, 0.0], [100.0, 100.0, 100.0]]
        )
        codes_pos, codes_neg = ltp_codes(
            patch, n_neighbors=4, radius=1.0, threshold=5.0
        )
        assert codes_pos[1, 1] == 0
        assert codes_neg[1, 1] == 0b0001

    def test_center_darker_than_all_neighbors_sets_every_positive_bit(self):
        patch = np.full((3, 3), 100.0)
        patch[1, 1] = 0.0
        codes_pos, codes_neg = ltp_codes(
            patch, n_neighbors=8, radius=1.0, threshold=5.0
        )
        # All eight positive bits set, no negative bits.
        assert codes_pos[1, 1] == 0b11111111
        assert codes_neg[1, 1] == 0

    def test_center_brighter_than_all_neighbors_sets_every_negative_bit(self):
        patch = np.zeros((3, 3))
        patch[1, 1] = 100.0
        codes_pos, codes_neg = ltp_codes(
            patch, n_neighbors=8, radius=1.0, threshold=5.0
        )
        assert codes_pos[1, 1] == 0
        assert codes_neg[1, 1] == 0b11111111

    def test_flat_patch_with_threshold_sets_no_bits(self):
        # No neighbor differs from the center by more than the threshold.
        patch = np.full((5, 5), 42.0)
        codes_pos, codes_neg = ltp_codes(
            patch, n_neighbors=8, radius=1.0, threshold=1.0
        )
        npt.assert_array_equal(codes_pos, np.zeros((5, 5), dtype=np.uint32))
        npt.assert_array_equal(codes_neg, np.zeros((5, 5), dtype=np.uint32))

    def test_matches_non_vectorized_loop(self):
        rng = np.random.default_rng(7)
        configs = [
            dict(n_neighbors=8, radius=1.0, threshold=5.0),
            dict(n_neighbors=4, radius=1.0, threshold=0.0),
            dict(n_neighbors=8, radius=2.0, threshold=3.0),
            dict(n_neighbors=12, radius=1.5, threshold=10.0),
            dict(n_neighbors=16, radius=2.5, threshold=0.0),
        ]
        shapes = [(6, 7), (5, 5), (8, 4), (3, 9)]
        for config in configs:
            for shape in shapes:
                with self.subTest(config=config, shape=shape):
                    patch = rng.uniform(0.0, 255.0, size=shape)
                    pos, neg = ltp_codes(patch, **config)
                    ref_pos, ref_neg = ltp_codes_loop(patch, **config)
                    npt.assert_array_equal(pos, ref_pos)
                    npt.assert_array_equal(neg, ref_neg)


def euler_totient(n: int) -> int:
    """Count integers in ``[1, n]`` that are coprime with ``n``.

    Args:
        n: Positive integer to evaluate.

    Returns:
        The value of Euler's totient ``phi(n)``.
    """
    result, d = n, 2
    n_remaining = n
    while d * d <= n_remaining:
        if n_remaining % d == 0:
            while n_remaining % d == 0:
                n_remaining //= d
            result -= result // d
        d += 1
    if n_remaining > 1:
        result -= result // n_remaining
    return result


def binary_necklace_count(n_bits: int) -> int:
    """Number of distinct binary necklaces of length ``n_bits``.

    This is the count of equivalence classes of ``n_bits``-bit codes under
    cyclic rotation, i.e. the number of rotation-invariant (ROR) bins. Given by
    Burnside's lemma: ``(1 / n) * sum_{d | n} phi(d) * 2^(n / d)``.

    Args:
        n_bits: Length of the binary codes (the number of LTP neighbors).

    Returns:
        Number of distinct rotation classes.
    """
    total = sum(
        euler_totient(d) * (2 ** (n_bits // d))
        for d in range(1, n_bits + 1)
        if n_bits % d == 0
    )
    return total // n_bits


def rotate_right(code: int, n_bits: int) -> int:
    """Rotate the ``n_bits`` low bits of ``code`` right by one position.

    Args:
        code: Code to rotate.
        n_bits: Bit width of the code.

    Returns:
        The rotated code.
    """
    return ((code >> 1) | ((code & 1) << (n_bits - 1))) & ((1 << n_bits) - 1)


def all_rotations(code: int, n_bits: int) -> set[int]:
    """Return every cyclic rotation of ``code`` (``n_bits`` wide)."""
    rotations = set()
    current = code
    for _ in range(n_bits):
        rotations.add(current)
        current = rotate_right(current, n_bits)
    return rotations


class RorEncodingTest(unittest.TestCase):
    def test_known_mapping_for_four_neighbors(self):
        # Hand-derived canonical (minimum-rotation) classes for 4-bit codes:
        #   {0}, {1,2,4,8}, {3,6,9,12}, {5,10}, {7,11,13,14}, {15}
        # remapped to dense bins 0..5 in ascending canonical order.
        codes = np.arange(16, dtype=np.uint32).reshape(1, 16)
        encoded, n_bins = ror_encoding(codes, n_neighbors=4)
        expected = [0, 1, 1, 2, 1, 3, 2, 4, 1, 2, 3, 4, 2, 4, 4, 5]
        npt.assert_array_equal(encoded.ravel(), expected)
        assert n_bins == 6

    def test_n_bins_equals_binary_necklace_count(self):
        for n_neighbors in (1, 2, 3, 4, 5, 8):
            with self.subTest(n_neighbors=n_neighbors):
                codes = np.arange(1 << n_neighbors, dtype=np.uint32)
                _, n_bins = ror_encoding(codes, n_neighbors=n_neighbors)
                assert n_bins == binary_necklace_count(n_neighbors)

    def test_rotations_share_a_bin(self):
        n_neighbors = 8
        all_codes = np.arange(1 << n_neighbors, dtype=np.uint32)
        encoded, _ = ror_encoding(all_codes, n_neighbors=n_neighbors)
        lut = encoded.ravel()
        for code in (0b00010110, 0b00000001, 0b10110100, 0b01010101):
            with self.subTest(code=code):
                bins = {lut[r] for r in all_rotations(code, n_neighbors)}
                assert len(bins) == 1

    def test_distinct_necklaces_get_distinct_bins(self):
        # One representative from each 4-bit rotation class -> all-distinct bins.
        representatives = np.array([0, 1, 3, 5, 7, 15], dtype=np.uint32)
        encoded, n_bins = ror_encoding(representatives, n_neighbors=4)
        assert len(np.unique(encoded)) == len(representatives)
        assert n_bins == len(representatives)

    def test_preserves_shape_and_returns_int_codes(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 256, size=(5, 7)).astype(np.uint32)
        encoded, _ = ror_encoding(codes, n_neighbors=8)
        assert encoded.shape == codes.shape
        assert np.issubdtype(encoded.dtype, np.integer)

    def test_zero_and_all_ones_are_singleton_classes(self):
        n_neighbors = 8
        codes = np.array([0, (1 << n_neighbors) - 1], dtype=np.uint32)
        encoded, n_bins = ror_encoding(codes, n_neighbors=n_neighbors)
        # All-zeros is the smallest canonical -> first bin; all-ones is the
        # largest canonical -> last bin.
        assert encoded[0] == 0
        assert encoded[1] == n_bins - 1

    def test_encoded_values_within_bin_range(self):
        all_codes = np.arange(256, dtype=np.uint32)
        encoded, n_bins = ror_encoding(all_codes, n_neighbors=8)
        assert encoded.min() >= 0
        assert encoded.max() == n_bins - 1

    def test_out_of_range_code_raises_value_error(self):
        # Codes must be < 2**n_neighbors; larger values are rejected.
        codes = np.array([16], dtype=np.uint32)  # only 0..15 valid for n=4
        with pytest.raises(ValueError, match="exceeds the number of codes"):
            ror_encoding(codes, n_neighbors=4)


class LocalTernaryPatternAndHistTest(unittest.TestCase):
    def test_output_is_1d_float32(self):
        patch = np.random.default_rng(0).uniform(0.0, 255.0, size=(8, 8))
        hist = local_ternary_pattern_and_hist(patch, n_neighbors=8)
        assert hist.ndim == 1
        assert hist.dtype == np.float32

    def test_length_is_twice_the_bin_count(self):
        patch = np.random.default_rng(1).uniform(0.0, 255.0, size=(8, 8))
        for n_neighbors in (4, 8):
            with self.subTest(n_neighbors=n_neighbors):
                hist = local_ternary_pattern_and_hist(patch, n_neighbors=n_neighbors)
                assert hist.shape == (2 * binary_necklace_count(n_neighbors),)

    def test_length_is_independent_of_patch_content_and_shape(self):
        rng = np.random.default_rng(2)
        first = local_ternary_pattern_and_hist(
            rng.uniform(0.0, 255.0, size=(8, 8)), n_neighbors=8
        )
        second = local_ternary_pattern_and_hist(
            rng.uniform(0.0, 255.0, size=(13, 4)), n_neighbors=8
        )
        third = local_ternary_pattern_and_hist(
            np.zeros((5, 5)), n_neighbors=8
        )
        assert first.shape == second.shape == third.shape

    def test_histogram_is_non_negative_and_normalized(self):
        patch = np.random.default_rng(3).uniform(0.0, 255.0, size=(10, 10))
        hist = local_ternary_pattern_and_hist(patch, n_neighbors=8)
        assert np.all(hist >= 0.0)
        # NOTE: normalization divides by (sum + 1e-6), so the result sums to
        # *almost* 1 rather than exactly 1.
        npt.assert_allclose(hist.sum(), 1.0, atol=1e-3)
        assert hist.sum() <= 1.0

    def test_flat_patch_mass_collapses_into_zero_bins(self):
        # With threshold > 0 no neighbor differs from the center, so every pixel
        # produces code 0 in both the positive and negative maps. All mass lands
        # in the first bin of each half (indices 0 and n_bins).
        n_bins = binary_necklace_count(8)
        hist = local_ternary_pattern_and_hist(
            np.full((6, 6), 50.0), n_neighbors=8, threshold=1.0
        )
        npt.assert_array_equal(np.flatnonzero(hist), [0, n_bins])

    def test_invalid_arguments_propagate_value_errors(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            local_ternary_pattern_and_hist(np.zeros((3, 3)), n_neighbors=0)
        with pytest.raises(ValueError, match="radius"):
            local_ternary_pattern_and_hist(np.zeros((3, 3)), radius=0.0)
        with pytest.raises(ValueError, match="threshold"):
            local_ternary_pattern_and_hist(np.zeros((3, 3)), threshold=-1.0)
        with pytest.raises(ValueError, match="2D"):
            local_ternary_pattern_and_hist(np.zeros((3, 3, 3)))
