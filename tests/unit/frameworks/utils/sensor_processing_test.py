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
import numpy.testing as npt
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tbp.monty.frameworks.utils.sensor_processing import (
    FLAT_THRESHOLD,
    arc_length_corrected_displacement,
    compute_arc_from_tangent_projection,
    directional_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
)
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    non_zero_magnitude_vectors,
    nonzero_orthogonal_vectors,
)

# Max tangent-plane displacement per step (meters).
MAX_PROJ = 0.05

# abs(curvature) = 1e-3 corresponds to 1 mm (sharp edge)
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
guard_params = regime_params(min_kp=1.0 + DEFAULT_TOLERANCE, max_kp=2.0)


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
        result = compute_arc_from_tangent_projection(0.5, curvature=1.0)
        npt.assert_allclose(result, np.pi / 6)

    def test_domain_guard_at_boundary(self):
        # kp = 1.0 exactly: guard fires, returns projection unchanged
        assert compute_arc_from_tangent_projection(1.0, curvature=1.0) == 1.0

    @given(tangent_projection=projections, curvature=curvatures)
    def test_corrected_length_geq_projection(self, tangent_projection, curvature):
        result = compute_arc_from_tangent_projection(tangent_projection, curvature)
        assert abs(result) >= abs(tangent_projection)

    @given(tangent_projection=projections, curvature=curvatures)
    def test_sign_preservation(self, tangent_projection, curvature):
        result = compute_arc_from_tangent_projection(tangent_projection, curvature)
        if tangent_projection > 0.0:
            assert result > 0.0
        elif tangent_projection < 0:
            assert result < 0.0
        else:
            assert result == 0.0

    @given(tangent_projection=projections, curvature=curvatures)
    def test_negating_projection_negates_result(self, tangent_projection, curvature):
        pos = compute_arc_from_tangent_projection(tangent_projection, curvature)
        neg = compute_arc_from_tangent_projection(-tangent_projection, curvature)
        assert neg == -1.0 * pos

    @given(tangent_projection=projections, curvature=curvatures)
    def test_curvature_sign_does_not_affect_result(self, tangent_projection, curvature):
        pos_k = compute_arc_from_tangent_projection(tangent_projection, curvature)
        neg_k = compute_arc_from_tangent_projection(tangent_projection, -curvature)
        assert pos_k == neg_k

    @given(params=flat_params)
    def test_flat_bypass_returns_projection(self, params):
        tangent_projection, curvature = params
        result = compute_arc_from_tangent_projection(tangent_projection, curvature)
        assert result == tangent_projection

    @given(params=guard_params)
    def test_domain_guard_returns_projection(self, params):
        tangent_projection, curvature = params
        result = compute_arc_from_tangent_projection(tangent_projection, curvature)
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
            DEFAULT_TOLERANCE * abs(k1), DEFAULT_TOLERANCE * abs(k2), DEFAULT_TOLERANCE
        )
        npt.assert_allclose(result, expected, atol=tol, rtol=DEFAULT_TOLERANCE)

    @given(
        movement_direction=non_zero_magnitude_vectors(),
        vectors=orthonormal_vectors(),
        ks=curvature_values(),
        a_scaler=st.floats(min_value=-1e3, max_value=1e3).filter(
            lambda x: abs(x) > DEFAULT_TOLERANCE
        ),
    )
    def test_non_orthogonal_pcs_raises(self, movement_direction, vectors, ks, a_scaler):
        pc1, _ = vectors
        k1, k2 = ks
        bad_pc2 = pc1 * a_scaler
        expected_msg = r"The pc1_dir and pc2_dir must be orthogonal\."
        with pytest.raises(ValueError, match=expected_msg):
            directional_curvature(
                movement_direction=movement_direction,
                k1=k1,
                k2=k2,
                pc1_dir=pc1,
                pc2_dir=bad_pc2,
            )

    @given(vectors=orthonormal_vectors(), ks=curvature_values())
    def test_out_of_plane_movement_raises(self, vectors, ks):
        pc1, pc2 = vectors
        k1, k2 = ks
        movement_direction = np.cross(pc1, pc2)
        expected_msg = (
            r"The movement_direction must lie in the plane"
            r" of pc1_dir and pc2_dir\."
        )
        with pytest.raises(ValueError, match=expected_msg):
            directional_curvature(
                movement_direction=movement_direction,
                k1=k1,
                k2=k2,
                pc1_dir=pc1,
                pc2_dir=pc2,
            )


class ArcLengthCorrectedDisplacementTest(unittest.TestCase):
    """Unit tests for the arc_length_corrected_displacement function."""

    def setUp(self):
        self.basis_u = np.array([1.0, 0.0, 0.0])
        self.basis_v = np.array([0.0, 1.0, 0.0])
        # Principal directions aligned with basis vectors
        self.pose_vectors = np.array(
            [
                [0.0, 0.0, 1.0],  # row 0: normal (unused by function)
                [1.0, 0.0, 0.0],  # row 1: principal dir 1
                [0.0, 1.0, 0.0],  # row 2: principal dir 2
            ]
        )

    @given(
        du=st.floats(min_value=-1, max_value=1),
        dv=st.floats(min_value=-1, max_value=1),
        k1=st.floats(min_value=-100, max_value=100),
        k2=st.floats(min_value=-100, max_value=100),
    )
    def test_arc_length_corrected_displacement_properties(self, du, dv, k1, k2):
        principal_curvatures = np.array([k1, k2])
        arc_u, arc_v = arc_length_corrected_displacement(
            du,
            dv,
            self.basis_u,
            self.basis_v,
            principal_curvatures,
            self.pose_vectors,
        )

        # 1. Sign preservation
        if du > 0:
            self.assertGreater(arc_u, 0)
        elif du < 0:
            self.assertLess(arc_u, 0)
        else:
            self.assertEqual(arc_u, 0)

        if dv > 0:
            self.assertGreater(arc_v, 0)
        elif dv < 0:
            self.assertLess(arc_v, 0)
        else:
            self.assertEqual(arc_v, 0)

        # 2. Arc length >= chord length (absolute)
        self.assertGreaterEqual(abs(arc_u), abs(du) - 1e-10)
        self.assertGreaterEqual(abs(arc_v), abs(dv) - 1e-10)

        # 3. Symmetry with respect to displacement
        arc_u_neg, arc_v_neg = arc_length_corrected_displacement(
            -du,
            -dv,
            self.basis_u,
            self.basis_v,
            principal_curvatures,
            self.pose_vectors,
        )
        self.assertAlmostEqual(arc_u_neg, -arc_u)
        self.assertAlmostEqual(arc_v_neg, -arc_v)

        # 4. Symmetry with respect to curvature sign (f(p, k) == f(p, -k))
        arc_u_nk, arc_v_nk = arc_length_corrected_displacement(
            du,
            dv,
            self.basis_u,
            self.basis_v,
            -principal_curvatures,
            self.pose_vectors,
        )
        self.assertEqual(arc_u_nk, arc_u)
        self.assertEqual(arc_v_nk, arc_v)

    def test_zero_curvature_returns_unchanged(self):
        result = arc_length_corrected_displacement(
            0.5,
            0.3,
            self.basis_u,
            self.basis_v,
            np.array([0.0, 0.0]),
            self.pose_vectors,
        )
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 0.3)

    def test_axes_corrected_independently(self):
        # k1=1.0 along basis_u, k2=0.0 along basis_v
        arc_u, arc_v = arc_length_corrected_displacement(
            0.5,
            0.5,
            self.basis_u,
            self.basis_v,
            np.array([1.0, 0.0]),
            self.pose_vectors,
        )
        # u axis should be corrected (k=1, p=0.5 => arcsin(0.5)/1 = pi/6)
        self.assertAlmostEqual(arc_u, np.pi / 6)
        # v axis has zero curvature, unchanged
        self.assertAlmostEqual(arc_v, 0.5)

    def test_symmetric_curvature_corrects_both(self):
        # k1 = k2 = 1.0, so both axes get same correction
        arc_u, arc_v = arc_length_corrected_displacement(
            0.5,
            0.5,
            self.basis_u,
            self.basis_v,
            np.array([1.0, 1.0]),
            self.pose_vectors,
        )
        expected = np.pi / 6
        self.assertAlmostEqual(arc_u, expected)
        self.assertAlmostEqual(arc_v, expected)

    def test_zero_displacement_returns_zero(self):
        arc_u, arc_v = arc_length_corrected_displacement(
            0.0,
            0.0,
            self.basis_u,
            self.basis_v,
            np.array([5.0, 5.0]),
            self.pose_vectors,
        )
        self.assertEqual(arc_u, 0.0)
        self.assertEqual(arc_v, 0.0)

    def test_negative_displacement_preserved(self):
        arc_u, arc_v = arc_length_corrected_displacement(
            -0.5,
            0.5,
            self.basis_u,
            self.basis_v,
            np.array([1.0, 1.0]),
            self.pose_vectors,
        )
        self.assertAlmostEqual(arc_u, -np.pi / 6)
        self.assertAlmostEqual(arc_v, np.pi / 6)

    def test_arc_at_least_as_long_as_chord(self):
        arc_u, arc_v = arc_length_corrected_displacement(
            0.3,
            0.4,
            self.basis_u,
            self.basis_v,
            np.array([2.0, 2.0]),
            self.pose_vectors,
        )
        self.assertGreaterEqual(abs(arc_u), 0.3)
        self.assertGreaterEqual(abs(arc_v), 0.4)
