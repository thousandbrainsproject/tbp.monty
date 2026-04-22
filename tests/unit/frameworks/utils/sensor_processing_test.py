# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

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
    directional_curvature,
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

    @given(
        vectors=orthonormal_vectors(),
        a_scaler=st.floats(min_value=-1e3, max_value=1e3).filter(
            lambda x: abs(x) > DEFAULT_TOLERANCE
        ),
    )
    def test_non_orthogonal_pcs_raises(self, vectors, a_scaler):
        pc1, _ = vectors
        bad_pc2 = pc1 * a_scaler
        with pytest.raises(ValueError, match="must be orthogonal"):
            directional_curvature(
                movement_direction=Mock(),
                k1=Mock(),
                k2=Mock(),
                pc1_dir=pc1,
                pc2_dir=bad_pc2,
            )

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
