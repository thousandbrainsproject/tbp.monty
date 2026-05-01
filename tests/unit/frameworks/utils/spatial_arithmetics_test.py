# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest

import numpy as np
import numpy.typing as npt
import quaternion as qt
from hypothesis import assume, example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)
from tbp.monty.math import DEFAULT_TOLERANCE


@st.composite
def vectors_3d(
    draw: st.DrawFn,
    min_value: float = -1e6,
    max_value: float = 1e6,
    dtype: npt.DTypeLike = np.float32,
):
    # TODO(scottcanoe): Reconsider np.float32 as the default dtype.
    return draw(
        arrays(
            dtype=dtype,
            shape=3,
            elements=st.floats(min_value=min_value, max_value=max_value, width=32),
        )
    )


@st.composite
def nonzero_magnitude_vectors(
    draw: st.DrawFn,
    min_value: float = -1e6,
    max_value: float = 1e6,
    dtype: npt.DTypeLike = np.float32,
):
    # TODO(scottcanoe): Reconsider np.float32 as the default dtype.
    return draw(
        vectors_3d(min_value=min_value, max_value=max_value, dtype=dtype).filter(
            lambda v: np.linalg.norm(v) > DEFAULT_TOLERANCE
        )
    )


unit_vectors = nonzero_magnitude_vectors().map(lambda v: normalize(v))


@st.composite
def nonzero_orthogonal_vectors(draw: st.DrawFn):
    random_base = normalize(draw(nonzero_magnitude_vectors()))
    n = normalize(draw(nonzero_magnitude_vectors()))
    v = np.cross(random_base, n)
    assume(np.linalg.norm(v) > DEFAULT_TOLERANCE)
    return v, n


@st.composite
def quaternions(draw: st.DrawFn):
    wxyz = draw(
        arrays(
            dtype=np.float64,
            shape=4,
            elements=st.floats(min_value=-1, max_value=1, width=32),
        ).filter(lambda v: np.linalg.norm(v) > DEFAULT_TOLERANCE)
    )
    wxyz = normalize(wxyz)
    return qt.quaternion(*wxyz)


rotation_objs = quaternions().map(lambda q: Rotation.from_quat([q.x, q.y, q.z, q.w]))
rotation_matrices = quaternions().map(
    lambda q: Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
)


class NormalizeTest(unittest.TestCase):
    @given(nonzero_magnitude_vectors())
    def test_preserves_direction(self, v):
        norm = np.linalg.norm(v)
        result = normalize(v)
        tol = max(DEFAULT_TOLERANCE * norm, DEFAULT_TOLERANCE)
        np.testing.assert_allclose(result * norm, v, atol=tol, rtol=tol)

    @given(nonzero_magnitude_vectors())
    def test_idempotent(self, v):
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_allclose(
            twice, once, atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE
        )

    def test_zero_vector_raises(self):
        v = np.zeros(3, dtype=float)
        with self.assertRaises(ValueError):
            normalize(v)

    @given(
        epsilon=st.floats(min_value=DEFAULT_TOLERANCE, max_value=1e-2),
        scale=st.floats(min_value=0.01, max_value=0.99),
    )
    def test_custom_epsilon(self, epsilon, scale):
        v = np.array([epsilon * scale, 0.0, 0.0])
        with self.assertRaises(ValueError):
            normalize(v, epsilon=epsilon)

    @given(nonzero_magnitude_vectors())
    def test_result_has_unit_norm(self, v):
        result = normalize(v)
        np.testing.assert_allclose(
            np.linalg.norm(result), 1.0, atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE
        )


class ProjectOntoTangentPlaneTest(unittest.TestCase):
    @given(
        a_vector=nonzero_magnitude_vectors(),
        a_scalar=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    )
    def test_a_vector_parallel_to_normal(self, a_vector, a_scalar):
        parallel_vector = a_scalar * a_vector
        result = project_onto_tangent_plane(parallel_vector, a_vector)
        tol = max(
            DEFAULT_TOLERANCE * np.linalg.norm(parallel_vector),
            DEFAULT_TOLERANCE * np.linalg.norm(a_vector),
            DEFAULT_TOLERANCE,
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=tol)

    @given(nonzero_orthogonal_vectors())
    def test_a_vector_perpendicular_to_normal(self, orthogonal_vectors):
        a_vector, a_normal = orthogonal_vectors
        result = project_onto_tangent_plane(a_vector, a_normal)
        tol = max(DEFAULT_TOLERANCE * np.linalg.norm(a_vector), DEFAULT_TOLERANCE)
        np.testing.assert_allclose(result, a_vector, atol=tol, rtol=DEFAULT_TOLERANCE)

    @given(a_vector=vectors_3d(), a_normal=nonzero_magnitude_vectors())
    def test_result_is_orthogonal_to_normal(self, a_vector, a_normal):
        result = project_onto_tangent_plane(a_vector, a_normal)
        tol = max(DEFAULT_TOLERANCE * np.linalg.norm(a_vector), DEFAULT_TOLERANCE)
        np.testing.assert_allclose(np.dot(result, normalize(a_normal)), 0.0, atol=tol)

    @given(a_vector=vectors_3d(), a_normal=nonzero_magnitude_vectors())
    def test_projection_is_idempotent(self, a_vector, a_normal):
        once = project_onto_tangent_plane(a_vector, a_normal)
        twice = project_onto_tangent_plane(once, a_normal)
        tol = max(DEFAULT_TOLERANCE * np.linalg.norm(a_vector), DEFAULT_TOLERANCE)
        np.testing.assert_allclose(twice, once, atol=tol, rtol=DEFAULT_TOLERANCE)


class TangentFrameTest(unittest.TestCase):
    def _assert_orthonormal_frame(
        self, frame: TangentFrame, normal: np.ndarray, tol: float = DEFAULT_TOLERANCE
    ) -> None:
        """Assert (basis_u, basis_v, normal) form an orthonormal right-handed frame."""
        u, v = frame.basis_u, frame.basis_v
        # Check unit norm
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=tol, rtol=tol)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=tol, rtol=tol)
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=tol, rtol=tol)

        # Check orthogonality
        np.testing.assert_allclose(np.dot(u, v), 0.0, atol=tol, rtol=0)
        np.testing.assert_allclose(np.dot(u, normal), 0.0, atol=tol, rtol=0)
        np.testing.assert_allclose(np.dot(v, normal), 0.0, atol=tol, rtol=0)

        # Check right-handedness
        np.testing.assert_allclose(np.cross(u, v), normal, atol=tol, rtol=tol)
        np.testing.assert_allclose(np.cross(v, u), -normal, atol=tol, rtol=tol)

    @given(n=unit_vectors)
    @example(
        n=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    @example(
        n=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    @example(
        n=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    def test_init_creates_orthonormal_frame(self, n):
        frame = TangentFrame(n)
        self._assert_orthonormal_frame(frame, n)

    def test_accumulated_transports_stay_orthonormal(self):
        rng = np.random.RandomState(42)
        n = normalize(rng.randn(3))
        frame = TangentFrame(n)
        for _ in range(1000):
            n_new = normalize(n + 0.05 * rng.randn(3))
            frame.transport(n_new)
            n = n_new
        self._assert_orthonormal_frame(frame, n, tol=DEFAULT_TOLERANCE)

    @given(n=unit_vectors)
    def test_transport_to_same_normal_is_noop(self, n):
        frame = TangentFrame(n)
        u_before, v_before = frame.basis_u.copy(), frame.basis_v.copy()
        frame.transport(n)
        np.testing.assert_allclose(
            frame.basis_u, u_before, atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE
        )
        np.testing.assert_allclose(
            frame.basis_v, v_before, atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE
        )

    @given(n1=unit_vectors, n2=unit_vectors)
    @example(
        n1=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        n2=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    def test_transport_preserves_orthonormality(self, n1, n2):
        frame = TangentFrame(n1)
        frame.transport(n2)
        self._assert_orthonormal_frame(frame, n2)
