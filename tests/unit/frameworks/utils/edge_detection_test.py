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

import numpy as np
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.utils.edge_detection import (
    EdgeDetector,
    StructureTensor,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.spatial_arithmetics_test import unit_vectors

PATCH_SIZE = 64
SURFACE_NORMAL_3D = np.array([0.0, 0.0, 1.0])

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)
a_scalar = st.floats(min_value=DEFAULT_TOLERANCE, max_value=100.0)


@st.composite
def surface_geometry(draw):
    """Generate matching surface normal and tangent frame."""
    surface_normal_3d = draw(unit_vectors)
    return surface_normal_3d, TangentFrame(surface_normal_3d)


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
    return StructureTensor(xx=Jxx, yy=Jyy, xy=Jxy)


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


def assert_pose_2d_is_orthonormal(pose_2d: np.ndarray) -> None:
    """Assert that pose rows form an orthonormal right-handed local 2D basis."""
    normal, tangent, perpendicular = pose_2d

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=DEFAULT_TOLERANCE)

    # Check unit norm.
    np.testing.assert_allclose(np.linalg.norm(tangent), 1.0, atol=DEFAULT_TOLERANCE)
    np.testing.assert_allclose(
        np.linalg.norm(perpendicular), 1.0, atol=DEFAULT_TOLERANCE
    )

    # Check orthogonality.
    np.testing.assert_allclose(np.dot(normal, tangent), 0.0, atol=DEFAULT_TOLERANCE)
    np.testing.assert_allclose(
        np.dot(normal, perpendicular), 0.0, atol=DEFAULT_TOLERANCE
    )
    np.testing.assert_allclose(
        np.dot(tangent, perpendicular), 0.0, atol=DEFAULT_TOLERANCE
    )

    # Check right-handedness.
    np.testing.assert_allclose(
        np.cross(tangent, perpendicular), normal, atol=DEFAULT_TOLERANCE
    )


@st.composite
def camera_extrinsic_matrix(draw):
    """Draw a random 4x4 camera-to-world matrix.

    Returns:
         world_camera: 4x4 camera-to-world matrix.
    """
    rng = np.random.default_rng()
    rot_3x3 = Rotation.random(random_state=rng).as_matrix()
    tx, ty, tz = (draw(st.floats(min_value=-100.0, max_value=100.0)) for _ in range(3))
    world_to_cam = np.identity(4)
    world_to_cam[:3, :3] = rot_3x3
    world_to_cam[:3, 3] = [tx, ty, tz]
    return world_to_cam


@st.composite
def sensor_observation(draw, patterns=None):
    if patterns is None:
        patterns = ["uniform", "vertical_edge", "horizontal_edge", "diagonal_edge"]

    pattern = draw(st.sampled_from(patterns))
    rgb = make_rgb_patch(PATCH_SIZE, pattern)
    alpha = np.full((PATCH_SIZE, PATCH_SIZE, 1), fill_value=255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)

    depth = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    world_camera = draw(camera_extrinsic_matrix())
    return SensorObservation(
        rgba=rgba,
        depth=depth,
        world_camera=world_camera,
    )


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
    return (h, w), Ix, Iy, radius, sigma_r


@st.composite
def center_check_inputs(draw):
    """Generate valid inputs for _passes_center_check.

    Uses center_weight_inputs to get realistic (weights, total_weight) pairs
    with total_weight > 0. Filters the rare zero-weight case.

    Returns:
        Tuple of (weights, total_weight, gradient_theta, max_center_offset).
    """
    rng = np.random.default_rng()
    weights = rng.uniform(0.0, 1.0, size=(PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
    total_weight = weights.sum()
    assume(total_weight > 0)
    gradient_theta = draw(angles)
    max_center_offset = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=50))
    )
    return weights, total_weight, gradient_theta, max_center_offset


class StructureTensorTest(unittest.TestCase):
    def test_eigenvalues_match_analytical(self):
        t = StructureTensor(xx=3.0, yy=1.0, xy=1.0)
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(
            lambda_min, 2.0 - np.sqrt(2.0), atol=DEFAULT_TOLERANCE
        )
        np.testing.assert_allclose(
            lambda_max, 2.0 + np.sqrt(2.0), atol=DEFAULT_TOLERANCE
        )

    @given(t=structure_tensors())
    @example(t=StructureTensor(xx=0.0, yy=0.0, xy=0.0))
    def test_eigenvalues_ordered(self, t):
        lambda_min, lambda_max = t.eigenvalues
        assert lambda_min <= lambda_max

    @given(t=structure_tensors())
    @example(t=StructureTensor(xx=0.0, yy=9.0, xy=0.0))
    def test_edge_strength_nonnegative(self, t):
        assert t.edge_strength >= 0.0

    @given(t=structure_tensors())
    @example(t=StructureTensor(xx=4.0, yy=0.0, xy=0.0))
    def test_coherence_in_unit_interval(self, t):
        assert 0.0 <= t.coherence <= 1.0

    @given(t=structure_tensors())
    @example(t=StructureTensor(xx=4.0, yy=0.0, xy=0.0))
    def test_edge_angle_range(self, t):
        assert 0.0 <= t.edge_angle <= np.pi

    @given(t=structure_tensors())
    def test_eigenvalue_trace_equals_jxx_plus_jyy(self, t):
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(
            lambda_min + lambda_max, t.xx + t.yy, atol=DEFAULT_TOLERANCE
        )

    @given(t=structure_tensors())
    def test_eigenvalue_product_equals_determinant(self, t):
        lambda_min, lambda_max = t.eigenvalues
        np.testing.assert_allclose(
            lambda_min * lambda_max, t.xx * t.yy - t.xy**2, atol=DEFAULT_TOLERANCE
        )

    @given(k=a_scalar)
    def test_isotropic_coherence_is_zero(self, k):
        t = StructureTensor(xx=k, yy=k, xy=0.0)
        np.testing.assert_allclose(t.coherence, 0.0, atol=DEFAULT_TOLERANCE)

    @given(t=structure_tensors(), k=a_scalar)
    def test_scaling_multiplies_edge_strength(self, t, k):
        scaled = StructureTensor(xx=k * t.xx, yy=k * t.yy, xy=k * t.xy)
        np.testing.assert_allclose(
            scaled.edge_strength, np.sqrt(k) * t.edge_strength, atol=DEFAULT_TOLERANCE
        )

    @given(t=structure_tensors(), k=a_scalar)
    @example(t=StructureTensor(xx=4.0, yy=0.0, xy=0.0), k=2.0)
    @example(t=StructureTensor(xx=0.0, yy=9.0, xy=0.0), k=3.0)
    def test_scaling_preserves_gradient_theta(self, t, k):
        scaled = StructureTensor(xx=k * t.xx, yy=k * t.yy, xy=k * t.xy)
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
        detector = EdgeDetector()

        weights, total_weight = detector._compute_center_weights((h, w), Ix, Iy)

        assert total_weight == 0.0
        assert np.all(weights == 0.0)

    def test_center_pixel_has_maximum_weight(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.ones((h, w), dtype=np.float32)  # noqa: N806
        Iy = np.ones((h, w), dtype=np.float32)  # noqa: N806
        detector = EdgeDetector(radius=1000.0)

        weights, _ = detector._compute_center_weights((h, w), Ix, Iy)

        r0, c0 = h // 2, w // 2
        assert weights[r0, c0] == np.max(weights)

    def test_pixel_just_outside_radius_is_zero(self):
        h, w = PATCH_SIZE, PATCH_SIZE
        Ix = np.ones((h, w), dtype=np.float32)  # noqa: N806
        Iy = np.ones((h, w), dtype=np.float32)  # noqa: N806
        detector = EdgeDetector(radius=2.0)

        weights, _ = detector._compute_center_weights((h, w), Ix, Iy)

        r0, c0 = h // 2, w // 2
        assert weights[r0 + 3, c0] == 0.0

    @given(inputs=center_weight_inputs())
    def test_weights_nonnegative(self, inputs):
        shape, Ix, Iy, radius, sigma_r = inputs  # noqa: N806
        detector = EdgeDetector(radius=radius, sigma_r=sigma_r)

        weights, _ = detector._compute_center_weights(shape, Ix, Iy)

        assert np.all(weights >= 0.0)

    @given(inputs=center_weight_inputs())
    def test_total_weight_equals_sum_of_weights(self, inputs):
        shape, Ix, Iy, radius, sigma_r = inputs  # noqa: N806
        detector = EdgeDetector(radius=radius, sigma_r=sigma_r)

        weights, total_weight = detector._compute_center_weights(shape, Ix, Iy)

        np.testing.assert_allclose(total_weight, np.sum(weights))

    @given(inputs=center_weight_inputs())
    def test_output_shape_matches_input(self, inputs):
        shape, Ix, Iy, radius, sigma_r = inputs  # noqa: N806
        detector = EdgeDetector(radius=radius, sigma_r=sigma_r)

        weights, _ = detector._compute_center_weights(shape, Ix, Iy)

        assert weights.shape == shape

    @given(inputs=center_weight_inputs())
    def test_pixels_beyond_radius_have_zero_weight(self, inputs):
        shape, Ix, Iy, radius, sigma_r = inputs  # noqa: N806
        h, w = shape
        r0, c0 = h // 2, w // 2
        detector = EdgeDetector(radius=radius, sigma_r=sigma_r)

        weights, _ = detector._compute_center_weights(shape, Ix, Iy)

        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d = np.sqrt((rows - r0) ** 2 + (cols - c0) ** 2)
        assert np.all(weights[d > radius] == 0.0)

    @given(inputs=center_weight_inputs(), k=a_scalar)
    def test_gradient_scaling_scales_weights_quadratically(self, inputs, k):
        shape, Ix, Iy, radius, sigma_r = inputs  # noqa: N806
        detector = EdgeDetector(radius=radius, sigma_r=sigma_r)

        weights, total_weight = detector._compute_center_weights(shape, Ix, Iy)
        weights_scaled, total_weight_scaled = detector._compute_center_weights(
            shape, k * Ix, k * Iy
        )

        tol = max(DEFAULT_TOLERANCE * k**2, DEFAULT_TOLERANCE)
        np.testing.assert_allclose(
            weights_scaled, k**2 * weights, rtol=DEFAULT_TOLERANCE, atol=tol
        )
        np.testing.assert_allclose(
            total_weight_scaled, k**2 * total_weight, rtol=DEFAULT_TOLERANCE, atol=tol
        )


class PassesCenterCheckTest(unittest.TestCase):
    @given(inputs=center_check_inputs())
    def test_none_offset_always_true(self, inputs):
        weights, total_weight, gradient_theta, _ = inputs
        detector = EdgeDetector()
        assert detector._passes_center_check(weights, total_weight, gradient_theta)

    @given(
        inputs=center_weight_inputs(),
        theta=angles,
        offset=st.integers(min_value=1, max_value=100),
    )
    def test_symmetric_weights_pass_any_nonneg_offset(self, inputs, theta, offset):
        # center_weight_inputs produces radially symmetric weights (radial Gaussian
        # * uniform gradient magnitude), so sum(weights*(cols-c0)) ~= 0 and
        # sum(weights*(rows-r0)) ~= 0, giving d_center ~= 0 for any theta.
        shape, Ix, Iy, radius, sigma_r = inputs  # noqa: N806
        detector = EdgeDetector(
            radius=radius, sigma_r=sigma_r, max_center_offset=offset
        )
        weights, total_weight = detector._compute_center_weights(shape, Ix, Iy)

        assume(total_weight > 0)
        assert detector._passes_center_check(weights, total_weight, theta)


class TestEdgeDetector(unittest.TestCase):
    def test_kernel_size_is_not_odd(self):
        with pytest.raises(ValueError, match="must be odd"):
            EdgeDetector(kernel_size=2)

    @given(
        obs=sensor_observation(patterns=["vertical_edge"]),
        geometry=surface_geometry(),
    )
    def test_surface_normal_is_not_none(self, obs, geometry):
        detector = EdgeDetector()
        _, tangent_frame = geometry

        with pytest.raises(ValueError, match="surface_normal is required"):
            detector(obs, surface_normal=None, tangent_frame=tangent_frame)

    @given(
        obs=sensor_observation(patterns=["vertical_edge"]),
        geometry=surface_geometry(),
    )
    def test_tangent_frame_is_not_none(self, obs, geometry):
        detector = EdgeDetector()
        surface_normal_3d, _ = geometry

        with pytest.raises(ValueError, match="tangent_frame is required"):
            detector(obs, surface_normal=surface_normal_3d, tangent_frame=None)

    @given(
        obs=sensor_observation(patterns=["uniform"]),
        geometry=surface_geometry(),
    )
    def test_uniform_patch_returns_zero_strength(self, obs, geometry):
        detector = EdgeDetector()
        surface_normal_3d, tangent_frame = geometry

        edge = detector(obs, surface_normal_3d, tangent_frame)

        assert edge.strength == 0.0
        assert edge.coherence == 0.0
        assert edge.angle is None

    @given(
        obs=sensor_observation(patterns=["vertical_edge"]),
        geometry=surface_geometry(),
    )
    def test_vertical_edge_detected(self, obs, geometry):
        detector = EdgeDetector()
        surface_normal_3d, tangent_frame = geometry

        edge = detector(obs, surface_normal_3d, tangent_frame)

        assert edge.strength > 0.0
        assert edge.coherence > 0.0
        assert edge.angle is not None
        assert abs(edge.angle - np.pi / 2) < 0.3

    @given(
        obs=sensor_observation(patterns=["horizontal_edge"]),
        geometry=surface_geometry(),
    )
    def test_horizontal_edge_orientation(self, obs, geometry):
        detector = EdgeDetector()
        surface_normal_3d, tangent_frame = geometry

        edge = detector(obs, surface_normal_3d, tangent_frame)

        # Horizontal edge tangent is always pi (structure tensor is sign-invariant)
        assert edge.angle is not None
        assert abs(edge.angle - np.pi) < 0.3

    @given(
        obs=sensor_observation(patterns=["diagonal_edge"]),
        geometry=surface_geometry(),
    )
    def test_diagonal_edge_detected(self, obs, geometry):
        detector = EdgeDetector()
        surface_normal_3d, tangent_frame = geometry

        edge = detector(obs, surface_normal_3d, tangent_frame)

        assert edge.strength > 0.0
        assert edge.coherence > 0.0
        assert edge.angle is not None

    def test_center_offset_rejects_off_center_edge(self):
        # Edge at right boundary, not at center
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        patch[:, PATCH_SIZE - 4 :, :] = 255
        alpha = np.ones((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        rgba = np.concatenate((patch, alpha), axis=-1)
        obs = SensorObservation(
            rgba=rgba,
            depth=np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32),
            world_camera=np.identity(4),
        )
        # Set radius to force total_weight > DEFAULT_TOLERANCE
        detector = EdgeDetector(radius=PATCH_SIZE // 2, max_center_offset=1)
        surface_normal_3d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        tangent_frame = TangentFrame(surface_normal_3d)

        edge = detector(obs, surface_normal_3d, tangent_frame)

        assert edge.strength == 0.0
        assert edge.coherence == 0.0
        assert edge.angle is None

    @given(obs=sensor_observation(), geometry=surface_geometry())
    def test_output_ranges_valid(self, obs, geometry):
        detector = EdgeDetector()
        surface_normal_3d, tangent_frame = geometry

        edge = detector(obs, surface_normal_3d, tangent_frame)

        assert edge.strength >= 0.0
        assert 0.0 <= edge.coherence <= 1.0
        assert edge.angle is None or 0.0 <= edge.angle <= np.pi

    @given(angle=angles, geometry=surface_geometry())
    @example(
        angle=0.0,
        geometry=(
            np.array([0.0, 0.0, 1.0]),
            TangentFrame(np.array([0.0, 0.0, 1.0])),
        ),
    )
    @example(
        angle=np.pi / 2,
        geometry=(
            np.array([0.0, 0.0, 1.0]),
            TangentFrame(np.array([0.0, 0.0, 1.0])),
        ),
    )
    def test_angle_to_pose_2d_maps_identity_camera_to_local_tangent_frame(
        self, angle, geometry
    ):
        detector = EdgeDetector()
        world_camera = np.identity(4)
        surface_normal_3d, tangent_frame = geometry

        pose_2d = detector._angle_to_pose_2d(
            angle,
            world_camera,
            surface_normal=surface_normal_3d,
            tangent_frame=tangent_frame,
        )

        edge_world = np.array([np.cos(angle), -np.sin(angle), 0.0])
        edge_tangent_world = project_onto_tangent_plane(edge_world, surface_normal_3d)
        if np.linalg.norm(edge_tangent_world) < DEFAULT_TOLERANCE:
            edge_tangent_world = tangent_frame.basis_u
        else:
            edge_tangent_world = normalize(edge_tangent_world)

        expected_tangent_2d = np.array(
            [
                np.dot(edge_tangent_world, tangent_frame.basis_u),
                np.dot(edge_tangent_world, tangent_frame.basis_v),
                0.0,
            ]
        )

        assert_pose_2d_is_orthonormal(pose_2d)
        np.testing.assert_allclose(
            pose_2d[1], expected_tangent_2d, atol=DEFAULT_TOLERANCE
        )

    def test_angle_to_pose_2d_uses_transported_tangent_frame(self):
        detector = EdgeDetector()
        world_camera = np.identity(4)
        world_camera[:3, :3] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        )
        surface_normal = np.array([0.0, 1.0, 0.0])
        tangent_frame = TangentFrame(np.array([0.0, 0.0, 1.0]))
        tangent_frame.transport(surface_normal)

        pose_2d = detector._angle_to_pose_2d(
            np.pi / 2,
            world_camera,
            surface_normal=surface_normal,
            tangent_frame=tangent_frame,
        )

        np.testing.assert_allclose(
            pose_2d,
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            atol=DEFAULT_TOLERANCE,
        )

    def test_angle_to_pose_2d_falls_back_to_basis_u_for_degenerate_projection(self):
        detector = EdgeDetector()
        surface_normal = np.array([1.0, 0.0, 0.0])
        tangent_frame = TangentFrame(surface_normal)

        pose_2d = detector._angle_to_pose_2d(
            0.0,
            np.identity(4),
            surface_normal=surface_normal,
            tangent_frame=tangent_frame,
        )

        assert_pose_2d_is_orthonormal(pose_2d)
        np.testing.assert_allclose(
            pose_2d,
            np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [-0.0, 1.0, 0.0],
                ]
            ),
            atol=DEFAULT_TOLERANCE,
        )
