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
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.utils.edge_detection import (
    DEFAULT_POSE_2D,
    EdgeDetector,
    StructureTensor,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)
from tbp.monty.math import DEFAULT_TOLERANCE
from tests.unit.frameworks.utils.spatial_arithmetics_test import (
    rotation_matrices,
    unit_vectors,
)

PATCH_SIZE = 64
UNIFORM_PATCH = np.full((PATCH_SIZE, PATCH_SIZE, 3), 128, dtype=np.uint8)

angles = st.floats(min_value=-2 * np.pi, max_value=2 * np.pi)

# Detected edge angle is in range (0, pi].
edge_angles = st.floats(min_value=DEFAULT_TOLERANCE, max_value=np.pi)
a_scalar = st.floats(min_value=DEFAULT_TOLERANCE, max_value=100.0)


@dataclass(frozen=True)
class EdgeDetectorGroundTruth:
    name: str
    angle: float
    camera_orientation: np.ndarray  # 3x3
    surface_normal: np.ndarray
    expected_pose_2d: np.ndarray


cases = [
    EdgeDetectorGroundTruth(
        name="identity_camera_vertical_edge",
        angle=np.pi / 2,
        camera_orientation=np.identity(3),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        expected_pose_2d=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    ),
    EdgeDetectorGroundTruth(
        name="identity_camera_diagonal_edge",
        angle=np.pi / 4,
        camera_orientation=np.identity(3),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        expected_pose_2d=np.array(
            [
                [0.0, 0.0, 1.0],
                [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
                [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            ]
        ),
    ),
    EdgeDetectorGroundTruth(
        name="identity_camera_horizontal_edge",
        angle=np.pi,
        camera_orientation=np.identity(3),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        expected_pose_2d=np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        ),
    ),
    EdgeDetectorGroundTruth(
        name="rotated_camera_surface_x_horizontal_edge",
        angle=np.pi,
        camera_orientation=np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        surface_normal=np.array([1.0, 0.0, 0.0]),
        expected_pose_2d=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    ),
    EdgeDetectorGroundTruth(
        name="camera_y_to_world_z_surface_y_diagonal_edge",
        angle=np.pi / 4,
        camera_orientation=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        surface_normal=np.array([0.0, 1.0, 0.0]),
        expected_pose_2d=np.array(
            [
                [0.0, 0.0, 1.0],
                [-np.sqrt(0.5), -np.sqrt(0.5), 0.0],
                [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            ]
        ),
    ),
]


@st.composite
def surface_geometry(draw):
    """Generate matching surface normal and tangent frame.

    Returns:
        Tuple of surface normal and tangent frame.
    """
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


def make_angled_edge_patch(angle_rad: float, size: int = PATCH_SIZE) -> np.ndarray:
    """Generate a synthetic RGB patch with a half-plane edge through the center.

    Returns:
        RGB patch containing the synthetic edge.
    """
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    center = (size - 1) / 2
    signed_distance = (cols - center) * -np.sin(angle_rad) + (rows - center) * np.cos(
        angle_rad
    )

    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[signed_distance >= 0] = 255
    return patch


def compute_expected_pose_2d(
    angle: float,
    world_camera: np.ndarray,
    surface_normal: np.ndarray,
    tangent_frame: TangentFrame,
) -> np.ndarray:
    """Compute the expected local 2D edge pose from an image-space angle.

    Returns:
        Expected pose vectors expressed in the local tangent frame.
    """
    rotation = world_camera[:3, :3]
    image_x_world = rotation @ np.array([1.0, 0.0, 0.0])
    image_y_world = rotation @ np.array([0.0, 1.0, 0.0])

    edge_world = np.cos(angle) * image_x_world - np.sin(angle) * image_y_world
    edge_tangent_world = project_onto_tangent_plane(edge_world, surface_normal)

    if np.linalg.norm(edge_tangent_world) < DEFAULT_TOLERANCE:
        edge_tangent_world = tangent_frame.basis_u
    else:
        edge_tangent_world = normalize(edge_tangent_world)

    tangent_2d = normalize(
        np.array(
            [
                np.dot(edge_tangent_world, tangent_frame.basis_u),
                np.dot(edge_tangent_world, tangent_frame.basis_v),
                0.0,
            ]
        )
    )

    return np.array(
        [
            [0.0, 0.0, 1.0],
            tangent_2d,
            [-tangent_2d[1], tangent_2d[0], 0.0],
        ]
    )


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


def sensor_observation(
    angle: float | None = None, camera_orientation: np.ndarray | None = None
) -> SensorObservation:
    """Build a minimal SensorObservation.

    Args:
        angle: If None, create a uniform rgb patch (no edge).
            Otherwise a Hypothesis strategy to draw from.
        camera_orientation: A 3x3 rotation matrix. If None, the identity matrix is used.

    Returns:
        SensorObservation.
    """
    rgb = UNIFORM_PATCH if angle is None else make_angled_edge_patch(angle)
    alpha = np.full((PATCH_SIZE, PATCH_SIZE, 1), fill_value=255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=-1)

    depth = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    world_camera = np.identity(4)
    if camera_orientation is not None:
        world_camera[:3, :3] = camera_orientation.copy()

    return SensorObservation(
        rgba=rgba,
        depth=depth,
        world_camera=world_camera,
    )


def angle_distance(actual: float, expected: float) -> float:
    """Smallest angular distance between undirected edge orientations.

    Returns:
        Absolute angular distance modulo pi.
    """
    return abs((actual - expected + np.pi / 2) % np.pi - np.pi / 2)


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


class TestEdgeDetector(unittest.TestCase):
    def test_kernel_size_is_not_odd(self):
        with pytest.raises(ValueError, match="must be odd"):
            EdgeDetector(kernel_size=2)

    @given(
        angle=edge_angles,
        camera_orientation=rotation_matrices,
        geometry=surface_geometry(),
    )
    def test_surface_normal_is_not_none(self, angle, camera_orientation, geometry):
        detector = EdgeDetector()
        _, tangent_frame = geometry
        obs = sensor_observation(angle=angle, camera_orientation=camera_orientation)

        with pytest.raises(ValueError, match="surface_normal is required"):
            detector(obs, surface_normal=None, tangent_frame=tangent_frame)

    @given(
        angle=edge_angles,
        camera_orientation=rotation_matrices,
        geometry=surface_geometry(),
    )
    def test_tangent_frame_is_not_none(self, angle, camera_orientation, geometry):
        detector = EdgeDetector()
        surface_normal_3d, _ = geometry
        obs = sensor_observation(angle=angle, camera_orientation=camera_orientation)

        with pytest.raises(ValueError, match="tangent_frame is required"):
            detector(obs, surface_normal=surface_normal_3d, tangent_frame=None)

    @given(
        angle=st.just(None),
        camera_orientation=rotation_matrices,
        geometry=surface_geometry(),
    )
    def test_uniform_patch_returns_no_edge(self, angle, camera_orientation, geometry):
        detector = EdgeDetector()
        surface_normal_3d, tangent_frame = geometry
        obs = sensor_observation(angle=angle, camera_orientation=camera_orientation)
        edge = detector(obs, surface_normal_3d, tangent_frame)

        assert edge.strength == 0.0
        assert edge.coherence == 0.0
        assert edge.angle is None
        np.testing.assert_allclose(edge.pose_2d, DEFAULT_POSE_2D)
        assert edge.is_geometric_edge is False
        assert edge.has_edge is False

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

    @given(
        angle=edge_angles,
        camera_orientation=rotation_matrices,
        geometry=surface_geometry(),
    )
    def test_angled_edge_patch_reports_expected_angle(
        self, angle, camera_orientation, geometry
    ):
        detector = EdgeDetector()
        surface_normal, tangent_frame = geometry
        obs = sensor_observation(angle=angle, camera_orientation=camera_orientation)

        edge = detector(obs, surface_normal, tangent_frame)
        assert angle_distance(edge.angle, angle) < 0.1

    def test_edge_detection_for_closed_form_solutions(self):
        detector = EdgeDetector()
        for case in cases:
            with self.subTest(case=case.name):
                surface_normal = case.surface_normal
                tangent_frame = TangentFrame(surface_normal)
                obs = sensor_observation(case.angle, case.camera_orientation)

                edge = detector(obs, surface_normal, tangent_frame)

                assert_pose_2d_is_orthonormal(edge.pose_2d)
                np.testing.assert_allclose(
                    edge.pose_2d, case.expected_pose_2d, atol=0.1
                )

    def test_angle_to_pose_2d_uses_transported_tangent_frame(self):
        detector = EdgeDetector()
        tangent_frame = TangentFrame(np.array([0.0, 0.0, 1.0]))

        for normal in [
            normalize(np.array([0.0, 0.2, 1.0])),
            normalize(np.array([0.0, 0.5, 1.0])),
            normalize(np.array([0.0, 1.0, 1.0])),
            normalize(np.array([0.0, 1.0, 0.0])),
        ]:
            tangent_frame.transport(normal)

        surface_normal = tangent_frame.normal
        angle = np.pi / 2
        world_camera = np.identity(4)

        obs = sensor_observation(angle, world_camera[:3, :3])
        edge = detector(obs, surface_normal, tangent_frame)

        np.testing.assert_allclose(
            edge.pose_2d,
            compute_expected_pose_2d(
                angle,
                world_camera,
                surface_normal,
                tangent_frame,
            ),
            atol=0.1,
        )
