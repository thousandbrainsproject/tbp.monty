# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = ["RotationSubsystem"]


class RotationSubsystem:
    """Handle rotation hypotheses via pose vector alignment.

    Rotation is NOT path-integrated via grid cells. Instead, rotation
    hypotheses are inferred from feature alignment at each step and refined
    through evidence accumulation. This matches the biological separation
    of grid cells (translation) from head direction cells (orientation).

    The key operation is: given sensed and stored pose vectors (surface
    normal + principal curvature direction), compute compatible rotations
    by aligning the two orthonormal frames. This typically produces 2
    rotations per location (180 degree curvature ambiguity), or more on
    isotropic surfaces.
    """

    def __init__(self, num_isotropic_samples: int = 8):
        """Initialise the rotation subsystem.

        Args:
            num_isotropic_samples: Number of rotation samples to generate
                when the surface is isotropic (curvature directions undefined).
        """
        self.num_isotropic_samples = num_isotropic_samples

    def initialise_rotations(
        self,
        sensed_normal: np.ndarray,
        sensed_curvature_dir: np.ndarray,
        stored_normal: np.ndarray,
        stored_curvature_dir: np.ndarray,
        pc1_is_pc2: bool = False,
    ) -> list[np.ndarray]:
        """Compute compatible rotations by aligning sensed and stored pose vectors.

        For a non-isotropic surface, produces 2 rotations (curvature
        direction can be aligned or flipped 180 degrees). For an isotropic
        surface, produces num_isotropic_samples rotations uniformly sampled
        around the normal axis.

        Args:
            sensed_normal: Sensed surface normal, shape (3,).
            sensed_curvature_dir: Sensed principal curvature direction, shape (3,).
            stored_normal: Stored surface normal, shape (3,).
            stored_curvature_dir: Stored principal curvature direction, shape (3,).
            pc1_is_pc2: Whether the surface is isotropic (principal curvatures
                are equal, so curvature direction is undefined).

        Returns:
            List of 3x3 rotation matrices. Each R satisfies (approximately):
                R @ sensed_normal ~= stored_normal
                R @ sensed_curvature_dir ~= +/- stored_curvature_dir
        """
        rotations = []

        if pc1_is_pc2:
            # Isotropic surface: sample rotations around the normal axis
            R_align = self._align_single_vector(sensed_normal, stored_normal)
            for angle in np.linspace(
                0, 2 * np.pi, self.num_isotropic_samples, endpoint=False
            ):
                R_around_normal = self._axis_angle_rotation(stored_normal, angle)
                rotations.append(R_around_normal @ R_align)
        else:
            # Build orthonormal frames from normal + curvature direction
            sensed_frame = self._build_frame(sensed_normal, sensed_curvature_dir)
            stored_frame = self._build_frame(stored_normal, stored_curvature_dir)

            # R that maps sensed_frame -> stored_frame
            # stored = R @ sensed => R = stored @ sensed^T
            R1 = stored_frame @ sensed_frame.T
            rotations.append(R1)

            # 180-degree curvature ambiguity: flip curvature direction
            stored_frame_flipped = self._build_frame(
                stored_normal, -stored_curvature_dir
            )
            R2 = stored_frame_flipped @ sensed_frame.T
            rotations.append(R2)

        return rotations

    @staticmethod
    def morphology_evidence(
        sensed_normal: np.ndarray,
        sensed_curvature_dir: np.ndarray,
        stored_normal: np.ndarray,
        stored_curvature_dir: np.ndarray,
        rotation: np.ndarray,
    ) -> float:
        """Compute morphology evidence from pose vector alignment quality.

        Applies the hypothesis rotation to the sensed vectors and compares
        with stored vectors. Normal similarity is in [-1, 1]; curvature
        direction similarity uses absolute cosine (ambiguity) in [0, 1].

        The combined evidence is in [-1, 1]: negative means poor alignment,
        positive means good alignment.

        Args:
            sensed_normal: Sensed surface normal, shape (3,).
            sensed_curvature_dir: Sensed principal curvature direction, shape (3,).
            stored_normal: Stored surface normal, shape (3,).
            stored_curvature_dir: Stored principal curvature direction, shape (3,).
            rotation: 3x3 rotation matrix (hypothesis rotation R_k).

        Returns:
            Evidence score in [-1, 1].
        """
        rotated_normal = rotation @ sensed_normal
        rotated_curv = rotation @ sensed_curvature_dir

        # Normal similarity: full cosine in [-1, 1]
        normal_sim = float(np.dot(rotated_normal, stored_normal))

        # Curvature direction: absolute cosine (180-degree ambiguity) in [0, 1]
        curv_sim = float(abs(np.dot(rotated_curv, stored_curvature_dir)))

        # Map curvature similarity from [0, 1] to [-1, 1] and average
        return 0.5 * normal_sim + 0.5 * (2.0 * curv_sim - 1.0)

    @staticmethod
    def _align_single_vector(
        source: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """Compute rotation that aligns source vector to target vector.

        Args:
            source: Source unit vector, shape (3,).
            target: Target unit vector, shape (3,).

        Returns:
            3x3 rotation matrix.
        """
        source = source / (np.linalg.norm(source) + 1e-10)
        target = target / (np.linalg.norm(target) + 1e-10)

        v = np.cross(source, target)
        c = np.dot(source, target)

        if c < -0.9999:
            # Nearly anti-parallel: find any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(source[0]) < 0.9 else np.array([0, 1, 0])
            perp = perp - np.dot(perp, source) * source
            perp = perp / np.linalg.norm(perp)
            return -np.eye(3) + 2.0 * np.outer(perp, perp)

        # Rodrigues' formula
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        return np.eye(3) + vx + vx @ vx / (1.0 + c)

    @staticmethod
    def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
        """Compute rotation matrix from axis-angle representation.

        Args:
            axis: Unit rotation axis, shape (3,).
            angle: Rotation angle in radians.

        Returns:
            3x3 rotation matrix.
        """
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        return Rotation.from_rotvec(angle * axis).as_matrix()

    @staticmethod
    def _build_frame(
        normal: np.ndarray, curvature_dir: np.ndarray
    ) -> np.ndarray:
        """Build an orthonormal frame from normal and curvature direction.

        The frame is [normal, curvature_dir_orthogonalised, cross_product].

        Args:
            normal: Surface normal, shape (3,).
            curvature_dir: Principal curvature direction, shape (3,).

        Returns:
            3x3 matrix where each row is a frame vector.
        """
        n = normal / (np.linalg.norm(normal) + 1e-10)
        # Ensure curvature direction is orthogonal to normal
        c = curvature_dir - np.dot(curvature_dir, n) * n
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            # Curvature direction parallel to normal — pick arbitrary
            perp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
            c = perp - np.dot(perp, n) * n
            c_norm = np.linalg.norm(c)
        c = c / c_norm
        t = np.cross(n, c)
        return np.array([n, c, t])
