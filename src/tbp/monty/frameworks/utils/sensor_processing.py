# Copyright 2025-2026 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging

import numpy as np
import torch
from numpy.typing import ArrayLike

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_angle_torch,
    get_right_hand_angle,
    non_singular_mat,
    normalize,
)

logger = logging.getLogger(__name__)

FLAT_THRESHOLD = 0.001


def arc_from_projection(
    tangent_projection: float,
    curvature: float,
    threshold: float = FLAT_THRESHOLD,
) -> float:
    """Correct displacement to true arc length on a curved surface.

    When a sensor moves along a curved surface, the straight-line displacement
    measured in the tangent plane underestimates the true distance traveled
    along the curve. This function corrects that by converting the displacement
    projection back to the actual arc length.

    The correction assumes that the surface is locally a circle. This approximation
    holds well when curvature is approximately constant over the displacement,
    but is inaccurate for surfaces with rapidly varying curvature.

    The relationship between arc length and its tangent-plane projection on a
    circle of curvature k is:

        tangent_projection = sin(k * arc_length) / k
        arc_length        = arcsin(k * tangent_projection) / k

    Reference:
        Do Carmo, M.P. "Differential Geometry of Curves and Surfaces",
        2nd ed., Dover, 2016, Section 3-2.

    Note:
        The formula works for both convex and concave surfaces because the
        arc-to-projection geometry on a circle is the same regardless of the sign
        of curvature.

    Args:
        tangent_projection: Signed displacement component projected onto a
            tangent-plane basis direction.
        curvature: Normal curvature along the basis direction (from Euler's
            formula). May be positive (convex) or negative (concave).
        threshold: Skip correction when |k * p| < threshold (the flat
            approximation is already accurate).

    Returns:
        Estimated signed arc length. Returns tangent_projection unchanged if
        |k * p| < threshold (arc-chord difference is negligible) or
        |k * p| >= 1.0 (arcsin domain guard).
    """
    abs_k = abs(curvature)
    abs_p = abs(tangent_projection)
    kp = abs_k * abs_p

    if kp < threshold:
        # When |k * p| < 0.001, the difference in arcsin(k*p) and k*p
        # is negligible and no arc-length correction is needed
        return tangent_projection

    if kp >= 1.0:
        # |k * p| >= 1.0 means the chord length p is greater than the radius of
        # curvature, i.e. we move beyond the most extreme visible point of the surface,
        # assuming uniform curvature. Either we are on a different surface, or the
        # curvature is not uniform (the surface curves back), and the system will
        # underestimate 2D distance traveled. The higher-level solution is a policy
        # that takes smaller steps in 3D space to better estimate movements in 2D space.
        logger.warning(
            "Arc correction skipped: |k*p| = %.4f >= 1.0 "
            "(tangent_projection=%.6f, curvature=%.6f)",
            kp,
            tangent_projection,
            curvature,
        )
        return tangent_projection

    arc_length = np.arcsin(kp) / abs_k
    return float(np.copysign(arc_length, tangent_projection))


def is_orthogonal(v1: ArrayLike, v2: ArrayLike, tolerance: float = 1e-6) -> bool:
    dot = np.dot(v1, v2)
    return np.allclose(dot, 0.0, atol=tolerance)


def is_unit_vector(vector: ArrayLike, tolerance: float = 1e-6) -> bool:
    return np.allclose(np.linalg.norm(vector), 1.0, atol=tolerance)


def is_coplanar(
    basis_1: ArrayLike, basis_2: ArrayLike, vector: ArrayLike, tolerance: float = 1e-6
) -> bool:
    plane_normal = np.cross(basis_1, basis_2)
    out_of_plane_magnitude = abs(np.dot(vector, plane_normal))
    return np.allclose(out_of_plane_magnitude, 0.0, atol=tolerance)


def directional_curvature(
    movement_direction: ArrayLike,
    k1: float,
    k2: float,
    pc1_dir: np.ndarray,
    pc2_dir: np.ndarray,
) -> float:
    """Compute normal curvature in a given direction via Euler's curvature formula.

    Returns the scalar normal curvature of the surface along `movement_direction`,
    given the two principal curvatures and their directions.

    k(theta) = k1 * cos^2(theta) + k2 * sin^2(theta)

    where theta is the angle between `movement_direction` and `pc1_dir`.

    This formula is only valid when `pc1_dir` and `pc2_dir` are the principal
    curvature directions and not for arbitrary orthonormal vectors.

    Reference: Weisstein, Eric W. "Euler Curvature Formula." MathWorld.
    https://mathworld.wolfram.com/EulerCurvatureFormula.html

    Args:
        movement_direction: Direction vector (will be normalized).
        k1: First principal curvature (corresponds to pc1_dir).
        k2: Second principal curvature (corresponds to pc2_dir).
        pc1_dir: First principal curvature direction (unit vector in tangent plane).
        pc2_dir: Second principal curvature direction (unit vector in tangent plane).

    Returns:
        Normal curvature in the given direction.

    Raises:
        ValueError: If pc1_dir and pc2_dir are not orthogonal, or if
            movement_direction does not lie in the plane spanned by pc1_dir
            and pc2_dir.
    """
    if not is_orthogonal(pc1_dir, pc2_dir):
        raise ValueError("The pc1_dir and pc2_dir must be orthogonal.")

    if not (is_unit_vector(pc1_dir) and is_unit_vector(pc2_dir)):
        raise ValueError("The pc1_dir and pc2_dir must be unit vectors.")

    if np.allclose(np.linalg.norm(movement_direction), 0.0):
        return 0.0

    move_hat = normalize(movement_direction)

    if not is_coplanar(pc1_dir, pc2_dir, move_hat):
        raise ValueError(
            "The movement_direction must lie in the plane of pc1_dir and pc2_dir."
        )

    cos_theta_squared = np.dot(move_hat, pc1_dir) ** 2
    sin_theta_squared = 1.0 - cos_theta_squared
    return k1 * cos_theta_squared + k2 * sin_theta_squared


def arc_length_corrected_displacement(
    du: float,
    dv: float,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    principal_curvatures: np.ndarray,
    curvature_pose_vectors: np.ndarray,
) -> tuple[float, float]:
    """Convert chord-length displacements to arc-length along each basis axis.

    Uses Euler's formula to find the normal curvature in each basis direction,
    then corrects the flat-plane displacement to the corresponding arc length.

    Args:
        du: Displacement along basis_u (chord length).
        dv: Displacement along basis_v (chord length).
        basis_u: First tangent-frame basis vector.
        basis_v: Second tangent-frame basis vector.
        principal_curvatures: Array [k1, k2] of principal curvature magnitudes.
        curvature_pose_vectors: Pose matrix whose rows [1] and [2] are the
            principal curvature directions.

    Returns:
        (arc_u, arc_v): Arc-length-corrected displacements.
    """
    k_u = directional_curvature(
        basis_u,
        principal_curvatures[0],
        principal_curvatures[1],
        curvature_pose_vectors[1],
        curvature_pose_vectors[2],
    )
    k_v = directional_curvature(
        basis_v,
        principal_curvatures[0],
        principal_curvatures[1],
        curvature_pose_vectors[1],
        curvature_pose_vectors[2],
    )
    return (
        arc_from_projection(du, k_u),
        arc_from_projection(dv, k_v),
    )


def surface_normal_naive(point_cloud, patch_radius_frac=2.5):
    """Estimate surface normal.

    This is a very simplified alternative to open3d's estimate_normals where we
    make use of several assumptions specific to our case:
    - we know which locations are neighboring locations from the camera patch
      arrangement
    - we only need the surface normal at the center of the patch

    TODO: Calculate surface normal from multiple points at different distances (tan_len
          values) and then take the average of them. Test if this improves robustness
          to raw sensor noise.

    Args:
        point_cloud: List of 3D coordinates with flags indicating whether each
            point lies on the object. Shape = [n, 4].
        patch_radius_frac: Fraction of observation size to use for SN calculation.
            Default of 2.5 means that we look half_obs_dim//2.5 to the left, right, up
            and down. With a resolution of 64x64 that would be 12 pixels. The
            calculated tan_len (in this example 12) describes the distance of pixels
            used to span up the two tangent vectors to calculate the surface normals.
            These two vectors are then used to calculate the surface normal by taking
            the cross product. If we set tan_len to a larger value, the surface normal
            is more influenced by the global shape of the patch.

    Returns:
        norm: Estimated surface normal at the center of the patch.
        valid_sn: Boolean indicating whether the surface normal was valid (True by
            default); an invalid surface normal means there were not enough points in
            the patch to make any estimate of the surface normal.
    """
    obs_dim = int(np.sqrt(point_cloud.shape[0]))
    half_obs_dim = obs_dim // 2
    center_id = half_obs_dim + obs_dim * half_obs_dim
    assert patch_radius_frac > 1, "patch_radius_frac needs to be > 1"
    tan_len = int(half_obs_dim // patch_radius_frac)
    found_surface_normal = False
    valid_sn = True
    while not found_surface_normal:
        center_id_up = half_obs_dim + obs_dim * (half_obs_dim - tan_len)
        center_id_down = half_obs_dim + obs_dim * (half_obs_dim + tan_len)

        vecup = point_cloud[center_id_up, :3] - point_cloud[center_id, :3]
        vecdown = point_cloud[center_id_down, :3] - point_cloud[center_id, :3]
        vecright = point_cloud[center_id + tan_len, :3] - point_cloud[center_id, :3]
        vecleft = point_cloud[center_id - tan_len, :3] - point_cloud[center_id, :3]

        vecup_norm = normalize(vecup)
        vecdown_norm = normalize(vecdown)
        vecright_norm = normalize(vecright)
        vecleft_norm = normalize(vecleft)

        # Check if tan_len up and right land on the object and calculate the
        # surface normal from those
        norm1, norm2 = None, None
        if (point_cloud[center_id_up, 3] > 0) and (
            point_cloud[center_id + tan_len, 3] > 0
        ):
            norm1 = -np.cross(vecup_norm, vecright_norm)
        # Check if tan_len down and left land on the object and calculate the
        # surface normal from those
        if (point_cloud[center_id_down, 3] > 0) and (
            point_cloud[center_id - tan_len, 3] > 0
        ):
            norm2 = -np.cross(vecdown_norm, vecleft_norm)

        # If any of the surrounding points were not on the object only use one sn.
        if norm1 is None:
            norm1 = norm2
        if norm2 is None:
            norm2 = norm1
        if norm1 is None and norm2 is None:
            # Try the opposite
            if (point_cloud[center_id_up, 3] > 0) and (
                point_cloud[center_id - tan_len, 3] > 0
            ):
                norm1 = np.cross(vecup_norm, vecleft_norm)

            # Check if tan_len down and left land on the object and calculate
            # the surface normal from those

            if (point_cloud[center_id_down, 3] > 0) and (
                point_cloud[center_id + tan_len, 3] > 0
            ):
                norm2 = np.cross(vecdown_norm, vecright_norm)
            if norm1 is None:
                norm1 = norm2
            if norm2 is None:
                norm2 = norm1
        if norm1 is not None:
            found_surface_normal = True
        else:
            # If none of the combinations worked, 3/4 of the points are off the object,
            # So try a smaller tan_len
            tan_len = tan_len // 2
            if tan_len < 1:
                norm1 = norm2 = np.array([0.0, 0.0, 1.0])
                valid_sn = False
                found_surface_normal = True
    norm = np.mean([norm1, norm2], axis=0)
    # Norm = np.cross(vec1_norm, vec2_norm)
    norm = normalize(norm)

    return norm, valid_sn


def surface_normal_ordinary_least_squares(
    sensor_frame_data, world_camera, center_id, neighbor_patch_frac=3.2
):
    """Extracts the surface normal direction from a noisy point cloud.

    Uses ordinary least-squares fitting with error minimization along the view
    direction.

    Args:
        sensor_frame_data: Point cloud in sensor coordinates (assumes the full
            patch is provided, i.e., no preliminary filtering of off-object points).
        world_camera: Matrix defining the sensor-to-world frame transformation.
        center_id: ID of the center point in the point cloud.
        neighbor_patch_frac: Fraction of the patch width that defines the
            local neighborhood within which to perform the least-squares fitting.

    Returns:
        surface_normal: Estimated surface normal at the center of the patch.
        valid_sn: Boolean indicating whether the surface normal was valid; defaults
            to True. An invalid surface normal means there were not enough points in
            the patch to make any estimate of the surface normal.
    """
    point_cloud = sensor_frame_data.copy()
    # Make sure that patch center is on the object
    if point_cloud[center_id, 3] > 0:
        # Define local neighborhood for least-squares fitting
        # Only use neighbors that lie on an object to extract surface normals
        neighbors_on_obj = center_neighbors(point_cloud, center_id, neighbor_patch_frac)

        # Solve linear least-square regression: X^{T}X w = X^{T}y <==> Aw = b
        x_mat = neighbors_on_obj.copy()
        x_mat[:, 2] = np.ones((neighbors_on_obj.shape[0],))
        y = neighbors_on_obj[:, 2]
        a_mat = np.matmul(x_mat.T, x_mat)
        b = np.matmul(x_mat.T, y)

        valid_sn = True
        if non_singular_mat(a_mat):
            w = np.linalg.solve(a_mat, b)

            # Compute surface normal from fitted weights and normalize it
            surface_normal = np.ones((3,))
            surface_normal[:2] = -w[:2].copy()
            surface_normal = normalize(surface_normal)

            # Make sure surface normal points upwards
            if surface_normal[2] < 0:
                surface_normal *= -1

            # Express surface normal back to world coordinate frame
            surface_normal = np.matmul(world_camera[:3, :3], surface_normal)

        else:  # Not enough point to compute
            surface_normal = np.array([0.0, 0.0, 1.0])
            valid_sn = False
            logger.debug("Warning : Singular matrix encountered in get_surface_normal!")

    # Patch center does not lie on an object
    else:
        surface_normal = np.array([0.0, 0.0, 1.0])
        valid_sn = False
        logger.debug("Warning : Patch center does not lie on an object!")

    return surface_normal, valid_sn


def surface_normal_total_least_squares(
    point_cloud_base, center_id, view_dir, neighbor_patch_frac=3.2
):
    """Extracts the surface normal direction from a noisy point-cloud.

    Uses total least-squares fitting. Error minimization is independent of the view
    direction.

    Args:
        point_cloud_base: Point cloud in world coordinates (assumes the full
            patch is provided, i.e., no preliminary filtering of off-object points).
        center_id: ID of the center point in the point cloud.
        view_dir: Viewing direction used to adjust the sign of the estimated
            surface normal.
        neighbor_patch_frac: Fraction of the patch width that defines the
            local neighborhood within which to perform the least-squares fitting.

    Returns:
        norm: Estimated surface normal at the center of the patch.
        valid_sn: Boolean indicating whether the surface normal was valid; defaults
            to True. An invalid surface normal means there were not enough points in
            the patch to make any estimate of the surface normal.
    """
    point_cloud = point_cloud_base.copy()
    # Make sure that patch center is on the object
    if point_cloud[center_id, 3] > 0:
        # Define local neighborhood for least-squares fitting
        # Only use neighbors that lie on an object to extract surface normals
        neighbors_on_obj = center_neighbors(point_cloud, center_id, neighbor_patch_frac)

        # Compute matrix M and p_mean for TLS regression
        n_points = neighbors_on_obj.shape[0]
        x_mat = neighbors_on_obj.copy()
        p_mean = 1 / n_points * np.mean(x_mat, axis=0, keepdims=True).T
        m_mat = 1 / n_points * np.matmul(x_mat.T, x_mat) - np.matmul(p_mean, p_mean.T)

        try:
            # Find eigenvector of M with the minimum eigenvalue
            eig_val, eig_vec = np.linalg.eig(m_mat)
            n_dir = eig_vec[:, np.argmin(eig_val)]
            valid_sn = True

            # Align SN with viewing direction
            if np.dot(view_dir, n_dir) < 0:
                n_dir *= -1
        except np.linalg.LinAlgError:
            n_dir = np.array([0.0, 0.0, 1.0])
            valid_sn = False
            logger.debug(
                "Warning : Non-diagonalizable matrix for surface normal estimation!"
            )

    # Patch center does not lie on an object
    else:
        n_dir = np.array([0.0, 0.0, 1.0])
        valid_sn = False
        logger.debug("Warning : Patch center does not lie on an object!")

    return n_dir, valid_sn


# Old implementation for principal curvature extraction; refer to the
# Review the get_principal_curvatures() function for the new implementation.
def curvature_at_point(point_cloud, center_id, normal):
    """Compute principal curvatures from a point cloud.

    Computes the two principal curvatures of a 2D surface and the corresponding
    principal directions.

    Args:
        point_cloud: Point cloud (2D numpy array) on which the local surface is
            approximated.
        center_id: Center point around which the local curvature is estimated.
        normal: Surface normal at the center point.

    Returns:
        k1: First principal curvature.
        k2: Second principal curvature.
        dir1: First principal direction.
        dir2: Second principal direction.
    """
    if point_cloud[center_id, 3] > 0:
        on_obj = point_cloud[:, 3] > 0
        adjusted_center_id = sum(on_obj[:center_id])
        point_cloud = point_cloud[on_obj, :3]
        # Step 1) project point coordinates onto local reference frame computed based
        # On the normal:

        # Get local reference frame (ev,fv,nv) at x:
        nv = normalize(normal)  # In case normal is not normalized

        # Find two directions ev and fv orthogonal to nv:
        e = np.zeros(3)

        if abs(nv[0]) < 0.5:
            e[0] = 1
        elif abs(nv[1]) < 0.5:
            e[1] = 1
        else:
            e[2] = 1

        f = np.cross(nv, e)
        fv = normalize(f)
        ev = np.cross(fv, nv)

        x = point_cloud[adjusted_center_id]
        dx = point_cloud - x  # Our point x is the origin of the local reference frame

        # Compute projections:
        u = np.dot(dx, ev)
        v = np.dot(dx, fv)
        w = np.dot(dx, nv)

        # Step 2) do least-squares fit to get the parameters of the quadratic form
        # Quadratic form: w=a*u*u+b*v*v+c*u*v+d*u+e*v:
        data = np.zeros((len(point_cloud), 5))

        data[:, 0] = np.multiply(u, u)
        data[:, 1] = np.multiply(v, v)
        data[:, 2] = np.multiply(u, v)
        data[:, 3] = u
        data[:, 4] = v

        beta = np.dot(np.transpose(data), w)
        a = np.dot(np.transpose(data), data)

        # Rarely, "a" can be singular, causing numpy to throw an error.
        # This appears to be caused by the surface-agent gathering observations that
        # are largely off the object, but not entirely (e.g. <25% visible), resulting
        # in a system with insufficient data to be solvable.
        if non_singular_mat(a):
            params = np.linalg.solve(a, beta)

            # Step 3) compute 1st and 2nd fundamental forms guv and buv:
            guv = np.zeros((2, 2))
            guv[0, 0] = 1 + params[3] * params[3]
            guv[0, 1] = params[3] * params[4]
            guv[1, 0] = guv[0, 1]
            guv[1, 1] = 1 + params[4] * params[4]

            buv = np.zeros((2, 2))
            buv[0, 0] = 2 * params[0]
            buv[0, 1] = params[2]
            buv[1, 0] = buv[0, 1]
            buv[1, 1] = 2 * params[1]

            # Step 4) compute the principle curvatures and directions:
            # TODO: here convex PCs are negative but I think they should be positive
            m = np.linalg.inv(guv).dot(buv)
            eigval, eigvec = np.linalg.eig(m)
            idx = eigval.argsort()[::-1]
            eigval_sorted = eigval[idx]
            eigvec_sorted = eigvec[:, idx]

            k1 = eigval_sorted[0]
            k2 = eigval_sorted[1]

            # TODO: sometimes dir1 and dir2 are not orthogonal, why?
            # Principal directions in the same coordinate frame as points:
            dir1 = eigvec_sorted[0, 0] * ev + eigvec_sorted[1, 0] * fv
            dir2 = eigvec_sorted[0, 1] * ev + eigvec_sorted[1, 1] * fv
            if get_right_hand_angle(dir1, dir2, nv) < 0:
                # Always have dir2 point to the righthand side of dir1
                dir2 = -dir2

            valid_pc = True

        else:
            k1, k2, dir1, dir2 = 0, 0, [0, 0, 0], [0, 0, 0]
            valid_pc = False
            logger.debug(
                "Warning : Singular matrix encountered in get-curvature-at-point!"
            )

    else:
        k1, k2, dir1, dir2 = 0, 0, [0, 0, 0], [0, 0, 0]
        valid_pc = False

    return k1, k2, dir1, dir2, valid_pc


def principal_curvatures(
    point_cloud_base,
    center_id,
    n_dir,
    neighbor_patch_frac=2.13,
    weighted=True,
    fit_intercept=True,
):
    """Compute principal curvatures from a point cloud.

    Computes the two principal curvatures of a 2D surface and the corresponding
    principal directions.

    Args:
        point_cloud_base: Point cloud (2D numpy array) based on which the 2D
            surface is approximated.
        center_id: Center point around which the local curvature is estimated.
        n_dir: Surface normal at the center point.
        neighbor_patch_frac: Fraction of the patch width that defines the standard
            deviation of the Gaussian distribution used to sample the weights;
            this defines a local neighborhood for principal curvature computation.
        weighted: Boolean flag that determines if regression is weighted.
            The weighting scheme is defined in :func:`weight_matrix`.
        fit_intercept: Boolean flag that determines whether to fit an intercept
            term for the regression.

    Returns:
        k1: First principal curvature.
        k2: Second principal curvature.
        dir1: First principal direction.
        dir2: Second principal direction.
    """
    point_cloud = point_cloud_base.copy()
    if point_cloud[center_id, 3] > 0:
        # Make sure point positions are expressed relative to the center point
        point_cloud[:, :3] -= point_cloud[center_id, :3]

        # Filter out points that are not on the object
        on_obj = point_cloud[:, 3] > 0
        point_cloud = point_cloud[on_obj, :3]

        # Find two directions u_dir and v_dir orthogonal to surface normal (n_dir):
        # If n_dir's z coef is 0 then normal is pointing in (x,y) plane
        u_dir = (
            np.array([1.0, 0.0, -n_dir[0] / n_dir[2]])
            if n_dir[2]
            else np.array([0.0, 0.0, 1.0])
        )
        u_dir /= np.linalg.norm(u_dir)
        v_dir = np.cross(n_dir, u_dir)
        v_dir /= np.linalg.norm(v_dir)

        # Project point coordinates onto local reference frame
        u = np.matmul(point_cloud, u_dir)
        v = np.matmul(point_cloud, v_dir)
        n = np.matmul(point_cloud, n_dir)

        # Compute the basis functions (features) for quadratic regression (only
        # Fit the intercept if fit_intercept = True)
        # Quadratic equation: n = a * u^2 + b * v^2 + c * u * v + d * u + e * v (+ d)
        n_features = 6 if fit_intercept else 5
        x_mat = np.zeros((len(point_cloud), n_features))
        x_mat[:, 0] = np.multiply(u, u)
        x_mat[:, 1] = np.multiply(v, v)
        x_mat[:, 2] = np.multiply(u, v)
        x_mat[:, 3] = u
        x_mat[:, 4] = v
        if fit_intercept:
            x_mat[:, 5] = np.ones(u.shape)

        # Quadratic regression comes down to solving a linear system: A * u = b
        # Expressions for A and b differ depending on if regression is weighted.
        if weighted:
            # Compute the weights for weighted least-square regression
            n_points = on_obj.shape[0]
            weights = weight_matrix(
                n_points, center_id, neighbor_patch_frac=neighbor_patch_frac
            )
            weights = weights[on_obj, :]  # Filter off-object points

            # Extract matrices for weighted least-squares regression.
            # A = X.T * W * X and b = X.T * W * n
            a_mat = np.matmul(x_mat.T, np.multiply(weights, x_mat))
            b = np.matmul(x_mat.T, np.multiply(weights, n[:, np.newaxis]))
        else:
            # Extract matrices for non-weighted least-squares regression.
            # A = X.T * X and b = X.T * n
            a_mat = np.matmul(x_mat.T, x_mat)
            b = np.matmul(x_mat.T, n[:, np.newaxis])

        # Rarely, "a" can be singular, causing numpy to throw an error.
        # This appears to be caused by the touch-sensor gathering observations that
        # are largely off the object, but not entirely (e.g. <25% visible), resulting
        # in a system with insufficient data to be solvable.
        if non_singular_mat(a_mat):
            # Step 2) do least-squares fit to get the parameters of the quadratic form
            params = np.linalg.solve(a_mat, b)

            # Step 3) compute 1st and 2nd fundamental forms guv and buv:
            # TODO: Extract improved surface normal estimate from fitted curve
            guv = np.zeros((2, 2))
            guv[0, 0] = 1 + params[3] * params[3]
            guv[0, 1] = params[3] * params[4]
            guv[1, 0] = guv[0, 1]
            guv[1, 1] = 1 + params[4] * params[4]

            buv = np.zeros((2, 2))
            buv[0, 0] = 2 * params[0]
            buv[0, 1] = params[2]
            buv[1, 0] = buv[0, 1]
            buv[1, 1] = 2 * params[1]

            # Step 4) compute the principle curvatures and directions:
            # TODO: here convex PCs are negative but I think they should be positive
            m = np.linalg.inv(guv).dot(buv)
            eigval, eigvec = np.linalg.eig(m)
            idx = eigval.argsort()[::-1]
            eigval_sorted = eigval[idx]
            eigvec_sorted = eigvec[:, idx]

            k1 = eigval_sorted[0]
            k2 = eigval_sorted[1]

            # TODO: sometimes dir1 and dir2 are not orthogonal, why?
            # Principal directions in the same coordinate frame as points:
            pc1_dir = eigvec_sorted[0, 0] * u_dir + eigvec_sorted[1, 0] * v_dir
            pc2_dir = eigvec_sorted[0, 1] * u_dir + eigvec_sorted[1, 1] * v_dir
            if get_right_hand_angle(pc1_dir, pc2_dir, n_dir) < 0:
                # Always have dir2 point to the righthand side of dir1
                pc2_dir = -pc2_dir
            valid_pc = True

        else:
            k1, k2, pc1_dir, pc2_dir = 0, 0, [0, 0, 0], [0, 0, 0]
            valid_pc = False
            logger.debug(
                "Warning : Singular matrix encountered in get-curvature-at-point!"
            )

    else:
        k1, k2, pc1_dir, pc2_dir = 0, 0, [0, 0, 0], [0, 0, 0]
        valid_pc = False

    return k1, k2, pc1_dir, pc2_dir, valid_pc


def center_neighbors(point_cloud, center_id, neighbor_patch_frac):
    """Get neighbors within a given neighborhood of the patch center.

    Returns:
        Locations and semantic IDs of all points within a given neighborhood of the
        patch center that lie on an object.
    """
    # Set patch center as origin of coordinate frame
    point_cloud[:, :3] -= point_cloud[center_id, :3]

    # Extract high-level parameters
    n_points = point_cloud.shape[0]
    patch_width = int(np.sqrt(n_points))
    neighbor_radius = patch_width / neighbor_patch_frac

    # Compute pixel distances to patch center (in pixel space).
    dist_to_center = pixel_dist_to_center(n_points, patch_width, center_id)

    # Use distances to define local neighborhood.
    is_neighbor = dist_to_center <= neighbor_radius
    neighbors_idx = is_neighbor.reshape((n_points,))
    neighbors = point_cloud[neighbors_idx, :]

    # Filter out points that do not lie on an object
    return neighbors[neighbors[:, 3] > 0, :3]


def weight_matrix(n_points, center_id, neighbor_patch_frac=2.13):
    """Extract individual pixel weights for least-squares fitting.

    Each pixel weight is sampled from a Gaussian distribution based on its distance
    to the patch center.

    Args:
        n_points: Total number of points in the full RGB-D square patch.
        center_id: ID of the center point in the point cloud.
        neighbor_patch_frac: Fraction of the patch width that defines the standard
            deviation of the Gaussian distribution used to sample the weights.

    Returns:
        Diagonal weight matrix of shape (n_points, 1).
    """
    # Extract center and all its neighbors
    patch_width = int(np.sqrt(n_points))
    sigma = patch_width / neighbor_patch_frac

    # Compute pixel distances to patch center (in pixel space).
    dist_to_center = pixel_dist_to_center(n_points, patch_width, center_id)

    # Compute weight matrix based on those distances
    w_coefs = (
        1.0
        / (np.sqrt(2 * np.pi) * sigma)
        * np.exp(-np.square(dist_to_center) / (2 * sigma**2))
    )
    w_diag = w_coefs.reshape((n_points, 1))
    w_diag /= np.sum(w_diag)
    return w_diag


def pixel_dist_to_center(n_points, patch_width, center_id):
    """Extract the relative distance of each pixel to the patch center (in pixel space).

    Args:
        n_points: Total number of points in the patch.
        patch_width: Width of the square patch.
        center_id: ID of the patch center.

    Returns:
        Relative distance of each pixel to the patch center (in pixel space).
    """
    # Get coordinates (in pixel space) of all pixels in the patch
    point_idx = np.arange(n_points).reshape(patch_width, patch_width)
    x, y = np.meshgrid(np.arange(patch_width), np.arange(patch_width))
    pos = np.dstack((x, y))

    # Compute relative distance to patch center
    pos_center = pos[point_idx == center_id]
    return np.linalg.norm(pos - pos_center, axis=2)


def point_pair_features(pos_i, pos_j, normal_i, normal_j):
    """Return point pair features between two points.

    Args:
        pos_i: Location of point 1.
        pos_j: Location of point 2.
        normal_i: Surface normal of point 1.
        normal_j: Surface normal of point 2.

    Returns:
        Point pair features.
    """
    pseudo = pos_j - pos_i
    return torch.stack(
        [
            pseudo.norm(p=2),
            get_angle_torch(normal_i, pseudo),
            get_angle_torch(normal_j, pseudo),
            get_angle_torch(normal_i, normal_j),
        ]
    )


def scale_clip(to_scale, clip):
    """Clip values into a range and scale with the square root.

    This can be used to bring Gaussian and mean curvatures into a reasonable range
    and remove outliers, which makes it easier to handle noise.
    The sign is preserved before applying the square root.

    Args:
        to_scale: Array where each element should be scaled.
        clip: Range to which the array values should be clipped.

    Returns:
        Scaled values of the array.
    """
    to_scale = np.clip(to_scale, -clip, clip)
    negative = to_scale < 0
    scaled = np.sqrt(np.abs(to_scale))
    if len(scaled.shape) == 0:  # Just a scalar value
        if negative:
            scaled = scaled * -1
    else:  # An array
        scaled[negative] = scaled[negative] * -1
    return scaled


def log_sign(to_scale):
    """Apply symlog to the input array, preserving sign.

    This implementation ensures that the sign of the input values is preserved and
    avoids extreme outputs when values are close to 0.

    Args:
        to_scale: Array to scale.

    Returns:
        Scaled values of the array.
    """
    to_scale = np.asarray(to_scale)
    sign = np.sign(to_scale)  # Preserve sign
    abs_vals = np.abs(to_scale)
    log_vals = np.log(abs_vals + 1)  # Avoid extreme values around 0
    return sign * log_vals
