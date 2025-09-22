# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np
import torch

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_angle_torch,
    get_right_hand_angle,
    non_singular_mat,
)

logger = logging.getLogger(__name__)


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
        point_cloud: list of 3d coordinates and whether the points are on the
            object or not. shape = [n, 4]
        patch_radius_frac: Fraction of observation size to use for SN calculation.
            Default of 2.5 means that we look half_obs_dim//2.5 to the left, right, up
            and down. With a resolution of 64x64 that would be 12 pixels. The
            calculated tan_len (in this example 12) describes the distance of pixels
            used to span up the two tangent vectors to calculate the surface normals.
            These two vectors are then used to calculate the surface normal by taking
            the cross product. If we set tan_len to a larger value the surface normal
            is more influenced by the global shape of the patch.

    Returns:
        norm: Estimated surface normal at center of patch
        valid_sn: Boolean for whether the surface normal was valid or not (True by
            default); an invalid surface normal means there were not enough points in
            the patch to make any estimate of the surface normal
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

        vecup_norm = vecup / np.linalg.norm(vecup)
        vecdown_norm = vecdown / np.linalg.norm(vecdown)
        vecright_norm = vecright / np.linalg.norm(vecright)
        vecleft_norm = vecleft / np.linalg.norm(vecleft)

        # check if tan_len up and right end up on the object and calculate sn from those
        norm1, norm2 = None, None
        if (point_cloud[center_id_up, 3] > 0) and (
            point_cloud[center_id + tan_len, 3] > 0
        ):
            norm1 = -np.cross(vecup_norm, vecright_norm)
        # check if tan_len down and left end up on the object and calculate
        # sn from those
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

            # check if tan_len down and left end up on the object and calculate
            # sn from those

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
            # if none of the combinations worked then 3/4 points are off the object
            # -> try a smaller tan_len
            tan_len = tan_len // 2
            if tan_len < 1:
                norm1 = norm2 = [0, 0, 1]
                valid_sn = False
                found_surface_normal = True
    norm = np.mean([norm1, norm2], axis=0)
    # norm = np.cross(vec1_norm, vec2_norm)
    norm = norm / np.linalg.norm(norm)

    return norm, valid_sn


def surface_normal_ordinary_least_squares(
    sensor_frame_data, world_camera, center_id, neighbor_patch_frac=3.2
):
    """Extracts the surface normal direction from a noisy point-cloud.

    Uses ordinary least-square fitting with error minimization along the view
    direction.

    Args:
        sensor_frame_data: point-cloud in sensor coordinates (assumes full
            patch is provided i.e. no preliminary filtering of off-object points).
        world_camera: matrix defining sensor-to-world frame transformation.
        center_id: id of the center point in point_cloud.
        neighbor_patch_frac: fraction of the patch width that defines the
            local neighborhood within which to perform the least-squares fitting.

    Returns:
        surface_normal: Estimated surface normal at center of patch
        valid_sn: Boolean for whether the surface normal was valid or not. Defaults
            to True. An invalid surface normal means there were not enough points in
            the patch to make any estimate of the surface normal
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
            surface_normal = surface_normal / np.linalg.norm(surface_normal)

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

    Uses total least-square fitting. Error minimization is independent of view
    direction.

    Args:
        point_cloud_base: point-cloud in world coordinates (assumes full
            patch is provided i.e. no preliminary filtering of off-object points).
        center_id: id of the center point in point_cloud.
        view_dir: viewing direction used to adjust the sign of the estimated
            surface normal.
        neighbor_patch_frac: fraction of the patch width that defines the
            local neighborhood within which to perform the least-squares fitting.

    Returns:
        norm: Estimate surface normal at center of patch
        valid_sn: Boolean for whether the surface normal was valid or not. Defaults
            to True. An invalid surface normal means there were not enough points in
            the patch to make any estimate of the surface normal
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
            # find eigenvector of M with min eigenvalue
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


# Old implementation for principal curvature extraction. Refer to
# get_principal_curvatures() function for new implementation.
def curvature_at_point(point_cloud, center_id, normal):
    """Compute principal curvatures from point cloud.

    Computes the two principal curvatures of a 2D surface and corresponding
    principal directions

    Args:
        point_cloud: point cloud (2d numpy array) based on which the 2D
            surface is approximated
        center_id: center point around which the local curvature is
            estimated
        normal: surface normal at the center point

    Returns:
        k1:     first principal curvature
        k2:     second principal curvature
        dir1:   first principal direction
        dir2:   second principal direction
    """
    if point_cloud[center_id, 3] > 0:
        on_obj = point_cloud[:, 3] > 0
        adjusted_center_id = sum(on_obj[:center_id])
        point_cloud = point_cloud[on_obj, :3]
        # Step 1) project point coordinates onto local reference frame computed based
        # on the normal:

        # get local reference frame (ev,fv,nv) at x:
        nv = normal / np.linalg.norm(normal)  # in case normal is not normalized

        # find two directions ev and fv orthogonal to nv:
        e = np.zeros(3)

        if abs(nv[0]) < 0.5:
            e[0] = 1
        elif abs(nv[1]) < 0.5:
            e[1] = 1
        else:
            e[2] = 1

        f = np.cross(nv, e)
        fv = f / np.linalg.norm(f)
        ev = np.cross(fv, nv)

        x = point_cloud[adjusted_center_id]
        dx = point_cloud - x  # our point x is the origin of the local reference frame

        # compute projections:
        u = np.dot(dx, ev)
        v = np.dot(dx, fv)
        w = np.dot(dx, nv)

        # Step 2) do least-squares fit to get the parameters of the quadratic form
        # w=a*u*u+b*v*v+c*u*v+d*u+e*v:
        data = np.zeros((len(point_cloud), 5))

        data[:, 0] = np.multiply(u, u)
        data[:, 1] = np.multiply(v, v)
        data[:, 2] = np.multiply(u, v)
        data[:, 3] = u
        data[:, 4] = v

        beta = np.dot(np.transpose(data), w)
        a = np.dot(np.transpose(data), data)

        # Rarely, "a" can be singular, causing numpy to throw an error; appears
        # to be caused by surface-agent gathering observations that are largely off the
        # object, but not entirely (e.g. <25% visible), resulting in a system
        # with insufficient data to be solvable
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
            # principal directions in the same coordinate frame as points:
            dir1 = eigvec_sorted[0, 0] * ev + eigvec_sorted[1, 0] * fv
            dir2 = eigvec_sorted[0, 1] * ev + eigvec_sorted[1, 1] * fv
            if get_right_hand_angle(dir1, dir2, nv) < 0:
                # always have dir2 point to the righthand side of dir1
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
    """Compute principal curvatures from point cloud.

    Computes the two principal curvatures of a 2D surface and corresponding
    principal directions

    Args:
        point_cloud_base: point cloud (2d numpy array) based on which the 2D
            surface is approximated
        center_id: center point around which the local curvature is
            estimated
        n_dir: surface normal at the center point
        neighbor_patch_frac: fraction of the patch width that defines the std
            of the gaussian distribution used to sample the weights. Defines a
            local neighborhood for principal curvature computation.
        weighted: boolean flag that determines if regression is weighted or not.
            Weighting scheme is defined in get_weight_matrix.
        fit_intercept: boolean flag that determines whether to fit an intercept
                term for the regression.

    Returns:
        k1:     first principal curvature
        k2:     second principal curvature
        dir1:   first principal direction
        dir2:   second principal direction
    """
    point_cloud = point_cloud_base.copy()
    if point_cloud[center_id, 3] > 0:
        # Make sure point positions are expressed relative to the center point
        point_cloud[:, :3] -= point_cloud[center_id, :3]

        # Filter out points that are not on the object
        on_obj = point_cloud[:, 3] > 0
        point_cloud = point_cloud[on_obj, :3]

        # find two directions u_dir and v_dir orthogonal to surface normal (n_dir):
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
        # fit the intercept if fit_intercept = True)
        # n = a * u^2 + b * v^2 + c * u * v + d * u + e * v (+ d)
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

        # Rarely, "a" can be singular, causing numpy to throw an error; appears
        # to be caused by touch-sensor gathering observations that are largely off the
        # object, but not entirely (e.g. <25% visible), resulting in a system
        # with insufficient data to be solvable
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
            # principal directions in the same coordinate frame as points:
            pc1_dir = eigvec_sorted[0, 0] * u_dir + eigvec_sorted[1, 0] * v_dir
            pc2_dir = eigvec_sorted[0, 1] * u_dir + eigvec_sorted[1, 1] * v_dir
            if get_right_hand_angle(pc1_dir, pc2_dir, n_dir) < 0:
                # always have dir2 point to the righthand side of dir1
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
        Locations and semantic id of all points within a given neighborhood of the
        patch center which lie on an object.
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
    neighbors_on_obj = neighbors[neighbors[:, 3] > 0, :3]
    return neighbors_on_obj


def weight_matrix(n_points, center_id, neighbor_patch_frac=2.13):
    """Extracts individual pixel weights for least-squares fitting.

    Weight for each pixel is sampled from a gaussian distribution based on its distance
    to the patch center.

    Args:
        n_points: total number of points in the full RGB-D square patch.
        center_id: id of the center point in point_cloud.
        neighbor_patch_frac: fraction of the patch width that defines the std
            of the gaussian distribution used to sample the weights.

    Returns:
        w_diag
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
    """Extracts the relative distance of each pixel to patch center (in pixel space).

    Returns:
        Relative distance of each pixel to patch center (in pixel space)
    """
    # Get coordinates (in pixel space) of all pixels in the patch
    point_idx = np.arange(n_points).reshape(patch_width, patch_width)
    x, y = np.meshgrid(np.arange(patch_width), np.arange(patch_width))
    pos = np.dstack((x, y))

    # Compute relative distance to patch center
    pos_center = pos[point_idx == center_id]
    dist_to_center = np.linalg.norm(pos - pos_center, axis=2)
    return dist_to_center


def point_pair_features(pos_i, pos_j, normal_i, normal_j):
    """Get point pair features between two points.

    Args:
        pos_i: Location of point 1
        pos_j: Location of point 2
        normal_i: Surface normal of point 1
        normal_j: Surface normal of point 2

    Returns:
        Point pair feature
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
    """Clip values into range and scale with sqrt.

    This can be used to get gaussian and mean curvatures into
    a reasonable range and remove outliers. Makes it easier to deal
    with noise.
    Preserves sign before applying sqrt.

    Args:
        to_scale: array where each element should be scaled.
        clip: range to which the array values should be clipped.

    Returns:
        scaled values of array.
    """
    to_scale = np.clip(to_scale, -clip, clip)
    negative = to_scale < 0
    scaled = np.sqrt(np.abs(to_scale))
    if len(scaled.shape) == 0:  # just a scalar value
        if negative:
            scaled = scaled * -1
    else:  # an array
        scaled[negative] = scaled[negative] * -1
    return scaled


def log_sign(to_scale):
    """Apply symlog to input array, preserving sign.

    This implementation makes sure to preserve the sign of the input values and to
    avoid extreme outputs when values are close to 0.

    Args:
        to_scale: array to scale.

    Returns:
        Scaled values of array.
    """
    to_scale = np.asarray(to_scale)
    sign = np.sign(to_scale)  # preserve sign
    abs_vals = np.abs(to_scale)
    log_vals = np.log(abs_vals + 1)  # avoid extreme values around 0
    scaled_vals = sign * log_vals
    return scaled_vals
