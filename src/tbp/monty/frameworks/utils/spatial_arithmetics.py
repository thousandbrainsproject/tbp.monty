# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import logging
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def rotations_to_quats(rotations, invert=False):
    # We get euler rotations from feature LM
    if rotations.ndim == 2:
        quats = euler_to_quats(rotations, invert=invert)
    # We get rotation matrices from evidence LM
    elif rotations.ndim == 3:
        quats = rot_mats_to_quats(rotations, invert=invert)
    else:
        raise ValueError("Invalid rotation matrix shape")
    return quats


def rot_mats_to_quats(rot_mats, invert=False):
    """Convert rotation matrices to quaternions.

    Args:
        rot_mats: Rotation matrices
        invert: Whether to invert the rotation. Defaults to False.

    Returns:
        Quaternions
    """
    quats = []
    for rotation_matrix in rot_mats:
        rotation = Rotation.from_matrix(rotation_matrix)
        if invert:
            rotation = rotation.inv()
        quats.append(rotation.as_quat())
    return quats


def euler_to_quats(euler_rots, invert=False):
    """Convert euler rotations to rotation matrices.

    Args:
        euler_rots: Euler rotations
        invert: Whether to invert the rotation. Defaults to False.

    Returns:
        Quaternions
    """
    quats = []
    for euler_rot in euler_rots:
        rot_mat = Rotation.from_euler("xyz", euler_rot, degrees=True)
        if invert:
            rot_mat = rot_mat.inv()
        quats.append(rot_mat.as_quat())
    return quats


def get_angle(vec1, vec2):
    """Get angle between two vectors.

    NOTE: for efficiency reasons we assume vec1 and vec2 are already
    normalized (which is the case for point normals and curvature
    directions).

    Args:
        vec1: Vector 1
        vec2: Vector 2

    Returns:
        angle in radians
    """
    # unit_vector_1 = vec1 / np.linalg.norm(vec1)
    # unit_vector_2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(vec1, vec2)
    angle = np.arccos(np.clip(dot_product, -1, 1))
    return angle


def get_angle_beefed_up(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'.

    If one of the vectors is undefined, return arbitrarily large distance

    If one of the vectors is the zero vector, return arbitrarily large distance

    Also enforces that vectors are unit vectors (therefore less efficient than
    the standard get_angle)

    >>> angle_between_vecs((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between_vecs((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between_vecs((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    if v1 is None or v2 is None:
        return np.inf

    if np.all(v1 == 0) or np.all(v2 == 0):
        return np.inf

    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)

    result = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    if result is np.nan:
        result = np.inf

    assert result >= 0, f"Angle between is negative : {result}"

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_angles_for_all_hypotheses(hyp_f, query_f):
    """Get all angles for hypotheses and their neighbors at once.

    hyp_f shape = (num_hyp, num_nn, 3)
    query_f shape = (num_hyp, 3)
        for each hypothesis we want to get num_nn angles.
    return shape = (num_hyp, num_nn)

    Args:
        hyp_f (num_hyp, num_nn, 3): ?
        query_f (num_hyp, 3): ?

    Returns:
        ?
    """
    dot_product = np.einsum("ijk,ik->ij", hyp_f, query_f)
    angle = np.arccos(np.clip(dot_product, -1, 1))
    return angle


def get_angle_torch(v1, v2):
    """Get angle between two torch vectors.

    Args:
        v1: Vector 1
        v2: Vector 2

    Returns:
        angle in radians
    """
    return torch.atan2(torch.cross(v1, v2).norm(p=2), (v1 * v2).sum())


def check_orthonormal(matrix):
    is_orthogonal = np.mean(np.abs(np.linalg.inv(matrix) - matrix.T)) < 0.01
    if not is_orthogonal:
        logging.debug(
            "not orthogonal. Error: "
            f"{np.mean(np.abs(np.linalg.inv(matrix) - matrix.T))}"
        )
    is_normal = np.mean(np.abs(np.linalg.norm(matrix, axis=1) - [1, 1, 1])) < 0.01
    if not is_normal:
        logging.debug(
            "not normal. Error: "
            f"{np.mean(np.abs(np.linalg.norm(matrix, axis=1) - [1, 1, 1]))}"
        )
    return is_orthogonal and is_normal


def align_orthonormal_vectors(m1, m2, as_scipy=True):
    """Calculate the rotation that aligns two sets of orthonormal vectors.

    Args:
        m1: First set of orthonormal vectors.
        m2: Second set of orthonormal vectors to align with.
        as_scipy: Whether to return a scipy rotation object or a rotation matrix.
            Defaults to True.

    Returns:
        ?
    """
    # assert check_orthonormal(m1), "m1 is not orthonormal"
    # assert check_orthonormal(m2), "m2 is not orthonormal"
    # to get rotation matrix between m1 and m2 calculate
    # m1*m2.T (can do m2.T instead of m2.inv because vectors
    # are orthogonal). Since vectors here are rows instead
    # of columns we apply T to m1 instead of m2.
    rot_mat = np.matmul(m1.T, m2)
    if as_scipy:
        rotation = Rotation.from_matrix(rot_mat)
        # Check that recovered rotation actually works
        # Error will be there if orthonormal vectors are mirrored
        # (right hand vs, left hand side). This is just left in here for
        # backward compatibility. The Evidence LM doesn't use the error anymore
        # and mirrored vectors will be corrected by the sensor module.
        error = np.mean(np.abs(rotation.inv().apply(m1) - m2))
        return rotation, error
    else:
        return rot_mat, None


def align_multiple_orthonormal_vectors(ms1, ms2, as_scipy=True):
    """Calculate rotations between multiple orthonormal vector sets.

    Args:
        ms1: multiple orthonormal vectors. shape = (N, 3, 3)
        ms2: orthonormal vectors to align with. shape = (3, 3)
        as_scipy: Whether to return a list of N scipy.Rotation objects or
            a np.array of rotation matrices (N, 3, 3).

    Returns:
        List of N Rotations that align ms2 with each element in ms1.
    """
    transpose_mts1 = np.transpose(ms1, axes=[0, 2, 1])
    rot_mats = np.matmul(transpose_mts1, ms2)
    if as_scipy:
        all_rotations = []
        for rot_mat in rot_mats:
            all_rotations.append(Rotation.from_matrix(rot_mat))
        return all_rotations
    else:
        return rot_mats


def get_right_hand_angle(v1, v2, pn):
    # some numpy bug (https://github.com/microsoft/pylance-release/issues/3277)
    # cp = lambda v1, v2: np.cross(v1, v2)
    # a = np.dot(cp(v1, v2), pn)
    a = np.dot(np.cross(v1, v2), pn)
    b = np.dot(v1, v2)
    rha = np.arctan2(a, b)
    return rha


def non_singular_mat(a):
    """Return True if a matrix is non-singular, i.e. can be inverted.

    Uses the condition number of the matrix, which will approach
    a very large value, given by (1 / sys.float_info.epsilon)
    (where epsilon is the smallest possible floating-point difference)
    """
    if np.linalg.cond(a) < 1 / sys.float_info.epsilon:
        return True
    else:
        return False


def get_more_directions_in_plane(vecs, n_poses):
    """Get a list of unit vectors, evenly spaced in a plane orthogonal to vecs[0].

    This is used to sample possible poses orthogonal to the point normal when the
    curvature directions are undefined (like on a flat surface).

    Args:
        vecs: Vector to get more directions in plane for
        n_poses: Number of poses to get

    Returns:
        list: List of vectors evenly spaced in a plane orthogonal to vecs[0]
    """
    new_vecs = [vecs]
    angles = np.linspace(0, 2 * np.pi, n_poses + 1)
    # First and last angle are same as original vec (0 & 1)
    for angle in angles[:-1]:
        new_vec = np.cos(angle) * vecs[1] + np.sin(angle) * vecs[2]
        new_vec2 = np.cross(vecs[0], new_vec)
        new_vecs.append([vecs[0], new_vec, new_vec2])
    return new_vecs[1:]


def get_unique_rotations(poses, similarity_th, get_reverse_r=True):
    """Get unique scipy.Rotations out of a list, given a similarity threshold.

    Args:
        poses: List of poses to get unique rotations from
        similarity_th: Similarity threshold
        get_reverse_r: Whether to get the reverse rotation. Defaults to True.

    Returns:
        euler_poses: Unique euler poses
        r_poses: Unique rotations corresponding to euler_poses
    """
    unique_poses = []
    euler_poses = []
    r_poses = []
    for path_poses in poses:
        for pose in path_poses:
            if pose_is_new(unique_poses, pose, similarity_th):
                unique_poses.append(pose)
                if get_reverse_r:
                    r_pose = pose.inv()
                    euler_pose = pose.inv().as_euler("xyz", degrees=True)
                    euler_pose = np.round(euler_pose, 3) % 360
                else:
                    r_pose = pose
                    euler_pose = pose.as_euler("xyz", degrees=True)
                    euler_pose = np.round(euler_pose, 3) % 360
                r_poses.append(r_pose)
                euler_poses.append(euler_pose)
    return euler_poses, r_poses


def pose_is_new(all_poses, new_pose, similarity_th):
    """Check if a pose is different from a list of poses.

    Use the magnitude of the difference between quaternions as a measure for
    similarity and check that it is below pose_similarity_threshold.

    Returns:
        bool: True if the pose is new, False otherwise
    """
    for pose in all_poses:
        d = new_pose * pose.inv()
        if d.magnitude() < similarity_th:
            return False
    return True


def rotate_pose_dependent_features(features, ref_frame_rots):
    """Rotate pose_vectors given a list of rotation matrices.

    Args:
        features: dict of features with pose vectors to rotate.
            pose vectors have shape (3, 3)
        ref_frame_rots: Rotation matrices to rotate pose features by. Can either be
            - A single scipy rotation (as used in FeatureGraphLM)
            - An array of rotation matrices of shape (N, 3, 3) or (3, 3) (as used in
            EvidenceGraphLM).

    Returns:
        dict: Original features but with the pose_vectors rotated. If multiple
            rotations were given, pose_vectors entry will now contain multiple
            entries of shape (N, 3, 3).
    """
    pose_transformed_features = copy.deepcopy(features)
    old_pv = pose_transformed_features["pose_vectors"]
    assert old_pv.shape == (
        3,
        3,
    ), f"pose_vectors in features need to be 3x3 matrices."
    if isinstance(ref_frame_rots, Rotation):
        rotated_pv = ref_frame_rots.apply(old_pv)
    else:
        # Transpose pose vectors so each vector is a column (otherwise .dot matmul
        # produces slightly different results)
        rotated_pv = ref_frame_rots.dot(old_pv.T)
        # Transpose last two axies so each pose vector is a row again
        rotated_pv = rotated_pv.transpose((0, 2, 1))
    pose_transformed_features["pose_vectors"] = rotated_pv
    return pose_transformed_features


def rotate_multiple_pose_dependent_features(features, ref_frame_rot):
    """Rotate point normal and curv dirs given a rotation matrix.

    Args:
        features: dict of features with pose vectors to rotate.
            Pose vectors have shape (N, 9)
        ref_frame_rot: scipy rotation to rotate pose vectors with.

    Returns:
        dict: Features with rotated pose vectors
    """
    pose_vecs = features["pose_vectors"]
    num_pose_vecs = pose_vecs.shape[0]
    assert pose_vecs.shape[-1] == 9, "pose_vectors in features should be flattened."
    # Reshape to (num_pose_vecs * 3, 3) to be able to rotate all vectors at once
    pose_vecs = pose_vecs.reshape((-1, 3))
    # rotate all vectors by ref_frame_rot
    rotated_pose_vecs = ref_frame_rot.apply(pose_vecs)
    # shape back to original shape (num_pose_vecs, 9)
    rotated_pose_vecs = rotated_pose_vecs.reshape((num_pose_vecs, 9))
    features["pose_vectors"] = rotated_pose_vecs
    return features


def apply_rf_transform_to_points(
    locations,
    features,
    location_rel_model,
    object_location_rel_body,
    object_rotation,
    object_scale=1,  # noqa: ARG001
):
    """Apply location and rotation transform to locations and features.

    These transforms tell us how to transform new observations into the existing
    model reference frame. They are calculated from the detected object pose.

    Args:
        locations: Locations to transform (in body reference frame). Shape (N, 3)
        features: Features to transform (in body reference frame). Shape (N, F)
        location_rel_model: Detected location of the sensor on the object (object
            reference frame).
        object_location_rel_body: Location of the sensor in the body reference
            frame.
        object_rotation: Rotation of the object in the world relative to the
            learned model of the object. Expresses how the object model needs to be
            rotated to be consistent with the observations. To transfor the observed
            locations (rel. body) into the models reference frame, the inverse of
            this rotation is applied.
        object_scale: Scale of the object relative to the model. Not used yet.

    Note:
        Function can also be used in different contexts besides transforming points
        from body to object centric reference frame.

    Returns:
        transformed_locations: Transformed locations
        features: Transformed features
    """
    # Two reference points that should be the same
    ref_point1 = np.array(location_rel_model)
    ref_point2 = np.array(object_location_rel_body)
    # Center new locations around 0 so that rotation doesn't displace them
    zero_centered_locs = locations - ref_point2
    # Apply detected rotation to new locations
    rotated_locs = object_rotation.inv().apply(zero_centered_locs)
    # Apply the offset between the two reference points
    ref_point_offset = ref_point1 - ref_point2
    location_offset = rotated_locs + ref_point_offset
    # Undo the 0 centering to bring back into model reference frame
    transformed_locations = location_offset + ref_point2
    features = rotate_multiple_pose_dependent_features(features, object_rotation.inv())
    return transformed_locations, features
