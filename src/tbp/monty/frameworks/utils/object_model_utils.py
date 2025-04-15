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
    get_angle,
    get_right_hand_angle,
)


class NumpyGraph:
    """Alternative way to represent graphs without using torch.

    Speeds up runtime significantly.
    """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def torch_graph_to_numpy(torch_graph):
    """Turn torch geometric data structure into dict with numpy arrays.

    Args:
        torch_graph: Torch geometric data structure.

    Returns:
        NumpyGraph.
    """
    numpy_graph = dict()
    for key in list(torch_graph.keys):
        if isinstance(torch_graph[key], torch.Tensor):
            numpy_graph[key] = np.array(torch_graph[key])
        else:
            numpy_graph[key] = torch_graph[key]
    return NumpyGraph(numpy_graph)


def already_in_list(
    existing_points, new_point, features, clean_ids, query_id, graph_delta_thresholds
):
    """Check if a given point is already in a list of points.

    Args:
        existing_points: List of x,y,z locations
        new_point: new location
        features: all features (both existing and candidate points)
        clean_ids: indices (w.r.t "features") that have been accepted into the graph
            and are compared to
        query_id: index (w.r.t "features") that is currently being considered
        graph_delta_thresholds: Dictionary of thresholds used to determine whether a
            point should be considered sufficiently different so as to be included in
            the graph

    Returns:
        bool: Whether the point is already in the list
    """
    in_list = False

    # Check first for nearness of the points in Euclidean space (vectorized)
    matching_dist_points = (
        np.linalg.norm((existing_points - new_point), axis=1)
        < graph_delta_thresholds["distance"]
    )

    assert (
        "on_object" not in graph_delta_thresholds.keys()
    ), "Don't pass feature-change SM delta_thresholds for graph_delta_thresholds"

    # Iterate through old graph points that do match distance-wise, performing
    # additional checks
    for match_idx in np.where(matching_dist_points)[0]:
        # Convert to the indexing that is used for the features (provides correct
        # index even if we've excluded some previous points from being added to the
        # graph)
        feature_idx = clean_ids[match_idx]

        # Go through all the feature comparisons; if the point is redundant
        # according to all of these comparisons, then the point can be considered
        # to be in the list
        redundant_point = True
        # TODO: What to do when a feature is received that is not in
        # graph_delta_thresholds? Currently it will not be considered when looking at
        # redundancy of the point.
        for feature in graph_delta_thresholds.keys():
            if feature in features.keys():
                if feature == "hsv":
                    # Only consider hue, not saturation and lightness at this point
                    match_hue = features[feature][feature_idx][0]
                    current_hue = features[feature][query_id][0]

                    assert np.all(
                        np.array(graph_delta_thresholds[feature][1:]) == 1
                    ), "Only considering hue"

                    hue_d = min(
                        abs(current_hue - match_hue),
                        1 - abs(current_hue - match_hue),
                    )  # Use circular difference to reflect angular nature of hue

                    if hue_d > graph_delta_thresholds[feature][0]:
                        logging.debug(
                            f"Interesting point because of {feature} : {hue_d}"
                        )
                        redundant_point = False
                        break
                elif feature == "pose_vectors":
                    # TODO S: currently just looking at first pose vector (pn)
                    angle_between = get_angle(
                        features["pose_vectors"][feature_idx][:3],
                        features["pose_vectors"][query_id][:3],
                    )
                    if angle_between > graph_delta_thresholds["pose_vectors"][0]:
                        redundant_point = False
                        break

                elif feature == "distance":
                    pass
                    # Already dealt with in vectorized form

                else:
                    delta_change = np.abs(
                        features[feature][feature_idx] - features[feature][query_id]
                    )
                    if len(delta_change.shape) > 0:
                        for i, dc in enumerate(delta_change):
                            if dc > graph_delta_thresholds[feature][i]:
                                logging.debug(
                                    f"Interesting point because of {feature} : {dc}"
                                )
                                redundant_point = False
                                break
                    else:
                        if delta_change > graph_delta_thresholds[feature]:
                            logging.debug(
                                "Interesting point because of "
                                f"{feature} : {delta_change}"
                            )
                            redundant_point = False
                            break

        if redundant_point:
            # Considered in the list if all features above (incl. distance) are
            # sufficiently similar so as to be redundant
            in_list = True
            break
        # Otherwise, the point *may* be interesting - needs to be interesting
        # when compared to *all* distance-matched points however; i.e. as long as
        # it is redundant with one distance-matched point, then it is in the list

    return in_list


def remove_close_points(point_cloud, features, graph_delta_thresholds, old_graph_index):
    """Remove points from a point cloud unless sufficiently far away.

    Points are removed unless sufficiently far away either by Euclidean distance, or
    feature-space.

    Args:
        point_cloud: List of 3D points
        features: ?
        graph_delta_thresholds: dictionary of thresholds; if the L-2 distance
            between the locations of two observations (or other feature-distance
            measure) is below all of the given thresholds, then a point will be
            considered insufficiently interesting to be added
        old_graph_index: If the graph is not new, the index associated with the
            final point in the old graph; we will skip this when checking for sameness,
            as they will already have been compared in the past to one-another, saving
            computation.

    Returns:
        List of 3D points that are sufficiently novel w.r.t one-another, along
        with their associated indices.
    """
    clean_ids = list(
        range(old_graph_index)
    )  # Create list of indices that are already clean / have been checked

    new_points = list(point_cloud[:old_graph_index])

    if graph_delta_thresholds is None:
        # Asign mutable default value
        graph_delta_thresholds = dict(
            distance=0.001,
        )
    if "pose_vectors" not in graph_delta_thresholds.keys():
        # By default, we will still consider a nearby point as new if the difference
        # in point-normals suggests it is on the other side of an object
        # NOTE: currently not looking at curvature directions/second pose vector
        graph_delta_thresholds["pose_vectors"] = [np.pi / 2, np.pi * 2, np.pi * 2]

    for i, p in enumerate(point_cloud[old_graph_index:], start=old_graph_index):
        if len(new_points) == 0:
            # Skip linalg operation in already_in_list for the first point the first
            # time a graph is learned:
            new_points.append(p)
            clean_ids.append(i)

        elif not already_in_list(
            new_points, p, features, clean_ids, i, graph_delta_thresholds
        ):
            new_points.append(p)
            clean_ids.append(i)

    return np.array(new_points), clean_ids


def increment_sparse_tensor_by_count(old_tensor, indices):
    # If an index is in indices multiple times, it will be counted multiple times
    unique_indices, counts = np.unique(indices, axis=0, return_counts=True)
    # If indices don't have dimensionality of grid, add column of zeros.
    # This is the case when adding obs counts since 4th dimension is just the count
    # and is not included in the indices.
    if unique_indices.shape[-1] != old_tensor.ndim:
        zeros_column = np.zeros((unique_indices.shape[0], 1))
        unique_indices = np.hstack((unique_indices, zeros_column))
    # Build a new sparse tensor with the new indices and values
    new_indices = torch.tensor(unique_indices.T, dtype=torch.long)
    new_values = torch.tensor(counts, dtype=torch.int64)
    # Return coalesced tensor (make representation more efficient)
    # Mostly removes duplicate entries which we should never have here but also allows
    # us to call .indices() instead of needing to use ._indices(). May remove this
    # if time overhead is too high but is pretty small atm (5.1e-5s).
    new_sparse_tensor = torch.sparse_coo_tensor(
        new_indices, new_values, old_tensor.shape
    ).coalesce()
    # Add the new sparse tensor to the old one
    new_tensor = old_tensor + new_sparse_tensor
    return new_tensor


def get_values_from_dense_last_dim(tensor, index_3d):
    """Get values from 4d tensor at indices in last dimension.

    This function assumes that the entries in the last dimension are dense.
    This is the case in all our sparse tensors where the first 3 dimensions
    represent the 3d location (sparse) and the 4th represents values at this
    location (dense).

    Returns:
        List of values.
    """
    last_dim_size = tensor.shape[-1]
    values = [
        float(tensor[index_3d[0], index_3d[1], index_3d[2], n])
        for n in range(last_dim_size)
    ]
    return values


def expand_index_dims(indices_3d, last_dim_size):
    """Expand 3d indices to 4d indices by adding a 4th dimension with size.

    Args:
        indices_3d: 3d indices that should be comverted to 4d
        last_dim_size: desired size of the 4th dimension (will be filled with
            arange indices from 0 to last_dim_size-1)

    Returns:
        Tensor of 4d indices.
    """
    indices_3d = np.array(indices_3d)
    indices_4d = np.repeat(indices_3d, last_dim_size, axis=0)
    fourth_dim_indices = np.tile(np.arange(last_dim_size), indices_3d.shape[0])
    indices_4d = np.column_stack((indices_4d, fourth_dim_indices))
    return torch.tensor(indices_4d.T, dtype=torch.long)


def get_cubic_patches(arr_shape, centers, size):
    """Cut a cubic patch around a center id out of a 3d array.

    NOTE: Currently not used. Was implemented for draft of nn search in grid.

    Returns:
        New centers and mask.
    """
    # Create a meshgrid for the offsets
    offsets = np.array(
        np.meshgrid(
            np.arange(-size, size + 1),
            np.arange(-size, size + 1),
            np.arange(-size, size + 1),
        )
    ).T.reshape(-1, 3)

    # Add offsets to the center indices
    new_centers = centers[:, np.newaxis, :] + offsets

    mask = np.any((new_centers < 0) | (new_centers >= np.array(arr_shape[:3])), axis=-1)

    return new_centers, mask


def pose_vector_mean(pose_vecs, pose_fully_defined):
    """Calculate mean of pose vectors.

    This takes into account that point normals may contain observations from two
    surface sides and curvature directions have an ambiguous direction. It also
    enforces them to stay orthogonal.

    If not pose_fully_defined, the curvature directions are meaningless and we just
    return the first observation. Theoretically this shouldn't matter but it can save
    some computation time.

    Returns:
        ?
    """
    # Check the angle between all point normals relative to the first curvature
    # directions. Then look at how many are positive vs. negative and use the ones
    # that make up the majority. So if 5 pns point one way and 10 in the opposite,
    # we will use the 10 and discard the rest. This avoids averaging over pns that
    # are from opposite sides of an objects surface.
    valid_pose_vecs = np.where(np.any(pose_vecs, axis=1))[0]
    if len(valid_pose_vecs) == 0:
        logging.debug(f"no valid pose vecs: {pose_vecs}")
        return None, False
    # TODO: more generic names
    pns = pose_vecs[valid_pose_vecs, :3]
    cds1 = pose_vecs[valid_pose_vecs, 3:6]
    cds2 = pose_vecs[valid_pose_vecs, 6:9]
    pns_to_use = get_right_hand_angle(pns, cds1[0], cds2[0]) > 0
    if (sum(pns_to_use) < len(pns_to_use) // 2) or (sum(pns_to_use) == 0):
        pns_to_use = np.logical_not(pns_to_use)
    # Take the mean of all pns pointing in the same half sphere spanned by the cds.
    norm_mean = np.mean(pns[pns_to_use], axis=0)
    # Make sure the mean vector still has unit length.
    normed_norm_mean = norm_mean / np.linalg.norm(norm_mean)

    if sum(pose_fully_defined) < len(pose_fully_defined) // 2:
        # print(
        #     "cd dirs not sufficiently defined, averaging only introduces noise"
        # )
        # Just take 1st one. Shouldn't matter since cd should not be used anyways if
        # not pose_fully_defined. Only has a small effect on sampled possible poses.
        pv_means = np.hstack([normed_norm_mean, cds1[0], cds2[0]])
        use_cds_to_update = False
    else:
        # Find cds pointing in opposing directions and invert them. This is needed
        # because the curvature directions are ambiguous and both directions are
        # equivalent. If we average over opposing directions, we will get noise.
        cd1_dirs = get_right_hand_angle(cds1, cds2[0], normed_norm_mean) < 0
        cds1[cd1_dirs] = -cds1[cd1_dirs]
        cd1_mean = np.mean(cds1, axis=0)
        normed_cd1_mean = cd1_mean / np.linalg.norm(cd1_mean)
        # Get the second cd by calculating a vector orthogonal to cd1 and pn.
        cd2_mean = np.cross(normed_norm_mean, normed_cd1_mean)
        normed_cd2_mean = cd2_mean / np.linalg.norm(cd2_mean)
        if get_right_hand_angle(normed_cd1_mean, cd2_mean, normed_norm_mean) < 0:
            normed_cd2_mean = -normed_cd2_mean
        use_cds_to_update = True
        pv_means = np.hstack([normed_norm_mean, normed_cd1_mean, normed_cd2_mean])

    assert not np.any(np.isnan(pv_means)), "NaN in pose vector mean"
    return pv_means, use_cds_to_update


def get_most_common_bool(booleans):
    """Get most common value out of a list of boolean values.

    Returns:
        True when we have equally many True as False entries.
    """
    if np.ndim(booleans) > 2:
        raise NotImplementedError(
            "get_most_common_bool not implemented for features"
            "with more than 1 bool value. Current bool shape: "
            f"{np.array(booleans).shape} D={np.ndim(booleans)}"
        )
    else:
        booleans = np.array(booleans).flatten()
    if sum(booleans) >= sum(np.logical_not(booleans)):
        return True
    else:
        return False


def circular_mean(values):
    """Calculate the mean of a circular value such as hue where 0==1.

    Returns:
        Mean value.
    """
    # convert to radians
    angles = np.array(values) * 2 * np.pi
    # calculate circular mean
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    # Make sure returned values are positive
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    # convert back to [0, 1] range
    mean = mean_angle / (2 * np.pi)
    return mean


def build_point_cloud_graph(locations, features, feature_mapping):
    """Build a graph from observations without edges.

    Args:
        locations: array of x, y, z positions in space
        features: dictionary of features at locations
        feature_mapping: ?

    Returns:
        A NumpyGraph containing the observed features at locations.
    """
    graph = NumpyGraph(dict(x=features, pos=locations, feature_mapping=feature_mapping))

    return graph
