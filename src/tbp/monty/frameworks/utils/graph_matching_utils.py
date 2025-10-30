# Copyright 2025 Thousand Brains Project
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
import math
from itertools import permutations

import numpy as np
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.spatial_arithmetics import get_more_directions_in_plane

logger = logging.getLogger(__name__)


def get_correct_k_n(k_n, num_datapoints):
    """Determine k_n given the number of datapoints.

    k_n specified in hyperparameter may not be possible to achive with the number
    of data points collected. This function checks for that and adjusts k_n.

    Args:
        k_n: current number k neareast neighbors specified for graph building
        num_datapoints: number of observations available to build the graph.

    Returns:
        adjusted k_n
    """
    # k_n will need to be one larger than how many edges you like since the
    # first nearest edge of a node is itself.
    k_n = k_n + 1

    if num_datapoints <= k_n:
        if num_datapoints > 2:
            k_n = num_datapoints - 1
            logger.debug(f"Setting k_n to {k_n}")
        else:
            logger.error("not enough observations collected to build graph.")
            return None
    return k_n


def get_unique_paths(possible_paths, threshold=0.01):
    """Get all unique paths in a list of possible paths.

    Args:
        possible_paths: List of possible paths (locations)
        threshold: minimun distance between two parts to be considered
            different. defaults to 0.01

    Returns:
        List of unique paths (locations)
    """
    unique_paths = []
    for path in possible_paths:
        path_already_in_up = False
        if len(unique_paths) == 0:
            unique_paths.append(path)
        else:
            for up in unique_paths:
                path_already_in_up = (
                    np.sum(np.linalg.norm(np.array(path) - np.array(up), axis=1))
                    < threshold
                )
                if path_already_in_up:
                    break
            if not path_already_in_up:
                unique_paths.append(path)
    return np.array(unique_paths)


def is_in_ranges(array, ranges):
    """Check for each element in an array whether it is in its specified range.

    Each element can have a different tolerance range.

    Returns:
        True if all elements are in their respective ranges, False otherwise.
    """
    for i, e in enumerate(array):
        if ranges[i][0] <= ranges[i][1]:
            element_in_range = e >= ranges[i][0] and e <= ranges[i][1]
        else:  # circular feature
            element_in_range = e >= ranges[i][0] or e <= ranges[i][1]
        if not element_in_range:
            return False
    return True


def get_uniform_initial_possible_poses(n_degrees_sampled=9):
    """Get initial list of possible poses.

    Args:
        n_degrees_sampled: Number of degrees sampled for each axis. Default = 9
            Which means tested degrees are in [  0.,  45.,  90., 135., 180., 225.,
            270., 315.] This results in 512 unique pose combinations.

    Of those, depending on the displacement vector, some are equivalent
    eg. [0,0,0] and [180,180,180] or [  0,  45,  90] and [180, 135, 270]
    (a, b, c) == (a + 180, -b + 180, c + 180)
    (see https://books.google.gr/books?id=rn3OBQAAQBAJ p.267)

    Returns:
        List of poses to test.
    """
    degrees = np.linspace(0, 360, n_degrees_sampled)[:-1]
    all_degrees = np.hstack([degrees, degrees, degrees])
    all_poses = list(permutations(all_degrees, 3))
    all_poses = np.unique(all_poses, axis=0)
    unique_poses = []
    dual_poses = []
    for pose in all_poses:
        dual_pose = np.array(
            [
                (pose[0] + 180) % 360,
                (-pose[1] + 180) % 360,
                (pose[2] + 180) % 360,
            ]
        )
        if list(pose) not in dual_poses:
            unique_poses.append(Rotation.from_euler("xyz", pose, degrees=True))
            dual_poses.append(list(dual_pose))

    # run faster by just going through poses that are currently tested:
    # unique_poses = [
    #     Rotation.from_euler("xyz", [0.0, r, 0.0], degrees=True)
    #     for r in np.linspace(0, 360, 9)[:-1]
    # ]
    return unique_poses


def get_initial_possible_poses(initial_possible_pose_type) -> list[Rotation] | None:
    """Initialize initial_possible_poses to test based on initial_possible_pose_type.

    Args:
        initial_possible_pose_type: How to sample initial possible poses.
            Options are:
            - "uniform": Sample uniformly from the space of possible poses.
            - "informed": Sample poses that are likely to be possible based on
                the object's geometry and the first observation.
            - list of euler angles: Use a list of predefiende poses to test (useful for
                debugging).

    Returns:
        List of initial possible poses to test or None if initial_possible_pose_type
        is "informed".
    """
    if initial_possible_pose_type == "uniform":
        initial_possible_poses = get_uniform_initial_possible_poses()
    elif initial_possible_pose_type == "informed":
        # In this case we initialize hypotheses with the first observation
        initial_possible_poses = None
    else:
        initial_possible_poses = [
            Rotation.from_euler("xyz", pose, degrees=True).inv()
            for pose in initial_possible_pose_type
        ]
    return initial_possible_poses


def add_pose_features_to_tolerances(tolerances, default_tolerances=20) -> dict:
    """Add default `pose_vectors` tolerances if not set.

    Returns:
        Tolerances dictionary with added pose_vectors if not set.
    """
    for input_channel in tolerances.keys():
        if "pose_vectors" not in tolerances[input_channel].keys():
            # NOTE: assumes that there are 3 pose vectors
            tolerances[input_channel]["pose_vectors"] = np.ones(3) * default_tolerances

        # Turn degree tolerances into radians
        tolerances[input_channel]["pose_vectors"] = [
            math.radians(deg) for deg in tolerances[input_channel]["pose_vectors"]
        ]
    return tolerances


def get_relevant_curvature(features):
    """Get the relevant curvature from features. Used to scale search sphere.

    In the case of principal_curvatures and principal_curvatures_log we use the
    maximum absolute curvature between the two values. Otherwise we just return
    the curvature value.

    Note:
        Not sure if other curvatures work as well as log curvatures since they
        may have too big of a range.

    Returns:
        Magnitude of sensed curvature (maximum if using two principal curvatures).
    """
    if "principal_curvatures_log" in features.keys():
        curvatures = features["principal_curvatures_log"]
        curvatures = np.max(np.abs(curvatures))
    elif "principal_curvatures" in features.keys():
        curvatures = features["principal_curvatures"]
        curvatures = np.max(np.abs(curvatures))
    elif "mean_curvature" in features.keys():
        curvatures = features["mean_curvature"]
    elif "mean_curvature_sc" in features.keys():
        curvatures = features["mean_curvature_sc"]
    elif "gaussian_curvature" in features.keys():
        curvatures = features["gaussian_curvature"]
    elif "gaussian_curvature_sc" in features.keys():
        curvatures = features["gaussian_curvature_sc"]
    else:
        logger.error(
            f"No curvatures contained in the features {list(features.keys())}."
        )
        # Return large curvature so we use an almost circular search sphere.
        curvatures = 10
    return curvatures


def get_scaled_evidences(evidences, per_object=False):
    """Scale evidences to be in range [-1, 1] for voting.

    This is useful so that the absolute evidence values don't distort the votes
    (for example if one LM has already had a lot more matching steps than another
    and the evidence is not bounded). It is also useful to keep the evidence added
    from a single voting step in a reasonable range.

    By default we normalize using the maximum and minimum evidence over all objects.
    There is also an option to scale the evidence for each object independently but
    that might give low evidence objects too much of a boost. We could probably
    remove this option.

    Returns:
        Scaled evidences.
    """
    scaled_evidences = {}
    if per_object:
        for graph_id in evidences.keys():
            if len(evidences[graph_id]):
                graph_evidences = evidences[graph_id]
                min_evidence = np.min(graph_evidences)
                max_evidence = np.max(graph_evidences)
                scaled_evidences[graph_id] = (evidences[graph_id] - min_evidence) / (
                    max_evidence - min_evidence
                )
                # put in range(-1, 1)
                scaled_evidences[graph_id] = (scaled_evidences[graph_id] - 0.5) * 2
    else:
        min_evidence = np.inf
        max_evidence = -np.inf
        for graph_id in evidences.keys():
            graph_evidences = evidences[graph_id]
            if len(graph_evidences):
                min_evidence = min(min_evidence, np.min(graph_evidences))
                max_evidence = max(max_evidence, np.max(graph_evidences))

        for graph_id in evidences.keys():
            graph_evidences = evidences[graph_id]
            if max_evidence >= 1:
                scaled_evidences[graph_id] = (graph_evidences - min_evidence) / (
                    max_evidence - min_evidence
                )
                # put in range(-1, 1)
                scaled_evidences[graph_id] = (scaled_evidences[graph_id] - 0.5) * 2
            else:
                # If largest value is <1, don't scale them -> don't increase any
                # evidences. Instead just make sure they are in the right range.
                scaled_evidences[graph_id] = np.clip(graph_evidences, -1, 1)
    return scaled_evidences


def get_custom_distances(nearest_node_locs, search_locs, search_sns, search_curvature):
    """Calculate custom distances modulated by surface normal and curvature.

    Args:
        nearest_node_locs: locations of nearest nodes to search_locs.
            shape=(num_hyp, max_nneighbors, 3)
        search_locs: search locations for each hypothesis.
            shape=(num_hyp, 3)
        search_sns: sensed surface normal rotated by hypothesis pose.
            shape=(num_hyp, 3)
        search_curvature: magnitude of sensed curvature (maximum if using
            two principal curvatures). Is used to modulate the search spheres
            thickness in the direction of the surface normal.
            shape=1

    Returns:
        custom_nearest_node_dists: custom distances of each nearest location
            from its search location taking into account the hypothesis point
            normal and sensed curvature.
            shape=(num_hyp, max_nneighbors)
    """
    # Calculate difference vectors between query point and all other points
    # expand_dims of search_locs so it has shape (num_hyp, 1, 3)
    # shape of differences = (num_hyp, max_nneighbors, 3)
    differences = nearest_node_locs - np.expand_dims(search_locs, axis=1)
    # Calculate the dot product between the query normal and the difference vectors
    # This tells us how well the points are aligned with the plane perpendicular to
    # the query normal. Points with dot product 0 are in this plane, higher
    # magnitudes of the dot product means they are further away from that plane
    # (-> should have larger distance).
    dot_products = np.einsum("ijk,ik->ij", differences, search_sns)
    # Calculate the euclidean distances. shape=(num_hyp, max_nneighbors)
    euclidean_dists = np.linalg.norm(differences, axis=2)
    # Calculate the total distances by adding the absolute dot product to the
    # euclidean distances. We multiply the dot product by 1/curvature to modulate
    # the flatness of the search sphere. If the curvature is large we want to be
    # able to go further out of the sphere while we want to stay close to the point
    # normal plane if we have a curvature close to 0.
    # To have a minimum wiggle room above and below the plane, even if we have 0
    # curvature (and to avoid division by 0) we add 0.5 to the denominator.
    # shape=(num_hyp, max_nneighbors).
    custom_nearest_node_dists = euclidean_dists + np.abs(dot_products) * (
        1 / (np.abs(search_curvature) + 0.5)
    )
    return custom_nearest_node_dists


# ====== Functions for detecting on new object ======
# TODO: These will be integrated into the goal-state-generator with the motor
# system refactor
def create_exponential_kernel(size, decay_rate):
    """Create an exponentially decaying kernel.

    Used to convolve e.g. evidence history when determining whether we are on a new
    object.

    Args:
        size: Size of the kernel.
        decay_rate: Decay rate of the kernel.

    Returns:
        Exponentially decaying kernel.
    """
    # Index so that exponential kernel applies 1.0 to the
    # most recent evidence change (i.e. full weighting)
    indices = np.arange(start=size - 1, stop=-1, step=-1)
    kernel = np.exp(-decay_rate * indices)
    return kernel


def detect_new_object_exponential(
    max_ev_per_step,
    detection_threshold=-1.0,
    decay_rate=3,
):
    """Detect we're on a new object using exponentially decaying evidence changes.

    Evidence changes from multiple steps into the past are convolved by exponentially
    decaying constants, such that more recent steps carry more significance.

    Args:
        max_ev_per_step: List of the maximum evidence (across all locations/poses)
            for the current MLH object, across all steps of the current episode
        detection_threshold: The total amount of negative evidence in the
            counter/sum that needs to be exceeded to determine that the LM has moved on
            to a new object
        decay_rate: The rate of decay that determines how much past evidence
            -drops contribute to the current estimate of change

    Returns:
        True if the total amount of negative evidence is less than or equal to the
        detection threshold, False otherwise.
    """
    ev_changes, _ = process_delta_evidence_values(max_ev_per_step)

    exp_kernel = create_exponential_kernel(size=len(ev_changes), decay_rate=decay_rate)

    # Update evidence values with exponential kernel
    ev_changes = np.multiply(exp_kernel, ev_changes)

    total_ev_changes = np.sum(ev_changes)

    return total_ev_changes <= detection_threshold


def detect_new_object_k_steps(
    max_ev_per_step,
    detection_threshold=-1.0,
    k=3,
    reset_at_positive_jump=False,
):
    """Detect we're on a new object using the evidence changes from multiple steps.

    Evidence changes from multiple steps into the past are considered. We look at the
    change in evidence over k discrete steps, weighing these equally.

    Args:
        max_ev_per_step: List of the maximum evidence (across all locations/poses)
            for the current MLH object, across all steps of the current episode
        detection_threshold: The total amount of negative evidence in the
            counter/sum that needs to be exceeded to determine that the LM has moved on
            to a new object
        k: How many steps into the past to look when summing the negative change in
            evidence
        reset_at_positive_jump: Boolean to "reset" the accumulated changes when
            there is a positive increase in evidence, i.e. k is further limited by this
            history

    Returns:
        True if the total evidence change is less than or equal to the detection
        threshold, False otherwise.
    """
    ev_changes, postive_jump_loc = process_delta_evidence_values(
        max_ev_per_step[-(k + 1) :]
        # NB if looking e.g. at 1 evidence step change in the past, then need 2
        # evidence values, hence k+1 used
    )

    # Further truncate if there were any positive jumps in evidence
    # Note we do not reset if evidence remains static (delta 0)
    if postive_jump_loc is not None and reset_at_positive_jump:
        ev_changes = ev_changes[postive_jump_loc:]

    total_ev_changes = np.sum(ev_changes)

    return total_ev_changes <= detection_threshold


def process_delta_evidence_values(max_ev_per_step):
    """Pre-process the max evidence values to get the change in evidence across steps.

    Clip the values to be less than or equal to 0.

    Also returns the index of the most recent positive change in evidence

    Returns:
        clipped_ev_changes: Clipped change in evidence across steps.
        postive_jump_loc: Index of the most recent positive change in evidence.
    """
    max_ev_per_step = np.array(max_ev_per_step)

    # Change in evidence across steps
    ev_changes = max_ev_per_step[1:] - max_ev_per_step[:-1]

    # Find the most recent postive jump in evidence before clipping
    positive_jumps = ev_changes > 0
    if np.any(positive_jumps):
        postive_jump_loc = np.where(positive_jumps)[0][-1]
    else:
        postive_jump_loc = None

    # Clip values as we are looking for instances of evidnece drops to suggest movement
    # onto a new object
    clipped_ev_changes = np.clip(ev_changes, -np.inf, 0)

    return clipped_ev_changes, postive_jump_loc


def find_step_on_new_object(
    stepwise_targets, primary_target, n_steps_off_primary_target
):
    """Returns the episode step at which we've moved off the primary target object.

    The returned episode step is the first step at which we've been off the primary
    target for a total of n_steps_off_primary_target consecutive steps.
    """
    off_primary_array = np.array(stepwise_targets != primary_target)

    # Create a sliding window of size k
    window = np.ones(n_steps_off_primary_target)

    # Use convolution to find when k Trues occur in a row
    conv = np.convolve(off_primary_array, window, "valid") == n_steps_off_primary_target

    # If conv contains at least one True, find the first occurrence
    if conv.any():
        return np.where(conv)[0][0] + n_steps_off_primary_target - 1
    else:
        return None


def possible_sensed_directions(
    sensed_directions: np.ndarray, num_hyps_per_node: int
) -> np.ndarray:
    """Returns the possible sensed directions for all nodes.

    This function determines the possible sensed directions for a given set of sensed
    directions. It relies on two different behaviors depending on the value of
    num_hyps_per_node.
        - If num_hyps_per_node equals 2: then pose is well defined (i.e., PC1 != PC2).
            A well defined pose does not distinguish between mirrored directions of PC1
            and PC2 (e.g., object can be upside down), therefore we sample both
            directions.
        - If num_hyps_per_node is not 2: this function samples additional poses in
            the plane perpendicular to the sensed surface normal.

    Arguments:
        sensed_directions: An array of sensed directions.
        num_hyps_per_node: Number of rotations to get for each node.

    Returns:
        possible_s_d : Possible sensed direction for all nodes at each rotation
    """
    if num_hyps_per_node == 2:
        possible_s_d = [
            sensed_directions.copy(),
            sensed_directions.copy(),
        ]
        possible_s_d[1][1:] = possible_s_d[1][1:] * -1
    else:
        possible_s_d = get_more_directions_in_plane(
            sensed_directions, num_hyps_per_node
        )
    return possible_s_d
