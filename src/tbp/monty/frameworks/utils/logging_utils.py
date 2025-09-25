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

import copy
import json
import logging
import os
from collections import deque
from itertools import chain
from pathlib import Path
from sys import getsizeof

import numpy as np
import numpy.typing as npt
import pandas as pd
import quaternion
import torch
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_unique_rotations,
    rotations_to_quats,
)

logger = logging.getLogger(__name__)


def load_stats(
    exp_path,
    load_train=True,
    load_eval=True,
    load_detailed=True,
    load_models=True,
    pretrained_dict=None,
):
    """Load experiment statistics from an experiment for analysis.

    Returns:
        train_stats: pandas DataFrame with training statistics
        eval_stats: pandas DataFrame with evaluation statistics
        detailed_stats: dict with detailed statistics
        lm_models: dict with loaded language models
    """
    train_stats, eval_stats, detailed_stats, lm_models = None, None, None, None
    if load_train:
        print("...loading and checking train statistics...")
        train_stats = pd.read_csv(os.path.join(exp_path, "train_stats.csv"))

    if load_eval:
        print("...loading and checking eval statistics...")
        eval_stats = pd.read_csv(os.path.join(exp_path, "eval_stats.csv"))

    if load_detailed:
        print("...loading detailed run statistics...")
        json_file = os.path.join(exp_path, "detailed_run_stats.json")
        try:
            with open(json_file, "r") as f:
                detailed_stats = json.load(f)
        except ValueError:
            detailed_stats = deserialize_json_chunks(json_file)
        f.close()

    if load_models:
        print("...loading LM models...")
        lm_models = load_models_from_dir(exp_path, pretrained_dict)

    return train_stats, eval_stats, detailed_stats, lm_models


def load_models_from_dir(exp_path, pretrained_dict=None):
    lm_models = {}

    if pretrained_dict is not None:
        lm_models["pretrained"] = {}
        state_dict = torch.load(os.path.join(pretrained_dict, "model.pt"))
        for lm_id in list(state_dict["lm_dict"].keys()):
            pretrained_models = state_dict["lm_dict"][lm_id]["graph_memory"]
            lm_models["pretrained"][lm_id] = pretrained_models

    for folder in os.listdir(exp_path):
        if folder.isnumeric():
            state_dict = torch.load(os.path.join(exp_path, folder, "model.pt"))
            for lm_id in list(state_dict["lm_dict"].keys()):
                epoch_models = state_dict["lm_dict"][lm_id]["graph_memory"]
                if folder not in lm_models.keys():
                    lm_models[folder] = {}
                lm_models[folder]["LM_" + str(lm_id)] = epoch_models
    return lm_models


def deserialize_json_chunks(json_file, start=0, stop=None, episodes=None):
    """Deserialize one episode at a time from json file.

    Only get episodes specified by arguments, which follow list / numpy like semantics.

    Note:
        assumes line counter is exactly in line with episode keys

    Args:
        json_file: full path to the json file to load
        start: int, get data starting at this episode
        stop: int, get data ending at this episode, not inclussive as usual in python
        episodes: iterable of ints with episodes to pull

    Returns:
        detailed_json: dict containing contents of file_handle
    """

    def should_get_episode(start, stop, episodes, counter):
        if episodes is not None:
            return counter in episodes
        else:
            return (counter >= start) and (counter < stop)

    detailed_json = {}
    stop = stop or np.inf
    with open(json_file, "r") as f:
        for line_counter, line in enumerate(f):
            if should_get_episode(start, stop, episodes, line_counter):
                # NOTE: json logging is only used at inference time and inference
                # episodes are independent and order does not matter. This hack fixes a
                # problem introduced from running in parallel: every episode had the
                # key 0 since it was its own experiment, so we update detailed_json with
                # line counter key instead of tmp_json key. This works for serial
                # episodes because order of execution is arbitrary, all that matters is
                # we know the parameters for that episode.
                tmp_json = json.loads(line)
                json_key = list(tmp_json.keys())[0]  # has only one key
                detailed_json[str(line_counter)] = tmp_json[json_key]
                del tmp_json

    if episodes is not None:
        str_episodes = [str(i) for i in episodes]
        if list(detailed_json.keys()) != str_episodes:
            print(
                "WARNING: episode keys did not equal json keys. This can happen if "
                "json file was not appended to in episode order. To manually load the"
                "whole file for debugging, run `deserialize_json_chunks(my_file)` with"
                "no further arguments"
            )
    return detailed_json


def get_object_graph_stats(graph_to_target, target_to_graph):
    n_objects_per_graph = [len(graph_to_target[k]) for k in graph_to_target.keys()]
    n_graphs_per_object = [len(target_to_graph[k]) for k in target_to_graph.keys()]
    results = dict(
        mean_objects_per_graph=np.mean(n_objects_per_graph),
        mean_graphs_per_object=np.mean(n_graphs_per_object),
    )
    return results


def matches_to_target_str(possible_matches, graph_to_target):
    """Get the possible target objects associated with each possible match.

    Targets are concatenated into a single string name for easy saving in a csv.

    Returns:
        dict: ?
    """
    possible_match_sources = set()
    for match in possible_matches:
        targets = graph_to_target[match]
        possible_match_sources.update(targets)

    sorted_targets = sorted(possible_match_sources)
    str_targets = "-".join(sorted_targets)
    return dict(possible_match_sources=str_targets)


def compute_unsupervised_stats(
    possible_matches, target, graph_to_target, target_to_graph
):
    """Compute statistics like how many graphs are built per object.

    Args:
        possible_matches: container of str that key into graph_to_target
        target: str ground truth name of object being presented
        graph_to_target: dict mapping each graph to the set of objects used to build
        target_to_graph: dict mapping each object to the set of graphs that used it

    Returns:
        dict: ?
    """
    object_is_known = target in target_to_graph
    result = None
    if object_is_known:
        # Seen this object in training, don't recognize
        result = "target_not_matched_(FN)"
        for candidate in possible_matches:
            if target in graph_to_target[candidate]:
                # Seen this object in training, and do recognize it
                result = "target_in_possible_matches_(TP)"
                break
    else:
        # Never seen this object, and don't recognize it
        result = "unknown_object_not_matched_(TN)"
        # TODO: couldn't we just check if len(possible_matches) > 0?
        for candidate in possible_matches:
            if target not in graph_to_target[candidate]:
                # Never seen this object, yet I mistakenly recognize
                result = "unknown_object_in_possible_matches_(FP)"
                break

    dict_stats = get_object_graph_stats(graph_to_target, target_to_graph)
    dict_stats.update(TFNP=result)
    str_targets = matches_to_target_str(possible_matches, graph_to_target)
    dict_stats.update(str_targets)
    return dict_stats


def get_reverse_rotation(rotation):
    return (np.array([360, 360, 360]) - np.array(rotation)) % 360


def get_unique_euler_poses(poses):
    """Get unique poses for an object from possible poses per path.

    Returns:
        unique_poses: array of unique poses
    """
    all_poses = []
    for path_poses in poses:
        for pose in path_poses:
            all_poses.append(pose)
    unique_poses = np.unique(all_poses, axis=0)
    return unique_poses


def check_rotation_accuracy(stats, last_n_step=1):
    pose_stats = []
    for episode in stats.keys():
        if len(stats[episode]["LM_0"]["possible_poses"]) >= last_n_step:
            target_object = stats[episode]["LM_0"]["target"]["object"]
            target_rotation = stats[episode]["LM_0"]["target"]["euler_rotation"]
            detected_rotations = stats[episode]["LM_0"]["possible_poses"][-last_n_step][
                target_object
            ]
            if len(detected_rotations) == 0:
                performance = "not_in_matches"
                detected_rotation = None
                num_paths, num_poses = 0, 0
            else:
                num_paths = len(detected_rotations)
                if num_paths > 1:
                    all_poses = []
                    for path in range(num_paths):
                        poses = np.array(detected_rotations[path])
                        if poses.shape[0] > 1:
                            for p in poses:
                                all_poses.append([p])
                        else:
                            all_poses.append(poses)
                    detected_rotations = np.array(all_poses)
                detected_rotations = np.array(detected_rotations)
                unique_poses = get_unique_euler_poses(detected_rotations)
                num_poses = unique_poses.shape[0]
                if num_poses == 1:
                    detected_rotation = unique_poses[0] % 360
                    dual_pose = np.array(
                        [
                            (detected_rotation[0] + 180) % 360,
                            (-detected_rotation[1] + 180) % 360,
                            (detected_rotation[2] + 180) % 360,
                        ]
                    )
                    if np.array_equal(
                        np.array(detected_rotation), np.array(target_rotation)
                    ) or np.array_equal(np.array(dual_pose), np.array(target_rotation)):
                        performance = "correct_rotation"
                    else:
                        performance = "wrong_rotation"
                else:
                    all_unique_poses = []
                    performance = "target_not_in_possible_poses"
                    for pose in unique_poses:
                        detected_rotation = pose % 360
                        dual_pose = np.array(
                            [
                                (detected_rotation[0] + 180) % 360,
                                (-detected_rotation[1] + 180) % 360,
                                (detected_rotation[2] + 180) % 360,
                            ]
                        )
                        all_unique_poses.append(detected_rotation)
                        all_unique_poses.append(dual_pose)
                        if np.array_equal(
                            np.array(detected_rotation),
                            np.array(target_rotation),
                        ) or np.array_equal(
                            np.array(dual_pose),
                            np.array(target_rotation),
                        ):
                            performance = "target_in_possible_poses"
                    detected_rotation = all_unique_poses
            pose_stats.append(
                pd.Series(
                    [
                        target_object,
                        str(target_rotation),
                        str(detected_rotation),
                        performance,
                        num_paths,
                        num_poses,
                    ]
                )
            )
    pose_stats = pd.concat(pose_stats, axis=1).T
    pose_stats.columns = [
        "object",
        "target_rotation",
        "detected_rotation",
        "primary_performance",
        "num_paths",
        "num_poses",
    ]
    return pose_stats


def check_detection_accuracy_at_step(stats, last_n_step=1):
    detection_stats = []
    for episode in stats.keys():
        possible_matches = stats[episode]["LM_0"]["possible_matches"]
        if len(possible_matches) >= last_n_step:
            target_object = stats[episode]["LM_0"]["target"]["object"]
            matches_at_step = possible_matches[-last_n_step]
            if len(matches_at_step) == 0:
                performance = "no matches"
            elif len(matches_at_step) == 1:
                if matches_at_step[0] == target_object:
                    performance = "correct"
                else:
                    performance = "wrong object"
            else:
                performance = "object not in possible matches"
                for possible_match in matches_at_step:
                    if possible_match == target_object:
                        performance = "object in possible matches"

            detection_stats.append(
                pd.Series(
                    [
                        target_object,
                        performance,
                    ]
                )
            )
    detection_stats = pd.concat(detection_stats, axis=1).T
    detection_stats.columns = [
        "object",
        "primary_performance",
    ]
    return detection_stats


def get_time_stats(all_ds, all_conditions) -> pd.DataFrame:
    """Get summary of run times in a dataframe for each condition.

    Args:
        all_ds: detailed stats (dict) for each condition
        all_conditions: name of each condition

    Returns:
        Runtime stats.
    """
    time_stats = []
    for i, detailed_stats in enumerate(all_ds):
        for episode in detailed_stats:
            times = detailed_stats[episode]["LM_0"]["relative_time"]
            for step in range(len(times)):
                time_stats.append(
                    pd.Series(
                        [
                            all_conditions[i],
                            times[step],
                            step,
                        ]
                    )
                )
    time_stats = pd.concat(time_stats, axis=1).T
    time_stats.columns = ["model_type", "time", "step"]
    return time_stats


def compute_pose_errors(
    predicted_rotation: Rotation, target_rotation: Rotation
) -> npt.NDArray[np.float64] | float:
    """Computes the angular pose errors between predicted and target rotations.

    Both inputs must be instances of `scipy.spatial.transform.Rotation`. The
    `predicted_rotation` may contain a single rotation or a list of rotations,
    while `target_rotation` must be exactly one rotation.

    The pose error is defined as the geodesic distance on SO(3) — the angle of the
    relative rotation between predicted and target. If `predicted_rotation` contains
    multiple rotations, this function returns the errors among them.

    Note that the `.inv()` operation in this method is due to how geodesic distance
    between two rotations is calculated, not a side-effect of whether the target
    rotation is stored in its normal form, or as its inverse. The function therefore
    assumes that the orientations are already in the same coordinate system before
    the comparison.

    Args:
        predicted_rotation: Predicted rotation(s). Can be a single or list of
            rotation.
        target_rotation: Target rotation. Must represent a single rotation.

    Returns:
        The angular errors in radians.
    """
    errors: npt.NDArray[np.float64] | float = (
        predicted_rotation * target_rotation.inv()
    ).magnitude()
    return errors


def compute_pose_error(
    predicted_rotation: Rotation, target_rotation: Rotation
) -> float:
    """Computes the minimum angular pose error between predicted and target rotations.

    See `compute_pose_errors` for more details.

    Args:
        predicted_rotation: Predicted rotation(s). Can be a single or list of
            rotation.
        target_rotation: Target rotation. Must represent a single rotation.

    Returns:
        The minimum angular error in radians.
    """
    error = np.min(compute_pose_errors(predicted_rotation, target_rotation))
    return error


def get_overall_pose_error(stats, lm_id="LM_0"):
    """Get mean pose error over all episodes.

    Note:
        This can now be obtained easier from the .csv stats.

    Args:
        stats: detailed stats
        lm_id: id of learning module

    Returns:
        mean pose error
    """
    errors = []
    for episode in stats.keys():
        detected = stats[episode][lm_id]["detected_rotation_quat"]
        if detected is not None:  # only checking accuracy on detected objects
            target = stats[episode][lm_id]["target"]["quat_rotation"]
            err = compute_pose_error(
                Rotation.from_quat(detected), Rotation.from_quat(target)
            )
            errors.append(err)
    return np.round(np.mean(errors), 4)


def print_overall_stats(stats):
    acc = (
        (
            len(stats[stats["primary_performance"] == "correct"])
            + len(stats[stats["primary_performance"] == "correct_mlh"])
        )
        / len(stats)
        * 100
    )
    print(f"Detected {np.round(acc, 2)}% correctly")
    rt = np.sum(stats["time"])
    rt_per_step = np.mean(stats["time"] / stats["monty_matching_steps"])
    print(
        f"overall run time: {np.round(rt, 2)} seconds "
        f"({np.round(rt / 60, 2)} minutes), "
        f"{np.round(rt / len(stats), 2)} seconds per episode, "
        f"{np.round(rt_per_step, 2)} seconds per step."
    )


def print_unsupervised_stats(stats, epoch_len):
    """Print stats of unsupervised learning experiment."""
    first_epoch_stats = stats[:epoch_len]
    later_epoch_stats = stats[epoch_len:]
    first_epoch_acc = (
        len(first_epoch_stats[first_epoch_stats["primary_performance"] == "no_match"])
        / len(first_epoch_stats)
        * 100
    )
    later_acc = (
        (
            len(
                later_epoch_stats[later_epoch_stats["primary_performance"] == "correct"]
            )
            + len(
                later_epoch_stats[
                    later_epoch_stats["primary_performance"] == "correct_mlh"
                ]
            )
        )
        / len(later_epoch_stats)
        * 100
    )
    print(
        f"Detected {np.round(first_epoch_acc, 2)}% correctly as new object"
        "in first epoch"
    )
    print(f"Detected {np.round(later_acc, 2)}% correctly after first epoch")
    print(f"Mean objects per graph: {list(stats['mean_objects_per_graph'])[-1]}")
    print(f"Mean graphs per object: {list(stats['mean_graphs_per_object'])[-1]}")
    print("Merged graphs:")
    for string in np.unique(list(stats["possible_match_sources"])):
        if "-" in string:
            print("     " + string)
    rt = np.sum(stats["time"])
    print(
        f"overall run time: {np.round(rt, 2)} seconds ({np.round(rt / 60, 2)} minutes),"
        f" {np.round(rt / len(stats), 2)} seconds per episode."
    )


def calculate_tpr(tp, fn):
    """Calculate True Positive Rate, aka sensitivity.

    Args:
        tp: true positives
        fn: false negatives

    Returns:
        True Positive Rate
    """
    if (tp + fn) == 0:
        return None
    else:
        return tp / (tp + fn)


def calculate_fpr(fp, tn):
    """Calculate False Positive Rate, aka specificity.

    Args:
        fp: false positives
        tn: true negatives

    Returns:
        False Positive Rate
    """
    if (fp + tn) == 0:
        return None
    else:
        return fp / (fp + tn)


###
# Functions used to help graph matching loggers
###


def get_graph_lm_episode_stats(lm):
    """Populate stats dictionary for one episode for a lm.

    Args:
        lm: Learning module for which to generate stats.

    Returns:
        dict with stats of one episode.
    """
    primary_performance = "patch_off_object"  # Performance on the primary target in
    # the environmnet, typically the target object we begin the episode on
    stepwise_performance = "patch_off_object"  # Performance relative to the object
    # the learning module is actually receiving sensory input from when it converges
    location = np.array([0, 0, 0])
    num_steps = 0
    result = None
    possible_matches = []
    rotation_error = None
    individual_ts_step = None
    individual_ts_perf = "patch_off_object"
    individual_ts_rotation_error = None

    if (
        len(lm.buffer.on_object) > 0 and lm.buffer.on_object[0] != 0
    ):  # TODO: update this?
        num_steps = lm.buffer.get_num_matching_steps()

        location = np.array(lm.buffer.get_current_location(input_channel="first"))
        possible_matches = lm.get_possible_matches()
        primary_performance = lm.terminal_state
        stepwise_performance = lm.terminal_state
        result = lm.detected_object
        if len(possible_matches) == 0:
            primary_performance = "no_match"
            stepwise_performance = "no_match"
        elif lm.primary_target == "no_label":
            primary_performance = "no_label"
            stepwise_performance = "no_label"
        # Exactly one match
        elif primary_performance == "match":
            target_to_graph = lm.graph_id_to_target[lm.detected_object]
            if lm.primary_target in target_to_graph:
                primary_performance = "correct"
                if lm.buffer.stats["symmetric_rotations"] is not None:
                    # Invert them since these are possible poses to rotate displacement
                    # not the object rotations.
                    detected_rotation = rotations_to_quats(
                        lm.buffer.stats["symmetric_rotations"], invert=True
                    )
                else:
                    detected_rotation = lm.buffer.stats["detected_rotation_quat"]
                rotation_error = np.round(
                    compute_pose_error(
                        Rotation.from_quat(detected_rotation),
                        Rotation.from_quat(lm.primary_target_rotation_quat),
                    ),
                    4,
                )
            else:
                primary_performance = "confused"

            if lm.stepwise_target_object in target_to_graph:
                stepwise_performance = "correct"
                # TODO eventually add rotation and translation error
            else:
                stepwise_performance = "confused"

        elif primary_performance == "time_out":
            result = possible_matches  # FIXME: not compatible with wandb logging
            if len(possible_matches) == 1:
                primary_performance = "pose_time_out"
                stepwise_performance = "pose_time_out"

        individual_ts_perf = "time_out"
        # TODO eventually consider adding stepwise stats for the below
        if lm.buffer.stats["individual_ts_reached_at_step"] is not None:
            individual_ts_step = lm.buffer.stats["individual_ts_reached_at_step"]
            if lm.buffer.stats["individual_ts_object"] is None:
                individual_ts_perf = "no_match"
            else:
                target_to_graph = lm.graph_id_to_target[
                    lm.buffer.stats["individual_ts_object"]
                ]
                if lm.primary_target in target_to_graph:
                    individual_ts_perf = "correct"
                    if lm.buffer.stats["symmetric_rotations_ts"] is not None:
                        detected_rotation_ts = rotations_to_quats(
                            lm.buffer.stats["symmetric_rotations_ts"],
                            invert=True,
                        )
                    else:
                        detected_rotation_ts = lm.buffer.stats["individual_ts_rot"]
                    individual_ts_rotation_error = np.round(
                        compute_pose_error(
                            Rotation.from_quat(detected_rotation_ts),
                            Rotation.from_quat(lm.primary_target_rotation_quat),
                        ),
                        4,
                    )
                else:
                    individual_ts_perf = "confused"

    relative_time = np.diff(np.array(lm.buffer.stats["time"]), prepend=0)
    lm.buffer.stats["relative_time"] = relative_time

    stats = {
        "primary_performance": primary_performance,
        "stepwise_performance": stepwise_performance,
        "num_steps": num_steps,
        "result": result,
        # TODO update the below so that we also log rotation error for the stepwise
        # object --> not currently implemented because the rotation of distractor
        # objects is not easily specified/recovered
        "rotation_error": rotation_error,
        "num_possible_matches": len(possible_matches),
        "detected_location": lm.detected_pose[:3],
        "detected_rotation": lm.detected_pose[3:6],
        "detected_scale": lm.detected_pose[6],
        "location_rel_body": location,
        "detected_path": lm.buffer.stats["detected_path"],
        "symmetry_evidence": lm.symmetry_evidence,
        "individual_ts_reached_at_step": individual_ts_step,
        "individual_ts_performance": individual_ts_perf,
        "individual_ts_rotation_error": individual_ts_rotation_error,
        "time": np.sum(lm.buffer.stats["relative_time"]),
    }

    graph_vs_object_stats = compute_unsupervised_stats(
        possible_matches,
        lm.primary_target,
        lm.graph_id_to_target,
        lm.target_to_graph_id,
    )
    stats.update(graph_vs_object_stats)
    return stats


def add_pose_lm_episode_stats(lm, stats):
    """Add possible poses of lm to episode stats.

    Args:
        lm: LM istance from which to add the statistics.
        stats: Statistics dictionary to update.

    Returns:
        Updated stats dictionary.
    """
    if hasattr(lm, "possible_poses") and (
        stats["primary_performance"] in ["correct", "confused", "pose_time_out"]
    ):
        possible_matches = lm.get_possible_matches()
        all_possible_poses = lm.possible_poses[possible_matches[0]]
        stats["possible_object_poses"] = get_unique_rotations(
            all_possible_poses, lm.pose_similarity_threshold
        )
        paths = np.array(lm.possible_paths[possible_matches[0]])
        stats["possible_object_locations"] = paths[:, -1]

        # FIXME: for some reason, we are getting the occasional pose in Scipy Rotation
        # format instead of float array. Find the source of the problem so we don't have
        # to run some extra fn every time we log to sanitize
        for i in range(len(stats["possible_object_poses"])):
            for j in range(len(stats["possible_object_poses"][i])):
                pose = stats["possible_object_poses"][i][j]
                if isinstance(pose, Rotation):
                    stats["possible_object_poses"][i][j] = pose.as_euler(
                        "xyz", degrees=True
                    )
    else:
        # All fields must be included in each update to enable periodic appending to csv
        stats["possible_object_poses"] = np.nan
        stats["possible_object_locations"] = np.nan
    return stats


def get_stats_per_lm(model, target):
    """Loop through lms and get stats.

    Args:
        model: model instance
        target: target object

    Returns:
        performance_dict: dict with stats per lm
    """
    performance_dict = {}
    primary_target_dict = target_data_to_dict(target)
    for i, lm in enumerate(model.learning_modules):
        lm_stats = get_graph_lm_episode_stats(lm)
        if hasattr(lm, "evidence"):
            lm_stats = add_evidence_lm_episode_stats(lm, lm_stats)
        else:
            lm_stats = add_pose_lm_episode_stats(lm, lm_stats)
        lm_stats = add_policy_episode_stats(lm, lm_stats)
        lm_stats["monty_steps"] = model.episode_steps
        lm_stats["monty_matching_steps"] = model.matching_steps
        performance_dict[f"LM_{i}"] = lm_stats
        performance_dict[f"LM_{i}"].update(primary_target_dict)
        # Add LM-specific target information
        performance_dict[f"LM_{i}"].update(
            {"stepwise_target_object": lm.stepwise_target_object}
        )
    return performance_dict


def add_policy_episode_stats(lm, stats):
    if "goal_state_achieved" in lm.buffer.stats.keys():
        stats["goal_states_attempted"] = len(lm.buffer.stats["goal_state_achieved"])
        stats["goal_state_achieved"] = np.sum(lm.buffer.stats["goal_state_achieved"])

    else:
        stats["goal_states_attempted"] = 0
        stats["goal_state_achieved"] = 0

    return stats


def add_evidence_lm_episode_stats(lm, stats):
    last_mlh = lm.get_current_mlh()

    stats["most_likely_object"] = last_mlh["graph_id"]
    stats["most_likely_location"] = last_mlh["location"]
    stats["most_likely_rotation"] = (
        last_mlh["rotation"].inv().as_euler("xyz", degrees=True)
    )
    stats["highest_evidence"] = last_mlh["evidence"]
    stats = calculate_performance(stats, "primary_performance", lm, lm.primary_target)
    stats = calculate_performance(
        stats, "stepwise_performance", lm, lm.stepwise_target_object
    )
    if stats["primary_performance"] == "correct_mlh":
        stats["rotation_error"] = np.round(
            compute_pose_error(
                last_mlh["rotation"].inv(),
                Rotation.from_quat(lm.primary_target_rotation_quat),
            ),
            4,
        )
    return stats


def calculate_performance(stats, performance_type, lm, target_object):
    """Calculate performance of an LM on a given target object.

    Args:
        stats: Statistics dictionary to update.
        performance_type: performance type index into stats
        lm: Learning module for which to generate stats.
        target_object: target (primary or stepwise) object for the LM to have converged
            to

    Returns:
        Updated stats dictionary.
    """
    if stats[performance_type] in ["time_out", "pose_time_out"]:
        # Check if the final result (object label) is consistent with the target
        if target_object in lm.graph_id_to_target[lm.get_current_mlh()["graph_id"]]:
            stats[performance_type] = "correct_mlh"
        else:
            stats[performance_type] = "confused_mlh"

    return stats


def target_data_to_dict(target):
    """Format target params to dict.

    Args:
        target: target params

    Returns:
        dict with target params
    """
    output_dict = {}
    output_dict["primary_target_object"] = target["object"]
    output_dict["primary_target_position"] = target["position"]
    output_dict["primary_target_rotation_euler"] = target["euler_rotation"]
    output_dict["primary_target_rotation_quat"] = quaternion.as_float_array(
        target["rotation"]
    )
    # Currently scale is applied uniformly along all dimensions
    output_dict["primary_target_scale"] = target["scale"][0]

    return output_dict


###
# Functions that assist handlers
###


def format_columns_for_wandb(lm_dict):
    """Various columns break wandb because we are playing fast and loose with types.

    Put any standardizations here.

    Args:
        lm_dict: dict, part of a larger dict ~ {LM_0: lm_dict, LM_1: lm_dict}

    Returns:
        formatted lm_dict
    """
    formatted_dict = copy.deepcopy(lm_dict)
    if "result" in formatted_dict:
        if isinstance(formatted_dict["result"], list):
            new_result = "^".join(formatted_dict["result"])
            formatted_dict["result"] = new_result

    return formatted_dict


def lm_stats_to_dataframe(stats, format_for_wandb=False):
    """Take in a dictionary and format into a dataframe.

    Example::

        {0: {LM_0: stats, LM_1: stats...}, 1:...} --> dataframe

    Currently we are reporting once per episode, so the loop over episodes is only over
    a singel key, value pair, but leaving it here because it is backward compatible.

    Returns:
        dataframe
    """
    df_list = []
    for episode in stats.values():
        lm_dict = {}
        # Loop over things like LM_*, SM_*, motor_system and get only LM_*
        for key in episode.keys():
            if isinstance(key, str):
                if key.startswith("LM_"):
                    if format_for_wandb:
                        lm_dict[key] = format_columns_for_wandb(
                            copy.deepcopy(episode[key])
                        )
                    else:
                        lm_dict[key] = episode[key]

        if len(lm_dict) > 0:
            df_list.append(pd.DataFrame.from_dict(lm_dict, orient="index"))

    big_df = pd.concat(df_list)
    big_df["lm_id"] = big_df.index
    return big_df


def maybe_rename_existing_file(log_file, extension, report_count):
    """Check if this run has already been executed.

    If so, change name of existing log file by appending _old to it.

    Args:
        log_file: full path to the file, e.g. ~/.../detailed_run_stats.json
        extension: str name of file type
        report_count: ?
    """
    if (report_count == 0) and (os.path.exists(log_file)):
        old_name = log_file.split(extension)[0]
        new_name = old_name + "_old" + extension

        logger.warning(
            f"Output file {log_file} already exists. This file will be moved"
            f" to {new_name}"
        )

        if os.path.exists(new_name):
            logger.warning(
                f"Output file {new_name} also already exists. This file will be removed"
                " before renaming."
            )
            os.remove(new_name)

        Path(log_file).rename(new_name)


def maybe_rename_existing_directory(path, report_count):
    if (report_count == 0) and os.path.exists(path):
        new_path = path + "_old"
        logger.warning(
            f"Output path {path} already exists. This path will be movedto {new_path}"
        )

        if os.path.exists(new_path):
            logger.warning(
                f"{new_path} also exists, and will be removed before renaming"
            )
            os.remove(new_path)

        Path(path).rename(new_path)


def get_rgba_frames_single_sm(observations):
    """Convert a time series of rgba observations into format for wandb.Video.

    Args:
        observations: episode_stats[sm][___observations]

    Returns:
        formatted observations
    """
    formatted_observations = []
    for step in range(len(observations)):
        # get data for this time step
        rgba = observations[step]["rgba"]

        # format according to wandb API: [time, channels, height, width]
        formatted_observations.append(np.moveaxis(rgba, [0, 1, 2], [1, 2, 0]))

    formatted_observations = np.array(formatted_observations)
    return formatted_observations


###
# Functions used for buffer size based logging
###


def total_size(o):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}


    The recursive recipe universally cited on stack exchange and blogs for gauging the
    size of python objets in memory.

    See Also:
        https://code.activestate.com/recipes/577504/
    """
    dict_handler = lambda d: chain.from_iterable(d.items())  # noqa: E731
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    # all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
