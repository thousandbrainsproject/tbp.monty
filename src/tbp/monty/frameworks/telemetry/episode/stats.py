# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.frameworks.telemetry.episode.schemas import LearningModuleStateTelemetry
from tbp.monty.frameworks.utils.logging_utils import (
    compute_pose_error,
    compute_unsupervised_stats,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_unique_rotations,
    rotations_to_quats,
)
from tbp.monty.geometry import Rotation


# TODO telemetry: convert to utils file? Declared as class for now to prevent name clash
class GraphLMStats:
    """Utility class substituting `logging_utils.get_stats_per_lm` and its callees."""

    @classmethod
    def get_lm_performance_stats(
        cls,
        lm: LearningModuleStateTelemetry,
        model: MontyBase,
        target: Mapping,
        target_data: Mapping,
        episode_seed: int,
    ) -> dict:
        """Generates the performance stats for a learning module.

        Inner-half equivalent of `logging_utils.get_stats_per_lm`.

        Args:
            lm: LM state event.
            model: Monty instance.
            target: Target object.
            target_data: Logging-friendly target object data.
            episode_seed: RNG seed used for the episode.

        Returns:
            LM performance stats.
        """
        lm_stats = cls.get_graph_lm_stats(lm, target)
        if lm.current_mlh is not None:
            lm_stats = cls.add_evidence_lm_episode_stats(lm, lm_stats, target)
        else:
            lm_stats = cls.add_pose_lm_episode_stats(lm, lm_stats)
        lm_stats = cls.add_policy_episode_stats(lm, lm_stats)
        lm_stats["monty_steps"] = model.episode_steps
        lm_stats["monty_matching_steps"] = model.matching_steps
        lm_stats["episode_seed"] = episode_seed
        lm_stats.update(target_data)

        # Add LM-specific target information
        lm_stats.update({"stepwise_target_object": lm.stepwise_target_object})
        return lm_stats

    @staticmethod
    def get_graph_lm_stats(lm: LearningModuleStateTelemetry, target: Mapping) -> dict:
        """Generates the stats dictionary for one episode for an LM.

        Equivalent of `logging_utils.get_graph_lm_episode_stats`.

        Returns:
            LM episode stats.
        """
        primary_target: str = target["object"]
        primary_target_rotation_quat: npt.ArrayLike = target["quat_rotation"]

        primary_performance = "patch_off_object"  # Performance on the primary target in
        # the environment, typically the target object we begin the episode on
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

        if lm.is_on_object:
            # TODO: update this?
            num_steps = lm.matching_steps
            location = np.array(lm.location)
            possible_matches = lm.possible_matches
            primary_performance = lm.terminal_state.value
            stepwise_performance = lm.terminal_state.value
            result = lm.detected_object
            if len(possible_matches) == 0:
                primary_performance = "no_match"
                stepwise_performance = "no_match"
            elif primary_target == "no_label":
                primary_performance = "no_label"
                stepwise_performance = "no_label"
            # Exactly one match
            elif primary_performance == "match":
                target_to_graph = lm.graph_id_to_target[lm.detected_object]
                if primary_target in target_to_graph:
                    primary_performance = "correct"
                    if lm.buffer_stats["symmetric_rotations"] is not None:
                        # Invert them since these are possible poses to rotate
                        # displacement, not the object rotations.
                        detected_rotation = rotations_to_quats(
                            lm.buffer_stats["symmetric_rotations"], invert=True
                        )
                    else:
                        detected_rotation = lm.buffer_stats["detected_rotation_quat"]
                    rotation_error = np.round(
                        compute_pose_error(
                            Rotation.from_quat(detected_rotation),
                            Rotation.from_quat(primary_target_rotation_quat),
                            return_degrees=True,
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

            elif primary_performance in ("time_out", "undefined"):
                result = possible_matches  # FIXME: not compatible with wandb logging
                # maybe join the list of strings?
                if len(possible_matches) == 1:
                    primary_performance = "pose_time_out"
                    stepwise_performance = "pose_time_out"

            individual_ts_perf = "time_out"
            # TODO eventually consider adding stepwise stats for the below
            if lm.buffer_stats["individual_ts_reached_at_step"] is not None:
                individual_ts_step = lm.buffer_stats["individual_ts_reached_at_step"]
                if lm.buffer_stats["individual_ts_object"] is None:
                    individual_ts_perf = "no_match"
                else:
                    target_to_graph = lm.graph_id_to_target[
                        lm.buffer_stats["individual_ts_object"]
                    ]
                    if primary_target in target_to_graph:
                        individual_ts_perf = "correct"
                        if lm.buffer_stats["symmetric_rotations_ts"] is not None:
                            detected_rotation_ts = rotations_to_quats(
                                lm.buffer_stats["symmetric_rotations_ts"],
                                invert=True,
                            )
                        else:
                            detected_rotation_ts = lm.buffer_stats["individual_ts_rot"]
                        individual_ts_rotation_error = np.round(
                            compute_pose_error(
                                Rotation.from_quat(detected_rotation_ts),
                                Rotation.from_quat(primary_target_rotation_quat),
                                return_degrees=True,
                            ),
                            4,
                        )
                    else:
                        individual_ts_perf = "confused"

        relative_time = np.diff(np.array(lm.buffer_stats["time"]), prepend=0)
        lm.buffer_stats["relative_time"] = relative_time

        detected_pose = lm.detected_pose
        if detected_pose is not None:
            detected_location = detected_pose[:3]
            detected_rotation = detected_pose[3:6]
            detected_scale = detected_pose[6]
        else:
            detected_location = None
            detected_rotation = None
            detected_scale = None

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
            "detected_location": detected_location,
            "detected_rotation": detected_rotation,
            "detected_scale": detected_scale,
            "location_rel_body": location,
            "detected_path": lm.buffer_stats["detected_path"],
            "symmetry_evidence": lm.symmetry_evidence,
            "individual_ts_reached_at_step": individual_ts_step,
            "individual_ts_performance": individual_ts_perf,
            "individual_ts_rotation_error": individual_ts_rotation_error,
            "time": np.sum(lm.buffer_stats["relative_time"]),
        }

        graph_vs_object_stats = compute_unsupervised_stats(
            possible_matches,
            primary_target,
            lm.graph_id_to_target,
            lm.target_to_graph_id,
        )
        stats.update(graph_vs_object_stats)
        return stats

    @classmethod
    def add_evidence_lm_episode_stats(
        cls,
        lm: LearningModuleStateTelemetry,
        stats: dict,
        target: Mapping,
    ) -> dict:
        """Equivalent of `logging_utils.add_evidence_lm_episode_stats`.

        Returns:
            Updated stats dictionary.
        """
        primary_target: str = target["object"]
        primary_target_rotation_quat: npt.ArrayLike = target["quat_rotation"]
        consistent_child_objects: Mapping = target["consistent_child_objects"]
        last_mlh = lm.current_mlh

        stats["most_likely_object"] = last_mlh["graph_id"]
        stats["most_likely_location"] = last_mlh["location"]
        stats["most_likely_rotation"] = (
            last_mlh["rotation"].inv().as_euler("xyz", degrees=True)
        )
        stats["highest_evidence"] = last_mlh["evidence"]
        stats["episode_avg_prediction_error"] = np.mean(
            lm.buffer_stats["mlh_prediction_error"]
        )
        stats = cls.calculate_performance(
            stats, "primary_performance", lm, primary_target
        )
        stats = cls.calculate_performance(
            stats, "stepwise_performance", lm, lm.stepwise_target_object
        )
        if stats["primary_performance"] == "correct_mlh":
            stats["rotation_error"] = np.round(
                compute_pose_error(
                    last_mlh["rotation"].inv(),
                    Rotation.from_quat(primary_target_rotation_quat),
                    return_degrees=True,
                ),
                4,
            )
        # Check if the most likely object is a consistent child object
        # Don't do this if the episode timed out, we had no match, or the detected
        # object was already an exact match with the label.
        if (
            stats["primary_performance"] in ["confused", "confused_mlh"]
            and consistent_child_objects
            and last_mlh["graph_id"] in consistent_child_objects
        ):
            stats["primary_performance"] = "consistent_child_obj"
        return stats

    @staticmethod
    def add_pose_lm_episode_stats(
        lm: LearningModuleStateTelemetry, stats: dict
    ) -> dict:
        """Add possible poses of an LM to episode stats.

        Args:
            lm: LM instance from which to add the statistics.
            stats: Statistics dictionary to update.

        Returns:
            Updated stats dictionary.
        """
        if lm.possible_poses and (
            stats["primary_performance"] in ["correct", "confused", "pose_time_out"]
        ):
            possible_matches = lm.possible_matches
            all_possible_poses = lm.possible_poses[possible_matches[0]]
            stats["possible_object_poses"] = get_unique_rotations(
                all_possible_poses, lm.pose_similarity_threshold
            )
            paths = np.array(lm.possible_paths[possible_matches[0]])
            stats["possible_object_locations"] = paths[:, -1]

            # FIXME: for some reason, we are getting the occasional pose in Scipy
            # Rotation format instead of float array. Find the source of the problem so
            # we don't have to run some extra fn every time we log to sanitize
            for i in range(len(stats["possible_object_poses"])):
                for j in range(len(stats["possible_object_poses"][i])):
                    pose = stats["possible_object_poses"][i][j]
                    if isinstance(pose, Rotation):
                        stats["possible_object_poses"][i][j] = pose.as_euler(
                            "xyz", degrees=True
                        )
        else:
            # All fields must be included in each update to enable periodic appending to
            # csv
            stats["possible_object_poses"] = np.nan
            stats["possible_object_locations"] = np.nan
        return stats

    @staticmethod
    def add_policy_episode_stats(lm: LearningModuleStateTelemetry, stats: dict) -> dict:
        if "goal_state_achieved" in lm.buffer_stats:
            stats["goal_states_attempted"] = len(lm.buffer_stats["goal_state_achieved"])
            stats["goal_state_achieved"] = np.sum(
                lm.buffer_stats["goal_state_achieved"]
            )

        else:
            stats["goal_states_attempted"] = 0
            stats["goal_state_achieved"] = 0
        return stats

    @staticmethod
    def calculate_performance(
        stats: dict,
        performance_type: str,
        lm: LearningModuleStateTelemetry,
        target_object,
    ) -> dict:
        """Calculate performance of an LM on a given target object.

        Args:
            stats: Statistics dictionary to update.
            performance_type: performance type index into stats
            lm: Learning module for which to generate stats.
            target_object: target (primary or stepwise) object for the LM to have
                converged to

        Returns:
            Updated stats dictionary.
        """
        if stats[performance_type] in ["time_out", "pose_time_out"]:
            # Check if the final result (object label) is consistent with the target
            if target_object in lm.graph_id_to_target[lm.current_mlh["graph_id"]]:
                stats[performance_type] = "correct_mlh"
            else:
                stats[performance_type] = "confused_mlh"
        return stats
