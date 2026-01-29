# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy

import numpy as np
import pandas as pd
import wandb
from sklearn.preprocessing import LabelEncoder

from tbp.monty.frameworks.loggers.exp_logger import BaseMontyLogger
from tbp.monty.frameworks.utils.logging_utils import (
    get_stats_per_lm,
    target_data_to_dict,
    total_size,
)

"""
Loggers define a pool of data in the attribute self.data. The job of loggers is to
create and manage this pool of data, including flushing it periodically. The job of
handlers is to grab subsets of the data pool, format it, and send it to a location
(wandb, or a certain file). The data pool can be updated at any of the callback points
in the experiment, for example, `post_episode`. The loggers receive a standard set of
arguments used for building the data pool, most notably a reference to the model.
During reporting, all handlers are called upon to send data to a destination.
Handlers also receive a standard set of arguments, most importantly the `self.data`
pool.

The Basic logger is associated with the BASIC monty_log_level specified in
config_args.logging. The Detailed logger is associated with the DETAILED logging
level. The Detailed logger's data pool contains a superset of the data in the
BASIC logger's `self.data` attribute.

The term logger is used unconventionally here. Normally, the thing being logged is a
string of text and the destination is a file or the terminal screen. Here, the logged
content is structured data. Perhaps the term DataReporter is more apt.

NOTE: previously, we updated the data pool every episode, and logged at most once per
episode. Reporting and flushing frequency were based on the size of self.data. To make
things easier to think about for handlers, we moved reporting and flushing both to take
place at the end of an episode. Why all the callbacks then, like post_step, post_train,
etc.? We are building a plane while flying it. Best not to throw the landing gear out
the window just because it isn't being used right now.
"""


class BasicGraphMatchingLogger(BaseMontyLogger):
    """Basic logger that logs or saves information when logging is called."""

    def __init__(
        self,
        handlers,
    ):
        """Initialize logger."""
        self.handlers = handlers
        self.data = dict(
            BASIC=dict(
                train_stats={},
                train_overall_stats={},
                train_targets={},
                train_actions={},
                train_timing={},
                eval_stats={},
                eval_overall_stats={},
                eval_actions={},
                eval_targets={},
                eval_timing={},
            ),
            DETAILED={},
        )
        self.overall_train_stats = dict(
            num_episodes=0,
            num_correct=0,
            num_correct_mlh=0,
            num_no_match=0,
            num_confused=0,
            num_confused_mlh=0,
            num_pose_time_out=0,
            num_time_out=0,
            num_patch_off_object=0,
            num_no_label=0,
            num_consistent_child_obj=0,
            num_correct_child_or_parent=0,
            num_correct_per_lm=0,
            num_correct_mlh_per_lm=0,
            num_consistent_child_obj_per_lm=0,
            num_no_match_per_lm=0,
            num_confused_per_lm=0,
            num_confused_mlh_per_lm=0,
            num_pose_time_out_per_lm=0,
            num_time_out_per_lm=0,
            num_patch_off_object_per_lm=0,
            num_no_label_per_lm=0,
            episode_correct=0,
            episode_correct_mlh=0,
            episode_no_match=0,
            episode_confused=0,
            episode_confused_mlh=0,
            episode_pose_time_out=0,
            episode_time_out=0,
            episode_avg_prediction_error=[],
            episode_lm_performances=[],
            # Total number of steps performed during the episode,
            # including steps where no sensory data was passed to the learning-modules:
            monty_steps=[],
            # Number of global monty *matching* steps. Counts steps when at least one LM
            # was updated:
            monty_matching_steps=[],
            # Number of steps associated with an individual LM processing data, i.e.
            # can differ across the LMs of a Monty model:
            episode_lm_steps=[],
            episode_lm_steps_indv_ts=[],
            episode_symmetry_evidence=[],
            rotation_errors=[],
            run_times=[],
            # Policy stats
            goal_states_attempted=0,
            goal_state_success_rate=0,
        )
        self.overall_eval_stats = copy.deepcopy(self.overall_train_stats)
        self.lms = []
        # Order of performance_options matters since we check them in sequence for each
        # lm. The lower in the list, the stronger it is to determine overall
        # performance. Performance lower down in the list will always trump higher-up
        # performance values. For example, if we have N LMs and one of them has
        # performance "correct", the overall episode performance is correct. It doesn't
        # matter if all the other LMs have a time_out performance.
        # The three strong terminal condition cases (no_match, confused, correct) need
        # to be listed last. The weaker time out conditions first so they don't
        # overwrite the performance of an LM that caused the episode to end.
        # TODO: what if 2 LMs reach a strong terminal state at the same step? For
        # example if one LM reaches the correct state and the other confused the
        # performance will be logged as correct. However, in this case we would
        # probably want to keep moving or log a conflicting performance.
        # TODO: If we have a time out and look at the mlh, we should take the majority
        # vote and not let correct_mlh win
        self.performance_options = [
            "patch_off_object",
            "no_label",
            "pose_time_out",
            "time_out",
            "consistent_child_obj",  # also counted if LM didn't converge
            "confused_mlh",
            "correct_mlh",
            "no_match",
            "confused",
            "correct",
        ]
        self.performance_encoder = LabelEncoder()
        self.performance_encoder.fit(self.performance_options)
        self.use_parallel_wandb_logging = False

        pd.set_option("display.max_rows", False)  # noqa: FBT003

    def flush(self):
        self.data = dict(
            BASIC=dict(
                train_stats={},
                train_overall_stats={},
                train_targets={},
                train_actions={},
                train_timing={},
                eval_stats={},
                eval_overall_stats={},
                eval_actions={},
                eval_targets={},
                eval_timing={},
            ),
            DETAILED={},
        )

    def log_episode(self, logger_args, output_dir, model):
        mode = model.experiment_mode
        episode = logger_args[f"{mode}_episodes"]

        for handler in self.handlers:
            handler.report_episode(self.data, output_dir, episode, mode)

        if not self.use_parallel_wandb_logging:
            # when logging in parallel to wandb we need to wait with flushing
            # until the parallel run script has retrieved the episode stats.
            self.flush()

    def maybe_log(self, logger_args, output_dir, model):
        """Left here in case we go back to size based logging.

        Remove if not used after code has stabilized.
        """
        # If we get above 10Mb, report, flush, continue
        if total_size(self.data) > 10_000_000:
            self.log(logger_args, output_dir, model)

    def post_episode(self, logger_args, output_dir, model):
        self.update_episode_data(logger_args, model)
        self.log_episode(logger_args, output_dir, model)

    def update_episode_data(self, logger_args, model):
        """Run get_stats_per_lm and add to overall stats.

        Store stats ~
            1 (episode)
                lm_0 (which lm)
                    stats
        """
        performance_dict = get_stats_per_lm(
            model, logger_args["target"], logger_args["episode_seed"]
        )
        target_dict = target_data_to_dict(logger_args["target"])
        if len(self.lms) == 0:  # first time function is called
            for lm in performance_dict.keys():
                if lm.startswith("LM_"):
                    self.lms.append(lm)

        mode = model.experiment_mode
        episode = logger_args[f"{mode}_episodes"]
        actions = model.motor_system._policy.action_sequence
        logger_time = {k: v for k, v in logger_args.items() if k != "target"}
        self.data["BASIC"][f"{mode}_stats"][episode] = performance_dict

        self.update_overall_stats(
            mode, episode, model.episode_steps, model.matching_steps
        )
        overall_stats = self.get_formatted_overall_stats(mode, episode)

        self.data["BASIC"][f"{mode}_overall_stats"][episode] = overall_stats
        self.data["BASIC"][f"{mode}_actions"][episode] = actions
        self.data["BASIC"][f"{mode}_targets"][episode] = target_dict
        self.data["BASIC"][f"{mode}_timing"][episode] = logger_time
        self.data["BASIC"][f"{mode}_stats"][episode]["target"] = target_dict

    def update_overall_stats(self, mode, episode, episode_steps, monty_matching_steps):
        """Update overall run stats for mode."""
        if mode == "train":
            stats = self.overall_train_stats
        else:
            stats = self.overall_eval_stats

        lm_performances = []
        for lm in self.lms:
            # This accumulates stats from all LM
            episode_stats = self.data["BASIC"][f"{mode}_stats"][episode][lm]
            performance = episode_stats["primary_performance"]

            if performance is not None:  # in pre training performance is None
                stats[f"num_{performance}_per_lm"] += 1
                lm_performances.append(performance)

            stats["rotation_errors"].append(episode_stats["rotation_error"])
            stats["run_times"].append(episode_stats["time"])
            stats["episode_lm_steps"].append(episode_stats["num_steps"])
            stats["episode_lm_steps_indv_ts"].append(
                episode_stats["individual_ts_reached_at_step"]
            )
            stats["episode_symmetry_evidence"].append(
                episode_stats["symmetry_evidence"]
            )
            stats["monty_steps"].append(episode_steps)
            stats["monty_matching_steps"].append(monty_matching_steps)
            # older LMs don't have prediction error stats
            if "episode_avg_prediction_error" in episode_stats:
                stats["episode_avg_prediction_error"].append(
                    episode_stats["episode_avg_prediction_error"]
                )

            if performance in {"consistent_child_obj", "correct", "correct_mlh"}:
                stats["num_correct_child_or_parent"] += 1

            stats["goal_states_attempted"] = episode_stats["goal_states_attempted"]

            stats["goal_state_success_rate"] = (
                episode_stats["goal_state_achieved"]
                / episode_stats["goal_states_attempted"]
                if episode_stats["goal_states_attempted"]
                else 0  # Handles division by 0
            )

        episode_performance = None
        stats["episode_lm_performances"].append(lm_performances)
        for p in self.performance_options:
            if p in lm_performances:
                # order of performance_options matters since we overwrite here!
                # episode_performance is only no_match if no lm had another
                # performance. That makes it possible for some lms to have no match
                # but still have an overall performance of correct (or other).
                episode_performance = p

        if episode_performance:
            for p in self.performance_options:
                stats[f"episode_{p}"] = int(p == episode_performance)
                stats[f"num_{p}"] += int(p == episode_performance)

        stats["num_episodes"] += 1

    def get_formatted_overall_stats(self, mode, episode):
        if mode == "train":
            stats = self.overall_train_stats
        else:
            stats = self.overall_eval_stats

        # Stores rotation errors if the object was recognized ("correct")
        correct_rotation_errors = [
            re for re in stats["rotation_errors"] if re is not None
        ]
        episode_re = [
            re for re in stats["rotation_errors"][-len(self.lms) :] if re is not None
        ]
        episode_individual_ts_steps = [
            steps
            for steps in stats["episode_lm_steps_indv_ts"][-len(self.lms) :]
            if steps is not None
        ]
        episode_lm_performances = self.performance_encoder.transform(
            stats["episode_lm_performances"][-1]
        )

        if len(episode_re) == 0:  # object was not recognized
            episode_re = [-1]

        overall_stats = {
            # % for performance per episode. This is the overall performance
            # of a Monty model, individual LMs may have different performances.
            # _mlh performances are determined using the most likely hypothesis
            # after a time out. For instance correct_mlh means that max steps
            # was reached without being confident enough about one object and pose
            # to classify it but the hypothesis with the highest evidence was
            # correct.
            "overall/percent_correct": (
                (stats["num_correct"] + stats["num_correct_mlh"])
                / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_no_match": (
                stats["num_no_match"] / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_confused": (
                (stats["num_confused"] + stats["num_confused_mlh"])
                / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_correct_mlh": (
                (stats["num_correct_mlh"]) / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_confused_mlh": (
                (stats["num_confused_mlh"]) / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_pose_time_out": (
                stats["num_pose_time_out"] / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_time_out": (
                stats["num_time_out"] / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_used_mlh_after_timeout": (
                (stats["num_correct_mlh"] + stats["num_confused_mlh"])
                / (stats["num_episodes"])
            )
            * 100,
            # Mean rotation error on all LMs that recognized the object
            "overall/avg_rotation_error": (
                np.mean(correct_rotation_errors)
                if len(correct_rotation_errors) > 0
                else np.nan
            ),
            "overall/avg_num_lm_steps": (
                np.mean(stats["episode_lm_steps"])
                if len(stats["episode_lm_steps"]) > 0
                else np.nan
            ),
            "overall/avg_num_monty_steps": (
                np.mean(stats["monty_steps"])
                if len(stats["monty_steps"]) > 0
                else np.nan
            ),
            "overall/avg_num_monty_matching_steps": (
                np.mean(stats["monty_matching_steps"])
                if len(stats["monty_matching_steps"]) > 0
                else np.nan
            ),
            "overall/avg_prediction_error": (
                np.mean(stats["episode_avg_prediction_error"])
                if len(stats["episode_avg_prediction_error"]) > 0
                else np.nan
            ),
            "overall/percent_consistent_child_obj": (
                stats["num_consistent_child_obj"] / (stats["num_episodes"])
            )
            * 100,
            "overall/percent_correct_child_or_parent": (
                stats["num_correct_child_or_parent"]
                / (stats["num_episodes"] * len(self.lms))
            )
            * 100,
            "overall/run_time": np.sum(stats["run_times"]) / len(self.lms),
            # NOTE: does not take into account different runtimes with multiple LMs
            "overall/avg_episode_run_time": (
                np.mean(stats["run_times"]) if len(stats["run_times"]) > 0 else np.nan
            ),
            "overall/num_episodes": stats["num_episodes"],
            # Stats for most recent episode
            # Performance of the overall Monty model
            "episode/correct": stats["episode_correct"] or stats["episode_correct_mlh"],
            "episode/no_match": stats["episode_no_match"],
            "episode/confused": (
                stats["episode_confused"] or stats["episode_confused_mlh"]
            ),
            "episode/correct_mlh": stats["episode_correct_mlh"],
            "episode/confused_mlh": stats["episode_confused_mlh"],
            "episode/pose_time_out": stats["episode_pose_time_out"],
            "episode/time_out": stats["episode_time_out"],
            "episode/consistent_child_obj": stats["episode_consistent_child_obj"],
            "episode/consistent_child_or_parent": (
                stats["episode_consistent_child_obj"]
                or stats["episode_correct"]
                or stats["episode_correct_mlh"]
            ),
            "episode/used_mlh_after_time_out": stats["episode_correct_mlh"]
            or stats["episode_confused_mlh"],
            "episode/rotation_error": (
                np.mean(episode_re) if len(episode_re) > 0 else np.nan
            ),
            # steps is the max number of steps of all LMs. Some LMs may have taken
            # less steps because they were not on the object all the time.
            "episode/lm_steps": np.max(stats["episode_lm_steps"][-len(self.lms) :]),
            "episode/monty_steps": stats["monty_steps"][-1],
            "episode/monty_matching_steps": stats["monty_matching_steps"][-1],
            "episode/mean_lm_steps_to_indv_ts": (
                np.mean(episode_individual_ts_steps)
                if len(episode_individual_ts_steps) > 0
                else np.nan
            ),
            "episode/run_time": np.max(stats["run_times"][-len(self.lms) :]),
            # Mean symmetry evidence with multiple LMs may be > required evidence
            # since one LM reaching its terminal condition doesn't mean all others do.
            "episode/symmetry_evidence": (
                np.mean(stats["episode_symmetry_evidence"][-len(self.lms) :])
                if len(stats["episode_symmetry_evidence"][-len(self.lms) :]) > 0
                else np.nan
            ),
            "episode/goal_states_attempted": stats["goal_states_attempted"],
            "episode/goal_state_success_rate": stats["goal_state_success_rate"],
            "episode/avg_prediction_error": stats["episode_avg_prediction_error"],
        }

        for p in self.performance_options:
            # % performance for each LM of the Monty model. For instance, some LMs
            # may have no_match but the overall model still recognized the object.
            if p == "correct":
                overall_stats["overall/percent_correct_per_lm"] = (
                    (stats["num_correct_per_lm"] + stats["num_correct_mlh_per_lm"])
                    / (stats["num_episodes"] * len(self.lms))
                ) * 100
            elif p == "confused":
                overall_stats["overall/percent_confused_per_lm"] = (
                    (stats["num_confused_per_lm"] + stats["num_confused_mlh_per_lm"])
                    / (stats["num_episodes"] * len(self.lms))
                ) * 100
            elif p in {"correct_mlh", "confused_mlh"}:
                # skip because they are already included in correct and confused stats
                pass
            else:
                overall_stats[f"overall/percent_{p}_per_lm"] = (
                    stats[f"num_{p}_per_lm"] / (stats["num_episodes"] * len(self.lms))
                ) * 100

        for lm in self.lms:
            lm_stats = self.data["BASIC"][f"{mode}_stats"][episode][lm]
            overall_stats[f"{lm}/episode/steps_to_individual_ts"] = lm_stats[
                "individual_ts_reached_at_step"
            ]
            overall_stats[f"{lm}/episode/individual_ts_rotation_error"] = lm_stats[
                "individual_ts_rotation_error"
            ]
            if "episode_avg_prediction_error" in lm_stats:
                overall_stats[f"{lm}/episode/avg_prediction_error"] = lm_stats[
                    "episode_avg_prediction_error"
                ]

        if len(self.lms) > 1:  # add histograms when running multiple LMs
            overall_stats["episode/rotation_error_per_lm"] = wandb.Histogram(episode_re)
            overall_stats["episode/steps_per_lm"] = wandb.Histogram(
                stats["episode_lm_steps"][-len(self.lms) :]
            )
            overall_stats["episode/steps_per_lm_indv_ts"] = wandb.Histogram(
                episode_individual_ts_steps
            )
            overall_stats["episode/symmetry_evidence_per_lm"] = wandb.Histogram(
                stats["episode_symmetry_evidence"][-len(self.lms) :]
            )
            overall_stats["episode/lm_performances"] = wandb.Histogram(
                episode_lm_performances
            )
            # filter out prediction errors that are nan
            prediction_errors = stats["episode_avg_prediction_error"][-len(self.lms) :]
            valid_prediction_errors = [e for e in prediction_errors if not np.isnan(e)]
            if valid_prediction_errors:
                overall_stats["episode/avg_prediction_error_dist"] = wandb.Histogram(
                    valid_prediction_errors
                )
                overall_stats["episode/avg_prediction_error"] = np.mean(
                    valid_prediction_errors
                )

        return overall_stats


class DetailedGraphMatchingLogger(BasicGraphMatchingLogger):
    """Log detailed stats as .json file by saving data for each LM and SM."""

    def __init__(self, handlers):
        """Initialize stats dicts."""
        super().__init__(handlers)

        self.train_episodes_to_total = {}
        self.eval_episodes_to_total = {}

    def log_episode(self, logger_args, output_dir, model):
        mode = model.experiment_mode
        episode = logger_args[f"{mode}_episodes"]
        kwargs = dict(
            train_episodes_to_total=self.train_episodes_to_total,
            eval_episodes_to_total=self.eval_episodes_to_total,
        )

        for handler in self.handlers:
            handler.report_episode(self.data, output_dir, episode, mode, **kwargs)

        self.flush()

    def post_episode(self, logger_args, output_dir, model):
        self.update_episode_data(logger_args, model)
        self.log_episode(logger_args, output_dir, model)

    def update_episode_data(self, logger_args, model):
        """Add episode data to overall buffer_data dict.

        Store stats ~
            1 (episode)
                lm_0 (which lm)
                    stats
        """
        # update train / eval stats
        super().update_episode_data(logger_args, model)

        episodes = logger_args["train_episodes"] + logger_args["eval_episodes"]
        self.train_episodes_to_total[logger_args["train_episodes"]] = episodes
        self.eval_episodes_to_total[logger_args["eval_episodes"]] = episodes

        buffer_data = {}
        for i, lm in enumerate(model.learning_modules):
            lm_dict = {}
            lm_dict.update(logger_args)
            lm_dict.update({"locations": lm.buffer.locations})
            lm_dict.update(lm.buffer.features)
            lm_dict.update({"displacements": lm.buffer.displacements})
            lm_dict.update(lm.buffer.stats)
            lm_dict.update(mode=model.experiment_mode)
            lm_dict.update({"stepwise_targets_list": lm.stepwise_targets_list})
            buffer_data[f"LM_{i}"] = lm_dict  # NOTE: probably same for all LMs

        for i, sm in enumerate(model.sensor_modules):
            if len(sm.state_dict()["raw_observations"]) > 0:
                buffer_data[f"SM_{i}"] = sm.state_dict()

        # TODO ensure will work with multiple, independent sensor agents
        buffer_data["motor_system"] = {}
        buffer_data["motor_system"]["action_sequence"] = (
            model.motor_system._policy.action_sequence
        )

        # Some motor systems store additional data specific to their policy, e.g. when
        # principal curvature has informed movements
        if hasattr(model.motor_system._policy, "action_details"):
            buffer_data["motor_system"]["action_details"] = (
                model.motor_system._policy.action_details
            )

        self.data["DETAILED"][episodes] = buffer_data


class SelectiveEvidenceLogger(BasicGraphMatchingLogger):
    """Log evidences as .json file by saving data for each LM and SM.

    This is slimmed down to only log data needed for object similarity analysis.
    Data logged:
        - evidences for each object and pose at the end of an episode
        - first frame of the view finder
    """

    def __init__(self, handlers):
        """Initialize stats dicts."""
        super().__init__(handlers)

        self.train_episodes_to_total = {}
        self.eval_episodes_to_total = {}

    def log_episode(self, logger_args, output_dir, model):
        mode = model.experiment_mode
        episode = logger_args[f"{mode}_episodes"]
        kwargs = dict(
            train_episodes_to_total=self.train_episodes_to_total,
            eval_episodes_to_total=self.eval_episodes_to_total,
        )

        for handler in self.handlers:
            handler.report_episode(self.data, output_dir, episode, mode, **kwargs)

        self.flush()

    def post_episode(self, logger_args, output_dir, model):
        self.update_episode_data(logger_args, model)
        self.log_episode(logger_args, output_dir, model)

    def update_episode_data(self, logger_args, model):
        """Add episode data to overall buffer_data dict."""
        # update train / eval stats
        super().update_episode_data(logger_args, model)

        episodes = logger_args["train_episodes"] + logger_args["eval_episodes"]
        self.train_episodes_to_total[logger_args["train_episodes"]] = episodes
        self.eval_episodes_to_total[logger_args["eval_episodes"]] = episodes

        buffer_data = {}
        for i, lm in enumerate(model.learning_modules):
            lm_dict = {}
            lm_dict.update(
                {
                    # Save evidences and hypotheses only for last step to save storage
                    "evidences_ls": lm.buffer.stats["evidences"][-1],
                    "possible_locations_ls": lm.buffer.stats["possible_locations"][-1],
                    # possible rotations don't change over time so no indexing here
                    "possible_rotations_ls": lm.buffer.stats["possible_rotations"],
                    # Save possible matches, mlh and symmetry evidence for all steps
                    "possible_matches": lm.buffer.stats["possible_matches"],
                    "current_mlh": lm.buffer.stats["current_mlh"],
                    "symmetry_evidence": lm.buffer.stats["symmetry_evidence"],
                }
            )
            buffer_data[f"LM_{i}"] = lm_dict

        for i, sm in enumerate(model.sensor_modules):
            if len(sm.state_dict()["raw_observations"]) > 0:
                # Only store first observation
                buffer_data[f"SM_{i}"] = sm.state_dict()["raw_observations"][0]

        self.data["DETAILED"][episodes] = buffer_data
