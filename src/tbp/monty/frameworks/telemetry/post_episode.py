# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Final, Sequence

import numpy as np
import wandb
from pydantic import Field
from sklearn.preprocessing import LabelEncoder
from typing_extensions import Self

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.loggers.monty_handlers import MontyHandler
from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.frameworks.telemetry.consumers import TelemetryConsumer
from tbp.monty.frameworks.telemetry.events import TelemetryEvent
from tbp.monty.frameworks.telemetry.producers import (
    TelemetryEmitter,
)
from tbp.monty.frameworks.utils.logging_utils import (
    get_stats_per_lm,
    target_data_to_dict,
)


class PostEpisodeTelemetry(TelemetryEvent):
    """Event produced by `MontyExperiment.post_episode()`."""

    SCHEMA_ID: Final[str] = "post_episode"

    episode_steps: int
    """Number of steps in which at least 1 LM received infos during exploration."""

    matching_steps: int
    """Number of steps in which at least 1 LM was updated. It is not the same as each
    individual LM's number of matching steps."""

    lm_stats: dict
    """Output of `get_stats_per_lm()`."""

    actions: list
    """Output of `MotorSystem.action_sequence`."""

    targets: dict
    """Output of `target_data_to_dict()`."""

    timing: dict
    """Output of `MontyExperiment.logger_args`."""

    extra: dict = Field(default_factory=dict)
    """Miscellaneous data to pass along to consumers."""

    @classmethod
    def from_logger_args(cls, logger_args: dict, model: MontyBase, emitter="") -> Self:
        """Constructs a `PostEpisodeTelemetry` from a `MontyExperiment`.

        Equivalent to the first half of `BasicGraphMatchingLogger.update_episode_data`.

        Args:
            logger_args: The dict from `MontyExperiment.logger_args`.
            model: The live `MontyBase` instance.
            emitter: Name of the emitting module.

        Returns:
            Populated `PostEpisodeTelemetry` instance.
        """
        performance_dict = get_stats_per_lm(
            model, logger_args["target"], logger_args["episode_seed"]
        )
        target_dict = target_data_to_dict(logger_args["target"])

        mode = model.experiment_mode
        episode = logger_args[f"{mode}_episodes"]
        actions = model.motor_system.action_sequence
        logger_time = {k: v for k, v in logger_args.items() if k != "target"}

        # TODO telemetry: keep here or move to consumer?
        performance_dict["target"] = target_dict

        return PostEpisodeTelemetry(
            emitter=emitter,
            mode=mode,
            episode=episode,
            step=model.total_steps,
            episode_steps=model.episode_steps,
            matching_steps=model.matching_steps,
            lm_stats=performance_dict,
            actions=actions,
            targets=target_dict,
            timing=logger_time,
        )


class ExperimentStatsTelemetry(TelemetryEvent):
    """Event produced by `PostEpisodeTelemetryConsumer`.

    Contains aggregated overall episode statistics of the experiment.
    """

    SCHEMA_ID: Final[str] = "experiment_stats"

    stats: dict
    """Stats dictionary to pass along to consumers."""

    @classmethod
    def from_parent(cls, stats: dict, parent: TelemetryEvent, emitter="") -> Self:
        """Constructs an `ExperimentStatsTelemetry` from a parent event.

        Copies `TelemetryEvent` fields from the parent into the new derived event.

        Args:
            stats: The aggregated stats dict to embed in the event, typically
                   `PostEpisodeTelemetryConsumer.data`.
            parent: The triggering event whose context fields are to be copied.
            emitter: Name of the emitting module. If empty, inherits from the parent.

        Returns:
            A new `ExperimentStatsTelemetry` instance.
        """
        return cls(
            stats=stats,
            emitter=(emitter if emitter else parent.emitter),
            mode=parent.mode,
            episode=parent.episode,
            step=parent.step,
        )


class PostEpisodeTelemetryConsumer(TelemetryConsumer):
    """Aggregates `PostEpisodeTelemetry` events and forwards the results to handlers.

    Replaces the aggregation and reporting logic of `BasicGraphMatchingLogger`
    (``update_overall_stats``, ``get_formatted_overall_stats``, ``log_episode``).
    Handlers receive the same `self.data` structure as before for compatibility.

    Also forwards the data to downstream consumers via `ExperimentStatsTelemetry`.
    """

    SCHEMA_IDS: Final[list[str]] = [PostEpisodeTelemetry.SCHEMA_ID]

    def __init__(
        self,
        level: int,
        handlers: Sequence[MontyHandler],
        output_dir: Path,
        **kwargs,
    ):
        """Initializes the post-episode consumer.

        Args:
            level: Logging level (``logging.DEBUG``, ``logging.INFO``, etc.) for
                   outgoing `ExperimentStatsTelemetry` event snapshots.
            handlers: `MontyHandler` instances that receive episode reports via
                      ``handler.report_episode()``. Replaces the handler list previously
                      owned by `BasicGraphMatchingLogger`.
            output_dir: Output path forwarded to handlers.
            **kwargs: Forwarded to parent class.
        """
        super().__init__(**kwargs)
        self.log_level = level
        self.handlers = handlers  # TODO telemetry: move handlers to separate consumer?
        self.output_dir = output_dir
        self.data = self._blank_data()
        self.overall_train_stats = self._blank_overall_stats()
        self.overall_eval_stats = self._blank_overall_stats()
        self.lms: list[str] = []
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
        self.telemetry = TelemetryEmitter("episode_stats", level=level)

    @staticmethod
    def _blank_data() -> dict:
        """Returns an empty `self.data` structure compatible with `MontyHandler`.

        Preserves the ``BASIC`` / ``DETAILED`` shape previously maintained by
        `BasicGraphMatchingLogger` for handler compatibility.
        """
        return dict(
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

    @staticmethod
    def _blank_overall_stats() -> dict:
        """Returns zeroed accumulators for overall train or eval stats.

        Used to initialize ``overall_train_stats`` and ``overall_eval_stats``. Each call
        to `_update_overall_stats()` appends or increments those variables.
        """
        return dict(
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

    # TODO telemetry: handlers currently closed by BasicGraphMatchingLogger
    # def close(self):
    #     for handler in self.handlers:
    #         handler.close()
    #         handler.close()

    def _consume(self, event: PostEpisodeTelemetry):
        """Processes a single `PostEpisodeTelemetry` event.

        Runs the full post-episode pipeline: updates accumulated episode data, emits an
        `ExperimentStatsTelemetry` snapshot, then forwards to handlers.

        Args:
            event: The event containing post-episode telemetry.
        """
        self._update_episode_data(event)
        self._emit_stats(event)
        self._log_episode(event)

    def _update_episode_data(self, event: PostEpisodeTelemetry):
        """Populates `self.data` with stats from the current episode.

        Mirrors the second half of `BasicGraphMatchingLogger.update_episode_data()`.

        Args:
            event: The event containing post-episode telemetry.
        """
        mode = event.mode
        episode = event.episode

        if not self.lms:  # first time function is called
            for lm in event.lm_stats:
                if lm.startswith("LM_"):
                    self.lms.append(lm)

        self.data["BASIC"][f"{mode}_stats"][episode] = event.lm_stats

        self._update_overall_stats(event)
        overall_stats = self._get_formatted_overall_stats(event)

        # # TODO telemetry: integrate with log level
        self.data["BASIC"][f"{mode}_overall_stats"][episode] = overall_stats
        self.data["BASIC"][f"{mode}_actions"][episode] = event.actions
        self.data["BASIC"][f"{mode}_targets"][episode] = event.targets
        self.data["BASIC"][f"{mode}_timing"][episode] = event.timing

    def _emit_stats(self, event: PostEpisodeTelemetry):
        """Emits an `ExperimentStatsTelemetry` snapshot after each episode.

        Wraps the current `self.data` accumulator in an `ExperimentStatsTelemetry`
        event and snapshots it via `self.telemetry`, emitting it to downstream
        consumers subscribed to ``"experiment_stats"``.

        Args:
            event: The triggering `PostEpisodeTelemetry` event, used as the parent for
                   context field inheritance.
        """
        self.telemetry.snapshot(
            level=self.log_level,
            event=ExperimentStatsTelemetry.from_parent(parent=event, stats=self.data),
        )

    def _log_episode(self, event: PostEpisodeTelemetry):
        """Forwards the accumulated episode data to all handlers.

        Mirrors `BasicGraphMatchingLogger.log_episode()`. Calls
        ``handler.report_episode()`` on each registered handler, then flushes the data
        unless parallel wandb logging is active.

        Args:
            event: The post-episode event supplying ``mode`` and ``episode``.
        """
        mode = event.mode
        episode = event.episode

        for handler in self.handlers:
            handler.report_episode(self.data, self.output_dir, episode, mode)

        if not self.use_parallel_wandb_logging:
            # when logging in parallel to wandb we need to wait with flushing
            # until the parallel run script has retrieved the episode stats.
            self._flush()

    def _flush(self):
        """Resets `self.data` to a blank structure."""
        self.data = self._blank_data()

    def _update_overall_stats(self, event: PostEpisodeTelemetry):
        """Accumulates per-episode stats into the running overall stats dict.

        Mirrors `BasicGraphMatchingLogger.update_overall_stats()`. Iterates over all
        LMs, updating per-LM and per-episode performance counters, rotation errors,
        step counts, symmetry evidence, and goal state stats. Determines the overall
        episode performance by scanning ``performance_options`` in priority order.

        Args:
            event: The event containing post-episode telemetry.
        """
        mode = event.mode
        episode = event.episode

        if mode is ExperimentMode.TRAIN:
            stats = self.overall_train_stats
        else:
            stats = self.overall_eval_stats

        lm_performances = []
        for lm in self.lms:
            # This accumulates stats from all LM
            episode_stats = self.data["BASIC"][f"{mode}_stats"][episode][lm]
            performance = episode_stats["primary_performance"]

            if performance is not None:  # in pre training, performance is None
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
            stats["monty_steps"].append(event.episode_steps)
            stats["monty_matching_steps"].append(event.matching_steps)
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

    def _get_formatted_overall_stats(self, event: PostEpisodeTelemetry) -> dict:
        """Formats the running overall stats into a flat dict for handlers and wandb.

        Mirrors `BasicGraphMatchingLogger.get_formatted_overall_stats()`. Computes
        percentage-based metrics, averages, and wandb histograms (when multiple LMs
        are present) from the accumulated state,

        Args:
            event: The post-episode event supplying ``mode`` and ``episode``.

        Returns:
            A dict of stats suitable to pass directly to wandb or a ``MontyHandler``.
        """
        mode = event.mode
        episode = event.episode

        if mode is ExperimentMode.TRAIN:
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
