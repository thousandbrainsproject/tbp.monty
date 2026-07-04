# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import wandb
from typing_extensions import override

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.loggers.monty_handlers import MontyHandler
from tbp.monty.frameworks.utils.logging_utils import (
    format_columns_for_wandb,
    get_rgba_frames_single_sm,
    lm_stats_to_dataframe,
)
from tbp.monty.frameworks.utils.plot_utils import mark_obs


class WandbWrapper(MontyHandler):
    """Container for wandb handlers.

    Loops over a series of handlers which log different information without committing
    (sending it to wandb).

    The wrapper finally commits all logs at once. This allows us to maintain control
    over the wandb global step. This class assumes reporting takes place once per
    episode; hence the wandb handlers have `report_episode` methods.
    """

    def __init__(
        self,
        wandb_handlers: list,
        run_name: str,
        wandb_group: str | None = None,
        config: dict | None = None,
        resume_wandb_run: bool = False,
        wandb_id: str | None = None,
    ):
        self.name = run_name
        self.group = wandb_group
        self.config = config
        self.wandb_logger = wandb.init(
            name=self.name,
            group=self.group,
            project="Monty",
            config=config,
            resume=resume_wandb_run,
            id=wandb_id,
        )
        self.wandb_handlers = [wandb_handler() for wandb_handler in wandb_handlers]

    def report_episode(
        self,
        data,
        output_dir,
        episode,
        mode: ExperimentMode = ExperimentMode.TRAIN,
        **kwargs,
    ):
        for handler in self.wandb_handlers:
            handler.report_episode(data, output_dir, episode, mode=mode, **kwargs)

        wandb.log({})  # TODO: What is this for?

    @classmethod
    def log_level(cls):
        return ""

    def close(self):
        self.wandb_logger.finish()


class WandbHandler(MontyHandler):
    """Parent class for wandb loggers."""

    def __init__(self):
        self.report_count = 0
        self.variable_length_columns = [
            "possible_object_poses",
            "possible_object_locations",
            "possible_object_sources",
            "possible_match_sources",
            "detected_path",
        ]
        self.post_init()

    def post_init(self):
        """Handle additional initialization for subclasses.

        Call this to handle any additional initialization for subclasses not
        covered by `WandbHandler`.
        """
        pass

    @classmethod
    def log_level(cls):
        return ""

    def report_episode(
        self, data, output_dir, mode: ExperimentMode = ExperimentMode.TRAIN, **kwargs
    ):
        pass

    def close(self):
        pass


class BasicWandbTableStatsHandler(WandbHandler):
    """Log LM episode stats to wandb as tables."""

    @classmethod
    def log_level(cls):
        return "BASIC"

    @override
    def report_episode(
        self,
        data,
        output_dir,
        episode,
        mode: ExperimentMode = ExperimentMode.TRAIN,
        **kwargs,
    ):
        ###
        # Log basic statistics
        # Ignore the episode value
        ###

        # Get stats data depending on mode (train or eval)
        basic_logs = data["BASIC"]
        mode_key = f"{mode}_stats"
        stats_table = f"{mode}_stats_table"
        stats = basic_logs.get(mode_key, {})

        # if len(stats) > 0:
        df = lm_stats_to_dataframe(stats, format_for_wandb=True)

        # Filter df to only include columns without variable length entries, like
        # possible_object_poses
        const_columns = list(set(df.columns) - set(self.variable_length_columns))
        const_df = df[const_columns]

        # shorthand for self.train_table = df or self.eval_table = df
        if not hasattr(self, stats_table):
            setattr(self, stats_table, const_df)
        else:  # Don't log first episode twice
            setattr(
                self,
                stats_table,
                pd.concat([getattr(self, stats_table), const_df]),
            )
        # print(getattr(self, stats_table))
        table = wandb.Table(dataframe=getattr(self, stats_table))
        wandb.log({stats_table: table}, commit=False)
        self.report_count += 1


class DetailedWandbTableStatsHandler(BasicWandbTableStatsHandler):
    """Log LM stats and actions to wandb as tables.

    This modified version of BasicWandbTableStatsHandler also logs the actions executed
    in each episode to wandb as tables (one table per episode).
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def log_level(cls):
        return "DETAILED"

    def report_episode(
        self,
        data,
        output_dir,
        episode,
        mode: ExperimentMode = ExperimentMode.TRAIN,
        **kwargs,
    ):
        super().report_episode(data, output_dir, episode, mode, **kwargs)
        basic_logs = data["BASIC"]
        # Get actions depending on mode (train or eval)
        action_key = f"{mode}_actions"
        action_data = basic_logs.get(action_key, {})

        assert len(action_data) == 1, "expected data for exactly one episode"
        # Log one table of actions per episode
        # for episode in action_data.keys():
        # TODO: is table the best format for this?

        episode = next(iter(action_data.keys()))
        table_name = f"{mode}_actions/episode_{episode}_table"
        actions = action_data[episode]
        for i, action in enumerate(actions):
            a = action[0]
            if a is not None:
                o = {}
                for key, value in dict(a).items():
                    if key in {"action", "agent_id"}:
                        continue  # don't duplicate action or agent_id in "params"
                    if isinstance(value, np.ndarray):
                        o[key] = value.tolist()
                    else:
                        o[key] = value
                actions[i][0] = {
                    f"{a.agent_id}": {"action": a.name, "params": json.dumps(o)}
                }
        actions_df = pd.DataFrame(actions)
        table = wandb.Table(dataframe=actions_df)
        wandb.log({table_name: table}, step=episode)


class BasicWandbChartStatsHandler(WandbHandler):
    """Log LM episode stats to wandb with one chart per measure."""

    def post_init(self):
        self.unsupervised_stats_by_mode = {}

    @classmethod
    def log_level(cls):
        return "BASIC"

    @override
    def report_episode(
        self,
        data,
        output_dir,
        episode,
        mode: ExperimentMode = ExperimentMode.TRAIN,
        **kwargs,
    ):
        basic_logs = data["BASIC"]
        mode_key = f"{mode}_overall_stats"
        stats = basic_logs.get(mode_key, {})

        wandb_stats = dict(stats[episode])
        wandb_stats.update(
            self.get_unsupervised_benchmark_stats(basic_logs, episode, mode)
        )

        wandb.log(wandb_stats, step=episode, commit=False)

    def get_unsupervised_benchmark_stats(self, basic_logs, episode, mode):
        """Log benchmark-style unsupervised learning stats to WandB.

        These stats mirror the metrics reported by print_unsupervised_stats:
        first-epoch correctness, post-first-epoch correctness, graph/object
        association stats, runtime, and average episode runtime.

        This method only reads existing BASIC logging data and does not change
        experiment behavior.

        Returns:
            A dictionary of unsupervised benchmark statistics to log to WandB.
        """
        if mode is not ExperimentMode.TRAIN:
            return {}

        mode_key = f"{mode}_stats"
        stats = basic_logs.get(mode_key, {})
        if len(stats) == 0:
            return {}

        episode_df = lm_stats_to_dataframe(stats, format_for_wandb=True)
        if episode_df.empty:
            return {}

        episode_df["episode"] = episode

        stats_table = f"{mode}_unsupervised_benchmark_stats"
        if stats_table not in self.unsupervised_stats_by_mode:
            self.unsupervised_stats_by_mode[stats_table] = episode_df
        else:
            self.unsupervised_stats_by_mode[stats_table] = pd.concat(
                [self.unsupervised_stats_by_mode[stats_table], episode_df],
                ignore_index=True,
            )

        stats_df = self.unsupervised_stats_by_mode[stats_table]

        required_columns = {
            "primary_performance",
            "primary_target_object",
            "mean_objects_per_graph",
            "mean_graphs_per_object",
            "time",
            "lm_id",
            "episode",
        }
        if not required_columns.issubset(stats_df.columns):
            return {}

        # The current unsupervised benchmarks use one LM. If multiple LMs are present,
        # use LM_0 to avoid counting one episode multiple times.
        lm_stats = stats_df[stats_df["lm_id"] == "LM_0"].copy()
        if lm_stats.empty:
            lm_stats = stats_df.drop_duplicates(subset="episode").copy()

        lm_stats = lm_stats.sort_values("episode").drop_duplicates(subset="episode")
        if lm_stats.empty:
            return {}

        latest_stats = lm_stats.iloc[-1]
        unsupervised_stats = {
            "unsupervised/mean_objects_per_graph": latest_stats[
                "mean_objects_per_graph"
            ],
            "unsupervised/mean_graphs_per_object": latest_stats[
                "mean_graphs_per_object"
            ],
        }

        epoch_len = self.infer_epoch_len(lm_stats)
        if epoch_len is not None:
            first_epoch_stats = lm_stats.iloc[:epoch_len]
            later_epoch_stats = lm_stats.iloc[epoch_len:]

            first_epoch_new = first_epoch_stats[
                first_epoch_stats["primary_performance"] == "no_match"
            ]
            unsupervised_stats["unsupervised/correct_first_epoch_percent"] = (
                len(first_epoch_new) / len(first_epoch_stats) * 100
            )

            if len(later_epoch_stats) > 0:
                later_correct = later_epoch_stats[
                    later_epoch_stats["primary_performance"].isin(
                        ["correct", "correct_mlh"]
                    )
                ]
                unsupervised_stats["unsupervised/correct_after_first_epoch_percent"] = (
                    len(later_correct) / len(later_epoch_stats) * 100
                )

        run_time_seconds = pd.to_numeric(lm_stats["time"]).sum()
        unsupervised_stats["unsupervised/run_time_minutes"] = run_time_seconds / 60
        unsupervised_stats["unsupervised/episode_run_time_seconds"] = (
            run_time_seconds / len(lm_stats)
        )

        return unsupervised_stats

    def infer_epoch_len(self, stats):
        """Infer epoch length from the first repeated target object.

        Unsupervised benchmark tables are computed with print_unsupervised_stats
        using a known epoch length. The WandB handler does not currently receive
        epoch_len directly, so infer it from the first repeated target object.

        Returns:
            The inferred epoch length, or None if it cannot be inferred.
        """
        seen_targets = set()
        for idx, target in enumerate(stats["primary_target_object"]):
            if pd.isna(target):
                continue
            if target in seen_targets:
                return idx
            seen_targets.add(target)

        return None

    def get_safe_columns_per_lm(self, stats):
        """Format each episode by looping over learning modules and formatting each one.

        Args:
            stats: dict ~ {LM_0: dict, LM_1: dict}

        Returns:
            The formatted stats.
        """
        safe_stats = {}
        for lm, lm_dict in stats.items():
            safe_lm_dict = {
                lm_col: lm_val
                for lm_col, lm_val in lm_dict.items()
                if lm_col not in self.variable_length_columns
            }
            formatted_lm_dict = format_columns_for_wandb(safe_lm_dict)
            safe_stats[lm] = formatted_lm_dict

        return safe_stats


class DetailedWandbHandler(WandbHandler):
    """Make animations from sequences of observations on wandb.

    NOTE: Not yet generalized for different model architectures. This assumes SM_0 is
    the patch and SM_1 is the view finder.
    """

    def post_init(self):
        self.report_key = "raw_rgba"

    def get_episode_frames(self, episode_stats):
        frames_per_sm = {}
        sm_ids = [sm for sm in episode_stats if sm.startswith("SM_")]
        for sm in sm_ids:
            observations = episode_stats[sm]["raw_observations"]
            frames_per_sm[sm] = get_rgba_frames_single_sm(observations)

        return frames_per_sm

    @override
    def report_episode(
        self,
        data,
        output_dir,
        episode,
        mode: ExperimentMode = ExperimentMode.TRAIN,
        **kwargs,
    ):
        # mode is ignored when reporting this episode

        detailed_stats = data["DETAILED"]
        frames_per_sm = self.get_episode_frames(detailed_stats[episode])
        for sm, frames in frames_per_sm.items():
            wandb.log(
                {
                    f"episode_{episode}_{self.report_key}_{sm}": wandb.Video(
                        frames, format="gif"
                    )
                },
                step=episode,
                commit=False,
            )


class DetailedWandbMarkedObsHandler(DetailedWandbHandler):
    """Just like DetailedWandbHandler, but use fancier observations.

    NOTE: This assumes SM_1 and SM_0 are the view finder and patch modules respectively,
          meaning this logger is specific to the model architecture.
    NOTE: This is slow, adding a few seconds per function call. The intended use
          case is for debugging and error analysis, so speed should not be an issue
          when the number of episodes is small, but probably do not use this if you are
          running a large number of experiments.
    """

    def post_init(self):
        self.report_key = "marked_obs"

    def get_episode_frames(self, episode_stats):
        frame_key = "patch_view"
        frame_dict = {frame_key: []}
        for step in range(len(episode_stats["SM_1"]["raw_observations"])):
            viz_obs = episode_stats["SM_1"]["raw_observations"][step]
            patch_obs = episode_stats["SM_0"]["raw_observations"][step]
            frame = mark_obs(viz_obs, patch_obs)
            wandb_frame = np.moveaxis(frame, [0, 1, 2], [1, 2, 0])  # format for wandb
            frame_dict[frame_key].append(wandb_frame)

        frame_dict[frame_key] = np.array(frame_dict[frame_key])
        return frame_dict
