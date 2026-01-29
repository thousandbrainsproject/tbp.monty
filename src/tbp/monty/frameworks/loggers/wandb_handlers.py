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

    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
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

    def report_episode(self, data, output_dir, mode="train", **kwargs):
        pass

    def close(self):
        pass


class BasicWandbTableStatsHandler(WandbHandler):
    """Log LM episode stats to wandb as tables."""

    @classmethod
    def log_level(cls):
        return "BASIC"

    @override
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
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

    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        super().report_episode(data, output_dir, episode, mode, **kwargs)
        basic_logs = data["BASIC"]
        # Get actions depending on mode (train or eval)
        action_key = f"{mode}_actions"
        action_data = basic_logs.get(action_key, {})

        assert len(action_data) == 1, "expected data for exactly one episode"
        # Log one table of actions per episode
        # for episode in action_data.keys():
        # TODO: is table the best format for this?

        episode = list(action_data.keys())[0]
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

    @classmethod
    def log_level(cls):
        return "BASIC"

    @override
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        basic_logs = data["BASIC"]
        mode_key = f"{mode}_overall_stats"
        stats = basic_logs.get(mode_key, {})
        wandb.log(stats[episode], step=episode, commit=False)

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
        sm_ids = [sm for sm in episode_stats.keys() if sm.startswith("SM_")]
        for sm in sm_ids:
            observations = episode_stats[sm]["raw_observations"]
            frames_per_sm[sm] = get_rgba_frames_single_sm(observations)

        return frames_per_sm

    @override
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
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
