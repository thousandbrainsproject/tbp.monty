# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import abc
import copy
import json
import logging
import os
from pprint import pformat

from tbp.monty.frameworks.actions.actions import ActionJSONEncoder
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.utils.logging_utils import (
    lm_stats_to_dataframe,
    maybe_rename_existing_file,
)

logger = logging.getLogger(__name__)

###
# Template for MontyHandler
###


class MontyHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def report_episode(self, **kwargs):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractclassmethod
    def log_level(self):
        """Handlers filter information from the data they receive.

        This class method specifies the level they filter at.
        """
        pass


###
# Handler classes
###


class DetailedJSONHandler(MontyHandler):
    """Grab any logs at the DETAILED level and append to a json file."""

    def __init__(
        self,
        episodes_to_save: list[int] | None = None,
        parallel_episode_index: int | None = None,
        save_per_episode: bool = False,
    ) -> None:
        """Initialize the DetailedJSONHandler.

        Args:
            episodes_to_save: List of episodes to save. If None, all episodes are
                appended to a consolidated file called detailed_run_stats.json.
            parallel_episode_index: Global episode number associated with
                this run when generated via parallel configs. Defaults to None.
            save_per_episode: Whether to save individual episode files or
                consolidate into a single detailed_run_stats.json file.
                Defaults to False.
        """
        self.report_count = 0
        self.saved_episode_count = 0
        self.parallel_episode_index = parallel_episode_index
        self.episodes_to_save = episodes_to_save
        self.save_per_episode = save_per_episode
        if self.episodes_to_save is not None and len(self.episodes_to_save) == 0:
            logger.info(
                "episodes_to_save is empty; no detailed episode files will be written"
            )

    @classmethod
    def log_level(cls):
        return "DETAILED"

    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        """Report episode data.

        Changed name to report episode since we are currently running with
        reporting and flushing exactly once per episode.
        """
        output_data = {}
        if mode == "train":
            local_total = kwargs["train_episodes_to_total"][episode]
            stats = data["BASIC"]["train_stats"][episode]

        elif mode == "eval":
            local_total = kwargs["eval_episodes_to_total"][episode]
            stats = data["BASIC"]["eval_stats"][episode]

        global_total = (
            self.parallel_episode_index
            if self.parallel_episode_index is not None
            else local_total
        )

        save_individual = self.save_per_episode and (
            self.episodes_to_save is None or global_total in self.episodes_to_save
        )

        if (
            self.episodes_to_save is not None
            and global_total not in self.episodes_to_save
        ):
            logger.debug(
                "Skipping detailed JSON for episode %s (not requested)", global_total
            )
            self.report_count += 1
            return

        output_data[global_total] = copy.deepcopy(stats)
        output_data[global_total].update(data["DETAILED"][local_total])

        if save_individual:
            episodes_dir = os.path.join(output_dir, "detailed_run_stats")
            os.makedirs(episodes_dir, exist_ok=True)

            episode_file = os.path.join(
                episodes_dir, f"episode_{global_total:06d}.json"
            )
            maybe_rename_existing_file(
                episode_file, ".json", 0 if self.saved_episode_count == 0 else 1
            )
            with open(episode_file, "w") as f:
                json.dump(
                    {global_total: output_data[global_total]},
                    f,
                    cls=BufferEncoder,
                    indent=2,
                )

            logger.debug(
                "Saved detailed JSON for episode %s to %s", global_total, episode_file
            )
            self.saved_episode_count += 1

        else:
            save_stats_path = os.path.join(output_dir, "detailed_run_stats.json")
            maybe_rename_existing_file(
                save_stats_path, ".json", 0 if self.saved_episode_count == 0 else 1
            )

            with open(save_stats_path, "a") as f:
                json.dump(
                    {global_total: output_data[global_total]}, f, cls=BufferEncoder
                )
                f.write(os.linesep)

            logger.debug(
                "Appended detailed stats for episode %s to %s",
                global_total,
                save_stats_path,
            )
            self.saved_episode_count += 1
        self.report_count += 1

    def close(self):
        pass


class BasicCSVStatsHandler(MontyHandler):
    """Grab any logs at the BASIC level and append to train or eval CSV files."""

    @classmethod
    def log_level(cls):
        return "BASIC"

    def __init__(self):
        """Initialize with empty dictionary to keep track of writes per file.

        We only want to include the header the first time we write to a file. This
        keeps track of writes per file so we can format the file properly.
        """
        self.reports_per_file = {}

    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        # Look for train_stats or eval_stats under BASIC logs
        basic_logs = data["BASIC"]
        mode_key = f"{mode}_stats"
        output_file = os.path.join(output_dir, f"{mode}_stats.csv")
        stats = basic_logs.get(mode_key, {})
        logger.debug(pformat(stats))

        # Remove file if it existed before to avoid appending to previous results file
        if output_file not in self.reports_per_file:
            self.reports_per_file[output_file] = 0
            maybe_rename_existing_file(output_file, ".csv", 0)
        else:
            self.reports_per_file[output_file] += 1

        # Format stats for a single episode as a dataframe
        dataframe = lm_stats_to_dataframe(stats)
        # Move most relevant columns to front
        if "most_likely_object" in dataframe:
            top_columns = [
                "primary_performance",
                "stepwise_performance",
                "num_steps",
                "rotation_error",
                "result",
                "most_likely_object",
                "primary_target_object",
                "stepwise_target_object",
                "highest_evidence",
                "time",
                "symmetry_evidence",
                "monty_steps",
                "monty_matching_steps",
                "individual_ts_performance",
                "individual_ts_reached_at_step",
                "primary_target_position",
                "primary_target_rotation_euler",
                "most_likely_rotation",
            ]
        else:
            top_columns = [
                "primary_performance",
                "stepwise_performance",
                "num_steps",
                "rotation_error",
                "result",
                "primary_target_object",
                "stepwise_target_object",
                "time",
                "symmetry_evidence",
                "monty_steps",
                "monty_matching_steps",
                "primary_target_position",
                "primary_target_rotation_euler",
            ]
        dataframe = self.move_columns_to_front(
            dataframe,
            top_columns,
        )

        # Only include header first time you write to this file
        header = self.reports_per_file[output_file] < 1
        dataframe.to_csv(output_file, mode="a", header=header)

    def move_columns_to_front(self, df, columns):
        for c_key in reversed(columns):
            df.insert(0, c_key, df.pop(c_key))
        return df

    def close(self):
        pass


class ReproduceEpisodeHandler(MontyHandler):
    @classmethod
    def log_level(cls):
        return "BASIC"

    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        # Set up data directory with reproducibility info for each episode
        if not hasattr(self, "data_dir"):
            self.data_dir = os.path.join(output_dir, "reproduce_episode_data")
            os.makedirs(self.data_dir, exist_ok=True)

        # TODO: store a pointer to the training model
        # something like if train_epochs == 0:
        #   use model_name_or_path
        # else:
        #   get checkpoint of most up to date model

        # Write data to action file
        action_file = f"{mode}_episode_{episode}_actions.jsonl"
        action_file_path = os.path.join(self.data_dir, action_file)
        actions = data["BASIC"][f"{mode}_actions"][episode]
        with open(action_file_path, "w") as f:
            for action in actions:
                f.write(f"{json.dumps(action[0], cls=ActionJSONEncoder)}\n")

        # Write data to object params / targets file
        object_file = f"{mode}_episode_{episode}_target.txt"
        object_file_path = os.path.join(self.data_dir, object_file)
        target = data["BASIC"][f"{mode}_targets"][episode]
        with open(object_file_path, "w") as f:
            json.dump(target, f, cls=BufferEncoder)

    def close(self):
        pass
