# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

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
        save_per_episode=True,
        save_consolidated=False,
        episodes_to_save=None,
        parallel_episode_index=None,
    ):
        """Initialize the DetailedJSONHandler.

        Args:
            save_per_episode (bool): If True, save each episode as a separate file
                in an 'episodes/' subdirectory. Defaults to True.
            save_consolidated (bool): If True, also maintain the original behavior
                of appending to a single consolidated file. Defaults to False.
            episodes_to_save (list or None): List of global episode numbers to save. If
                None, all episodes are saved. Defaults to None.
            parallel_episode_index (int or None): Global episode number associated with
                this run when generated via parallel configs. Defaults to None.
        """
        self.report_count = 0
        self.save_per_episode = save_per_episode
        self.save_consolidated = save_consolidated
        self.parallel_episode_index = (
            int(parallel_episode_index) if parallel_episode_index is not None else None
        )
        self.episodes_to_save = (
            None
            if episodes_to_save is None
            else {int(ep) for ep in episodes_to_save}
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

        if self.episodes_to_save is not None:
            if global_total not in self.episodes_to_save:
                print(f"Skipping save for episode {global_total} (not in episodes_to_save list)")
                self.report_count += 1
                return

        output_data[global_total] = copy.deepcopy(stats)
        output_data[global_total].update(data["DETAILED"][local_total])

        # Per-episode saving
        if self.save_per_episode:
            episodes_dir = os.path.join(output_dir, "episodes")
            os.makedirs(episodes_dir, exist_ok=True)

            episode_file = os.path.join(episodes_dir, f"episode_{global_total:06d}.json")
            with open(episode_file, "w") as f:
                json.dump({global_total: output_data[global_total]}, f, cls=BufferEncoder, indent=2)

            print(f"Episode {global_total} saved to {episode_file}")

        # Consolidated saving
        if self.save_consolidated:
            save_stats_path = os.path.join(output_dir, "detailed_run_stats.json")
            maybe_rename_existing_file(save_stats_path, ".json", self.report_count)

            with open(save_stats_path, "a") as f:
                json.dump({global_total: output_data[global_total]}, f, cls=BufferEncoder)
                f.write(os.linesep)

            print("Stats appended to " + save_stats_path)
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
