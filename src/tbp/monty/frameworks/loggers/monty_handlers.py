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

import abc
import copy
import json
import logging
import os
from pathlib import Path
from pprint import pformat
from typing import Container, Literal

from typing_extensions import override

from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.utils.logging_utils import (
    lm_stats_to_dataframe,
    maybe_rename_existing_file,
)

__all__ = ["BasicCSVStatsHandler", "DetailedJSONHandler", "MontyHandler"]

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

    @classmethod
    @abc.abstractmethod
    def log_level(cls):
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
        detailed_episodes_to_save: Container[int] | Literal["all"] = "all",
        detailed_save_per_episode: bool = False,
    ) -> None:
        """Initialize the DetailedJSONHandler.

        Args:
            detailed_episodes_to_save: Container of episodes to save or
                the string ``"all"`` (default) to include every episode.
            detailed_save_per_episode: Whether to save individual episode files or
                consolidate into a single detailed_run_stats.json file.
                Defaults to False.
        """
        self.already_renamed = False
        self.detailed_episodes_to_save = detailed_episodes_to_save
        self.detailed_save_per_episode = detailed_save_per_episode

    @classmethod
    def log_level(cls):
        return "DETAILED"

    def _should_save_episode(self, global_episode_id: int) -> bool:
        """Check if episode should be saved.

        Returns:
            True if episode should be saved, False otherwise.
        """
        return (
            self.detailed_episodes_to_save == "all"
            or global_episode_id in self.detailed_episodes_to_save
        )

    def get_detailed_stats(
        self,
        data,
        global_episode_id: int,
        local_episode: int,
        mode: Literal["train", "eval"],
    ) -> dict:
        """Get detailed episode stats.

        Returns:
            stats: Episode stats.
        """
        output_data = {}

        basic_stats = data["BASIC"][f"{mode}_stats"][local_episode]
        detailed_pool = data["DETAILED"]
        detailed_stats = detailed_pool.get(local_episode)
        if detailed_stats is None:
            detailed_stats = detailed_pool.get(global_episode_id)

        output_data[global_episode_id] = copy.deepcopy(basic_stats)
        output_data[global_episode_id].update(detailed_stats)

        return output_data

    def report_episode(self, data, output_dir, local_episode, mode="train", **kwargs):
        """Report episode data."""
        global_episode_id = kwargs[f"{mode}_episodes_to_total"][local_episode]

        if not self._should_save_episode(global_episode_id):
            logger.debug(
                "Skipping detailed JSON for episode %s (not requested)",
                global_episode_id,
            )
            return

        stats = self.get_detailed_stats(data, global_episode_id, local_episode, mode)

        if self.detailed_save_per_episode:
            self._save_per_episode(output_dir, global_episode_id, stats)
        else:
            self._save_all(global_episode_id, output_dir, stats)

    def _save_per_episode(self, output_dir: str, global_episode_id: int, stats: dict):
        """Save detailed stats for a single episode.

        Args:
            output_dir: Directory where results are written.
            global_episode_id: Combined train+eval episode id used for DETAILED logs.
            stats: Dictionary containing episode stats keyed by global episode id.
        """
        episodes_dir = Path(output_dir) / "detailed_run_stats"
        episodes_dir.mkdir(exist_ok=True, parents=True)

        episode_file = episodes_dir / f"episode_{global_episode_id:06d}.json"
        maybe_rename_existing_file(episode_file)

        with episode_file.open("w") as f:
            json.dump(
                {global_episode_id: stats[global_episode_id]},
                f,
                cls=BufferEncoder,
            )

        logger.debug(
            "Saved detailed JSON for episode %s to %s",
            global_episode_id,
            str(episode_file),
        )

    def _save_all(self, global_episode_id: int, output_dir: str, stats: dict):
        """Save detailed stats for all episodes."""
        save_stats_path = Path(output_dir) / "detailed_run_stats.json"
        if not self.already_renamed:
            maybe_rename_existing_file(save_stats_path)
            self.already_renamed = True

        with save_stats_path.open("a") as f:
            json.dump(
                {global_episode_id: stats[global_episode_id]},
                f,
                cls=BufferEncoder,
            )
            f.write(os.linesep)

        logger.debug(
            "Appended detailed stats for episode %s to %s",
            global_episode_id,
            str(save_stats_path),
        )

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

    @override
    def report_episode(self, data, output_dir, episode, mode="train", **kwargs):
        # episode is ignored when reporting stats to CSV
        # Look for train_stats or eval_stats under BASIC logs
        basic_logs = data["BASIC"]
        mode_key = f"{mode}_stats"
        output_file = Path(output_dir) / f"{mode}_stats.csv"
        stats = basic_logs.get(mode_key, {})
        logger.debug(pformat(stats))

        # Remove file if it existed before to avoid appending to previous results file
        if output_file not in self.reports_per_file:
            self.reports_per_file[output_file] = 0
            maybe_rename_existing_file(output_file)
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
