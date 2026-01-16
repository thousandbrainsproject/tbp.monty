# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import pytest

pytest.importorskip(
    "habitat_sim",
    reason="Habitat Sim optional dependency not installed.",
)

import tempfile
import unittest
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd

from tests.integration.reproducibility.config import hydra_config
from tests.integration.reproducibility.run import parallel_run, serial_run


def decode_arrays(s: Any, dtype: type) -> Any:
    """Converts a string-represented array to a numpy array.

    Args:
        s: A string-representation of a list/tuple/array.
        dtype: The dtype of the numpy array to return.

    Note:
        Adapted from https://github.com/thousandbrainsproject/tbp.tbs_sensorimotor_intelligence/blob/97636792abbbf19bd16b964617ca28a5fc3eae8b/scripts/data_utils.py#L99.

    Returns:
        A numpy array with the decoded string-represented array. If the input is not a
        string-representation of a list/tuple/array, the input is returned as-is.
    """
    if not isinstance(s, str):
        return s

    # Quick out for empty strings.
    if s == "":
        return ""

    # Remove the outer brackets and parentheses. If it doesn't have
    # brackets or parentheses, it's not an array, so just return it as-is.
    if s.startswith("["):
        s = s.strip("[]")
    elif s.startswith("("):
        s = s.strip("()")
    else:
        return s

    # Split the string into a list of elements.
    if "," in s:
        # list and tuples are comma-separated
        lst = [elt.strip() for elt in s.split(",")]
    else:
        # numpy arrays are space-separated
        lst = s.split()

    # arrays of strings are a special case - can return arrays with dtype 'object',
    # and we also need to strip quotes from each item.
    if np.issubdtype(dtype, np.str_):
        lst = [elt.strip("'\"") for elt in lst]
        if dtype is str:
            return np.array(lst, dtype=object)

        return np.array(lst, dtype=dtype)

    # Must replace 'None' with np.nan for float arrays.
    if np.issubdtype(dtype, np.floating):
        lst = [np.nan if elt == "None" else dtype(elt) for elt in lst]
    return np.array(lst)


def load_eval_stats(path: Path) -> pd.DataFrame:
    """Loads an `eval_stats.csv` file into a pandas dataframe.

    The main purpose of this function is to convert strings of arrays into arrays.
    Some columns contain arrays, but they're loaded as strings, for example:
    "[1.34, 232.33, 123.44]".

    Args:
        path: Path to an `eval_stats.csv` file.

    Note:
        Adapted from https://github.com/thousandbrainsproject/tbp.tbs_sensorimotor_intelligence/blob/97636792abbbf19bd16b964617ca28a5fc3eae8b/scripts/data_utils.py#L38.

    Returns:
        A pandas dataframe with the loaded `eval_stats.csv` file.
    """
    df = pd.read_csv(path)

    float_array_cols = [
        "detected_location",
        "detected_path",
        "detected_rotation",
        "location_rel_body",
        "primary_target_position",
        "primary_target_rotation_euler",
        "primary_target_rotation_quat",
    ]
    column_order = list(df.columns)
    df["result"] = df["result"].replace(np.nan, "")
    df["result"] = df["result"].apply(decode_arrays, args=[str])
    for col_name in float_array_cols:
        df[col_name] = df[col_name].apply(decode_arrays, args=[float])
    return df[column_order]


class EvalEpisodeTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(tempfile.mkdtemp())
        with hydra.initialize(version_base=None, config_path="../../../conf"):
            self.training_config = hydra_config(
                "reproducibility_supervised_training",
                self.output_dir,
                fixed_actions_path=(
                    Path(__file__).parent / "supervised_training_actions.jsonl"
                ),
            )
            parallel_run(self.training_config)

        self.model_name_or_path = (
            Path(self.training_config.experiment.config.logging.output_dir)
            / self.training_config.experiment.config.logging.run_name
            / "pretrained"
        )

    def test_eval_episode_results_are_equal(self):
        with hydra.initialize(version_base=None, config_path="../../../conf"):
            config = hydra_config(
                "reproducibility_eval_episodes",
                self.output_dir,
                model_name_or_path=self.model_name_or_path,
            )
            serial_run(config)
            eval_stats_path = (
                Path(config.experiment.config.logging.output_dir) / "eval_stats.csv"
            )
            scsv = load_eval_stats(eval_stats_path)

            eval_stats_path.unlink()

            config = hydra_config(
                "reproducibility_eval_episodes",
                self.output_dir,
                model_name_or_path=self.model_name_or_path,
            )
            parallel_run(config)
            eval_stats_path = (
                Path(config.experiment.config.logging.output_dir)
                / config.experiment.config.logging.run_name
                / "eval_stats.csv"
            )
            pcsv = load_eval_stats(eval_stats_path)

        # We have to drop these columns because they are not the same in the parallel
        # and serial runs. In particular, 'stepwise_performance' and
        # 'stepwise_target_object' are derived from the mapping between semantic IDs to
        # names which depend on the number of objects in the environment, and
        # environments only have one object in parallel experiments.
        drop_cols = ["time", "stepwise_performance", "stepwise_target_object"]
        scsv = scsv.drop(columns=drop_cols)
        pcsv = pcsv.drop(columns=drop_cols)

        pd.testing.assert_frame_equal(scsv, pcsv)
