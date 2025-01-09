# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import os
import pathlib
import pickle

import pandas as pd

# Load all experiment configurations from local project
from configs import CONFIGS

from tbp.monty.frameworks.config_utils.cmd_parser import create_rerun_parser
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict
from tbp.monty.frameworks.utils.follow_up_configs import (
    create_eval_config_multiple_episodes,
    recover_output_dir,
)

"""
Script to analyse results from an experiment, find episodes to rerun, generate the
associated configs with detailed logging, and supply the command to run them.

Run with `python make_detailed_follow_up_configs.py -e old_exp --episodes 3 1 4`
This will output two things:

1) It will pickle configs for re-running select episodes in configs/follow_ups
2) It will print the command to actually execute these experiments

The --episodes flag is optional, and defaults to any episodes with incorrect results
from the model (e.g. time out, confused, not recognied)

The intended use case is so you can run an experiment with many episodes without
detailed logging, which is expensive and memory intensive. Then, you can easily spot
check failed episodes with this script, for example by finding the exact time step
where the correct object is ruled out, or by watching animations to gather context.

NOTE, if you do not specify which episodes to rerun, this script will look for
eval_stats.csv, the output of BasicCSVStatsHandler. In this case, make sure the original
experiment uses it.
"""

if __name__ == "__main__":
    cmd_parser = create_rerun_parser(all_configs=CONFIGS)
    cmd_args = cmd_parser.parse_args()
    experiment = cmd_args.experiment
    follow_up_suffix = cmd_args.name
    rerun_episodes = cmd_args.episodes

    # Load results from experiment and find episodes of interest
    config = CONFIGS[experiment]
    config = config_to_dict(config)
    output_dir = recover_output_dir(config, experiment)

    if not rerun_episodes:
        # Find all episodes that failed (e.g. time out, confused, not recognized)
        eval_stats_file = os.path.join(output_dir, "eval_stats.csv")
        if not os.path.exists(eval_stats_file):
            raise FileNotFoundError(
                f"Could not find eval_stats.csv at location {eval_stats_file}"
                "Note, this script selects episodes to re-run based on output of"
                "`BasicCSVStatsHandler`. Make sure you add this to your config to use"
                "this script."
            )
        eval_stats = pd.read_csv(eval_stats_file)
        rerun_episodes = eval_stats.index[eval_stats["performance"] != "correct"]
        if len(rerun_episodes) == 0:
            raise ValueError(
                f"No episodes with incorrect results found in {eval_stats_file}"
            )
        print(f"\n\nFound {len(rerun_episodes)} episodes with incorrect results")

    # Generate single follow up config for error analysis in experiments/follow_ups
    follow_up_config = create_eval_config_multiple_episodes(
        config, experiment, rerun_episodes
    )

    # Pickle these configs in the follow_ups directory
    follow_up_dir = pathlib.Path(__file__).parent / "configs/follow_ups"
    follow_up_name = "_".join([experiment, follow_up_suffix])
    file_path = os.path.join(follow_up_dir, follow_up_name + ".pkl")
    with open(file_path, "wb") as f:
        pickle.dump(follow_up_config, f)

    print("\n\n")
    print(f"Config to rerun all episodes saved in {follow_up_dir}/{follow_up_name}")
    print("-" * 50)
    print("\n\n")

    # Print out the run.py -e command to run all follow ups
    command = f"python run.py -e {follow_up_name}"
    command += "\n\n"
    print("To rerun all episodes in a single experiment, use this command \n")
    print(command)
    print("To print the config to verify it is correct, add the -p flag")

    # Possible optimizations:
    # -- output dir is being parsed here and inside create_eval_episode, refactor
    # -- use wandb resume_run option
