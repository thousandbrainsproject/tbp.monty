# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import logging
import os
import pprint
import time

from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict


def merge_args(config, cmd_args=None):
    """Override experiment "config" parameters with command line args.

    Returns:
        Updated config with command line args.
    """
    if not cmd_args:
        return config

    exp_config = copy.deepcopy(config)
    exp_config.update(cmd_args.__dict__)
    del exp_config["experiments"]
    return exp_config


def print_config(config):
    """Print config with nice formatting if config_args.print_config is True."""
    print("\n\n")
    print("Printing config below")
    print("-" * 100)
    print(pprint.pformat(config))
    print("-" * 100)


def run(config):
    exp = config["experiment_class"]()
    with exp:
        exp.setup_experiment(config)

        # TODO: Later will want to evaluate every x episodes or epochs
        # this could probably be solved with just setting the logging freqency
        # Since each trainng loop already does everything that eval does.
        if exp.do_train:
            print("---------training---------")
            exp.train()

        if exp.do_eval:
            print("---------evaluating---------")
            exp.evaluate()


def main(all_configs, experiments=None):
    """Use this as "main" function when running monty experiments.

    A typical project `run.py` shoud look like this::

        # Load all experiment configurations from local project
        from experiments import CONFIGS
        from tbp.monty.frameworks.run import main

        if __name__ == "__main__":
            main(all_configs=CONFIGS)

    Args:
        all_configs: Dict containing all available experiment configurations.
            Usually each project would have its own list of experiment
            configurations
        experiments: Optional list of experiments to run, used to bypass the
            command line args
    """
    cmd_args = None
    if not experiments:
        cmd_parser = create_cmd_parser(experiments=list(all_configs.keys()))
        cmd_args = cmd_parser.parse_args()
        experiments = cmd_args.experiments

        if cmd_args.quiet_habitat_logs:
            os.environ["MAGNUM_LOG"] = "quiet"
            os.environ["HABITAT_SIM_LOG"] = "quiet"

    for experiment in experiments:
        exp = all_configs[experiment]
        exp_config = merge_args(exp, cmd_args)  # TODO: is this really even necessary?
        exp_config = config_to_dict(exp_config)

        # Update run_name and output dir with experiment name
        # NOTE: wandb args are further processed in monty_experiment
        if not exp_config["logging_config"]["run_name"]:
            exp_config["logging_config"]["run_name"] = experiment
        exp_config["logging_config"]["output_dir"] = os.path.join(
            exp_config["logging_config"]["output_dir"],
            exp_config["logging_config"]["run_name"],
        )
        # If we are not running in parallel, this should always be False
        exp_config["logging_config"]["log_parallel_wandb"] = False

        # Print config, including udpates to run name
        if cmd_args is not None:
            if cmd_args.print_config:
                print_config(exp_config)
                continue

        os.makedirs(exp_config["logging_config"]["output_dir"], exist_ok=True)
        start_time = time.time()
        run(exp_config)
        logging.info(f"Done running {experiment} in {time.time() - start_time} seconds")
