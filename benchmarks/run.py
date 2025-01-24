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
from dataclasses import fields

from configs.names import (
    NAMES,
    MontyWorldExperiments,
    MontyWorldHabitatExperiments,
    PretrainingExperiments,
    YcbExperiments,
)

from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser
from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.run import main  # noqa: E402

if __name__ == "__main__":
    cmd_args = None
    cmd_parser = create_cmd_parser(experiments=NAMES)
    cmd_args = cmd_parser.parse_args()
    experiments = cmd_args.experiments

    if cmd_args.quiet_habitat_logs:
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    monty_world_experiment_names = [
        field.name for field in fields(MontyWorldExperiments)
    ]
    monty_world_habitat_experiment_names = [
        field.name for field in fields(MontyWorldHabitatExperiments)
    ]
    pretraining_experiment_names = [
        field.name for field in fields(PretrainingExperiments)
    ]
    ycb_experiment_names = [field.name for field in fields(YcbExperiments)]

    CONFIGS = dict()
    for experiment in experiments:
        if experiment in monty_world_experiment_names:
            from configs.monty_world_experiments import CONFIGS as MONTY_WORLD

            CONFIGS.update(MONTY_WORLD)
        elif experiment in monty_world_habitat_experiment_names:
            from configs.monty_world_habitat_experiments import (
                CONFIGS as MONTY_WORLD_HABITAT,
            )

            CONFIGS.update(MONTY_WORLD_HABITAT)
        elif experiment in pretraining_experiment_names:
            from configs.pretraining_experiments import CONFIGS as PRETRAININGS

            CONFIGS.update(PRETRAININGS)
        elif experiment in ycb_experiment_names:
            from configs.ycb_experiments import CONFIGS as YCB

            CONFIGS.update(YCB)

    main(all_configs=CONFIGS, experiments=cmd_args.experiments)
