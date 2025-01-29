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
import sys

# Benchmarks is a scripts folder. However, we want to reuse and import
# scripts within the benchmarks folder and externally. This is done by adding
# the benchmarks folder to the system path and using fully qualified module
# names like benchmarks.configs.names when importing.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.expanduser(os.path.realpath(__file__))))
)

from dataclasses import fields

from benchmarks.configs.names import (
    NAMES,
    MontyWorldExperiments,
    MontyWorldHabitatExperiments,
    PretrainingExperiments,
    YcbExperiments,
)
from tbp.monty.frameworks.config_utils.cmd_parser import create_cmd_parser_parallel
from tbp.monty.frameworks.run_env import setup_env

setup_env()

from tbp.monty.frameworks.run_parallel import main  # noqa: E402

if __name__ == "__main__":
    cmd_args = None
    cmd_parser = create_cmd_parser_parallel(experiments=NAMES)
    cmd_args = cmd_parser.parse_args()
    experiment = cmd_args.experiment
    num_parallel = cmd_args.num_parallel
    quiet_habitat_logs = cmd_args.quiet_habitat_logs
    print_cfg = cmd_args.print_cfg
    is_unittest = False

    if quiet_habitat_logs:
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

    main(
        all_configs=CONFIGS,
        exp=CONFIGS[experiment],
        experiment=experiment,
        num_parallel=num_parallel,
        quiet_habitat_logs=quiet_habitat_logs,
        print_cfg=print_cfg,
        is_unittest=is_unittest,
    )
