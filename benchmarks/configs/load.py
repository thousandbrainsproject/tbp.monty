# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import fields

from benchmarks.configs.names import (
    MontyWorldExperiments,
    MontyWorldHabitatExperiments,
    MyExperiments,
    PretrainingExperiments,
    YcbExperiments,
)

__all__ = ["load_configs", "load_config"]


def load_configs(experiments: list[str]) -> dict:
    """Load the configuration groups for the given experiments.

    Args:
        experiments: The names of the experiments to import the configurations for.

    Returns:
        The imported configuration groups for the given experiments.
    """
    configs = dict()

    for experiment in experiments:
        configs.update(select_config(experiment))

    return configs


def load_config(experiment: str) -> dict:
    """Load the configuration group for the given experiment.

    Args:
        experiment: The name of the experiment to load the configuration for.

    Returns:
        The imported configuration group for the given experiment.
    """
    return select_config(experiment)


def select_config(experiment: str) -> dict:
    """Select and import the configuration group for the given experiment.

    Args:
        experiment: The name of the experiment to import the configuration for.

    Returns:
        The imported configuration group for the given experiment.

    Note:
        We import the configurations selectively to avoid importing uninstalled
        dependencies. For example, if someone does not install a specific simulator,
        then do not import the corresponding configuration group.
    """
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
    my_experiment_names = [field.name for field in fields(MyExperiments)]

    if experiment in monty_world_experiment_names:
        from benchmarks.configs.monty_world_experiments import (
            CONFIGS as MONTY_WORLD,
        )

        return MONTY_WORLD
    elif experiment in monty_world_habitat_experiment_names:
        from benchmarks.configs.monty_world_habitat_experiments import (
            CONFIGS as MONTY_WORLD_HABITAT,
        )

        return MONTY_WORLD_HABITAT
    elif experiment in pretraining_experiment_names:
        from benchmarks.configs.pretraining_experiments import (
            CONFIGS as PRETRAININGS,
        )

        return PRETRAININGS
    elif experiment in ycb_experiment_names:
        from benchmarks.configs.ycb_experiments import CONFIGS as YCB

        return YCB
    elif experiment in my_experiment_names:
        from benchmarks.configs.my_experiments import CONFIGS as MY_EXPERIMENTS

        return MY_EXPERIMENTS
