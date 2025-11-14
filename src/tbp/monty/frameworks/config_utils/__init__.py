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


def shrink_config(config: dict) -> dict:
    """Shrink the run size of a given experiment configuration.

    Reduce the number of epochs, objects, etc. to speed up an experiment. This
    is useful for testing.

    Args:
        config: The configuration dictionary to shrink.

    Returns:
        The shrunk configuration dictionary.
    """
    new_config = copy.deepcopy(config)

    new_config["experiment_args"].update(
        n_eval_epochs=1,
        n_train_epochs=1,
        max_train_steps=1,
        max_eval_steps=1,
        max_total_steps=2,
    )

    new_config["monty_config"]["monty_args"]["num_exploratory_steps"] = 1

    object_names = new_config["eval_env_interface_args"]["object_names"]
    if isinstance(object_names, list):
        _shrink_config_list(new_config["eval_env_interface_args"], "object_names")
    else:
        _shrink_config_list(object_names, "source_object_list")
        _shrink_config_list(object_names, "targets_list")
        object_names["num_distractors"] = 1

    return new_config


def _shrink_config_list(config, key, max_length=2):
    """Mutating helper to shrink lists in config dicts."""
    if key in config:
        config[key] = config[key][:max_length]
