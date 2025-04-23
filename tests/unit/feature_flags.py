# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
from typing import Dict

from tbp.monty.frameworks.config_utils.config_args import Dataclass
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    InformedEnvironmentDataLoaderEvalArgs,
    InformedEnvironmentDataLoaderTrainArgs,
)


def to_dict(maybe_dataclass: Dataclass | Dict) -> dict:
    """Convert a dataclass or dict to a dict.

    Args:
        maybe_dataclass: The object to convert to a dict.

    Returns:
        dict: The dict.
    """
    return asdict(maybe_dataclass) if is_dataclass(maybe_dataclass) else maybe_dataclass


def create_config_with_get_good_view_positioning_procedure(config):
    """Creates a duplicate configuration testing GetGoodView positioning procedure.

    Args:
        config (dict): The configuration to duplicate.

    Returns:
        dict: A duplicate of the configuration with the
            use_get_good_view_positioning_procedure feature flag enabled.
    """
    config_with_get_good_view_positioning_procedure = copy.deepcopy(config)
    if hasattr(config, "eval_dataloader_args"):
        eval_dataloader_args = copy.deepcopy(to_dict(config["eval_dataloader_args"]))
        eval_dataloader_args["use_get_good_view_positioning_procedure"] = True
        config_with_get_good_view_positioning_procedure["eval_dataloader_args"] = (
            InformedEnvironmentDataLoaderEvalArgs(**eval_dataloader_args)
        )
    train_dataloader_args = copy.deepcopy(to_dict(config["train_dataloader_args"]))
    train_dataloader_args["use_get_good_view_positioning_procedure"] = True
    config_with_get_good_view_positioning_procedure["train_dataloader_args"] = (
        InformedEnvironmentDataLoaderTrainArgs(**train_dataloader_args)
    )
    return config_with_get_good_view_positioning_procedure
