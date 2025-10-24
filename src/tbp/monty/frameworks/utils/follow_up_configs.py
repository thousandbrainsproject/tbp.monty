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
import json
import os

import torch

from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    PredefinedObjectInitializer,
)
from tbp.monty.frameworks.loggers.monty_handlers import DetailedJSONHandler
from tbp.monty.frameworks.loggers.wandb_handlers import DetailedWandbMarkedObsHandler
from tbp.monty.frameworks.utils.dataclass_utils import config_to_dict


def recover_output_dir(config, config_name):
    """Update output directory based on how run.py modifies on the fly.

    Returns:
        output_dir: Path to output directory.
    """
    output_dir = config["logging_config"]["output_dir"]
    if not config["logging_config"]["run_name"]:
        output_dir = os.path.join(config["logging_config"]["output_dir"], config_name)

    return output_dir


def recover_run_name(config, config_name):
    if not config["logging_config"]["run_name"]:
        return config_name
    else:
        return config["logging_config"]["run_name"]


def recover_wandb_id(output_dir):
    # 0 is just go specify any epoch subdirectory; ids should all be the same
    config_file_name = os.path.join(output_dir, "0", "config.pt")
    cfg = torch.load(config_file_name)
    config = config_to_dict(cfg)
    wandb_id = config["logging_config"]["wandb_id"]

    return wandb_id


def create_eval_episode_config(
    parent_config, parent_config_name, episode, update_run_dir=True
):
    """Generate a new config that runs only that episode with detailed logging.

    Inputs are the config for a previous experiment, and the episode we want to re-run.

    Args:
        parent_config: actual config used to run previous experiment
        parent_config_name: str key into configs with name of experiment to reproduce
        episode: int index of episode we want to reproduce
        update_run_dir: bool, update run directory or not based on of run.py modified

    Note:
        For now, assume we just care about re-running an eval episode

    Note:
        For now, assume dataloader_class is a subclass of
              `EnvironmentDataLoaderPerObject`

    Returns:
        Config for re-running the specified episode.
    """
    # Create config that loads latest checkpoint and reproduces an eval episode
    # 0) Get path to prev experiment data
    # 1) Setup policy
    # 2) Setup object config
    # 3) Update logging config
    # 4) Update run args; only do one epoch, one episode

    new_config = copy.deepcopy(config_to_dict(parent_config))

    # Determine parent output_dir, run_name, based on how run.py modifies on the fly
    output_dir = new_config["logging_config"]["output_dir"]
    run_name = new_config["logging_config"]["run_name"]
    wandb_id = None
    if update_run_dir:
        output_dir = recover_output_dir(new_config, parent_config_name)
        run_name = recover_run_name(new_config, parent_config_name)
        wandb_id = recover_wandb_id(output_dir)

    # 1) Use a policy that uses exact actions from previous episode
    motor_file = os.path.join(
        output_dir, "reproduce_episode_data", f"eval_episode_{episode}_actions.jsonl"
    )
    new_config["monty_config"]["motor_system_config"]["motor_system_args"][
        "policy_args"
    ]["file_name"] = motor_file

    # 2) Load object params from this episode into dataloader config
    object_params_file = os.path.join(
        output_dir, "reproduce_episode_data", f"eval_episode_{episode}_target.txt"
    )
    with open(object_params_file) as f:
        target_data = json.load(f)

    new_config["eval_dataloader_args"]["object_names"] = [
        target_data["primary_target_object"]
    ]
    new_config["eval_dataloader_args"]["object_init_sampler"] = (
        PredefinedObjectInitializer(
            positions=[target_data["primary_target_position"]],
            rotations=[target_data["primary_target_rotation_euler"]],
            # FIXME: target_scale is a float, need an array of floats for this
        )
    )

    # 3) Update logging config
    # First, make sure the detailed handlers are in there
    # TODO: update handlers used here as more sophisticated ones get made
    if DetailedJSONHandler not in new_config["logging_config"]["monty_handlers"]:
        new_config["logging_config"]["monty_handlers"].append(DetailedJSONHandler)
    if (
        DetailedWandbMarkedObsHandler
        not in new_config["logging_config"]["wandb_handlers"]
    ):
        new_config["logging_config"]["wandb_handlers"].append(
            DetailedWandbMarkedObsHandler
        )

    # Second, update the output directory, run_name, set resume to True
    new_output_dir = os.path.join(output_dir, f"eval_episode_{episode}_rerun")
    os.makedirs(new_output_dir, exist_ok=True)
    new_config["logging_config"]["output_dir"] = new_output_dir
    new_config["logging_config"]["run_name"] = run_name
    new_config["logging_config"]["resume_wandb_run"] = True
    new_config["logging_config"]["wandb_id"] = wandb_id
    new_config["logging_config"]["monty_log_level"] = "DETAILED"

    # 4) Make sure this config is set to only run for one episode
    new_config["experiment_args"]["n_eval_epochs"] = 1
    new_config["experiment_args"]["do_train"] = False

    return new_config


def create_eval_config_multiple_episodes(
    parent_config, parent_config_name, episodes, update_run_dir=True
):
    # NOTE: update_run_dir is assumed to be True, only added as an argument for
    # easier testing.
    new_config = copy.deepcopy(config_to_dict(parent_config))

    ###
    # General steps to update config that do not depend on specific episodes
    ###

    # Recover output dir based on how run.py modifies on the fly
    output_dir = new_config["logging_config"]["output_dir"]
    run_name = new_config["logging_config"]["run_name"]
    wandb_id = None
    if update_run_dir:
        output_dir = recover_output_dir(new_config, parent_config_name)
        run_name = recover_run_name(new_config, parent_config_name)
        wandb_id = recover_wandb_id(output_dir)

    # Add notes to indicate which episodes of the parent experiment were rerun
    notes = dict(rerun_episodes=episodes)
    if "notes" not in new_config:
        new_config["notes"] = notes
    else:
        new_config["notes"].update(notes)

    # Update the output directory to be a "rerun" subdir
    new_output_dir = os.path.join(output_dir, f"eval_rerun_episodes")
    os.makedirs(new_output_dir, exist_ok=True)
    new_config["logging_config"]["output_dir"] = new_output_dir
    new_config["logging_config"]["run_name"] = run_name
    new_config["logging_config"]["resume_wandb_run"] = True
    new_config["logging_config"]["wandb_id"] = wandb_id
    new_config["logging_config"]["monty_log_level"] = "DETAILED"

    # Turn training off; this feature is only intended for eval episodes
    new_config["experiment_args"]["do_train"] = False
    new_config["experiment_args"]["n_eval_epochs"] = 1

    # Add detailed handlers
    if DetailedJSONHandler not in new_config["logging_config"]["monty_handlers"]:
        new_config["logging_config"]["monty_handlers"].append(DetailedJSONHandler)

    ###
    # Accumulate episode-specific data: actions and object params
    ###

    file_names_per_episode = {}
    target_objects = []
    target_positions = []
    target_rotations = []
    for episode_counter, episode in enumerate(episodes):
        # Get actions from this episode
        motor_file = os.path.join(
            output_dir,
            "reproduce_episode_data",
            f"eval_episode_{episode}_actions.jsonl",
        )

        # Get object params from this episode
        object_params_file = os.path.join(
            output_dir, "reproduce_episode_data", f"eval_episode_{episode}_target.txt"
        )
        with open(object_params_file) as f:
            target_data = json.load(f)

        # Update accumulators
        file_names_per_episode[episode_counter] = motor_file
        target_objects.append(target_data["primary_target_object"])
        target_positions.append(target_data["primary_target_position"])
        target_rotations.append(target_data["primary_target_rotation_euler"])

    # Update config with episode-specific data
    new_config["eval_dataloader_args"]["object_names"] = target_objects
    new_config["eval_dataloader_args"]["object_init_sampler"] = (
        PredefinedObjectInitializer(
            positions=target_positions,
            rotations=target_rotations,
            change_every_episode=True,
        )
    )
    new_config["monty_config"]["motor_system_config"]["motor_system_args"][
        "policy_args"
    ]["file_names_per_episode"] = file_names_per_episode

    return new_config
