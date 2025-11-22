# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from tbp.monty.frameworks.config_utils.make_env_interface_configs import (
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
    output_dir = Path(config["logging"]["output_dir"])
    if not config["logging"]["run_name"]:
        output_dir = output_dir / config_name

    return output_dir


def recover_run_name(config, config_name):
    if not config["logging"]["run_name"]:
        return config_name

    return config["logging"]["run_name"]


def recover_wandb_id(output_dir):
    # 0 is just go specify any epoch subdirectory; ids should all be the same
    config_file_name = Path(output_dir) / "0" / "config.pt"
    cfg = torch.load(config_file_name)
    config = config_to_dict(cfg)
    return config["logging"]["wandb_id"]


def create_eval_episode_hydra_cfg(
    parent_config: DictConfig, episode: int
) -> DictConfig:
    """Returns a new Hydra config that runs only that episode with detailed logging.

    This doesn't support the `update_run_dir` option that the function below does.
    """
    new_cfg = copy.deepcopy(parent_config)
    # Make an alias for the actual config to shorten how far we need to dig into
    # the object tree
    exp_cfg = new_cfg.test.config

    output_dir = exp_cfg.logging.output_dir
    # 1) Use a policy that uses exact actions from previous episode
    motor_system_cfg = exp_cfg.monty_config.motor_system_config
    motor_system_cfg.motor_system_args.policy_args.file_name = str(
        Path(output_dir)
        / "reproduce_episode_data"
        / f"eval_episode_{episode}_actions.jsonl"
    )
    # 2) Load object params from this episode into environment interface config
    object_params_file = (
        Path(output_dir)
        / "reproduce_episode_data"
        / f"eval_episode_{episode}_target.txt"
    )
    with open(object_params_file) as f:
        target_data = json.load(f)

    exp_cfg.eval_env_interface_args.object_names = [
        target_data["primary_target_object"]
    ]

    # Need to create the config that Hydra expects using fully qualified
    # class names instead of sticking the class instance on the config.
    def class_path(klass: type) -> str:
        """Returns the fully qualified class path for a class."""
        return f"{klass.__module__}.{klass.__name__}"

    def monty_class(klass: type) -> str:
        """Returns the ${monty.class:...} resolver expression for a class."""
        return f"${{monty.class:{class_path(klass)}}}"

    exp_cfg.eval_env_interface_args.object_init_sampler = {
        "_target_": class_path(PredefinedObjectInitializer),
        "positions": [target_data["primary_target_position"]],
        "rotations": [target_data["primary_target_rotation_euler"]],
        # FIXME: target_scale is a float, need an array of floats for this
    }
    # 3) Update logging config
    # We have to convert the handlers from the config into a format that will match
    # the values we can look things up with.
    # TODO: update handlers used here as more sophisticated ones get made
    monty_handlers = OmegaConf.to_object(exp_cfg.logging.monty_handlers)
    if DetailedJSONHandler not in monty_handlers:
        exp_cfg.logging.monty_handlers.append(monty_class(DetailedJSONHandler))
    wandb_handlers = OmegaConf.to_object(exp_cfg.logging.wandb_handlers)
    if DetailedWandbMarkedObsHandler not in wandb_handlers:
        exp_cfg.logging.wandb_handlers.append(
            monty_class(DetailedWandbMarkedObsHandler)
        )
    # Second, update the output directory, run_name, set resume to True
    new_output_dir = Path(output_dir) / f"eval_episode_{episode}_rerun"
    new_output_dir.mkdir(exist_ok=True)
    exp_cfg.logging.output_dir = str(new_output_dir)
    exp_cfg.logging.resume_wandb_run = True
    exp_cfg.logging.wandb_id = None
    exp_cfg.logging.monty_log_level = "DETAILED"
    # 4) Make sure this config is set to only run for one episode
    exp_cfg.n_eval_epochs = 1
    exp_cfg.do_train = False

    # We've been using exp_cfg for convenience to shorten how far we
    # need to dig into the object tree, but we want to return the top
    # of the new config.
    return new_cfg


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
        For now, assume env_interface_class is a subclass of
              `EnvironmentInterfacePerObject`

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
    output_dir = Path(new_config["logging"]["output_dir"])
    run_name = new_config["logging"]["run_name"]
    wandb_id = None
    if update_run_dir:
        output_dir = recover_output_dir(new_config, parent_config_name)
        run_name = recover_run_name(new_config, parent_config_name)
        wandb_id = recover_wandb_id(output_dir)

    # 1) Use a policy that uses exact actions from previous episode
    motor_file = (
        output_dir / "reproduce_episode_data" / f"eval_episode_{episode}_actions.jsonl"
    )
    new_config["monty_config"]["motor_system_config"]["motor_system_args"][
        "policy_args"
    ]["file_name"] = motor_file

    # 2) Load object params from this episode into environment interface config
    object_params_file = (
        output_dir / "reproduce_episode_data" / f"eval_episode_{episode}_target.txt"
    )
    with open(object_params_file) as f:
        target_data = json.load(f)

    new_config["eval_env_interface_args"]["object_names"] = [
        target_data["primary_target_object"]
    ]
    new_config["eval_env_interface_args"]["object_init_sampler"] = (
        PredefinedObjectInitializer(
            positions=[target_data["primary_target_position"]],
            rotations=[target_data["primary_target_rotation_euler"]],
            # FIXME: target_scale is a float, need an array of floats for this
        )
    )

    # 3) Update logging config
    # First, make sure the detailed handlers are in there
    # TODO: update handlers used here as more sophisticated ones get made
    if DetailedJSONHandler not in new_config["logging"]["monty_handlers"]:
        new_config["logging"]["monty_handlers"].append(DetailedJSONHandler)
    if DetailedWandbMarkedObsHandler not in new_config["logging"]["wandb_handlers"]:
        new_config["logging"]["wandb_handlers"].append(DetailedWandbMarkedObsHandler)

    # Second, update the output directory, run_name, set resume to True
    new_output_dir = output_dir / f"eval_episode_{episode}_rerun"
    os.makedirs(new_output_dir, exist_ok=True)
    new_config["logging"]["output_dir"] = new_output_dir
    new_config["logging"]["run_name"] = run_name
    new_config["logging"]["resume_wandb_run"] = True
    new_config["logging"]["wandb_id"] = wandb_id
    new_config["logging"]["monty_log_level"] = "DETAILED"

    # 4) Make sure this config is set to only run for one episode
    new_config["experiment_args"]["n_eval_epochs"] = 1
    new_config["experiment_args"]["do_train"] = False

    return new_config


def create_eval_multiple_episodes_hydra_cfg(
    parent_config: DictConfig, episodes: list[int]
) -> DictConfig:
    """Returns a new Hydra config that runs only the episodes specified.

    This doesn't support the `update_run_dir` option that the function below does.
    """
    new_cfg = copy.deepcopy(parent_config)
    # Make an alias for the actual config to shorten how far we need to dig into
    # the object tree
    exp_cfg = new_cfg.test.config

    output_dir = exp_cfg.logging.output_dir

    # Add notes to indicate which episodes of the parent experiment were rerun.
    # This field might not exist to open the DictConfig up for adding keys
    with open_dict(new_cfg):
        exp_cfg.notes = {"rerun_episodes": episodes}

    # Update the output directory to be a "rerun" subdir
    new_output_dir = Path(output_dir) / "eval_rerun_episodes"
    new_output_dir.mkdir(exist_ok=True)
    exp_cfg.logging.output_dir = new_output_dir
    exp_cfg.logging.resume_wandb_run = False
    exp_cfg.logging.wandb_id = None
    exp_cfg.logging.monty_log_level = "DETAILED"

    # Turn training off; this feature is only intended for eval episodes
    exp_cfg.do_train = False
    exp_cfg.n_eval_epochs = 1

    # Add detailed handlers

    # Need to create the config that Hydra expects using fully qualified
    # class names instead of sticking the class instance on the config.
    def class_path(klass: type) -> str:
        """Returns the fully qualified class path for a class."""
        return f"{klass.__module__}.{klass.__name__}"

    def monty_class(klass: type) -> str:
        """Returns the ${monty.class:...} resolver expression for a class."""
        return f"${{monty.class:{class_path(klass)}}}"

    monty_handlers = OmegaConf.to_object(exp_cfg.logging.monty_handlers)
    if DetailedJSONHandler not in monty_handlers:
        exp_cfg.logging.monty_handlers.append(monty_class(DetailedJSONHandler))

    file_names_per_episode = {}
    target_objects = []
    target_positions = []
    target_rotations = []
    for episode_counter, episode in enumerate(episodes):
        # Get actions from this episode
        motor_file = (
            Path(output_dir)
            / "reproduce_episode_data"
            / f"eval_episode_{episode}_actions.jsonl"
        )

        # Get object params from this episode
        object_params_file = (
            Path(output_dir)
            / "reproduce_episode_data"
            / f"eval_episode_{episode}_target.txt"
        )
        with open(object_params_file) as f:
            target_data = json.load(f)

        # Update accumulators
        file_names_per_episode[episode_counter] = str(motor_file)
        target_objects.append(target_data["primary_target_object"])
        target_positions.append(target_data["primary_target_position"])
        target_rotations.append(target_data["primary_target_rotation_euler"])

    # Update config with episode-specific data
    exp_cfg.eval_env_interface_args.object_names = target_objects
    exp_cfg.eval_env_interface_args.object_init_sampler = {
        "_target_": class_path(PredefinedObjectInitializer),
        "positions": target_positions,
        "rotations": target_rotations,
        "change_every_episode": True,
    }
    motor_system_cfg = exp_cfg.monty_config.motor_system_config
    # This field might not exist to open the DictConfig up for adding keys
    with open_dict(new_cfg):
        motor_system_cfg.motor_system_args.policy_args.file_names_per_episode = (
            file_names_per_episode
        )

    return new_cfg


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
    output_dir = Path(new_config["logging"]["output_dir"])
    run_name = new_config["logging"]["run_name"]
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
    new_output_dir = output_dir / "eval_rerun_episodes"
    os.makedirs(new_output_dir, exist_ok=True)
    new_config["logging"]["output_dir"] = new_output_dir
    new_config["logging"]["run_name"] = run_name
    new_config["logging"]["resume_wandb_run"] = True
    new_config["logging"]["wandb_id"] = wandb_id
    new_config["logging"]["monty_log_level"] = "DETAILED"

    # Turn training off; this feature is only intended for eval episodes
    new_config["experiment_args"]["do_train"] = False
    new_config["experiment_args"]["n_eval_epochs"] = 1

    # Add detailed handlers
    if DetailedJSONHandler not in new_config["logging"]["monty_handlers"]:
        new_config["logging"]["monty_handlers"].append(DetailedJSONHandler)

    ###
    # Accumulate episode-specific data: actions and object params
    ###

    file_names_per_episode = {}
    target_objects = []
    target_positions = []
    target_rotations = []
    for episode_counter, episode in enumerate(episodes):
        # Get actions from this episode
        motor_file = (
            output_dir
            / "reproduce_episode_data"
            / f"eval_episode_{episode}_actions.jsonl"
        )

        # Get object params from this episode
        object_params_file = (
            output_dir / "reproduce_episode_data" / f"eval_episode_{episode}_target.txt"
        )
        with open(object_params_file) as f:
            target_data = json.load(f)

        # Update accumulators
        file_names_per_episode[episode_counter] = motor_file
        target_objects.append(target_data["primary_target_object"])
        target_positions.append(target_data["primary_target_position"])
        target_rotations.append(target_data["primary_target_rotation_euler"])

    # Update config with episode-specific data
    new_config["eval_env_interface_args"]["object_names"] = target_objects
    new_config["eval_env_interface_args"]["object_init_sampler"] = (
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
