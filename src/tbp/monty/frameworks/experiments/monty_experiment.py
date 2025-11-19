# Copyright 2025 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import copy
import datetime
import logging
import os
import pprint
from typing import Any, Literal

import numpy as np
import torch
from typing_extensions import Self

from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentInterface,
    EnvironmentInterfacePerObject,
    SaccadeOnImageEnvironmentInterface,
    SaccadeOnImageFromStreamEnvironmentInterface,
)
from tbp.monty.frameworks.environments.embodied_environment import EmbodiedEnvironment
from tbp.monty.frameworks.loggers.exp_logger import (
    BaseMontyLogger,
    LoggingCallbackHandler,
)
from tbp.monty.frameworks.loggers.wandb_handlers import WandbWrapper
from tbp.monty.frameworks.models.abstract_monty_classes import (
    LearningModule,
    SensorModule,
)
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.utils.dataclass_utils import (
    Dataclass,
    config_to_dict,
    get_subset_of_args,
)
from tbp.monty.frameworks.utils.live_plotter import LivePlotter

__all__ = ["MontyExperiment"]

logger = logging.getLogger("tbp.monty")


class MontyExperiment:
    """General Monty experiment class used to run sensorimotor experiments.

    This class implements the framework for setting up an environment interface and
    Monty model, the outermost loops for training and evaluating (including run epoch
    and episode).
    """

    def __init__(self, config: Dataclass | dict[str, Any]) -> None:
        """Initialize the experiment based on the provided configuration.

        Args:
            config: config specifying variables of the experiment.
        """
        # Copy the config and store it so we can modify it freely
        config = copy.deepcopy(config)
        config = config_to_dict(config)
        self.config = config

        self.unpack_experiment_args(config["experiment_args"])

        if self.show_sensor_output:
            self.live_plotter = LivePlotter()

    def setup_experiment(self, config: dict[str, Any]) -> None:
        """Set up the basic elements of a Monty experiment and initialize counters.

        Args:
            config: config specifying variables of the experiment.
        """
        self.init_loggers(self.config["logging_config"])
        self.model = self.init_model(
            monty_config=config["monty_config"],
            model_path=self.model_path,
        )
        self.load_environment_interfaces(config)
        self.init_monty_data_loggers(self.config["logging_config"])
        self.init_counters()

    ####
    # Methods for setting up an experiment
    ####

    def unpack_experiment_args(self, experiment_args):
        self.do_train = experiment_args["do_train"]
        self.do_eval = experiment_args["do_eval"]
        self.max_eval_steps = experiment_args["max_eval_steps"]
        self.max_train_steps = experiment_args["max_train_steps"]
        self.max_total_steps = experiment_args["max_total_steps"]
        self.n_eval_epochs = experiment_args["n_eval_epochs"]
        self.n_train_epochs = experiment_args["n_train_epochs"]
        self.model_path = experiment_args["model_name_or_path"]
        self.min_lms_match = experiment_args["min_lms_match"]
        self.rng = np.random.RandomState(experiment_args["seed"])
        self.show_sensor_output = experiment_args["show_sensor_output"]
        self.supervised_lm_ids = experiment_args["supervised_lm_ids"]
        if self.supervised_lm_ids == "all":
            self.supervised_lm_ids = list(
                self.config["monty_config"]["learning_module_configs"].keys()
            )

    def init_model(self, monty_config, model_path=None):
        """Initialize the Monty model.

        Args:
            monty_config: configuration for the Monty class.
            model_path: Optional model checkpoint. Can be full file name or just the
                directory containing the "model.pt" file saved from a previous run.

        Returns:
            Monty class instance

        Raises:
            TypeError: If `motor_system_class` is not a subclass of `MotorSystem` or
                `policy_class` is not a subclass of `MotorPolicy`.
        """
        monty_config = copy.deepcopy(monty_config)

        # Create learning modules
        learning_module_configs = monty_config.pop("learning_module_configs")
        learning_modules = {}
        for lm_id, lm_cfg in learning_module_configs.items():
            lm_class = lm_cfg["learning_module_class"]
            lm_args = lm_cfg["learning_module_args"]
            assert issubclass(lm_class, LearningModule)
            learning_modules[lm_id] = lm_class(**lm_args)
            learning_modules[lm_id].rng = self.rng
            learning_modules[lm_id].learning_module_id = lm_id

        # Create sensor modules
        sensor_module_configs = monty_config.pop("sensor_module_configs")
        sensor_modules = {}
        for sm_id, sm_cfg in sensor_module_configs.items():
            sm_class = sm_cfg["sensor_module_class"]
            sm_args = sm_cfg["sensor_module_args"]
            assert issubclass(sm_class, SensorModule)
            sensor_modules[sm_id] = sm_class(rng=self.rng, **sm_args)

        # Create motor system
        motor_system_config = monty_config.pop("motor_system_config")
        motor_system_class = motor_system_config["motor_system_class"]
        motor_system_args = motor_system_config["motor_system_args"]
        if not issubclass(motor_system_class, MotorSystem):
            raise TypeError(
                "motor_system_class must be a subclass of MotorSystem, got "
                f"{motor_system_class}"
            )
        policy_class = motor_system_args["policy_class"]
        policy_args = motor_system_args["policy_args"]
        if not issubclass(policy_class, MotorPolicy):
            raise TypeError(
                f"policy_class must be a subclass of MotorPolicy, got {policy_class}"
            )
        policy = policy_class(rng=self.rng, **policy_args)
        motor_system = motor_system_class(policy=policy)

        # Get mapping between sensor modules, learning modules and agents
        lm_len = len(learning_modules)
        sm_to_lm_matrix = monty_config.pop("sm_to_lm_matrix", [[]] * lm_len)
        lm_to_lm_matrix = monty_config.pop("lm_to_lm_matrix", [[]] * lm_len)
        lm_to_lm_vote_matrix = monty_config.pop("lm_to_lm_vote_matrix", [[]] * lm_len)
        sm_to_agent_dict = monty_config.pop("sm_to_agent_dict")

        # Create monty model
        # FIXME: Kept for backward compatibility
        monty_args = monty_config.pop("monty_args", {})
        monty_class = monty_config.pop("monty_class")
        model = monty_class(
            sensor_modules=list(sensor_modules.values()),
            learning_modules=list(learning_modules.values()),
            motor_system=motor_system,
            sm_to_agent_dict=sm_to_agent_dict,
            sm_to_lm_matrix=sm_to_lm_matrix,
            lm_to_lm_matrix=lm_to_lm_matrix,
            lm_to_lm_vote_matrix=lm_to_lm_vote_matrix,
            # Pass any leftover configuration paramters downstream to monty_class
            **monty_config,
            # FIXME: Kept for backward compatibility
            **monty_args,
        )
        model.min_lms_match = self.min_lms_match

        if monty_args["num_exploratory_steps"] > self.max_total_steps:
            new_max_steps = monty_args["num_exploratory_steps"] + self.max_train_steps
            print(
                "max_total_steps is set < num_exploratory_steps + max_train_steps."
                f" Resetting it to {new_max_steps}"
            )
            self.max_total_steps = new_max_steps

        # Load from checkpoint
        if model_path:
            if "model.pt" not in model_path:
                model_path = os.path.join(model_path, "model.pt")
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)

        return model

    def init_env(self, env_init_func, env_init_args):
        self.env = env_init_func(**env_init_args)
        assert isinstance(self.env, EmbodiedEnvironment)

    def load_environment_interfaces(self, config):
        # Initialize everything needed for environment interface
        env_interface_config = config["env_interface_config"]
        self.init_env(
            env_interface_config["env_init_func"], env_interface_config["env_init_args"]
        )

        # Initialize train environment interface if needed
        if config["experiment_args"]["do_train"]:
            env_interface_class = config["train_env_interface_class"]
            env_interface_args = dict(
                env=self.env,
                transform=env_interface_config["transform"],
                **config["train_env_interface_args"],
            )

            self.train_env_interface = self.create_env_interface(
                env_interface_class, env_interface_args
            )
        else:
            self.train_env_interface = None

        # Initialize eval environment interfaces if needed
        if config["experiment_args"]["do_eval"]:
            env_interface_class = config["eval_env_interface_class"]
            env_interface_args = dict(
                env=self.env,
                transform=env_interface_config["transform"],
                **config["eval_env_interface_args"],
            )

            self.eval_env_interface = self.create_env_interface(
                env_interface_class, env_interface_args
            )
        else:
            self.eval_env_interface = None

    def create_env_interface(self, env_interface_class, env_interface_args):
        """Environment interface used to collect data from environment observations.

        Args:
            env_interface_class: The class of the environment interface.
            env_interface_args: The arguments for the environment interface.

        Returns:
            The instantiated environment interface.

        Raises:
            TypeError: If `env_interface_class` is not a subclass of
                `EnvironmentInterface`
        """
        # training and validation are just different environment interfaces
        if not issubclass(env_interface_class, EnvironmentInterface):
            raise TypeError(
                "env_interface_class must be EnvironmentInterface (for now)"
            )

        env_interface = env_interface_class(
            **env_interface_args,
            motor_system=self.model.motor_system,
            rng=self.rng,
        )

        assert env_interface.motor_system is self.model.motor_system
        return env_interface

    def init_counters(self):
        # Initialize time stamp variables for logging
        self.total_train_steps = 0
        self.train_episodes = 0
        self.train_epochs = 0
        self.total_eval_steps = 0
        self.eval_episodes = 0
        self.eval_epochs = 0
        self.env_interface = None

    ####
    # Logging
    ####

    @property
    def logger_args(self):
        """Get current status of counters for the logger.

        Returns:
            dict with current expirent state.
        """
        args = dict(
            total_train_steps=self.total_train_steps,
            train_episodes=self.train_episodes,
            train_epochs=self.train_epochs,
            total_eval_steps=self.total_eval_steps,
            eval_episodes=self.eval_episodes,
            eval_epochs=self.eval_epochs,
        )
        # FIXME: 'target' attribute is specific to `EnvironmentInterfacePerObject`
        if isinstance(self.env_interface, EnvironmentInterfacePerObject):
            target = self.env_interface.primary_target
            if target is not None:
                target.update(
                    consistent_child_objects=self.env_interface.consistent_child_objects
                )
            args.update(target=target)
        return args

    def init_loggers(self, logging_config: dict[str, Any]) -> None:
        """Initialize logger with specified log level.

        Args:
            logging_config: Logging configuration.
        """
        # Unpack individual logging arguments
        self.python_log_level = logging_config["python_log_level"]
        self.log_to_file = logging_config["python_log_to_file"]
        self.log_to_stderr = logging_config["python_log_to_stderr"]
        self.output_dir = logging_config["output_dir"]
        self.run_name = logging_config["run_name"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Clear any existing tpb.monty logger handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)

        # Create basic python logging handlers
        python_logging_handlers: list[logging.Handler] = []
        if self.log_to_file:
            python_logging_handlers.append(
                logging.FileHandler(os.path.join(self.output_dir, "log.txt"), mode="w")
            )
        if self.log_to_stderr:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s"
                )
            )
            python_logging_handlers.append(handler)

        logger.setLevel(self.python_log_level)
        for handler in python_logging_handlers:
            logger.addHandler(handler)

        logger.info("logger initialized")
        logger.debug(pprint.pformat(self.config))

    def init_monty_data_loggers(self, logging_config: dict[str, Any]) -> None:
        """Initialize Monty data loggers.

        Args:
            logging_config: Logging configuration.
        """
        self.monty_log_level = logging_config["monty_log_level"]
        self.monty_handlers = logging_config["monty_handlers"]
        self.wandb_handlers = logging_config["wandb_handlers"]

        # Configure Monty logging
        monty_handlers = []
        has_detailed_logger = False
        for handler in self.monty_handlers:
            if handler.log_level() == "DETAILED":
                has_detailed_logger = True
            handler_args = get_subset_of_args(logging_config, handler.__init__)
            monty_handler = handler(**handler_args)
            monty_handlers.append(monty_handler)

        # Configure wandb logging
        if len(self.wandb_handlers) > 0:
            wandb_args = get_subset_of_args(logging_config, WandbWrapper.__init__)
            wandb_args.update(
                config=self.config,
                run_name=wandb_args["run_name"] + "_" + wandb_args["wandb_id"],
            )
            monty_handlers.append(WandbWrapper(**wandb_args))
            for handler in self.wandb_handlers:
                if handler.log_level() == "DETAILED":
                    has_detailed_logger = True

        if has_detailed_logger and self.monty_log_level != "DETAILED":
            logger.warning(
                f"Log level is set to {self.monty_log_level} but you "
                "specified a detailed logging handler. Setting log level "
                "to detailed."
            )
            self.monty_log_level = "DETAILED"

        if self.monty_log_level == "DETAILED" and not has_detailed_logger:
            logger.warning(
                "You are setting the monty logging level to DETAILED, but all your"
                "handlers are BASIC. Consider setting the level to BASIC, or adding a"
                "DETAILED handler"
            )

        for lm in self.model.learning_modules:
            lm.has_detailed_logger = has_detailed_logger

        if has_detailed_logger:
            for sm in self.model.sensor_modules:
                if hasattr(sm, "save_raw_obs") and not sm.save_raw_obs:
                    logger.warning(
                        "You are using a DETAILED logger with sensor module "
                        f"{sm.sensor_module_id} but 'save_raw_obs' is False. "
                        "Consider setting 'save_raw_obs' to True to log and visualize "
                        "the SM RGB raw values."
                    )

        # monty_log_level determines if we used Basic or Detailed logger
        # TODO: only defined for MontyForGraphMatching right now, need to add TM later
        # NOTE: later, more levels that Basic or Detailed could be added

        if self.monty_log_level in self.model.LOGGING_REGISTRY:
            logger_class = self.model.LOGGING_REGISTRY[self.monty_log_level]
            self.monty_logger = logger_class(handlers=monty_handlers)
        else:
            logger.warning(
                "Unable to match monty logger to log level. "
                "An empty logger will be used as a placeholder"
            )
            self.monty_logger = BaseMontyLogger(handlers=[])

        if "log_parallel_wandb" in logging_config.keys():
            self.monty_logger.use_parallel_wandb_logging = logging_config[
                "log_parallel_wandb"
            ]
        # Instantiate logging callback handler for custom monty loggers
        self.logger_handler = LoggingCallbackHandler(
            self.monty_logger, self.model, output_dir=self.output_dir
        )

    def get_epoch_state(self):
        mode = self.model.experiment_mode

        if mode == "train":
            epoch = self.train_epochs
            episode = self.train_episodes
        else:
            epoch = self.eval_epochs
            episode = self.eval_episodes

        return mode, epoch, episode

    ####
    # Methods for running the experiment
    ####

    def pre_step(self, _step, _observation):
        """Hook for anything you want to do before a step."""
        self.logger_handler.pre_step(self.logger_args)

    def post_step(self, _step, _observation):
        """Hook for anything you want to do after a step."""
        self.logger_handler.post_step(self.logger_args)

    def run_episode(self):
        """Run one episode until model.is_done."""
        self.pre_episode()
        for step, observation in enumerate(self.env_interface):
            self.pre_step(step, observation)
            self.model.step(observation)
            self.post_step(step, observation)
            if self.model.is_done or step >= self.max_steps:
                break
        self.post_episode(step)

    def pre_episode(self):
        """Call pre_episode on elements in experiment and set mode."""
        self.model.pre_episode()
        self.env_interface.pre_episode()

        self.max_steps = self.max_train_steps
        if self.model.experiment_mode != "train":
            self.max_steps = self.max_eval_steps

        self.logger_handler.pre_episode(self.logger_args)

        if self.show_sensor_output:
            self.live_plotter.initialize_online_plotting()

    def post_episode(self, steps):
        """Call post_episode on elements in experiment and increment counters.

        General order of post episode should be:
            logger_handler.post_episode
            model.post_episode
            increment counters
            env_interface.post_episode
        If the logger_handler is called later it will not log the correct
        episode ID and target object. If model.post_episode is called before the
        logger we have already updated the target to graph mapping and will never
        get 'confused'/'FP'.
        """
        self.logger_handler.post_episode(self.logger_args)
        self.model.post_episode()

        if self.model.experiment_mode == "train":
            self.train_episodes += 1
            self.total_train_steps += steps
        else:
            self.eval_episodes += 1
            self.total_eval_steps += steps

        # move down here, otherwise env_interface.primary_target is already changed
        self.env_interface.post_episode()

    def run_epoch(self):
        """Run epoch -> Run one episode for each object."""
        self.pre_epoch()
        if isinstance(self.env_interface, SaccadeOnImageFromStreamEnvironmentInterface):
            try:
                while True:
                    self.run_episode()
            except KeyboardInterrupt:
                logger.info("Data streaming interupted. Stopping experiment.")
        elif isinstance(self.env_interface, SaccadeOnImageEnvironmentInterface):
            num_episodes = len(self.env_interface.scenes)
            for _ in range(num_episodes):
                self.run_episode()
        elif isinstance(self.env_interface, EnvironmentInterfacePerObject):
            for object_name in self.env_interface.object_names:
                logger.info(f"Running a simulation to model object: {object_name}")
                self.run_episode()
        else:
            logger.info("Running single episode")
            self.run_episode()

        self.post_epoch()

    def pre_epoch(self):
        """Set environment interface and call sub pre_epoch functions."""
        self.env_interface = self.train_env_interface
        if self.model.experiment_mode != "train":
            self.env_interface = self.eval_env_interface

        self.env_interface.pre_epoch()
        self.logger_handler.pre_epoch(self.logger_args)

    def post_epoch(self):
        """Call sub post_epoch functions and save state dict."""
        # NOTE: maybe an option not to save everything every epoch?
        self.save_state_dict(
            output_dir=os.path.join(self.output_dir, f"{self.train_epochs}")
        )
        self.logger_handler.post_epoch(self.logger_args)

        if self.model.experiment_mode == "train":
            self.train_epochs += 1
            self.train_env_interface.post_epoch()
        else:
            self.eval_epochs += 1
            self.eval_env_interface.post_epoch()

    def train(self):
        """Run n_train_epochs."""
        self.logger_handler.pre_train(self.logger_args)
        self.model.set_experiment_mode("train")
        for _ in range(self.n_train_epochs):
            self.run_epoch()
        self.logger_handler.post_train(self.logger_args)

    def evaluate(self):
        """Run n_eval_epochs."""
        # TODO: check that number of eval epochs is at least as many as length
        # of environment interface number of rotations
        self.logger_handler.pre_eval(self.logger_args)
        self.model.set_experiment_mode("eval")
        for _ in range(self.n_eval_epochs):
            self.run_epoch()
        self.logger_handler.post_eval(self.logger_args)

    def state_dict(self):
        """Return state_dict with total steps."""
        return dict(
            total_train_steps=self.total_train_steps,
            train_episodes=self.train_episodes,
            train_epochs=self.train_epochs,
            total_eval_steps=self.total_eval_steps,
            eval_episodes=self.eval_episodes,
            eval_epochs=self.eval_epochs,
            time_stamp=datetime.datetime.now(),
        )

    def save_state_dict(self, output_dir=None):
        """Save state_dict of experiment and model."""
        model_state_dict = self.model.state_dict()
        exp_state_dict = self.state_dict()
        output_dir = output_dir if output_dir is not None else self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # When performing evaluation with parallel runs on a remote server
        # (assumed if we are using parallel wandb logging), then don't save models;
        # these can fill a huge amount of hard-disk memory before they are cleaned up
        # at the end of the experiment, and currently the model won't be changing
        # during evaluation
        # TODO can consider a save frequency for training as well; e.g. currently
        # with training from scratch, we save +++ data
        if (
            self.model.experiment_mode == "eval"
            and self.monty_logger.use_parallel_wandb_logging
        ):
            pass
        else:
            logger.info(f"saving model to {output_dir}")
            torch.save(model_state_dict, os.path.join(output_dir, "model.pt"))
            torch.save(exp_state_dict, os.path.join(output_dir, "exp_state_dict.pt"))
            torch.save(self.config, os.path.join(output_dir, "config.pt"))

    def load_state_dict(self, load_dir):
        """Load state_dict of previous experiment."""
        model_state_dict = torch.load(os.path.join(load_dir, "model.pt"))
        exp_state_dict = torch.load(os.path.join(load_dir, "exp_state_dict.pt"))
        config = torch.load(os.path.join(load_dir, "config.pt"))
        state_dict_keys = self.state_dict().keys()

        self.model.load_state_dict(model_state_dict)
        self.config = config
        for k in state_dict_keys:
            setattr(self, k, exp_state_dict[k])

    def close(self):
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
            self.env = None

        # Close monty logging
        self.logger_handler.close(self.logger_args)

        # Close python logging
        for handler in logger.handlers:
            logger.debug(f"Removing and closing python log handler: {handler}")
            logger.removeHandler(handler)
            handler.close()

    def __enter__(self) -> Self:
        """Context manager entry method.

        Returns:
            MontyExperiment self to allow assignment in a with statement.
        """
        self.setup_experiment(self.config)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> Literal[False]:
        """Context manager exit method.

        Ensure that we always close the environment if necessary.

        Returns:
            Whether to supress any exceptions that were raised.
        """
        self.close()
        return False  # don't silence exceptions inside the with block
