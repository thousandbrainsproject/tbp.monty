# Copyright 2025 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import copy
import datetime
import importlib
import logging
import os
import pprint

import numpy as np
import torch

from tbp.monty.frameworks.environments.embodied_data import (
    EnvironmentDataLoader,
    EnvironmentDataLoaderPerObject,
    EnvironmentDataset,
    SaccadeOnImageDataLoader,
    SaccadeOnImageFromStreamDataLoader,
)
from tbp.monty.frameworks.loggers.exp_logger import (
    BaseMontyLogger,
    LoggingCallbackHandler,
)
from tbp.monty.frameworks.loggers.wandb_handlers import WandbWrapper
from tbp.monty.frameworks.models.abstract_monty_classes import (
    LearningModule,
    SensorModule,
)
from tbp.monty.frameworks.models.monty_base import MontyBase
from tbp.monty.frameworks.models.motor_policies import MotorSystem
from tbp.monty.frameworks.utils.dataclass_utils import (
    config_to_dict,
    get_subset_of_args,
)

__all__ = {"MontyExperiment"}


class MontyExperiment:
    """General Monty experiment class used to run sensorimotor experiments.

    This class implements the framework for setting up a dataloader and Monty model,
    the outermost loops for training and evaluating (including run epoch and episode)
    """

    def setup_experiment(self, config):
        """Set up the basic elements of a Monty experiment and initialize counters.

        Args:
            config: config specifying variables of the experiment.
        """
        # Save a copy of the config used to specify the experiment before modifying
        config = copy.deepcopy(config)
        # Convert any dataclass back to dict for backward compatibility
        config = config_to_dict(config)
        self.config = config

        self.unpack_experiment_args(config["experiment_args"])
        self.model = self.init_model(
            monty_config=config["monty_config"],
            model_path=self.model_path,
        )
        self.load_dataset_and_dataloaders(config)
        self.init_loggers(self.config["logging_config"])
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

    def init_model(self, monty_config, model_path=None):
        """Initialize the Monty model.

        Args:
            monty_config: confguration for the Monty class.
            model_path: Optional model checkpoint. Can be full file name or just the
                directory containing the "model.pt" file saved from a previous run.

        Returns:
            Monty class instance
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
            sensor_modules[sm_id] = sm_class(**sm_args)
            sensor_modules[sm_id].rng = self.rng

        # Create motor system
        motor_system_config = monty_config.pop("motor_system_config")
        motor_system_class = motor_system_config["motor_system_class"]
        motor_system_args = motor_system_config["motor_system_args"]
        assert issubclass(motor_system_class, MotorSystem)
        motor_system = motor_system_class(rng=self.rng, **motor_system_args)

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

    def load_dataset_and_dataloaders(self, config):
        # Initialize everything needed for dataloader
        dataset_class = config["dataset_class"]
        dataset_args = config["dataset_args"]
        self.dataset = self.load_dataset(dataset_class, dataset_args)

        # Initialize train dataloaders if needed
        if config["experiment_args"]["do_train"]:
            dataloader_class = config["train_dataloader_class"]
            dataloader_args = config["train_dataloader_args"]
            self.train_dataloader = self.create_data_loader(
                dataloader_class, dataloader_args
            )
        else:
            self.train_dataloader = None

        # Initialize eval dataloaders if needed
        if config["experiment_args"]["do_eval"]:
            dataloader_class = config["eval_dataloader_class"]
            dataloader_args = config["eval_dataloader_args"]
            self.eval_dataloader = self.create_data_loader(
                dataloader_class, dataloader_args
            )
        else:
            self.eval_dataloader = None

    def load_dataset(self, dataset_class, dataset_args):
        """Instantiate a dataset.

        Possible splits include train and val for now, though this could change later
        based on how we implement validation for monty.

        Args:
            dataset_class: The class of the dataset.
            dataset_args: The arguments for the dataset.

        Returns:
            The instantiated dataset.

        Raises:
            TypeError: If `dataset_class` is not a subclass of `EnvironmentDataset`
        """
        # Require dataset_class to be EnvironmentDataset now, generalzie later
        if not issubclass(dataset_class, EnvironmentDataset):
            raise TypeError("dataset class must be EnvironmentDataset (for now)")

        dataset_args["rng"] = self.rng
        dataset = dataset_class(**dataset_args)
        return dataset

    def create_data_loader(self, dataloader_class, dataloader_args):
        """Dataloader used to collect data by sampling from dataset.

        Args:
            dataloader_class: The class of the dataloader.
            dataloader_args: The arguments for the dataloader.

        Returns:
            The instantiated dataloader.

        Raises:
            TypeError: If `dataloader_class` is not a subclass of
                `EnvironmentDataLoader`
        """
        # lump dataset and motor system into dataloader args
        # assume fixed dataset, training and validation are just different loaders
        if not issubclass(dataloader_class, EnvironmentDataLoader):
            raise TypeError("dataset class must be EnvironmentDataLoader (for now)")

        dataloader = dataloader_class(
            **dataloader_args,
            dataset=self.dataset,
            motor_system=self.model.motor_system,
            rng=self.rng,
        )

        assert dataloader.motor_system is self.model.motor_system
        return dataloader

    def init_counters(self):
        # Initialize time stamp variables for logging
        self.total_train_steps = 0
        self.train_episodes = 0
        self.train_epochs = 0
        self.total_eval_steps = 0
        self.eval_episodes = 0
        self.eval_epochs = 0
        self.dataloader = None

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
        # FIXME: 'target' attribute is specific to `EnvironmentDataLoaderPerObject`
        if isinstance(self.dataloader, EnvironmentDataLoaderPerObject):
            args.update(target=self.dataloader.primary_target)
        return args

    def init_loggers(self, logging_config):
        """Initialize logger with specified log level."""
        # Add experiment config so config can be passed to wandb
        all_logging_args = logging_config
        # all_logging_args.update(config=self.config)

        # Unpack individual logging arguments
        self.monty_log_level = all_logging_args["monty_log_level"]
        self.monty_handlers = all_logging_args["monty_handlers"]
        self.wandb_handlers = all_logging_args["wandb_handlers"]
        self.python_log_level = all_logging_args["python_log_level"]
        self.log_to_file = all_logging_args["python_log_to_file"]
        self.log_to_stdout = all_logging_args["python_log_to_stdout"]
        self.output_dir = all_logging_args["output_dir"]
        self.run_name = all_logging_args["run_name"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # If basic config has been set by a previous experiment, ipython, code editor,
        # or anything else, the config will not be properly set. importlib.reload gets
        # around this and ensures
        importlib.reload(logging)

        # Create basic python logging handlers
        python_logging_handlers = []
        if self.log_to_file:
            python_logging_handlers.append(
                logging.FileHandler(os.path.join(self.output_dir, "log.txt"), mode="w")
            )
        if self.log_to_stdout:
            python_logging_handlers.append(logging.StreamHandler())

        # Configure basic python logging
        logging.basicConfig(
            level=self.python_log_level,
            handlers=python_logging_handlers,
        )
        logging.info(f"Logger initialized at {datetime.datetime.now()}")
        logging.debug(pprint.pformat(self.config))

        # Configure Monty logging
        monty_handlers = []
        has_detailed_logger = False
        for handler in self.monty_handlers:
            if handler.log_level() == "DETAILED":
                has_detailed_logger = True
            handler_args = get_subset_of_args(all_logging_args, handler.__init__)
            monty_handler = handler(**handler_args)
            monty_handlers.append(monty_handler)

        # Configure wandb logging
        if len(self.wandb_handlers) > 0:
            wandb_args = get_subset_of_args(all_logging_args, WandbWrapper.__init__)
            wandb_args.update(
                config=self.config,
                run_name=wandb_args["run_name"] + "_" + wandb_args["wandb_id"],
            )
            monty_handlers.append(WandbWrapper(**wandb_args))
            for handler in self.wandb_handlers:
                if handler.log_level() == "DETAILED":
                    has_detailed_logger = True

        if has_detailed_logger and self.monty_log_level != "DETAILED":
            logging.warning(
                f"Log level is set to {self.monty_log_level} but you "
                "specified a detailed logging handler. Setting log level "
                "to detailed."
            )
            self.monty_log_level = "DETAILED"

        if self.monty_log_level == "DETAILED" and not has_detailed_logger:
            logging.warning(
                "You are setting the monty logging level to DETAILED, but all your"
                "handlers are BASIC. Consider setting the level to BASIC, or adding a"
                "DETAILED handler"
            )

        for lm in self.model.learning_modules:
            lm.has_detailed_logger = has_detailed_logger

        if has_detailed_logger or self.show_sensor_output:
            # If we log detailed stats we want to save sm raw obs by default.
            for sm in self.model.sensor_modules:
                sm.save_raw_obs = True

        # monty_log_level determines if we used Basic or Detailed logger
        # TODO: only defined for MontyForGraphMatching right now, need to add TM later
        # NOTE: later, more levels that Basic or Detailed could be added

        if isinstance(self.model, MontyBase):
            if self.monty_log_level in self.model.LOGGING_REGISTRY:
                logger_class = self.model.LOGGING_REGISTRY[self.monty_log_level]
                self.monty_logger = logger_class(handlers=monty_handlers)

            else:
                logging.warning(
                    "Unable to match monty logger to log level"
                    "An empty logger will be used as a placeholder"
                )
                self.monty_logger = BaseMontyLogger(handlers=[])
        else:
            raise (
                NotImplementedError,
                "Please implement a mapping from monty_log_level to a logger class"
                f"for models of type {type(self.model)}",
            )

        if "log_parallel_wandb" in all_logging_args.keys():
            self.monty_logger.use_parallel_wandb_logging = all_logging_args[
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

    def pre_step(self, step, observation):
        """Hook for anything you want to do before a step."""
        self.logger_handler.pre_step(self.logger_args)

    def post_step(self, step, observation):
        """Hook for anything you want to do after a step."""
        self.logger_handler.post_step(self.logger_args)

    def run_episode(self):
        """Run one episode until model.is_done."""
        self.pre_episode()
        for step, observation in enumerate(self.dataloader):
            self.pre_step(step, observation)
            self.model.step(observation)
            self.post_step(step, observation)
            if self.model.is_done or step >= self.max_steps:
                break
        self.post_episode(step)

    def pre_episode(self):
        """Call pre_episode on elements in experiment and set mode."""
        self.model.pre_episode()
        self.dataloader.pre_episode()

        self.max_steps = self.max_train_steps
        if not self.model.experiment_mode == "train":
            self.max_steps = self.max_eval_steps

        self.logger_handler.pre_episode(self.logger_args)

    def post_episode(self, steps):
        """Call post_episode on elements in experiment and increment counters.

        General order of post episode should be:
            logger_handler.post_episode
            model.post_episode
            increment counters
            dataloader.post_episode
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

        # move down here, otherwise dataloader.primary_target is already changed
        self.dataloader.post_episode()

    def run_epoch(self):
        """Run epoch -> Run one episode for each object."""
        self.pre_epoch()
        if isinstance(self.dataloader, SaccadeOnImageFromStreamDataLoader):
            try:
                while True:
                    self.run_episode()
            except KeyboardInterrupt:
                logging.info("Data streaming interupted. Stopping experiment.")
        elif isinstance(self.dataloader, SaccadeOnImageDataLoader):
            num_episodes = len(self.dataloader.scenes)
            for _ in range(num_episodes):
                self.run_episode()
        elif isinstance(self.dataloader, EnvironmentDataLoaderPerObject):
            for object_name in self.dataloader.object_names:
                logging.info(f"Running a simulation to model object: {object_name}")
                self.run_episode()
        else:
            logging.info("Running single episode")
            self.run_episode()

        self.post_epoch()

    def pre_epoch(self):
        """Set dataloader and call sub pre_epoch functions."""
        self.dataloader = self.train_dataloader
        if not self.model.experiment_mode == "train":
            self.dataloader = self.eval_dataloader

        self.dataloader.pre_epoch()
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
            self.train_dataloader.post_epoch()
        else:
            self.eval_epochs += 1
            self.eval_dataloader.post_epoch()

    def train(self):
        """Run n_train_epochs."""
        self.logger_handler.pre_train(self.logger_args)
        self.model.set_experiment_mode("train")
        for _ in range(self.n_train_epochs):
            self.run_epoch()
        self.logger_handler.post_train(self.logger_args)

        if not self.do_eval:
            self.close()

    def evaluate(self):
        """Run n_eval_epochs."""
        # TODO: check that number of eval epochs is at least as many as length
        # of dataloader number of rotations
        self.logger_handler.pre_eval(self.logger_args)
        self.model.set_experiment_mode("eval")
        for _ in range(self.n_eval_epochs):
            self.run_epoch()
        self.logger_handler.post_eval(self.logger_args)
        self.close()

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
            logging.info(f"saving model to {output_dir}")
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
        if isinstance(self.dataset, EnvironmentDataset):
            self.dataset.close()

        # Close monty logging
        self.logger_handler.close(self.logger_args)

        # Close python logging
        python_logger = logging.getLogger()
        for handler in python_logger.handlers:
            logging.debug(f"Removing and closing python log handler: {handler}")
            python_logger.removeHandler(handler)
            handler.close()

    def __enter__(self):
        """Context manager entry method.

        Returns:
            MontyExperiment self to allow assignment in a with statement.
        """
        # TODO: Move some of the initialization code from `setup_experiment` into this.
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Context manager exit method.

        Ensure that we always close the environment if necessary.

        Returns:
            bool to indicate whether to supress any exceptions that were raised.
        """
        # TODO: We call self.close inside `train` and `evaluate`.
        #   Those should probably be removed.
        self.close()
        return False  # don't silence exceptions inside the with block
