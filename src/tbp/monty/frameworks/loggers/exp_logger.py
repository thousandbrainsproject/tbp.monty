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
import pickle

"""
Based on https://github.com/huggingface/transformers/blob/1438c487df5ce38a7b2ae30877b3074b96a423dd/src/transformers/trainer_callback.py
"""


class LoggingCallbackHandler:
    """Calls a list of loggers on an event (eg post_train).

    Each logger receives:
    logger_args: dict with time stamps (steps, epochs, etc.) and
        dataloader.primary_target which contains object id and pose
    output_dir: Full path of the directory to store log files

    Note:
        This logger handler is intended primarily for logging
    """

    def __init__(self, loggers, model, output_dir):
        self.loggers = loggers
        if isinstance(loggers, BaseMontyLogger):
            self.loggers = [loggers]
        self.model = model
        self.output_dir = output_dir

    @property
    def logger_list(self):
        return "\n".join(logger.__class__.__name__ for logger in self.loggers)

    def pre_step(self, logger_args):
        self.call_event("pre_step", logger_args)

    def post_step(self, logger_args):
        self.call_event("post_step", logger_args)

    def pre_episode(self, logger_args):
        self.call_event("pre_episode", logger_args)

    def post_episode(self, logger_args):
        self.call_event("post_episode", logger_args)

    def pre_epoch(self, logger_args):
        self.call_event("pre_epoch", logger_args)

    def post_epoch(self, logger_args):
        self.call_event("post_epoch", logger_args)

    def pre_train(self, logger_args):
        self.call_event("pre_train", logger_args)

    def post_train(self, logger_args):
        self.call_event("post_train", logger_args)

    def pre_eval(self, logger_args):
        self.call_event("pre_eval", logger_args)

    def post_eval(self, logger_args):
        self.call_event("post_eval", logger_args)

    def close(self, logger_args):
        self.call_event("close", logger_args)

    def call_event(self, event, logger_args):
        for logger in self.loggers:
            getattr(logger, event)(
                logger_args=logger_args, output_dir=self.output_dir, model=self.model
            )


class BaseMontyLogger:
    """Basic logger that logs or saves information when logging is called."""

    def __init__(self, handlers):
        self.handlers = handlers

    def flush(self):
        pass

    def pre_step(self, logger_args, output_dir, model):
        pass

    def post_step(self, logger_args, output_dir, model):
        pass

    def pre_episode(self, logger_args, output_dir, model):
        pass

    def post_episode(self, logger_args, output_dir, model):
        pass

    def pre_epoch(self, logger_args, output_dir, model):
        pass

    def post_epoch(self, logger_args, output_dir, model):
        pass

    def pre_train(self, logger_args, output_dir, model):
        pass

    def post_train(self, logger_args, output_dir, model):
        pass

    def pre_eval(self, logger_args, output_dir, model):
        pass

    def post_eval(self, logger_args, output_dir, model):
        pass

    def close(self, logger_args, output_dir, model):
        for handler in self.handlers:
            handler.close()


class TestLogger(BaseMontyLogger):
    def __init__(self, handlers):
        self.handlers = handlers
        self.log = []

    def pre_episode(self, logger_args, output_dir, model):
        self.log.append("pre_episode")

    def post_episode(self, logger_args, output_dir, model):
        self.log.append("post_episode")

    def pre_epoch(self, logger_args, output_dir, model):
        self.log.append("pre_epoch")

    def post_epoch(self, logger_args, output_dir, model):
        self.log.append("post_epoch")

    def pre_train(self, logger_args, output_dir, model):
        self.log.append("pre_train")

    def post_train(self, logger_args, output_dir, model):
        self.log.append("post_train")

    def pre_eval(self, logger_args, output_dir, model):
        self.log.append("pre_eval")

    def post_eval(self, logger_args, output_dir, model):
        self.log.append("post_eval")

    def close(self, logger_args, output_dir, model):
        with open(os.path.join(output_dir, "fake_log.pkl"), "wb") as f:
            pickle.dump(self.log, f)

    def __deepcopy__(self, memo):
        # Do not create new copy of loggers. They are create by the tests outside
        # the experiment
        return self
