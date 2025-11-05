# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
import importlib

import hydra
from omegaconf import DictConfig, OmegaConf

from tbp.monty.frameworks.run_env import setup_env


@hydra.main(config_path=".", config_name="config", version_base=None)
def validate(cfg: DictConfig):
    # Force interpolation of the config to get errors
    OmegaConf.to_object(cfg)
    print(OmegaConf.to_yaml(cfg))

    app = hydra.utils.instantiate(cfg.experiment)

    # app.setup_experiment(cfg)
    app.init_loggers(cfg.experiment.config.logging)

    print("done")


def monty_class_resolver(class_name: str) -> type:
    """Returns a class object by fully qualified path.

    TODO: This is an interim solution to retrieve my_class in
      the my_class(**my_args) pattern.
    """
    parts = class_name.split(".")
    module = ".".join(parts[:-1])
    klass = parts[-1]
    module_obj = importlib.import_module(module)
    return getattr(module_obj, klass)


if __name__ == "__main__":
    setup_env()
    OmegaConf.register_new_resolver("monty.class", monty_class_resolver)
    validate()
