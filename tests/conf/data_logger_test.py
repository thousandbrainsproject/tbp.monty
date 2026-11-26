# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import hydra
from omegaconf import OmegaConf

from tbp.monty.frameworks.loggers.graph_matching_loggers import (
    DetailedGraphMatchingLogger,
)
from tbp.monty.frameworks.loggers.monty_handlers import (
    BasicCSVStatsHandler,
    DetailedJSONHandler,
)
from tests import HYDRA_ROOT


def test_select_detailed_data_logger_preserves_other_logging_settings() -> None:
    experiment = "experiment=randrot_noise_10distinctobj_dist_agent"
    logger_package = "experiment.config.logging.monty_data_logger"
    override = f"logging/data_logger@{logger_package}=detailed"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(HYDRA_ROOT)):
        defaults = hydra.compose(config_name="experiment", overrides=[experiment])
        detailed = hydra.compose(
            config_name="experiment", overrides=[experiment, override]
        )

    default_logging = defaults.experiment.config.logging
    detailed_logging = detailed.experiment.config.logging
    assert detailed_logging.monty_data_logger != default_logging.monty_data_logger

    default_settings = OmegaConf.to_container(default_logging, resolve=False)
    detailed_settings = OmegaConf.to_container(detailed_logging, resolve=False)
    assert isinstance(default_settings, dict)
    assert isinstance(detailed_settings, dict)
    del default_settings["monty_data_logger"]
    del detailed_settings["monty_data_logger"]
    assert detailed_settings == default_settings

    logger = hydra.utils.instantiate(detailed_logging.monty_data_logger)
    assert isinstance(logger, DetailedGraphMatchingLogger)
    assert [type(handler) for handler in logger.handlers] == [
        BasicCSVStatsHandler,
        DetailedJSONHandler,
    ]
