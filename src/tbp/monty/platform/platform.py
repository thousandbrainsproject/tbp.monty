# Copyright 2025-2026 Thousand Brains Project
# Copyright 2021-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from tbp.monty.core.experiment import ExperimentConfig
from tbp.monty.libs.telemetry import (
    BasicAsyncPipeline,
    Telemetry,
    Traceable,
)
from tbp.monty.libs.telemetry.emitters import SimpleEmitter

from .config import PlatformConfig

__all__ = ["Platform"]


class Platform(Traceable):
    _EMITTER_CLASS = SimpleEmitter

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config

    def init(self) -> None:
        pipeline = BasicAsyncPipeline()
        pipeline.start()
        Telemetry.get(pipeline)
        self.tel.info("Monty Platform initialized!")

    def run(self, experiment_config: ExperimentConfig) -> None:
        self.tel.info("Running experiment")
        print(experiment_config.to_yaml())

    def shutdown(self) -> None:
        self.tel.info("Monty Platform shutdown received!")
        Telemetry.shutdown()
