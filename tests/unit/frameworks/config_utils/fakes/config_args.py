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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from tbp.monty.frameworks.config_utils.config_args import (
    MontyArgs,
    MontyConfig,
    MotorSystemConfig,
)
from tbp.monty.frameworks.models.monty_base import MontyBase
from tests.unit.frameworks.models.fakes.learning_modules import FakeLearningModule
from tests.unit.frameworks.models.fakes.sensor_modules import FakeSensorModule


@dataclass
class FakeSingleCameraMontyConfig(MontyConfig):
    monty_class: Callable = MontyBase
    learning_module_configs: dataclass | Dict = field(
        default_factory=lambda: dict(
            learning_module_1=dict(
                learning_module_class=FakeLearningModule,
                learning_module_args={},
            )
        )
    )
    sensor_module_configs: dataclass | Dict = field(
        default_factory=lambda: dict(
            sensor_module_0=dict(
                sensor_module_class=FakeSensorModule,
                sensor_module_args=dict(sensor_module_id="sensor_id_0"),
            ),
        )
    )
    motor_system_config: dataclass | Dict = field(default_factory=MotorSystemConfig)
    sm_to_agent_dict: Dict = field(
        default_factory=lambda: dict(sensor_id_0="agent_id_0")
    )
    sm_to_lm_matrix: List = field(default_factory=lambda: [[0]])
    lm_to_lm_matrix: Optional[List] = None
    lm_to_lm_vote_matrix: Optional[List] = None
    monty_args: Dict | MontyArgs = field(default_factory=MontyArgs)
