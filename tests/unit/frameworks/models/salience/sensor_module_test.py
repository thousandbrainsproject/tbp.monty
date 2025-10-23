# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from parameterized import parameterized_class

from tbp.monty.frameworks.models.salience.sensor_module import HabitatSalienceSM
from tbp.monty.frameworks.models.salience.strategies import RGBADepthObservation


@parameterized_class(
    ("save_raw_obs", "is_exploring", "should_snapshot"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
@patch("tbp.monty.frameworks.models.salience.sensor_module.on_object_observation")
class HabitatSalienceSMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor_module = HabitatSalienceSM(
            rng=np.random.RandomState(42),
            sensor_module_id="test",
            salience_strategy_class=MagicMock,
            return_inhibitor_class=MagicMock,
            snapshot_telemetry_class=MagicMock,
        )
        self.state = {
            "rotation": "i'm a rotation",
            "location": "i'm a position",
        }

    def test_step_snapshots_raw_observation_as_needed(
        self,
        on_object_observation: MagicMock,
    ) -> None:
        self.sensor_module._save_raw_obs = self.save_raw_obs
        self.sensor_module.is_exploring = self.is_exploring
        data: dict[str, Any] = MagicMock()

        self.sensor_module.update_state(self.state)
        self.sensor_module.step(data)

        if self.should_snapshot:
            self.sensor_module._snapshot_telemetry.raw_observation.assert_called_once_with(  # type: ignore[attr-defined]
                data, self.state["rotation"], self.state["location"]
            )
        else:
            self.sensor_module._snapshot_telemetry.raw_observation.assert_not_called()  # type: ignore[attr-defined]

    def test_step_calls_salience_strategy(
        self,
        on_object_observation: MagicMock,
    ) -> None:
        data: dict[str, Any] = {
            "rgba": np.zeros((64, 64, 4)),
            "depth": np.zeros((64, 64)),
        }
        self.sensor_module.step(data)
        self.sensor_module._salience_strategy.assert_called_once_with(
            RGBADepthObservation(rgba=data["rgba"], depth=data["depth"])
        )
