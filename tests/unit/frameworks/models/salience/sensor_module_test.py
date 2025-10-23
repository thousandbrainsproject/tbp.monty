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
from parameterized import parameterized

from tbp.monty.frameworks.models.salience.sensor_module import HabitatSalienceSM


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

    @parameterized.expand(
        [
            (True, False, True),
            (True, True, False),
            (False, False, False),
            (False, True, False),
        ]
    )
    @patch("tbp.monty.frameworks.models.salience.sensor_module.on_object_observation")
    def test_step_snapshots_raw_observation_as_needed(
        self,
        save_raw_obs: bool,
        is_exploring: bool,
        should_snapshot: bool,
        on_object_observation: MagicMock,
    ) -> None:
        self.sensor_module._save_raw_obs = save_raw_obs
        self.sensor_module.is_exploring = is_exploring
        data: dict[str, Any] = MagicMock()

        self.sensor_module.update_state(self.state)
        self.sensor_module.step(data)

        if should_snapshot:
            self.sensor_module._snapshot_telemetry.raw_observation.assert_called_once_with(  # type: ignore[attr-defined]
                data, self.state["rotation"], self.state["location"]
            )
        else:
            self.sensor_module._snapshot_telemetry.raw_observation.assert_not_called()  # type: ignore[attr-defined]
