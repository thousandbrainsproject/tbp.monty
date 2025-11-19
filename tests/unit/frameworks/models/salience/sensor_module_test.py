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
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import numpy.testing as npt
import pytest
from parameterized import parameterized_class

from tbp.monty.frameworks.models.salience.on_object_observation import (
    OnObjectObservation,
)
from tbp.monty.frameworks.models.salience.sensor_module import (
    HabitatSalienceSM,
)
from tbp.monty.frameworks.models.states import GoalState


@pytest.fixture
def mocked_object_observation():
    with patch(
        "tbp.monty.frameworks.models.salience.sensor_module.on_object_observation"
    ):
        yield


@parameterized_class(
    ("save_raw_obs", "is_exploring", "should_snapshot"),
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, False),
    ],
)
@pytest.mark.usefixtures("mocked_object_observation")
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

    def test_step_snapshots_raw_observation_as_needed(self) -> None:
        self.sensor_module._save_raw_obs = self.save_raw_obs  # type: ignore[attr-defined]
        self.sensor_module.is_exploring = self.is_exploring  # type: ignore[attr-defined]
        data: dict[str, Any] = MagicMock()

        self.sensor_module.update_state(self.state)
        self.sensor_module.step(data)

        if self.should_snapshot:  # type: ignore[attr-defined]
            self.sensor_module._snapshot_telemetry.raw_observation.assert_called_once_with(  # type: ignore[attr-defined]
                data, self.state["rotation"], self.state["location"]
            )
        else:
            self.sensor_module._snapshot_telemetry.raw_observation.assert_not_called()  # type: ignore[attr-defined]

    def test_step_returns_no_percept(self) -> None:
        self.assertIsNone(self.sensor_module.step(MagicMock()))

    @patch("tbp.monty.frameworks.models.salience.sensor_module.on_object_observation")
    def test_step_proposes_goals_properly(
        self, on_object_observation: MagicMock
    ) -> None:
        self.sensor_module._salience_strategy.return_value = sentinel.salience_map  # type: ignore[attr-defined]
        locations = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        on_object_observation.return_value = OnObjectObservation(
            center_location=sentinel.center_location,
            locations=locations,
            salience=sentinel.salience_map,
        )
        self.sensor_module._return_inhibitor.return_value = sentinel.ior_weights  # type: ignore[attr-defined]
        salience = 0.1 * np.array([1, 2, 3])
        self.sensor_module._weight_salience = MagicMock(return_value=salience)  # type: ignore[method-assign]
        data: dict[str, Any] = {
            "rgba": np.zeros((64, 64, 4)),
            "depth": np.zeros((64, 64)),
        }

        self.sensor_module.step(data)
        goals = self.sensor_module.propose_goal_states()

        self.sensor_module._salience_strategy.assert_called_once_with(  # type: ignore[attr-defined]
            rgba=data["rgba"], depth=data["depth"]
        )
        on_object_observation.assert_called_once_with(data, sentinel.salience_map)
        self.sensor_module._return_inhibitor.assert_called_once_with(  # type: ignore[attr-defined]
            sentinel.center_location, locations
        )
        self.sensor_module._weight_salience.assert_called_once_with(
            sentinel.salience_map, sentinel.ior_weights
        )

        self.assertEqual(len(goals), locations.shape[0])
        for i, g in enumerate(goals):
            expected_goal = GoalState(
                location=locations[i],
                confidence=salience[i],
                use_state=True,
                morphological_features=None,
                non_morphological_features=None,
                goal_tolerances=None,
                sender_id="test",
                sender_type="SM",
            )
            # TODO: implement __eq__ for GoalState
            npt.assert_array_equal(g.location, expected_goal.location)
            self.assertEqual(g.confidence, expected_goal.confidence)
            self.assertEqual(g.use_state, expected_goal.use_state)
            self.assertEqual(
                g.morphological_features, expected_goal.morphological_features
            )
            self.assertEqual(
                g.non_morphological_features, expected_goal.non_morphological_features
            )
            self.assertEqual(g.goal_tolerances, expected_goal.goal_tolerances)
            self.assertEqual(g.sender_id, expected_goal.sender_id)
            self.assertEqual(g.sender_type, expected_goal.sender_type)


class HabitatSalienceSMPrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor_module = HabitatSalienceSM(
            rng=np.random.RandomState(42),
            sensor_module_id="test",
            salience_strategy_class=MagicMock,
            return_inhibitor_class=MagicMock,
            snapshot_telemetry_class=MagicMock,
        )

    def test_normalize_salience_does_clips_uniform_salience_between_0_and_1(
        self,
    ) -> None:
        salience = 2 * np.ones(10)
        normalized = self.sensor_module._normalize_salience(salience)
        npt.assert_array_equal(normalized, np.ones(10))

    def test_normalize_salience_normalizes_empty_salience(self) -> None:
        salience = np.array([])
        normalized = self.sensor_module._normalize_salience(salience)
        npt.assert_array_equal(normalized, np.array([]))

    def test_weight_salience_decays_randomizes_and_normalizes_salience_in_that_order(
        self,
    ) -> None:
        salience = np.array([1, 2, 3])
        ior_weights = np.array([0.1, 0.2, 0.3])
        self.sensor_module._decay_salience = MagicMock(return_value=sentinel.decayed)  # type: ignore[method-assign]
        self.sensor_module._randomize_salience = MagicMock(  # type: ignore[method-assign]
            return_value=sentinel.randomized
        )
        self.sensor_module._normalize_salience = MagicMock(  # type: ignore[method-assign]
            return_value=sentinel.normalized
        )

        weighted = self.sensor_module._weight_salience(salience, ior_weights)

        self.sensor_module._decay_salience.assert_called_once_with(
            salience, ior_weights
        )
        self.sensor_module._randomize_salience.assert_called_once_with(sentinel.decayed)
        self.sensor_module._normalize_salience.assert_called_once_with(
            sentinel.randomized
        )
        self.assertEqual(weighted, sentinel.normalized)
