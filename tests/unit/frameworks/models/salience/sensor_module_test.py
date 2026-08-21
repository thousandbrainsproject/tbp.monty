# Copyright 2025-2026 Thousand Brains Project
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
import numpy.typing as npt
import pytest
import quaternion as qt
from parameterized import parameterized_class

from tbp.monty.cmp import Goal
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.models.salience.on_object_observation import (
    OnObjectObservation,
)
from tbp.monty.frameworks.models.salience.sensor_module import (
    SalienceSM,
)
from tbp.monty.frameworks.models.salience.telemetry import SalienceSMTelemetry
from tbp.monty.frameworks.sensors import SensorID


class ArrayEqual:
    def __init__(self, arr: npt.ArrayLike):
        self.arr = arr

    def __eq__(self, other: npt.ArrayLike):
        return np.array_equal(self.arr, other)

    def __hash__(self):
        return hash(np.asarray(self.arr).tobytes())


@pytest.fixture
def mocked_object_observation():
    empty_obs = OnObjectObservation(
        center_location=None,
        locations=np.empty((0, 3)),
        salience=np.empty([]),
        on_object_mask=np.zeros((64, 64), dtype=bool),
        locations_map=np.zeros((64, 64, 3)),
    )
    with patch(
        "tbp.monty.frameworks.models.salience.sensor_module.on_object_observation",
        return_value=empty_obs,
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
class SalienceSMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            salience_strategy=MagicMock(return_value=np.array([])),
            return_inhibitor=MagicMock(return_value=np.array([])),
            snapshot_telemetry=MagicMock(),
        )
        self.observation = SensorObservation(
            rgba=np.zeros((64, 64, 4), dtype=np.uint8),
            depth=np.zeros((64, 64)),
        )
        self.default_sensor_state = SensorState(
            position=(0, 0, 0),
            rotation=qt.quaternion(1, 0, 0, 0),
        )
        self.state = AgentState(
            sensors={
                SensorID(self.sensor_module.sensor_module_id): self.default_sensor_state
            },
            position=self.default_sensor_state.position,
            rotation=self.default_sensor_state.rotation,
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState())

    def test_step_snapshots_raw_observation_as_needed(self) -> None:
        self.sensor_module._save_raw_obs = self.save_raw_obs  # type: ignore[attr-defined]
        self.sensor_module.is_exploring = self.is_exploring  # type: ignore[attr-defined]
        data: dict[str, Any] = MagicMock()

        self.sensor_module.update_state(self.state)
        self.sensor_module.step(self.ctx, data)

        if self.should_snapshot:  # type: ignore[attr-defined]
            self.sensor_module._snapshot_telemetry.raw_observation.assert_called_once_with(  # type: ignore[attr-defined]
                data, self.state.rotation, ArrayEqual(self.state.position)
            )
        else:
            self.sensor_module._snapshot_telemetry.raw_observation.assert_not_called()  # type: ignore[attr-defined]

    def test_step_returns_no_percept(self) -> None:
        self.assertIsNone(self.sensor_module.step(self.ctx, self.observation))

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
            on_object_mask=np.zeros((64, 64), dtype=bool),
            locations_map=np.zeros((64, 64, 3)),
        )
        self.sensor_module._return_inhibitor.return_value = sentinel.ior_weights  # type: ignore[attr-defined]
        salience = 0.1 * np.array([1, 2, 3])
        self.sensor_module._weight_salience = MagicMock(return_value=salience)  # type: ignore[method-assign]
        data = SensorObservation(
            rgba=np.zeros((64, 64, 4), dtype=np.uint8),
            depth=np.zeros((64, 64)),
        )

        self.sensor_module.step(self.ctx, data)
        goals = self.sensor_module.propose_goals()

        self.sensor_module._salience_strategy.assert_called_once_with(  # type: ignore[attr-defined]
            ctx=self.ctx, rgba=data["rgba"], depth=data["depth"]
        )
        on_object_observation.assert_called_once_with(data, sentinel.salience_map)
        self.sensor_module._return_inhibitor.assert_called_once_with(  # type: ignore[attr-defined]
            sentinel.center_location, locations
        )
        self.sensor_module._weight_salience.assert_called_once_with(
            self.ctx, sentinel.salience_map, sentinel.ior_weights
        )

        self.assertEqual(len(goals), locations.shape[0])
        for i, g in enumerate(goals):
            expected_goal = Goal(
                location=locations[i],
                confidence=salience[i],
                pass_message=False,
                process_features_in_lm=False,
                morphological_features=None,
                non_morphological_features=None,
                goal_tolerances=None,
                sender_id="test",
                sender_type="SM",
            )
            # TODO: implement __eq__ for GoalState
            np.testing.assert_array_equal(g.location, expected_goal.location)
            self.assertEqual(g.confidence, expected_goal.confidence)
            self.assertEqual(g.pass_message, expected_goal.pass_message)
            self.assertEqual(
                g.morphological_features, expected_goal.morphological_features
            )
            self.assertEqual(
                g.non_morphological_features, expected_goal.non_morphological_features
            )
            self.assertEqual(g.goal_tolerances, expected_goal.goal_tolerances)
            self.assertEqual(g.sender_id, expected_goal.sender_id)
            self.assertEqual(g.sender_type, expected_goal.sender_type)


class SalienceSMRegionTest(unittest.TestCase):
    """The region is the segmented surface, proposed as goals."""

    def setUp(self) -> None:
        # A 2x2 frame: three on-object pixels, of which the segmentation covers
        # (0, 0) and (1, 1). Locations encode their own pixel coordinates.
        self.on_object_mask = np.array([[True, True], [False, True]])
        self.locations_map = np.zeros((2, 2, 3))
        for row in range(2):
            for col in range(2):
                self.locations_map[row, col] = [row, col, 1.0]
        self.segmentation_map = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        # Weighted salience for the on-object pixels, in row-major order:
        # (0, 0), (0, 1), (1, 1).
        self.weighted_salience = np.array([0.1, 0.5, 0.9])

        self.segmentation_strategy = MagicMock(return_value=self.segmentation_map)
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            salience_strategy=MagicMock(return_value=sentinel.salience_map),
            return_inhibitor=MagicMock(return_value=sentinel.ior_weights),
            snapshot_telemetry=MagicMock(),
            segmentation_strategy=self.segmentation_strategy,
        )
        self.sensor_module._weight_salience = MagicMock(  # type: ignore[method-assign]
            return_value=self.weighted_salience
        )
        self.observation = SensorObservation(
            rgba=np.zeros((2, 2, 4), dtype=np.uint8),
            depth=np.zeros((2, 2)),
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState())

    def step(self) -> None:
        """Step the sensor module with the mocked observation pipeline."""
        pix_rows, pix_cols = np.where(self.on_object_mask)
        on_object = OnObjectObservation(
            center_location=None,
            locations=self.locations_map[pix_rows, pix_cols],
            salience=sentinel.salience_map,
            on_object_mask=self.on_object_mask,
            locations_map=self.locations_map,
        )
        with patch(
            "tbp.monty.frameworks.models.salience.sensor_module."
            "on_object_observation",
            return_value=on_object,
        ):
            self.sensor_module.step(self.ctx, self.observation)

    def test_region_is_the_on_object_part_of_the_segmented_surface(self) -> None:
        self.step()
        region = self.sensor_module.propose_region()
        locations = [g.location.tolist() for g in region]
        # (1, 0) is segmented but off-object; (0, 1) is on-object but outside
        # the segmentation.
        self.assertEqual(locations, [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])

    def test_region_goals_carry_the_weighted_salience(self) -> None:
        self.step()
        region = self.sensor_module.propose_region()
        self.assertEqual([g.confidence for g in region], [0.1, 0.9])

    def test_segmentation_strategy_receives_the_observation(self) -> None:
        self.step()
        self.segmentation_strategy.assert_called_once_with(
            ctx=self.ctx,
            rgba=self.observation["rgba"],
            depth=self.observation["depth"],
        )

    def test_without_a_segmentation_strategy_the_region_is_empty(self) -> None:
        self.sensor_module._segmentation_strategy = None
        self.step()
        self.assertEqual(self.sensor_module.propose_region(), [])

    def test_reset_clears_the_region(self) -> None:
        self.step()
        self.sensor_module.reset()
        self.assertEqual(self.sensor_module.propose_region(), [])


class SalienceSMTelemetryRecordingTest(unittest.TestCase):
    """Segmentation masks and regions are stashed when telemetry is supplied."""

    def setUp(self) -> None:
        # The same 2x2 mocked pipeline as SalienceSMRegionTest, with telemetry.
        self.on_object_mask = np.array([[True, True], [False, True]])
        self.locations_map = np.zeros((2, 2, 3))
        for row in range(2):
            for col in range(2):
                self.locations_map[row, col] = [row, col, 1.0]
        self.segmentation_map = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        self.weighted_salience = np.array([0.1, 0.5, 0.9])

        self.telemetry = SalienceSMTelemetry(save_segmentation=True)
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            salience_strategy=MagicMock(return_value=sentinel.salience_map),
            return_inhibitor=MagicMock(return_value=sentinel.ior_weights),
            snapshot_telemetry=self.telemetry,
            segmentation_strategy=MagicMock(return_value=self.segmentation_map),
        )
        self.sensor_module._weight_salience = MagicMock(  # type: ignore[method-assign]
            return_value=self.weighted_salience
        )
        self.observation = SensorObservation(
            rgba=np.zeros((2, 2, 4), dtype=np.uint8),
            depth=np.zeros((2, 2)),
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState())

    def step(self) -> None:
        """Step the sensor module with the mocked observation pipeline."""
        pix_rows, pix_cols = np.where(self.on_object_mask)
        on_object = OnObjectObservation(
            center_location=None,
            locations=self.locations_map[pix_rows, pix_cols],
            salience=sentinel.salience_map,
            on_object_mask=self.on_object_mask,
            locations_map=self.locations_map,
        )
        with patch(
            "tbp.monty.frameworks.models.salience.sensor_module."
            "on_object_observation",
            return_value=on_object,
        ):
            self.sensor_module.step(self.ctx, self.observation)

    def test_step_records_the_segmentation_mask_and_region(self) -> None:
        self.step()
        state = self.telemetry.state_dict()
        np.testing.assert_array_equal(
            state["segmentation_maps"][0], self.segmentation_map
        )
        self.assertEqual(state["regions"][0], self.sensor_module.propose_region())

    def test_step_does_not_record_while_exploring(self) -> None:
        self.sensor_module.is_exploring = True
        self.step()
        self.assertEqual(self.telemetry.state_dict()["segmentation_maps"], [])

    def test_without_a_segmentation_strategy_none_is_recorded(self) -> None:
        self.sensor_module._segmentation_strategy = None
        self.step()
        state = self.telemetry.state_dict()
        self.assertIsNone(state["segmentation_maps"][0])
        self.assertEqual(state["regions"][0], [])

    def test_state_dict_holds_snapshot_and_segmentation_telemetry(self) -> None:
        self.step()
        state = self.sensor_module.state_dict()
        self.assertEqual(
            set(state),
            {"raw_observations", "sm_properties", "segmentation_maps", "regions"},
        )

    def test_recording_is_off_unless_save_segmentation_is_set(self) -> None:
        self.sensor_module._snapshot_telemetry = SalienceSMTelemetry()
        self.step()
        state = self.sensor_module.state_dict()
        self.assertEqual(state["segmentation_maps"], [])
        self.assertEqual(state["regions"], [])

    def test_reset_discards_the_recordings(self) -> None:
        self.step()
        self.sensor_module.reset()
        self.assertEqual(self.telemetry.state_dict()["segmentation_maps"], [])


class SalienceSMPrivateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor_module = SalienceSM(
            sensor_module_id="test",
            salience_strategy=MagicMock(),
            return_inhibitor=MagicMock(),
            snapshot_telemetry=MagicMock(),
        )
        self.ctx = RuntimeContext(rng=np.random.RandomState())

    def test_normalize_salience_does_clips_uniform_salience_between_0_and_1(
        self,
    ) -> None:
        salience = 2 * np.ones(10)
        normalized = self.sensor_module._normalize_salience(salience)
        np.testing.assert_array_equal(normalized, np.ones(10))

    def test_normalize_salience_normalizes_empty_salience(self) -> None:
        salience = np.array([])
        normalized = self.sensor_module._normalize_salience(salience)
        np.testing.assert_array_equal(normalized, np.array([]))

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

        weighted = self.sensor_module._weight_salience(self.ctx, salience, ior_weights)

        self.sensor_module._decay_salience.assert_called_once_with(
            salience, ior_weights
        )
        self.sensor_module._randomize_salience.assert_called_once_with(
            self.ctx, sentinel.decayed
        )
        self.sensor_module._normalize_salience.assert_called_once_with(
            sentinel.randomized
        )
        self.assertEqual(weighted, sentinel.normalized)
