# Copyright 2026 Thousand Brains Project
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
from unittest.mock import MagicMock, sentinel

import numpy as np
import quaternion as qt
from hypothesis import given
from hypothesis import strategies as st

from tbp.monty.experiment.sensor_modules.telemetry import Snapshot
from tbp.monty.frameworks.models.motor_system_state import SensorState
from tbp.monty.sensor_modules.sensor_module import Payload
from tests.strategies import sentinels
from tests.strategies.motor_system_state import sensor_state


class SnapshotTest(unittest.TestCase):
    @given(observations=sentinels.distinct(prefix="observation"))
    def test_appends_observation_to_telemetry_raw_observations(
        self, observations: list[Any]
    ) -> None:
        initial_observations_len = len(observations)
        telemetry: dict[str, Any] = {"raw_observations": observations}
        payload = Payload(
            observation=sentinel.new_observation,
            percept=None,
            goals=[],
            region=None,
            telemetry=telemetry,
        )
        snapshot = Snapshot()

        payload = snapshot(ctx=MagicMock(), payload=payload)

        self.assertEqual(
            len(payload.telemetry.get("raw_observations", [])),
            initial_observations_len + 1,
        )
        self.assertIn(
            sentinel.new_observation, payload.telemetry.get("raw_observations", [])
        )

    @given(observations=sentinels.distinct(prefix="observation"))
    def test_reset_clears_previous_payload_telemetry_raw_observations(
        self, observations: list[Any]
    ) -> None:
        telemetry: dict[str, Any] = {"raw_observations": observations}
        payload = Payload(
            observation=sentinel.new_observation,
            percept=None,
            goals=[],
            region=None,
            telemetry=telemetry,
        )
        snapshot = Snapshot()

        snapshot.reset()
        payload = snapshot(ctx=MagicMock(), payload=payload)

        self.assertEqual(len(payload.telemetry.get("raw_observations", [])), 1)
        self.assertEqual(
            [sentinel.new_observation], payload.telemetry.get("raw_observations", [])
        )

    @given(sensor_states=st.lists(sensor_state()), sensor_state=sensor_state())
    def test_appends_sm_rotation_and_location_to_telemetry_sm_properties(
        self, sensor_states: list[SensorState], sensor_state: SensorState
    ) -> None:
        initial_sm_properties_len = len(sensor_states)
        telemetry: dict[str, Any] = {
            "sm_properties": [
                {
                    "sm_location": np.array(ss.position),
                    "sm_rotation": qt.as_float_array(ss.rotation),
                }
                for ss in sensor_states
            ]
        }
        payload = Payload(
            observation=MagicMock(),
            percept=None,
            goals=[],
            region=None,
            telemetry=telemetry,
        )
        ctx_mock = MagicMock()
        ctx_mock.sensor_state = sensor_state
        snapshot = Snapshot()

        payload = snapshot(ctx=ctx_mock, payload=payload)

        self.assertEqual(
            len(payload.telemetry.get("sm_properties", [])),
            initial_sm_properties_len + 1,
        )
        expected = {
            "sm_location": np.array(sensor_state.position),
            "sm_rotation": qt.as_float_array(sensor_state.rotation),
        }
        found = False
        for sm_property in payload.telemetry.get("sm_properties", []):
            if np.allclose(
                sm_property["sm_location"], expected["sm_location"]
            ) and np.allclose(sm_property["sm_rotation"], expected["sm_rotation"]):
                found = True
                break
        self.assertTrue(found)

    @given(sensor_states=st.lists(sensor_state()), sensor_state=sensor_state())
    def test_reset_clears_previous_payload_telemetry_sm_properties(
        self, sensor_states: list[SensorState], sensor_state: SensorState
    ) -> None:
        telemetry: dict[str, Any] = {
            "sm_properties": [
                {
                    "sm_location": np.array(ss.position),
                    "sm_rotation": qt.as_float_array(ss.rotation),
                }
                for ss in sensor_states
            ]
        }
        payload = Payload(
            observation=MagicMock(),
            percept=None,
            goals=[],
            region=None,
            telemetry=telemetry,
        )
        ctx_mock = MagicMock()
        ctx_mock.sensor_state = sensor_state
        snapshot = Snapshot()

        snapshot.reset()
        payload = snapshot(ctx=ctx_mock, payload=payload)

        self.assertEqual(len(payload.telemetry.get("sm_properties", [])), 1)
        expected = {
            "sm_location": np.array(sensor_state.position),
            "sm_rotation": qt.as_float_array(sensor_state.rotation),
        }
        found = False
        for sm_property in payload.telemetry.get("sm_properties", []):
            if np.allclose(
                sm_property["sm_location"], expected["sm_location"]
            ) and np.allclose(sm_property["sm_rotation"], expected["sm_rotation"]):
                found = True
                break
        self.assertTrue(found)
