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
from unittest.mock import MagicMock, sentinel

import numpy as np
import quaternion as qt

from tbp.monty.cmp import AttentionRegion
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.sensor_modules.sensor_module import (
    Payload,
    SensorModule,
    TransformContext,
)


class SensorModuleTest(unittest.TestCase):
    def assert_transform_ctx_equals(
        self, transform_ctx: TransformContext, expected_ctx: TransformContext
    ) -> None:
        self.assertEqual(transform_ctx.rng, expected_ctx.rng)
        self.assertIs(transform_ctx.agent_state, expected_ctx.agent_state)
        np.testing.assert_array_equal(
            transform_ctx.sensor_state.position, expected_ctx.sensor_state.position
        )
        self.assertEqual(
            transform_ctx.sensor_state.rotation, expected_ctx.sensor_state.rotation
        )
        self.assertFalse(transform_ctx.motor_only_step)
        self.assertFalse(transform_ctx.suppress_runtime_errors)

    def setUp(self) -> None:
        self._sensor_module_id = "test"
        self._default_sensor_state = SensorState(
            position=(0, 0, 0),
            rotation=qt.quaternion(1, 0, 0, 0),
        )
        self._state = AgentState(
            sensors={SensorID(self._sensor_module_id): self._default_sensor_state},
            position=(0, 0, 0),
            rotation=qt.quaternion(1, 0, 0, 0),
        )

    def test_step_no_transforms_returns_none(self) -> None:
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        percept = sensor_module.step(ctx, MagicMock())

        self.assertIsNone(percept)

    def test_step_no_transforms_prepares_empty_goals_list_for_propose_goals(
        self,
    ) -> None:
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, MagicMock())
        goals = sensor_module.propose_goals()

        self.assertEqual(goals, [])

    def test_step_no_transforms_prepares_empty_region_for_propose_region(self) -> None:
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, MagicMock())
        region = sensor_module.propose_region()

        np.testing.assert_array_equal(
            region.locations, AttentionRegion.empty().locations
        )
        np.testing.assert_array_equal(region.weights, AttentionRegion.empty().weights)

    def test_step_no_transforms_prepares_empty_telemetry_for_payload(self) -> None:
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, MagicMock())
        memento = sensor_module.state_dict()

        self.assertEqual(memento["telemetry"], {})

    def test_step_invokes_transforms_in_order(self) -> None:
        observation = MagicMock()
        transform1 = MagicMock()
        transform1.return_value = sentinel.transform1_payload
        transform2 = MagicMock()
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[transform1, transform2],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, observation)

        transform1_ctx = transform1.call_args_list[0].args[0]
        self.assert_transform_ctx_equals(
            transform1_ctx,
            TransformContext(
                rng=ctx.rng,
                agent_state=self._state,
                sensor_state=self._default_sensor_state,
                motor_only_step=False,
                suppress_runtime_errors=ctx.suppress_runtime_errors,
            ),
        )
        transform1_payload = transform1.call_args_list[0].args[1]
        self.assertEqual(
            transform1_payload,
            Payload(
                observation=observation,
                percept=None,
                goals=[],
                region=None,
                telemetry={},
            ),
        )
        transform2_ctx = transform2.call_args_list[0].args[0]
        self.assert_transform_ctx_equals(
            transform2_ctx,
            TransformContext(
                rng=ctx.rng,
                agent_state=self._state,
                sensor_state=self._default_sensor_state,
                motor_only_step=False,
                suppress_runtime_errors=ctx.suppress_runtime_errors,
            ),
        )
        transform2_payload = transform2.call_args_list[0].args[1]
        self.assertEqual(transform2_payload, sentinel.transform1_payload)

    def test_step_returns_last_transforms_payload_percept(self) -> None:
        observation = MagicMock()
        transform1 = MagicMock()
        transform1.return_value = sentinel.transform1_payload
        transform2 = MagicMock()
        transform2.return_value = Payload(
            observation=observation,
            percept=sentinel.transform2_percept,
            goals=[],
            region=None,
            telemetry={},
        )
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[transform1, transform2],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        percept = sensor_module.step(ctx, observation)

        self.assertEqual(percept, sentinel.transform2_percept)
        transform2_ctx = transform2.call_args_list[0].args[0]
        self.assert_transform_ctx_equals(
            transform2_ctx,
            TransformContext(
                rng=ctx.rng,
                agent_state=self._state,
                sensor_state=self._default_sensor_state,
                motor_only_step=False,
                suppress_runtime_errors=ctx.suppress_runtime_errors,
            ),
        )
        transform2_payload = transform2.call_args_list[0].args[1]
        self.assertEqual(transform2_payload, sentinel.transform1_payload)

    def test_step_prepares_last_transforms_payload_goals_for_propose_goals(
        self,
    ) -> None:
        observation = MagicMock()
        transform1 = MagicMock()
        transform1.return_value = sentinel.transform1_payload
        transform2 = MagicMock()
        transform2.return_value = Payload(
            observation=observation,
            percept=sentinel.transform2_percept,
            goals=sentinel.transform2_goals,
            region=None,
            telemetry={},
        )
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[transform1, transform2],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, observation)
        goals = sensor_module.propose_goals()

        self.assertEqual(goals, sentinel.transform2_goals)
        transform2_ctx = transform2.call_args_list[0].args[0]
        self.assert_transform_ctx_equals(
            transform2_ctx,
            TransformContext(
                rng=ctx.rng,
                agent_state=self._state,
                sensor_state=self._default_sensor_state,
                motor_only_step=False,
                suppress_runtime_errors=ctx.suppress_runtime_errors,
            ),
        )
        transform2_payload = transform2.call_args_list[0].args[1]
        self.assertEqual(transform2_payload, sentinel.transform1_payload)

    def test_step_prepares_last_transforms_payload_region_for_propose_region(
        self,
    ) -> None:
        observation = MagicMock()
        transform1 = MagicMock()
        transform1.return_value = sentinel.transform1_payload
        transform2 = MagicMock()
        transform2.return_value = Payload(
            observation=observation,
            percept=sentinel.transform2_percept,
            goals=[],
            region=sentinel.transform2_region,
            telemetry={},
        )
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[transform1, transform2],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, observation)
        region = sensor_module.propose_region()

        self.assertEqual(region, sentinel.transform2_region)
        transform2_ctx = transform2.call_args_list[0].args[0]
        self.assert_transform_ctx_equals(
            transform2_ctx,
            TransformContext(
                rng=ctx.rng,
                agent_state=self._state,
                sensor_state=self._default_sensor_state,
                motor_only_step=False,
                suppress_runtime_errors=ctx.suppress_runtime_errors,
            ),
        )
        transform2_payload = transform2.call_args_list[0].args[1]
        self.assertEqual(transform2_payload, sentinel.transform1_payload)

    def test_step_prepares_last_transforms_payload_telemetry_for_state_dict(
        self,
    ) -> None:
        observation = MagicMock()
        transform1 = MagicMock()
        transform1.return_value = sentinel.transform1_payload
        transform2 = MagicMock()
        transform2.return_value = Payload(
            observation=observation,
            percept=None,
            goals=[],
            region=None,
            telemetry=sentinel.transform2_telemetry,
        )
        sensor_module = SensorModule(
            sensor_module_id="test",
            sensor_id=SensorID("test"),
            transforms=[transform1, transform2],
        )
        ctx = RuntimeContext(rng=np.random.RandomState(0))
        sensor_module.update_state(self._state)

        sensor_module.step(ctx, observation)
        memento = sensor_module.state_dict()

        self.assertIs(memento["telemetry"], sentinel.transform2_telemetry)
        transform2_ctx = transform2.call_args_list[0].args[0]
        self.assert_transform_ctx_equals(
            transform2_ctx,
            TransformContext(
                rng=ctx.rng,
                agent_state=self._state,
                sensor_state=self._default_sensor_state,
                motor_only_step=False,
                suppress_runtime_errors=ctx.suppress_runtime_errors,
            ),
        )
        transform2_payload = transform2.call_args_list[0].args[1]
        self.assertEqual(transform2_payload, sentinel.transform1_payload)
