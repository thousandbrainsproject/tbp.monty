# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import Collection

import quaternion as qt
from typing_extensions import Self

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import SensorObservation
from tbp.monty.frameworks.models.motor_system_state import AgentState, SensorState
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.sensor_modules.transform import TransformContext, TransformPipeline


class SensorModule:

    _agent_state: AgentState
    _goals: list[Goal]
    _sensor_id: SensorID
    _sensor_module_id: str
    _sensor_state: SensorState
    _transform_pipeline: TransformPipeline

    def __init__(
        self: Self,
        sensor_module_id: str,
        sensor_id: SensorID,
        transform_pipeline: TransformPipeline | None = None,
    ) -> None:
        self._sensor_module_id = sensor_module_id
        self._sensor_id = sensor_id
        self._transform_pipeline = (
            transform_pipeline
            if transform_pipeline is not None
            else TransformPipeline([])
        )

    @property
    def sensor_module_id(self) -> str:
        return self._sensor_module_id

    def propose_goals(self) -> Collection[Goal]:
        return self._goals

    def step(
        self: Self,
        ctx: RuntimeContext,
        observation: SensorObservation,
        motor_only_step: bool = False,
    ) -> Message | None:
        """Process an observation into a percept and goals.

        Args:
            ctx: The runtime context.
            observation: Sensor observation.
            motor_only_step: Whether the current step is a motor-only step.

        Returns:
            An optional percept with features and morphological features.
        """
        transform_ctx = TransformContext(
            rng=ctx.rng,
            state=self._agent_state,
            motor_only_step=motor_only_step,
        )
        _, percept, goals = self._transform_pipeline(
            transform_ctx, observation, None, []
        )
        self._goals = goals
        return percept

    def update_state(self: Self, agent: AgentState) -> None:
        self._agent_state = agent
        sensor = agent.sensors[self._sensor_id]
        self._sensor_state = SensorState(
            position=agent.position
            + qt.rotate_vectors(agent.rotation, sensor.position),
            rotation=agent.rotation * sensor.rotation,
        )
